
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/76b7a81be9b27c2e92c4951280c0a8da664b997c', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '76b7a81be9b27c2e92c4951280c0a8da664b997c', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/76b7a81be9b27c2e92c4951280c0a8da664b997c

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/76b7a81be9b27c2e92c4951280c0a8da664b997c

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f4ff068cf28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-17 12:14:08.999256
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-17 12:14:09.002936
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-17 12:14:09.005992
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-17 12:14:09.009025
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f4ffc4563c8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354453.4375
Epoch 2/10

1/1 [==============================] - 0s 105ms/step - loss: 260406.2656
Epoch 3/10

1/1 [==============================] - 0s 103ms/step - loss: 167947.8438
Epoch 4/10

1/1 [==============================] - 0s 98ms/step - loss: 97775.4844
Epoch 5/10

1/1 [==============================] - 0s 99ms/step - loss: 56355.2422
Epoch 6/10

1/1 [==============================] - 0s 98ms/step - loss: 33527.3750
Epoch 7/10

1/1 [==============================] - 0s 100ms/step - loss: 21121.3613
Epoch 8/10

1/1 [==============================] - 0s 109ms/step - loss: 14150.5420
Epoch 9/10

1/1 [==============================] - 0s 103ms/step - loss: 10044.7881
Epoch 10/10

1/1 [==============================] - 0s 100ms/step - loss: 7542.6650

  #### Inference Need return ypred, ytrue ######################### 
[[ 8.2369310e-01 -5.2478516e-01  9.3614161e-03  1.4110944e+00
  -7.1628034e-02 -1.3394301e-01  1.4468534e+00 -7.0223224e-01
  -8.9721274e-01 -4.0947664e-01  8.8413936e-01 -3.9279789e-01
   1.3245974e+00 -3.5920143e-03 -2.4041303e-02 -4.1516539e-01
  -4.3092626e-01  6.2100226e-01  2.9361308e-02  1.3576729e+00
  -2.5319171e-01  9.8834997e-01  2.7998966e-01  4.3860674e-03
   1.1325760e+00  1.5660770e+00 -6.9570291e-01  3.0299756e-01
   2.8733069e-01  7.1673989e-01  3.2823437e-01  1.2233876e+00
   3.8480544e-01  7.9827738e-01  1.1132150e+00  1.3352137e+00
   6.7126381e-01 -8.1862736e-01  1.3771514e+00 -3.5296637e-01
   5.0163347e-01 -6.7630976e-01 -3.4115678e-01 -1.7710036e-01
  -1.4761084e+00 -7.9543942e-01  3.2884714e-01  8.3330345e-01
   1.7625213e-01  5.2181679e-01 -1.8043101e-02 -1.4654338e+00
  -4.0039811e-01  3.5926968e-01  3.7952754e-01 -2.5759152e-01
   1.5589572e-01 -6.3724774e-01 -5.3047907e-01 -2.7113575e-01
   5.9344232e-02  8.7313193e-01  3.3580929e-01 -2.5504422e-01
   9.7579741e-01  1.0393685e-01  1.2083354e+00 -6.6379452e-01
   7.0006406e-01 -1.5004520e+00  3.3376604e-01  7.1757984e-01
  -1.1656496e-01  1.5679411e+00  1.1139526e+00  5.1768589e-01
   8.5573041e-01  9.1432166e-01 -7.6913500e-01 -3.5322037e-01
  -8.3323163e-01  6.8714112e-01  3.9878882e-02  1.3980489e+00
  -5.0930709e-02  6.4030731e-01 -1.9896552e-01 -1.5532200e+00
  -2.5188088e-01 -1.8685499e-01 -8.3307976e-01  1.4779449e+00
  -1.7423687e+00  4.9677342e-01  6.8293941e-01 -4.1835368e-01
   8.1127107e-02 -1.4554487e+00 -3.8581178e-01  1.1552001e+00
  -1.7213091e+00  4.9468458e-02  5.9809613e-01  1.0863791e+00
   6.2156957e-01  1.3551762e+00 -1.5044609e-01 -8.2531464e-01
  -2.1632296e-01 -9.0862751e-02  1.8562114e+00  1.3195639e+00
  -7.6815081e-01  1.2774439e+00 -1.5503910e+00  5.4727817e-01
   1.3794125e+00  6.2197715e-01 -6.9999379e-01 -1.7678563e-01
  -1.1543822e-02  8.0368690e+00  6.4563370e+00  5.7248192e+00
   7.8987417e+00  7.4945664e+00  6.3124685e+00  8.6980152e+00
   5.8599300e+00  7.7216263e+00  6.9031796e+00  8.3998718e+00
   7.5694942e+00  7.9873109e+00  6.9640870e+00  7.5759187e+00
   8.1470251e+00  5.6243696e+00  8.6551857e+00  7.8494859e+00
   7.0756769e+00  7.2139959e+00  5.6079979e+00  8.5622826e+00
   8.4948339e+00  6.3018789e+00  8.4379673e+00  7.8487082e+00
   6.3559861e+00  8.1103086e+00  7.5388174e+00  8.9033899e+00
   7.1243525e+00  8.5261021e+00  7.5172424e+00  6.3509045e+00
   8.1576786e+00  7.8691010e+00  7.7870526e+00  6.5611205e+00
   7.1634159e+00  7.0945072e+00  7.7107258e+00  5.9467330e+00
   6.0630326e+00  6.6814647e+00  7.5570922e+00  7.3822017e+00
   7.8953443e+00  7.8887854e+00  7.4514785e+00  7.2999606e+00
   6.9992285e+00  9.0688553e+00  6.0058770e+00  7.3956213e+00
   5.4063044e+00  7.5515013e+00  7.1048894e+00  8.1894798e+00
   1.1233778e+00  4.7459173e-01  1.5522386e+00  8.5720444e-01
   1.1396220e+00  2.3848493e+00  9.6544254e-01  1.5165070e+00
   2.3730345e+00  1.0819950e+00  1.4060121e+00  5.7505751e-01
   6.0420406e-01  2.4438272e+00  4.5756555e-01  1.6606364e+00
   1.4268119e+00  5.3541976e-01  1.2640033e+00  1.7740315e+00
   1.9364777e+00  2.1378288e+00  1.1374667e+00  2.0282631e+00
   5.9774864e-01  1.3334388e+00  5.3885990e-01  1.7048370e+00
   2.1142659e+00  2.8261089e-01  1.6706935e+00  3.3492613e-01
   1.0730075e+00  3.3982629e-01  4.2212170e-01  1.7468939e+00
   1.4771909e+00  8.0890149e-01  2.0190678e+00  1.1004214e+00
   4.1019452e-01  3.9673829e-01  2.8397899e+00  1.3985463e+00
   2.0383084e+00  7.8337538e-01  1.0458429e+00  2.2798982e+00
   6.2723339e-01  1.1722248e+00  5.2706701e-01  3.2447582e-01
   1.3414797e+00  2.1665344e+00  1.7224096e+00  9.8863202e-01
   1.7296510e+00  2.4432459e+00  8.2260644e-01  8.4253085e-01
   1.1403260e+00  7.5347483e-01  1.7055509e+00  1.3143301e+00
   2.2905393e+00  1.4267677e+00  1.5603101e+00  8.4093803e-01
   8.2282150e-01  1.2798452e+00  1.4653326e+00  9.7844374e-01
   8.6894369e-01  3.9319456e-01  1.6164942e+00  2.4326930e+00
   3.3616531e-01  4.1413271e-01  8.3879232e-01  1.4378932e+00
   1.2203059e+00  4.1506928e-01  1.3157868e-01  4.8889971e-01
   2.8926353e+00  1.5696596e+00  8.6566389e-01  6.8514192e-01
   1.0535560e+00  9.5889920e-01  2.9350626e-01  1.8122067e+00
   1.1669017e+00  4.6155113e-01  2.3380846e-01  2.0356698e+00
   1.3839002e+00  1.0040481e+00  2.1061273e+00  7.7666926e-01
   2.1760781e+00  8.0711216e-01  9.0889657e-01  1.4479749e+00
   2.8711492e-01  4.3556571e-01  6.9547331e-01  1.3567204e+00
   2.2030334e+00  1.5698720e+00  4.2030013e-01  1.1396905e+00
   2.7436912e-01  2.1966434e+00  2.2061574e-01  6.7558360e-01
   7.8687441e-01  8.7331355e-02  7.2152120e-01  1.9426646e+00
   8.7361574e-02  9.1327200e+00  9.0303164e+00  6.7611947e+00
   6.3750443e+00  8.8816175e+00  8.5222893e+00  7.4355712e+00
   7.6159496e+00  6.9869037e+00  6.9547539e+00  7.9876590e+00
   8.2035055e+00  8.8266745e+00  9.1403303e+00  7.2991710e+00
   6.9336734e+00  8.0019217e+00  7.7386007e+00  8.1551180e+00
   6.0735898e+00  8.7178068e+00  8.3441763e+00  6.9728799e+00
   8.3337355e+00  7.5876994e+00  8.1495638e+00  8.3386497e+00
   7.7095747e+00  8.9814205e+00  9.2219486e+00  7.1859832e+00
   6.9380212e+00  8.1143646e+00  8.3091621e+00  7.0198236e+00
   9.3963232e+00  7.3727384e+00  9.1260462e+00  7.2248421e+00
   8.4885769e+00  6.3346877e+00  7.6374135e+00  8.2861462e+00
   8.5160809e+00  7.2799144e+00  7.9970241e+00  7.9511962e+00
   7.0386114e+00  8.4197693e+00  7.6387997e+00  7.7870049e+00
   8.0027246e+00  7.7868419e+00  7.8239970e+00  7.0506439e+00
   7.5935535e+00  8.1070814e+00  7.8755522e+00  9.1585989e+00
  -8.5588255e+00 -2.2821019e+00  2.0631990e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-17 12:14:18.118978
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.1039
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-17 12:14:18.123997
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9062.34
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-17 12:14:18.127619
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.1961
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-17 12:14:18.131843
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -810.597
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139980970243408
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139978440995224
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139978440995728
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139978440996232
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139978440996736
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139978440997240

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f4fdbff9da0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.461442
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.427355
grad_step = 000002, loss = 0.404432
grad_step = 000003, loss = 0.379343
grad_step = 000004, loss = 0.352624
grad_step = 000005, loss = 0.326060
grad_step = 000006, loss = 0.299754
grad_step = 000007, loss = 0.283359
grad_step = 000008, loss = 0.269376
grad_step = 000009, loss = 0.255762
grad_step = 000010, loss = 0.242808
grad_step = 000011, loss = 0.229252
grad_step = 000012, loss = 0.217699
grad_step = 000013, loss = 0.208493
grad_step = 000014, loss = 0.201008
grad_step = 000015, loss = 0.193916
grad_step = 000016, loss = 0.186753
grad_step = 000017, loss = 0.178870
grad_step = 000018, loss = 0.170248
grad_step = 000019, loss = 0.161595
grad_step = 000020, loss = 0.153354
grad_step = 000021, loss = 0.145624
grad_step = 000022, loss = 0.138687
grad_step = 000023, loss = 0.132240
grad_step = 000024, loss = 0.125700
grad_step = 000025, loss = 0.119053
grad_step = 000026, loss = 0.112816
grad_step = 000027, loss = 0.107381
grad_step = 000028, loss = 0.102449
grad_step = 000029, loss = 0.097316
grad_step = 000030, loss = 0.091830
grad_step = 000031, loss = 0.086471
grad_step = 000032, loss = 0.081578
grad_step = 000033, loss = 0.077056
grad_step = 000034, loss = 0.072821
grad_step = 000035, loss = 0.068907
grad_step = 000036, loss = 0.065143
grad_step = 000037, loss = 0.061375
grad_step = 000038, loss = 0.057812
grad_step = 000039, loss = 0.054578
grad_step = 000040, loss = 0.051483
grad_step = 000041, loss = 0.048502
grad_step = 000042, loss = 0.045678
grad_step = 000043, loss = 0.042905
grad_step = 000044, loss = 0.040248
grad_step = 000045, loss = 0.037887
grad_step = 000046, loss = 0.035705
grad_step = 000047, loss = 0.033558
grad_step = 000048, loss = 0.031524
grad_step = 000049, loss = 0.029585
grad_step = 000050, loss = 0.027727
grad_step = 000051, loss = 0.026023
grad_step = 000052, loss = 0.024423
grad_step = 000053, loss = 0.022889
grad_step = 000054, loss = 0.021459
grad_step = 000055, loss = 0.020093
grad_step = 000056, loss = 0.018809
grad_step = 000057, loss = 0.017645
grad_step = 000058, loss = 0.016529
grad_step = 000059, loss = 0.015467
grad_step = 000060, loss = 0.014478
grad_step = 000061, loss = 0.013539
grad_step = 000062, loss = 0.012679
grad_step = 000063, loss = 0.011877
grad_step = 000064, loss = 0.011128
grad_step = 000065, loss = 0.010444
grad_step = 000066, loss = 0.009792
grad_step = 000067, loss = 0.009189
grad_step = 000068, loss = 0.008635
grad_step = 000069, loss = 0.008111
grad_step = 000070, loss = 0.007637
grad_step = 000071, loss = 0.007195
grad_step = 000072, loss = 0.006791
grad_step = 000073, loss = 0.006420
grad_step = 000074, loss = 0.006075
grad_step = 000075, loss = 0.005764
grad_step = 000076, loss = 0.005470
grad_step = 000077, loss = 0.005201
grad_step = 000078, loss = 0.004955
grad_step = 000079, loss = 0.004728
grad_step = 000080, loss = 0.004525
grad_step = 000081, loss = 0.004334
grad_step = 000082, loss = 0.004160
grad_step = 000083, loss = 0.003997
grad_step = 000084, loss = 0.003848
grad_step = 000085, loss = 0.003712
grad_step = 000086, loss = 0.003586
grad_step = 000087, loss = 0.003470
grad_step = 000088, loss = 0.003361
grad_step = 000089, loss = 0.003262
grad_step = 000090, loss = 0.003170
grad_step = 000091, loss = 0.003085
grad_step = 000092, loss = 0.003004
grad_step = 000093, loss = 0.002931
grad_step = 000094, loss = 0.002863
grad_step = 000095, loss = 0.002800
grad_step = 000096, loss = 0.002742
grad_step = 000097, loss = 0.002688
grad_step = 000098, loss = 0.002638
grad_step = 000099, loss = 0.002592
grad_step = 000100, loss = 0.002549
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002510
grad_step = 000102, loss = 0.002474
grad_step = 000103, loss = 0.002441
grad_step = 000104, loss = 0.002411
grad_step = 000105, loss = 0.002383
grad_step = 000106, loss = 0.002358
grad_step = 000107, loss = 0.002334
grad_step = 000108, loss = 0.002313
grad_step = 000109, loss = 0.002294
grad_step = 000110, loss = 0.002277
grad_step = 000111, loss = 0.002261
grad_step = 000112, loss = 0.002247
grad_step = 000113, loss = 0.002233
grad_step = 000114, loss = 0.002222
grad_step = 000115, loss = 0.002211
grad_step = 000116, loss = 0.002201
grad_step = 000117, loss = 0.002192
grad_step = 000118, loss = 0.002184
grad_step = 000119, loss = 0.002177
grad_step = 000120, loss = 0.002170
grad_step = 000121, loss = 0.002163
grad_step = 000122, loss = 0.002157
grad_step = 000123, loss = 0.002151
grad_step = 000124, loss = 0.002145
grad_step = 000125, loss = 0.002139
grad_step = 000126, loss = 0.002134
grad_step = 000127, loss = 0.002129
grad_step = 000128, loss = 0.002123
grad_step = 000129, loss = 0.002118
grad_step = 000130, loss = 0.002113
grad_step = 000131, loss = 0.002107
grad_step = 000132, loss = 0.002102
grad_step = 000133, loss = 0.002096
grad_step = 000134, loss = 0.002090
grad_step = 000135, loss = 0.002085
grad_step = 000136, loss = 0.002080
grad_step = 000137, loss = 0.002074
grad_step = 000138, loss = 0.002068
grad_step = 000139, loss = 0.002065
grad_step = 000140, loss = 0.002065
grad_step = 000141, loss = 0.002061
grad_step = 000142, loss = 0.002053
grad_step = 000143, loss = 0.002045
grad_step = 000144, loss = 0.002040
grad_step = 000145, loss = 0.002040
grad_step = 000146, loss = 0.002035
grad_step = 000147, loss = 0.002024
grad_step = 000148, loss = 0.002021
grad_step = 000149, loss = 0.002020
grad_step = 000150, loss = 0.002014
grad_step = 000151, loss = 0.002009
grad_step = 000152, loss = 0.002010
grad_step = 000153, loss = 0.002009
grad_step = 000154, loss = 0.001997
grad_step = 000155, loss = 0.001992
grad_step = 000156, loss = 0.001994
grad_step = 000157, loss = 0.001987
grad_step = 000158, loss = 0.001979
grad_step = 000159, loss = 0.001978
grad_step = 000160, loss = 0.001976
grad_step = 000161, loss = 0.001967
grad_step = 000162, loss = 0.001964
grad_step = 000163, loss = 0.001963
grad_step = 000164, loss = 0.001958
grad_step = 000165, loss = 0.001957
grad_step = 000166, loss = 0.001967
grad_step = 000167, loss = 0.001989
grad_step = 000168, loss = 0.002019
grad_step = 000169, loss = 0.001982
grad_step = 000170, loss = 0.001941
grad_step = 000171, loss = 0.001943
grad_step = 000172, loss = 0.001962
grad_step = 000173, loss = 0.001950
grad_step = 000174, loss = 0.001925
grad_step = 000175, loss = 0.001927
grad_step = 000176, loss = 0.001947
grad_step = 000177, loss = 0.001946
grad_step = 000178, loss = 0.001928
grad_step = 000179, loss = 0.001909
grad_step = 000180, loss = 0.001909
grad_step = 000181, loss = 0.001920
grad_step = 000182, loss = 0.001921
grad_step = 000183, loss = 0.001910
grad_step = 000184, loss = 0.001895
grad_step = 000185, loss = 0.001892
grad_step = 000186, loss = 0.001896
grad_step = 000187, loss = 0.001898
grad_step = 000188, loss = 0.001897
grad_step = 000189, loss = 0.001889
grad_step = 000190, loss = 0.001882
grad_step = 000191, loss = 0.001875
grad_step = 000192, loss = 0.001872
grad_step = 000193, loss = 0.001871
grad_step = 000194, loss = 0.001872
grad_step = 000195, loss = 0.001874
grad_step = 000196, loss = 0.001875
grad_step = 000197, loss = 0.001877
grad_step = 000198, loss = 0.001876
grad_step = 000199, loss = 0.001873
grad_step = 000200, loss = 0.001868
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001864
grad_step = 000202, loss = 0.001857
grad_step = 000203, loss = 0.001851
grad_step = 000204, loss = 0.001846
grad_step = 000205, loss = 0.001842
grad_step = 000206, loss = 0.001839
grad_step = 000207, loss = 0.001836
grad_step = 000208, loss = 0.001835
grad_step = 000209, loss = 0.001835
grad_step = 000210, loss = 0.001837
grad_step = 000211, loss = 0.001843
grad_step = 000212, loss = 0.001859
grad_step = 000213, loss = 0.001884
grad_step = 000214, loss = 0.001929
grad_step = 000215, loss = 0.001919
grad_step = 000216, loss = 0.001890
grad_step = 000217, loss = 0.001843
grad_step = 000218, loss = 0.001830
grad_step = 000219, loss = 0.001845
grad_step = 000220, loss = 0.001857
grad_step = 000221, loss = 0.001848
grad_step = 000222, loss = 0.001820
grad_step = 000223, loss = 0.001809
grad_step = 000224, loss = 0.001829
grad_step = 000225, loss = 0.001838
grad_step = 000226, loss = 0.001817
grad_step = 000227, loss = 0.001806
grad_step = 000228, loss = 0.001815
grad_step = 000229, loss = 0.001813
grad_step = 000230, loss = 0.001796
grad_step = 000231, loss = 0.001791
grad_step = 000232, loss = 0.001801
grad_step = 000233, loss = 0.001801
grad_step = 000234, loss = 0.001789
grad_step = 000235, loss = 0.001788
grad_step = 000236, loss = 0.001793
grad_step = 000237, loss = 0.001791
grad_step = 000238, loss = 0.001783
grad_step = 000239, loss = 0.001779
grad_step = 000240, loss = 0.001781
grad_step = 000241, loss = 0.001783
grad_step = 000242, loss = 0.001779
grad_step = 000243, loss = 0.001776
grad_step = 000244, loss = 0.001779
grad_step = 000245, loss = 0.001785
grad_step = 000246, loss = 0.001791
grad_step = 000247, loss = 0.001802
grad_step = 000248, loss = 0.001817
grad_step = 000249, loss = 0.001842
grad_step = 000250, loss = 0.001839
grad_step = 000251, loss = 0.001820
grad_step = 000252, loss = 0.001778
grad_step = 000253, loss = 0.001754
grad_step = 000254, loss = 0.001757
grad_step = 000255, loss = 0.001770
grad_step = 000256, loss = 0.001778
grad_step = 000257, loss = 0.001771
grad_step = 000258, loss = 0.001761
grad_step = 000259, loss = 0.001751
grad_step = 000260, loss = 0.001743
grad_step = 000261, loss = 0.001741
grad_step = 000262, loss = 0.001744
grad_step = 000263, loss = 0.001748
grad_step = 000264, loss = 0.001747
grad_step = 000265, loss = 0.001739
grad_step = 000266, loss = 0.001729
grad_step = 000267, loss = 0.001723
grad_step = 000268, loss = 0.001723
grad_step = 000269, loss = 0.001728
grad_step = 000270, loss = 0.001730
grad_step = 000271, loss = 0.001727
grad_step = 000272, loss = 0.001721
grad_step = 000273, loss = 0.001715
grad_step = 000274, loss = 0.001712
grad_step = 000275, loss = 0.001711
grad_step = 000276, loss = 0.001714
grad_step = 000277, loss = 0.001717
grad_step = 000278, loss = 0.001719
grad_step = 000279, loss = 0.001717
grad_step = 000280, loss = 0.001714
grad_step = 000281, loss = 0.001715
grad_step = 000282, loss = 0.001725
grad_step = 000283, loss = 0.001740
grad_step = 000284, loss = 0.001771
grad_step = 000285, loss = 0.001790
grad_step = 000286, loss = 0.001814
grad_step = 000287, loss = 0.001780
grad_step = 000288, loss = 0.001731
grad_step = 000289, loss = 0.001692
grad_step = 000290, loss = 0.001697
grad_step = 000291, loss = 0.001727
grad_step = 000292, loss = 0.001736
grad_step = 000293, loss = 0.001716
grad_step = 000294, loss = 0.001686
grad_step = 000295, loss = 0.001679
grad_step = 000296, loss = 0.001695
grad_step = 000297, loss = 0.001708
grad_step = 000298, loss = 0.001702
grad_step = 000299, loss = 0.001683
grad_step = 000300, loss = 0.001670
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001672
grad_step = 000302, loss = 0.001681
grad_step = 000303, loss = 0.001687
grad_step = 000304, loss = 0.001683
grad_step = 000305, loss = 0.001673
grad_step = 000306, loss = 0.001664
grad_step = 000307, loss = 0.001661
grad_step = 000308, loss = 0.001664
grad_step = 000309, loss = 0.001667
grad_step = 000310, loss = 0.001669
grad_step = 000311, loss = 0.001666
grad_step = 000312, loss = 0.001662
grad_step = 000313, loss = 0.001658
grad_step = 000314, loss = 0.001657
grad_step = 000315, loss = 0.001660
grad_step = 000316, loss = 0.001668
grad_step = 000317, loss = 0.001682
grad_step = 000318, loss = 0.001707
grad_step = 000319, loss = 0.001750
grad_step = 000320, loss = 0.001796
grad_step = 000321, loss = 0.001851
grad_step = 000322, loss = 0.001804
grad_step = 000323, loss = 0.001725
grad_step = 000324, loss = 0.001670
grad_step = 000325, loss = 0.001667
grad_step = 000326, loss = 0.001674
grad_step = 000327, loss = 0.001670
grad_step = 000328, loss = 0.001691
grad_step = 000329, loss = 0.001686
grad_step = 000330, loss = 0.001642
grad_step = 000331, loss = 0.001643
grad_step = 000332, loss = 0.001665
grad_step = 000333, loss = 0.001647
grad_step = 000334, loss = 0.001635
grad_step = 000335, loss = 0.001647
grad_step = 000336, loss = 0.001642
grad_step = 000337, loss = 0.001630
grad_step = 000338, loss = 0.001634
grad_step = 000339, loss = 0.001634
grad_step = 000340, loss = 0.001623
grad_step = 000341, loss = 0.001623
grad_step = 000342, loss = 0.001630
grad_step = 000343, loss = 0.001625
grad_step = 000344, loss = 0.001615
grad_step = 000345, loss = 0.001616
grad_step = 000346, loss = 0.001620
grad_step = 000347, loss = 0.001614
grad_step = 000348, loss = 0.001610
grad_step = 000349, loss = 0.001612
grad_step = 000350, loss = 0.001612
grad_step = 000351, loss = 0.001607
grad_step = 000352, loss = 0.001605
grad_step = 000353, loss = 0.001606
grad_step = 000354, loss = 0.001605
grad_step = 000355, loss = 0.001601
grad_step = 000356, loss = 0.001598
grad_step = 000357, loss = 0.001598
grad_step = 000358, loss = 0.001598
grad_step = 000359, loss = 0.001596
grad_step = 000360, loss = 0.001593
grad_step = 000361, loss = 0.001593
grad_step = 000362, loss = 0.001593
grad_step = 000363, loss = 0.001591
grad_step = 000364, loss = 0.001589
grad_step = 000365, loss = 0.001589
grad_step = 000366, loss = 0.001590
grad_step = 000367, loss = 0.001592
grad_step = 000368, loss = 0.001595
grad_step = 000369, loss = 0.001604
grad_step = 000370, loss = 0.001619
grad_step = 000371, loss = 0.001651
grad_step = 000372, loss = 0.001693
grad_step = 000373, loss = 0.001762
grad_step = 000374, loss = 0.001794
grad_step = 000375, loss = 0.001800
grad_step = 000376, loss = 0.001716
grad_step = 000377, loss = 0.001622
grad_step = 000378, loss = 0.001587
grad_step = 000379, loss = 0.001613
grad_step = 000380, loss = 0.001656
grad_step = 000381, loss = 0.001660
grad_step = 000382, loss = 0.001610
grad_step = 000383, loss = 0.001572
grad_step = 000384, loss = 0.001583
grad_step = 000385, loss = 0.001615
grad_step = 000386, loss = 0.001619
grad_step = 000387, loss = 0.001587
grad_step = 000388, loss = 0.001564
grad_step = 000389, loss = 0.001568
grad_step = 000390, loss = 0.001578
grad_step = 000391, loss = 0.001582
grad_step = 000392, loss = 0.001578
grad_step = 000393, loss = 0.001567
grad_step = 000394, loss = 0.001556
grad_step = 000395, loss = 0.001553
grad_step = 000396, loss = 0.001560
grad_step = 000397, loss = 0.001567
grad_step = 000398, loss = 0.001560
grad_step = 000399, loss = 0.001548
grad_step = 000400, loss = 0.001542
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001546
grad_step = 000402, loss = 0.001551
grad_step = 000403, loss = 0.001549
grad_step = 000404, loss = 0.001543
grad_step = 000405, loss = 0.001540
grad_step = 000406, loss = 0.001539
grad_step = 000407, loss = 0.001538
grad_step = 000408, loss = 0.001535
grad_step = 000409, loss = 0.001533
grad_step = 000410, loss = 0.001533
grad_step = 000411, loss = 0.001532
grad_step = 000412, loss = 0.001529
grad_step = 000413, loss = 0.001526
grad_step = 000414, loss = 0.001524
grad_step = 000415, loss = 0.001524
grad_step = 000416, loss = 0.001524
grad_step = 000417, loss = 0.001523
grad_step = 000418, loss = 0.001520
grad_step = 000419, loss = 0.001517
grad_step = 000420, loss = 0.001515
grad_step = 000421, loss = 0.001515
grad_step = 000422, loss = 0.001514
grad_step = 000423, loss = 0.001513
grad_step = 000424, loss = 0.001511
grad_step = 000425, loss = 0.001509
grad_step = 000426, loss = 0.001508
grad_step = 000427, loss = 0.001507
grad_step = 000428, loss = 0.001507
grad_step = 000429, loss = 0.001506
grad_step = 000430, loss = 0.001505
grad_step = 000431, loss = 0.001505
grad_step = 000432, loss = 0.001506
grad_step = 000433, loss = 0.001510
grad_step = 000434, loss = 0.001516
grad_step = 000435, loss = 0.001528
grad_step = 000436, loss = 0.001544
grad_step = 000437, loss = 0.001568
grad_step = 000438, loss = 0.001589
grad_step = 000439, loss = 0.001605
grad_step = 000440, loss = 0.001598
grad_step = 000441, loss = 0.001576
grad_step = 000442, loss = 0.001544
grad_step = 000443, loss = 0.001528
grad_step = 000444, loss = 0.001529
grad_step = 000445, loss = 0.001543
grad_step = 000446, loss = 0.001553
grad_step = 000447, loss = 0.001547
grad_step = 000448, loss = 0.001526
grad_step = 000449, loss = 0.001504
grad_step = 000450, loss = 0.001488
grad_step = 000451, loss = 0.001486
grad_step = 000452, loss = 0.001493
grad_step = 000453, loss = 0.001502
grad_step = 000454, loss = 0.001506
grad_step = 000455, loss = 0.001496
grad_step = 000456, loss = 0.001483
grad_step = 000457, loss = 0.001472
grad_step = 000458, loss = 0.001470
grad_step = 000459, loss = 0.001475
grad_step = 000460, loss = 0.001479
grad_step = 000461, loss = 0.001478
grad_step = 000462, loss = 0.001471
grad_step = 000463, loss = 0.001464
grad_step = 000464, loss = 0.001461
grad_step = 000465, loss = 0.001463
grad_step = 000466, loss = 0.001466
grad_step = 000467, loss = 0.001469
grad_step = 000468, loss = 0.001469
grad_step = 000469, loss = 0.001466
grad_step = 000470, loss = 0.001463
grad_step = 000471, loss = 0.001463
grad_step = 000472, loss = 0.001464
grad_step = 000473, loss = 0.001470
grad_step = 000474, loss = 0.001473
grad_step = 000475, loss = 0.001481
grad_step = 000476, loss = 0.001479
grad_step = 000477, loss = 0.001480
grad_step = 000478, loss = 0.001468
grad_step = 000479, loss = 0.001458
grad_step = 000480, loss = 0.001445
grad_step = 000481, loss = 0.001438
grad_step = 000482, loss = 0.001436
grad_step = 000483, loss = 0.001438
grad_step = 000484, loss = 0.001442
grad_step = 000485, loss = 0.001442
grad_step = 000486, loss = 0.001440
grad_step = 000487, loss = 0.001434
grad_step = 000488, loss = 0.001428
grad_step = 000489, loss = 0.001423
grad_step = 000490, loss = 0.001421
grad_step = 000491, loss = 0.001421
grad_step = 000492, loss = 0.001422
grad_step = 000493, loss = 0.001423
grad_step = 000494, loss = 0.001424
grad_step = 000495, loss = 0.001424
grad_step = 000496, loss = 0.001423
grad_step = 000497, loss = 0.001423
grad_step = 000498, loss = 0.001423
grad_step = 000499, loss = 0.001424
grad_step = 000500, loss = 0.001425
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001427
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

  date_run                              2020-05-17 12:14:41.216332
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.270401
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-17 12:14:41.222953
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.184911
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-17 12:14:41.230007
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.151208
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-17 12:14:41.235898
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.80979
metric_name                                             r2_score
Name: 11, dtype: object 

  


### Running {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepARTrainingNetwork: 26844
100%|██████████| 10/10 [00:02<00:00,  4.09it/s, avg_epoch_loss=5.21]
INFO:root:Epoch[0] Elapsed time 2.447 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.209227
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.20922703742981 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4ff82d7e80> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepFactorTrainingNetwork: 12466
100%|██████████| 10/10 [00:01<00:00,  7.99it/s, avg_epoch_loss=3.59e+3]
INFO:root:Epoch[0] Elapsed time 1.252 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=3590.403646
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 3590.4036458333335 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4f3f73e6a0> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU
INFO:gluonts.model.wavenet._estimator:Using dilation depth 10 and receptive field length 1024

  #### Fit  ####################################################### 
INFO:root:using training windows of length = 12
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in WaveNet: 97636
 30%|███       | 3/10 [00:12<00:28,  4.10s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:23<00:15,  3.96s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:34<00:03,  3.88s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:37<00:00,  3.80s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 37.964 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.866617
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.866617202758789 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4f281b7160> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in TransformerTrainingNetwork: 33911
100%|██████████| 10/10 [00:01<00:00,  5.61it/s, avg_epoch_loss=5.79]
INFO:root:Epoch[0] Elapsed time 1.783 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.790056
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.790056037902832 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4f281beda0> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepStateTrainingNetwork: 28054
 10%|█         | 1/10 [02:12<19:46, 131.80s/it, avg_epoch_loss=0.412] 20%|██        | 2/10 [05:13<19:33, 146.66s/it, avg_epoch_loss=0.399] 30%|███       | 3/10 [08:58<19:51, 170.15s/it, avg_epoch_loss=0.39]  40%|████      | 4/10 [12:37<18:30, 185.03s/it, avg_epoch_loss=0.385] 50%|█████     | 5/10 [16:16<16:16, 195.22s/it, avg_epoch_loss=0.384] 60%|██████    | 6/10 [20:11<13:47, 206.94s/it, avg_epoch_loss=0.383] 70%|███████   | 7/10 [23:58<10:39, 213.03s/it, avg_epoch_loss=0.381] 80%|████████  | 8/10 [27:53<07:19, 219.79s/it, avg_epoch_loss=0.378] 90%|█████████ | 9/10 [31:46<03:43, 223.55s/it, avg_epoch_loss=0.376]100%|██████████| 10/10 [35:53<00:00, 230.75s/it, avg_epoch_loss=0.375]100%|██████████| 10/10 [35:54<00:00, 215.41s/it, avg_epoch_loss=0.375]
INFO:root:Epoch[0] Elapsed time 2154.094 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.374571
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3745713621377945 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4f280b2128> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in GaussianProcessTrainingNetwork: 14
100%|██████████| 10/10 [00:02<00:00,  3.55it/s, avg_epoch_loss=415]
INFO:root:Epoch[0] Elapsed time 2.841 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=414.652022
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 414.65202175008733 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4f3f7cca20> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in SimpleFeedForwardTrainingNetwork: 20323
100%|██████████| 10/10 [00:00<00:00, 42.26it/s, avg_epoch_loss=5.15]
INFO:root:Epoch[0] Elapsed time 0.238 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.151470
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.1514702320098875 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f4f440c0438> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} 1 validation error for MLPEncoderModel
layer_sizes
  field required (type=value_error.missing) 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/timeseries/test02/model_list.json 

                        date_run  ...            metric_name
0   2020-05-17 12:14:08.999256  ...    mean_absolute_error
1   2020-05-17 12:14:09.002936  ...     mean_squared_error
2   2020-05-17 12:14:09.005992  ...  median_absolute_error
3   2020-05-17 12:14:09.009025  ...               r2_score
4   2020-05-17 12:14:18.118978  ...    mean_absolute_error
5   2020-05-17 12:14:18.123997  ...     mean_squared_error
6   2020-05-17 12:14:18.127619  ...  median_absolute_error
7   2020-05-17 12:14:18.131843  ...               r2_score
8   2020-05-17 12:14:41.216332  ...    mean_absolute_error
9   2020-05-17 12:14:41.222953  ...     mean_squared_error
10  2020-05-17 12:14:41.230007  ...  median_absolute_error
11  2020-05-17 12:14:41.235898  ...               r2_score

[12 rows x 6 columns] 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
    mpars['encoder'] = MLPEncoder()   #bug in seq2seq
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 424, in init_wrapper
    model = PydanticModel(**{**nmargs, **kwargs})
  File "pydantic/main.py", line 283, in pydantic.main.BaseModel.__init__
pydantic.error_wrappers.ValidationError: 1 validation error for MLPEncoderModel
layer_sizes
  field required (type=value_error.missing)
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa502564be0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4b1d789e8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4b4f21e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4b1d789e8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa502564be0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4b1d789e8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4b4f21e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4b1d789e8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa502564be0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4ac739fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 238, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 238, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 238, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 238, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 238, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 238, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 238, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 238, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 238, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 238, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa4b4f21e48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} 'data_info' 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/cnn/mnist 

  Empty DataFrame
Columns: [date_run, model_uri, json, dataset_uri, metric, metric_name]
Index: [] 
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 238, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f5066c4d198> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=dfd378eaec44a7c4128d49af02d4a7ae3338f29947cb3071ff175f074ecebd24
  Stored in directory: /tmp/pip-ephem-wheel-cache-drjjdkhz/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f4fff92aef0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 4472832/17464789 [======>.......................] - ETA: 0s
11468800/17464789 [==================>...........] - ETA: 0s
16408576/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-17 12:53:12.396936: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-17 12:53:12.412621: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-17 12:53:12.412814: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5559a491c280 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-17 12:53:12.412833: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 15s - loss: 7.5593 - accuracy: 0.5070
 2000/25000 [=>............................] - ETA: 11s - loss: 7.6590 - accuracy: 0.5005
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.6615 - accuracy: 0.5003 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.5785 - accuracy: 0.5058
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6636 - accuracy: 0.5002
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7024 - accuracy: 0.4977
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7170 - accuracy: 0.4967
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7356 - accuracy: 0.4955
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7416 - accuracy: 0.4951
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7310 - accuracy: 0.4958
11000/25000 [============>.................] - ETA: 4s - loss: 7.7335 - accuracy: 0.4956
12000/25000 [=============>................] - ETA: 4s - loss: 7.7318 - accuracy: 0.4958
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7209 - accuracy: 0.4965
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7071 - accuracy: 0.4974
15000/25000 [=================>............] - ETA: 3s - loss: 7.7249 - accuracy: 0.4962
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7030 - accuracy: 0.4976
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7000 - accuracy: 0.4978
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6888 - accuracy: 0.4986
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6852 - accuracy: 0.4988
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6751 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6608 - accuracy: 0.5004
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 10s 388us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-17 12:53:29.320413
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-17 12:53:29.320413  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:02<75:50:54, 3.16kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:02<53:19:26, 4.49kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:02<37:22:29, 6.41kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<26:09:20, 9.15kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:03<18:15:15, 13.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.38M/862M [00:03<12:41:39, 18.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.6M/862M [00:03<8:49:59, 26.7kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.0M/862M [00:03<6:09:36, 38.1kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.2M/862M [00:03<4:17:13, 54.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.5M/862M [00:03<2:59:28, 77.6kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.6M/862M [00:03<2:04:56, 111kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 35.1M/862M [00:03<1:27:12, 158kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.4M/862M [00:03<1:00:49, 225kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.6M/862M [00:04<42:28, 321kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 48.7M/862M [00:04<29:37, 458kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:04<21:23, 631kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:06<16:48, 798kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:06<14:05, 952kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:07<10:25, 1.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:08<09:35, 1.39MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:08<08:22, 1.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:09<06:16, 2.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:10<07:02, 1.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:10<07:44, 1.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:11<06:00, 2.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:11<04:24, 3.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:12<07:49, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:12<06:49, 1.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:13<05:06, 2.58MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:14<06:41, 1.96MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:14<07:24, 1.78MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:15<05:45, 2.28MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:15<04:09, 3.15MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:16<28:33, 458kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:16<21:20, 613kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:16<15:14, 856kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:18<13:41, 950kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:18<10:56, 1.19MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:18<07:58, 1.63MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:20<08:36, 1.50MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.0M/862M [00:20<07:20, 1.76MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.6M/862M [00:20<05:25, 2.38MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:22<06:50, 1.88MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:22<07:24, 1.74MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:22<05:46, 2.23MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<04:09, 3.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:24<21:17, 601kB/s] .vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:24<16:13, 789kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:24<11:39, 1.10MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:26<11:08, 1.14MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:26<10:29, 1.21MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:26<07:53, 1.61MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<05:40, 2.23MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<10:29, 1.21MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<08:39, 1.46MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:28<06:21, 1.99MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<07:24, 1.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:30<06:28, 1.95MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:30<04:47, 2.62MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<06:18, 1.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<05:40, 2.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:32<04:13, 2.95MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<05:56, 2.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<06:41, 1.86MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<05:13, 2.38MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<03:48, 3.26MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:36<11:35, 1.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<09:23, 1.32MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:36<06:52, 1.80MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<08:42, 1.41MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<07:17, 1.68MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<05:46, 2.12MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<04:10, 2.92MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<12:53, 946kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<11:24, 1.07MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:42<08:34, 1.42MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<07:57, 1.52MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<06:51, 1.77MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:44<05:06, 2.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<06:19, 1.90MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<06:52, 1.75MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:46<05:25, 2.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<05:43, 2.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:48<05:14, 2.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:48<03:58, 3.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<05:34, 2.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<06:19, 1.88MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<05:02, 2.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<03:39, 3.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<1:27:13, 136kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<1:02:14, 190kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:52<43:46, 270kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<33:18, 353kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<25:43, 458kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:54<18:35, 633kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<13:06, 893kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<1:32:57, 126kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<1:06:13, 177kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:56<46:33, 251kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<35:12, 331kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<27:06, 429kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:58<19:33, 594kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<13:46, 840kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<1:32:19, 125kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<1:05:47, 176kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:00<46:11, 250kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<34:57, 329kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:01<26:47, 429kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:02<19:14, 597kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<13:36, 842kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<14:30, 788kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:03<11:18, 1.01MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:04<08:09, 1.40MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<08:21, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:05<07:00, 1.62MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:06<05:08, 2.20MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<06:17, 1.80MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:07<05:32, 2.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:07<04:07, 2.73MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<05:33, 2.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:09<05:01, 2.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:09<03:47, 2.95MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<05:17, 2.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<04:50, 2.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<03:39, 3.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<05:10, 2.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:13<05:52, 1.89MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:13<04:37, 2.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<03:19, 3.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<45:35, 242kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<33:02, 333kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<23:20, 471kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:17<18:51, 581kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:17<14:18, 765kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<10:16, 1.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<09:43, 1.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<07:55, 1.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:19<05:46, 1.88MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<06:34, 1.65MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<06:47, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:21<05:12, 2.07MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<03:46, 2.86MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<10:36, 1.01MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:23<08:32, 1.26MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:23<06:14, 1.72MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<06:51, 1.56MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:25<05:52, 1.81MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:25<04:20, 2.45MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:27<05:32, 1.92MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:27<06:02, 1.76MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:27<04:45, 2.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<05:01, 2.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<04:36, 2.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<03:29, 3.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<04:52, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<04:28, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<03:20, 3.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<04:49, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<05:29, 1.90MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:33<04:17, 2.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<03:06, 3.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<11:26, 904kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<09:02, 1.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<06:34, 1.57MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<07:00, 1.47MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<05:56, 1.73MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<04:22, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<05:29, 1.86MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<05:55, 1.72MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:39<04:34, 2.22MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<03:20, 3.04MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<08:16, 1.22MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<06:54, 1.46MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:43<05:54, 1.70MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:43<04:23, 2.28MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:45<05:21, 1.86MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:45<05:54, 1.69MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:45<04:34, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<03:19, 2.98MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<07:03, 1.41MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<05:48, 1.71MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:47<04:19, 2.29MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<05:18, 1.86MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<05:43, 1.72MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:49<04:26, 2.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<03:12, 3.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<15:04, 650kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<11:33, 847kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:51<08:18, 1.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<08:05, 1.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<07:37, 1.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:53<05:50, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:11, 2.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<1:11:52, 134kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<51:06, 189kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:55<35:57, 268kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<25:11, 381kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<56:08, 171kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<41:15, 232kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<29:19, 326kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<20:31, 464kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:58<9:23:42, 16.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<6:35:18, 24.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<4:36:10, 34.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:00<3:14:48, 48.5kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<2:18:14, 68.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:00<1:37:03, 97.1kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<1:07:43, 139kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<56:09, 167kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:02<40:13, 233kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:02<28:19, 330kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<21:56, 424kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:04<17:14, 540kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<12:31, 742kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<10:13, 904kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<08:06, 1.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<05:53, 1.56MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:08<06:15, 1.47MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:08<05:18, 1.73MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<03:57, 2.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<04:54, 1.86MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:10<05:17, 1.72MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<04:05, 2.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<02:58, 3.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:12<06:56, 1.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<05:46, 1.56MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<04:15, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<05:04, 1.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<05:23, 1.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<04:10, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:00, 2.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:16<08:22, 1.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:16<06:47, 1.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<04:57, 1.79MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:18<05:31, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<04:44, 1.86MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:18<03:32, 2.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<04:32, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<04:08, 2.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<03:08, 2.78MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<04:12, 2.06MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<03:49, 2.27MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<02:56, 2.94MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:24<04:02, 2.13MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:24<03:55, 2.19MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<02:57, 2.90MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:26<04:02, 2.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:26<03:44, 2.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<02:49, 3.01MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<03:56, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<03:48, 2.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<02:49, 2.99MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<02:05, 4.02MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<1:39:22, 84.7kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:30<1:10:23, 119kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<49:22, 170kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:32<36:21, 230kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<26:17, 317kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<18:33, 448kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:34<14:52, 557kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<12:06, 684kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<08:52, 931kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<07:30, 1.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<06:06, 1.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:36<04:28, 1.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<05:00, 1.63MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<05:10, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:38<04:01, 2.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<04:06, 1.96MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<03:42, 2.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:40<02:48, 2.87MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<03:48, 2.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:42<04:17, 1.87MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<03:24, 2.35MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<03:39, 2.17MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<03:21, 2.36MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:44<02:33, 3.09MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<03:37, 2.17MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<04:04, 1.93MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<03:11, 2.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<02:19, 3.35MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<05:16, 1.48MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<04:21, 1.78MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<03:13, 2.42MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<02:20, 3.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:49<20:19, 380kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:49<15:47, 489kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<11:23, 677kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<08:01, 956kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:51<11:33, 663kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<08:52, 863kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<06:22, 1.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<06:13, 1.22MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<05:54, 1.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<04:31, 1.68MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:14, 2.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<56:01, 134kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<39:56, 188kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<28:01, 267kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<21:17, 350kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<16:24, 454kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<11:47, 631kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<08:18, 890kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:59<09:40, 764kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:59<07:31, 980kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<05:24, 1.36MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<05:29, 1.33MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:01<05:20, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<04:06, 1.78MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:03<04:01, 1.80MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<03:34, 2.03MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<02:40, 2.70MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:05<03:32, 2.02MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:05<03:57, 1.81MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:05<03:07, 2.29MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<03:20, 2.13MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<03:03, 2.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<02:18, 3.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<03:15, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:09<03:40, 1.92MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:09<03:16, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<02:27, 2.86MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:11<03:10, 2.19MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:11<02:57, 2.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<02:12, 3.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<03:09, 2.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<03:39, 1.89MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<02:51, 2.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<02:07, 3.24MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:15<03:36, 1.89MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:15<03:15, 2.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<02:26, 2.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<03:16, 2.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<03:40, 1.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:17<02:54, 2.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<03:06, 2.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<02:46, 2.41MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<02:07, 3.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<01:33, 4.26MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<14:00, 474kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<11:09, 594kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:21<08:06, 817kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<05:42, 1.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<14:09, 464kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<10:35, 619kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:23<07:33, 865kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:25<07:20, 884kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:27<05:42, 1.13MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:27<04:17, 1.49MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<03:07, 2.04MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<03:50, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<04:25, 1.43MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<03:27, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<02:32, 2.48MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:30<03:19, 1.89MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:31<02:58, 2.11MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:31<02:13, 2.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<01:37, 3.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<1:02:47, 99.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:33<45:11, 138kB/s]   .vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:33<31:50, 195kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<22:14, 277kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<18:36, 331kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:35<13:38, 450kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<09:40, 633kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<08:09, 747kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:36<06:59, 869kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<05:12, 1.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:41, 1.63MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<45:28, 132kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<32:25, 185kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:39<22:45, 263kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<17:13, 345kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<13:15, 448kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<09:31, 623kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<06:45, 874kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:42<06:26, 912kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<05:06, 1.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<03:42, 1.58MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<03:56, 1.47MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<03:22, 1.72MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<02:29, 2.31MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<03:04, 1.87MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<03:28, 1.65MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<02:59, 1.92MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<02:19, 2.46MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<01:43, 3.31MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<03:19, 1.71MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<03:02, 1.86MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<02:18, 2.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:50<02:43, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:50<02:36, 2.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<01:57, 2.84MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<01:27, 3.82MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:52<05:48, 952kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:52<04:44, 1.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<03:28, 1.58MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<03:31, 1.55MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<03:08, 1.74MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<02:19, 2.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<02:44, 1.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:56<02:28, 2.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<01:52, 2.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<02:31, 2.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<02:19, 2.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<01:45, 3.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<02:26, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<02:47, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<02:12, 2.37MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:02<02:22, 2.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:02<02:11, 2.37MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<01:39, 3.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<02:21, 2.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<02:10, 2.36MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<01:38, 3.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<02:20, 2.16MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<02:39, 1.90MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:06<02:07, 2.38MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<01:31, 3.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<05:40, 877kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<04:29, 1.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<03:15, 1.52MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<03:24, 1.44MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:10<02:53, 1.70MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<02:07, 2.29MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:12<02:37, 1.84MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<02:50, 1.70MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<02:13, 2.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<02:19, 2.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:14<02:07, 2.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<01:36, 2.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<02:13, 2.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<02:01, 2.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:16<01:31, 3.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:18<02:09, 2.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:18<01:59, 2.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<01:29, 3.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<02:07, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<01:57, 2.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<01:28, 3.08MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<02:05, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<01:55, 2.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<01:27, 3.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<02:03, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<02:20, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:24<01:49, 2.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<01:21, 3.23MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:11, 1.99MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<01:59, 2.19MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<01:28, 2.93MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:27<02:01, 2.12MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<01:48, 2.38MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<1:12:44, 59.0kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:29<50:39, 83.5kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<35:51, 118kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<25:02, 168kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:31<18:19, 227kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<13:14, 314kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:32<09:18, 444kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<07:24, 553kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<06:02, 676kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:34<04:23, 927kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<03:07, 1.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<03:39, 1.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<02:58, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:36<02:09, 1.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:37<02:25, 1.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:37<02:30, 1.58MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:38<01:55, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<01:22, 2.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<03:50, 1.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:39<03:04, 1.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<02:13, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<02:26, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<02:05, 1.82MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<01:32, 2.46MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:43<01:57, 1.92MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<01:44, 2.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<01:17, 2.87MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<01:46, 2.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<01:59, 1.85MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<01:33, 2.36MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:06, 3.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:47<05:51, 617kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<04:27, 809kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<03:10, 1.13MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<03:02, 1.17MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<02:50, 1.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<02:09, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:51<02:03, 1.69MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:51<01:47, 1.93MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:20, 2.57MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:53<01:43, 1.98MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:53<01:32, 2.20MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:09, 2.91MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:55<01:35, 2.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:55<01:27, 2.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<01:04, 3.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:57<01:31, 2.14MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:57<01:43, 1.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<01:22, 2.37MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<01:28, 2.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<01:21, 2.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<01:00, 3.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<01:26, 2.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<01:38, 1.90MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:01<01:16, 2.43MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<00:55, 3.34MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<03:07, 981kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:03<02:29, 1.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<01:48, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<01:57, 1.53MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:05<01:58, 1.52MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:05<01:31, 1.95MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<01:31, 1.92MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<01:21, 2.14MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:07<01:01, 2.84MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<01:22, 2.08MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<01:32, 1.85MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<01:12, 2.37MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<00:51, 3.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<02:12, 1.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<01:49, 1.52MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:20, 2.06MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<01:33, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<01:22, 1.99MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:01, 2.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:15<01:19, 1.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:15<01:28, 1.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:15<01:09, 2.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<01:12, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<01:06, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<00:50, 3.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:18<01:09, 2.16MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:19<01:19, 1.90MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<01:01, 2.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<00:44, 3.31MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:20<01:40, 1.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:21<01:25, 1.71MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<01:02, 2.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:22<01:16, 1.86MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:23<01:08, 2.09MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:23<00:50, 2.77MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<01:07, 2.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<01:15, 1.84MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:25<00:59, 2.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<01:02, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<00:57, 2.35MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<00:42, 3.09MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<01:00, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<01:08, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:29<00:53, 2.42MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<00:38, 3.31MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:30<01:44, 1.21MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:30<01:25, 1.46MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:31<01:02, 1.99MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:32<01:11, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:32<01:14, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<00:57, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<00:40, 2.90MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:34<01:40, 1.18MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<01:22, 1.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<00:59, 1.94MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<01:07, 1.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<01:10, 1.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<00:54, 2.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:38<00:55, 1.99MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:38<00:49, 2.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<00:36, 2.95MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:40<00:49, 2.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:40<00:56, 1.87MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:40<00:44, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:31, 3.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:42<06:59, 242kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:42<05:03, 334kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<03:31, 472kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:44<02:47, 582kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:44<02:16, 712kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<01:39, 967kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:08, 1.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:46<12:01, 129kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:46<08:32, 181kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<05:55, 257kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<04:23, 338kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<03:12, 461kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<02:14, 648kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<01:52, 759kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<01:26, 977kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:50<01:01, 1.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<01:01, 1.32MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<00:50, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:52<00:36, 2.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:43, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<00:38, 2.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:54<00:28, 2.63MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:20, 3.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<00:48, 1.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:56<00:42, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<00:31, 2.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<00:35, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<00:41, 1.66MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<00:32, 2.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:22, 2.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<00:36, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<00:32, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<00:23, 2.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:02<00:29, 2.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:02<00:32, 1.84MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:02<00:29, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<00:22, 2.67MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:15, 3.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:04<00:41, 1.35MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:04<00:36, 1.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:04<00:26, 2.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:27, 1.87MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:25, 2.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:18, 2.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:22, 2.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:25, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:08<00:20, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:14, 3.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:28, 1.53MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:10<00:24, 1.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:17, 2.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<00:20, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<00:18, 2.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:12<00:13, 2.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<00:17, 2.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<00:15, 2.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<00:11, 2.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:07, 4.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:15<02:02, 256kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<01:28, 353kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:59, 497kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:44, 608kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:33, 792kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:18<00:23, 1.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:20, 1.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:16, 1.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:20<00:11, 1.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:07, 2.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:21<03:47, 83.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<02:38, 118kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:22<01:41, 168kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:23<01:05, 227kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:46, 314kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:24<00:29, 442kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:25<00:19, 552kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:14, 720kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:08, 1.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:05, 1.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:27<00:09, 678kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:07, 881kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:28<00:03, 1.22MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:02, 1.23MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:29<00:01, 1.54MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 2.06MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<162:25:23,  1.46s/it]  0%|          | 796/400000 [00:01<113:28:28,  1.02s/it]  0%|          | 1585/400000 [00:01<79:16:45,  1.40it/s]  1%|          | 2418/400000 [00:01<55:23:00,  1.99it/s]  1%|          | 3295/400000 [00:01<38:41:12,  2.85it/s]  1%|          | 4074/400000 [00:01<27:01:54,  4.07it/s]  1%|          | 4927/400000 [00:02<18:53:07,  5.81it/s]  1%|▏         | 5696/400000 [00:02<13:11:53,  8.30it/s]  2%|▏         | 6496/400000 [00:02<9:13:26, 11.85it/s]   2%|▏         | 7275/400000 [00:02<6:26:53, 16.92it/s]  2%|▏         | 8055/400000 [00:02<4:30:32, 24.15it/s]  2%|▏         | 8830/400000 [00:02<3:09:15, 34.45it/s]  2%|▏         | 9637/400000 [00:02<2:12:26, 49.12it/s]  3%|▎         | 10480/400000 [00:02<1:32:44, 70.00it/s]  3%|▎         | 11345/400000 [00:02<1:05:00, 99.65it/s]  3%|▎         | 12165/400000 [00:02<45:39, 141.60it/s]   3%|▎         | 12977/400000 [00:03<32:07, 200.78it/s]  3%|▎         | 13789/400000 [00:03<22:41, 283.68it/s]  4%|▎         | 14590/400000 [00:03<16:05, 399.06it/s]  4%|▍         | 15414/400000 [00:03<11:28, 558.49it/s]  4%|▍         | 16240/400000 [00:03<08:14, 775.36it/s]  4%|▍         | 17059/400000 [00:03<05:59, 1064.43it/s]  4%|▍         | 17872/400000 [00:03<04:26, 1435.90it/s]  5%|▍         | 18703/400000 [00:03<03:19, 1909.84it/s]  5%|▍         | 19513/400000 [00:03<02:33, 2471.63it/s]  5%|▌         | 20317/400000 [00:03<02:02, 3105.08it/s]  5%|▌         | 21112/400000 [00:04<01:40, 3787.75it/s]  5%|▌         | 21902/400000 [00:04<01:24, 4450.59it/s]  6%|▌         | 22710/400000 [00:04<01:13, 5142.94it/s]  6%|▌         | 23516/400000 [00:04<01:05, 5768.28it/s]  6%|▌         | 24312/400000 [00:04<00:59, 6286.75it/s]  6%|▋         | 25131/400000 [00:04<00:55, 6756.46it/s]  6%|▋         | 25932/400000 [00:04<00:53, 6971.39it/s]  7%|▋         | 26722/400000 [00:04<00:51, 7225.57it/s]  7%|▋         | 27510/400000 [00:04<00:50, 7406.65it/s]  7%|▋         | 28298/400000 [00:05<00:50, 7414.78it/s]  7%|▋         | 29135/400000 [00:05<00:48, 7676.55it/s]  7%|▋         | 29945/400000 [00:05<00:47, 7797.21it/s]  8%|▊         | 30743/400000 [00:05<00:47, 7850.02it/s]  8%|▊         | 31582/400000 [00:05<00:46, 8004.30it/s]  8%|▊         | 32420/400000 [00:05<00:45, 8111.51it/s]  8%|▊         | 33239/400000 [00:05<00:45, 7991.87it/s]  9%|▊         | 34044/400000 [00:05<00:45, 8001.13it/s]  9%|▊         | 34848/400000 [00:05<00:46, 7814.68it/s]  9%|▉         | 35656/400000 [00:05<00:46, 7892.20it/s]  9%|▉         | 36534/400000 [00:06<00:44, 8138.39it/s]  9%|▉         | 37382/400000 [00:06<00:44, 8236.43it/s] 10%|▉         | 38209/400000 [00:06<00:44, 8194.58it/s] 10%|▉         | 39031/400000 [00:06<00:44, 8061.10it/s] 10%|▉         | 39840/400000 [00:06<00:45, 7922.59it/s] 10%|█         | 40635/400000 [00:06<00:45, 7928.95it/s] 10%|█         | 41430/400000 [00:06<00:45, 7895.66it/s] 11%|█         | 42221/400000 [00:06<00:45, 7854.27it/s] 11%|█         | 43024/400000 [00:06<00:45, 7904.46it/s] 11%|█         | 43850/400000 [00:06<00:44, 8006.43it/s] 11%|█         | 44704/400000 [00:07<00:43, 8158.02it/s] 11%|█▏        | 45547/400000 [00:07<00:43, 8234.01it/s] 12%|█▏        | 46372/400000 [00:07<00:43, 8143.59it/s] 12%|█▏        | 47193/400000 [00:07<00:43, 8162.68it/s] 12%|█▏        | 48010/400000 [00:07<00:43, 8083.70it/s] 12%|█▏        | 48820/400000 [00:07<00:44, 7979.89it/s] 12%|█▏        | 49642/400000 [00:07<00:43, 8049.21it/s] 13%|█▎        | 50485/400000 [00:07<00:42, 8158.29it/s] 13%|█▎        | 51370/400000 [00:07<00:41, 8351.44it/s] 13%|█▎        | 52259/400000 [00:07<00:40, 8505.14it/s] 13%|█▎        | 53165/400000 [00:08<00:40, 8663.20it/s] 14%|█▎        | 54070/400000 [00:08<00:39, 8773.92it/s] 14%|█▎        | 54950/400000 [00:08<00:39, 8708.31it/s] 14%|█▍        | 55823/400000 [00:08<00:39, 8604.84it/s] 14%|█▍        | 56685/400000 [00:08<00:40, 8414.73it/s] 14%|█▍        | 57529/400000 [00:08<00:41, 8285.19it/s] 15%|█▍        | 58360/400000 [00:08<00:41, 8150.48it/s] 15%|█▍        | 59177/400000 [00:08<00:41, 8117.18it/s] 15%|█▌        | 60008/400000 [00:08<00:41, 8172.26it/s] 15%|█▌        | 60832/400000 [00:08<00:41, 8191.04it/s] 15%|█▌        | 61655/400000 [00:09<00:41, 8201.92it/s] 16%|█▌        | 62476/400000 [00:09<00:41, 8127.42it/s] 16%|█▌        | 63290/400000 [00:09<00:41, 8092.81it/s] 16%|█▌        | 64100/400000 [00:09<00:41, 8012.63it/s] 16%|█▌        | 64902/400000 [00:09<00:42, 7882.90it/s] 16%|█▋        | 65692/400000 [00:09<00:42, 7854.04it/s] 17%|█▋        | 66515/400000 [00:09<00:41, 7960.34it/s] 17%|█▋        | 67320/400000 [00:09<00:41, 7984.48it/s] 17%|█▋        | 68119/400000 [00:09<00:41, 7949.28it/s] 17%|█▋        | 68957/400000 [00:09<00:41, 8072.86it/s] 17%|█▋        | 69788/400000 [00:10<00:40, 8139.92it/s] 18%|█▊        | 70620/400000 [00:10<00:40, 8191.66it/s] 18%|█▊        | 71440/400000 [00:10<00:40, 8078.65it/s] 18%|█▊        | 72282/400000 [00:10<00:40, 8178.03it/s] 18%|█▊        | 73101/400000 [00:10<00:39, 8172.86it/s] 18%|█▊        | 73964/400000 [00:10<00:39, 8300.34it/s] 19%|█▊        | 74815/400000 [00:10<00:38, 8360.97it/s] 19%|█▉        | 75652/400000 [00:10<00:39, 8163.35it/s] 19%|█▉        | 76470/400000 [00:10<00:39, 8124.66it/s] 19%|█▉        | 77321/400000 [00:11<00:39, 8233.18it/s] 20%|█▉        | 78172/400000 [00:11<00:38, 8313.62it/s] 20%|█▉        | 79005/400000 [00:11<00:38, 8253.57it/s] 20%|█▉        | 79832/400000 [00:11<00:38, 8228.89it/s] 20%|██        | 80656/400000 [00:11<00:39, 8164.12it/s] 20%|██        | 81473/400000 [00:11<00:39, 8116.97it/s] 21%|██        | 82286/400000 [00:11<00:39, 8089.82it/s] 21%|██        | 83104/400000 [00:11<00:39, 8116.26it/s] 21%|██        | 83916/400000 [00:11<00:39, 8013.44it/s] 21%|██        | 84745/400000 [00:11<00:38, 8092.12it/s] 21%|██▏       | 85583/400000 [00:12<00:38, 8174.69it/s] 22%|██▏       | 86423/400000 [00:12<00:38, 8240.78it/s] 22%|██▏       | 87252/400000 [00:12<00:37, 8252.90it/s] 22%|██▏       | 88078/400000 [00:12<00:38, 8162.30it/s] 22%|██▏       | 88895/400000 [00:12<00:38, 8084.00it/s] 22%|██▏       | 89704/400000 [00:12<00:38, 8057.41it/s] 23%|██▎       | 90513/400000 [00:12<00:38, 8064.05it/s] 23%|██▎       | 91343/400000 [00:12<00:37, 8132.70it/s] 23%|██▎       | 92157/400000 [00:12<00:38, 8065.20it/s] 23%|██▎       | 93010/400000 [00:12<00:37, 8196.99it/s] 23%|██▎       | 93851/400000 [00:13<00:37, 8259.55it/s] 24%|██▎       | 94678/400000 [00:13<00:37, 8247.53it/s] 24%|██▍       | 95524/400000 [00:13<00:36, 8307.84it/s] 24%|██▍       | 96356/400000 [00:13<00:37, 8201.99it/s] 24%|██▍       | 97191/400000 [00:13<00:36, 8245.32it/s] 25%|██▍       | 98017/400000 [00:13<00:36, 8191.00it/s] 25%|██▍       | 98837/400000 [00:13<00:37, 8072.55it/s] 25%|██▍       | 99662/400000 [00:13<00:36, 8124.55it/s] 25%|██▌       | 100476/400000 [00:13<00:37, 8094.77it/s] 25%|██▌       | 101289/400000 [00:13<00:36, 8104.22it/s] 26%|██▌       | 102119/400000 [00:14<00:36, 8160.38it/s] 26%|██▌       | 102936/400000 [00:14<00:36, 8064.73it/s] 26%|██▌       | 103743/400000 [00:14<00:37, 8003.99it/s] 26%|██▌       | 104544/400000 [00:14<00:36, 7991.38it/s] 26%|██▋       | 105357/400000 [00:14<00:36, 8030.31it/s] 27%|██▋       | 106161/400000 [00:14<00:36, 8028.03it/s] 27%|██▋       | 106982/400000 [00:14<00:36, 8080.20it/s] 27%|██▋       | 107801/400000 [00:14<00:36, 8112.62it/s] 27%|██▋       | 108613/400000 [00:14<00:36, 7954.67it/s] 27%|██▋       | 109430/400000 [00:14<00:36, 8015.91it/s] 28%|██▊       | 110264/400000 [00:15<00:35, 8108.67it/s] 28%|██▊       | 111079/400000 [00:15<00:35, 8120.46it/s] 28%|██▊       | 111892/400000 [00:15<00:35, 8082.45it/s] 28%|██▊       | 112701/400000 [00:15<00:35, 8033.08it/s] 28%|██▊       | 113525/400000 [00:15<00:35, 8093.77it/s] 29%|██▊       | 114345/400000 [00:15<00:35, 8123.58it/s] 29%|██▉       | 115158/400000 [00:15<00:35, 8082.56it/s] 29%|██▉       | 115967/400000 [00:15<00:35, 8082.13it/s] 29%|██▉       | 116776/400000 [00:15<00:35, 7967.72it/s] 29%|██▉       | 117574/400000 [00:15<00:35, 7880.31it/s] 30%|██▉       | 118389/400000 [00:16<00:35, 7958.08it/s] 30%|██▉       | 119211/400000 [00:16<00:34, 8034.27it/s] 30%|███       | 120041/400000 [00:16<00:34, 8109.88it/s] 30%|███       | 120853/400000 [00:16<00:34, 8066.63it/s] 30%|███       | 121708/400000 [00:16<00:33, 8203.98it/s] 31%|███       | 122530/400000 [00:16<00:34, 8114.82it/s] 31%|███       | 123366/400000 [00:16<00:33, 8184.27it/s] 31%|███       | 124186/400000 [00:16<00:33, 8177.71it/s] 31%|███▏      | 125005/400000 [00:16<00:34, 8043.45it/s] 31%|███▏      | 125811/400000 [00:16<00:34, 8032.00it/s] 32%|███▏      | 126640/400000 [00:17<00:33, 8106.66it/s] 32%|███▏      | 127466/400000 [00:17<00:33, 8151.81it/s] 32%|███▏      | 128300/400000 [00:17<00:33, 8204.67it/s] 32%|███▏      | 129121/400000 [00:17<00:33, 8048.09it/s] 32%|███▏      | 129927/400000 [00:17<00:33, 8006.21it/s] 33%|███▎      | 130729/400000 [00:17<00:33, 7921.00it/s] 33%|███▎      | 131543/400000 [00:17<00:33, 7982.91it/s] 33%|███▎      | 132356/400000 [00:17<00:33, 8025.77it/s] 33%|███▎      | 133160/400000 [00:17<00:33, 7908.47it/s] 33%|███▎      | 133984/400000 [00:18<00:33, 8001.86it/s] 34%|███▎      | 134806/400000 [00:18<00:32, 8065.09it/s] 34%|███▍      | 135615/400000 [00:18<00:32, 8068.08it/s] 34%|███▍      | 136424/400000 [00:18<00:32, 8073.99it/s] 34%|███▍      | 137232/400000 [00:18<00:32, 8040.06it/s] 35%|███▍      | 138055/400000 [00:18<00:32, 8093.79it/s] 35%|███▍      | 138871/400000 [00:18<00:32, 8111.28it/s] 35%|███▍      | 139683/400000 [00:18<00:32, 8031.37it/s] 35%|███▌      | 140487/400000 [00:18<00:32, 8016.22it/s] 35%|███▌      | 141289/400000 [00:18<00:32, 7923.78it/s] 36%|███▌      | 142115/400000 [00:19<00:32, 8021.70it/s] 36%|███▌      | 142922/400000 [00:19<00:31, 8034.71it/s] 36%|███▌      | 143726/400000 [00:19<00:32, 7995.58it/s] 36%|███▌      | 144541/400000 [00:19<00:31, 8039.24it/s] 36%|███▋      | 145346/400000 [00:19<00:31, 8025.64it/s] 37%|███▋      | 146166/400000 [00:19<00:31, 8075.80it/s] 37%|███▋      | 146986/400000 [00:19<00:31, 8110.20it/s] 37%|███▋      | 147798/400000 [00:19<00:32, 7693.17it/s] 37%|███▋      | 148582/400000 [00:19<00:32, 7735.72it/s] 37%|███▋      | 149359/400000 [00:19<00:33, 7566.54it/s] 38%|███▊      | 150119/400000 [00:20<00:33, 7497.30it/s] 38%|███▊      | 150872/400000 [00:20<00:33, 7371.84it/s] 38%|███▊      | 151612/400000 [00:20<00:33, 7334.93it/s] 38%|███▊      | 152348/400000 [00:20<00:33, 7299.53it/s] 38%|███▊      | 153129/400000 [00:20<00:33, 7443.12it/s] 38%|███▊      | 153970/400000 [00:20<00:31, 7707.55it/s] 39%|███▊      | 154745/400000 [00:20<00:32, 7582.76it/s] 39%|███▉      | 155529/400000 [00:20<00:31, 7658.11it/s] 39%|███▉      | 156297/400000 [00:20<00:31, 7618.35it/s] 39%|███▉      | 157061/400000 [00:20<00:32, 7455.52it/s] 39%|███▉      | 157852/400000 [00:21<00:31, 7583.82it/s] 40%|███▉      | 158691/400000 [00:21<00:30, 7808.48it/s] 40%|███▉      | 159503/400000 [00:21<00:30, 7898.73it/s] 40%|████      | 160303/400000 [00:21<00:30, 7928.73it/s] 40%|████      | 161099/400000 [00:21<00:30, 7937.01it/s] 40%|████      | 161895/400000 [00:21<00:29, 7943.75it/s] 41%|████      | 162706/400000 [00:21<00:29, 7990.41it/s] 41%|████      | 163527/400000 [00:21<00:29, 8052.71it/s] 41%|████      | 164333/400000 [00:21<00:29, 7970.67it/s] 41%|████▏     | 165131/400000 [00:21<00:29, 7877.84it/s] 41%|████▏     | 165921/400000 [00:22<00:29, 7884.26it/s] 42%|████▏     | 166729/400000 [00:22<00:29, 7940.74it/s] 42%|████▏     | 167559/400000 [00:22<00:28, 8042.86it/s] 42%|████▏     | 168402/400000 [00:22<00:28, 8154.83it/s] 42%|████▏     | 169219/400000 [00:22<00:28, 7986.33it/s] 43%|████▎     | 170019/400000 [00:22<00:28, 7937.23it/s] 43%|████▎     | 170880/400000 [00:22<00:28, 8127.16it/s] 43%|████▎     | 171754/400000 [00:22<00:27, 8301.24it/s] 43%|████▎     | 172615/400000 [00:22<00:27, 8387.57it/s] 43%|████▎     | 173456/400000 [00:23<00:27, 8190.28it/s] 44%|████▎     | 174333/400000 [00:23<00:27, 8355.81it/s] 44%|████▍     | 175218/400000 [00:23<00:26, 8497.05it/s] 44%|████▍     | 176106/400000 [00:23<00:26, 8607.14it/s] 44%|████▍     | 176969/400000 [00:23<00:25, 8610.21it/s] 44%|████▍     | 177832/400000 [00:23<00:26, 8326.89it/s] 45%|████▍     | 178668/400000 [00:23<00:27, 8051.07it/s] 45%|████▍     | 179478/400000 [00:23<00:27, 7990.32it/s] 45%|████▌     | 180310/400000 [00:23<00:27, 8085.72it/s] 45%|████▌     | 181121/400000 [00:23<00:27, 8031.24it/s] 45%|████▌     | 181931/400000 [00:24<00:27, 8051.30it/s] 46%|████▌     | 182738/400000 [00:24<00:26, 8047.65it/s] 46%|████▌     | 183544/400000 [00:24<00:27, 7914.59it/s] 46%|████▌     | 184374/400000 [00:24<00:26, 8026.27it/s] 46%|████▋     | 185178/400000 [00:24<00:26, 8016.25it/s] 47%|████▋     | 186010/400000 [00:24<00:26, 8104.18it/s] 47%|████▋     | 186865/400000 [00:24<00:25, 8232.77it/s] 47%|████▋     | 187691/400000 [00:24<00:25, 8239.40it/s] 47%|████▋     | 188516/400000 [00:24<00:25, 8193.96it/s] 47%|████▋     | 189336/400000 [00:24<00:26, 7904.52it/s] 48%|████▊     | 190161/400000 [00:25<00:26, 8003.83it/s] 48%|████▊     | 190964/400000 [00:25<00:26, 7762.85it/s] 48%|████▊     | 191770/400000 [00:25<00:26, 7848.20it/s] 48%|████▊     | 192613/400000 [00:25<00:25, 8013.90it/s] 48%|████▊     | 193417/400000 [00:25<00:25, 8015.33it/s] 49%|████▊     | 194298/400000 [00:25<00:24, 8236.71it/s] 49%|████▉     | 195190/400000 [00:25<00:24, 8430.24it/s] 49%|████▉     | 196108/400000 [00:25<00:23, 8641.02it/s] 49%|████▉     | 196976/400000 [00:25<00:24, 8370.17it/s] 49%|████▉     | 197818/400000 [00:25<00:24, 8281.21it/s] 50%|████▉     | 198673/400000 [00:26<00:24, 8358.52it/s] 50%|████▉     | 199546/400000 [00:26<00:23, 8465.83it/s] 50%|█████     | 200464/400000 [00:26<00:23, 8666.41it/s] 50%|█████     | 201352/400000 [00:26<00:22, 8727.12it/s] 51%|█████     | 202227/400000 [00:26<00:23, 8556.03it/s] 51%|█████     | 203095/400000 [00:26<00:22, 8592.49it/s] 51%|█████     | 203956/400000 [00:26<00:22, 8552.29it/s] 51%|█████     | 204813/400000 [00:26<00:23, 8256.50it/s] 51%|█████▏    | 205642/400000 [00:26<00:23, 8240.93it/s] 52%|█████▏    | 206469/400000 [00:27<00:23, 8174.20it/s] 52%|█████▏    | 207291/400000 [00:27<00:23, 8185.22it/s] 52%|█████▏    | 208111/400000 [00:27<00:23, 8170.83it/s] 52%|█████▏    | 208929/400000 [00:27<00:23, 8132.95it/s] 52%|█████▏    | 209743/400000 [00:27<00:23, 8055.30it/s] 53%|█████▎    | 210550/400000 [00:27<00:23, 7961.69it/s] 53%|█████▎    | 211349/400000 [00:27<00:23, 7969.54it/s] 53%|█████▎    | 212182/400000 [00:27<00:23, 8073.97it/s] 53%|█████▎    | 212991/400000 [00:27<00:23, 8006.03it/s] 53%|█████▎    | 213812/400000 [00:27<00:23, 8063.68it/s] 54%|█████▎    | 214619/400000 [00:28<00:23, 7931.27it/s] 54%|█████▍    | 215440/400000 [00:28<00:23, 8010.18it/s] 54%|█████▍    | 216253/400000 [00:28<00:22, 8043.81it/s] 54%|█████▍    | 217090/400000 [00:28<00:22, 8137.08it/s] 54%|█████▍    | 217920/400000 [00:28<00:22, 8182.11it/s] 55%|█████▍    | 218739/400000 [00:28<00:22, 8129.81it/s] 55%|█████▍    | 219553/400000 [00:28<00:22, 8109.59it/s] 55%|█████▌    | 220365/400000 [00:28<00:22, 8081.12it/s] 55%|█████▌    | 221192/400000 [00:28<00:21, 8135.42it/s] 56%|█████▌    | 222006/400000 [00:28<00:22, 8022.22it/s] 56%|█████▌    | 222809/400000 [00:29<00:22, 7978.26it/s] 56%|█████▌    | 223622/400000 [00:29<00:21, 8021.32it/s] 56%|█████▌    | 224425/400000 [00:29<00:22, 7949.84it/s] 56%|█████▋    | 225227/400000 [00:29<00:21, 7969.25it/s] 57%|█████▋    | 226026/400000 [00:29<00:21, 7975.26it/s] 57%|█████▋    | 226824/400000 [00:29<00:21, 7968.17it/s] 57%|█████▋    | 227621/400000 [00:29<00:21, 7955.65it/s] 57%|█████▋    | 228435/400000 [00:29<00:21, 8007.94it/s] 57%|█████▋    | 229247/400000 [00:29<00:21, 8039.04it/s] 58%|█████▊    | 230052/400000 [00:29<00:21, 8008.86it/s] 58%|█████▊    | 230870/400000 [00:30<00:20, 8057.86it/s] 58%|█████▊    | 231676/400000 [00:30<00:20, 8040.04it/s] 58%|█████▊    | 232497/400000 [00:30<00:20, 8087.80it/s] 58%|█████▊    | 233310/400000 [00:30<00:20, 8099.42it/s] 59%|█████▊    | 234121/400000 [00:30<00:20, 8029.10it/s] 59%|█████▊    | 234925/400000 [00:30<00:20, 7971.87it/s] 59%|█████▉    | 235744/400000 [00:30<00:20, 8034.93it/s] 59%|█████▉    | 236613/400000 [00:30<00:19, 8218.91it/s] 59%|█████▉    | 237439/400000 [00:30<00:19, 8231.03it/s] 60%|█████▉    | 238263/400000 [00:30<00:19, 8227.93it/s] 60%|█████▉    | 239087/400000 [00:31<00:20, 8042.15it/s] 60%|█████▉    | 239893/400000 [00:31<00:20, 8004.27it/s] 60%|██████    | 240734/400000 [00:31<00:19, 8120.62it/s] 60%|██████    | 241590/400000 [00:31<00:19, 8245.42it/s] 61%|██████    | 242434/400000 [00:31<00:18, 8300.88it/s] 61%|██████    | 243266/400000 [00:31<00:19, 8202.57it/s] 61%|██████    | 244097/400000 [00:31<00:18, 8233.21it/s] 61%|██████    | 244923/400000 [00:31<00:18, 8240.83it/s] 61%|██████▏   | 245771/400000 [00:31<00:18, 8308.89it/s] 62%|██████▏   | 246606/400000 [00:31<00:18, 8320.93it/s] 62%|██████▏   | 247439/400000 [00:32<00:18, 8292.45it/s] 62%|██████▏   | 248280/400000 [00:32<00:18, 8325.55it/s] 62%|██████▏   | 249113/400000 [00:32<00:18, 8087.20it/s] 62%|██████▏   | 249979/400000 [00:32<00:18, 8250.76it/s] 63%|██████▎   | 250807/400000 [00:32<00:18, 8242.46it/s] 63%|██████▎   | 251633/400000 [00:32<00:18, 8034.83it/s] 63%|██████▎   | 252451/400000 [00:32<00:18, 8076.69it/s] 63%|██████▎   | 253274/400000 [00:32<00:18, 8120.80it/s] 64%|██████▎   | 254088/400000 [00:32<00:18, 7978.41it/s] 64%|██████▎   | 254888/400000 [00:32<00:18, 7977.51it/s] 64%|██████▍   | 255687/400000 [00:33<00:18, 7920.68it/s] 64%|██████▍   | 256495/400000 [00:33<00:18, 7966.42it/s] 64%|██████▍   | 257341/400000 [00:33<00:17, 8107.10it/s] 65%|██████▍   | 258201/400000 [00:33<00:17, 8246.42it/s] 65%|██████▍   | 259027/400000 [00:33<00:17, 8209.92it/s] 65%|██████▍   | 259849/400000 [00:33<00:17, 8111.75it/s] 65%|██████▌   | 260689/400000 [00:33<00:17, 8193.67it/s] 65%|██████▌   | 261547/400000 [00:33<00:16, 8305.39it/s] 66%|██████▌   | 262379/400000 [00:33<00:16, 8294.44it/s] 66%|██████▌   | 263215/400000 [00:34<00:16, 8311.28it/s] 66%|██████▌   | 264047/400000 [00:34<00:16, 8212.63it/s] 66%|██████▌   | 264869/400000 [00:34<00:16, 8101.24it/s] 66%|██████▋   | 265695/400000 [00:34<00:16, 8146.09it/s] 67%|██████▋   | 266532/400000 [00:34<00:16, 8211.26it/s] 67%|██████▋   | 267362/400000 [00:34<00:16, 8235.21it/s] 67%|██████▋   | 268186/400000 [00:34<00:16, 8209.12it/s] 67%|██████▋   | 269008/400000 [00:34<00:16, 8182.41it/s] 67%|██████▋   | 269851/400000 [00:34<00:15, 8253.42it/s] 68%|██████▊   | 270677/400000 [00:34<00:15, 8169.66it/s] 68%|██████▊   | 271495/400000 [00:35<00:15, 8105.68it/s] 68%|██████▊   | 272306/400000 [00:35<00:15, 8031.33it/s] 68%|██████▊   | 273126/400000 [00:35<00:15, 8081.09it/s] 68%|██████▊   | 273984/400000 [00:35<00:15, 8222.97it/s] 69%|██████▊   | 274831/400000 [00:35<00:15, 8292.84it/s] 69%|██████▉   | 275662/400000 [00:35<00:15, 8283.86it/s] 69%|██████▉   | 276491/400000 [00:35<00:15, 8167.30it/s] 69%|██████▉   | 277309/400000 [00:35<00:15, 8152.47it/s] 70%|██████▉   | 278125/400000 [00:35<00:14, 8143.31it/s] 70%|██████▉   | 278940/400000 [00:35<00:15, 7963.26it/s] 70%|██████▉   | 279738/400000 [00:36<00:15, 7966.32it/s] 70%|███████   | 280536/400000 [00:36<00:15, 7955.95it/s] 70%|███████   | 281333/400000 [00:36<00:14, 7946.16it/s] 71%|███████   | 282136/400000 [00:36<00:14, 7970.08it/s] 71%|███████   | 282949/400000 [00:36<00:14, 8015.75it/s] 71%|███████   | 283751/400000 [00:36<00:14, 8000.52it/s] 71%|███████   | 284552/400000 [00:36<00:14, 7995.41it/s] 71%|███████▏  | 285352/400000 [00:36<00:14, 7898.92it/s] 72%|███████▏  | 286161/400000 [00:36<00:14, 7954.88it/s] 72%|███████▏  | 286978/400000 [00:36<00:14, 8017.88it/s] 72%|███████▏  | 287781/400000 [00:37<00:14, 8006.58it/s] 72%|███████▏  | 288611/400000 [00:37<00:13, 8091.74it/s] 72%|███████▏  | 289421/400000 [00:37<00:13, 8017.63it/s] 73%|███████▎  | 290224/400000 [00:37<00:13, 7915.34it/s] 73%|███████▎  | 291017/400000 [00:37<00:13, 7864.37it/s] 73%|███████▎  | 291804/400000 [00:37<00:13, 7849.98it/s] 73%|███████▎  | 292590/400000 [00:37<00:13, 7803.10it/s] 73%|███████▎  | 293371/400000 [00:37<00:13, 7778.29it/s] 74%|███████▎  | 294176/400000 [00:37<00:13, 7856.43it/s] 74%|███████▎  | 294975/400000 [00:37<00:13, 7895.84it/s] 74%|███████▍  | 295786/400000 [00:38<00:13, 7956.68it/s] 74%|███████▍  | 296601/400000 [00:38<00:12, 8012.83it/s] 74%|███████▍  | 297403/400000 [00:38<00:12, 7907.88it/s] 75%|███████▍  | 298235/400000 [00:38<00:12, 8026.59it/s] 75%|███████▍  | 299070/400000 [00:38<00:12, 8119.48it/s] 75%|███████▍  | 299903/400000 [00:38<00:12, 8179.65it/s] 75%|███████▌  | 300722/400000 [00:38<00:12, 8164.64it/s] 75%|███████▌  | 301539/400000 [00:38<00:12, 8054.63it/s] 76%|███████▌  | 302352/400000 [00:38<00:12, 8074.51it/s] 76%|███████▌  | 303160/400000 [00:38<00:12, 8066.67it/s] 76%|███████▌  | 303975/400000 [00:39<00:11, 8090.41it/s] 76%|███████▌  | 304785/400000 [00:39<00:11, 8022.10it/s] 76%|███████▋  | 305588/400000 [00:39<00:12, 7867.33it/s] 77%|███████▋  | 306388/400000 [00:39<00:11, 7905.89it/s] 77%|███████▋  | 307180/400000 [00:39<00:11, 7903.03it/s] 77%|███████▋  | 307988/400000 [00:39<00:11, 7954.08it/s] 77%|███████▋  | 308791/400000 [00:39<00:11, 7976.70it/s] 77%|███████▋  | 309589/400000 [00:39<00:11, 7944.03it/s] 78%|███████▊  | 310398/400000 [00:39<00:11, 7987.23it/s] 78%|███████▊  | 311197/400000 [00:39<00:11, 7973.74it/s] 78%|███████▊  | 311995/400000 [00:40<00:11, 7926.41it/s] 78%|███████▊  | 312814/400000 [00:40<00:10, 8002.73it/s] 78%|███████▊  | 313615/400000 [00:40<00:10, 7935.54it/s] 79%|███████▊  | 314426/400000 [00:40<00:10, 7985.99it/s] 79%|███████▉  | 315225/400000 [00:40<00:10, 7925.18it/s] 79%|███████▉  | 316020/400000 [00:40<00:10, 7930.02it/s] 79%|███████▉  | 316814/400000 [00:40<00:10, 7921.23it/s] 79%|███████▉  | 317607/400000 [00:40<00:10, 7904.10it/s] 80%|███████▉  | 318398/400000 [00:40<00:10, 7850.51it/s] 80%|███████▉  | 319184/400000 [00:40<00:10, 7717.72it/s] 80%|███████▉  | 319980/400000 [00:41<00:10, 7788.65it/s] 80%|████████  | 320782/400000 [00:41<00:10, 7855.17it/s] 80%|████████  | 321569/400000 [00:41<00:10, 7750.21it/s] 81%|████████  | 322409/400000 [00:41<00:09, 7932.31it/s] 81%|████████  | 323261/400000 [00:41<00:09, 8099.86it/s] 81%|████████  | 324073/400000 [00:41<00:09, 8076.60it/s] 81%|████████  | 324911/400000 [00:41<00:09, 8164.06it/s] 81%|████████▏ | 325729/400000 [00:41<00:09, 8118.10it/s] 82%|████████▏ | 326548/400000 [00:41<00:09, 8138.04it/s] 82%|████████▏ | 327398/400000 [00:42<00:08, 8243.15it/s] 82%|████████▏ | 328224/400000 [00:42<00:08, 8221.57it/s] 82%|████████▏ | 329047/400000 [00:42<00:08, 8221.16it/s] 82%|████████▏ | 329870/400000 [00:42<00:08, 8139.17it/s] 83%|████████▎ | 330685/400000 [00:42<00:08, 8064.31it/s] 83%|████████▎ | 331492/400000 [00:42<00:08, 7745.46it/s] 83%|████████▎ | 332270/400000 [00:42<00:08, 7721.78it/s] 83%|████████▎ | 333103/400000 [00:42<00:08, 7894.46it/s] 83%|████████▎ | 333895/400000 [00:42<00:08, 7900.39it/s] 84%|████████▎ | 334692/400000 [00:42<00:08, 7919.28it/s] 84%|████████▍ | 335550/400000 [00:43<00:07, 8104.65it/s] 84%|████████▍ | 336371/400000 [00:43<00:07, 8134.76it/s] 84%|████████▍ | 337187/400000 [00:43<00:07, 8141.57it/s] 85%|████████▍ | 338034/400000 [00:43<00:07, 8235.82it/s] 85%|████████▍ | 338859/400000 [00:43<00:07, 8215.81it/s] 85%|████████▍ | 339682/400000 [00:43<00:07, 8171.71it/s] 85%|████████▌ | 340507/400000 [00:43<00:07, 8192.33it/s] 85%|████████▌ | 341333/400000 [00:43<00:07, 8210.99it/s] 86%|████████▌ | 342204/400000 [00:43<00:06, 8352.19it/s] 86%|████████▌ | 343041/400000 [00:43<00:06, 8253.42it/s] 86%|████████▌ | 343900/400000 [00:44<00:06, 8349.75it/s] 86%|████████▌ | 344736/400000 [00:44<00:06, 8337.21it/s] 86%|████████▋ | 345571/400000 [00:44<00:06, 8075.91it/s] 87%|████████▋ | 346400/400000 [00:44<00:06, 8136.76it/s] 87%|████████▋ | 347216/400000 [00:44<00:06, 8061.77it/s] 87%|████████▋ | 348052/400000 [00:44<00:06, 8147.73it/s] 87%|████████▋ | 348899/400000 [00:44<00:06, 8240.20it/s] 87%|████████▋ | 349731/400000 [00:44<00:06, 8263.41it/s] 88%|████████▊ | 350580/400000 [00:44<00:05, 8327.59it/s] 88%|████████▊ | 351414/400000 [00:44<00:05, 8295.40it/s] 88%|████████▊ | 352285/400000 [00:45<00:05, 8413.31it/s] 88%|████████▊ | 353128/400000 [00:45<00:05, 8319.19it/s] 88%|████████▊ | 353961/400000 [00:45<00:05, 8138.49it/s] 89%|████████▊ | 354782/400000 [00:45<00:05, 8158.04it/s] 89%|████████▉ | 355599/400000 [00:45<00:05, 8071.63it/s] 89%|████████▉ | 356408/400000 [00:45<00:05, 7994.58it/s] 89%|████████▉ | 357209/400000 [00:45<00:05, 7931.00it/s] 90%|████████▉ | 358011/400000 [00:45<00:05, 7956.61it/s] 90%|████████▉ | 358808/400000 [00:45<00:05, 7563.94it/s] 90%|████████▉ | 359569/400000 [00:45<00:05, 7528.93it/s] 90%|█████████ | 360354/400000 [00:46<00:05, 7622.36it/s] 90%|█████████ | 361133/400000 [00:46<00:05, 7670.65it/s] 90%|█████████ | 361953/400000 [00:46<00:04, 7820.87it/s] 91%|█████████ | 362778/400000 [00:46<00:04, 7940.07it/s] 91%|█████████ | 363574/400000 [00:46<00:04, 7904.94it/s] 91%|█████████ | 364377/400000 [00:46<00:04, 7940.17it/s] 91%|█████████▏| 365185/400000 [00:46<00:04, 7978.96it/s] 91%|█████████▏| 365994/400000 [00:46<00:04, 8009.82it/s] 92%|█████████▏| 366805/400000 [00:46<00:04, 8038.34it/s] 92%|█████████▏| 367610/400000 [00:47<00:04, 7989.34it/s] 92%|█████████▏| 368410/400000 [00:47<00:03, 7924.88it/s] 92%|█████████▏| 369252/400000 [00:47<00:03, 8066.81it/s] 93%|█████████▎| 370100/400000 [00:47<00:03, 8186.04it/s] 93%|█████████▎| 370943/400000 [00:47<00:03, 8255.16it/s] 93%|█████████▎| 371770/400000 [00:47<00:03, 8103.82it/s] 93%|█████████▎| 372582/400000 [00:47<00:03, 8031.94it/s] 93%|█████████▎| 373391/400000 [00:47<00:03, 8048.74it/s] 94%|█████████▎| 374222/400000 [00:47<00:03, 8122.46it/s] 94%|█████████▍| 375036/400000 [00:47<00:03, 8125.47it/s] 94%|█████████▍| 375850/400000 [00:48<00:02, 8067.79it/s] 94%|█████████▍| 376688/400000 [00:48<00:02, 8157.55it/s] 94%|█████████▍| 377505/400000 [00:48<00:02, 8144.29it/s] 95%|█████████▍| 378320/400000 [00:48<00:02, 8118.84it/s] 95%|█████████▍| 379133/400000 [00:48<00:02, 8097.79it/s] 95%|█████████▍| 379943/400000 [00:48<00:02, 7893.15it/s] 95%|█████████▌| 380755/400000 [00:48<00:02, 7958.16it/s] 95%|█████████▌| 381555/400000 [00:48<00:02, 7969.49it/s] 96%|█████████▌| 382389/400000 [00:48<00:02, 8076.10it/s] 96%|█████████▌| 383207/400000 [00:48<00:02, 8103.03it/s] 96%|█████████▌| 384018/400000 [00:49<00:01, 8070.42it/s] 96%|█████████▌| 384860/400000 [00:49<00:01, 8170.80it/s] 96%|█████████▋| 385678/400000 [00:49<00:01, 8138.05it/s] 97%|█████████▋| 386493/400000 [00:49<00:01, 8087.65it/s] 97%|█████████▋| 387330/400000 [00:49<00:01, 8168.74it/s] 97%|█████████▋| 388148/400000 [00:49<00:01, 8064.34it/s] 97%|█████████▋| 388976/400000 [00:49<00:01, 8126.48it/s] 97%|█████████▋| 389798/400000 [00:49<00:01, 8153.55it/s] 98%|█████████▊| 390622/400000 [00:49<00:01, 8178.37it/s] 98%|█████████▊| 391454/400000 [00:49<00:01, 8220.24it/s] 98%|█████████▊| 392277/400000 [00:50<00:00, 8161.75it/s] 98%|█████████▊| 393122/400000 [00:50<00:00, 8245.68it/s] 98%|█████████▊| 393961/400000 [00:50<00:00, 8286.32it/s] 99%|█████████▊| 394817/400000 [00:50<00:00, 8364.31it/s] 99%|█████████▉| 395668/400000 [00:50<00:00, 8406.80it/s] 99%|█████████▉| 396510/400000 [00:50<00:00, 8356.26it/s] 99%|█████████▉| 397366/400000 [00:50<00:00, 8414.48it/s]100%|█████████▉| 398245/400000 [00:50<00:00, 8522.57it/s]100%|█████████▉| 399099/400000 [00:50<00:00, 8525.97it/s]100%|█████████▉| 399952/400000 [00:50<00:00, 8308.04it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7849.33it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f1d3011e940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.012124871320819991 	 Accuracy: 49
Train Epoch: 1 	 Loss: 0.01086499001270154 	 Accuracy: 76

  model saves at 76% accuracy 

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
2020-05-17 13:02:35.654474: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-17 13:02:35.659411: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-17 13:02:35.659578: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55819e5ee0f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-17 13:02:35.659593: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1cd9532908> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6513 - accuracy: 0.5010
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4673 - accuracy: 0.5130 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.4366 - accuracy: 0.5150
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.3830 - accuracy: 0.5185
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5041 - accuracy: 0.5106
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5848 - accuracy: 0.5053
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5374 - accuracy: 0.5084
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5478 - accuracy: 0.5077
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5678 - accuracy: 0.5064
10000/25000 [===========>..................] - ETA: 5s - loss: 7.5562 - accuracy: 0.5072
11000/25000 [============>.................] - ETA: 4s - loss: 7.5454 - accuracy: 0.5079
12000/25000 [=============>................] - ETA: 4s - loss: 7.5478 - accuracy: 0.5077
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5675 - accuracy: 0.5065
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5812 - accuracy: 0.5056
15000/25000 [=================>............] - ETA: 3s - loss: 7.5900 - accuracy: 0.5050
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5832 - accuracy: 0.5054
17000/25000 [===================>..........] - ETA: 2s - loss: 7.5764 - accuracy: 0.5059
18000/25000 [====================>.........] - ETA: 2s - loss: 7.5866 - accuracy: 0.5052
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6239 - accuracy: 0.5028
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6344 - accuracy: 0.5021
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6374 - accuracy: 0.5019
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6290 - accuracy: 0.5025
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 10s 393us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f1c94e555f8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f1cac0d8358> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.6742 - crf_viterbi_accuracy: 0.1467 - val_loss: 1.5721 - val_crf_viterbi_accuracy: 0.1067

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

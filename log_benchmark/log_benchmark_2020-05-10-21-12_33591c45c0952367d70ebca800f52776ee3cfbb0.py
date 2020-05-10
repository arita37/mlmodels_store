
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/33591c45c0952367d70ebca800f52776ee3cfbb0', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '33591c45c0952367d70ebca800f52776ee3cfbb0', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/33591c45c0952367d70ebca800f52776ee3cfbb0

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/33591c45c0952367d70ebca800f52776ee3cfbb0

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f0007c88fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 21:12:27.765664
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-10 21:12:27.769659
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-10 21:12:27.772881
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-10 21:12:27.776223
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f0013a4c3c8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 357024.0625
Epoch 2/10

1/1 [==============================] - 0s 100ms/step - loss: 269605.6562
Epoch 3/10

1/1 [==============================] - 0s 110ms/step - loss: 165005.1094
Epoch 4/10

1/1 [==============================] - 0s 97ms/step - loss: 97025.4375
Epoch 5/10

1/1 [==============================] - 0s 94ms/step - loss: 55500.0508
Epoch 6/10

1/1 [==============================] - 0s 96ms/step - loss: 31187.4844
Epoch 7/10

1/1 [==============================] - 0s 99ms/step - loss: 17892.9629
Epoch 8/10

1/1 [==============================] - 0s 92ms/step - loss: 11026.7656
Epoch 9/10

1/1 [==============================] - 0s 102ms/step - loss: 7377.8076
Epoch 10/10

1/1 [==============================] - 0s 93ms/step - loss: 5334.9902

  #### Inference Need return ypred, ytrue ######################### 
[[ 8.64271343e-01  5.98760605e-01  3.27289343e-01  4.74041194e-01
  -3.03047538e-01 -2.09214211e-01 -1.24892533e+00  5.55433989e-01
   3.96339327e-01  3.27399015e-01 -1.05112982e+00 -1.60870814e+00
  -2.06119865e-01 -2.22286010e+00  5.67355812e-01 -6.13937378e-01
   6.46741927e-01  3.39593053e-01 -4.06524748e-01  2.96848714e-02
  -3.43599319e-02  2.36988604e-01  5.17393470e-01  2.27866173e-01
  -3.32456589e-01  9.28280532e-01  1.06559563e+00 -4.78443056e-01
  -4.72484887e-01  9.44818437e-01 -1.00483716e+00 -7.15736568e-01
  -7.82482266e-01  1.89910567e+00  8.26460540e-01 -1.88117683e+00
   9.21241403e-01 -7.07429469e-01  1.08998902e-01 -1.11840403e+00
  -5.61541915e-02  3.88085842e-03  2.22607732e-01 -1.34015012e+00
   1.27570510e+00  2.78222978e-01  3.19518894e-01  1.07015920e+00
   2.98398614e-01  8.19809213e-02  1.21676230e+00 -1.27271438e+00
  -3.68017524e-01  7.00317860e-01  1.32813096e-01 -1.27611899e+00
   3.63698184e-01  4.28492069e-01 -6.04676187e-01 -4.08516526e-01
  -2.87749767e-01  8.03838634e+00  1.08811159e+01  9.13156509e+00
   9.23987198e+00  9.94490337e+00  1.10545282e+01  9.39412403e+00
   1.03733139e+01  8.53457928e+00  9.27170372e+00  9.26672268e+00
   9.18501091e+00  8.83470535e+00  1.04082289e+01  8.20659733e+00
   8.65927124e+00  9.37516975e+00  1.06393766e+01  9.19125462e+00
   9.02062702e+00  1.08268147e+01  9.77917957e+00  1.04979038e+01
   1.09636984e+01  8.98726177e+00  9.06884098e+00  9.66674900e+00
   8.93405247e+00  9.42108059e+00  9.72228718e+00  1.04531364e+01
   7.85652637e+00  9.62272549e+00  1.07315712e+01  9.44602585e+00
   8.84535789e+00  1.15348215e+01  9.71672535e+00  9.52615452e+00
   1.06468487e+01  9.38287544e+00  7.85611343e+00  9.41446495e+00
   1.12217016e+01  1.17404461e+01  8.68915749e+00  8.82837105e+00
   1.03056841e+01  9.24108505e+00  9.59935951e+00  8.26312256e+00
   1.11133194e+01  8.60929394e+00  1.05921602e+01  1.01321621e+01
   9.72437382e+00  9.88063717e+00  8.88800049e+00  7.90365553e+00
  -4.57750827e-01 -1.07659352e+00  3.42823863e-02 -1.54186308e-01
   6.42185450e-01  9.85133469e-01  2.08662724e+00  1.23445714e+00
   1.68473554e+00  2.18277860e+00 -2.17854530e-01  5.20689189e-01
  -5.21008968e-02  5.57806134e-01  4.74967808e-01  6.20195687e-01
   1.85759592e+00  2.43750811e-02 -1.61016452e+00  6.01540446e-01
  -6.75461471e-01  1.46763778e+00  2.11805868e+00 -2.86222219e-01
   4.27893579e-01  1.08635545e+00 -1.35028720e-01 -2.21181273e-01
   2.14955419e-01 -1.25198817e+00  3.61596644e-01 -1.04603100e+00
   4.53549087e-01  1.15271175e+00  6.72413051e-01 -5.42747498e-01
   8.94402742e-01 -4.41841543e-01  7.39786506e-01 -1.19571686e+00
   1.31777614e-01  4.02151346e-01 -3.26632023e-01 -7.81495273e-01
   8.93587470e-02 -1.20642483e+00 -2.23723602e+00  7.36547172e-01
   1.19719654e-01 -1.86506212e-01  4.27474827e-01 -1.17419410e+00
  -5.90836406e-01  9.60597992e-01 -7.51513839e-01  2.09372473e+00
   1.62077713e+00 -1.14022911e+00  2.16610861e+00  3.76819789e-01
   5.22037923e-01  1.94191313e+00  1.81987989e+00  8.86427879e-01
   2.01717901e+00  3.33286810e+00  1.36048222e+00  1.04884720e+00
   1.44421077e+00  1.39519989e-01  3.36803198e+00  2.18855762e+00
   1.14554095e+00  1.75913906e+00  4.01910603e-01  1.76662588e+00
   3.10099483e-01  1.38217998e+00  1.17927217e+00  1.98135257e-01
   7.34619319e-01  1.26729727e+00  5.19489884e-01  8.83795381e-01
   2.33500481e+00  2.17726469e-01  2.92607403e+00  1.77390039e-01
   1.10566437e+00  1.63356519e+00  1.39573801e+00  4.63190913e-01
   1.14570153e+00  1.59104967e+00  1.03129840e+00  1.64086974e+00
   6.63573027e-01  1.05275869e+00  1.35223532e+00  1.38711929e+00
   5.93859196e-01  9.35044706e-01  2.80718148e-01  5.59772134e-01
   1.90053225e+00  6.85405850e-01  4.36253071e-01  2.03927088e+00
   2.92853355e-01  8.94563138e-01  1.51798916e+00  6.51372492e-01
   2.48717022e+00  1.76576853e-01  2.96979284e+00  2.99867868e+00
   1.75905955e+00  2.14485812e+00  5.92571259e-01  6.49115443e-02
   1.76794767e-01  9.39346886e+00  8.61750889e+00  9.52438354e+00
   9.67262077e+00  1.02886515e+01  1.09940443e+01  8.45762539e+00
   9.54696465e+00  9.06931019e+00  1.13240309e+01  1.06268425e+01
   9.02376175e+00  1.00310755e+01  9.74792957e+00  1.13462305e+01
   9.40924263e+00  7.18405676e+00  9.02384853e+00  9.10846138e+00
   1.00464334e+01  1.02084913e+01  8.16783047e+00  8.86344242e+00
   8.47847462e+00  9.10166931e+00  1.13568459e+01  8.53624725e+00
   1.13094091e+01  1.05241108e+01  1.07075930e+01  8.62426281e+00
   1.05963993e+01  9.00810814e+00  1.01337643e+01  8.68277550e+00
   1.05517197e+01  1.02669239e+01  8.99349213e+00  1.07779942e+01
   7.66811895e+00  9.02977848e+00  9.95410156e+00  8.31801414e+00
   9.29315090e+00  1.05635548e+01  1.11632910e+01  9.09098721e+00
   1.10062475e+01  1.01008053e+01  9.84873962e+00  1.01946869e+01
   1.15333195e+01  1.00823755e+01  1.02534504e+01  8.80276203e+00
   7.55007982e+00  1.07209826e+01  9.32793713e+00  1.07285566e+01
   2.34444594e+00  7.31044769e-01  2.35345030e+00  1.82225382e+00
   9.60134685e-01  2.17648602e+00  1.81039894e+00  7.65239000e-01
   7.79651880e-01  3.00786686e+00  7.83414602e-01  9.40665007e-02
   1.12802982e-01  8.40087295e-01  1.05783343e+00  3.57864380e-01
   1.80205762e-01  7.05590189e-01  3.85885656e-01  3.25399160e-01
   1.19282007e+00  8.61883163e-02  6.14424169e-01  1.11652017e+00
   5.32011747e-01  1.48311234e+00  7.50288844e-01  5.34358799e-01
   2.37112999e-01  7.06639349e-01  5.73397875e-01  1.33655930e+00
   1.81388402e+00  2.89657950e-01  2.79293609e+00  1.89061236e+00
   6.89216316e-01  1.14689207e+00  1.79486990e+00  4.24231291e-01
   1.67508960e-01  2.25995588e+00  1.05506325e+00  8.76681685e-01
   7.49158978e-01  7.27695465e-01  1.69950092e+00  1.92720592e+00
   4.20154572e-01  1.58100700e+00  2.17902040e+00  1.54995978e+00
   9.37419057e-01  2.32210970e+00  1.40974438e+00  2.59103715e-01
   1.21410120e+00  1.08747864e+00  1.32140279e+00  4.90468860e-01
  -1.01131744e+01  8.70510101e+00 -6.05623388e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 21:12:37.340461
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.3713
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-10 21:12:37.344207
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8556.79
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-10 21:12:37.347482
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.1615
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-10 21:12:37.350642
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -765.321
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139637765315440
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139636538076296
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139636538076800
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139636537675960
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139636537676464
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139636537676968

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f00079883c8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.572447
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.544043
grad_step = 000002, loss = 0.525373
grad_step = 000003, loss = 0.507106
grad_step = 000004, loss = 0.488841
grad_step = 000005, loss = 0.472697
grad_step = 000006, loss = 0.461720
grad_step = 000007, loss = 0.450993
grad_step = 000008, loss = 0.439853
grad_step = 000009, loss = 0.427267
grad_step = 000010, loss = 0.415444
grad_step = 000011, loss = 0.405272
grad_step = 000012, loss = 0.395689
grad_step = 000013, loss = 0.385630
grad_step = 000014, loss = 0.374744
grad_step = 000015, loss = 0.363275
grad_step = 000016, loss = 0.351948
grad_step = 000017, loss = 0.341401
grad_step = 000018, loss = 0.331440
grad_step = 000019, loss = 0.321357
grad_step = 000020, loss = 0.310954
grad_step = 000021, loss = 0.300548
grad_step = 000022, loss = 0.290404
grad_step = 000023, loss = 0.280704
grad_step = 000024, loss = 0.271341
grad_step = 000025, loss = 0.261896
grad_step = 000026, loss = 0.252243
grad_step = 000027, loss = 0.242860
grad_step = 000028, loss = 0.234048
grad_step = 000029, loss = 0.225536
grad_step = 000030, loss = 0.216949
grad_step = 000031, loss = 0.208369
grad_step = 000032, loss = 0.200161
grad_step = 000033, loss = 0.192378
grad_step = 000034, loss = 0.184719
grad_step = 000035, loss = 0.177080
grad_step = 000036, loss = 0.169693
grad_step = 000037, loss = 0.162684
grad_step = 000038, loss = 0.155945
grad_step = 000039, loss = 0.149232
grad_step = 000040, loss = 0.142654
grad_step = 000041, loss = 0.136448
grad_step = 000042, loss = 0.130448
grad_step = 000043, loss = 0.124537
grad_step = 000044, loss = 0.118836
grad_step = 000045, loss = 0.113418
grad_step = 000046, loss = 0.108173
grad_step = 000047, loss = 0.103028
grad_step = 000048, loss = 0.098114
grad_step = 000049, loss = 0.093454
grad_step = 000050, loss = 0.088915
grad_step = 000051, loss = 0.084501
grad_step = 000052, loss = 0.080340
grad_step = 000053, loss = 0.076363
grad_step = 000054, loss = 0.072501
grad_step = 000055, loss = 0.068813
grad_step = 000056, loss = 0.065330
grad_step = 000057, loss = 0.061977
grad_step = 000058, loss = 0.058758
grad_step = 000059, loss = 0.055730
grad_step = 000060, loss = 0.052833
grad_step = 000061, loss = 0.050050
grad_step = 000062, loss = 0.047428
grad_step = 000063, loss = 0.044937
grad_step = 000064, loss = 0.042552
grad_step = 000065, loss = 0.040303
grad_step = 000066, loss = 0.038174
grad_step = 000067, loss = 0.036136
grad_step = 000068, loss = 0.034221
grad_step = 000069, loss = 0.032410
grad_step = 000070, loss = 0.030681
grad_step = 000071, loss = 0.029054
grad_step = 000072, loss = 0.027517
grad_step = 000073, loss = 0.026052
grad_step = 000074, loss = 0.024675
grad_step = 000075, loss = 0.023371
grad_step = 000076, loss = 0.022131
grad_step = 000077, loss = 0.020965
grad_step = 000078, loss = 0.019859
grad_step = 000079, loss = 0.018806
grad_step = 000080, loss = 0.017816
grad_step = 000081, loss = 0.016873
grad_step = 000082, loss = 0.015980
grad_step = 000083, loss = 0.015137
grad_step = 000084, loss = 0.014334
grad_step = 000085, loss = 0.013575
grad_step = 000086, loss = 0.012854
grad_step = 000087, loss = 0.012170
grad_step = 000088, loss = 0.011524
grad_step = 000089, loss = 0.010909
grad_step = 000090, loss = 0.010328
grad_step = 000091, loss = 0.009777
grad_step = 000092, loss = 0.009098
grad_step = 000093, loss = 0.008564
grad_step = 000094, loss = 0.008074
grad_step = 000095, loss = 0.007601
grad_step = 000096, loss = 0.007192
grad_step = 000097, loss = 0.006784
grad_step = 000098, loss = 0.006433
grad_step = 000099, loss = 0.006079
grad_step = 000100, loss = 0.005734
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.005431
grad_step = 000102, loss = 0.005157
grad_step = 000103, loss = 0.004893
grad_step = 000104, loss = 0.004629
grad_step = 000105, loss = 0.004328
grad_step = 000106, loss = 0.004069
grad_step = 000107, loss = 0.003872
grad_step = 000108, loss = 0.003704
grad_step = 000109, loss = 0.003554
grad_step = 000110, loss = 0.003373
grad_step = 000111, loss = 0.003192
grad_step = 000112, loss = 0.003033
grad_step = 000113, loss = 0.002913
grad_step = 000114, loss = 0.002819
grad_step = 000115, loss = 0.002742
grad_step = 000116, loss = 0.002650
grad_step = 000117, loss = 0.002550
grad_step = 000118, loss = 0.002445
grad_step = 000119, loss = 0.002357
grad_step = 000120, loss = 0.002291
grad_step = 000121, loss = 0.002243
grad_step = 000122, loss = 0.002211
grad_step = 000123, loss = 0.002184
grad_step = 000124, loss = 0.002158
grad_step = 000125, loss = 0.002131
grad_step = 000126, loss = 0.002084
grad_step = 000127, loss = 0.002032
grad_step = 000128, loss = 0.001981
grad_step = 000129, loss = 0.001944
grad_step = 000130, loss = 0.001921
grad_step = 000131, loss = 0.001912
grad_step = 000132, loss = 0.001910
grad_step = 000133, loss = 0.001917
grad_step = 000134, loss = 0.001925
grad_step = 000135, loss = 0.001936
grad_step = 000136, loss = 0.001936
grad_step = 000137, loss = 0.001916
grad_step = 000138, loss = 0.001880
grad_step = 000139, loss = 0.001842
grad_step = 000140, loss = 0.001814
grad_step = 000141, loss = 0.001803
grad_step = 000142, loss = 0.001807
grad_step = 000143, loss = 0.001818
grad_step = 000144, loss = 0.001833
grad_step = 000145, loss = 0.001845
grad_step = 000146, loss = 0.001849
grad_step = 000147, loss = 0.001839
grad_step = 000148, loss = 0.001817
grad_step = 000149, loss = 0.001789
grad_step = 000150, loss = 0.001766
grad_step = 000151, loss = 0.001755
grad_step = 000152, loss = 0.001756
grad_step = 000153, loss = 0.001765
grad_step = 000154, loss = 0.001776
grad_step = 000155, loss = 0.001786
grad_step = 000156, loss = 0.001792
grad_step = 000157, loss = 0.001791
grad_step = 000158, loss = 0.001782
grad_step = 000159, loss = 0.001772
grad_step = 000160, loss = 0.001763
grad_step = 000161, loss = 0.001767
grad_step = 000162, loss = 0.001770
grad_step = 000163, loss = 0.001788
grad_step = 000164, loss = 0.001792
grad_step = 000165, loss = 0.001803
grad_step = 000166, loss = 0.001784
grad_step = 000167, loss = 0.001779
grad_step = 000168, loss = 0.001774
grad_step = 000169, loss = 0.001770
grad_step = 000170, loss = 0.001760
grad_step = 000171, loss = 0.001724
grad_step = 000172, loss = 0.001702
grad_step = 000173, loss = 0.001705
grad_step = 000174, loss = 0.001720
grad_step = 000175, loss = 0.001733
grad_step = 000176, loss = 0.001729
grad_step = 000177, loss = 0.001710
grad_step = 000178, loss = 0.001686
grad_step = 000179, loss = 0.001679
grad_step = 000180, loss = 0.001689
grad_step = 000181, loss = 0.001699
grad_step = 000182, loss = 0.001702
grad_step = 000183, loss = 0.001697
grad_step = 000184, loss = 0.001692
grad_step = 000185, loss = 0.001680
grad_step = 000186, loss = 0.001673
grad_step = 000187, loss = 0.001674
grad_step = 000188, loss = 0.001680
grad_step = 000189, loss = 0.001685
grad_step = 000190, loss = 0.001688
grad_step = 000191, loss = 0.001697
grad_step = 000192, loss = 0.001701
grad_step = 000193, loss = 0.001711
grad_step = 000194, loss = 0.001721
grad_step = 000195, loss = 0.001747
grad_step = 000196, loss = 0.001754
grad_step = 000197, loss = 0.001767
grad_step = 000198, loss = 0.001744
grad_step = 000199, loss = 0.001709
grad_step = 000200, loss = 0.001668
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001648
grad_step = 000202, loss = 0.001659
grad_step = 000203, loss = 0.001679
grad_step = 000204, loss = 0.001692
grad_step = 000205, loss = 0.001684
grad_step = 000206, loss = 0.001666
grad_step = 000207, loss = 0.001641
grad_step = 000208, loss = 0.001633
grad_step = 000209, loss = 0.001640
grad_step = 000210, loss = 0.001649
grad_step = 000211, loss = 0.001656
grad_step = 000212, loss = 0.001654
grad_step = 000213, loss = 0.001648
grad_step = 000214, loss = 0.001637
grad_step = 000215, loss = 0.001626
grad_step = 000216, loss = 0.001623
grad_step = 000217, loss = 0.001627
grad_step = 000218, loss = 0.001632
grad_step = 000219, loss = 0.001639
grad_step = 000220, loss = 0.001648
grad_step = 000221, loss = 0.001663
grad_step = 000222, loss = 0.001683
grad_step = 000223, loss = 0.001728
grad_step = 000224, loss = 0.001764
grad_step = 000225, loss = 0.001825
grad_step = 000226, loss = 0.001768
grad_step = 000227, loss = 0.001690
grad_step = 000228, loss = 0.001608
grad_step = 000229, loss = 0.001618
grad_step = 000230, loss = 0.001679
grad_step = 000231, loss = 0.001674
grad_step = 000232, loss = 0.001629
grad_step = 000233, loss = 0.001600
grad_step = 000234, loss = 0.001627
grad_step = 000235, loss = 0.001658
grad_step = 000236, loss = 0.001629
grad_step = 000237, loss = 0.001600
grad_step = 000238, loss = 0.001603
grad_step = 000239, loss = 0.001621
grad_step = 000240, loss = 0.001625
grad_step = 000241, loss = 0.001604
grad_step = 000242, loss = 0.001595
grad_step = 000243, loss = 0.001605
grad_step = 000244, loss = 0.001618
grad_step = 000245, loss = 0.001627
grad_step = 000246, loss = 0.001615
grad_step = 000247, loss = 0.001620
grad_step = 000248, loss = 0.001641
grad_step = 000249, loss = 0.001656
grad_step = 000250, loss = 0.001660
grad_step = 000251, loss = 0.001639
grad_step = 000252, loss = 0.001616
grad_step = 000253, loss = 0.001600
grad_step = 000254, loss = 0.001572
grad_step = 000255, loss = 0.001559
grad_step = 000256, loss = 0.001556
grad_step = 000257, loss = 0.001555
grad_step = 000258, loss = 0.001563
grad_step = 000259, loss = 0.001577
grad_step = 000260, loss = 0.001588
grad_step = 000261, loss = 0.001587
grad_step = 000262, loss = 0.001576
grad_step = 000263, loss = 0.001564
grad_step = 000264, loss = 0.001554
grad_step = 000265, loss = 0.001541
grad_step = 000266, loss = 0.001530
grad_step = 000267, loss = 0.001524
grad_step = 000268, loss = 0.001521
grad_step = 000269, loss = 0.001520
grad_step = 000270, loss = 0.001523
grad_step = 000271, loss = 0.001529
grad_step = 000272, loss = 0.001537
grad_step = 000273, loss = 0.001540
grad_step = 000274, loss = 0.001543
grad_step = 000275, loss = 0.001546
grad_step = 000276, loss = 0.001549
grad_step = 000277, loss = 0.001555
grad_step = 000278, loss = 0.001556
grad_step = 000279, loss = 0.001563
grad_step = 000280, loss = 0.001558
grad_step = 000281, loss = 0.001552
grad_step = 000282, loss = 0.001532
grad_step = 000283, loss = 0.001514
grad_step = 000284, loss = 0.001497
grad_step = 000285, loss = 0.001488
grad_step = 000286, loss = 0.001488
grad_step = 000287, loss = 0.001495
grad_step = 000288, loss = 0.001508
grad_step = 000289, loss = 0.001520
grad_step = 000290, loss = 0.001539
grad_step = 000291, loss = 0.001553
grad_step = 000292, loss = 0.001564
grad_step = 000293, loss = 0.001557
grad_step = 000294, loss = 0.001542
grad_step = 000295, loss = 0.001516
grad_step = 000296, loss = 0.001496
grad_step = 000297, loss = 0.001482
grad_step = 000298, loss = 0.001482
grad_step = 000299, loss = 0.001491
grad_step = 000300, loss = 0.001495
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001496
grad_step = 000302, loss = 0.001490
grad_step = 000303, loss = 0.001486
grad_step = 000304, loss = 0.001480
grad_step = 000305, loss = 0.001473
grad_step = 000306, loss = 0.001471
grad_step = 000307, loss = 0.001473
grad_step = 000308, loss = 0.001473
grad_step = 000309, loss = 0.001471
grad_step = 000310, loss = 0.001469
grad_step = 000311, loss = 0.001466
grad_step = 000312, loss = 0.001462
grad_step = 000313, loss = 0.001457
grad_step = 000314, loss = 0.001452
grad_step = 000315, loss = 0.001450
grad_step = 000316, loss = 0.001450
grad_step = 000317, loss = 0.001453
grad_step = 000318, loss = 0.001462
grad_step = 000319, loss = 0.001487
grad_step = 000320, loss = 0.001536
grad_step = 000321, loss = 0.001635
grad_step = 000322, loss = 0.001760
grad_step = 000323, loss = 0.001913
grad_step = 000324, loss = 0.001819
grad_step = 000325, loss = 0.001661
grad_step = 000326, loss = 0.001471
grad_step = 000327, loss = 0.001512
grad_step = 000328, loss = 0.001654
grad_step = 000329, loss = 0.001572
grad_step = 000330, loss = 0.001463
grad_step = 000331, loss = 0.001496
grad_step = 000332, loss = 0.001568
grad_step = 000333, loss = 0.001542
grad_step = 000334, loss = 0.001448
grad_step = 000335, loss = 0.001467
grad_step = 000336, loss = 0.001513
grad_step = 000337, loss = 0.001471
grad_step = 000338, loss = 0.001446
grad_step = 000339, loss = 0.001467
grad_step = 000340, loss = 0.001459
grad_step = 000341, loss = 0.001447
grad_step = 000342, loss = 0.001455
grad_step = 000343, loss = 0.001439
grad_step = 000344, loss = 0.001422
grad_step = 000345, loss = 0.001431
grad_step = 000346, loss = 0.001435
grad_step = 000347, loss = 0.001418
grad_step = 000348, loss = 0.001407
grad_step = 000349, loss = 0.001422
grad_step = 000350, loss = 0.001425
grad_step = 000351, loss = 0.001404
grad_step = 000352, loss = 0.001398
grad_step = 000353, loss = 0.001410
grad_step = 000354, loss = 0.001407
grad_step = 000355, loss = 0.001395
grad_step = 000356, loss = 0.001395
grad_step = 000357, loss = 0.001398
grad_step = 000358, loss = 0.001393
grad_step = 000359, loss = 0.001387
grad_step = 000360, loss = 0.001390
grad_step = 000361, loss = 0.001391
grad_step = 000362, loss = 0.001385
grad_step = 000363, loss = 0.001379
grad_step = 000364, loss = 0.001380
grad_step = 000365, loss = 0.001380
grad_step = 000366, loss = 0.001377
grad_step = 000367, loss = 0.001373
grad_step = 000368, loss = 0.001373
grad_step = 000369, loss = 0.001374
grad_step = 000370, loss = 0.001371
grad_step = 000371, loss = 0.001368
grad_step = 000372, loss = 0.001367
grad_step = 000373, loss = 0.001367
grad_step = 000374, loss = 0.001366
grad_step = 000375, loss = 0.001364
grad_step = 000376, loss = 0.001362
grad_step = 000377, loss = 0.001363
grad_step = 000378, loss = 0.001365
grad_step = 000379, loss = 0.001372
grad_step = 000380, loss = 0.001390
grad_step = 000381, loss = 0.001432
grad_step = 000382, loss = 0.001494
grad_step = 000383, loss = 0.001582
grad_step = 000384, loss = 0.001622
grad_step = 000385, loss = 0.001558
grad_step = 000386, loss = 0.001448
grad_step = 000387, loss = 0.001370
grad_step = 000388, loss = 0.001390
grad_step = 000389, loss = 0.001433
grad_step = 000390, loss = 0.001425
grad_step = 000391, loss = 0.001393
grad_step = 000392, loss = 0.001359
grad_step = 000393, loss = 0.001362
grad_step = 000394, loss = 0.001385
grad_step = 000395, loss = 0.001382
grad_step = 000396, loss = 0.001358
grad_step = 000397, loss = 0.001338
grad_step = 000398, loss = 0.001352
grad_step = 000399, loss = 0.001372
grad_step = 000400, loss = 0.001353
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001330
grad_step = 000402, loss = 0.001333
grad_step = 000403, loss = 0.001353
grad_step = 000404, loss = 0.001365
grad_step = 000405, loss = 0.001352
grad_step = 000406, loss = 0.001354
grad_step = 000407, loss = 0.001376
grad_step = 000408, loss = 0.001402
grad_step = 000409, loss = 0.001422
grad_step = 000410, loss = 0.001427
grad_step = 000411, loss = 0.001424
grad_step = 000412, loss = 0.001422
grad_step = 000413, loss = 0.001376
grad_step = 000414, loss = 0.001333
grad_step = 000415, loss = 0.001304
grad_step = 000416, loss = 0.001302
grad_step = 000417, loss = 0.001314
grad_step = 000418, loss = 0.001324
grad_step = 000419, loss = 0.001333
grad_step = 000420, loss = 0.001331
grad_step = 000421, loss = 0.001313
grad_step = 000422, loss = 0.001292
grad_step = 000423, loss = 0.001281
grad_step = 000424, loss = 0.001284
grad_step = 000425, loss = 0.001294
grad_step = 000426, loss = 0.001299
grad_step = 000427, loss = 0.001297
grad_step = 000428, loss = 0.001295
grad_step = 000429, loss = 0.001291
grad_step = 000430, loss = 0.001295
grad_step = 000431, loss = 0.001306
grad_step = 000432, loss = 0.001341
grad_step = 000433, loss = 0.001405
grad_step = 000434, loss = 0.001484
grad_step = 000435, loss = 0.001539
grad_step = 000436, loss = 0.001512
grad_step = 000437, loss = 0.001378
grad_step = 000438, loss = 0.001285
grad_step = 000439, loss = 0.001280
grad_step = 000440, loss = 0.001342
grad_step = 000441, loss = 0.001377
grad_step = 000442, loss = 0.001319
grad_step = 000443, loss = 0.001259
grad_step = 000444, loss = 0.001254
grad_step = 000445, loss = 0.001298
grad_step = 000446, loss = 0.001322
grad_step = 000447, loss = 0.001275
grad_step = 000448, loss = 0.001240
grad_step = 000449, loss = 0.001255
grad_step = 000450, loss = 0.001281
grad_step = 000451, loss = 0.001285
grad_step = 000452, loss = 0.001268
grad_step = 000453, loss = 0.001266
grad_step = 000454, loss = 0.001281
grad_step = 000455, loss = 0.001285
grad_step = 000456, loss = 0.001305
grad_step = 000457, loss = 0.001301
grad_step = 000458, loss = 0.001282
grad_step = 000459, loss = 0.001239
grad_step = 000460, loss = 0.001218
grad_step = 000461, loss = 0.001222
grad_step = 000462, loss = 0.001225
grad_step = 000463, loss = 0.001215
grad_step = 000464, loss = 0.001215
grad_step = 000465, loss = 0.001220
grad_step = 000466, loss = 0.001207
grad_step = 000467, loss = 0.001193
grad_step = 000468, loss = 0.001191
grad_step = 000469, loss = 0.001190
grad_step = 000470, loss = 0.001185
grad_step = 000471, loss = 0.001185
grad_step = 000472, loss = 0.001194
grad_step = 000473, loss = 0.001203
grad_step = 000474, loss = 0.001211
grad_step = 000475, loss = 0.001222
grad_step = 000476, loss = 0.001252
grad_step = 000477, loss = 0.001273
grad_step = 000478, loss = 0.001312
grad_step = 000479, loss = 0.001310
grad_step = 000480, loss = 0.001294
grad_step = 000481, loss = 0.001237
grad_step = 000482, loss = 0.001175
grad_step = 000483, loss = 0.001146
grad_step = 000484, loss = 0.001157
grad_step = 000485, loss = 0.001186
grad_step = 000486, loss = 0.001206
grad_step = 000487, loss = 0.001209
grad_step = 000488, loss = 0.001182
grad_step = 000489, loss = 0.001159
grad_step = 000490, loss = 0.001148
grad_step = 000491, loss = 0.001159
grad_step = 000492, loss = 0.001173
grad_step = 000493, loss = 0.001174
grad_step = 000494, loss = 0.001163
grad_step = 000495, loss = 0.001145
grad_step = 000496, loss = 0.001124
grad_step = 000497, loss = 0.001109
grad_step = 000498, loss = 0.001103
grad_step = 000499, loss = 0.001110
grad_step = 000500, loss = 0.001133
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001176
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

  date_run                              2020-05-10 21:12:56.182670
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.320021
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-10 21:12:56.188397
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.312305
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-10 21:12:56.195670
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.142944
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-10 21:12:56.200806
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -3.74558
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
0   2020-05-10 21:12:27.765664  ...    mean_absolute_error
1   2020-05-10 21:12:27.769659  ...     mean_squared_error
2   2020-05-10 21:12:27.772881  ...  median_absolute_error
3   2020-05-10 21:12:27.776223  ...               r2_score
4   2020-05-10 21:12:37.340461  ...    mean_absolute_error
5   2020-05-10 21:12:37.344207  ...     mean_squared_error
6   2020-05-10 21:12:37.347482  ...  median_absolute_error
7   2020-05-10 21:12:37.350642  ...               r2_score
8   2020-05-10 21:12:56.182670  ...    mean_absolute_error
9   2020-05-10 21:12:56.188397  ...     mean_squared_error
10  2020-05-10 21:12:56.195670  ...  median_absolute_error
11  2020-05-10 21:12:56.200806  ...               r2_score

[12 rows x 6 columns] 
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 36%|███▋      | 3596288/9912422 [00:00<00:00, 35262177.00it/s]9920512it [00:00, 36534522.97it/s]                             
0it [00:00, ?it/s]32768it [00:00, 575395.43it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160553.90it/s]1654784it [00:00, 11218073.83it/s]                         
0it [00:00, ?it/s]8192it [00:00, 166428.06it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5355c3cba8> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f52f3391d30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5355bffe80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f52f3391cc0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5355c48908> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5355c48fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f52f338c080> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5355c3cba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f52f3391cc0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5355c3cba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f52f338cef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc96bf031d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=5205b1b8785b9a1152ef19c289b4dab70ec4ff3cce5c70a952bf95b456016e85
  Stored in directory: /tmp/pip-ephem-wheel-cache-f2as0bvd/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc904bf2860> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2727936/17464789 [===>..........................] - ETA: 0s
11419648/17464789 [==================>...........] - ETA: 0s
16506880/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-10 21:14:22.207494: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 21:14:22.212020: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-10 21:14:22.212218: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ce53275460 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 21:14:22.212234: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.9120 - accuracy: 0.4840
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7510 - accuracy: 0.4945 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6564 - accuracy: 0.5007
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5593 - accuracy: 0.5070
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6114 - accuracy: 0.5036
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6487 - accuracy: 0.5012
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5987 - accuracy: 0.5044
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5631 - accuracy: 0.5067
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5525 - accuracy: 0.5074
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5685 - accuracy: 0.5064
11000/25000 [============>.................] - ETA: 3s - loss: 7.6304 - accuracy: 0.5024
12000/25000 [=============>................] - ETA: 3s - loss: 7.5989 - accuracy: 0.5044
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5970 - accuracy: 0.5045
14000/25000 [===============>..............] - ETA: 2s - loss: 7.5790 - accuracy: 0.5057
15000/25000 [=================>............] - ETA: 2s - loss: 7.5838 - accuracy: 0.5054
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5765 - accuracy: 0.5059
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6197 - accuracy: 0.5031
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6453 - accuracy: 0.5014
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6368 - accuracy: 0.5019
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6597 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6747 - accuracy: 0.4995
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6736 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6600 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 7s 272us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-10 21:14:35.523273
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-10 21:14:35.523273  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-10 21:14:41.393645: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-10 21:14:41.399210: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-10 21:14:41.399367: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56527e9e4a70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-10 21:14:41.399381: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fb7bdf12c18> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4789 - crf_viterbi_accuracy: 0.1067 - val_loss: 1.4098 - val_crf_viterbi_accuracy: 0.0667

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fb7bfe37f98> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6513 - accuracy: 0.5010
 2000/25000 [=>............................] - ETA: 7s - loss: 7.8276 - accuracy: 0.4895 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6768 - accuracy: 0.4993
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6590 - accuracy: 0.5005
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6482 - accuracy: 0.5012
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6754 - accuracy: 0.4994
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6264 - accuracy: 0.5026
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6564 - accuracy: 0.5007
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6804 - accuracy: 0.4991
11000/25000 [============>.................] - ETA: 3s - loss: 7.6638 - accuracy: 0.5002
12000/25000 [=============>................] - ETA: 3s - loss: 7.6526 - accuracy: 0.5009
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6253 - accuracy: 0.5027
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6261 - accuracy: 0.5026
15000/25000 [=================>............] - ETA: 2s - loss: 7.6370 - accuracy: 0.5019
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6379 - accuracy: 0.5019
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6615 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6771 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6674 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6761 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6532 - accuracy: 0.5009
25000/25000 [==============================] - 7s 270us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fb75e993470> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 140, in benchmark_run
    metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 60, in metric_eval
    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
AttributeError: module 'sklearn.metrics' has no attribute 'accuracy, f1_score'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_tch.transformer_classifier notfound, No module named 'util_transformer', tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.py", line 164, in fit
    output_path      = out_pars["model_path"]
KeyError: 'model_path'
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:03<98:04:31, 2.44kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:03<68:52:59, 3.48kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:03<48:15:55, 4.96kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:03<33:46:23, 7.08kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:03<23:34:10, 10.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.54M/862M [00:04<16:23:12, 14.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.2M/862M [00:04<11:23:47, 20.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:04<7:55:33, 29.5kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.8M/862M [00:04<5:30:36, 42.1kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.5M/862M [00:04<3:49:57, 60.1kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:04<2:39:57, 85.8kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:04<1:51:16, 123kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 46.8M/862M [00:04<1:17:45, 175kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:05<54:32, 247kB/s]  .vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:05<38:07, 352kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:07<39:35, 339kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:07<29:58, 448kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:07<21:26, 625kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:07<15:06, 885kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:09<41:01, 326kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:09<30:29, 438kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:09<21:44, 613kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:11<17:46, 747kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:11<15:06, 879kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:11<11:19, 1.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.7M/862M [00:11<08:06, 1.63MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:13<10:21, 1.28MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:13<08:37, 1.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:13<06:22, 2.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:15<07:30, 1.75MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:15<07:57, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.2M/862M [00:15<06:13, 2.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:15<04:29, 2.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:17<12:37:03, 17.3kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:17<8:51:03, 24.6kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:17<6:11:20, 35.1kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:19<4:22:14, 49.6kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:19<3:06:11, 69.9kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:19<2:10:46, 99.4kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:19<1:31:27, 142kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:21<1:10:25, 184kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:21<50:22, 257kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:21<35:30, 364kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:21<24:56, 516kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:23<1:09:30, 185kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:23<49:56, 258kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:23<35:13, 365kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:25<27:34, 464kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:25<21:54, 584kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:25<15:53, 805kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:25<11:15, 1.13MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:27<14:28, 880kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:27<11:26, 1.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:27<08:16, 1.54MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<08:45, 1.45MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<08:43, 1.45MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:29<06:38, 1.90MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:29<04:49, 2.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<09:11, 1.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<07:44, 1.63MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:31<05:44, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<06:55, 1.81MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<07:24, 1.69MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:33<05:48, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<06:04, 2.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:35<05:31, 2.25MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:35<04:08, 3.00MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<05:47, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:37<06:34, 1.88MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:37<05:14, 2.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<05:39, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<05:14, 2.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:39<03:58, 3.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<05:37, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<05:11, 2.36MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:41<03:56, 3.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<05:38, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<05:10, 2.35MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:43<03:55, 3.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<05:36, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<05:09, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:45<03:55, 3.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<05:35, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<05:08, 2.34MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:47<03:54, 3.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<05:33, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:48<05:06, 2.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:49<03:49, 3.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<05:31, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<05:06, 2.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:51<03:51, 3.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<05:28, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<06:21, 1.86MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<05:00, 2.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<03:38, 3.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<59:05, 199kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<42:22, 278kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:54<29:54, 393kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<23:35, 496kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<18:53, 620kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:56<13:48, 847kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<11:30, 1.01MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<09:14, 1.26MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:58<06:43, 1.73MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:00<07:25, 1.56MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<06:22, 1.82MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:00<04:44, 2.43MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<06:02, 1.91MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<05:23, 2.13MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:02<04:03, 2.82MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<05:32, 2.06MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<05:02, 2.27MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:04<03:48, 2.99MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<05:21, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<04:53, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:06<03:42, 3.06MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<05:15, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<04:49, 2.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:08<03:39, 3.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<05:12, 2.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<05:57, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<04:38, 2.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:10<03:24, 3.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<07:00, 1.59MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<06:02, 1.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:12<04:27, 2.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<05:43, 1.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<06:22, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:14<04:57, 2.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<03:37, 3.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<07:18, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<06:15, 1.76MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:16<04:36, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<05:47, 1.89MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<06:17, 1.74MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:18<04:51, 2.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<03:32, 3.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<07:26, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<06:08, 1.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:20<04:30, 2.41MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<03:17, 3.28MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<45:52, 236kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<34:18, 315kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<24:31, 440kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<18:50, 571kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<14:17, 751kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:24<10:12, 1.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<09:37, 1.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<08:54, 1.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<06:46, 1.57MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<06:26, 1.65MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<05:37, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<04:11, 2.52MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<05:22, 1.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<05:54, 1.78MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<04:35, 2.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<03:20, 3.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<08:21, 1.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<06:54, 1.51MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:32<05:04, 2.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<05:59, 1.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<06:18, 1.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<04:56, 2.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<05:07, 2.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<04:38, 2.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:36<03:30, 2.94MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<04:51, 2.12MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<05:29, 1.87MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<04:18, 2.38MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<03:06, 3.28MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<21:35, 472kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<16:09, 631kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:40<11:33, 880kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<10:26, 971kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:41<09:21, 1.08MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<07:02, 1.44MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<05:02, 2.00MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<37:36, 268kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:43<27:21, 368kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:44<19:21, 518kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:45<15:49, 632kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:45<12:05, 826kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:45<08:39, 1.15MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<08:23, 1.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<06:52, 1.44MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:47<05:00, 1.97MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<05:51, 1.68MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<06:06, 1.62MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:49<04:41, 2.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<03:24, 2.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<07:44, 1.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<06:26, 1.52MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:51<04:44, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<05:35, 1.74MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<05:53, 1.65MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:53<04:32, 2.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<03:17, 2.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:55<07:16, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:55<05:56, 1.63MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:55<04:23, 2.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:57<05:17, 1.81MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:57<05:38, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<04:21, 2.20MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:57<03:10, 3.00MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:59<06:33, 1.45MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<05:33, 1.71MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:59<04:07, 2.30MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:01<05:05, 1.86MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<05:34, 1.70MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<04:23, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<03:09, 2.97MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:03<1:02:20, 150kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<44:35, 210kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:03<31:20, 298kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<24:01, 388kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<18:42, 497kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:05<13:29, 689kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<09:30, 973kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<13:44, 673kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<10:34, 874kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:07<07:35, 1.21MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:09<07:26, 1.23MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:09<06:08, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:09<04:29, 2.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:11<05:16, 1.73MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:11<04:37, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<03:25, 2.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:13<04:32, 1.99MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:13<04:05, 2.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<03:05, 2.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:15<04:16, 2.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:15<04:49, 1.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<03:50, 2.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:17<04:06, 2.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:17<03:47, 2.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<02:50, 3.13MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:19<04:03, 2.17MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<04:38, 1.90MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<03:38, 2.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<02:37, 3.33MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:21<22:29, 390kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:21<16:38, 526kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<11:48, 740kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<10:16, 846kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:23<08:58, 969kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<06:42, 1.29MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:23<04:45, 1.81MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:24<8:22:13, 17.2kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:25<5:52:10, 24.5kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:25<4:05:58, 34.9kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:26<2:53:28, 49.3kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:27<2:03:07, 69.5kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<1:26:25, 98.8kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:27<1:00:22, 141kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<46:04, 184kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<33:05, 256kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<23:16, 363kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<18:12, 462kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<13:36, 618kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<09:42, 863kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:32<08:44, 956kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<06:57, 1.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<05:02, 1.65MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:34<05:29, 1.51MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:34<05:31, 1.50MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<04:13, 1.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:35<03:03, 2.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:36<05:42, 1.44MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<04:50, 1.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:36<03:35, 2.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<04:24, 1.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<03:55, 2.07MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:38<02:56, 2.75MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<03:57, 2.04MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<03:36, 2.24MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:40<02:43, 2.96MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:42<03:46, 2.12MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:42<03:20, 2.39MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:42<02:32, 3.14MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:44<03:39, 2.17MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:44<04:10, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<03:17, 2.41MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<02:22, 3.33MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:46<13:33, 580kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:46<10:17, 764kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:46<07:23, 1.06MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:48<06:57, 1.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:48<06:28, 1.20MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<04:52, 1.60MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<03:30, 2.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:50<06:11, 1.25MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:50<05:08, 1.50MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:50<03:46, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:52<04:25, 1.73MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:52<03:52, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<02:52, 2.65MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:54<03:48, 1.99MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:54<03:26, 2.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:54<02:33, 2.95MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<03:35, 2.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<03:59, 1.88MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<03:10, 2.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:58<03:25, 2.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:58<03:10, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:58<02:22, 3.12MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<03:22, 2.19MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<03:07, 2.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:00<02:21, 3.11MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<03:22, 2.17MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<03:06, 2.35MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:02<02:19, 3.13MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:04<03:21, 2.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:04<03:47, 1.91MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<02:57, 2.44MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:04<02:08, 3.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:06<07:09, 1.00MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:06<05:45, 1.25MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<04:12, 1.70MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<04:35, 1.55MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:08<04:44, 1.50MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:08<03:37, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<02:37, 2.70MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<06:09, 1.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:10<05:04, 1.39MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<03:41, 1.90MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<04:11, 1.67MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:12<03:39, 1.91MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<02:42, 2.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<03:30, 1.97MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:14<03:09, 2.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:14<02:22, 2.89MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:15<03:16, 2.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:16<03:41, 1.85MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:16<02:52, 2.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<02:06, 3.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<04:02, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:18<03:32, 1.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:18<02:38, 2.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<03:24, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<03:05, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<02:18, 2.90MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<03:09, 2.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<02:53, 2.30MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:22<02:09, 3.05MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<03:04, 2.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<03:29, 1.88MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:24<02:46, 2.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:25<02:58, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:25<02:45, 2.35MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:25<02:03, 3.13MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:27<02:56, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:27<02:43, 2.36MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:27<02:03, 3.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:29<02:55, 2.17MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<03:24, 1.87MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<02:42, 2.35MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:30<01:57, 3.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:31<24:02, 262kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:31<17:27, 360kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:31<12:18, 509kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:33<10:01, 621kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:33<08:17, 751kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:33<06:03, 1.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<04:18, 1.43MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:35<05:38, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:35<04:34, 1.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<03:19, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:37<03:44, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:37<03:52, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<03:01, 2.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:10, 2.76MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:39<44:26, 135kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:39<31:40, 190kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:39<22:14, 269kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:41<16:51, 353kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<12:24, 479kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:41<08:48, 672kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:43<07:30, 784kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:43<06:27, 910kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:43<04:46, 1.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:43<03:22, 1.72MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:45<08:29, 685kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:45<06:31, 889kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:45<04:42, 1.23MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:47<04:37, 1.24MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<04:24, 1.30MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<03:22, 1.70MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:49<03:15, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:49<02:52, 1.98MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:49<02:07, 2.66MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:51<02:47, 2.01MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:51<02:31, 2.22MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:51<01:53, 2.93MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:53<02:38, 2.10MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:53<02:19, 2.39MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:53<01:45, 3.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<02:31, 2.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<02:19, 2.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:55<01:44, 3.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<02:29, 2.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<02:48, 1.93MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:57<02:13, 2.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<01:36, 3.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:59<5:09:21, 17.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:59<3:36:50, 24.6kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<2:31:10, 35.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:01<1:46:20, 49.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:01<1:14:53, 70.2kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:01<52:18, 100kB/s]   .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:03<37:34, 138kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:03<27:20, 190kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:03<19:19, 268kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:03<13:31, 381kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<11:30, 446kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<08:34, 598kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:05<06:05, 835kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<05:25, 932kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:07<04:49, 1.05MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:07<03:38, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<02:34, 1.93MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<42:30, 117kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:09<30:13, 165kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:09<21:09, 234kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<15:51, 310kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<12:05, 407kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:11<08:39, 567kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:11<06:04, 801kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:12<07:17, 665kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:12<05:36, 865kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:13<04:00, 1.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<03:54, 1.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<03:13, 1.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:15<02:21, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<02:45, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<02:53, 1.63MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:17<02:15, 2.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:17<01:37, 2.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:18<34:18, 135kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:18<24:28, 190kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:18<17:09, 269kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:20<12:58, 353kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<09:32, 479kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:20<06:45, 673kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<05:45, 784kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<04:57, 910kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:22<03:41, 1.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<02:36, 1.70MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<33:31, 133kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<23:53, 186kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<16:44, 263kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:26<12:38, 346kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:26<09:13, 474kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<06:31, 665kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<05:33, 775kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<04:45, 904kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<03:32, 1.21MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:30<03:08, 1.35MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:30<02:37, 1.61MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<01:55, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<02:19, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<02:02, 2.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:32<01:30, 2.73MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:34<02:00, 2.03MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:34<02:14, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<01:46, 2.30MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<01:16, 3.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:36<3:53:40, 17.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:36<2:43:43, 24.6kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:36<1:53:58, 35.1kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:38<1:19:58, 49.5kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:38<56:17, 70.3kB/s]  .vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:38<39:14, 100kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:40<28:08, 138kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:40<20:26, 190kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<14:25, 269kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<10:03, 382kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:42<08:48, 434kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:42<06:33, 582kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:42<04:39, 814kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<04:06, 916kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<03:39, 1.02MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:44<02:45, 1.36MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<01:56, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:46<10:14, 360kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:46<07:32, 488kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:46<05:19, 687kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<04:32, 797kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<03:54, 924kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:48<02:55, 1.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:50<02:35, 1.37MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:50<02:10, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:50<01:36, 2.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<01:55, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:52<01:41, 2.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:52<01:16, 2.72MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:54<01:41, 2.02MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:54<01:31, 2.23MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:54<01:08, 2.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<00:58, 3.40MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<2:38:53, 21.1kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:56<1:50:57, 30.0kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<1:16:29, 42.9kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:57<1:02:02, 52.8kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:58<44:03, 74.3kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:58<30:52, 106kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<21:47, 147kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<15:34, 206kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:00<10:53, 292kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<08:15, 380kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:02<06:05, 514kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:02<04:18, 723kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<03:41, 831kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<02:53, 1.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:04<02:05, 1.46MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<02:09, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<02:07, 1.42MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:06<01:37, 1.83MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:07<01:35, 1.84MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<01:25, 2.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:08<01:03, 2.73MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:09<01:24, 2.04MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<01:16, 2.23MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<00:57, 2.95MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:11<01:19, 2.12MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<01:29, 1.87MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<01:09, 2.39MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:12<00:50, 3.26MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<01:56, 1.40MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<01:35, 1.71MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<01:09, 2.32MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<00:50, 3.17MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:15<07:34, 351kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:15<05:50, 455kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:15<04:12, 628kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:55, 889kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<11:25, 227kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<08:14, 314kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<05:46, 444kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:19<04:34, 551kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:19<03:27, 729kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<02:26, 1.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:21<02:16, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:21<01:50, 1.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<01:20, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<01:29, 1.60MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<01:31, 1.56MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:23<01:10, 2.00MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<01:11, 1.95MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:25<01:04, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:25<00:47, 2.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<01:04, 2.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<01:12, 1.86MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:27<00:56, 2.39MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<00:40, 3.25MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<01:31, 1.43MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:29<01:17, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:29<00:56, 2.28MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:31<01:08, 1.85MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:31<01:13, 1.72MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:31<00:56, 2.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<00:40, 3.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:33<01:32, 1.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:33<01:17, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<00:56, 2.13MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:35<01:06, 1.78MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:35<00:58, 2.01MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:35<00:43, 2.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:37<00:56, 2.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:37<01:02, 1.81MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:37<00:48, 2.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<00:34, 3.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<01:59, 920kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<01:34, 1.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:39<01:07, 1.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<01:11, 1.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<01:11, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:41<00:54, 1.94MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:41<00:38, 2.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:43<01:04, 1.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:43<00:55, 1.83MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:43<00:40, 2.47MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<00:50, 1.93MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:45<00:55, 1.76MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:45<00:42, 2.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:30, 3.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:33, 2.81MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<1:09:59, 22.3kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:47<48:40, 31.8kB/s]  .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<32:58, 45.5kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<25:43, 58.1kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:49<18:16, 81.6kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:49<12:45, 116kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<08:50, 161kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:51<06:19, 225kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:51<04:22, 318kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<03:17, 412kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:53<02:26, 553kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:53<01:42, 777kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<01:27, 883kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<01:17, 995kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:55<00:57, 1.34MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:55<00:39, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<00:58, 1.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<00:48, 1.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:57<00:35, 2.03MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:39, 1.73MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:34, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:59<00:25, 2.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:32, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:36, 1.77MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<00:28, 2.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:01<00:19, 3.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<07:28, 136kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:02<05:18, 190kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<03:38, 270kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<02:40, 353kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:04<02:03, 457kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:04<01:27, 635kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:04<01:00, 895kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<01:03, 834kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:49, 1.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:06<00:34, 1.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:34, 1.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:33, 1.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:08<00:25, 1.87MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:17, 2.59MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<01:04, 689kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:49, 893kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:34, 1.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:32, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:26, 1.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:12<00:18, 2.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:20, 1.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:21, 1.64MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<00:16, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:11, 2.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<00:56, 571kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:42, 753kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:16<00:28, 1.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<00:25, 1.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:20, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<00:13, 1.86MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:14, 1.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:12, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:08, 2.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:10, 1.94MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:08, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<00:06, 2.86MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:07, 2.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:08, 1.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:24<00:06, 2.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:03, 3.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<00:07, 1.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<00:06, 1.78MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:03, 2.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:03, 1.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:04, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:28<00:02, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:28<00:01, 3.06MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:01, 1.65MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:01, 1.90MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:00, 2.54MB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 885/400000 [00:00<00:45, 8849.65it/s]  0%|          | 1762/400000 [00:00<00:45, 8823.42it/s]  1%|          | 2607/400000 [00:00<00:45, 8705.49it/s]  1%|          | 3488/400000 [00:00<00:45, 8735.43it/s]  1%|          | 4359/400000 [00:00<00:45, 8726.27it/s]  1%|▏         | 5256/400000 [00:00<00:44, 8796.60it/s]  2%|▏         | 6110/400000 [00:00<00:45, 8717.60it/s]  2%|▏         | 6988/400000 [00:00<00:44, 8735.72it/s]  2%|▏         | 7885/400000 [00:00<00:44, 8803.57it/s]  2%|▏         | 8795/400000 [00:01<00:44, 8888.44it/s]  2%|▏         | 9660/400000 [00:01<00:45, 8523.26it/s]  3%|▎         | 10499/400000 [00:01<00:46, 8403.88it/s]  3%|▎         | 11340/400000 [00:01<00:46, 8404.58it/s]  3%|▎         | 12217/400000 [00:01<00:45, 8509.09it/s]  3%|▎         | 13090/400000 [00:01<00:45, 8572.48it/s]  3%|▎         | 13980/400000 [00:01<00:44, 8666.55it/s]  4%|▎         | 14868/400000 [00:01<00:44, 8727.07it/s]  4%|▍         | 15757/400000 [00:01<00:43, 8772.77it/s]  4%|▍         | 16647/400000 [00:01<00:43, 8809.09it/s]  4%|▍         | 17528/400000 [00:02<00:43, 8719.21it/s]  5%|▍         | 18400/400000 [00:02<00:43, 8712.03it/s]  5%|▍         | 19275/400000 [00:02<00:43, 8721.90it/s]  5%|▌         | 20148/400000 [00:02<00:43, 8655.10it/s]  5%|▌         | 21023/400000 [00:02<00:43, 8683.01it/s]  5%|▌         | 21903/400000 [00:02<00:43, 8716.00it/s]  6%|▌         | 22779/400000 [00:02<00:43, 8728.14it/s]  6%|▌         | 23663/400000 [00:02<00:42, 8760.05it/s]  6%|▌         | 24540/400000 [00:02<00:42, 8760.63it/s]  6%|▋         | 25417/400000 [00:02<00:42, 8757.34it/s]  7%|▋         | 26293/400000 [00:03<00:42, 8732.77it/s]  7%|▋         | 27172/400000 [00:03<00:42, 8749.67it/s]  7%|▋         | 28053/400000 [00:03<00:42, 8767.26it/s]  7%|▋         | 28935/400000 [00:03<00:42, 8780.85it/s]  7%|▋         | 29814/400000 [00:03<00:42, 8776.74it/s]  8%|▊         | 30692/400000 [00:03<00:42, 8774.77it/s]  8%|▊         | 31574/400000 [00:03<00:41, 8787.14it/s]  8%|▊         | 32455/400000 [00:03<00:41, 8793.95it/s]  8%|▊         | 33336/400000 [00:03<00:41, 8796.82it/s]  9%|▊         | 34218/400000 [00:03<00:41, 8802.77it/s]  9%|▉         | 35099/400000 [00:04<00:41, 8753.57it/s]  9%|▉         | 35975/400000 [00:04<00:41, 8749.44it/s]  9%|▉         | 36856/400000 [00:04<00:41, 8767.18it/s]  9%|▉         | 37737/400000 [00:04<00:41, 8777.98it/s] 10%|▉         | 38615/400000 [00:04<00:41, 8674.68it/s] 10%|▉         | 39495/400000 [00:04<00:41, 8710.68it/s] 10%|█         | 40377/400000 [00:04<00:41, 8741.58it/s] 10%|█         | 41258/400000 [00:04<00:40, 8761.68it/s] 11%|█         | 42137/400000 [00:04<00:40, 8768.83it/s] 11%|█         | 43014/400000 [00:04<00:42, 8444.16it/s] 11%|█         | 43862/400000 [00:05<00:42, 8439.09it/s] 11%|█         | 44744/400000 [00:05<00:41, 8548.40it/s] 11%|█▏        | 45624/400000 [00:05<00:41, 8620.86it/s] 12%|█▏        | 46503/400000 [00:05<00:40, 8669.90it/s] 12%|█▏        | 47379/400000 [00:05<00:40, 8696.31it/s] 12%|█▏        | 48250/400000 [00:05<00:40, 8678.64it/s] 12%|█▏        | 49131/400000 [00:05<00:40, 8715.54it/s] 13%|█▎        | 50003/400000 [00:05<00:40, 8704.30it/s] 13%|█▎        | 50876/400000 [00:05<00:40, 8709.75it/s] 13%|█▎        | 51748/400000 [00:05<00:40, 8695.97it/s] 13%|█▎        | 52618/400000 [00:06<00:40, 8684.02it/s] 13%|█▎        | 53496/400000 [00:06<00:39, 8710.64it/s] 14%|█▎        | 54368/400000 [00:06<00:40, 8543.51it/s] 14%|█▍        | 55224/400000 [00:06<00:40, 8470.38it/s] 14%|█▍        | 56093/400000 [00:06<00:40, 8534.50it/s] 14%|█▍        | 56972/400000 [00:06<00:39, 8607.75it/s] 14%|█▍        | 57844/400000 [00:06<00:39, 8639.79it/s] 15%|█▍        | 58717/400000 [00:06<00:39, 8664.44it/s] 15%|█▍        | 59596/400000 [00:06<00:39, 8701.60it/s] 15%|█▌        | 60468/400000 [00:06<00:38, 8706.90it/s] 15%|█▌        | 61360/400000 [00:07<00:38, 8768.54it/s] 16%|█▌        | 62244/400000 [00:07<00:38, 8787.74it/s] 16%|█▌        | 63131/400000 [00:07<00:38, 8810.84it/s] 16%|█▌        | 64013/400000 [00:07<00:38, 8788.91it/s] 16%|█▌        | 64892/400000 [00:07<00:38, 8704.82it/s] 16%|█▋        | 65766/400000 [00:07<00:38, 8714.39it/s] 17%|█▋        | 66670/400000 [00:07<00:37, 8808.11it/s] 17%|█▋        | 67553/400000 [00:07<00:37, 8813.73it/s] 17%|█▋        | 68435/400000 [00:07<00:38, 8716.46it/s] 17%|█▋        | 69308/400000 [00:07<00:38, 8511.85it/s] 18%|█▊        | 70211/400000 [00:08<00:38, 8658.73it/s] 18%|█▊        | 71121/400000 [00:08<00:37, 8783.88it/s] 18%|█▊        | 72001/400000 [00:08<00:37, 8786.11it/s] 18%|█▊        | 72881/400000 [00:08<00:37, 8778.37it/s] 18%|█▊        | 73763/400000 [00:08<00:37, 8788.09it/s] 19%|█▊        | 74643/400000 [00:08<00:37, 8787.55it/s] 19%|█▉        | 75523/400000 [00:08<00:36, 8777.60it/s] 19%|█▉        | 76402/400000 [00:08<00:37, 8722.08it/s] 19%|█▉        | 77286/400000 [00:08<00:36, 8755.02it/s] 20%|█▉        | 78162/400000 [00:08<00:37, 8620.19it/s] 20%|█▉        | 79041/400000 [00:09<00:37, 8667.98it/s] 20%|█▉        | 79928/400000 [00:09<00:36, 8724.87it/s] 20%|██        | 80809/400000 [00:09<00:36, 8749.25it/s] 20%|██        | 81685/400000 [00:09<00:36, 8751.99it/s] 21%|██        | 82561/400000 [00:09<00:36, 8708.95it/s] 21%|██        | 83433/400000 [00:09<00:36, 8639.02it/s] 21%|██        | 84298/400000 [00:09<00:36, 8606.39it/s] 21%|██▏       | 85178/400000 [00:09<00:36, 8662.71it/s] 22%|██▏       | 86051/400000 [00:09<00:36, 8681.87it/s] 22%|██▏       | 86927/400000 [00:09<00:35, 8704.63it/s] 22%|██▏       | 87805/400000 [00:10<00:35, 8726.21it/s] 22%|██▏       | 88690/400000 [00:10<00:35, 8761.34it/s] 22%|██▏       | 89571/400000 [00:10<00:35, 8774.56it/s] 23%|██▎       | 90454/400000 [00:10<00:35, 8788.68it/s] 23%|██▎       | 91333/400000 [00:10<00:35, 8781.21it/s] 23%|██▎       | 92227/400000 [00:10<00:34, 8825.37it/s] 23%|██▎       | 93110/400000 [00:10<00:34, 8799.58it/s] 24%|██▎       | 94005/400000 [00:10<00:34, 8841.42it/s] 24%|██▎       | 94896/400000 [00:10<00:34, 8859.68it/s] 24%|██▍       | 95792/400000 [00:10<00:34, 8888.59it/s] 24%|██▍       | 96687/400000 [00:11<00:34, 8905.27it/s] 24%|██▍       | 97600/400000 [00:11<00:33, 8971.47it/s] 25%|██▍       | 98510/400000 [00:11<00:33, 9008.19it/s] 25%|██▍       | 99411/400000 [00:11<00:33, 8987.41it/s] 25%|██▌       | 100310/400000 [00:11<00:33, 8814.93it/s] 25%|██▌       | 101193/400000 [00:11<00:33, 8816.03it/s] 26%|██▌       | 102076/400000 [00:11<00:33, 8798.72it/s] 26%|██▌       | 102957/400000 [00:11<00:33, 8774.87it/s] 26%|██▌       | 103872/400000 [00:11<00:33, 8882.39it/s] 26%|██▌       | 104771/400000 [00:12<00:33, 8913.07it/s] 26%|██▋       | 105663/400000 [00:12<00:33, 8908.53it/s] 27%|██▋       | 106581/400000 [00:12<00:32, 8986.18it/s] 27%|██▋       | 107492/400000 [00:12<00:32, 9021.21it/s] 27%|██▋       | 108395/400000 [00:12<00:32, 8966.92it/s] 27%|██▋       | 109292/400000 [00:12<00:32, 8898.30it/s] 28%|██▊       | 110183/400000 [00:12<00:32, 8885.30it/s] 28%|██▊       | 111072/400000 [00:12<00:32, 8880.54it/s] 28%|██▊       | 111964/400000 [00:12<00:32, 8889.52it/s] 28%|██▊       | 112863/400000 [00:12<00:32, 8917.53it/s] 28%|██▊       | 113763/400000 [00:13<00:32, 8940.82it/s] 29%|██▊       | 114658/400000 [00:13<00:32, 8843.31it/s] 29%|██▉       | 115543/400000 [00:13<00:32, 8696.96it/s] 29%|██▉       | 116430/400000 [00:13<00:32, 8745.85it/s] 29%|██▉       | 117334/400000 [00:13<00:32, 8830.61it/s] 30%|██▉       | 118234/400000 [00:13<00:31, 8878.60it/s] 30%|██▉       | 119134/400000 [00:13<00:31, 8912.08it/s] 30%|███       | 120070/400000 [00:13<00:30, 9040.23it/s] 30%|███       | 120988/400000 [00:13<00:30, 9078.81it/s] 30%|███       | 121903/400000 [00:13<00:30, 9097.55it/s] 31%|███       | 122814/400000 [00:14<00:30, 9086.05it/s] 31%|███       | 123723/400000 [00:14<00:30, 9086.77it/s] 31%|███       | 124632/400000 [00:14<00:31, 8801.28it/s] 31%|███▏      | 125551/400000 [00:14<00:30, 8912.93it/s] 32%|███▏      | 126452/400000 [00:14<00:30, 8939.86it/s] 32%|███▏      | 127348/400000 [00:14<00:30, 8865.99it/s] 32%|███▏      | 128257/400000 [00:14<00:30, 8929.84it/s] 32%|███▏      | 129175/400000 [00:14<00:30, 9001.21it/s] 33%|███▎      | 130076/400000 [00:14<00:30, 8976.71it/s] 33%|███▎      | 130975/400000 [00:14<00:30, 8778.54it/s] 33%|███▎      | 131855/400000 [00:15<00:30, 8654.71it/s] 33%|███▎      | 132760/400000 [00:15<00:30, 8769.58it/s] 33%|███▎      | 133662/400000 [00:15<00:30, 8841.88it/s] 34%|███▎      | 134548/400000 [00:15<00:30, 8613.34it/s] 34%|███▍      | 135428/400000 [00:15<00:30, 8667.00it/s] 34%|███▍      | 136297/400000 [00:15<00:30, 8596.71it/s] 34%|███▍      | 137174/400000 [00:15<00:30, 8647.53it/s] 35%|███▍      | 138041/400000 [00:15<00:30, 8653.90it/s] 35%|███▍      | 138925/400000 [00:15<00:29, 8707.34it/s] 35%|███▍      | 139797/400000 [00:15<00:29, 8699.29it/s] 35%|███▌      | 140675/400000 [00:16<00:29, 8721.04it/s] 35%|███▌      | 141597/400000 [00:16<00:29, 8863.20it/s] 36%|███▌      | 142523/400000 [00:16<00:28, 8977.48it/s] 36%|███▌      | 143450/400000 [00:16<00:28, 9061.72it/s] 36%|███▌      | 144357/400000 [00:16<00:28, 8996.86it/s] 36%|███▋      | 145258/400000 [00:16<00:29, 8753.78it/s] 37%|███▋      | 146148/400000 [00:16<00:28, 8794.86it/s] 37%|███▋      | 147034/400000 [00:16<00:28, 8812.58it/s] 37%|███▋      | 147937/400000 [00:16<00:28, 8874.67it/s] 37%|███▋      | 148859/400000 [00:16<00:27, 8973.17it/s] 37%|███▋      | 149765/400000 [00:17<00:27, 8998.99it/s] 38%|███▊      | 150684/400000 [00:17<00:27, 9053.31it/s] 38%|███▊      | 151598/400000 [00:17<00:27, 9076.84it/s] 38%|███▊      | 152517/400000 [00:17<00:27, 9108.15it/s] 38%|███▊      | 153429/400000 [00:17<00:27, 9054.43it/s] 39%|███▊      | 154353/400000 [00:17<00:26, 9107.67it/s] 39%|███▉      | 155265/400000 [00:17<00:27, 9061.82it/s] 39%|███▉      | 156172/400000 [00:17<00:27, 8994.51it/s] 39%|███▉      | 157072/400000 [00:17<00:27, 8926.91it/s] 39%|███▉      | 157966/400000 [00:17<00:27, 8922.10it/s] 40%|███▉      | 158871/400000 [00:18<00:26, 8957.63it/s] 40%|███▉      | 159796/400000 [00:18<00:26, 9042.69it/s] 40%|████      | 160708/400000 [00:18<00:26, 9063.82it/s] 40%|████      | 161633/400000 [00:18<00:26, 9118.59it/s] 41%|████      | 162570/400000 [00:18<00:25, 9191.74it/s] 41%|████      | 163490/400000 [00:18<00:25, 9191.14it/s] 41%|████      | 164434/400000 [00:18<00:25, 9264.37it/s] 41%|████▏     | 165361/400000 [00:18<00:25, 9257.86it/s] 42%|████▏     | 166317/400000 [00:18<00:25, 9343.97it/s] 42%|████▏     | 167287/400000 [00:18<00:24, 9446.87it/s] 42%|████▏     | 168234/400000 [00:19<00:24, 9452.04it/s] 42%|████▏     | 169180/400000 [00:19<00:24, 9379.96it/s] 43%|████▎     | 170119/400000 [00:19<00:24, 9333.18it/s] 43%|████▎     | 171053/400000 [00:19<00:24, 9290.96it/s] 43%|████▎     | 171983/400000 [00:19<00:24, 9187.65it/s] 43%|████▎     | 172911/400000 [00:19<00:24, 9213.50it/s] 43%|████▎     | 173860/400000 [00:19<00:24, 9294.40it/s] 44%|████▎     | 174790/400000 [00:19<00:24, 9251.72it/s] 44%|████▍     | 175752/400000 [00:19<00:23, 9356.69it/s] 44%|████▍     | 176709/400000 [00:19<00:23, 9419.29it/s] 44%|████▍     | 177652/400000 [00:20<00:23, 9397.98it/s] 45%|████▍     | 178609/400000 [00:20<00:23, 9448.20it/s] 45%|████▍     | 179555/400000 [00:20<00:24, 9156.50it/s] 45%|████▌     | 180473/400000 [00:20<00:24, 8939.63it/s] 45%|████▌     | 181372/400000 [00:20<00:24, 8953.13it/s] 46%|████▌     | 182281/400000 [00:20<00:24, 8991.51it/s] 46%|████▌     | 183209/400000 [00:20<00:23, 9075.06it/s] 46%|████▌     | 184118/400000 [00:20<00:23, 8998.83it/s] 46%|████▋     | 185019/400000 [00:20<00:24, 8945.01it/s] 46%|████▋     | 185949/400000 [00:21<00:23, 9048.58it/s] 47%|████▋     | 186865/400000 [00:21<00:23, 9080.08it/s] 47%|████▋     | 187813/400000 [00:21<00:23, 9195.62it/s] 47%|████▋     | 188734/400000 [00:21<00:22, 9191.78it/s] 47%|████▋     | 189691/400000 [00:21<00:22, 9299.43it/s] 48%|████▊     | 190622/400000 [00:21<00:22, 9295.45it/s] 48%|████▊     | 191553/400000 [00:21<00:23, 8817.68it/s] 48%|████▊     | 192493/400000 [00:21<00:23, 8982.51it/s] 48%|████▊     | 193396/400000 [00:21<00:23, 8969.31it/s] 49%|████▊     | 194297/400000 [00:21<00:22, 8960.38it/s] 49%|████▉     | 195196/400000 [00:22<00:23, 8870.10it/s] 49%|████▉     | 196092/400000 [00:22<00:22, 8895.00it/s] 49%|████▉     | 197017/400000 [00:22<00:22, 8997.56it/s] 49%|████▉     | 197966/400000 [00:22<00:22, 9137.18it/s] 50%|████▉     | 198882/400000 [00:22<00:22, 9108.76it/s] 50%|████▉     | 199794/400000 [00:22<00:21, 9105.04it/s] 50%|█████     | 200726/400000 [00:22<00:21, 9167.62it/s] 50%|█████     | 201664/400000 [00:22<00:21, 9229.94it/s] 51%|█████     | 202600/400000 [00:22<00:21, 9265.97it/s] 51%|█████     | 203527/400000 [00:22<00:21, 9229.39it/s] 51%|█████     | 204462/400000 [00:23<00:21, 9262.74it/s] 51%|█████▏    | 205389/400000 [00:23<00:21, 9215.95it/s] 52%|█████▏    | 206327/400000 [00:23<00:20, 9263.35it/s] 52%|█████▏    | 207254/400000 [00:23<00:20, 9249.14it/s] 52%|█████▏    | 208189/400000 [00:23<00:20, 9278.91it/s] 52%|█████▏    | 209118/400000 [00:23<00:20, 9198.58it/s] 53%|█████▎    | 210039/400000 [00:23<00:20, 9168.07it/s] 53%|█████▎    | 210957/400000 [00:23<00:21, 8828.28it/s] 53%|█████▎    | 211843/400000 [00:23<00:21, 8785.30it/s] 53%|█████▎    | 212725/400000 [00:23<00:21, 8794.11it/s] 53%|█████▎    | 213609/400000 [00:24<00:21, 8805.13it/s] 54%|█████▎    | 214491/400000 [00:24<00:21, 8715.87it/s] 54%|█████▍    | 215414/400000 [00:24<00:20, 8863.53it/s] 54%|█████▍    | 216324/400000 [00:24<00:20, 8930.57it/s] 54%|█████▍    | 217237/400000 [00:24<00:20, 8986.62it/s] 55%|█████▍    | 218146/400000 [00:24<00:20, 9016.53it/s] 55%|█████▍    | 219049/400000 [00:24<00:20, 9013.30it/s] 55%|█████▍    | 219979/400000 [00:24<00:19, 9097.37it/s] 55%|█████▌    | 220927/400000 [00:24<00:19, 9207.55it/s] 55%|█████▌    | 221857/400000 [00:24<00:19, 9233.94it/s] 56%|█████▌    | 222781/400000 [00:25<00:19, 9173.41it/s] 56%|█████▌    | 223699/400000 [00:25<00:19, 9141.11it/s] 56%|█████▌    | 224627/400000 [00:25<00:19, 9181.33it/s] 56%|█████▋    | 225546/400000 [00:25<00:19, 9114.70it/s] 57%|█████▋    | 226458/400000 [00:25<00:19, 8841.08it/s] 57%|█████▋    | 227373/400000 [00:25<00:19, 8931.19it/s] 57%|█████▋    | 228272/400000 [00:25<00:19, 8947.36it/s] 57%|█████▋    | 229194/400000 [00:25<00:18, 9027.19it/s] 58%|█████▊    | 230113/400000 [00:25<00:18, 9073.66it/s] 58%|█████▊    | 231059/400000 [00:25<00:18, 9185.64it/s] 58%|█████▊    | 231979/400000 [00:26<00:18, 9073.41it/s] 58%|█████▊    | 232888/400000 [00:26<00:18, 9050.60it/s] 58%|█████▊    | 233834/400000 [00:26<00:18, 9166.88it/s] 59%|█████▊    | 234799/400000 [00:26<00:17, 9304.27it/s] 59%|█████▉    | 235731/400000 [00:26<00:17, 9234.85it/s] 59%|█████▉    | 236688/400000 [00:26<00:17, 9332.07it/s] 59%|█████▉    | 237623/400000 [00:26<00:17, 9275.43it/s] 60%|█████▉    | 238586/400000 [00:26<00:17, 9376.18it/s] 60%|█████▉    | 239525/400000 [00:26<00:17, 9302.74it/s] 60%|██████    | 240456/400000 [00:27<00:17, 9281.03it/s] 60%|██████    | 241385/400000 [00:27<00:17, 9203.73it/s] 61%|██████    | 242306/400000 [00:27<00:17, 9124.13it/s] 61%|██████    | 243220/400000 [00:27<00:17, 9127.64it/s] 61%|██████    | 244161/400000 [00:27<00:16, 9210.23it/s] 61%|██████▏   | 245083/400000 [00:27<00:16, 9161.21it/s] 62%|██████▏   | 246031/400000 [00:27<00:16, 9252.89it/s] 62%|██████▏   | 246967/400000 [00:27<00:16, 9283.60it/s] 62%|██████▏   | 247958/400000 [00:27<00:16, 9463.02it/s] 62%|██████▏   | 248906/400000 [00:27<00:16, 9405.63it/s] 62%|██████▏   | 249877/400000 [00:28<00:15, 9492.57it/s] 63%|██████▎   | 250845/400000 [00:28<00:15, 9545.52it/s] 63%|██████▎   | 251801/400000 [00:28<00:15, 9366.86it/s] 63%|██████▎   | 252739/400000 [00:28<00:16, 9124.98it/s] 63%|██████▎   | 253654/400000 [00:28<00:16, 9046.66it/s] 64%|██████▎   | 254561/400000 [00:28<00:16, 8971.61it/s] 64%|██████▍   | 255460/400000 [00:28<00:16, 8917.28it/s] 64%|██████▍   | 256353/400000 [00:28<00:16, 8668.22it/s] 64%|██████▍   | 257223/400000 [00:28<00:16, 8586.10it/s] 65%|██████▍   | 258084/400000 [00:28<00:16, 8592.84it/s] 65%|██████▍   | 258961/400000 [00:29<00:16, 8644.19it/s] 65%|██████▍   | 259839/400000 [00:29<00:16, 8684.32it/s] 65%|██████▌   | 260709/400000 [00:29<00:16, 8678.37it/s] 65%|██████▌   | 261590/400000 [00:29<00:15, 8715.83it/s] 66%|██████▌   | 262495/400000 [00:29<00:15, 8811.49it/s] 66%|██████▌   | 263377/400000 [00:29<00:15, 8799.53it/s] 66%|██████▌   | 264266/400000 [00:29<00:15, 8824.08it/s] 66%|██████▋   | 265149/400000 [00:29<00:15, 8779.11it/s] 67%|██████▋   | 266053/400000 [00:29<00:15, 8852.83it/s] 67%|██████▋   | 266951/400000 [00:29<00:14, 8888.56it/s] 67%|██████▋   | 267841/400000 [00:30<00:15, 8792.11it/s] 67%|██████▋   | 268723/400000 [00:30<00:14, 8798.59it/s] 67%|██████▋   | 269604/400000 [00:30<00:14, 8782.05it/s] 68%|██████▊   | 270508/400000 [00:30<00:14, 8856.10it/s] 68%|██████▊   | 271394/400000 [00:30<00:14, 8771.85it/s] 68%|██████▊   | 272285/400000 [00:30<00:14, 8810.34it/s] 68%|██████▊   | 273167/400000 [00:30<00:14, 8772.83it/s] 69%|██████▊   | 274045/400000 [00:30<00:14, 8770.29it/s] 69%|██████▊   | 274938/400000 [00:30<00:14, 8817.26it/s] 69%|██████▉   | 275832/400000 [00:30<00:14, 8850.77it/s] 69%|██████▉   | 276720/400000 [00:31<00:13, 8858.71it/s] 69%|██████▉   | 277607/400000 [00:31<00:13, 8860.67it/s] 70%|██████▉   | 278494/400000 [00:31<00:13, 8825.65it/s] 70%|██████▉   | 279377/400000 [00:31<00:13, 8825.73it/s] 70%|███████   | 280260/400000 [00:31<00:13, 8820.07it/s] 70%|███████   | 281151/400000 [00:31<00:13, 8846.54it/s] 71%|███████   | 282040/400000 [00:31<00:13, 8858.52it/s] 71%|███████   | 282926/400000 [00:31<00:13, 8772.30it/s] 71%|███████   | 283804/400000 [00:31<00:13, 8738.50it/s] 71%|███████   | 284709/400000 [00:31<00:13, 8828.95it/s] 71%|███████▏  | 285612/400000 [00:32<00:12, 8886.40it/s] 72%|███████▏  | 286511/400000 [00:32<00:12, 8916.28it/s] 72%|███████▏  | 287406/400000 [00:32<00:12, 8924.49it/s] 72%|███████▏  | 288321/400000 [00:32<00:12, 8990.78it/s] 72%|███████▏  | 289227/400000 [00:32<00:12, 9010.70it/s] 73%|███████▎  | 290129/400000 [00:32<00:12, 8986.74it/s] 73%|███████▎  | 291048/400000 [00:32<00:12, 9043.80it/s] 73%|███████▎  | 291953/400000 [00:32<00:11, 9017.52it/s] 73%|███████▎  | 292855/400000 [00:32<00:11, 8950.62it/s] 73%|███████▎  | 293751/400000 [00:32<00:11, 8941.56it/s] 74%|███████▎  | 294658/400000 [00:33<00:11, 8978.59it/s] 74%|███████▍  | 295557/400000 [00:33<00:11, 8794.54it/s] 74%|███████▍  | 296444/400000 [00:33<00:11, 8815.58it/s] 74%|███████▍  | 297342/400000 [00:33<00:11, 8859.37it/s] 75%|███████▍  | 298235/400000 [00:33<00:11, 8878.31it/s] 75%|███████▍  | 299125/400000 [00:33<00:11, 8883.01it/s] 75%|███████▌  | 300027/400000 [00:33<00:11, 8921.17it/s] 75%|███████▌  | 300920/400000 [00:33<00:11, 8876.10it/s] 75%|███████▌  | 301808/400000 [00:33<00:11, 8694.25it/s] 76%|███████▌  | 302710/400000 [00:33<00:11, 8786.73it/s] 76%|███████▌  | 303630/400000 [00:34<00:10, 8904.41it/s] 76%|███████▌  | 304565/400000 [00:34<00:10, 9032.74it/s] 76%|███████▋  | 305482/400000 [00:34<00:10, 9073.40it/s] 77%|███████▋  | 306398/400000 [00:34<00:10, 9097.97it/s] 77%|███████▋  | 307322/400000 [00:34<00:10, 9139.49it/s] 77%|███████▋  | 308237/400000 [00:34<00:10, 9129.83it/s] 77%|███████▋  | 309151/400000 [00:34<00:09, 9130.46it/s] 78%|███████▊  | 310073/400000 [00:34<00:09, 9155.16it/s] 78%|███████▊  | 311009/400000 [00:34<00:09, 9215.35it/s] 78%|███████▊  | 311931/400000 [00:34<00:09, 9127.18it/s] 78%|███████▊  | 312845/400000 [00:35<00:09, 9017.34it/s] 78%|███████▊  | 313748/400000 [00:35<00:09, 8958.33it/s] 79%|███████▊  | 314645/400000 [00:35<00:09, 8916.11it/s] 79%|███████▉  | 315537/400000 [00:35<00:09, 8907.23it/s] 79%|███████▉  | 316435/400000 [00:35<00:09, 8928.34it/s] 79%|███████▉  | 317329/400000 [00:35<00:09, 8903.54it/s] 80%|███████▉  | 318226/400000 [00:35<00:09, 8920.64it/s] 80%|███████▉  | 319119/400000 [00:35<00:09, 8905.42it/s] 80%|████████  | 320035/400000 [00:35<00:08, 8979.59it/s] 80%|████████  | 320948/400000 [00:36<00:08, 9023.62it/s] 80%|████████  | 321851/400000 [00:36<00:08, 9009.56it/s] 81%|████████  | 322753/400000 [00:36<00:08, 8942.41it/s] 81%|████████  | 323649/400000 [00:36<00:08, 8946.96it/s] 81%|████████  | 324544/400000 [00:36<00:08, 8829.12it/s] 81%|████████▏ | 325429/400000 [00:36<00:08, 8835.06it/s] 82%|████████▏ | 326313/400000 [00:36<00:08, 8825.75it/s] 82%|████████▏ | 327196/400000 [00:36<00:08, 8799.85it/s] 82%|████████▏ | 328092/400000 [00:36<00:08, 8847.09it/s] 82%|████████▏ | 328977/400000 [00:36<00:08, 8768.91it/s] 82%|████████▏ | 329875/400000 [00:37<00:07, 8831.14it/s] 83%|████████▎ | 330790/400000 [00:37<00:07, 8922.29it/s] 83%|████████▎ | 331683/400000 [00:37<00:07, 8658.55it/s] 83%|████████▎ | 332577/400000 [00:37<00:07, 8741.04it/s] 83%|████████▎ | 333478/400000 [00:37<00:07, 8818.85it/s] 84%|████████▎ | 334374/400000 [00:37<00:07, 8859.13it/s] 84%|████████▍ | 335267/400000 [00:37<00:07, 8878.94it/s] 84%|████████▍ | 336160/400000 [00:37<00:07, 8892.24it/s] 84%|████████▍ | 337050/400000 [00:37<00:07, 8839.59it/s] 84%|████████▍ | 337935/400000 [00:37<00:07, 8826.58it/s] 85%|████████▍ | 338838/400000 [00:38<00:06, 8885.25it/s] 85%|████████▍ | 339742/400000 [00:38<00:06, 8929.83it/s] 85%|████████▌ | 340636/400000 [00:38<00:06, 8904.15it/s] 85%|████████▌ | 341527/400000 [00:38<00:06, 8752.40it/s] 86%|████████▌ | 342409/400000 [00:38<00:06, 8771.64it/s] 86%|████████▌ | 343307/400000 [00:38<00:06, 8832.69it/s] 86%|████████▌ | 344196/400000 [00:38<00:06, 8849.26it/s] 86%|████████▋ | 345098/400000 [00:38<00:06, 8899.62it/s] 86%|████████▋ | 345989/400000 [00:38<00:06, 8657.63it/s] 87%|████████▋ | 346857/400000 [00:38<00:06, 8592.12it/s] 87%|████████▋ | 347748/400000 [00:39<00:06, 8684.92it/s] 87%|████████▋ | 348645/400000 [00:39<00:05, 8768.45it/s] 87%|████████▋ | 349523/400000 [00:39<00:05, 8763.23it/s] 88%|████████▊ | 350406/400000 [00:39<00:05, 8782.72it/s] 88%|████████▊ | 351291/400000 [00:39<00:05, 8800.03it/s] 88%|████████▊ | 352196/400000 [00:39<00:05, 8871.40it/s] 88%|████████▊ | 353091/400000 [00:39<00:05, 8894.55it/s] 88%|████████▊ | 353989/400000 [00:39<00:05, 8917.02it/s] 89%|████████▊ | 354881/400000 [00:39<00:05, 8905.04it/s] 89%|████████▉ | 355775/400000 [00:39<00:04, 8914.29it/s] 89%|████████▉ | 356667/400000 [00:40<00:04, 8894.73it/s] 89%|████████▉ | 357557/400000 [00:40<00:04, 8892.75it/s] 90%|████████▉ | 358447/400000 [00:40<00:04, 8863.86it/s] 90%|████████▉ | 359334/400000 [00:40<00:04, 8859.68it/s] 90%|█████████ | 360221/400000 [00:40<00:04, 8835.08it/s] 90%|█████████ | 361105/400000 [00:40<00:04, 8828.72it/s] 90%|█████████ | 361988/400000 [00:40<00:04, 8812.81it/s] 91%|█████████ | 362874/400000 [00:40<00:04, 8824.25it/s] 91%|█████████ | 363757/400000 [00:40<00:04, 8813.48it/s] 91%|█████████ | 364650/400000 [00:40<00:03, 8846.04it/s] 91%|█████████▏| 365535/400000 [00:41<00:03, 8837.95it/s] 92%|█████████▏| 366419/400000 [00:41<00:03, 8827.92it/s] 92%|█████████▏| 367302/400000 [00:41<00:03, 8821.22it/s] 92%|█████████▏| 368185/400000 [00:41<00:03, 8820.15it/s] 92%|█████████▏| 369094/400000 [00:41<00:03, 8898.52it/s] 92%|█████████▏| 369985/400000 [00:41<00:03, 8887.43it/s] 93%|█████████▎| 370875/400000 [00:41<00:03, 8888.30it/s] 93%|█████████▎| 371764/400000 [00:41<00:03, 8886.23it/s] 93%|█████████▎| 372660/400000 [00:41<00:03, 8905.82it/s] 93%|█████████▎| 373560/400000 [00:41<00:02, 8932.58it/s] 94%|█████████▎| 374463/400000 [00:42<00:02, 8960.58it/s] 94%|█████████▍| 375374/400000 [00:42<00:02, 9003.22it/s] 94%|█████████▍| 376275/400000 [00:42<00:02, 8750.62it/s] 94%|█████████▍| 377152/400000 [00:42<00:02, 8663.26it/s] 95%|█████████▍| 378065/400000 [00:42<00:02, 8796.38it/s] 95%|█████████▍| 378947/400000 [00:42<00:02, 8790.56it/s] 95%|█████████▍| 379865/400000 [00:42<00:02, 8903.57it/s] 95%|█████████▌| 380759/400000 [00:42<00:02, 8912.33it/s] 95%|█████████▌| 381664/400000 [00:42<00:02, 8951.80it/s] 96%|█████████▌| 382574/400000 [00:42<00:01, 8993.20it/s] 96%|█████████▌| 383488/400000 [00:43<00:01, 9035.83it/s] 96%|█████████▌| 384414/400000 [00:43<00:01, 9099.58it/s] 96%|█████████▋| 385329/400000 [00:43<00:01, 9111.95it/s] 97%|█████████▋| 386241/400000 [00:43<00:01, 9018.43it/s] 97%|█████████▋| 387144/400000 [00:43<00:01, 9011.66it/s] 97%|█████████▋| 388046/400000 [00:43<00:01, 9011.03it/s] 97%|█████████▋| 388948/400000 [00:43<00:01, 9007.14it/s] 97%|█████████▋| 389862/400000 [00:43<00:01, 9045.33it/s] 98%|█████████▊| 390767/400000 [00:43<00:01, 9013.85it/s] 98%|█████████▊| 391669/400000 [00:43<00:00, 9013.66it/s] 98%|█████████▊| 392592/400000 [00:44<00:00, 9077.25it/s] 98%|█████████▊| 393500/400000 [00:44<00:00, 8999.07it/s] 99%|█████████▊| 394419/400000 [00:44<00:00, 9055.43it/s] 99%|█████████▉| 395329/400000 [00:44<00:00, 9065.99it/s] 99%|█████████▉| 396243/400000 [00:44<00:00, 9086.78it/s] 99%|█████████▉| 397152/400000 [00:44<00:00, 9084.27it/s]100%|█████████▉| 398077/400000 [00:44<00:00, 9133.13it/s]100%|█████████▉| 399005/400000 [00:44<00:00, 9174.70it/s]100%|█████████▉| 399923/400000 [00:44<00:00, 9154.65it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8910.22it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb75e3c8a58> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01089526740607615 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.010982483725085306 	 Accuracy: 62

  model saves at 62% accuracy 

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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

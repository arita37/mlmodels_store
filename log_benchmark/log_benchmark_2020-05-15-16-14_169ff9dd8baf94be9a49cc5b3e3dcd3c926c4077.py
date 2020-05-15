
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/169ff9dd8baf94be9a49cc5b3e3dcd3c926c4077

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7efc76866fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 16:14:29.804566
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 16:14:29.808284
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 16:14:29.811335
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 16:14:29.814326
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7efc8287e4a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352208.8750
Epoch 2/10

1/1 [==============================] - 0s 105ms/step - loss: 271077.2188
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 178486.1094
Epoch 4/10

1/1 [==============================] - 0s 100ms/step - loss: 105392.1016
Epoch 5/10

1/1 [==============================] - 0s 98ms/step - loss: 59532.4336
Epoch 6/10

1/1 [==============================] - 0s 97ms/step - loss: 34602.1992
Epoch 7/10

1/1 [==============================] - 0s 97ms/step - loss: 21484.3125
Epoch 8/10

1/1 [==============================] - 0s 105ms/step - loss: 14310.8682
Epoch 9/10

1/1 [==============================] - 0s 101ms/step - loss: 10141.7012
Epoch 10/10

1/1 [==============================] - 0s 90ms/step - loss: 7588.2056

  #### Inference Need return ypred, ytrue ######################### 
[[-1.39036179e+00 -7.96137452e-01  8.08778524e-01  1.09502733e+00
  -1.80908263e-01 -5.83698869e-01  1.00940013e+00 -6.92255497e-01
   1.33108735e+00 -2.33058834e+00 -3.05460691e-01 -7.57493973e-01
  -1.36702156e+00  2.45795876e-01  8.29463303e-01  1.36077464e+00
   2.06202820e-01  4.06402767e-01 -1.93874443e+00  8.30685973e-01
  -3.34847212e-01 -6.10952914e-01  5.42778373e-02  5.20475090e-01
   3.49045157e-01  1.19079351e-02 -1.67371440e+00  6.77284598e-01
  -1.50737858e+00 -1.42843172e-01  1.68959010e+00 -1.80285186e-01
   8.42967629e-01 -2.41089165e-02  1.59711301e-01 -1.07202685e+00
  -2.92105436e-01  4.91590679e-01 -2.50947535e-01 -3.85305166e-01
   1.49467242e+00 -9.24747229e-01  1.97741121e-01  3.26623917e-02
   1.67175162e+00  5.42007208e-01 -1.43458098e-01 -5.64255238e-01
   1.99611425e+00 -2.01473951e+00 -7.30995834e-01 -2.80801177e-01
  -3.22309583e-02 -1.28459588e-01 -2.14931083e+00 -2.48942170e-02
   9.09480095e-01 -1.73895717e+00 -8.00732791e-01 -5.01211166e-01
  -9.06789377e-02  6.45150137e+00  7.55277109e+00  5.62204981e+00
   6.84021521e+00  8.57197952e+00  6.63128805e+00  7.27517080e+00
   8.39374256e+00  8.29193020e+00  5.61844444e+00  7.38221550e+00
   7.83065748e+00  5.37294722e+00  9.28242588e+00  7.03146791e+00
   7.50648737e+00  9.57337666e+00  5.54274702e+00  8.76450348e+00
   7.74126625e+00  7.01693821e+00  6.88891172e+00  8.59048176e+00
   7.18093252e+00  6.17365742e+00  8.41601944e+00  7.41585827e+00
   8.04503632e+00  6.85706854e+00  6.86342812e+00  6.65505934e+00
   8.68065453e+00  7.54848480e+00  6.51858616e+00  8.26699924e+00
   7.21873331e+00  6.48724699e+00  6.81250620e+00  7.51235771e+00
   6.51094723e+00  7.59233379e+00  8.29264164e+00  8.41921425e+00
   7.50688696e+00  5.99400806e+00  5.83251905e+00  6.10991526e+00
   8.40700722e+00  7.86811543e+00  6.75327492e+00  7.70483923e+00
   7.83236933e+00  6.39124775e+00  7.01255226e+00  8.08752441e+00
   9.30638123e+00  8.11681271e+00  7.82782698e+00  7.89024401e+00
  -3.29328775e-01  9.71984804e-01 -3.45683455e-01  1.08949733e+00
   7.85353065e-01 -1.12065226e-01 -8.61816883e-01 -7.69101739e-01
   4.41202253e-01  5.88989675e-01  3.58499229e-01  9.69923019e-01
   3.22881401e-01  1.14863467e+00  1.06733632e+00  7.92638600e-01
  -1.57442296e+00  8.01664472e-01  4.09990966e-01  1.29817796e+00
   5.72253913e-02 -3.90172243e-01 -1.15237975e+00 -1.80655718e-01
  -9.84787464e-01  4.43450123e-01  9.12115574e-01  9.22439396e-02
  -4.25657034e-02 -5.19407630e-01  2.47309372e-01 -7.50246346e-01
   1.55441928e+00  3.82123172e-01 -7.92488158e-01 -1.75022471e+00
  -1.27381408e+00  1.51046872e-01 -6.38011098e-01 -7.43495226e-02
  -4.00600433e-02  4.48016018e-01  4.58997637e-01  1.14777803e+00
   2.17555761e+00  1.40559629e-01 -1.57430768e+00  1.98750567e+00
   5.53615391e-01  1.21184039e+00 -1.52801871e+00  3.83810878e-01
  -7.87663877e-01  9.01186168e-01 -8.32580984e-01  1.28698480e+00
  -8.79320949e-02  1.98547304e-01  8.41028810e-01  2.01876014e-01
   1.82709575e+00  4.46934640e-01  1.50771141e+00  1.42998791e+00
   1.19990253e+00  8.60841036e-01  1.04035580e+00  1.76613867e-01
   6.99418068e-01  9.72497165e-01  8.28817964e-01  4.86993790e-01
   1.51254773e+00  6.61848009e-01  7.37589300e-01  1.26965415e+00
   5.13857543e-01  1.88144505e-01  1.89228892e+00  7.38288283e-01
   1.31516719e+00  1.26126170e-01  1.31150913e+00  5.55646479e-01
   1.20716465e+00  8.62748027e-02  2.23332942e-01  6.77271307e-01
   1.71515250e+00  6.17733479e-01  4.20644701e-01  6.87379837e-01
   1.43853974e+00  3.39018703e-01  9.26441133e-01  3.14222956e+00
   1.77514076e+00  1.87055707e+00  7.51899123e-01  4.42963243e-01
   1.53922343e+00  1.10421216e+00  1.73140407e-01  1.00392962e+00
   6.78670883e-01  2.24248600e+00  1.15024042e+00  2.25852919e+00
   5.42707086e-01  1.04515731e+00  1.33419633e-01  2.27662981e-01
   1.46720040e+00  7.88329720e-01  2.59433460e+00  1.47446752e+00
   9.01675224e-01  2.08848524e+00  7.82003522e-01  1.29221773e+00
   1.13219619e-01  8.21206284e+00  7.87639332e+00  8.68495560e+00
   6.94089842e+00  9.45350552e+00  7.17333364e+00  7.09636402e+00
   8.74621487e+00  8.28892517e+00  8.54102039e+00  8.19024754e+00
   9.32094002e+00  9.17152500e+00  9.04431534e+00  9.29598904e+00
   7.00615454e+00  6.57565498e+00  7.11136818e+00  6.59901857e+00
   7.43857527e+00  6.85533762e+00  7.85150480e+00  8.50247192e+00
   8.95670795e+00  6.84007072e+00  7.82475090e+00  9.24291039e+00
   7.49565792e+00  8.57109547e+00  8.02998543e+00  8.08916664e+00
   6.70470524e+00  7.73782921e+00  9.19911957e+00  8.27941799e+00
   8.16015816e+00  7.46684551e+00  9.11901569e+00  6.40197659e+00
   6.77043152e+00  6.76640034e+00  8.78844833e+00  7.59820366e+00
   8.97009087e+00  8.18531895e+00  7.32272053e+00  8.64767075e+00
   6.58654642e+00  7.02822733e+00  7.31119823e+00  8.27500820e+00
   8.35282421e+00  7.14092445e+00  7.64189053e+00  7.50539351e+00
   5.96692085e+00  8.12729263e+00  7.18976688e+00  6.58161402e+00
   1.42677534e+00  1.40489316e+00  2.17053270e+00  3.81250024e-01
   1.25339425e+00  6.16543591e-01  3.41768980e-01  8.61489713e-01
   2.53437948e+00  1.11484087e+00  1.37343919e+00  1.14147782e-01
   7.81136870e-01  1.27290475e+00  1.61950004e+00  3.57465744e-01
   8.36342931e-01  1.40072584e+00  2.40524173e-01  7.26330161e-01
   5.39566815e-01  6.11749649e-01  5.16267240e-01  7.46875763e-01
   2.73468018e+00  1.99766397e+00  5.81345022e-01  1.91492653e+00
   2.16277504e+00  2.36342669e-01  7.46194124e-01  5.92913866e-01
   1.02702379e+00  2.83920765e+00  2.39693880e+00  2.02563453e+00
   2.74667048e+00  2.33451796e+00  1.14401007e+00  1.45130885e+00
   4.46667075e-01  4.59642768e-01  5.59997201e-01  4.01442409e-01
   5.09678066e-01  2.34721661e-01  3.87582064e-01  8.07501853e-01
   1.21922076e+00  2.40180159e+00  2.40067303e-01  2.27016497e+00
   5.24241388e-01  2.07328260e-01  8.52823079e-01  4.27350581e-01
   1.98672795e+00  2.17314959e+00  3.22875082e-01  4.36149299e-01
  -5.37001562e+00  2.64125562e+00 -1.28812675e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 16:14:38.510617
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.5369
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 16:14:38.514362
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8958.97
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 16:14:38.517671
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.0397
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 16:14:38.520781
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -801.339
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139622445544896
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139621486604808
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139621486605312
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139621486605816
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139621486606320
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139621486606824

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7efc624a0e10> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.579980
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.543892
grad_step = 000002, loss = 0.514782
grad_step = 000003, loss = 0.483503
grad_step = 000004, loss = 0.448790
grad_step = 000005, loss = 0.413786
grad_step = 000006, loss = 0.384424
grad_step = 000007, loss = 0.365379
grad_step = 000008, loss = 0.348831
grad_step = 000009, loss = 0.324372
grad_step = 000010, loss = 0.301815
grad_step = 000011, loss = 0.288293
grad_step = 000012, loss = 0.277891
grad_step = 000013, loss = 0.265708
grad_step = 000014, loss = 0.250401
grad_step = 000015, loss = 0.233521
grad_step = 000016, loss = 0.217111
grad_step = 000017, loss = 0.202536
grad_step = 000018, loss = 0.191356
grad_step = 000019, loss = 0.179788
grad_step = 000020, loss = 0.167094
grad_step = 000021, loss = 0.155703
grad_step = 000022, loss = 0.146059
grad_step = 000023, loss = 0.136773
grad_step = 000024, loss = 0.127335
grad_step = 000025, loss = 0.117917
grad_step = 000026, loss = 0.108504
grad_step = 000027, loss = 0.100090
grad_step = 000028, loss = 0.092445
grad_step = 000029, loss = 0.084932
grad_step = 000030, loss = 0.077176
grad_step = 000031, loss = 0.070094
grad_step = 000032, loss = 0.063725
grad_step = 000033, loss = 0.058061
grad_step = 000034, loss = 0.052621
grad_step = 000035, loss = 0.047308
grad_step = 000036, loss = 0.042367
grad_step = 000037, loss = 0.038048
grad_step = 000038, loss = 0.034045
grad_step = 000039, loss = 0.030336
grad_step = 000040, loss = 0.026813
grad_step = 000041, loss = 0.023709
grad_step = 000042, loss = 0.021016
grad_step = 000043, loss = 0.018536
grad_step = 000044, loss = 0.016261
grad_step = 000045, loss = 0.014243
grad_step = 000046, loss = 0.012527
grad_step = 000047, loss = 0.011052
grad_step = 000048, loss = 0.009684
grad_step = 000049, loss = 0.008512
grad_step = 000050, loss = 0.007582
grad_step = 000051, loss = 0.006804
grad_step = 000052, loss = 0.006122
grad_step = 000053, loss = 0.005465
grad_step = 000054, loss = 0.004925
grad_step = 000055, loss = 0.004538
grad_step = 000056, loss = 0.004230
grad_step = 000057, loss = 0.003977
grad_step = 000058, loss = 0.003771
grad_step = 000059, loss = 0.003625
grad_step = 000060, loss = 0.003500
grad_step = 000061, loss = 0.003378
grad_step = 000062, loss = 0.003279
grad_step = 000063, loss = 0.003210
grad_step = 000064, loss = 0.003168
grad_step = 000065, loss = 0.003113
grad_step = 000066, loss = 0.003055
grad_step = 000067, loss = 0.003006
grad_step = 000068, loss = 0.002966
grad_step = 000069, loss = 0.002926
grad_step = 000070, loss = 0.002881
grad_step = 000071, loss = 0.002846
grad_step = 000072, loss = 0.002811
grad_step = 000073, loss = 0.002767
grad_step = 000074, loss = 0.002716
grad_step = 000075, loss = 0.002675
grad_step = 000076, loss = 0.002643
grad_step = 000077, loss = 0.002612
grad_step = 000078, loss = 0.002578
grad_step = 000079, loss = 0.002546
grad_step = 000080, loss = 0.002519
grad_step = 000081, loss = 0.002490
grad_step = 000082, loss = 0.002462
grad_step = 000083, loss = 0.002437
grad_step = 000084, loss = 0.002417
grad_step = 000085, loss = 0.002397
grad_step = 000086, loss = 0.002377
grad_step = 000087, loss = 0.002360
grad_step = 000088, loss = 0.002348
grad_step = 000089, loss = 0.002339
grad_step = 000090, loss = 0.002338
grad_step = 000091, loss = 0.002348
grad_step = 000092, loss = 0.002370
grad_step = 000093, loss = 0.002394
grad_step = 000094, loss = 0.002395
grad_step = 000095, loss = 0.002352
grad_step = 000096, loss = 0.002278
grad_step = 000097, loss = 0.002234
grad_step = 000098, loss = 0.002245
grad_step = 000099, loss = 0.002281
grad_step = 000100, loss = 0.002293
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002263
grad_step = 000102, loss = 0.002222
grad_step = 000103, loss = 0.002211
grad_step = 000104, loss = 0.002230
grad_step = 000105, loss = 0.002249
grad_step = 000106, loss = 0.002242
grad_step = 000107, loss = 0.002216
grad_step = 000108, loss = 0.002199
grad_step = 000109, loss = 0.002203
grad_step = 000110, loss = 0.002217
grad_step = 000111, loss = 0.002221
grad_step = 000112, loss = 0.002210
grad_step = 000113, loss = 0.002194
grad_step = 000114, loss = 0.002185
grad_step = 000115, loss = 0.002187
grad_step = 000116, loss = 0.002194
grad_step = 000117, loss = 0.002196
grad_step = 000118, loss = 0.002190
grad_step = 000119, loss = 0.002181
grad_step = 000120, loss = 0.002172
grad_step = 000121, loss = 0.002168
grad_step = 000122, loss = 0.002169
grad_step = 000123, loss = 0.002171
grad_step = 000124, loss = 0.002173
grad_step = 000125, loss = 0.002172
grad_step = 000126, loss = 0.002169
grad_step = 000127, loss = 0.002164
grad_step = 000128, loss = 0.002159
grad_step = 000129, loss = 0.002154
grad_step = 000130, loss = 0.002151
grad_step = 000131, loss = 0.002149
grad_step = 000132, loss = 0.002148
grad_step = 000133, loss = 0.002147
grad_step = 000134, loss = 0.002146
grad_step = 000135, loss = 0.002147
grad_step = 000136, loss = 0.002148
grad_step = 000137, loss = 0.002151
grad_step = 000138, loss = 0.002157
grad_step = 000139, loss = 0.002168
grad_step = 000140, loss = 0.002188
grad_step = 000141, loss = 0.002223
grad_step = 000142, loss = 0.002277
grad_step = 000143, loss = 0.002342
grad_step = 000144, loss = 0.002383
grad_step = 000145, loss = 0.002352
grad_step = 000146, loss = 0.002243
grad_step = 000147, loss = 0.002141
grad_step = 000148, loss = 0.002128
grad_step = 000149, loss = 0.002189
grad_step = 000150, loss = 0.002244
grad_step = 000151, loss = 0.002228
grad_step = 000152, loss = 0.002160
grad_step = 000153, loss = 0.002113
grad_step = 000154, loss = 0.002129
grad_step = 000155, loss = 0.002172
grad_step = 000156, loss = 0.002183
grad_step = 000157, loss = 0.002150
grad_step = 000158, loss = 0.002110
grad_step = 000159, loss = 0.002103
grad_step = 000160, loss = 0.002126
grad_step = 000161, loss = 0.002145
grad_step = 000162, loss = 0.002136
grad_step = 000163, loss = 0.002108
grad_step = 000164, loss = 0.002091
grad_step = 000165, loss = 0.002095
grad_step = 000166, loss = 0.002109
grad_step = 000167, loss = 0.002114
grad_step = 000168, loss = 0.002104
grad_step = 000169, loss = 0.002088
grad_step = 000170, loss = 0.002078
grad_step = 000171, loss = 0.002078
grad_step = 000172, loss = 0.002084
grad_step = 000173, loss = 0.002088
grad_step = 000174, loss = 0.002085
grad_step = 000175, loss = 0.002076
grad_step = 000176, loss = 0.002067
grad_step = 000177, loss = 0.002061
grad_step = 000178, loss = 0.002060
grad_step = 000179, loss = 0.002061
grad_step = 000180, loss = 0.002062
grad_step = 000181, loss = 0.002062
grad_step = 000182, loss = 0.002058
grad_step = 000183, loss = 0.002053
grad_step = 000184, loss = 0.002048
grad_step = 000185, loss = 0.002042
grad_step = 000186, loss = 0.002038
grad_step = 000187, loss = 0.002035
grad_step = 000188, loss = 0.002034
grad_step = 000189, loss = 0.002038
grad_step = 000190, loss = 0.002047
grad_step = 000191, loss = 0.002055
grad_step = 000192, loss = 0.002050
grad_step = 000193, loss = 0.002032
grad_step = 000194, loss = 0.002022
grad_step = 000195, loss = 0.002030
grad_step = 000196, loss = 0.002045
grad_step = 000197, loss = 0.002054
grad_step = 000198, loss = 0.002052
grad_step = 000199, loss = 0.002060
grad_step = 000200, loss = 0.002094
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002160
grad_step = 000202, loss = 0.002255
grad_step = 000203, loss = 0.002345
grad_step = 000204, loss = 0.002410
grad_step = 000205, loss = 0.002405
grad_step = 000206, loss = 0.002304
grad_step = 000207, loss = 0.002163
grad_step = 000208, loss = 0.002020
grad_step = 000209, loss = 0.002031
grad_step = 000210, loss = 0.002153
grad_step = 000211, loss = 0.002200
grad_step = 000212, loss = 0.002143
grad_step = 000213, loss = 0.002059
grad_step = 000214, loss = 0.002019
grad_step = 000215, loss = 0.002030
grad_step = 000216, loss = 0.002067
grad_step = 000217, loss = 0.002100
grad_step = 000218, loss = 0.002079
grad_step = 000219, loss = 0.002014
grad_step = 000220, loss = 0.001983
grad_step = 000221, loss = 0.002013
grad_step = 000222, loss = 0.002049
grad_step = 000223, loss = 0.002047
grad_step = 000224, loss = 0.002012
grad_step = 000225, loss = 0.001990
grad_step = 000226, loss = 0.001997
grad_step = 000227, loss = 0.002007
grad_step = 000228, loss = 0.002003
grad_step = 000229, loss = 0.001992
grad_step = 000230, loss = 0.001991
grad_step = 000231, loss = 0.001998
grad_step = 000232, loss = 0.001992
grad_step = 000233, loss = 0.001978
grad_step = 000234, loss = 0.001971
grad_step = 000235, loss = 0.001977
grad_step = 000236, loss = 0.001987
grad_step = 000237, loss = 0.001986
grad_step = 000238, loss = 0.001978
grad_step = 000239, loss = 0.001970
grad_step = 000240, loss = 0.001969
grad_step = 000241, loss = 0.001970
grad_step = 000242, loss = 0.001969
grad_step = 000243, loss = 0.001966
grad_step = 000244, loss = 0.001964
grad_step = 000245, loss = 0.001965
grad_step = 000246, loss = 0.001969
grad_step = 000247, loss = 0.001975
grad_step = 000248, loss = 0.001982
grad_step = 000249, loss = 0.001990
grad_step = 000250, loss = 0.002001
grad_step = 000251, loss = 0.002021
grad_step = 000252, loss = 0.002038
grad_step = 000253, loss = 0.002056
grad_step = 000254, loss = 0.002060
grad_step = 000255, loss = 0.002062
grad_step = 000256, loss = 0.002063
grad_step = 000257, loss = 0.002062
grad_step = 000258, loss = 0.002048
grad_step = 000259, loss = 0.002022
grad_step = 000260, loss = 0.001993
grad_step = 000261, loss = 0.001977
grad_step = 000262, loss = 0.001978
grad_step = 000263, loss = 0.001997
grad_step = 000264, loss = 0.002010
grad_step = 000265, loss = 0.002020
grad_step = 000266, loss = 0.001995
grad_step = 000267, loss = 0.001970
grad_step = 000268, loss = 0.001970
grad_step = 000269, loss = 0.001993
grad_step = 000270, loss = 0.002008
grad_step = 000271, loss = 0.001996
grad_step = 000272, loss = 0.001980
grad_step = 000273, loss = 0.001981
grad_step = 000274, loss = 0.001998
grad_step = 000275, loss = 0.002018
grad_step = 000276, loss = 0.002022
grad_step = 000277, loss = 0.002029
grad_step = 000278, loss = 0.002037
grad_step = 000279, loss = 0.002061
grad_step = 000280, loss = 0.002093
grad_step = 000281, loss = 0.002121
grad_step = 000282, loss = 0.002128
grad_step = 000283, loss = 0.002116
grad_step = 000284, loss = 0.002088
grad_step = 000285, loss = 0.002054
grad_step = 000286, loss = 0.002009
grad_step = 000287, loss = 0.001965
grad_step = 000288, loss = 0.001933
grad_step = 000289, loss = 0.001923
grad_step = 000290, loss = 0.001933
grad_step = 000291, loss = 0.001951
grad_step = 000292, loss = 0.001969
grad_step = 000293, loss = 0.001982
grad_step = 000294, loss = 0.001994
grad_step = 000295, loss = 0.001999
grad_step = 000296, loss = 0.001997
grad_step = 000297, loss = 0.001986
grad_step = 000298, loss = 0.001969
grad_step = 000299, loss = 0.001953
grad_step = 000300, loss = 0.001940
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001929
grad_step = 000302, loss = 0.001921
grad_step = 000303, loss = 0.001914
grad_step = 000304, loss = 0.001911
grad_step = 000305, loss = 0.001911
grad_step = 000306, loss = 0.001914
grad_step = 000307, loss = 0.001918
grad_step = 000308, loss = 0.001922
grad_step = 000309, loss = 0.001925
grad_step = 000310, loss = 0.001930
grad_step = 000311, loss = 0.001938
grad_step = 000312, loss = 0.001949
grad_step = 000313, loss = 0.001964
grad_step = 000314, loss = 0.001984
grad_step = 000315, loss = 0.002012
grad_step = 000316, loss = 0.002049
grad_step = 000317, loss = 0.002102
grad_step = 000318, loss = 0.002158
grad_step = 000319, loss = 0.002220
grad_step = 000320, loss = 0.002248
grad_step = 000321, loss = 0.002243
grad_step = 000322, loss = 0.002174
grad_step = 000323, loss = 0.002072
grad_step = 000324, loss = 0.001971
grad_step = 000325, loss = 0.001909
grad_step = 000326, loss = 0.001903
grad_step = 000327, loss = 0.001940
grad_step = 000328, loss = 0.001991
grad_step = 000329, loss = 0.002024
grad_step = 000330, loss = 0.002020
grad_step = 000331, loss = 0.001981
grad_step = 000332, loss = 0.001934
grad_step = 000333, loss = 0.001904
grad_step = 000334, loss = 0.001900
grad_step = 000335, loss = 0.001912
grad_step = 000336, loss = 0.001924
grad_step = 000337, loss = 0.001927
grad_step = 000338, loss = 0.001924
grad_step = 000339, loss = 0.001920
grad_step = 000340, loss = 0.001916
grad_step = 000341, loss = 0.001912
grad_step = 000342, loss = 0.001901
grad_step = 000343, loss = 0.001890
grad_step = 000344, loss = 0.001882
grad_step = 000345, loss = 0.001881
grad_step = 000346, loss = 0.001886
grad_step = 000347, loss = 0.001895
grad_step = 000348, loss = 0.001904
grad_step = 000349, loss = 0.001908
grad_step = 000350, loss = 0.001911
grad_step = 000351, loss = 0.001907
grad_step = 000352, loss = 0.001903
grad_step = 000353, loss = 0.001898
grad_step = 000354, loss = 0.001895
grad_step = 000355, loss = 0.001894
grad_step = 000356, loss = 0.001893
grad_step = 000357, loss = 0.001890
grad_step = 000358, loss = 0.001885
grad_step = 000359, loss = 0.001880
grad_step = 000360, loss = 0.001875
grad_step = 000361, loss = 0.001872
grad_step = 000362, loss = 0.001872
grad_step = 000363, loss = 0.001873
grad_step = 000364, loss = 0.001878
grad_step = 000365, loss = 0.001885
grad_step = 000366, loss = 0.001897
grad_step = 000367, loss = 0.001907
grad_step = 000368, loss = 0.001922
grad_step = 000369, loss = 0.001934
grad_step = 000370, loss = 0.001956
grad_step = 000371, loss = 0.001990
grad_step = 000372, loss = 0.002056
grad_step = 000373, loss = 0.002156
grad_step = 000374, loss = 0.002292
grad_step = 000375, loss = 0.002428
grad_step = 000376, loss = 0.002510
grad_step = 000377, loss = 0.002459
grad_step = 000378, loss = 0.002311
grad_step = 000379, loss = 0.002115
grad_step = 000380, loss = 0.002019
grad_step = 000381, loss = 0.001948
grad_step = 000382, loss = 0.001958
grad_step = 000383, loss = 0.002037
grad_step = 000384, loss = 0.002098
grad_step = 000385, loss = 0.002054
grad_step = 000386, loss = 0.001907
grad_step = 000387, loss = 0.001852
grad_step = 000388, loss = 0.001922
grad_step = 000389, loss = 0.001989
grad_step = 000390, loss = 0.001988
grad_step = 000391, loss = 0.001920
grad_step = 000392, loss = 0.001878
grad_step = 000393, loss = 0.001885
grad_step = 000394, loss = 0.001901
grad_step = 000395, loss = 0.001920
grad_step = 000396, loss = 0.001912
grad_step = 000397, loss = 0.001898
grad_step = 000398, loss = 0.001885
grad_step = 000399, loss = 0.001862
grad_step = 000400, loss = 0.001849
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001857
grad_step = 000402, loss = 0.001879
grad_step = 000403, loss = 0.001886
grad_step = 000404, loss = 0.001864
grad_step = 000405, loss = 0.001839
grad_step = 000406, loss = 0.001833
grad_step = 000407, loss = 0.001844
grad_step = 000408, loss = 0.001855
grad_step = 000409, loss = 0.001853
grad_step = 000410, loss = 0.001843
grad_step = 000411, loss = 0.001836
grad_step = 000412, loss = 0.001833
grad_step = 000413, loss = 0.001836
grad_step = 000414, loss = 0.001835
grad_step = 000415, loss = 0.001832
grad_step = 000416, loss = 0.001828
grad_step = 000417, loss = 0.001828
grad_step = 000418, loss = 0.001829
grad_step = 000419, loss = 0.001828
grad_step = 000420, loss = 0.001825
grad_step = 000421, loss = 0.001820
grad_step = 000422, loss = 0.001818
grad_step = 000423, loss = 0.001819
grad_step = 000424, loss = 0.001821
grad_step = 000425, loss = 0.001821
grad_step = 000426, loss = 0.001819
grad_step = 000427, loss = 0.001817
grad_step = 000428, loss = 0.001815
grad_step = 000429, loss = 0.001814
grad_step = 000430, loss = 0.001813
grad_step = 000431, loss = 0.001812
grad_step = 000432, loss = 0.001811
grad_step = 000433, loss = 0.001809
grad_step = 000434, loss = 0.001807
grad_step = 000435, loss = 0.001806
grad_step = 000436, loss = 0.001806
grad_step = 000437, loss = 0.001805
grad_step = 000438, loss = 0.001805
grad_step = 000439, loss = 0.001805
grad_step = 000440, loss = 0.001804
grad_step = 000441, loss = 0.001804
grad_step = 000442, loss = 0.001804
grad_step = 000443, loss = 0.001804
grad_step = 000444, loss = 0.001805
grad_step = 000445, loss = 0.001808
grad_step = 000446, loss = 0.001812
grad_step = 000447, loss = 0.001820
grad_step = 000448, loss = 0.001833
grad_step = 000449, loss = 0.001855
grad_step = 000450, loss = 0.001893
grad_step = 000451, loss = 0.001954
grad_step = 000452, loss = 0.002058
grad_step = 000453, loss = 0.002210
grad_step = 000454, loss = 0.002441
grad_step = 000455, loss = 0.002669
grad_step = 000456, loss = 0.002840
grad_step = 000457, loss = 0.002714
grad_step = 000458, loss = 0.002343
grad_step = 000459, loss = 0.001925
grad_step = 000460, loss = 0.001792
grad_step = 000461, loss = 0.001967
grad_step = 000462, loss = 0.002199
grad_step = 000463, loss = 0.002226
grad_step = 000464, loss = 0.002006
grad_step = 000465, loss = 0.001819
grad_step = 000466, loss = 0.001838
grad_step = 000467, loss = 0.001974
grad_step = 000468, loss = 0.002023
grad_step = 000469, loss = 0.001918
grad_step = 000470, loss = 0.001817
grad_step = 000471, loss = 0.001829
grad_step = 000472, loss = 0.001899
grad_step = 000473, loss = 0.001910
grad_step = 000474, loss = 0.001846
grad_step = 000475, loss = 0.001803
grad_step = 000476, loss = 0.001821
grad_step = 000477, loss = 0.001855
grad_step = 000478, loss = 0.001844
grad_step = 000479, loss = 0.001802
grad_step = 000480, loss = 0.001790
grad_step = 000481, loss = 0.001812
grad_step = 000482, loss = 0.001828
grad_step = 000483, loss = 0.001807
grad_step = 000484, loss = 0.001776
grad_step = 000485, loss = 0.001775
grad_step = 000486, loss = 0.001794
grad_step = 000487, loss = 0.001804
grad_step = 000488, loss = 0.001786
grad_step = 000489, loss = 0.001765
grad_step = 000490, loss = 0.001765
grad_step = 000491, loss = 0.001779
grad_step = 000492, loss = 0.001786
grad_step = 000493, loss = 0.001776
grad_step = 000494, loss = 0.001761
grad_step = 000495, loss = 0.001758
grad_step = 000496, loss = 0.001764
grad_step = 000497, loss = 0.001769
grad_step = 000498, loss = 0.001765
grad_step = 000499, loss = 0.001757
grad_step = 000500, loss = 0.001753
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001755
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

  date_run                              2020-05-15 16:14:57.357786
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.225687
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 16:14:57.364266
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.119397
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 16:14:57.372278
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.137753
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 16:14:57.377328
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.814286
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
0   2020-05-15 16:14:29.804566  ...    mean_absolute_error
1   2020-05-15 16:14:29.808284  ...     mean_squared_error
2   2020-05-15 16:14:29.811335  ...  median_absolute_error
3   2020-05-15 16:14:29.814326  ...               r2_score
4   2020-05-15 16:14:38.510617  ...    mean_absolute_error
5   2020-05-15 16:14:38.514362  ...     mean_squared_error
6   2020-05-15 16:14:38.517671  ...  median_absolute_error
7   2020-05-15 16:14:38.520781  ...               r2_score
8   2020-05-15 16:14:57.357786  ...    mean_absolute_error
9   2020-05-15 16:14:57.364266  ...     mean_squared_error
10  2020-05-15 16:14:57.372278  ...  median_absolute_error
11  2020-05-15 16:14:57.377328  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cbda59898> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 25%|██▌       | 2482176/9912422 [00:00<00:00, 24806170.03it/s] 54%|█████▍    | 5398528/9912422 [00:00<00:00, 25065886.69it/s] 82%|████████▏ | 8151040/9912422 [00:00<00:00, 25201062.70it/s] 99%|█████████▉| 9822208/9912422 [00:00<00:00, 18636311.87it/s]9920512it [00:00, 13050783.43it/s]                             
0it [00:00, ?it/s]32768it [00:00, 546040.55it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 159179.35it/s]1654784it [00:00, 11243460.95it/s]                         
0it [00:00, ?it/s]8192it [00:00, 188153.98it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cbda59a90> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3c6d254048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cbda59a90> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cbda59898> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 

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
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3c70408dd8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3c6d254048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3c70408dd8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3c6d254048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3c6d2540b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
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
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f3cbda11e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fd5e7eaa1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=a8d5dd570a2992c5e62a4c79ce4a0d75fe9a7ad4ffd0391cda4b5d64b47b6ae7
  Stored in directory: /tmp/pip-ephem-wheel-cache-aluia5i0/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd57fca5710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
  614400/17464789 [>.............................] - ETA: 1s
 1482752/17464789 [=>............................] - ETA: 1s
 2392064/17464789 [===>..........................] - ETA: 0s
 3399680/17464789 [====>.........................] - ETA: 0s
 4456448/17464789 [======>.......................] - ETA: 0s
 5627904/17464789 [========>.....................] - ETA: 0s
 6815744/17464789 [==========>...................] - ETA: 0s
 8011776/17464789 [============>.................] - ETA: 0s
 9314304/17464789 [==============>...............] - ETA: 0s
10723328/17464789 [=================>............] - ETA: 0s
12206080/17464789 [===================>..........] - ETA: 0s
13770752/17464789 [======================>.......] - ETA: 0s
15409152/17464789 [=========================>....] - ETA: 0s
17096704/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 16:16:24.749624: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 16:16:24.754206: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095230000 Hz
2020-05-15 16:16:24.754430: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5614f10e0290 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 16:16:24.754446: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7126 - accuracy: 0.4970
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7050 - accuracy: 0.4975 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7075 - accuracy: 0.4973
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6781 - accuracy: 0.4992
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6390 - accuracy: 0.5018
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6334 - accuracy: 0.5022
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6907 - accuracy: 0.4984
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6858 - accuracy: 0.4988
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6421 - accuracy: 0.5016
11000/25000 [============>.................] - ETA: 3s - loss: 7.6262 - accuracy: 0.5026
12000/25000 [=============>................] - ETA: 3s - loss: 7.6104 - accuracy: 0.5037
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6218 - accuracy: 0.5029
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6217 - accuracy: 0.5029
15000/25000 [=================>............] - ETA: 2s - loss: 7.6431 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6618 - accuracy: 0.5003
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6630 - accuracy: 0.5002
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6504 - accuracy: 0.5011
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6424 - accuracy: 0.5016
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6398 - accuracy: 0.5017
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6330 - accuracy: 0.5022
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6600 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 7s 277us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 16:16:38.269536
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 16:16:38.269536  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<16:36:32, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<11:51:29, 20.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<8:21:17, 28.7kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:00<5:51:28, 40.8kB/s].vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:01<4:05:27, 58.3kB/s].vector_cache/glove.6B.zip:   1%|          | 8.23M/862M [00:01<2:50:58, 83.2kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:01<1:59:12, 119kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.2M/862M [00:01<1:22:59, 170kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.9M/862M [00:01<57:46, 242kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 27.2M/862M [00:01<40:24, 344kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.0M/862M [00:01<28:12, 490kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.8M/862M [00:01<19:45, 697kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.0M/862M [00:01<13:51, 988kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.3M/862M [00:02<09:45, 1.40MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.0M/862M [00:02<06:54, 1.97MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<05:20, 2.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<05:38, 2.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<07:35, 1.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:04<06:07, 2.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.4M/862M [00:05<04:30, 2.97MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<08:49, 1.51MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<07:34, 1.76MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:06<05:42, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:07<04:09, 3.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<1:09:13, 192kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<49:59, 266kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 66.4M/862M [00:08<35:19, 375kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<27:25, 482kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<22:10, 596kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:10<16:09, 817kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:11<11:26, 1.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<15:52, 828kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:12<12:26, 1.06MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:12<09:01, 1.45MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<09:24, 1.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:14<09:14, 1.41MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:14<07:08, 1.83MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<07:05, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<06:18, 2.06MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:16<04:41, 2.76MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<06:18, 2.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<05:46, 2.24MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:18<04:19, 2.99MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<06:04, 2.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<06:53, 1.87MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:20<05:24, 2.38MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:20<03:57, 3.24MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<08:10, 1.57MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<07:01, 1.82MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:22<05:14, 2.44MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<06:38, 1.92MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<05:56, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:24<04:25, 2.87MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:07, 2.07MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:35, 2.27MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:10, 3.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:55, 2.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:25, 2.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:06, 3.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:51, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:20, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:03, 3.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:46, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:18, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:01, 3.08MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:04, 2.04MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:34<06:46, 1.83MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:17, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<03:52, 3.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:28, 1.45MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:11, 1.71MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:17, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:34, 1.86MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<07:06, 1.73MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:31, 2.22MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<03:59, 3.05MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<13:50, 880kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<10:57, 1.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<07:57, 1.53MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:24, 1.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:23, 1.44MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:29, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:28, 1.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:48, 2.08MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<04:22, 2.75MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:50, 2.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:18, 2.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:00, 2.98MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:37, 2.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:09, 2.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<03:54, 3.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:31, 2.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:17, 1.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:00, 2.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:23, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<04:59, 2.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<03:44, 3.14MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:22, 2.18MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<04:55, 2.37MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:44, 3.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:22, 2.17MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<04:56, 2.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:45, 3.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:21, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:06, 1.89MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:46, 2.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<03:31, 3.27MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:45, 1.70MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:54, 1.95MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:25, 2.60MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:45, 1.99MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:29, 1.76MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:08, 2.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:43, 3.06MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<43:34, 261kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<31:29, 361kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<22:17, 509kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<15:41, 720kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<38:18, 295kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<29:10, 387kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<20:58, 538kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<16:26, 683kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<12:39, 887kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<09:07, 1.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<08:59, 1.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<08:34, 1.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<06:32, 1.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:41, 2.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<2:10:55, 84.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<1:32:44, 120kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<1:05:03, 170kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<47:58, 230kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<35:48, 308kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<25:33, 431kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<17:55, 612kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<29:32, 371kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<21:48, 502kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<15:30, 704kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<13:22, 814kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<10:17, 1.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<07:26, 1.46MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<05:20, 2.03MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<47:19, 229kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<35:19, 306kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<25:10, 429kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<17:41, 608kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<20:12, 532kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<15:15, 704kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<10:56, 980kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<10:07, 1.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:10, 1.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<05:57, 1.79MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<06:40, 1.59MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:45, 1.84MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<04:14, 2.50MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:06, 3.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<2:07:40, 82.6kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<1:31:24, 115kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<1:04:23, 164kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<45:04, 233kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<36:21, 288kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<26:31, 395kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<18:47, 556kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<15:33, 669kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<11:57, 870kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<08:37, 1.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<08:26, 1.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<08:00, 1.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<06:07, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:56, 1.73MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<05:14, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<03:55, 2.61MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:06, 2.00MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:40, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<04:29, 2.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<03:15, 3.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<1:14:58, 135kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<53:30, 189kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<37:36, 269kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<28:35, 352kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<20:47, 484kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<23:45, 423kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<16:37, 602kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<35:27, 282kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<25:50, 387kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<18:17, 545kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<14:57, 663kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<12:24, 800kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<09:10, 1.08MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:59, 1.23MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<06:36, 1.49MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:50, 2.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<05:40, 1.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<04:58, 1.97MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<03:42, 2.62MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<04:53, 1.98MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<05:25, 1.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<04:17, 2.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:33, 2.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<04:12, 2.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:11, 3.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:27, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:05, 1.88MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:01, 2.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:55<02:53, 3.28MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<57:45, 165kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<41:23, 230kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<29:06, 326kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<22:31, 419kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<17:41, 534kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<12:50, 734kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<10:28, 895kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<08:08, 1.15MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<05:53, 1.59MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<04:14, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<21:05, 441kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<15:42, 592kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<11:11, 828kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<10:00, 923kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<07:46, 1.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:38, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<04:03, 2.26MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<37:29, 245kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<28:06, 326kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<20:03, 456kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<14:06, 646kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<14:34, 625kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<11:08, 816kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<07:58, 1.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<07:41, 1.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<06:05, 1.48MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:47, 1.88MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<03:25, 2.62MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<18:52, 475kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<14:08, 633kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<10:06, 884kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<09:08, 973kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<08:15, 1.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<06:12, 1.43MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<04:33, 1.94MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<05:14, 1.69MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:34, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<03:24, 2.57MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:26, 1.97MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:53, 1.79MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:53, 2.25MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:18<02:49, 3.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<1:03:52, 136kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<45:33, 191kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<32:00, 270kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<24:19, 354kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<18:50, 457kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<13:36, 632kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<10:51, 788kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<08:20, 1.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<06:10, 1.38MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<04:23, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<22:04, 384kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<16:20, 519kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<11:35, 729kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<10:03, 837kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<07:53, 1.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<05:41, 1.47MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<05:57, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<05:48, 1.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:29, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:27, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:00, 2.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<03:00, 2.74MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:01, 2.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:35, 1.79MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:33, 2.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<02:38, 3.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:16, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:49, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<02:52, 2.82MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:53, 2.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:22, 1.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:28, 2.32MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<02:30, 3.18MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<07:11, 1.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:52, 1.36MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<04:18, 1.85MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:50, 1.64MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<05:01, 1.58MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<03:51, 2.05MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<02:49, 2.80MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<05:05, 1.55MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<04:21, 1.80MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<03:14, 2.41MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<04:05, 1.91MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:26, 1.75MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:27, 2.25MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:29, 3.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<08:56, 865kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<07:03, 1.09MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:07, 1.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<05:20, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:31, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:20, 2.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<04:07, 1.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:39, 2.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<02:43, 2.78MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:41, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:09, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:14, 2.32MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<02:19, 3.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<14:32, 513kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<10:57, 680kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<07:50, 947kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<07:11, 1.03MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:46, 1.28MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:11, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:40, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<04:00, 1.83MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<02:58, 2.45MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:47, 1.91MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:09, 1.75MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:16, 2.21MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<03:26, 2.09MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:08, 2.29MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:22, 3.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:19, 2.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:46, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:00, 2.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:10, 3.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<51:49, 136kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<36:58, 190kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<25:57, 270kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<19:43, 354kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<14:29, 481kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<10:17, 675kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<08:47, 785kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<07:33, 913kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<05:35, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<03:59, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<05:50, 1.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<04:47, 1.43MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<03:29, 1.95MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:01, 1.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:11, 1.61MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:16, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:15<02:20, 2.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<57:02, 117kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<40:34, 165kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<28:27, 234kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<21:22, 310kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<16:19, 406kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<11:44, 564kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<09:12, 712kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<07:07, 919kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<05:06, 1.28MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<05:04, 1.28MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<04:52, 1.33MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:41, 1.75MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:23<02:38, 2.43MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<07:07, 902kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<05:37, 1.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<04:04, 1.57MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:19, 1.47MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:19, 1.47MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:18, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<02:22, 2.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<06:24, 981kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<05:07, 1.23MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<03:43, 1.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:03, 1.53MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:28, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:34, 2.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:13, 1.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:30, 1.75MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:43, 2.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<01:58, 3.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<05:09, 1.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<04:13, 1.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<03:06, 1.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<03:34, 1.68MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:43, 1.61MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:51, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:37<02:04, 2.87MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<04:07, 1.44MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:29, 1.70MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<02:35, 2.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:10, 1.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:25, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:38, 2.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<01:54, 3.04MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<04:59, 1.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:05, 1.42MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:59, 1.93MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:25, 1.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:58, 1.92MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:13, 2.57MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:53, 1.96MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:10, 1.78MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:27, 2.29MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<01:47, 3.14MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:27, 1.26MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:42, 1.51MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:43, 2.05MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:11, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<03:25, 1.61MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:38, 2.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<01:55, 2.86MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:28, 1.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:01, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:14, 2.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:48, 1.92MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<02:30, 2.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<01:53, 2.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:33, 2.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:54, 1.83MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:17, 2.31MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:26, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:15, 2.33MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<01:41, 3.08MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:23, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:12, 2.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<01:40, 3.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:22, 2.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:43, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:09, 2.36MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:02<01:33, 3.25MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<4:52:20, 17.3kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<3:24:53, 24.6kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<2:22:48, 35.2kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<1:40:25, 49.6kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<1:10:43, 70.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<49:22, 100kB/s]   .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<34:22, 143kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<46:30, 106kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<33:00, 149kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<23:05, 211kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<17:12, 282kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<12:26, 389kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<08:46, 549kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<06:09, 776kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<23:31, 203kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<16:55, 282kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<11:53, 399kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<09:22, 503kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<07:30, 627kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<05:27, 860kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<03:50, 1.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<06:26, 721kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<04:58, 933kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<03:33, 1.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:32, 1.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:24, 1.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:36, 1.74MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:32, 1.77MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:14, 2.01MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:40, 2.67MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:11, 2.02MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:59, 2.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:29, 2.94MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:04, 2.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:20, 1.86MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:49, 2.38MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<01:19, 3.27MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<05:22, 800kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<04:11, 1.02MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:01, 1.41MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<03:05, 1.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<03:02, 1.39MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:19, 1.81MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:40, 2.49MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<03:12, 1.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:40, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<01:57, 2.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<02:19, 1.77MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:58, 2.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:28, 2.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:58, 2.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:12, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:44, 2.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<01:51, 2.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<01:42, 2.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:16, 3.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:47, 2.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:04, 1.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:37, 2.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:10, 3.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<03:04, 1.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:32, 1.50MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:52, 2.03MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:10, 1.72MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:17, 1.64MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:45, 2.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:15, 2.93MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<03:51, 954kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<03:04, 1.20MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<02:13, 1.64MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:23, 1.51MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:24, 1.50MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:51, 1.93MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:51, 1.91MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:39, 2.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<01:13, 2.85MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:40, 2.08MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:31, 2.28MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:08, 3.00MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:36, 2.13MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:28, 2.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:05, 3.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:33, 2.14MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:25, 2.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:04, 3.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:31, 2.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:43, 1.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:20, 2.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<00:58, 3.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:42, 1.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:13, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:37, 1.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:51, 1.69MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:36, 1.94MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:12, 2.58MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:33, 1.97MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:21, 2.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:01, 2.98MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:24, 2.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:36, 1.86MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:14, 2.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<00:54, 3.25MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:52, 1.56MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:37, 1.80MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:12, 2.41MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:29, 1.92MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:37, 1.75MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:16, 2.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<00:53, 3.11MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<07:28, 373kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<05:30, 505kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<03:53, 710kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<03:19, 819kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:53, 943kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:08, 1.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:54, 1.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:36, 1.65MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:10, 2.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:24, 1.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:15, 2.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<00:55, 2.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:14, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:07, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:50, 2.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:09, 2.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:03, 2.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<00:47, 3.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:06, 2.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:11, 1.98MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<00:57, 2.47MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<00:42, 3.30MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:11, 1.95MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:03, 2.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:47, 2.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:04, 2.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:12, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:57, 2.34MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:25<00:40, 3.22MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<18:25, 118kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<13:04, 166kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<09:05, 236kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<06:45, 312kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<05:08, 408kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<03:40, 568kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:29<02:32, 805kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<06:12, 328kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<04:32, 446kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<03:11, 627kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:39, 740kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:15, 870kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:39, 1.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:09, 1.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:00, 949kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:35, 1.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<01:08, 1.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:12, 1.51MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:01, 1.77MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:45, 2.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:56, 1.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:01, 1.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:47, 2.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:33, 3.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:33, 1.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:15, 1.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:54, 1.83MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:00, 1.62MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:01, 1.57MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:47, 2.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:47, 1.96MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:42, 2.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:31, 2.88MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:42, 2.09MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:38, 2.30MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:28, 3.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:39, 2.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:36, 2.33MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:27, 3.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:37, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:34, 2.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:25, 3.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:35, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:32, 2.34MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:24, 3.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:33, 2.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:38, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:30, 2.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:31, 2.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:28, 2.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:21, 3.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:29, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:33, 1.90MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:26, 2.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:18, 3.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<02:47, 360kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<02:02, 489kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<01:25, 686kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:10, 798kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:00, 923kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:44, 1.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:38, 1.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:31, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:22, 2.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:26, 1.82MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:28, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:21, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:21, 2.06MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:19, 2.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:14, 2.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:18, 2.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:21, 1.88MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:16, 2.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:11, 3.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:26, 1.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:22, 1.60MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:15, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:17, 1.79MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:15, 2.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:11, 2.68MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:13, 2.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<00:07, 3.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<01:30, 260kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<01:04, 358kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:42, 505kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:31, 617kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:23, 808kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:15, 1.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:13, 1.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:10, 1.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:06, 1.94MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.67MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.61MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:02, 2.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:07, 951kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:05, 1.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:03, 1.64MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 1.51MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.77MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 2.37MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 856/400000 [00:00<00:46, 8559.91it/s]  0%|          | 1716/400000 [00:00<00:46, 8571.70it/s]  1%|          | 2580/400000 [00:00<00:46, 8591.35it/s]  1%|          | 3442/400000 [00:00<00:46, 8597.50it/s]  1%|          | 4295/400000 [00:00<00:46, 8574.91it/s]  1%|▏         | 5142/400000 [00:00<00:46, 8542.92it/s]  1%|▏         | 5979/400000 [00:00<00:46, 8489.90it/s]  2%|▏         | 6835/400000 [00:00<00:46, 8510.11it/s]  2%|▏         | 7677/400000 [00:00<00:46, 8480.99it/s]  2%|▏         | 8491/400000 [00:01<00:46, 8363.29it/s]  2%|▏         | 9351/400000 [00:01<00:46, 8430.56it/s]  3%|▎         | 10188/400000 [00:01<00:46, 8405.44it/s]  3%|▎         | 11022/400000 [00:01<00:46, 8385.33it/s]  3%|▎         | 11891/400000 [00:01<00:45, 8474.35it/s]  3%|▎         | 12750/400000 [00:01<00:45, 8508.21it/s]  3%|▎         | 13630/400000 [00:01<00:44, 8591.22it/s]  4%|▎         | 14497/400000 [00:01<00:44, 8612.58it/s]  4%|▍         | 15369/400000 [00:01<00:44, 8642.77it/s]  4%|▍         | 16242/400000 [00:01<00:44, 8667.27it/s]  4%|▍         | 17108/400000 [00:02<00:44, 8640.36it/s]  4%|▍         | 17989/400000 [00:02<00:43, 8688.95it/s]  5%|▍         | 18858/400000 [00:02<00:44, 8556.49it/s]  5%|▍         | 19714/400000 [00:02<00:46, 8263.96it/s]  5%|▌         | 20558/400000 [00:02<00:45, 8313.77it/s]  5%|▌         | 21424/400000 [00:02<00:44, 8413.42it/s]  6%|▌         | 22277/400000 [00:02<00:44, 8446.27it/s]  6%|▌         | 23123/400000 [00:02<00:44, 8419.45it/s]  6%|▌         | 23978/400000 [00:02<00:44, 8455.97it/s]  6%|▌         | 24825/400000 [00:02<00:44, 8367.13it/s]  6%|▋         | 25679/400000 [00:03<00:44, 8417.06it/s]  7%|▋         | 26543/400000 [00:03<00:44, 8480.85it/s]  7%|▋         | 27415/400000 [00:03<00:43, 8550.09it/s]  7%|▋         | 28290/400000 [00:03<00:43, 8606.30it/s]  7%|▋         | 29162/400000 [00:03<00:42, 8637.56it/s]  8%|▊         | 30027/400000 [00:03<00:42, 8620.11it/s]  8%|▊         | 30890/400000 [00:03<00:42, 8604.36it/s]  8%|▊         | 31751/400000 [00:03<00:43, 8553.55it/s]  8%|▊         | 32607/400000 [00:03<00:44, 8283.06it/s]  8%|▊         | 33447/400000 [00:03<00:44, 8316.04it/s]  9%|▊         | 34281/400000 [00:04<00:44, 8214.75it/s]  9%|▉         | 35129/400000 [00:04<00:44, 8291.76it/s]  9%|▉         | 35960/400000 [00:04<00:44, 8125.79it/s]  9%|▉         | 36802/400000 [00:04<00:44, 8210.89it/s]  9%|▉         | 37673/400000 [00:04<00:43, 8351.85it/s] 10%|▉         | 38517/400000 [00:04<00:43, 8375.35it/s] 10%|▉         | 39378/400000 [00:04<00:42, 8444.31it/s] 10%|█         | 40248/400000 [00:04<00:42, 8517.88it/s] 10%|█         | 41101/400000 [00:04<00:42, 8467.77it/s] 10%|█         | 41949/400000 [00:04<00:42, 8339.29it/s] 11%|█         | 42814/400000 [00:05<00:42, 8427.88it/s] 11%|█         | 43677/400000 [00:05<00:41, 8487.00it/s] 11%|█         | 44527/400000 [00:05<00:42, 8350.71it/s] 11%|█▏        | 45364/400000 [00:05<00:43, 8143.74it/s] 12%|█▏        | 46181/400000 [00:05<00:44, 8026.40it/s] 12%|█▏        | 47045/400000 [00:05<00:43, 8198.81it/s] 12%|█▏        | 47911/400000 [00:05<00:42, 8331.04it/s] 12%|█▏        | 48775/400000 [00:05<00:41, 8419.99it/s] 12%|█▏        | 49619/400000 [00:05<00:42, 8323.30it/s] 13%|█▎        | 50486/400000 [00:05<00:41, 8423.99it/s] 13%|█▎        | 51339/400000 [00:06<00:41, 8453.39it/s] 13%|█▎        | 52201/400000 [00:06<00:40, 8502.55it/s] 13%|█▎        | 53070/400000 [00:06<00:40, 8557.82it/s] 13%|█▎        | 53942/400000 [00:06<00:40, 8602.99it/s] 14%|█▎        | 54818/400000 [00:06<00:39, 8646.72it/s] 14%|█▍        | 55688/400000 [00:06<00:39, 8661.48it/s] 14%|█▍        | 56556/400000 [00:06<00:39, 8665.93it/s] 14%|█▍        | 57427/400000 [00:06<00:39, 8678.68it/s] 15%|█▍        | 58296/400000 [00:06<00:39, 8669.84it/s] 15%|█▍        | 59164/400000 [00:06<00:40, 8482.33it/s] 15%|█▌        | 60014/400000 [00:07<00:40, 8349.23it/s] 15%|█▌        | 60861/400000 [00:07<00:40, 8383.72it/s] 15%|█▌        | 61741/400000 [00:07<00:39, 8502.84it/s] 16%|█▌        | 62620/400000 [00:07<00:39, 8586.50it/s] 16%|█▌        | 63502/400000 [00:07<00:38, 8654.22it/s] 16%|█▌        | 64373/400000 [00:07<00:38, 8670.72it/s] 16%|█▋        | 65241/400000 [00:07<00:38, 8593.85it/s] 17%|█▋        | 66120/400000 [00:07<00:38, 8650.58it/s] 17%|█▋        | 67000/400000 [00:07<00:38, 8692.89it/s] 17%|█▋        | 67877/400000 [00:08<00:38, 8714.68it/s] 17%|█▋        | 68749/400000 [00:08<00:38, 8695.39it/s] 17%|█▋        | 69629/400000 [00:08<00:37, 8726.46it/s] 18%|█▊        | 70508/400000 [00:08<00:37, 8745.20it/s] 18%|█▊        | 71387/400000 [00:08<00:37, 8757.12it/s] 18%|█▊        | 72266/400000 [00:08<00:37, 8765.91it/s] 18%|█▊        | 73143/400000 [00:08<00:37, 8746.66it/s] 19%|█▊        | 74018/400000 [00:08<00:38, 8463.57it/s] 19%|█▊        | 74867/400000 [00:08<00:38, 8356.07it/s] 19%|█▉        | 75714/400000 [00:08<00:38, 8387.91it/s] 19%|█▉        | 76590/400000 [00:09<00:38, 8493.77it/s] 19%|█▉        | 77462/400000 [00:09<00:37, 8558.29it/s] 20%|█▉        | 78343/400000 [00:09<00:37, 8630.54it/s] 20%|█▉        | 79219/400000 [00:09<00:37, 8666.48it/s] 20%|██        | 80091/400000 [00:09<00:36, 8681.38it/s] 20%|██        | 80962/400000 [00:09<00:36, 8687.15it/s] 20%|██        | 81832/400000 [00:09<00:36, 8663.85it/s] 21%|██        | 82699/400000 [00:09<00:36, 8636.60it/s] 21%|██        | 83573/400000 [00:09<00:36, 8666.99it/s] 21%|██        | 84446/400000 [00:09<00:36, 8683.75it/s] 21%|██▏       | 85315/400000 [00:10<00:36, 8683.79it/s] 22%|██▏       | 86189/400000 [00:10<00:36, 8698.92it/s] 22%|██▏       | 87064/400000 [00:10<00:35, 8711.62it/s] 22%|██▏       | 87944/400000 [00:10<00:35, 8737.86it/s] 22%|██▏       | 88818/400000 [00:10<00:35, 8727.35it/s] 22%|██▏       | 89691/400000 [00:10<00:35, 8723.97it/s] 23%|██▎       | 90564/400000 [00:10<00:35, 8718.56it/s] 23%|██▎       | 91440/400000 [00:10<00:35, 8729.33it/s] 23%|██▎       | 92313/400000 [00:10<00:35, 8704.29it/s] 23%|██▎       | 93184/400000 [00:10<00:35, 8699.42it/s] 24%|██▎       | 94054/400000 [00:11<00:35, 8691.76it/s] 24%|██▎       | 94925/400000 [00:11<00:35, 8694.41it/s] 24%|██▍       | 95795/400000 [00:11<00:36, 8427.13it/s] 24%|██▍       | 96666/400000 [00:11<00:35, 8507.51it/s] 24%|██▍       | 97532/400000 [00:11<00:35, 8552.12it/s] 25%|██▍       | 98405/400000 [00:11<00:35, 8602.72it/s] 25%|██▍       | 99277/400000 [00:11<00:34, 8635.06it/s] 25%|██▌       | 100146/400000 [00:11<00:34, 8650.82it/s] 25%|██▌       | 101020/400000 [00:11<00:34, 8677.24it/s] 25%|██▌       | 101897/400000 [00:11<00:34, 8703.28it/s] 26%|██▌       | 102768/400000 [00:12<00:34, 8587.21it/s] 26%|██▌       | 103628/400000 [00:12<00:35, 8287.47it/s] 26%|██▌       | 104460/400000 [00:12<00:36, 8146.28it/s] 26%|██▋       | 105337/400000 [00:12<00:35, 8321.38it/s] 27%|██▋       | 106188/400000 [00:12<00:35, 8374.16it/s] 27%|██▋       | 107028/400000 [00:12<00:35, 8260.46it/s] 27%|██▋       | 107856/400000 [00:12<00:35, 8139.61it/s] 27%|██▋       | 108697/400000 [00:12<00:35, 8217.46it/s] 27%|██▋       | 109556/400000 [00:12<00:34, 8323.81it/s] 28%|██▊       | 110390/400000 [00:12<00:35, 8220.51it/s] 28%|██▊       | 111214/400000 [00:13<00:35, 8169.73it/s] 28%|██▊       | 112032/400000 [00:13<00:35, 8109.71it/s] 28%|██▊       | 112851/400000 [00:13<00:35, 8131.03it/s] 28%|██▊       | 113665/400000 [00:13<00:35, 8104.10it/s] 29%|██▊       | 114476/400000 [00:13<00:35, 8043.54it/s] 29%|██▉       | 115283/400000 [00:13<00:35, 8049.84it/s] 29%|██▉       | 116089/400000 [00:13<00:36, 7734.18it/s] 29%|██▉       | 116961/400000 [00:13<00:35, 8004.03it/s] 29%|██▉       | 117832/400000 [00:13<00:34, 8201.13it/s] 30%|██▉       | 118696/400000 [00:13<00:33, 8325.84it/s] 30%|██▉       | 119567/400000 [00:14<00:33, 8436.97it/s] 30%|███       | 120433/400000 [00:14<00:32, 8500.54it/s] 30%|███       | 121304/400000 [00:14<00:32, 8560.10it/s] 31%|███       | 122176/400000 [00:14<00:32, 8605.16it/s] 31%|███       | 123054/400000 [00:14<00:31, 8655.11it/s] 31%|███       | 123928/400000 [00:14<00:31, 8679.84it/s] 31%|███       | 124800/400000 [00:14<00:31, 8689.12it/s] 31%|███▏      | 125671/400000 [00:14<00:31, 8694.37it/s] 32%|███▏      | 126543/400000 [00:14<00:31, 8700.49it/s] 32%|███▏      | 127416/400000 [00:14<00:31, 8707.00it/s] 32%|███▏      | 128287/400000 [00:15<00:31, 8527.92it/s] 32%|███▏      | 129141/400000 [00:15<00:32, 8330.61it/s] 32%|███▏      | 129976/400000 [00:15<00:32, 8218.24it/s] 33%|███▎      | 130800/400000 [00:15<00:33, 8144.66it/s] 33%|███▎      | 131616/400000 [00:15<00:33, 8039.70it/s] 33%|███▎      | 132422/400000 [00:15<00:33, 7963.64it/s] 33%|███▎      | 133259/400000 [00:15<00:33, 8080.66it/s] 34%|███▎      | 134133/400000 [00:15<00:32, 8266.58it/s] 34%|███▍      | 135008/400000 [00:15<00:31, 8405.64it/s] 34%|███▍      | 135880/400000 [00:16<00:31, 8494.96it/s] 34%|███▍      | 136756/400000 [00:16<00:30, 8572.30it/s] 34%|███▍      | 137633/400000 [00:16<00:30, 8629.16it/s] 35%|███▍      | 138511/400000 [00:16<00:30, 8671.25it/s] 35%|███▍      | 139394/400000 [00:16<00:29, 8718.10it/s] 35%|███▌      | 140267/400000 [00:16<00:29, 8706.31it/s] 35%|███▌      | 141151/400000 [00:16<00:29, 8745.03it/s] 36%|███▌      | 142026/400000 [00:16<00:29, 8737.36it/s] 36%|███▌      | 142900/400000 [00:16<00:29, 8679.31it/s] 36%|███▌      | 143780/400000 [00:16<00:29, 8712.51it/s] 36%|███▌      | 144662/400000 [00:17<00:29, 8742.45it/s] 36%|███▋      | 145540/400000 [00:17<00:29, 8753.05it/s] 37%|███▋      | 146421/400000 [00:17<00:28, 8766.55it/s] 37%|███▋      | 147298/400000 [00:17<00:29, 8463.27it/s] 37%|███▋      | 148147/400000 [00:17<00:30, 8329.13it/s] 37%|███▋      | 148983/400000 [00:17<00:30, 8252.95it/s] 37%|███▋      | 149810/400000 [00:17<00:30, 8237.21it/s] 38%|███▊      | 150678/400000 [00:17<00:29, 8364.63it/s] 38%|███▊      | 151553/400000 [00:17<00:29, 8476.14it/s] 38%|███▊      | 152430/400000 [00:17<00:28, 8561.36it/s] 38%|███▊      | 153302/400000 [00:18<00:28, 8607.83it/s] 39%|███▊      | 154183/400000 [00:18<00:28, 8666.71it/s] 39%|███▉      | 155051/400000 [00:18<00:28, 8670.68it/s] 39%|███▉      | 155919/400000 [00:18<00:28, 8564.40it/s] 39%|███▉      | 156777/400000 [00:18<00:29, 8348.79it/s] 39%|███▉      | 157652/400000 [00:18<00:28, 8463.93it/s] 40%|███▉      | 158531/400000 [00:18<00:28, 8556.95it/s] 40%|███▉      | 159389/400000 [00:18<00:28, 8493.81it/s] 40%|████      | 160240/400000 [00:18<00:28, 8394.77it/s] 40%|████      | 161112/400000 [00:18<00:28, 8487.78it/s] 40%|████      | 161975/400000 [00:19<00:27, 8527.42it/s] 41%|████      | 162853/400000 [00:19<00:27, 8601.42it/s] 41%|████      | 163714/400000 [00:19<00:27, 8491.84it/s] 41%|████      | 164588/400000 [00:19<00:27, 8562.13it/s] 41%|████▏     | 165470/400000 [00:19<00:27, 8636.07it/s] 42%|████▏     | 166349/400000 [00:19<00:26, 8681.09it/s] 42%|████▏     | 167229/400000 [00:19<00:26, 8716.40it/s] 42%|████▏     | 168104/400000 [00:19<00:26, 8725.94it/s] 42%|████▏     | 168980/400000 [00:19<00:26, 8735.56it/s] 42%|████▏     | 169860/400000 [00:19<00:26, 8754.49it/s] 43%|████▎     | 170736/400000 [00:20<00:26, 8591.74it/s] 43%|████▎     | 171596/400000 [00:20<00:27, 8386.55it/s] 43%|████▎     | 172437/400000 [00:20<00:27, 8260.91it/s] 43%|████▎     | 173265/400000 [00:20<00:27, 8172.03it/s] 44%|████▎     | 174106/400000 [00:20<00:27, 8241.25it/s] 44%|████▎     | 174932/400000 [00:20<00:27, 8056.45it/s] 44%|████▍     | 175806/400000 [00:20<00:27, 8248.20it/s] 44%|████▍     | 176674/400000 [00:20<00:26, 8372.11it/s] 44%|████▍     | 177549/400000 [00:20<00:26, 8481.57it/s] 45%|████▍     | 178425/400000 [00:21<00:25, 8561.70it/s] 45%|████▍     | 179306/400000 [00:21<00:25, 8632.27it/s] 45%|████▌     | 180186/400000 [00:21<00:25, 8680.85it/s] 45%|████▌     | 181055/400000 [00:21<00:25, 8673.58it/s] 45%|████▌     | 181929/400000 [00:21<00:25, 8691.73it/s] 46%|████▌     | 182810/400000 [00:21<00:24, 8724.90it/s] 46%|████▌     | 183691/400000 [00:21<00:24, 8747.71it/s] 46%|████▌     | 184569/400000 [00:21<00:24, 8754.88it/s] 46%|████▋     | 185448/400000 [00:21<00:24, 8765.22it/s] 47%|████▋     | 186325/400000 [00:21<00:24, 8744.36it/s] 47%|████▋     | 187203/400000 [00:22<00:24, 8753.90it/s] 47%|████▋     | 188081/400000 [00:22<00:24, 8760.49it/s] 47%|████▋     | 188958/400000 [00:22<00:24, 8748.00it/s] 47%|████▋     | 189837/400000 [00:22<00:23, 8757.69it/s] 48%|████▊     | 190713/400000 [00:22<00:23, 8740.68it/s] 48%|████▊     | 191593/400000 [00:22<00:23, 8755.58it/s] 48%|████▊     | 192469/400000 [00:22<00:23, 8729.29it/s] 48%|████▊     | 193348/400000 [00:22<00:23, 8747.37it/s] 49%|████▊     | 194225/400000 [00:22<00:23, 8751.88it/s] 49%|████▉     | 195101/400000 [00:22<00:23, 8736.99it/s] 49%|████▉     | 195981/400000 [00:23<00:23, 8754.55it/s] 49%|████▉     | 196865/400000 [00:23<00:23, 8778.04it/s] 49%|████▉     | 197743/400000 [00:23<00:23, 8776.70it/s] 50%|████▉     | 198621/400000 [00:23<00:23, 8544.11it/s] 50%|████▉     | 199477/400000 [00:23<00:23, 8357.27it/s] 50%|█████     | 200315/400000 [00:23<00:24, 8314.56it/s] 50%|█████     | 201196/400000 [00:23<00:23, 8455.20it/s] 51%|█████     | 202076/400000 [00:23<00:23, 8553.98it/s] 51%|█████     | 202953/400000 [00:23<00:22, 8615.86it/s] 51%|█████     | 203816/400000 [00:23<00:23, 8439.77it/s] 51%|█████     | 204675/400000 [00:24<00:23, 8481.54it/s] 51%|█████▏    | 205547/400000 [00:24<00:22, 8550.16it/s] 52%|█████▏    | 206426/400000 [00:24<00:22, 8619.30it/s] 52%|█████▏    | 207304/400000 [00:24<00:22, 8666.63it/s] 52%|█████▏    | 208172/400000 [00:24<00:22, 8485.65it/s] 52%|█████▏    | 209022/400000 [00:24<00:22, 8457.10it/s] 52%|█████▏    | 209901/400000 [00:24<00:22, 8553.82it/s] 53%|█████▎    | 210777/400000 [00:24<00:21, 8612.14it/s] 53%|█████▎    | 211655/400000 [00:24<00:21, 8660.71it/s] 53%|█████▎    | 212522/400000 [00:24<00:21, 8582.74it/s] 53%|█████▎    | 213381/400000 [00:25<00:22, 8455.69it/s] 54%|█████▎    | 214228/400000 [00:25<00:22, 8169.72it/s] 54%|█████▍    | 215048/400000 [00:25<00:22, 8090.12it/s] 54%|█████▍    | 215860/400000 [00:25<00:22, 8068.85it/s] 54%|█████▍    | 216735/400000 [00:25<00:22, 8261.02it/s] 54%|█████▍    | 217609/400000 [00:25<00:21, 8398.92it/s] 55%|█████▍    | 218486/400000 [00:25<00:21, 8504.75it/s] 55%|█████▍    | 219365/400000 [00:25<00:21, 8586.13it/s] 55%|█████▌    | 220245/400000 [00:25<00:20, 8648.45it/s] 55%|█████▌    | 221120/400000 [00:25<00:20, 8676.73it/s] 55%|█████▌    | 221989/400000 [00:26<00:20, 8660.73it/s] 56%|█████▌    | 222860/400000 [00:26<00:20, 8674.03it/s] 56%|█████▌    | 223735/400000 [00:26<00:20, 8696.57it/s] 56%|█████▌    | 224612/400000 [00:26<00:20, 8717.83it/s] 56%|█████▋    | 225484/400000 [00:26<00:20, 8707.63it/s] 57%|█████▋    | 226355/400000 [00:26<00:20, 8526.74it/s] 57%|█████▋    | 227209/400000 [00:26<00:20, 8357.36it/s] 57%|█████▋    | 228047/400000 [00:26<00:20, 8270.82it/s] 57%|█████▋    | 228876/400000 [00:26<00:20, 8193.22it/s] 57%|█████▋    | 229697/400000 [00:26<00:20, 8191.22it/s] 58%|█████▊    | 230517/400000 [00:27<00:21, 8064.65it/s] 58%|█████▊    | 231325/400000 [00:27<00:21, 8017.42it/s] 58%|█████▊    | 232155/400000 [00:27<00:20, 8099.30it/s] 58%|█████▊    | 233016/400000 [00:27<00:20, 8244.47it/s] 58%|█████▊    | 233894/400000 [00:27<00:19, 8396.80it/s] 59%|█████▊    | 234777/400000 [00:27<00:19, 8519.67it/s] 59%|█████▉    | 235654/400000 [00:27<00:19, 8592.28it/s] 59%|█████▉    | 236525/400000 [00:27<00:18, 8627.09it/s] 59%|█████▉    | 237398/400000 [00:27<00:18, 8655.98it/s] 60%|█████▉    | 238265/400000 [00:28<00:18, 8605.77it/s] 60%|█████▉    | 239145/400000 [00:28<00:18, 8660.84it/s] 60%|██████    | 240023/400000 [00:28<00:18, 8696.14it/s] 60%|██████    | 240893/400000 [00:28<00:18, 8658.30it/s] 60%|██████    | 241769/400000 [00:28<00:18, 8687.25it/s] 61%|██████    | 242647/400000 [00:28<00:18, 8712.05it/s] 61%|██████    | 243527/400000 [00:28<00:17, 8736.86it/s] 61%|██████    | 244405/400000 [00:28<00:17, 8748.45it/s] 61%|██████▏   | 245285/400000 [00:28<00:17, 8761.85it/s] 62%|██████▏   | 246162/400000 [00:28<00:17, 8709.27it/s] 62%|██████▏   | 247034/400000 [00:29<00:17, 8711.98it/s] 62%|██████▏   | 247913/400000 [00:29<00:17, 8735.02it/s] 62%|██████▏   | 248795/400000 [00:29<00:17, 8758.97it/s] 62%|██████▏   | 249671/400000 [00:29<00:17, 8757.63it/s] 63%|██████▎   | 250547/400000 [00:29<00:17, 8758.07it/s] 63%|██████▎   | 251423/400000 [00:29<00:16, 8748.95it/s] 63%|██████▎   | 252298/400000 [00:29<00:16, 8748.93it/s] 63%|██████▎   | 253173/400000 [00:29<00:16, 8646.58it/s] 64%|██████▎   | 254038/400000 [00:29<00:17, 8387.44it/s] 64%|██████▎   | 254879/400000 [00:29<00:17, 8333.34it/s] 64%|██████▍   | 255744/400000 [00:30<00:17, 8425.17it/s] 64%|██████▍   | 256618/400000 [00:30<00:16, 8515.84it/s] 64%|██████▍   | 257497/400000 [00:30<00:16, 8593.75it/s] 65%|██████▍   | 258370/400000 [00:30<00:16, 8632.36it/s] 65%|██████▍   | 259239/400000 [00:30<00:16, 8649.44it/s] 65%|██████▌   | 260105/400000 [00:30<00:16, 8555.92it/s] 65%|██████▌   | 260962/400000 [00:30<00:16, 8324.40it/s] 65%|██████▌   | 261797/400000 [00:30<00:16, 8236.54it/s] 66%|██████▌   | 262663/400000 [00:30<00:16, 8357.77it/s] 66%|██████▌   | 263531/400000 [00:30<00:16, 8450.49it/s] 66%|██████▌   | 264378/400000 [00:31<00:16, 8246.86it/s] 66%|██████▋   | 265205/400000 [00:31<00:16, 7992.78it/s] 67%|██████▋   | 266008/400000 [00:31<00:16, 7953.62it/s] 67%|██████▋   | 266845/400000 [00:31<00:16, 8072.83it/s] 67%|██████▋   | 267721/400000 [00:31<00:16, 8266.45it/s] 67%|██████▋   | 268589/400000 [00:31<00:15, 8384.95it/s] 67%|██████▋   | 269468/400000 [00:31<00:15, 8502.33it/s] 68%|██████▊   | 270343/400000 [00:31<00:15, 8574.60it/s] 68%|██████▊   | 271224/400000 [00:31<00:14, 8641.32it/s] 68%|██████▊   | 272103/400000 [00:31<00:14, 8685.29it/s] 68%|██████▊   | 272973/400000 [00:32<00:14, 8656.02it/s] 68%|██████▊   | 273852/400000 [00:32<00:14, 8693.48it/s] 69%|██████▊   | 274734/400000 [00:32<00:14, 8729.39it/s] 69%|██████▉   | 275613/400000 [00:32<00:14, 8746.99it/s] 69%|██████▉   | 276490/400000 [00:32<00:14, 8752.59it/s] 69%|██████▉   | 277366/400000 [00:32<00:14, 8724.38it/s] 70%|██████▉   | 278239/400000 [00:32<00:13, 8725.81it/s] 70%|██████▉   | 279120/400000 [00:32<00:13, 8748.39it/s] 70%|██████▉   | 279995/400000 [00:32<00:13, 8734.60it/s] 70%|███████   | 280879/400000 [00:32<00:13, 8763.83it/s] 70%|███████   | 281756/400000 [00:33<00:13, 8695.96it/s] 71%|███████   | 282626/400000 [00:33<00:13, 8437.84it/s] 71%|███████   | 283472/400000 [00:33<00:14, 8195.47it/s] 71%|███████   | 284295/400000 [00:33<00:14, 8183.61it/s] 71%|███████▏  | 285163/400000 [00:33<00:13, 8325.30it/s] 71%|███████▏  | 285998/400000 [00:33<00:13, 8315.51it/s] 72%|███████▏  | 286831/400000 [00:33<00:14, 8049.58it/s] 72%|███████▏  | 287704/400000 [00:33<00:13, 8240.93it/s] 72%|███████▏  | 288575/400000 [00:33<00:13, 8374.52it/s] 72%|███████▏  | 289416/400000 [00:34<00:13, 8220.87it/s] 73%|███████▎  | 290241/400000 [00:34<00:13, 8145.84it/s] 73%|███████▎  | 291112/400000 [00:34<00:13, 8305.93it/s] 73%|███████▎  | 291945/400000 [00:34<00:13, 8118.70it/s] 73%|███████▎  | 292760/400000 [00:34<00:13, 7981.12it/s] 73%|███████▎  | 293575/400000 [00:34<00:13, 8029.82it/s] 74%|███████▎  | 294442/400000 [00:34<00:12, 8210.40it/s] 74%|███████▍  | 295313/400000 [00:34<00:12, 8353.72it/s] 74%|███████▍  | 296189/400000 [00:34<00:12, 8471.31it/s] 74%|███████▍  | 297067/400000 [00:34<00:12, 8559.97it/s] 74%|███████▍  | 297941/400000 [00:35<00:11, 8612.22it/s] 75%|███████▍  | 298814/400000 [00:35<00:11, 8647.15it/s] 75%|███████▍  | 299685/400000 [00:35<00:11, 8664.85it/s] 75%|███████▌  | 300553/400000 [00:35<00:11, 8571.02it/s] 75%|███████▌  | 301430/400000 [00:35<00:11, 8628.52it/s] 76%|███████▌  | 302309/400000 [00:35<00:11, 8673.56it/s] 76%|███████▌  | 303181/400000 [00:35<00:11, 8684.93it/s] 76%|███████▌  | 304063/400000 [00:35<00:10, 8723.93it/s] 76%|███████▌  | 304936/400000 [00:35<00:10, 8719.88it/s] 76%|███████▋  | 305810/400000 [00:35<00:10, 8723.36it/s] 77%|███████▋  | 306683/400000 [00:36<00:11, 8314.83it/s] 77%|███████▋  | 307519/400000 [00:36<00:11, 8049.84it/s] 77%|███████▋  | 308329/400000 [00:36<00:11, 8005.09it/s] 77%|███████▋  | 309133/400000 [00:36<00:11, 7966.79it/s] 77%|███████▋  | 309933/400000 [00:36<00:11, 7970.24it/s] 78%|███████▊  | 310732/400000 [00:36<00:11, 7854.51it/s] 78%|███████▊  | 311596/400000 [00:36<00:10, 8074.13it/s] 78%|███████▊  | 312480/400000 [00:36<00:10, 8287.54it/s] 78%|███████▊  | 313358/400000 [00:36<00:10, 8429.12it/s] 79%|███████▊  | 314242/400000 [00:36<00:10, 8547.19it/s] 79%|███████▉  | 315118/400000 [00:37<00:09, 8609.00it/s] 79%|███████▉  | 315983/400000 [00:37<00:09, 8620.98it/s] 79%|███████▉  | 316862/400000 [00:37<00:09, 8669.88it/s] 79%|███████▉  | 317741/400000 [00:37<00:09, 8704.07it/s] 80%|███████▉  | 318617/400000 [00:37<00:09, 8719.39it/s] 80%|███████▉  | 319490/400000 [00:37<00:09, 8571.07it/s] 80%|████████  | 320349/400000 [00:37<00:09, 8232.30it/s] 80%|████████  | 321176/400000 [00:37<00:09, 8083.06it/s] 81%|████████  | 322037/400000 [00:37<00:09, 8233.64it/s] 81%|████████  | 322914/400000 [00:38<00:09, 8385.70it/s] 81%|████████  | 323789/400000 [00:38<00:08, 8489.56it/s] 81%|████████  | 324655/400000 [00:38<00:08, 8537.47it/s] 81%|████████▏ | 325532/400000 [00:38<00:08, 8603.31it/s] 82%|████████▏ | 326410/400000 [00:38<00:08, 8654.38it/s] 82%|████████▏ | 327287/400000 [00:38<00:08, 8687.91it/s] 82%|████████▏ | 328166/400000 [00:38<00:08, 8717.28it/s] 82%|████████▏ | 329039/400000 [00:38<00:08, 8702.20it/s] 82%|████████▏ | 329919/400000 [00:38<00:08, 8729.79it/s] 83%|████████▎ | 330795/400000 [00:38<00:07, 8736.01it/s] 83%|████████▎ | 331669/400000 [00:39<00:07, 8732.02it/s] 83%|████████▎ | 332548/400000 [00:39<00:07, 8747.27it/s] 83%|████████▎ | 333423/400000 [00:39<00:07, 8723.59it/s] 84%|████████▎ | 334301/400000 [00:39<00:07, 8737.75it/s] 84%|████████▍ | 335180/400000 [00:39<00:07, 8753.14it/s] 84%|████████▍ | 336056/400000 [00:39<00:07, 8701.10it/s] 84%|████████▍ | 336927/400000 [00:39<00:07, 8693.69it/s] 84%|████████▍ | 337797/400000 [00:39<00:07, 8695.40it/s] 85%|████████▍ | 338676/400000 [00:39<00:07, 8722.61it/s] 85%|████████▍ | 339556/400000 [00:39<00:06, 8743.21it/s] 85%|████████▌ | 340436/400000 [00:40<00:06, 8758.71it/s] 85%|████████▌ | 341312/400000 [00:40<00:06, 8733.10it/s] 86%|████████▌ | 342186/400000 [00:40<00:06, 8727.08it/s] 86%|████████▌ | 343061/400000 [00:40<00:06, 8732.48it/s] 86%|████████▌ | 343941/400000 [00:40<00:06, 8751.94it/s] 86%|████████▌ | 344820/400000 [00:40<00:06, 8760.89it/s] 86%|████████▋ | 345697/400000 [00:40<00:06, 8759.31it/s] 87%|████████▋ | 346573/400000 [00:40<00:06, 8605.06it/s] 87%|████████▋ | 347435/400000 [00:40<00:06, 8337.81it/s] 87%|████████▋ | 348309/400000 [00:40<00:06, 8452.09it/s] 87%|████████▋ | 349185/400000 [00:41<00:05, 8542.02it/s] 88%|████████▊ | 350046/400000 [00:41<00:05, 8562.21it/s] 88%|████████▊ | 350904/400000 [00:41<00:05, 8511.40it/s] 88%|████████▊ | 351756/400000 [00:41<00:05, 8267.14it/s] 88%|████████▊ | 352585/400000 [00:41<00:05, 8139.54it/s] 88%|████████▊ | 353401/400000 [00:41<00:05, 8093.58it/s] 89%|████████▊ | 354212/400000 [00:41<00:05, 7976.46it/s] 89%|████████▉ | 355050/400000 [00:41<00:05, 8092.93it/s] 89%|████████▉ | 355929/400000 [00:41<00:05, 8289.56it/s] 89%|████████▉ | 356808/400000 [00:41<00:05, 8432.11it/s] 89%|████████▉ | 357691/400000 [00:42<00:04, 8545.25it/s] 90%|████████▉ | 358566/400000 [00:42<00:04, 8605.28it/s] 90%|████████▉ | 359435/400000 [00:42<00:04, 8627.63it/s] 90%|█████████ | 360305/400000 [00:42<00:04, 8648.56it/s] 90%|█████████ | 361179/400000 [00:42<00:04, 8675.38it/s] 91%|█████████ | 362048/400000 [00:42<00:04, 8654.95it/s] 91%|█████████ | 362914/400000 [00:42<00:04, 8583.00it/s] 91%|█████████ | 363773/400000 [00:42<00:04, 8306.64it/s] 91%|█████████ | 364606/400000 [00:42<00:04, 8304.56it/s] 91%|█████████▏| 365484/400000 [00:42<00:04, 8438.96it/s] 92%|█████████▏| 366360/400000 [00:43<00:03, 8531.02it/s] 92%|█████████▏| 367239/400000 [00:43<00:03, 8606.42it/s] 92%|█████████▏| 368112/400000 [00:43<00:03, 8640.29it/s] 92%|█████████▏| 368977/400000 [00:43<00:03, 8616.16it/s] 92%|█████████▏| 369851/400000 [00:43<00:03, 8652.39it/s] 93%|█████████▎| 370728/400000 [00:43<00:03, 8685.43it/s] 93%|█████████▎| 371610/400000 [00:43<00:03, 8722.64it/s] 93%|█████████▎| 372483/400000 [00:43<00:03, 8703.68it/s] 93%|█████████▎| 373354/400000 [00:43<00:03, 8697.24it/s] 94%|█████████▎| 374224/400000 [00:43<00:02, 8690.68it/s] 94%|█████████▍| 375094/400000 [00:44<00:02, 8603.41it/s] 94%|█████████▍| 375968/400000 [00:44<00:02, 8641.87it/s] 94%|█████████▍| 376833/400000 [00:44<00:02, 8642.41it/s] 94%|█████████▍| 377706/400000 [00:44<00:02, 8667.67it/s] 95%|█████████▍| 378584/400000 [00:44<00:02, 8700.70it/s] 95%|█████████▍| 379463/400000 [00:44<00:02, 8724.48it/s] 95%|█████████▌| 380336/400000 [00:44<00:02, 8659.23it/s] 95%|█████████▌| 381204/400000 [00:44<00:02, 8663.83it/s] 96%|█████████▌| 382080/400000 [00:44<00:02, 8691.93it/s] 96%|█████████▌| 382961/400000 [00:44<00:01, 8725.66it/s] 96%|█████████▌| 383834/400000 [00:45<00:01, 8696.04it/s] 96%|█████████▌| 384711/400000 [00:45<00:01, 8715.83it/s] 96%|█████████▋| 385583/400000 [00:45<00:01, 8708.05it/s] 97%|█████████▋| 386461/400000 [00:45<00:01, 8728.66it/s] 97%|█████████▋| 387343/400000 [00:45<00:01, 8753.21it/s] 97%|█████████▋| 388219/400000 [00:45<00:01, 8751.83it/s] 97%|█████████▋| 389095/400000 [00:45<00:01, 8726.29it/s] 97%|█████████▋| 389968/400000 [00:45<00:01, 8701.31it/s] 98%|█████████▊| 390846/400000 [00:45<00:01, 8723.96it/s] 98%|█████████▊| 391725/400000 [00:45<00:00, 8743.46it/s] 98%|█████████▊| 392603/400000 [00:46<00:00, 8753.52it/s] 98%|█████████▊| 393479/400000 [00:46<00:00, 8745.75it/s] 99%|█████████▊| 394354/400000 [00:46<00:00, 8736.59it/s] 99%|█████████▉| 395230/400000 [00:46<00:00, 8742.81it/s] 99%|█████████▉| 396106/400000 [00:46<00:00, 8745.11it/s] 99%|█████████▉| 396987/400000 [00:46<00:00, 8763.36it/s] 99%|█████████▉| 397866/400000 [00:46<00:00, 8770.22it/s]100%|█████████▉| 398744/400000 [00:46<00:00, 8740.26it/s]100%|█████████▉| 399625/400000 [00:46<00:00, 8760.14it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8523.98it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f6fcb099940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011195742862248455 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.010809392434697486 	 Accuracy: 74

  model saves at 74% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15837 out of table with 15738 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15837 out of table with 15738 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-15 16:25:46.541519: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 16:25:46.545401: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095230000 Hz
2020-05-15 16:25:46.545553: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564e314e7e50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 16:25:46.545566: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f6fd6c0af98> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7433 - accuracy: 0.4950
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6896 - accuracy: 0.4985 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6922 - accuracy: 0.4983
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7395 - accuracy: 0.4952
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7034 - accuracy: 0.4976
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7612 - accuracy: 0.4938
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7411 - accuracy: 0.4951
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7452 - accuracy: 0.4949
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7245 - accuracy: 0.4962
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7464 - accuracy: 0.4948
11000/25000 [============>.................] - ETA: 3s - loss: 7.7307 - accuracy: 0.4958
12000/25000 [=============>................] - ETA: 3s - loss: 7.7458 - accuracy: 0.4948
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7480 - accuracy: 0.4947
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7039 - accuracy: 0.4976
15000/25000 [=================>............] - ETA: 2s - loss: 7.6738 - accuracy: 0.4995
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6724 - accuracy: 0.4996
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6747 - accuracy: 0.4995
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6615 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6368 - accuracy: 0.5019
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6429 - accuracy: 0.5016
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6389 - accuracy: 0.5018
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6617 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6640 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6602 - accuracy: 0.5004
25000/25000 [==============================] - 7s 278us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f6f2b0390f0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f6f2b001ef0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.5455 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.4569 - val_crf_viterbi_accuracy: 0.0133

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

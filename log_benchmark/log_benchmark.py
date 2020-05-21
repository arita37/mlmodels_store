
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '51b64e342c7b2661e79b8abaa33db92672ae95c7', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fef29c9de80> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-21 12:14:16.449656
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-21 12:14:16.453324
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-21 12:14:16.456370
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-21 12:14:16.459569
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fef35a67320> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354016.3125
Epoch 2/10

1/1 [==============================] - 0s 110ms/step - loss: 269913.2812
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 186745.5469
Epoch 4/10

1/1 [==============================] - 0s 93ms/step - loss: 121468.1016
Epoch 5/10

1/1 [==============================] - 0s 93ms/step - loss: 78661.2188
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 52201.2773
Epoch 7/10

1/1 [==============================] - 0s 99ms/step - loss: 35640.2617
Epoch 8/10

1/1 [==============================] - 0s 93ms/step - loss: 25142.2285
Epoch 9/10

1/1 [==============================] - 0s 91ms/step - loss: 18500.4258
Epoch 10/10

1/1 [==============================] - 0s 93ms/step - loss: 14206.7803

  #### Inference Need return ypred, ytrue ######################### 
[[-0.94140875 -0.1717309   1.6906374   0.5752399  -0.03568253  0.5161906
   0.1425     -0.18306996  0.7131743  -0.27125117 -1.29512    -0.7451145
   0.05990416  0.40089035  1.4056506  -1.012149    1.4107744   0.08796042
   0.54313207  0.7706739   0.8751159   0.582454    0.5878904   0.26805675
   0.473413   -0.5579102   1.243897   -0.37695011  0.8563049  -1.6464667
   0.41433576  0.99142456 -0.82441205 -1.4991708  -1.0778255   0.18672654
  -0.40407458 -0.21438181 -0.17898442 -0.87171006 -0.29911247  0.38793385
  -1.0918708  -0.58918047 -0.7769408   0.5379947  -0.34040475  0.7793375
  -0.83337176  0.294872    0.19254045 -1.2560592  -0.11695409 -0.3026915
  -0.09424064 -0.22175816  0.76553094 -0.2372618   0.32215434  0.21419977
  -0.09418327  5.3170624   5.1746526   3.611123    4.249423    5.672055
   5.3605933   5.8006883   4.7981324   3.677352    4.7038956   5.4025893
   5.135629    4.9742446   4.79555     5.926383    5.6729803   4.5687947
   4.734825    4.329396    4.85985     5.423412    5.4413476   5.546266
   5.789979    5.143298    5.830013    5.0223675   4.386522    4.1634274
   5.4613934   5.01189     3.5307558   5.019582    4.3542857   4.489886
   5.0618153   6.4200277   4.7378807   4.723408    4.8256955   4.6893363
   4.6092086   4.8050947   5.091391    5.4424214   4.973111    5.187018
   6.2834673   4.21733     5.7992325   6.719028    5.27681     5.6160617
   5.1477356   5.0802593   4.5183244   4.3108463   4.6334887   4.5744767
   0.37680942  1.0913911   0.27648443  1.0374162  -0.22853386 -0.7006584
  -0.5739242   1.2823342   0.32603556 -0.81859934  0.92192173 -1.085681
  -0.5553584   0.6606811  -0.3727166   0.31090283 -1.324351    0.23390499
  -1.0241951  -0.2986839  -0.974414    0.6955278   0.5778147   0.57655007
   1.1221551   1.0561639  -0.34337807  0.27719012  0.4789688   0.9399935
   0.29234657  0.63308084  1.742235   -1.562383    0.4878823  -0.5593113
  -0.3514265   0.03973168 -0.26484334  0.15463603  0.7816877   0.46065158
  -1.5980637  -0.432836    0.407233    0.98539156  0.23904781  0.709198
  -0.11868972  0.49414784  0.25601012 -1.3673701   0.15508446  1.2121234
  -0.958537    1.124202   -0.3465697   0.26769468 -0.2766757   0.03485465
   0.30520064  0.6315714   1.5063131   0.87208927  0.44077408  1.2196969
   0.7445711   1.8215177   0.82228494  0.5699141   1.4468161   1.1318237
   1.5048816   0.36827928  0.41735625  0.64300203  0.6379297   1.4251235
   1.1907314   2.4938107   0.7475594   0.9895224   1.0683789   1.3224802
   2.0176287   1.0339482   0.6705227   0.3306812   0.52971286  0.2749951
   1.9422388   2.5823646   0.86153114  1.1283774   0.93183064  2.1945236
   1.0212973   0.5976453   0.2200836   1.7624259   1.7116352   0.44128108
   0.5839398   1.853159    1.0356896   0.9017521   0.7087445   0.673895
   1.4602727   1.9054799   1.1866882   0.5769746   1.4358402   1.3482809
   0.6124295   2.2261977   0.6608796   0.37330878  0.3357637   1.6094027
   0.0451318   7.029656    7.150513    6.266234    5.726708    4.5313864
   5.536222    5.2371244   6.7914333   6.55519     6.568003    4.3667955
   4.4171553   5.450899    6.058654    5.582108    5.665799    5.2885337
   6.2628922   5.9762006   5.2434907   4.5916886   7.1919417   5.6229033
   4.3981943   5.1660013   4.8450494   4.544823    5.3183465   6.7330675
   7.0886226   5.975547    4.9070907   5.2491746   6.8931904   5.183033
   4.45159     6.012698    5.744041    6.639362    4.3998165   5.8051324
   5.5730867   5.6326904   6.8814816   6.568466    5.6861324   6.734257
   6.0444307   5.899673    5.788512    5.782769    4.355907    5.5373597
   4.423104    7.1008205   5.662097    6.730588    5.549443    6.5665164
   0.42898148  0.6917223   1.3038915   1.3197111   1.364541    2.2596812
   0.24444646  1.6149358   1.785422    1.0717671   0.5052449   0.5533216
   0.75822484  0.42460966  0.4817052   0.9840051   1.7059886   0.17061996
   0.29236937  2.3344655   0.41844106  1.887229    0.86292565  0.6038343
   0.7556233   1.2150667   1.680022    0.37081254  1.8353121   0.7057918
   0.26682627  0.18957376  1.0716891   0.29060423  1.2462248   0.6644493
   1.079965    1.8731933   0.44153428  0.40320867  1.0282764   0.5024141
   1.5529437   1.4986827   1.9169517   1.8046998   0.4059825   2.1826048
   0.67483425  0.38365692  0.8800053   2.081149    0.4421128   1.1979713
   1.1929337   0.23196018  0.5916042   0.2731067   1.0374764   0.28499055
  -1.320411    6.676684   -0.21892059]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-21 12:14:26.360152
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.2484
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-21 12:14:26.363779
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9475.78
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-21 12:14:26.367485
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.2331
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-21 12:14:26.370632
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -847.623
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140664832848336
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140663874441792
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140663874442296
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140663874442800
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140663874443304
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140663874443808

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fef318e9e10> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.561268
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.527569
grad_step = 000002, loss = 0.496945
grad_step = 000003, loss = 0.463597
grad_step = 000004, loss = 0.428719
grad_step = 000005, loss = 0.397585
grad_step = 000006, loss = 0.374334
grad_step = 000007, loss = 0.364725
grad_step = 000008, loss = 0.353675
grad_step = 000009, loss = 0.335104
grad_step = 000010, loss = 0.318585
grad_step = 000011, loss = 0.307305
grad_step = 000012, loss = 0.298511
grad_step = 000013, loss = 0.289694
grad_step = 000014, loss = 0.279672
grad_step = 000015, loss = 0.267845
grad_step = 000016, loss = 0.255122
grad_step = 000017, loss = 0.244157
grad_step = 000018, loss = 0.235903
grad_step = 000019, loss = 0.227353
grad_step = 000020, loss = 0.216713
grad_step = 000021, loss = 0.205984
grad_step = 000022, loss = 0.196806
grad_step = 000023, loss = 0.188524
grad_step = 000024, loss = 0.180020
grad_step = 000025, loss = 0.171263
grad_step = 000026, loss = 0.162639
grad_step = 000027, loss = 0.154365
grad_step = 000028, loss = 0.146563
grad_step = 000029, loss = 0.139298
grad_step = 000030, loss = 0.132020
grad_step = 000031, loss = 0.124667
grad_step = 000032, loss = 0.117851
grad_step = 000033, loss = 0.111401
grad_step = 000034, loss = 0.104972
grad_step = 000035, loss = 0.098889
grad_step = 000036, loss = 0.093222
grad_step = 000037, loss = 0.087605
grad_step = 000038, loss = 0.082297
grad_step = 000039, loss = 0.077319
grad_step = 000040, loss = 0.072473
grad_step = 000041, loss = 0.067894
grad_step = 000042, loss = 0.063572
grad_step = 000043, loss = 0.059462
grad_step = 000044, loss = 0.055669
grad_step = 000045, loss = 0.051962
grad_step = 000046, loss = 0.048423
grad_step = 000047, loss = 0.044871
grad_step = 000048, loss = 0.041404
grad_step = 000049, loss = 0.038226
grad_step = 000050, loss = 0.035507
grad_step = 000051, loss = 0.033076
grad_step = 000052, loss = 0.030694
grad_step = 000053, loss = 0.028302
grad_step = 000054, loss = 0.026074
grad_step = 000055, loss = 0.024109
grad_step = 000056, loss = 0.022333
grad_step = 000057, loss = 0.020607
grad_step = 000058, loss = 0.018932
grad_step = 000059, loss = 0.017405
grad_step = 000060, loss = 0.016077
grad_step = 000061, loss = 0.014917
grad_step = 000062, loss = 0.013785
grad_step = 000063, loss = 0.012679
grad_step = 000064, loss = 0.011663
grad_step = 000065, loss = 0.010754
grad_step = 000066, loss = 0.009944
grad_step = 000067, loss = 0.009229
grad_step = 000068, loss = 0.008553
grad_step = 000069, loss = 0.007902
grad_step = 000070, loss = 0.007307
grad_step = 000071, loss = 0.006782
grad_step = 000072, loss = 0.006313
grad_step = 000073, loss = 0.005884
grad_step = 000074, loss = 0.005485
grad_step = 000075, loss = 0.005119
grad_step = 000076, loss = 0.004795
grad_step = 000077, loss = 0.004511
grad_step = 000078, loss = 0.004247
grad_step = 000079, loss = 0.004000
grad_step = 000080, loss = 0.003779
grad_step = 000081, loss = 0.003584
grad_step = 000082, loss = 0.003412
grad_step = 000083, loss = 0.003260
grad_step = 000084, loss = 0.003117
grad_step = 000085, loss = 0.002983
grad_step = 000086, loss = 0.002865
grad_step = 000087, loss = 0.002769
grad_step = 000088, loss = 0.002683
grad_step = 000089, loss = 0.002604
grad_step = 000090, loss = 0.002534
grad_step = 000091, loss = 0.002471
grad_step = 000092, loss = 0.002420
grad_step = 000093, loss = 0.002378
grad_step = 000094, loss = 0.002338
grad_step = 000095, loss = 0.002303
grad_step = 000096, loss = 0.002274
grad_step = 000097, loss = 0.002251
grad_step = 000098, loss = 0.002231
grad_step = 000099, loss = 0.002213
grad_step = 000100, loss = 0.002196
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002182
grad_step = 000102, loss = 0.002171
grad_step = 000103, loss = 0.002161
grad_step = 000104, loss = 0.002151
grad_step = 000105, loss = 0.002146
grad_step = 000106, loss = 0.002135
grad_step = 000107, loss = 0.002128
grad_step = 000108, loss = 0.002121
grad_step = 000109, loss = 0.002115
grad_step = 000110, loss = 0.002106
grad_step = 000111, loss = 0.002102
grad_step = 000112, loss = 0.002095
grad_step = 000113, loss = 0.002088
grad_step = 000114, loss = 0.002083
grad_step = 000115, loss = 0.002075
grad_step = 000116, loss = 0.002069
grad_step = 000117, loss = 0.002064
grad_step = 000118, loss = 0.002056
grad_step = 000119, loss = 0.002049
grad_step = 000120, loss = 0.002043
grad_step = 000121, loss = 0.002035
grad_step = 000122, loss = 0.002028
grad_step = 000123, loss = 0.002021
grad_step = 000124, loss = 0.002013
grad_step = 000125, loss = 0.002007
grad_step = 000126, loss = 0.001998
grad_step = 000127, loss = 0.001989
grad_step = 000128, loss = 0.001981
grad_step = 000129, loss = 0.001974
grad_step = 000130, loss = 0.001970
grad_step = 000131, loss = 0.001971
grad_step = 000132, loss = 0.001974
grad_step = 000133, loss = 0.001964
grad_step = 000134, loss = 0.001936
grad_step = 000135, loss = 0.001928
grad_step = 000136, loss = 0.001940
grad_step = 000137, loss = 0.001940
grad_step = 000138, loss = 0.001928
grad_step = 000139, loss = 0.001903
grad_step = 000140, loss = 0.001892
grad_step = 000141, loss = 0.001890
grad_step = 000142, loss = 0.001892
grad_step = 000143, loss = 0.001909
grad_step = 000144, loss = 0.001911
grad_step = 000145, loss = 0.001928
grad_step = 000146, loss = 0.001869
grad_step = 000147, loss = 0.001869
grad_step = 000148, loss = 0.001908
grad_step = 000149, loss = 0.001865
grad_step = 000150, loss = 0.001852
grad_step = 000151, loss = 0.001864
grad_step = 000152, loss = 0.001850
grad_step = 000153, loss = 0.001843
grad_step = 000154, loss = 0.001839
grad_step = 000155, loss = 0.001836
grad_step = 000156, loss = 0.001846
grad_step = 000157, loss = 0.001845
grad_step = 000158, loss = 0.001847
grad_step = 000159, loss = 0.001828
grad_step = 000160, loss = 0.001820
grad_step = 000161, loss = 0.001814
grad_step = 000162, loss = 0.001811
grad_step = 000163, loss = 0.001809
grad_step = 000164, loss = 0.001809
grad_step = 000165, loss = 0.001812
grad_step = 000166, loss = 0.001806
grad_step = 000167, loss = 0.001802
grad_step = 000168, loss = 0.001794
grad_step = 000169, loss = 0.001789
grad_step = 000170, loss = 0.001787
grad_step = 000171, loss = 0.001786
grad_step = 000172, loss = 0.001787
grad_step = 000173, loss = 0.001791
grad_step = 000174, loss = 0.001816
grad_step = 000175, loss = 0.001831
grad_step = 000176, loss = 0.001876
grad_step = 000177, loss = 0.001801
grad_step = 000178, loss = 0.001791
grad_step = 000179, loss = 0.001825
grad_step = 000180, loss = 0.001775
grad_step = 000181, loss = 0.001764
grad_step = 000182, loss = 0.001798
grad_step = 000183, loss = 0.001772
grad_step = 000184, loss = 0.001764
grad_step = 000185, loss = 0.001782
grad_step = 000186, loss = 0.001752
grad_step = 000187, loss = 0.001744
grad_step = 000188, loss = 0.001754
grad_step = 000189, loss = 0.001743
grad_step = 000190, loss = 0.001736
grad_step = 000191, loss = 0.001742
grad_step = 000192, loss = 0.001747
grad_step = 000193, loss = 0.001742
grad_step = 000194, loss = 0.001736
grad_step = 000195, loss = 0.001737
grad_step = 000196, loss = 0.001741
grad_step = 000197, loss = 0.001733
grad_step = 000198, loss = 0.001725
grad_step = 000199, loss = 0.001724
grad_step = 000200, loss = 0.001727
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001728
grad_step = 000202, loss = 0.001725
grad_step = 000203, loss = 0.001723
grad_step = 000204, loss = 0.001727
grad_step = 000205, loss = 0.001740
grad_step = 000206, loss = 0.001765
grad_step = 000207, loss = 0.001784
grad_step = 000208, loss = 0.001809
grad_step = 000209, loss = 0.001779
grad_step = 000210, loss = 0.001755
grad_step = 000211, loss = 0.001735
grad_step = 000212, loss = 0.001714
grad_step = 000213, loss = 0.001693
grad_step = 000214, loss = 0.001670
grad_step = 000215, loss = 0.001680
grad_step = 000216, loss = 0.001705
grad_step = 000217, loss = 0.001710
grad_step = 000218, loss = 0.001709
grad_step = 000219, loss = 0.001709
grad_step = 000220, loss = 0.001695
grad_step = 000221, loss = 0.001675
grad_step = 000222, loss = 0.001653
grad_step = 000223, loss = 0.001645
grad_step = 000224, loss = 0.001652
grad_step = 000225, loss = 0.001663
grad_step = 000226, loss = 0.001674
grad_step = 000227, loss = 0.001677
grad_step = 000228, loss = 0.001678
grad_step = 000229, loss = 0.001676
grad_step = 000230, loss = 0.001673
grad_step = 000231, loss = 0.001668
grad_step = 000232, loss = 0.001662
grad_step = 000233, loss = 0.001655
grad_step = 000234, loss = 0.001646
grad_step = 000235, loss = 0.001640
grad_step = 000236, loss = 0.001627
grad_step = 000237, loss = 0.001618
grad_step = 000238, loss = 0.001610
grad_step = 000239, loss = 0.001609
grad_step = 000240, loss = 0.001614
grad_step = 000241, loss = 0.001623
grad_step = 000242, loss = 0.001640
grad_step = 000243, loss = 0.001660
grad_step = 000244, loss = 0.001691
grad_step = 000245, loss = 0.001716
grad_step = 000246, loss = 0.001736
grad_step = 000247, loss = 0.001719
grad_step = 000248, loss = 0.001671
grad_step = 000249, loss = 0.001627
grad_step = 000250, loss = 0.001609
grad_step = 000251, loss = 0.001632
grad_step = 000252, loss = 0.001634
grad_step = 000253, loss = 0.001639
grad_step = 000254, loss = 0.001630
grad_step = 000255, loss = 0.001619
grad_step = 000256, loss = 0.001616
grad_step = 000257, loss = 0.001586
grad_step = 000258, loss = 0.001566
grad_step = 000259, loss = 0.001559
grad_step = 000260, loss = 0.001573
grad_step = 000261, loss = 0.001594
grad_step = 000262, loss = 0.001588
grad_step = 000263, loss = 0.001574
grad_step = 000264, loss = 0.001563
grad_step = 000265, loss = 0.001560
grad_step = 000266, loss = 0.001561
grad_step = 000267, loss = 0.001556
grad_step = 000268, loss = 0.001552
grad_step = 000269, loss = 0.001543
grad_step = 000270, loss = 0.001538
grad_step = 000271, loss = 0.001537
grad_step = 000272, loss = 0.001542
grad_step = 000273, loss = 0.001551
grad_step = 000274, loss = 0.001564
grad_step = 000275, loss = 0.001581
grad_step = 000276, loss = 0.001597
grad_step = 000277, loss = 0.001614
grad_step = 000278, loss = 0.001617
grad_step = 000279, loss = 0.001608
grad_step = 000280, loss = 0.001581
grad_step = 000281, loss = 0.001550
grad_step = 000282, loss = 0.001528
grad_step = 000283, loss = 0.001525
grad_step = 000284, loss = 0.001545
grad_step = 000285, loss = 0.001552
grad_step = 000286, loss = 0.001560
grad_step = 000287, loss = 0.001522
grad_step = 000288, loss = 0.001499
grad_step = 000289, loss = 0.001500
grad_step = 000290, loss = 0.001512
grad_step = 000291, loss = 0.001522
grad_step = 000292, loss = 0.001505
grad_step = 000293, loss = 0.001491
grad_step = 000294, loss = 0.001489
grad_step = 000295, loss = 0.001497
grad_step = 000296, loss = 0.001509
grad_step = 000297, loss = 0.001515
grad_step = 000298, loss = 0.001526
grad_step = 000299, loss = 0.001524
grad_step = 000300, loss = 0.001529
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001538
grad_step = 000302, loss = 0.001565
grad_step = 000303, loss = 0.001617
grad_step = 000304, loss = 0.001698
grad_step = 000305, loss = 0.001819
grad_step = 000306, loss = 0.001850
grad_step = 000307, loss = 0.001814
grad_step = 000308, loss = 0.001595
grad_step = 000309, loss = 0.001468
grad_step = 000310, loss = 0.001521
grad_step = 000311, loss = 0.001628
grad_step = 000312, loss = 0.001634
grad_step = 000313, loss = 0.001517
grad_step = 000314, loss = 0.001471
grad_step = 000315, loss = 0.001524
grad_step = 000316, loss = 0.001542
grad_step = 000317, loss = 0.001523
grad_step = 000318, loss = 0.001502
grad_step = 000319, loss = 0.001483
grad_step = 000320, loss = 0.001475
grad_step = 000321, loss = 0.001476
grad_step = 000322, loss = 0.001496
grad_step = 000323, loss = 0.001504
grad_step = 000324, loss = 0.001463
grad_step = 000325, loss = 0.001442
grad_step = 000326, loss = 0.001459
grad_step = 000327, loss = 0.001479
grad_step = 000328, loss = 0.001474
grad_step = 000329, loss = 0.001444
grad_step = 000330, loss = 0.001436
grad_step = 000331, loss = 0.001447
grad_step = 000332, loss = 0.001451
grad_step = 000333, loss = 0.001446
grad_step = 000334, loss = 0.001439
grad_step = 000335, loss = 0.001434
grad_step = 000336, loss = 0.001431
grad_step = 000337, loss = 0.001427
grad_step = 000338, loss = 0.001428
grad_step = 000339, loss = 0.001432
grad_step = 000340, loss = 0.001429
grad_step = 000341, loss = 0.001421
grad_step = 000342, loss = 0.001413
grad_step = 000343, loss = 0.001412
grad_step = 000344, loss = 0.001416
grad_step = 000345, loss = 0.001417
grad_step = 000346, loss = 0.001415
grad_step = 000347, loss = 0.001410
grad_step = 000348, loss = 0.001406
grad_step = 000349, loss = 0.001403
grad_step = 000350, loss = 0.001403
grad_step = 000351, loss = 0.001403
grad_step = 000352, loss = 0.001400
grad_step = 000353, loss = 0.001396
grad_step = 000354, loss = 0.001393
grad_step = 000355, loss = 0.001392
grad_step = 000356, loss = 0.001391
grad_step = 000357, loss = 0.001391
grad_step = 000358, loss = 0.001391
grad_step = 000359, loss = 0.001390
grad_step = 000360, loss = 0.001389
grad_step = 000361, loss = 0.001388
grad_step = 000362, loss = 0.001390
grad_step = 000363, loss = 0.001395
grad_step = 000364, loss = 0.001407
grad_step = 000365, loss = 0.001429
grad_step = 000366, loss = 0.001471
grad_step = 000367, loss = 0.001535
grad_step = 000368, loss = 0.001636
grad_step = 000369, loss = 0.001690
grad_step = 000370, loss = 0.001709
grad_step = 000371, loss = 0.001565
grad_step = 000372, loss = 0.001420
grad_step = 000373, loss = 0.001372
grad_step = 000374, loss = 0.001440
grad_step = 000375, loss = 0.001501
grad_step = 000376, loss = 0.001462
grad_step = 000377, loss = 0.001395
grad_step = 000378, loss = 0.001389
grad_step = 000379, loss = 0.001410
grad_step = 000380, loss = 0.001415
grad_step = 000381, loss = 0.001397
grad_step = 000382, loss = 0.001393
grad_step = 000383, loss = 0.001401
grad_step = 000384, loss = 0.001379
grad_step = 000385, loss = 0.001361
grad_step = 000386, loss = 0.001362
grad_step = 000387, loss = 0.001377
grad_step = 000388, loss = 0.001386
grad_step = 000389, loss = 0.001370
grad_step = 000390, loss = 0.001352
grad_step = 000391, loss = 0.001340
grad_step = 000392, loss = 0.001339
grad_step = 000393, loss = 0.001349
grad_step = 000394, loss = 0.001354
grad_step = 000395, loss = 0.001352
grad_step = 000396, loss = 0.001339
grad_step = 000397, loss = 0.001329
grad_step = 000398, loss = 0.001328
grad_step = 000399, loss = 0.001330
grad_step = 000400, loss = 0.001329
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001325
grad_step = 000402, loss = 0.001322
grad_step = 000403, loss = 0.001321
grad_step = 000404, loss = 0.001320
grad_step = 000405, loss = 0.001319
grad_step = 000406, loss = 0.001317
grad_step = 000407, loss = 0.001316
grad_step = 000408, loss = 0.001313
grad_step = 000409, loss = 0.001310
grad_step = 000410, loss = 0.001305
grad_step = 000411, loss = 0.001300
grad_step = 000412, loss = 0.001296
grad_step = 000413, loss = 0.001293
grad_step = 000414, loss = 0.001291
grad_step = 000415, loss = 0.001290
grad_step = 000416, loss = 0.001288
grad_step = 000417, loss = 0.001287
grad_step = 000418, loss = 0.001286
grad_step = 000419, loss = 0.001285
grad_step = 000420, loss = 0.001285
grad_step = 000421, loss = 0.001287
grad_step = 000422, loss = 0.001292
grad_step = 000423, loss = 0.001299
grad_step = 000424, loss = 0.001318
grad_step = 000425, loss = 0.001348
grad_step = 000426, loss = 0.001410
grad_step = 000427, loss = 0.001497
grad_step = 000428, loss = 0.001654
grad_step = 000429, loss = 0.001841
grad_step = 000430, loss = 0.001967
grad_step = 000431, loss = 0.001946
grad_step = 000432, loss = 0.001592
grad_step = 000433, loss = 0.001434
grad_step = 000434, loss = 0.001460
grad_step = 000435, loss = 0.001577
grad_step = 000436, loss = 0.001551
grad_step = 000437, loss = 0.001330
grad_step = 000438, loss = 0.001352
grad_step = 000439, loss = 0.001522
grad_step = 000440, loss = 0.001452
grad_step = 000441, loss = 0.001266
grad_step = 000442, loss = 0.001322
grad_step = 000443, loss = 0.001433
grad_step = 000444, loss = 0.001330
grad_step = 000445, loss = 0.001249
grad_step = 000446, loss = 0.001332
grad_step = 000447, loss = 0.001352
grad_step = 000448, loss = 0.001249
grad_step = 000449, loss = 0.001269
grad_step = 000450, loss = 0.001311
grad_step = 000451, loss = 0.001260
grad_step = 000452, loss = 0.001237
grad_step = 000453, loss = 0.001273
grad_step = 000454, loss = 0.001253
grad_step = 000455, loss = 0.001224
grad_step = 000456, loss = 0.001247
grad_step = 000457, loss = 0.001248
grad_step = 000458, loss = 0.001216
grad_step = 000459, loss = 0.001217
grad_step = 000460, loss = 0.001239
grad_step = 000461, loss = 0.001218
grad_step = 000462, loss = 0.001200
grad_step = 000463, loss = 0.001211
grad_step = 000464, loss = 0.001215
grad_step = 000465, loss = 0.001198
grad_step = 000466, loss = 0.001195
grad_step = 000467, loss = 0.001201
grad_step = 000468, loss = 0.001193
grad_step = 000469, loss = 0.001185
grad_step = 000470, loss = 0.001189
grad_step = 000471, loss = 0.001188
grad_step = 000472, loss = 0.001177
grad_step = 000473, loss = 0.001175
grad_step = 000474, loss = 0.001178
grad_step = 000475, loss = 0.001175
grad_step = 000476, loss = 0.001168
grad_step = 000477, loss = 0.001168
grad_step = 000478, loss = 0.001167
grad_step = 000479, loss = 0.001162
grad_step = 000480, loss = 0.001157
grad_step = 000481, loss = 0.001157
grad_step = 000482, loss = 0.001155
grad_step = 000483, loss = 0.001151
grad_step = 000484, loss = 0.001149
grad_step = 000485, loss = 0.001149
grad_step = 000486, loss = 0.001149
grad_step = 000487, loss = 0.001150
grad_step = 000488, loss = 0.001157
grad_step = 000489, loss = 0.001180
grad_step = 000490, loss = 0.001229
grad_step = 000491, loss = 0.001337
grad_step = 000492, loss = 0.001497
grad_step = 000493, loss = 0.001621
grad_step = 000494, loss = 0.001700
grad_step = 000495, loss = 0.001325
grad_step = 000496, loss = 0.001176
grad_step = 000497, loss = 0.001262
grad_step = 000498, loss = 0.001287
grad_step = 000499, loss = 0.001217
grad_step = 000500, loss = 0.001217
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001231
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

  date_run                              2020-05-21 12:14:43.785946
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.279969
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-21 12:14:43.791704
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.208215
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-21 12:14:43.799147
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.138055
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-21 12:14:43.804584
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -2.1639
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
100%|██████████| 10/10 [00:02<00:00,  4.10it/s, avg_epoch_loss=5.24]
INFO:root:Epoch[0] Elapsed time 2.440 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.242081
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.242081260681152 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fef1564f320> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  8.18it/s, avg_epoch_loss=3.59e+3]
INFO:root:Epoch[0] Elapsed time 1.223 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=3590.403646
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 3590.4036458333335 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7feedc02cf28> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
 30%|███       | 3/10 [00:12<00:29,  4.25s/it, avg_epoch_loss=6.94] 60%|██████    | 6/10 [00:23<00:16,  4.09s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:35<00:03,  3.97s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:38<00:00,  3.87s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 38.706 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.863318
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.863318347930909 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fee7cd32588> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:02<00:00,  4.86it/s, avg_epoch_loss=5.78]
INFO:root:Epoch[0] Elapsed time 2.059 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.780079
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.7800788402557375 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fee60694e48> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
 10%|█         | 1/10 [02:09<19:26, 129.57s/it, avg_epoch_loss=0.412] 20%|██        | 2/10 [05:08<19:15, 144.44s/it, avg_epoch_loss=0.399] 30%|███       | 3/10 [08:38<19:07, 163.98s/it, avg_epoch_loss=0.39]  40%|████      | 4/10 [12:49<19:01, 190.26s/it, avg_epoch_loss=0.385] 50%|█████     | 5/10 [16:35<16:44, 200.92s/it, avg_epoch_loss=0.384] 60%|██████    | 6/10 [20:05<13:34, 203.65s/it, avg_epoch_loss=0.383] 70%|███████   | 7/10 [23:30<10:11, 203.92s/it, avg_epoch_loss=0.381] 80%|████████  | 8/10 [27:13<06:59, 209.75s/it, avg_epoch_loss=0.378] 90%|█████████ | 9/10 [30:50<03:31, 211.75s/it, avg_epoch_loss=0.376]100%|██████████| 10/10 [34:22<00:00, 211.96s/it, avg_epoch_loss=0.375]100%|██████████| 10/10 [34:22<00:00, 206.26s/it, avg_epoch_loss=0.375]
INFO:root:Epoch[0] Elapsed time 2062.637 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.374571
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3745713621377945 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fee60657128> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  5.67it/s, avg_epoch_loss=415]
INFO:root:Epoch[0] Elapsed time 1.914 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=414.652022
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 414.65202175008733 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7feedc29c710> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:00<00:00, 43.03it/s, avg_epoch_loss=5.14]
INFO:root:Epoch[0] Elapsed time 0.233 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.136962
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.136961555480957 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7feedc182208> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
0   2020-05-21 12:14:16.449656  ...    mean_absolute_error
1   2020-05-21 12:14:16.453324  ...     mean_squared_error
2   2020-05-21 12:14:16.456370  ...  median_absolute_error
3   2020-05-21 12:14:16.459569  ...               r2_score
4   2020-05-21 12:14:26.360152  ...    mean_absolute_error
5   2020-05-21 12:14:26.363779  ...     mean_squared_error
6   2020-05-21 12:14:26.367485  ...  median_absolute_error
7   2020-05-21 12:14:26.370632  ...               r2_score
8   2020-05-21 12:14:43.785946  ...    mean_absolute_error
9   2020-05-21 12:14:43.791704  ...     mean_squared_error
10  2020-05-21 12:14:43.799147  ...  median_absolute_error
11  2020-05-21 12:14:43.804584  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa577075da0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa51e9afa20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa521b58d30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa51e9afa20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa577075da0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa51e9afa20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa521b58d30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa51e9afa20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa577075da0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa51e9afa20> <class 'mlmodels.model_tch.torchhub.Model'>

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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa521b58d30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} 'data_info' 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/cnn/mnist 

  Empty DataFrame
Columns: [date_run, model_uri, json, dataset_uri, metric, metric_name]
Index: [] 
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fea0dfb50b8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=503c22ad5e6844ba033c17108130f2406fac2756593e7fab1b385ff5e2d50c14
  Stored in directory: /tmp/pip-ephem-wheel-cache-kd7wmcck/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fe9adce0c88> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1466368/17464789 [=>............................] - ETA: 0s
 5079040/17464789 [=======>......................] - ETA: 0s
10330112/17464789 [================>.............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-21 12:51:42.058544: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-21 12:51:42.070761: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095070000 Hz
2020-05-21 12:51:42.070955: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e4d98dc6b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 12:51:42.070971: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.8813 - accuracy: 0.4860
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7203 - accuracy: 0.4965 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7024 - accuracy: 0.4977
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6436 - accuracy: 0.5015
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6942 - accuracy: 0.4982
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6308 - accuracy: 0.5023
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6644 - accuracy: 0.5001
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6283 - accuracy: 0.5025
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6377 - accuracy: 0.5019
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6344 - accuracy: 0.5021
11000/25000 [============>.................] - ETA: 3s - loss: 7.6067 - accuracy: 0.5039
12000/25000 [=============>................] - ETA: 3s - loss: 7.6270 - accuracy: 0.5026
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6557 - accuracy: 0.5007
15000/25000 [=================>............] - ETA: 2s - loss: 7.6881 - accuracy: 0.4986
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7097 - accuracy: 0.4972
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7027 - accuracy: 0.4976
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6973 - accuracy: 0.4980
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6852 - accuracy: 0.4988
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6965 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6776 - accuracy: 0.4993
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6910 - accuracy: 0.4984
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6873 - accuracy: 0.4987
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6883 - accuracy: 0.4986
25000/25000 [==============================] - 7s 280us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-21 12:51:55.625222
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-21 12:51:55.625222  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<39:11:52, 6.11kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<27:40:52, 8.65kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:01<19:25:54, 12.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 868k/862M [00:01<13:36:24, 17.6kB/s].vector_cache/glove.6B.zip:   0%|          | 3.51M/862M [00:01<9:29:57, 25.1kB/s].vector_cache/glove.6B.zip:   1%|          | 7.95M/862M [00:01<6:37:00, 35.9kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.6M/862M [00:02<4:36:29, 51.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:02<3:12:32, 73.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:02<2:14:17, 104kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.3M/862M [00:02<1:33:31, 149kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:02<1:05:19, 212kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.5M/862M [00:02<45:32, 303kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:02<31:51, 431kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.5M/862M [00:02<22:16, 613kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.6M/862M [00:02<15:40, 869kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.4M/862M [00:03<11:04, 1.22MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:03<08:20, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<07:44, 1.74MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<07:27, 1.80MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<05:43, 2.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:07<06:24, 2.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<06:03, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<04:37, 2.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<06:00, 2.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<07:01, 1.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<05:37, 2.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:09<04:04, 3.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:11<34:36, 382kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:11<25:21, 521kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.4M/862M [00:11<18:07, 728kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:11<12:47, 1.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:13<1:01:55, 212kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<44:40, 294kB/s]  .vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:13<31:32, 416kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:15<25:06, 521kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<18:54, 692kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<13:29, 967kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:17<12:31, 1.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:17<10:05, 1.29MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:17<07:22, 1.76MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:19<08:13, 1.57MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:19<07:04, 1.83MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:19<05:15, 2.45MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:21<06:43, 1.91MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:21<06:01, 2.13MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:21<04:32, 2.83MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:23<06:12, 2.07MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:23<05:38, 2.27MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:23<04:15, 3.00MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:25<05:59, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:25<06:47, 1.88MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:25<05:23, 2.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<03:53, 3.25MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<21:11, 598kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<15:57, 794kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<11:33, 1.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<08:14, 1.53MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<32:33, 387kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<25:20, 497kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<18:15, 689kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<12:53, 973kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<18:03, 694kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<13:54, 901kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<10:02, 1.25MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<09:56, 1.25MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<10:20, 1.21MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<08:32, 1.46MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<07:35, 1.63MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<06:36, 1.88MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<04:53, 2.53MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<06:16, 1.96MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<06:54, 1.78MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<05:21, 2.29MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<03:56, 3.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<07:16, 1.69MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<06:22, 1.92MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<04:43, 2.59MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:07, 1.99MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<06:54, 1.76MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<05:27, 2.23MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<03:56, 3.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<44:00, 275kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<32:05, 378kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<22:41, 533kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<18:39, 646kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<15:32, 775kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<11:29, 1.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<08:09, 1.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<11:28:28, 17.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<8:02:53, 24.8kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<5:37:35, 35.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<3:58:24, 50.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<2:49:17, 70.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<1:58:55, 100kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<1:23:10, 143kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<1:03:47, 186kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<45:50, 258kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<32:19, 366kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<25:18, 465kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<18:56, 621kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<13:30, 870kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<12:09, 963kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<09:42, 1.21MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<07:02, 1.66MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:41, 1.51MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:34, 1.77MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<04:53, 2.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:09, 1.88MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:41, 1.73MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:12, 2.22MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<03:46, 3.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<11:19, 1.02MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<09:06, 1.26MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<06:40, 1.72MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:18, 1.57MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:25, 1.54MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:43, 1.99MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<04:08, 2.74MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<08:59, 1.26MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<07:28, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:31, 2.05MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<06:28, 1.75MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<06:50, 1.65MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:21, 2.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<03:50, 2.92MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<5:30:17, 34.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<3:52:12, 48.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<2:42:27, 68.9kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<1:55:58, 96.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<1:22:17, 136kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<57:45, 193kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<42:53, 259kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<31:09, 356kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<22:02, 502kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<17:59, 613kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<14:50, 742kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<10:52, 1.01MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<07:43, 1.42MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<12:12, 898kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<09:39, 1.13MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<07:02, 1.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<07:27, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<07:27, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:42, 1.90MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<04:06, 2.64MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<12:49, 843kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<10:06, 1.07MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<07:20, 1.47MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<07:39, 1.41MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<07:41, 1.40MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<05:56, 1.81MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:16, 2.50MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<1:19:10, 135kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<56:29, 189kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<39:43, 268kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<30:12, 351kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<23:19, 455kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<16:46, 632kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<11:55, 886kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<11:27, 921kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<09:05, 1.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:35, 1.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<07:03, 1.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<07:06, 1.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<05:25, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<03:57, 2.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:56, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<05:56, 1.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<04:25, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:30, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<06:00, 1.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<04:44, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<03:26, 2.99MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<1:15:51, 135kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<53:56, 190kB/s]  .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<38:10, 269kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<26:43, 382kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<28:19, 360kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<20:52, 488kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<14:50, 686kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<12:44, 795kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:39<09:47, 1.04MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<07:23, 1.37MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:15, 1.92MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<21:35, 466kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<15:59, 629kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<11:30, 873kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<08:08, 1.23MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<45:43, 219kB/s] .vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<32:59, 303kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<23:14, 429kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<16:20, 607kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<1:25:21, 116kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<1:01:47, 161kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<43:40, 227kB/s]  .vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [01:46<30:36, 323kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<25:40, 384kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<18:59, 519kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<13:34, 725kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<09:35, 1.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<21:37, 453kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<17:08, 571kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<12:24, 788kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<09:01, 1.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<08:19, 1.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<06:40, 1.46MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<05:12, 1.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:49, 2.53MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:36, 1.72MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:09, 1.87MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<03:52, 2.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<02:52, 3.35MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<08:01, 1.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<06:51, 1.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<05:03, 1.89MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<03:40, 2.60MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<09:51, 965kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<07:54, 1.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:45, 1.65MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<06:14, 1.51MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<05:15, 1.80MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:56, 2.39MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<02:53, 3.26MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<09:11, 1.02MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<08:33, 1.10MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<06:30, 1.44MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<04:39, 2.00MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<13:09, 708kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<09:59, 930kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<07:12, 1.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<05:14, 1.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<07:49, 1.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<06:23, 1.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:41, 1.97MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:25, 2.68MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<08:21, 1.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<06:49, 1.34MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:58, 1.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<05:30, 1.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<04:46, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<03:32, 2.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<04:36, 1.96MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:08, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<03:05, 2.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<04:18, 2.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<03:55, 2.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<02:58, 3.00MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<04:15, 2.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<04:48, 1.85MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<03:45, 2.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<02:45, 3.21MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<05:34, 1.58MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<04:47, 1.84MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<03:34, 2.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<04:32, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<04:04, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<03:04, 2.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<04:11, 2.07MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<03:49, 2.27MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<02:52, 3.01MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<04:02, 2.13MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<03:41, 2.33MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<02:45, 3.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<03:58, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:25<03:41, 2.31MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<02:46, 3.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<03:56, 2.15MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<03:38, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<02:43, 3.11MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<03:52, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:29<03:34, 2.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<02:42, 3.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<03:51, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<04:27, 1.87MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<03:30, 2.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<02:40, 3.10MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:45, 2.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<03:22, 2.46MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<02:31, 3.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<01:57, 4.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:04, 1.62MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<04:35, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<03:28, 2.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:03, 2.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<03:51, 2.11MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<02:54, 2.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<02:08, 3.77MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<10:15, 787kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<08:09, 989kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<05:54, 1.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<04:14, 1.89MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<11:30, 695kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<08:53, 899kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<06:29, 1.23MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:38, 1.71MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<07:59, 993kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<06:23, 1.24MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:38, 1.70MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<05:07, 1.54MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<07:28, 1.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<05:20, 1.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<05:30, 1.41MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:33, 1.71MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<03:29, 2.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<02:30, 3.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<16:34, 466kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<13:14, 583kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<09:36, 803kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<06:51, 1.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<07:00, 1.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:42, 1.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<04:07, 1.85MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:59, 2.54MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<43:04, 176kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<30:54, 245kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<21:45, 347kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<16:55, 444kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<12:36, 596kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<08:59, 833kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<08:01, 929kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<06:22, 1.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:36, 1.61MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<04:58, 1.48MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:15, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:08, 2.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<02:20, 3.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<05:19, 1.37MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:54, 1.49MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<03:41, 1.98MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<02:42, 2.69MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<04:43, 1.53MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<04:20, 1.67MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<03:15, 2.22MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<02:22, 3.03MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<08:01, 895kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<06:35, 1.09MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<04:51, 1.47MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<03:27, 2.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<26:59, 263kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<19:47, 359kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<14:02, 505kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<11:11, 629kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<08:41, 809kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<06:15, 1.12MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<04:27, 1.57MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<13:47, 506kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<10:29, 663kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<07:30, 924kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<05:20, 1.29MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<10:16, 672kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<07:59, 863kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:45, 1.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<04:06, 1.66MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<10:18, 662kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<08:00, 853kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<05:46, 1.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<05:27, 1.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<04:28, 1.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<03:20, 2.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<02:25, 2.77MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<06:51, 975kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<05:34, 1.20MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:03, 1.64MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<04:14, 1.56MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<04:27, 1.49MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:27, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<02:33, 2.57MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<03:22, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<03:02, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:17, 2.85MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<03:07, 2.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<02:51, 2.27MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:09, 2.99MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<03:01, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<02:45, 2.32MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:03, 3.11MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<01:31, 4.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<16:48, 378kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<12:18, 516kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<08:49, 717kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<06:14, 1.01MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:30<07:33, 831kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<05:55, 1.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<04:17, 1.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<04:27, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<03:45, 1.65MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:44, 2.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<01:59, 3.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<47:01, 131kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<33:31, 183kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<23:29, 260kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<16:31, 368kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<14:36, 416kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<10:58, 553kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<07:49, 772kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<05:34, 1.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<06:53, 871kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<05:35, 1.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:05, 1.46MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:56, 2.03MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<08:17, 715kB/s] .vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<06:32, 906kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<04:45, 1.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:25, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<05:17, 1.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<04:34, 1.28MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:23, 1.73MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:27, 2.36MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<04:24, 1.31MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<03:54, 1.48MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<02:54, 1.98MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:07, 2.70MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<04:23, 1.31MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<03:46, 1.52MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<02:48, 2.03MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:06, 1.82MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<02:52, 1.96MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<02:10, 2.58MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<01:35, 3.53MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<1:07:09, 83.2kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<47:39, 117kB/s]   .vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<33:24, 167kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<23:19, 237kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<21:27, 257kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<15:48, 349kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<11:13, 490kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<07:51, 694kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<5:26:04, 16.7kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<3:48:32, 23.8kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<2:39:22, 34.0kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<1:52:02, 48.0kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<1:18:54, 68.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<55:05, 97.1kB/s]  .vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<39:32, 134kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<28:11, 188kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<19:46, 267kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<14:59, 350kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<11:07, 471kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<07:53, 662kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<05:34, 930kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<07:29, 690kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<05:49, 887kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<04:11, 1.23MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<04:01, 1.27MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:18, 1.54MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:28, 2.06MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<01:47, 2.81MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<04:03, 1.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:22, 1.49MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:29, 2.01MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:50, 1.74MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:01, 1.65MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:20, 2.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<01:43, 2.86MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:37, 1.86MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:20, 2.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<01:45, 2.77MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:21, 2.05MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:08, 2.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:36, 2.97MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:14, 2.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:03, 2.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:31, 3.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:11, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:00, 2.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:30, 3.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:09, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<01:58, 2.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<01:29, 3.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:06, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<01:52, 2.44MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<01:25, 3.19MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<01:02, 4.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<17:43, 253kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<12:51, 349kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<09:03, 492kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<07:20, 602kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<05:34, 792kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:58, 1.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<03:47, 1.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<03:06, 1.40MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:15, 1.91MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:35, 1.66MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:41, 1.59MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:03, 2.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:29, 2.85MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<03:24, 1.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:48, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:03, 2.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<02:24, 1.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<02:06, 1.97MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:34, 2.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:38<05:01, 810kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:38<04:02, 1.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:38<03:14, 1.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:38<02:35, 1.56MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:38<02:09, 1.88MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:39<01:44, 2.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:39<01:43, 2.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:39<01:26, 2.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:39<01:14, 3.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:39<01:06, 3.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:39<00:58, 4.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:39<00:53, 4.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:39<00:47, 5.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:39<00:44, 5.40MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:40<00:41, 5.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<00:38, 6.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:41<03:55, 1.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:41<03:27, 1.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:41<02:35, 1.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:41<01:57, 1.99MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:41<01:32, 2.53MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:41<01:12, 3.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:43<02:52, 1.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:43<04:46, 808kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:43<03:59, 968kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:43<02:59, 1.29MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:43<02:13, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:43<01:42, 2.24MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:43<01:19, 2.89MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:45<03:15, 1.16MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:45<03:32, 1.07MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:45<02:47, 1.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:45<02:05, 1.80MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:45<01:34, 2.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<01:15, 2.97MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:47<03:32, 1.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:47<06:08, 606kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:47<05:11, 718kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:48<03:48, 975kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:48<02:50, 1.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:48<02:06, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:48<01:36, 2.29MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:49<02:32, 1.44MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:49<03:00, 1.22MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:49<02:22, 1.54MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:49<01:51, 1.96MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:50<01:26, 2.50MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:50<01:07, 3.19MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:50<00:54, 3.95MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:51<03:24, 1.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:51<03:31, 1.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:51<02:41, 1.33MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:51<02:05, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:52<01:35, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:52<01:12, 2.94MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:52<00:59, 3.57MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:53<42:16, 83.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:53<30:39, 115kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:53<21:39, 162kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:53<15:18, 229kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:53<10:46, 324kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:54<07:39, 453kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:54<05:26, 635kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:55<11:04, 312kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:55<08:48, 391kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:55<06:26, 535kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:55<04:35, 745kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:56<03:19, 1.02MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<02:25, 1.40MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:57<04:00, 844kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:57<03:51, 876kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:57<02:55, 1.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:57<02:12, 1.52MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:57<01:38, 2.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:58<01:16, 2.60MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:59<02:12, 1.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:59<03:36, 918kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:59<03:03, 1.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:59<02:16, 1.45MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:00<01:43, 1.90MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:00<01:17, 2.51MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:00<01:01, 3.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:01<03:10, 1.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:07<33:57, 95.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [05:07<23:50, 135kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:07<16:44, 192kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:07<11:49, 271kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:07<08:19, 383kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:07<05:53, 537kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:08<04:24, 714kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:08<03:08, 994kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:08<02:15, 1.38MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:10<03:35, 857kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:10<03:13, 953kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:10<02:24, 1.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:10<01:45, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:10<01:18, 2.31MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:12<02:13, 1.36MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:12<03:01, 994kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:12<02:29, 1.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:12<01:50, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:12<01:22, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:12<01:01, 2.88MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:14<02:45, 1.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:14<03:23, 868kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:14<02:43, 1.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:14<02:00, 1.46MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:14<01:27, 1.98MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:16<02:11, 1.32MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:16<02:14, 1.28MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:16<01:44, 1.64MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:16<01:17, 2.20MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:16<00:56, 2.96MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:18<04:06, 684kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:18<04:14, 661kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:18<03:18, 847kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:18<02:23, 1.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:18<01:43, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:20<02:18, 1.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:20<02:57, 926kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:20<02:23, 1.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:20<01:44, 1.55MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:20<01:18, 2.06MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:20<00:57, 2.77MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:22<04:33, 585kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:22<04:29, 593kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:22<03:28, 766kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:22<02:30, 1.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:22<01:48, 1.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:24<02:18, 1.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:24<02:54, 894kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:24<02:21, 1.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:24<01:43, 1.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:24<01:15, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:24<00:56, 2.68MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:26<13:13, 191kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:26<09:51, 256kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:26<07:01, 358kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:26<05:01, 496kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:28<04:01, 611kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:28<04:01, 612kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:28<03:06, 788kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:28<02:14, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:28<01:37, 1.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:28<01:11, 2.01MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:30<04:10, 573kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:30<04:13, 566kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:30<03:15, 732kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:30<02:20, 1.01MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:30<01:41, 1.39MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:32<02:04, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:32<02:36, 892kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:32<02:06, 1.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:32<01:32, 1.49MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:32<01:07, 2.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:34<01:44, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:34<02:20, 963kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:34<01:54, 1.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:34<01:24, 1.59MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:34<01:01, 2.15MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:36<01:43, 1.26MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:36<02:11, 997kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:36<01:47, 1.22MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:36<01:18, 1.65MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:36<00:57, 2.24MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:37<02:00, 1.06MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:38<02:21, 899kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:38<01:52, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:38<01:22, 1.52MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:38<00:59, 2.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:39<02:07, 964kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:40<02:23, 854kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:40<01:53, 1.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:40<01:21, 1.48MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:40<00:59, 2.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:41<01:33, 1.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:42<01:49, 1.08MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:42<01:26, 1.36MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:42<01:03, 1.85MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:42<00:45, 2.54MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:43<04:00, 476kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:44<03:28, 547kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:44<02:34, 739kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:44<01:48, 1.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:44<01:18, 1.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:45<02:02, 899kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:46<02:02, 899kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:46<01:34, 1.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:46<01:07, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:47<01:13, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:48<01:26, 1.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:48<01:07, 1.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:48<00:49, 2.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:48<00:35, 2.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:49<02:13, 765kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:49<02:03, 826kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:50<01:32, 1.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:50<01:05, 1.52MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:50<00:47, 2.08MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:51<04:11, 390kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:51<03:22, 483kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:52<02:27, 659kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:52<01:42, 925kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:53<01:44, 898kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:53<01:37, 957kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:54<01:14, 1.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:54<00:52, 1.74MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:55<01:14, 1.21MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:55<01:15, 1.19MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:56<00:56, 1.56MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:56<00:40, 2.13MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:57<00:50, 1.70MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:57<00:55, 1.55MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:58<00:42, 1.98MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:58<00:30, 2.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:59<01:05, 1.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:59<01:04, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [06:00<00:48, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:00<00:33, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:02<01:00, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:02<01:20, 956kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:02<01:05, 1.17MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:02<00:47, 1.60MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:02<00:32, 2.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:04<09:52, 123kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:04<07:10, 169kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:04<05:01, 239kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:04<03:25, 340kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:06<02:51, 401kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:06<02:16, 502kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:06<01:39, 686kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:06<01:07, 965kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:08<01:17, 833kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:08<01:11, 900kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:08<00:53, 1.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:08<00:37, 1.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:10<00:52, 1.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:10<00:51, 1.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:10<00:38, 1.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:10<00:27, 2.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:12<00:38, 1.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:12<00:40, 1.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:12<00:31, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:12<00:21, 2.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:14<00:39, 1.32MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:14<00:35, 1.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:14<00:26, 1.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:15<00:26, 1.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:16<00:28, 1.68MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:16<00:21, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:16<00:15, 2.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:17<00:31, 1.39MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:18<00:28, 1.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:18<00:20, 2.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:19<00:20, 1.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:20<00:21, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:20<00:16, 2.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:21<00:16, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:22<00:18, 1.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:22<00:14, 2.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:23<00:14, 2.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:24<00:15, 1.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:24<00:12, 2.49MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:25<00:12, 2.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:25<00:13, 1.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:26<00:10, 2.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:27<00:10, 2.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:27<00:11, 2.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:28<00:08, 2.51MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:29<00:08, 2.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:29<00:09, 1.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:30<00:07, 2.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:31<00:06, 2.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:31<00:07, 1.99MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:31<00:05, 2.55MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:32<00:03, 3.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:33<00:05, 1.86MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:33<00:05, 1.77MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:33<00:04, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:34<00:02, 3.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:35<00:14, 461kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:35<00:11, 585kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:35<00:07, 803kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:36<00:02, 1.13MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:37<00:07, 350kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:37<00:05, 457kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:37<00:02, 633kB/s].vector_cache/glove.6B.zip: 862MB [06:37, 2.17MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 832/400000 [00:00<00:47, 8316.02it/s]  0%|          | 1687/400000 [00:00<00:47, 8382.61it/s]  1%|          | 2559/400000 [00:00<00:46, 8479.00it/s]  1%|          | 3425/400000 [00:00<00:46, 8532.14it/s]  1%|          | 4282/400000 [00:00<00:46, 8543.23it/s]  1%|▏         | 5155/400000 [00:00<00:45, 8598.34it/s]  2%|▏         | 6036/400000 [00:00<00:45, 8658.21it/s]  2%|▏         | 6911/400000 [00:00<00:45, 8685.07it/s]  2%|▏         | 7783/400000 [00:00<00:45, 8693.21it/s]  2%|▏         | 8671/400000 [00:01<00:44, 8746.78it/s]  2%|▏         | 9558/400000 [00:01<00:44, 8783.32it/s]  3%|▎         | 10427/400000 [00:01<00:44, 8754.71it/s]  3%|▎         | 11309/400000 [00:01<00:44, 8772.87it/s]  3%|▎         | 12180/400000 [00:01<00:44, 8753.67it/s]  3%|▎         | 13050/400000 [00:01<00:44, 8711.92it/s]  3%|▎         | 13932/400000 [00:01<00:44, 8743.71it/s]  4%|▎         | 14804/400000 [00:01<00:44, 8562.71it/s]  4%|▍         | 15678/400000 [00:01<00:44, 8613.48it/s]  4%|▍         | 16547/400000 [00:01<00:44, 8633.77it/s]  4%|▍         | 17426/400000 [00:02<00:44, 8678.40it/s]  5%|▍         | 18309/400000 [00:02<00:43, 8720.84it/s]  5%|▍         | 19186/400000 [00:02<00:43, 8735.39it/s]  5%|▌         | 20060/400000 [00:02<00:43, 8729.60it/s]  5%|▌         | 20933/400000 [00:02<00:44, 8557.49it/s]  5%|▌         | 21798/400000 [00:02<00:44, 8584.33it/s]  6%|▌         | 22671/400000 [00:02<00:43, 8625.26it/s]  6%|▌         | 23534/400000 [00:02<00:46, 8076.53it/s]  6%|▌         | 24407/400000 [00:02<00:45, 8262.04it/s]  6%|▋         | 25257/400000 [00:02<00:44, 8331.50it/s]  7%|▋         | 26117/400000 [00:03<00:44, 8410.11it/s]  7%|▋         | 26972/400000 [00:03<00:44, 8449.99it/s]  7%|▋         | 27820/400000 [00:03<00:44, 8382.13it/s]  7%|▋         | 28676/400000 [00:03<00:44, 8433.64it/s]  7%|▋         | 29524/400000 [00:03<00:43, 8445.62it/s]  8%|▊         | 30374/400000 [00:03<00:43, 8461.48it/s]  8%|▊         | 31255/400000 [00:03<00:43, 8561.97it/s]  8%|▊         | 32134/400000 [00:03<00:42, 8626.63it/s]  8%|▊         | 32998/400000 [00:03<00:42, 8561.82it/s]  8%|▊         | 33855/400000 [00:03<00:42, 8541.31it/s]  9%|▊         | 34710/400000 [00:04<00:42, 8522.91it/s]  9%|▉         | 35577/400000 [00:04<00:42, 8565.15it/s]  9%|▉         | 36434/400000 [00:04<00:42, 8562.02it/s]  9%|▉         | 37302/400000 [00:04<00:42, 8595.40it/s] 10%|▉         | 38162/400000 [00:04<00:42, 8516.54it/s] 10%|▉         | 39018/400000 [00:04<00:42, 8528.44it/s] 10%|▉         | 39881/400000 [00:04<00:42, 8556.97it/s] 10%|█         | 40737/400000 [00:04<00:42, 8503.79it/s] 10%|█         | 41595/400000 [00:04<00:42, 8525.45it/s] 11%|█         | 42448/400000 [00:04<00:42, 8501.47it/s] 11%|█         | 43302/400000 [00:05<00:41, 8511.89it/s] 11%|█         | 44161/400000 [00:05<00:41, 8534.10it/s] 11%|█▏        | 45015/400000 [00:05<00:42, 8420.90it/s] 11%|█▏        | 45859/400000 [00:05<00:42, 8417.16it/s] 12%|█▏        | 46717/400000 [00:05<00:41, 8463.65it/s] 12%|█▏        | 47564/400000 [00:05<00:41, 8425.20it/s] 12%|█▏        | 48407/400000 [00:05<00:42, 8370.31it/s] 12%|█▏        | 49259/400000 [00:05<00:41, 8414.64it/s] 13%|█▎        | 50111/400000 [00:05<00:41, 8444.90it/s] 13%|█▎        | 50956/400000 [00:05<00:41, 8429.70it/s] 13%|█▎        | 51813/400000 [00:06<00:41, 8470.28it/s] 13%|█▎        | 52672/400000 [00:06<00:40, 8503.05it/s] 13%|█▎        | 53523/400000 [00:06<00:40, 8490.68it/s] 14%|█▎        | 54393/400000 [00:06<00:40, 8552.32it/s] 14%|█▍        | 55254/400000 [00:06<00:40, 8567.65it/s] 14%|█▍        | 56128/400000 [00:06<00:39, 8618.25it/s] 14%|█▍        | 56992/400000 [00:06<00:39, 8623.28it/s] 14%|█▍        | 57855/400000 [00:06<00:40, 8530.75it/s] 15%|█▍        | 58709/400000 [00:06<00:40, 8408.85it/s] 15%|█▍        | 59551/400000 [00:06<00:41, 8293.79it/s] 15%|█▌        | 60416/400000 [00:07<00:40, 8396.52it/s] 15%|█▌        | 61280/400000 [00:07<00:40, 8466.59it/s] 16%|█▌        | 62144/400000 [00:07<00:39, 8516.51it/s] 16%|█▌        | 63010/400000 [00:07<00:39, 8557.92it/s] 16%|█▌        | 63888/400000 [00:07<00:38, 8621.63it/s] 16%|█▌        | 64776/400000 [00:07<00:38, 8697.15it/s] 16%|█▋        | 65664/400000 [00:07<00:38, 8750.44it/s] 17%|█▋        | 66549/400000 [00:07<00:37, 8779.62it/s] 17%|█▋        | 67428/400000 [00:07<00:38, 8682.07it/s] 17%|█▋        | 68297/400000 [00:07<00:38, 8537.39it/s] 17%|█▋        | 69184/400000 [00:08<00:38, 8634.11it/s] 18%|█▊        | 70049/400000 [00:08<00:38, 8618.03it/s] 18%|█▊        | 70912/400000 [00:08<00:38, 8540.82it/s] 18%|█▊        | 71771/400000 [00:08<00:38, 8552.68it/s] 18%|█▊        | 72627/400000 [00:08<00:38, 8500.77it/s] 18%|█▊        | 73502/400000 [00:08<00:38, 8572.92it/s] 19%|█▊        | 74382/400000 [00:08<00:37, 8638.92it/s] 19%|█▉        | 75254/400000 [00:08<00:37, 8660.17it/s] 19%|█▉        | 76137/400000 [00:08<00:37, 8707.70it/s] 19%|█▉        | 77009/400000 [00:08<00:37, 8688.75it/s] 19%|█▉        | 77899/400000 [00:09<00:36, 8748.42it/s] 20%|█▉        | 78780/400000 [00:09<00:36, 8765.76it/s] 20%|█▉        | 79667/400000 [00:09<00:36, 8794.78it/s] 20%|██        | 80558/400000 [00:09<00:36, 8828.07it/s] 20%|██        | 81441/400000 [00:09<00:36, 8731.72it/s] 21%|██        | 82318/400000 [00:09<00:36, 8741.99it/s] 21%|██        | 83200/400000 [00:09<00:36, 8763.21it/s] 21%|██        | 84079/400000 [00:09<00:36, 8770.71it/s] 21%|██        | 84957/400000 [00:09<00:35, 8763.77it/s] 21%|██▏       | 85834/400000 [00:10<00:36, 8696.11it/s] 22%|██▏       | 86704/400000 [00:10<00:36, 8658.66it/s] 22%|██▏       | 87571/400000 [00:10<00:36, 8610.58it/s] 22%|██▏       | 88452/400000 [00:10<00:35, 8669.44it/s] 22%|██▏       | 89326/400000 [00:10<00:35, 8689.00it/s] 23%|██▎       | 90196/400000 [00:10<00:35, 8610.82it/s] 23%|██▎       | 91058/400000 [00:10<00:35, 8609.01it/s] 23%|██▎       | 91920/400000 [00:10<00:36, 8551.66it/s] 23%|██▎       | 92789/400000 [00:10<00:35, 8591.35it/s] 23%|██▎       | 93649/400000 [00:10<00:36, 8426.02it/s] 24%|██▎       | 94510/400000 [00:11<00:36, 8480.01it/s] 24%|██▍       | 95377/400000 [00:11<00:35, 8533.95it/s] 24%|██▍       | 96248/400000 [00:11<00:35, 8584.40it/s] 24%|██▍       | 97107/400000 [00:11<00:35, 8457.29it/s] 24%|██▍       | 97954/400000 [00:11<00:35, 8429.65it/s] 25%|██▍       | 98798/400000 [00:11<00:35, 8429.65it/s] 25%|██▍       | 99642/400000 [00:11<00:35, 8393.62it/s] 25%|██▌       | 100504/400000 [00:11<00:35, 8457.50it/s] 25%|██▌       | 101384/400000 [00:11<00:34, 8554.96it/s] 26%|██▌       | 102256/400000 [00:11<00:34, 8603.73it/s] 26%|██▌       | 103117/400000 [00:12<00:34, 8501.45it/s] 26%|██▌       | 103968/400000 [00:12<00:35, 8431.92it/s] 26%|██▌       | 104839/400000 [00:12<00:34, 8509.63it/s] 26%|██▋       | 105704/400000 [00:12<00:34, 8550.29it/s] 27%|██▋       | 106560/400000 [00:12<00:34, 8552.95it/s] 27%|██▋       | 107424/400000 [00:12<00:34, 8577.34it/s] 27%|██▋       | 108286/400000 [00:12<00:33, 8588.34it/s] 27%|██▋       | 109153/400000 [00:12<00:33, 8612.40it/s] 28%|██▊       | 110018/400000 [00:12<00:33, 8621.37it/s] 28%|██▊       | 110881/400000 [00:12<00:33, 8605.68it/s] 28%|██▊       | 111744/400000 [00:13<00:33, 8611.17it/s] 28%|██▊       | 112607/400000 [00:13<00:33, 8616.72it/s] 28%|██▊       | 113469/400000 [00:13<00:33, 8561.05it/s] 29%|██▊       | 114343/400000 [00:13<00:33, 8612.98it/s] 29%|██▉       | 115222/400000 [00:13<00:32, 8663.83it/s] 29%|██▉       | 116090/400000 [00:13<00:32, 8666.39it/s] 29%|██▉       | 116960/400000 [00:13<00:32, 8674.30it/s] 29%|██▉       | 117842/400000 [00:13<00:32, 8716.40it/s] 30%|██▉       | 118714/400000 [00:13<00:32, 8704.92it/s] 30%|██▉       | 119585/400000 [00:13<00:32, 8705.75it/s] 30%|███       | 120458/400000 [00:14<00:32, 8711.44it/s] 30%|███       | 121333/400000 [00:14<00:31, 8721.79it/s] 31%|███       | 122206/400000 [00:14<00:32, 8616.12it/s] 31%|███       | 123068/400000 [00:14<00:32, 8539.36it/s] 31%|███       | 123938/400000 [00:14<00:32, 8584.73it/s] 31%|███       | 124804/400000 [00:14<00:31, 8605.98it/s] 31%|███▏      | 125667/400000 [00:14<00:31, 8612.60it/s] 32%|███▏      | 126529/400000 [00:14<00:31, 8577.55it/s] 32%|███▏      | 127401/400000 [00:14<00:31, 8618.32it/s] 32%|███▏      | 128281/400000 [00:14<00:31, 8671.10it/s] 32%|███▏      | 129155/400000 [00:15<00:31, 8688.99it/s] 33%|███▎      | 130025/400000 [00:15<00:31, 8663.59it/s] 33%|███▎      | 130900/400000 [00:15<00:30, 8687.54it/s] 33%|███▎      | 131769/400000 [00:15<00:30, 8683.36it/s] 33%|███▎      | 132647/400000 [00:15<00:30, 8709.29it/s] 33%|███▎      | 133521/400000 [00:15<00:30, 8716.34it/s] 34%|███▎      | 134393/400000 [00:15<00:30, 8569.27it/s] 34%|███▍      | 135252/400000 [00:15<00:30, 8572.96it/s] 34%|███▍      | 136130/400000 [00:15<00:30, 8632.87it/s] 34%|███▍      | 136994/400000 [00:15<00:30, 8538.78it/s] 34%|███▍      | 137867/400000 [00:16<00:30, 8592.79it/s] 35%|███▍      | 138727/400000 [00:16<00:30, 8579.23it/s] 35%|███▍      | 139596/400000 [00:16<00:30, 8611.07it/s] 35%|███▌      | 140461/400000 [00:16<00:30, 8621.19it/s] 35%|███▌      | 141343/400000 [00:16<00:29, 8679.38it/s] 36%|███▌      | 142213/400000 [00:16<00:29, 8683.64it/s] 36%|███▌      | 143082/400000 [00:16<00:29, 8582.40it/s] 36%|███▌      | 143941/400000 [00:16<00:29, 8576.75it/s] 36%|███▌      | 144799/400000 [00:16<00:29, 8564.74it/s] 36%|███▋      | 145656/400000 [00:16<00:29, 8554.03it/s] 37%|███▋      | 146512/400000 [00:17<00:29, 8489.23it/s] 37%|███▋      | 147378/400000 [00:17<00:29, 8537.51it/s] 37%|███▋      | 148252/400000 [00:17<00:29, 8596.28it/s] 37%|███▋      | 149126/400000 [00:17<00:29, 8638.34it/s] 37%|███▋      | 149991/400000 [00:17<00:28, 8634.50it/s] 38%|███▊      | 150855/400000 [00:17<00:28, 8615.03it/s] 38%|███▊      | 151723/400000 [00:17<00:28, 8632.33it/s] 38%|███▊      | 152606/400000 [00:17<00:28, 8689.98it/s] 38%|███▊      | 153478/400000 [00:17<00:28, 8698.72it/s] 39%|███▊      | 154355/400000 [00:17<00:28, 8717.41it/s] 39%|███▉      | 155227/400000 [00:18<00:28, 8698.12it/s] 39%|███▉      | 156097/400000 [00:18<00:28, 8683.00it/s] 39%|███▉      | 156966/400000 [00:18<00:27, 8681.81it/s] 39%|███▉      | 157835/400000 [00:18<00:27, 8661.47it/s] 40%|███▉      | 158721/400000 [00:18<00:27, 8717.33it/s] 40%|███▉      | 159607/400000 [00:18<00:27, 8759.41it/s] 40%|████      | 160484/400000 [00:18<00:27, 8734.53it/s] 40%|████      | 161369/400000 [00:18<00:27, 8768.32it/s] 41%|████      | 162246/400000 [00:18<00:27, 8684.52it/s] 41%|████      | 163115/400000 [00:18<00:27, 8651.11it/s] 41%|████      | 163981/400000 [00:19<00:27, 8554.67it/s] 41%|████      | 164837/400000 [00:19<00:27, 8551.67it/s] 41%|████▏     | 165704/400000 [00:19<00:27, 8585.38it/s] 42%|████▏     | 166568/400000 [00:19<00:27, 8600.93it/s] 42%|████▏     | 167452/400000 [00:19<00:26, 8669.15it/s] 42%|████▏     | 168320/400000 [00:19<00:26, 8610.80it/s] 42%|████▏     | 169196/400000 [00:19<00:26, 8653.61it/s] 43%|████▎     | 170062/400000 [00:19<00:26, 8609.50it/s] 43%|████▎     | 170936/400000 [00:19<00:26, 8647.45it/s] 43%|████▎     | 171801/400000 [00:19<00:26, 8600.49it/s] 43%|████▎     | 172662/400000 [00:20<00:26, 8467.63it/s] 43%|████▎     | 173514/400000 [00:20<00:26, 8481.45it/s] 44%|████▎     | 174372/400000 [00:20<00:26, 8508.19it/s] 44%|████▍     | 175242/400000 [00:20<00:26, 8563.28it/s] 44%|████▍     | 176099/400000 [00:20<00:26, 8506.08it/s] 44%|████▍     | 176977/400000 [00:20<00:25, 8585.63it/s] 44%|████▍     | 177850/400000 [00:20<00:25, 8625.16it/s] 45%|████▍     | 178713/400000 [00:20<00:26, 8463.26it/s] 45%|████▍     | 179584/400000 [00:20<00:25, 8533.89it/s] 45%|████▌     | 180451/400000 [00:21<00:25, 8571.88it/s] 45%|████▌     | 181337/400000 [00:21<00:25, 8655.62it/s] 46%|████▌     | 182207/400000 [00:21<00:25, 8668.78it/s] 46%|████▌     | 183075/400000 [00:21<00:25, 8637.20it/s] 46%|████▌     | 183947/400000 [00:21<00:24, 8660.89it/s] 46%|████▌     | 184822/400000 [00:21<00:24, 8685.39it/s] 46%|████▋     | 185691/400000 [00:21<00:24, 8676.21it/s] 47%|████▋     | 186559/400000 [00:21<00:24, 8676.84it/s] 47%|████▋     | 187436/400000 [00:21<00:24, 8704.38it/s] 47%|████▋     | 188316/400000 [00:21<00:24, 8732.40it/s] 47%|████▋     | 189190/400000 [00:22<00:24, 8707.05it/s] 48%|████▊     | 190061/400000 [00:22<00:24, 8624.59it/s] 48%|████▊     | 190934/400000 [00:22<00:24, 8653.21it/s] 48%|████▊     | 191800/400000 [00:22<00:24, 8631.88it/s] 48%|████▊     | 192668/400000 [00:22<00:23, 8644.63it/s] 48%|████▊     | 193535/400000 [00:22<00:23, 8649.71it/s] 49%|████▊     | 194401/400000 [00:22<00:23, 8606.80it/s] 49%|████▉     | 195262/400000 [00:22<00:23, 8583.78it/s] 49%|████▉     | 196121/400000 [00:22<00:23, 8567.71it/s] 49%|████▉     | 196997/400000 [00:22<00:23, 8622.64it/s] 49%|████▉     | 197867/400000 [00:23<00:23, 8645.07it/s] 50%|████▉     | 198748/400000 [00:23<00:23, 8693.43it/s] 50%|████▉     | 199618/400000 [00:23<00:23, 8670.19it/s] 50%|█████     | 200486/400000 [00:23<00:23, 8670.42it/s] 50%|█████     | 201374/400000 [00:23<00:22, 8730.90it/s] 51%|█████     | 202248/400000 [00:23<00:22, 8733.18it/s] 51%|█████     | 203122/400000 [00:23<00:22, 8725.73it/s] 51%|█████     | 203995/400000 [00:23<00:22, 8710.89it/s] 51%|█████     | 204867/400000 [00:23<00:22, 8652.79it/s] 51%|█████▏    | 205734/400000 [00:23<00:22, 8657.00it/s] 52%|█████▏    | 206606/400000 [00:24<00:22, 8673.90it/s] 52%|█████▏    | 207478/400000 [00:24<00:22, 8686.18it/s] 52%|█████▏    | 208347/400000 [00:24<00:22, 8676.75it/s] 52%|█████▏    | 209215/400000 [00:24<00:22, 8623.57it/s] 53%|█████▎    | 210079/400000 [00:24<00:22, 8627.44it/s] 53%|█████▎    | 210942/400000 [00:24<00:21, 8608.43it/s] 53%|█████▎    | 211824/400000 [00:24<00:21, 8669.06it/s] 53%|█████▎    | 212700/400000 [00:24<00:21, 8696.11it/s] 53%|█████▎    | 213570/400000 [00:24<00:21, 8583.22it/s] 54%|█████▎    | 214445/400000 [00:24<00:21, 8631.26it/s] 54%|█████▍    | 215314/400000 [00:25<00:21, 8646.48it/s] 54%|█████▍    | 216180/400000 [00:25<00:21, 8650.25it/s] 54%|█████▍    | 217050/400000 [00:25<00:21, 8663.69it/s] 54%|█████▍    | 217917/400000 [00:25<00:21, 8644.43it/s] 55%|█████▍    | 218796/400000 [00:25<00:20, 8684.93it/s] 55%|█████▍    | 219665/400000 [00:25<00:20, 8631.51it/s] 55%|█████▌    | 220551/400000 [00:25<00:20, 8696.60it/s] 55%|█████▌    | 221428/400000 [00:25<00:20, 8715.75it/s] 56%|█████▌    | 222305/400000 [00:25<00:20, 8730.57it/s] 56%|█████▌    | 223184/400000 [00:25<00:20, 8746.42it/s] 56%|█████▌    | 224077/400000 [00:26<00:19, 8797.89it/s] 56%|█████▌    | 224976/400000 [00:26<00:19, 8853.34it/s] 56%|█████▋    | 225869/400000 [00:26<00:19, 8873.51it/s] 57%|█████▋    | 226757/400000 [00:26<00:19, 8749.05it/s] 57%|█████▋    | 227633/400000 [00:26<00:19, 8713.71it/s] 57%|█████▋    | 228505/400000 [00:26<00:19, 8703.78it/s] 57%|█████▋    | 229376/400000 [00:26<00:19, 8673.18it/s] 58%|█████▊    | 230244/400000 [00:26<00:19, 8610.56it/s] 58%|█████▊    | 231106/400000 [00:26<00:19, 8561.14it/s] 58%|█████▊    | 231963/400000 [00:26<00:19, 8541.71it/s] 58%|█████▊    | 232857/400000 [00:27<00:19, 8656.42it/s] 58%|█████▊    | 233724/400000 [00:27<00:19, 8527.82it/s] 59%|█████▊    | 234599/400000 [00:27<00:19, 8592.61it/s] 59%|█████▉    | 235459/400000 [00:27<00:19, 8544.04it/s] 59%|█████▉    | 236334/400000 [00:27<00:19, 8603.54it/s] 59%|█████▉    | 237207/400000 [00:27<00:18, 8639.50it/s] 60%|█████▉    | 238095/400000 [00:27<00:18, 8709.78it/s] 60%|█████▉    | 238974/400000 [00:27<00:18, 8732.16it/s] 60%|█████▉    | 239848/400000 [00:27<00:18, 8670.26it/s] 60%|██████    | 240718/400000 [00:27<00:18, 8678.56it/s] 60%|██████    | 241599/400000 [00:28<00:18, 8717.17it/s] 61%|██████    | 242471/400000 [00:28<00:18, 8671.30it/s] 61%|██████    | 243354/400000 [00:28<00:17, 8716.63it/s] 61%|██████    | 244226/400000 [00:28<00:18, 8491.15it/s] 61%|██████▏   | 245105/400000 [00:28<00:18, 8576.15it/s] 61%|██████▏   | 245989/400000 [00:28<00:17, 8653.04it/s] 62%|██████▏   | 246881/400000 [00:28<00:17, 8729.44it/s] 62%|██████▏   | 247759/400000 [00:28<00:17, 8742.98it/s] 62%|██████▏   | 248634/400000 [00:28<00:17, 8692.16it/s] 62%|██████▏   | 249504/400000 [00:28<00:17, 8689.23it/s] 63%|██████▎   | 250374/400000 [00:29<00:17, 8672.86it/s] 63%|██████▎   | 251256/400000 [00:29<00:17, 8713.61it/s] 63%|██████▎   | 252143/400000 [00:29<00:16, 8759.25it/s] 63%|██████▎   | 253020/400000 [00:29<00:16, 8695.77it/s] 63%|██████▎   | 253892/400000 [00:29<00:16, 8702.10it/s] 64%|██████▎   | 254763/400000 [00:29<00:16, 8689.19it/s] 64%|██████▍   | 255633/400000 [00:29<00:16, 8663.65it/s] 64%|██████▍   | 256504/400000 [00:29<00:16, 8676.83it/s] 64%|██████▍   | 257372/400000 [00:29<00:16, 8528.87it/s] 65%|██████▍   | 258238/400000 [00:29<00:16, 8566.80it/s] 65%|██████▍   | 259096/400000 [00:30<00:16, 8431.20it/s] 65%|██████▍   | 259940/400000 [00:30<00:16, 8431.60it/s] 65%|██████▌   | 260787/400000 [00:30<00:16, 8442.09it/s] 65%|██████▌   | 261633/400000 [00:30<00:16, 8444.79it/s] 66%|██████▌   | 262498/400000 [00:30<00:16, 8503.57it/s] 66%|██████▌   | 263372/400000 [00:30<00:15, 8572.68it/s] 66%|██████▌   | 264241/400000 [00:30<00:15, 8607.00it/s] 66%|██████▋   | 265102/400000 [00:30<00:15, 8571.78it/s] 66%|██████▋   | 265960/400000 [00:30<00:15, 8549.15it/s] 67%|██████▋   | 266816/400000 [00:30<00:15, 8469.81it/s] 67%|██████▋   | 267690/400000 [00:31<00:15, 8547.40it/s] 67%|██████▋   | 268546/400000 [00:31<00:15, 8460.08it/s] 67%|██████▋   | 269393/400000 [00:31<00:15, 8447.60it/s] 68%|██████▊   | 270252/400000 [00:31<00:15, 8487.87it/s] 68%|██████▊   | 271131/400000 [00:31<00:15, 8575.98it/s] 68%|██████▊   | 272011/400000 [00:31<00:14, 8639.94it/s] 68%|██████▊   | 272876/400000 [00:31<00:14, 8635.98it/s] 68%|██████▊   | 273740/400000 [00:31<00:14, 8629.68it/s] 69%|██████▊   | 274604/400000 [00:31<00:14, 8629.09it/s] 69%|██████▉   | 275468/400000 [00:31<00:14, 8611.79it/s] 69%|██████▉   | 276332/400000 [00:32<00:14, 8618.46it/s] 69%|██████▉   | 277194/400000 [00:32<00:14, 8600.51it/s] 70%|██████▉   | 278055/400000 [00:32<00:14, 8596.11it/s] 70%|██████▉   | 278915/400000 [00:32<00:14, 8577.47it/s] 70%|██████▉   | 279778/400000 [00:32<00:13, 8590.43it/s] 70%|███████   | 280642/400000 [00:32<00:13, 8603.42it/s] 70%|███████   | 281506/400000 [00:32<00:13, 8613.42it/s] 71%|███████   | 282374/400000 [00:32<00:13, 8629.69it/s] 71%|███████   | 283237/400000 [00:32<00:13, 8617.02it/s] 71%|███████   | 284099/400000 [00:32<00:13, 8599.29it/s] 71%|███████   | 284964/400000 [00:33<00:13, 8611.79it/s] 71%|███████▏  | 285826/400000 [00:33<00:13, 8606.20it/s] 72%|███████▏  | 286702/400000 [00:33<00:13, 8651.55it/s] 72%|███████▏  | 287587/400000 [00:33<00:12, 8709.24it/s] 72%|███████▏  | 288459/400000 [00:33<00:12, 8690.93it/s] 72%|███████▏  | 289338/400000 [00:33<00:12, 8719.26it/s] 73%|███████▎  | 290218/400000 [00:33<00:12, 8741.41it/s] 73%|███████▎  | 291093/400000 [00:33<00:12, 8722.32it/s] 73%|███████▎  | 291981/400000 [00:33<00:12, 8766.46it/s] 73%|███████▎  | 292858/400000 [00:34<00:12, 8723.92it/s] 73%|███████▎  | 293736/400000 [00:34<00:12, 8738.81it/s] 74%|███████▎  | 294610/400000 [00:34<00:12, 8733.56it/s] 74%|███████▍  | 295488/400000 [00:34<00:11, 8745.04it/s] 74%|███████▍  | 296366/400000 [00:34<00:11, 8753.55it/s] 74%|███████▍  | 297243/400000 [00:34<00:11, 8755.80it/s] 75%|███████▍  | 298135/400000 [00:34<00:11, 8803.42it/s] 75%|███████▍  | 299021/400000 [00:34<00:11, 8819.48it/s] 75%|███████▍  | 299918/400000 [00:34<00:11, 8862.12it/s] 75%|███████▌  | 300815/400000 [00:34<00:11, 8892.42it/s] 75%|███████▌  | 301705/400000 [00:35<00:11, 8682.24it/s] 76%|███████▌  | 302575/400000 [00:35<00:11, 8527.31it/s] 76%|███████▌  | 303447/400000 [00:35<00:11, 8581.07it/s] 76%|███████▌  | 304307/400000 [00:35<00:11, 8498.24it/s] 76%|███████▋  | 305158/400000 [00:35<00:11, 8460.74it/s] 77%|███████▋  | 306005/400000 [00:35<00:11, 8436.57it/s] 77%|███████▋  | 306866/400000 [00:35<00:10, 8485.63it/s] 77%|███████▋  | 307715/400000 [00:35<00:10, 8457.27it/s] 77%|███████▋  | 308569/400000 [00:35<00:10, 8481.82it/s] 77%|███████▋  | 309448/400000 [00:35<00:10, 8569.49it/s] 78%|███████▊  | 310306/400000 [00:36<00:10, 8534.74it/s] 78%|███████▊  | 311160/400000 [00:36<00:10, 8497.96it/s] 78%|███████▊  | 312011/400000 [00:36<00:10, 8476.71it/s] 78%|███████▊  | 312864/400000 [00:36<00:10, 8491.45it/s] 78%|███████▊  | 313714/400000 [00:36<00:10, 8384.88it/s] 79%|███████▊  | 314562/400000 [00:36<00:10, 8411.31it/s] 79%|███████▉  | 315425/400000 [00:36<00:09, 8473.38it/s] 79%|███████▉  | 316284/400000 [00:36<00:09, 8507.34it/s] 79%|███████▉  | 317135/400000 [00:36<00:09, 8466.70it/s] 79%|███████▉  | 317982/400000 [00:36<00:09, 8456.32it/s] 80%|███████▉  | 318842/400000 [00:37<00:09, 8497.94it/s] 80%|███████▉  | 319696/400000 [00:37<00:09, 8510.17it/s] 80%|████████  | 320548/400000 [00:37<00:09, 8511.28it/s] 80%|████████  | 321408/400000 [00:37<00:09, 8536.18it/s] 81%|████████  | 322269/400000 [00:37<00:09, 8557.36it/s] 81%|████████  | 323125/400000 [00:37<00:08, 8553.94it/s] 81%|████████  | 323983/400000 [00:37<00:08, 8561.69it/s] 81%|████████  | 324854/400000 [00:37<00:08, 8603.43it/s] 81%|████████▏ | 325715/400000 [00:37<00:08, 8541.46it/s] 82%|████████▏ | 326601/400000 [00:37<00:08, 8633.22it/s] 82%|████████▏ | 327481/400000 [00:38<00:08, 8681.72it/s] 82%|████████▏ | 328350/400000 [00:38<00:08, 8649.85it/s] 82%|████████▏ | 329216/400000 [00:38<00:08, 8532.79it/s] 83%|████████▎ | 330072/400000 [00:38<00:08, 8539.70it/s] 83%|████████▎ | 330932/400000 [00:38<00:08, 8557.58it/s] 83%|████████▎ | 331789/400000 [00:38<00:07, 8538.17it/s] 83%|████████▎ | 332651/400000 [00:38<00:07, 8561.37it/s] 83%|████████▎ | 333522/400000 [00:38<00:07, 8605.10it/s] 84%|████████▎ | 334399/400000 [00:38<00:07, 8651.60it/s] 84%|████████▍ | 335265/400000 [00:38<00:07, 8646.85it/s] 84%|████████▍ | 336130/400000 [00:39<00:07, 8621.25it/s] 84%|████████▍ | 337006/400000 [00:39<00:07, 8661.91it/s] 84%|████████▍ | 337876/400000 [00:39<00:07, 8670.35it/s] 85%|████████▍ | 338753/400000 [00:39<00:07, 8698.82it/s] 85%|████████▍ | 339627/400000 [00:39<00:06, 8710.88it/s] 85%|████████▌ | 340499/400000 [00:39<00:06, 8642.71it/s] 85%|████████▌ | 341364/400000 [00:39<00:06, 8612.25it/s] 86%|████████▌ | 342234/400000 [00:39<00:06, 8636.62it/s] 86%|████████▌ | 343109/400000 [00:39<00:06, 8667.55it/s] 86%|████████▌ | 343977/400000 [00:39<00:06, 8671.11it/s] 86%|████████▌ | 344845/400000 [00:40<00:06, 8620.48it/s] 86%|████████▋ | 345710/400000 [00:40<00:06, 8629.26it/s] 87%|████████▋ | 346582/400000 [00:40<00:06, 8656.02it/s] 87%|████████▋ | 347448/400000 [00:40<00:06, 8637.12it/s] 87%|████████▋ | 348317/400000 [00:40<00:05, 8650.85it/s] 87%|████████▋ | 349183/400000 [00:40<00:05, 8634.24it/s] 88%|████████▊ | 350051/400000 [00:40<00:05, 8646.53it/s] 88%|████████▊ | 350928/400000 [00:40<00:05, 8680.95it/s] 88%|████████▊ | 351797/400000 [00:40<00:05, 8615.89it/s] 88%|████████▊ | 352674/400000 [00:40<00:05, 8658.73it/s] 88%|████████▊ | 353544/400000 [00:41<00:05, 8669.87it/s] 89%|████████▊ | 354415/400000 [00:41<00:05, 8680.24it/s] 89%|████████▉ | 355284/400000 [00:41<00:05, 8670.55it/s] 89%|████████▉ | 356160/400000 [00:41<00:05, 8696.86it/s] 89%|████████▉ | 357036/400000 [00:41<00:04, 8714.01it/s] 89%|████████▉ | 357908/400000 [00:41<00:04, 8531.36it/s] 90%|████████▉ | 358776/400000 [00:41<00:04, 8573.99it/s] 90%|████████▉ | 359635/400000 [00:41<00:04, 8529.43it/s] 90%|█████████ | 360502/400000 [00:41<00:04, 8570.19it/s] 90%|█████████ | 361386/400000 [00:41<00:04, 8649.32it/s] 91%|█████████ | 362252/400000 [00:42<00:04, 8651.90it/s] 91%|█████████ | 363119/400000 [00:42<00:04, 8655.69it/s] 91%|█████████ | 363985/400000 [00:42<00:04, 8611.01it/s] 91%|█████████ | 364855/400000 [00:42<00:04, 8635.41it/s] 91%|█████████▏| 365722/400000 [00:42<00:03, 8645.39it/s] 92%|█████████▏| 366587/400000 [00:42<00:03, 8637.53it/s] 92%|█████████▏| 367451/400000 [00:42<00:03, 8615.18it/s] 92%|█████████▏| 368321/400000 [00:42<00:03, 8639.90it/s] 92%|█████████▏| 369187/400000 [00:42<00:03, 8644.30it/s] 93%|█████████▎| 370055/400000 [00:42<00:03, 8654.72it/s] 93%|█████████▎| 370921/400000 [00:43<00:03, 8634.41it/s] 93%|█████████▎| 371785/400000 [00:43<00:03, 8612.88it/s] 93%|█████████▎| 372647/400000 [00:43<00:03, 8588.24it/s] 93%|█████████▎| 373506/400000 [00:43<00:03, 8540.80it/s] 94%|█████████▎| 374368/400000 [00:43<00:02, 8563.36it/s] 94%|█████████▍| 375234/400000 [00:43<00:02, 8592.08it/s] 94%|█████████▍| 376096/400000 [00:43<00:02, 8599.53it/s] 94%|█████████▍| 376963/400000 [00:43<00:02, 8619.69it/s] 94%|█████████▍| 377836/400000 [00:43<00:02, 8652.41it/s] 95%|█████████▍| 378704/400000 [00:43<00:02, 8659.82it/s] 95%|█████████▍| 379572/400000 [00:44<00:02, 8664.54it/s] 95%|█████████▌| 380439/400000 [00:44<00:02, 8645.16it/s] 95%|█████████▌| 381304/400000 [00:44<00:02, 8621.11it/s] 96%|█████████▌| 382182/400000 [00:44<00:02, 8665.41it/s] 96%|█████████▌| 383050/400000 [00:44<00:01, 8668.86it/s] 96%|█████████▌| 383917/400000 [00:44<00:01, 8658.50it/s] 96%|█████████▌| 384783/400000 [00:44<00:01, 8607.88it/s] 96%|█████████▋| 385648/400000 [00:44<00:01, 8620.09it/s] 97%|█████████▋| 386511/400000 [00:44<00:01, 8618.56it/s] 97%|█████████▋| 387378/400000 [00:44<00:01, 8633.14it/s] 97%|█████████▋| 388242/400000 [00:45<00:01, 8333.56it/s] 97%|█████████▋| 389106/400000 [00:45<00:01, 8420.45it/s] 97%|█████████▋| 389970/400000 [00:45<00:01, 8485.04it/s] 98%|█████████▊| 390846/400000 [00:45<00:01, 8565.21it/s] 98%|█████████▊| 391719/400000 [00:45<00:00, 8612.38it/s] 98%|█████████▊| 392592/400000 [00:45<00:00, 8645.17it/s] 98%|█████████▊| 393466/400000 [00:45<00:00, 8671.60it/s] 99%|█████████▊| 394349/400000 [00:45<00:00, 8718.07it/s] 99%|█████████▉| 395222/400000 [00:45<00:00, 8708.87it/s] 99%|█████████▉| 396094/400000 [00:45<00:00, 8704.94it/s] 99%|█████████▉| 396965/400000 [00:46<00:00, 8649.42it/s] 99%|█████████▉| 397836/400000 [00:46<00:00, 8666.44it/s]100%|█████████▉| 398709/400000 [00:46<00:00, 8683.93it/s]100%|█████████▉| 399587/400000 [00:46<00:00, 8712.05it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8612.89it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb72ff5cd68> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010934889060744912 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.010936846501851161 	 Accuracy: 72

  model saves at 72% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15701 out of table with 15699 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15701 out of table with 15699 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-21 13:01:24.278051: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-21 13:01:24.282432: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095070000 Hz
2020-05-21 13:01:24.282569: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559601951990 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 13:01:24.282584: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fb6db5b7fd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.2373 - accuracy: 0.5280
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5056 - accuracy: 0.5105 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.4417 - accuracy: 0.5147
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.4136 - accuracy: 0.5165
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.4213 - accuracy: 0.5160
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5465 - accuracy: 0.5078
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5768 - accuracy: 0.5059
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5459 - accuracy: 0.5079
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5491 - accuracy: 0.5077
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5302 - accuracy: 0.5089
11000/25000 [============>.................] - ETA: 3s - loss: 7.5872 - accuracy: 0.5052
12000/25000 [=============>................] - ETA: 3s - loss: 7.6181 - accuracy: 0.5032
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6171 - accuracy: 0.5032
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6151 - accuracy: 0.5034
15000/25000 [=================>............] - ETA: 2s - loss: 7.6094 - accuracy: 0.5037
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6197 - accuracy: 0.5031
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6405 - accuracy: 0.5017
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6334 - accuracy: 0.5022
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6190 - accuracy: 0.5031
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6260 - accuracy: 0.5027
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6535 - accuracy: 0.5009
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6492 - accuracy: 0.5011
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6493 - accuracy: 0.5011
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6519 - accuracy: 0.5010
25000/25000 [==============================] - 7s 279us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fb695602860> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fb68c221da0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4008 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.3788 - val_crf_viterbi_accuracy: 0.0133

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

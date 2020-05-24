
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'dbbd1e3505a2b3043e7688c1260e13ddacd09d91', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f3505dd3eb8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-24 20:18:05.655545
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-24 20:18:05.659063
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-24 20:18:05.661746
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-24 20:18:05.664302
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f3511b9d320> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354121.2500
Epoch 2/10

1/1 [==============================] - 0s 99ms/step - loss: 262331.8438
Epoch 3/10

1/1 [==============================] - 0s 93ms/step - loss: 161954.8125
Epoch 4/10

1/1 [==============================] - 0s 89ms/step - loss: 89878.9219
Epoch 5/10

1/1 [==============================] - 0s 93ms/step - loss: 48204.5156
Epoch 6/10

1/1 [==============================] - 0s 88ms/step - loss: 27420.4434
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 17176.9180
Epoch 8/10

1/1 [==============================] - 0s 89ms/step - loss: 11718.7549
Epoch 9/10

1/1 [==============================] - 0s 93ms/step - loss: 8544.9951
Epoch 10/10

1/1 [==============================] - 0s 93ms/step - loss: 6588.7441

  #### Inference Need return ypred, ytrue ######################### 
[[-1.0664891  -0.5647168  -0.26242304 -1.4469792   0.6047671  -1.4992859
  -0.36301553 -0.30321324 -1.5630342  -0.2572859   1.1247206  -1.2211766
   1.5980511   0.10323906  1.2591354  -1.6676645  -0.08011482 -0.2531232
   1.3127183   0.6660288  -0.6837356  -1.1753958   1.2238737  -0.02809089
   1.1240206  -0.41278177  1.1526693  -0.7033573   1.0061039   0.68516016
  -1.5088965   1.197498    0.90887123  0.4572113   0.6564963  -1.516618
   1.7421956   0.55307555  0.1405443  -0.33350456  0.28946584  1.7750278
   0.87676734  1.1276855  -1.6210533   0.4115728   1.0635532  -0.85444653
   1.5956942  -0.19686317  0.29410696  0.5243664  -0.52599144 -1.5802108
  -1.4974991   0.76631546 -1.837667    0.879465   -1.8252537   1.2493483
   0.92300034 -0.7141334  -0.12702134 -2.328268   -0.07679284 -1.4250331
   0.88541585 -0.5220367   1.7141181   0.82222056 -0.76266694  1.7025326
  -0.22296959  0.8878602   0.11111331 -1.1293085  -0.80030346 -0.8980468
  -1.3455656  -0.49285674 -0.27045038 -0.41665146  0.3047343   2.015329
   0.66977113 -0.5647128   1.4750313   0.6486017  -0.9667773  -0.48930895
  -0.9278427   0.19380051 -0.703687   -1.2896414   0.77747864 -0.76586777
   0.88597435 -0.34966022 -0.22879438 -1.8691933  -1.0996342   1.608552
   0.67125946 -0.37894917 -0.88930535 -1.4814444  -1.4042653   0.6342014
   1.0591274  -0.7027169   0.35215828 -0.31129587 -1.4066749  -0.8056818
   0.22143388 -1.6282198  -0.6390507  -1.1987677   1.2522342   0.80445665
   0.47096536  6.360672    7.3861704   8.320159    8.17627     7.4662976
   9.14058     7.955407    9.127804    9.156531    9.321829    9.985733
   8.165415    8.837562    7.936877    8.734181    6.9271817   7.248482
   8.413622    7.8455067   8.103955    7.7150807   7.7337084   8.713093
   7.4981785   7.706914    8.396208    8.783039    8.7784395   7.8912745
   6.607551    7.7713833   6.418819    6.7761774   7.736153    7.758979
   9.619848    8.263205    9.65594     7.4718914   8.104621    7.302083
   7.9451995   9.260726   10.03491     8.124373    8.420598    7.283254
   8.996241    8.614524    8.288471    8.575758    7.567863    8.335921
   9.036536    6.264589    8.857609    7.837659   10.114811    5.085
   2.2498548   1.3482802   1.7305291   0.8361861   0.820845    2.1768098
   1.8737385   0.1372878   2.2055674   2.8185792   1.8637958   1.5250368
   1.1589179   1.0451553   0.7167455   1.9543526   0.53369975  0.31203616
   2.6869326   1.4306363   0.49108994  2.2558274   1.5814042   2.4519138
   2.5136533   1.1796219   0.25815707  1.1436131   2.7604566   1.0389268
   1.9108481   0.82113624  0.24329245  1.6551733   0.10634565  0.67339367
   2.5608034   2.6370049   0.9827112   0.5973948   0.44910514  1.0387511
   1.7799276   0.7759914   0.48591626  0.9832833   2.073578    3.3072586
   0.56238025  1.9144444   0.22655404  0.77421874  1.6734269   0.76671916
   0.61287785  0.19618094  1.7084267   2.002431    1.2410471   1.2464077
   1.4333539   1.6615558   0.74555594  0.48127162  1.5741742   0.6973202
   3.2613964   0.8930608   0.9403119   2.619655    1.5112092   3.286625
   0.5553882   1.2698187   1.1679769   0.49319196  1.1369276   1.786841
   0.8370605   0.12422431  1.059874    0.8129893   1.116233    1.4679598
   0.2133888   0.73150885  0.83929074  2.9563608   0.48933583  1.2588906
   0.8362497   0.26210868  1.3123263   1.0291926   1.7494767   0.56205595
   0.8466347   0.38882113  0.94813746  0.73167753  0.24263293  0.5578067
   1.9474108   0.763772    0.43647826  0.1442461   2.1232157   0.2251324
   1.3000667   0.5439786   1.2848591   0.78508306  0.46836555  0.45302236
   1.3205156   0.10051608  1.5772315   2.3630695   0.5618006   1.2953238
   0.23640138  8.464473    8.426526    8.219123    8.85714     8.154855
   9.788132    8.398409    6.8677664   8.360241    9.3425     10.228123
   8.371042    8.498001    8.297703    8.807337    6.664288    8.481988
   9.408699    7.478601    8.053719    7.758379    6.650344    8.854239
   8.193636    8.278496    8.539132    8.486193    8.165517    8.178592
   7.61582     8.360912    6.901868    8.27551     9.4946375   7.987862
   7.816045    9.689162    8.985724    6.891304    8.387943    7.528691
   9.637199    7.8696537   7.942317    9.374409    8.801035    8.788258
   8.780845    6.969597    9.297491    9.008581    8.930497    8.781251
   8.391272    9.901896    8.173382    7.559012    7.9340625   9.462892
  -4.2035103  -6.0480127   9.404826  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-24 20:18:13.529428
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.6915
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-24 20:18:13.533154
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8805.11
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-24 20:18:13.536022
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.9354
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-24 20:18:13.538725
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -787.56
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139865349050832
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139864407089672
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139864407090176
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139864407090680
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139864407091184
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139864407091688

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f3505dd39e8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.428737
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.395124
grad_step = 000002, loss = 0.371189
grad_step = 000003, loss = 0.347844
grad_step = 000004, loss = 0.327727
grad_step = 000005, loss = 0.317901
grad_step = 000006, loss = 0.310421
grad_step = 000007, loss = 0.302685
grad_step = 000008, loss = 0.292037
grad_step = 000009, loss = 0.281728
grad_step = 000010, loss = 0.273922
grad_step = 000011, loss = 0.267554
grad_step = 000012, loss = 0.260771
grad_step = 000013, loss = 0.253067
grad_step = 000014, loss = 0.244785
grad_step = 000015, loss = 0.236465
grad_step = 000016, loss = 0.228504
grad_step = 000017, loss = 0.218520
grad_step = 000018, loss = 0.208085
grad_step = 000019, loss = 0.198002
grad_step = 000020, loss = 0.188686
grad_step = 000021, loss = 0.179483
grad_step = 000022, loss = 0.171142
grad_step = 000023, loss = 0.163787
grad_step = 000024, loss = 0.157313
grad_step = 000025, loss = 0.151466
grad_step = 000026, loss = 0.145911
grad_step = 000027, loss = 0.140197
grad_step = 000028, loss = 0.134180
grad_step = 000029, loss = 0.128161
grad_step = 000030, loss = 0.122590
grad_step = 000031, loss = 0.117495
grad_step = 000032, loss = 0.112587
grad_step = 000033, loss = 0.107882
grad_step = 000034, loss = 0.103267
grad_step = 000035, loss = 0.098531
grad_step = 000036, loss = 0.093760
grad_step = 000037, loss = 0.089250
grad_step = 000038, loss = 0.085171
grad_step = 000039, loss = 0.081458
grad_step = 000040, loss = 0.077879
grad_step = 000041, loss = 0.074229
grad_step = 000042, loss = 0.070517
grad_step = 000043, loss = 0.066900
grad_step = 000044, loss = 0.063552
grad_step = 000045, loss = 0.060537
grad_step = 000046, loss = 0.057689
grad_step = 000047, loss = 0.054766
grad_step = 000048, loss = 0.051804
grad_step = 000049, loss = 0.049064
grad_step = 000050, loss = 0.046618
grad_step = 000051, loss = 0.044303
grad_step = 000052, loss = 0.042018
grad_step = 000053, loss = 0.039798
grad_step = 000054, loss = 0.037687
grad_step = 000055, loss = 0.035694
grad_step = 000056, loss = 0.033813
grad_step = 000057, loss = 0.032022
grad_step = 000058, loss = 0.030298
grad_step = 000059, loss = 0.028644
grad_step = 000060, loss = 0.027064
grad_step = 000061, loss = 0.025582
grad_step = 000062, loss = 0.024218
grad_step = 000063, loss = 0.022927
grad_step = 000064, loss = 0.021660
grad_step = 000065, loss = 0.020439
grad_step = 000066, loss = 0.019310
grad_step = 000067, loss = 0.018268
grad_step = 000068, loss = 0.017263
grad_step = 000069, loss = 0.016288
grad_step = 000070, loss = 0.015369
grad_step = 000071, loss = 0.014523
grad_step = 000072, loss = 0.013730
grad_step = 000073, loss = 0.012972
grad_step = 000074, loss = 0.012254
grad_step = 000075, loss = 0.011579
grad_step = 000076, loss = 0.010945
grad_step = 000077, loss = 0.010352
grad_step = 000078, loss = 0.009795
grad_step = 000079, loss = 0.009264
grad_step = 000080, loss = 0.008761
grad_step = 000081, loss = 0.008297
grad_step = 000082, loss = 0.007868
grad_step = 000083, loss = 0.007462
grad_step = 000084, loss = 0.007076
grad_step = 000085, loss = 0.006719
grad_step = 000086, loss = 0.006390
grad_step = 000087, loss = 0.006079
grad_step = 000088, loss = 0.005786
grad_step = 000089, loss = 0.005513
grad_step = 000090, loss = 0.005261
grad_step = 000091, loss = 0.005027
grad_step = 000092, loss = 0.004809
grad_step = 000093, loss = 0.004605
grad_step = 000094, loss = 0.004415
grad_step = 000095, loss = 0.004240
grad_step = 000096, loss = 0.004079
grad_step = 000097, loss = 0.003927
grad_step = 000098, loss = 0.003785
grad_step = 000099, loss = 0.003656
grad_step = 000100, loss = 0.003537
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003425
grad_step = 000102, loss = 0.003323
grad_step = 000103, loss = 0.003230
grad_step = 000104, loss = 0.003152
grad_step = 000105, loss = 0.003085
grad_step = 000106, loss = 0.003025
grad_step = 000107, loss = 0.002939
grad_step = 000108, loss = 0.002858
grad_step = 000109, loss = 0.002806
grad_step = 000110, loss = 0.002771
grad_step = 000111, loss = 0.002731
grad_step = 000112, loss = 0.002668
grad_step = 000113, loss = 0.002614
grad_step = 000114, loss = 0.002582
grad_step = 000115, loss = 0.002564
grad_step = 000116, loss = 0.002548
grad_step = 000117, loss = 0.002515
grad_step = 000118, loss = 0.002473
grad_step = 000119, loss = 0.002434
grad_step = 000120, loss = 0.002410
grad_step = 000121, loss = 0.002400
grad_step = 000122, loss = 0.002403
grad_step = 000123, loss = 0.002422
grad_step = 000124, loss = 0.002400
grad_step = 000125, loss = 0.002368
grad_step = 000126, loss = 0.002316
grad_step = 000127, loss = 0.002306
grad_step = 000128, loss = 0.002330
grad_step = 000129, loss = 0.002322
grad_step = 000130, loss = 0.002307
grad_step = 000131, loss = 0.002267
grad_step = 000132, loss = 0.002266
grad_step = 000133, loss = 0.002292
grad_step = 000134, loss = 0.002283
grad_step = 000135, loss = 0.002271
grad_step = 000136, loss = 0.002238
grad_step = 000137, loss = 0.002227
grad_step = 000138, loss = 0.002232
grad_step = 000139, loss = 0.002244
grad_step = 000140, loss = 0.002266
grad_step = 000141, loss = 0.002250
grad_step = 000142, loss = 0.002233
grad_step = 000143, loss = 0.002209
grad_step = 000144, loss = 0.002209
grad_step = 000145, loss = 0.002226
grad_step = 000146, loss = 0.002222
grad_step = 000147, loss = 0.002213
grad_step = 000148, loss = 0.002197
grad_step = 000149, loss = 0.002196
grad_step = 000150, loss = 0.002209
grad_step = 000151, loss = 0.002216
grad_step = 000152, loss = 0.002231
grad_step = 000153, loss = 0.002214
grad_step = 000154, loss = 0.002202
grad_step = 000155, loss = 0.002187
grad_step = 000156, loss = 0.002190
grad_step = 000157, loss = 0.002200
grad_step = 000158, loss = 0.002193
grad_step = 000159, loss = 0.002183
grad_step = 000160, loss = 0.002179
grad_step = 000161, loss = 0.002185
grad_step = 000162, loss = 0.002193
grad_step = 000163, loss = 0.002195
grad_step = 000164, loss = 0.002197
grad_step = 000165, loss = 0.002187
grad_step = 000166, loss = 0.002180
grad_step = 000167, loss = 0.002172
grad_step = 000168, loss = 0.002171
grad_step = 000169, loss = 0.002174
grad_step = 000170, loss = 0.002176
grad_step = 000171, loss = 0.002176
grad_step = 000172, loss = 0.002172
grad_step = 000173, loss = 0.002168
grad_step = 000174, loss = 0.002164
grad_step = 000175, loss = 0.002163
grad_step = 000176, loss = 0.002164
grad_step = 000177, loss = 0.002165
grad_step = 000178, loss = 0.002168
grad_step = 000179, loss = 0.002170
grad_step = 000180, loss = 0.002174
grad_step = 000181, loss = 0.002173
grad_step = 000182, loss = 0.002173
grad_step = 000183, loss = 0.002165
grad_step = 000184, loss = 0.002158
grad_step = 000185, loss = 0.002152
grad_step = 000186, loss = 0.002151
grad_step = 000187, loss = 0.002154
grad_step = 000188, loss = 0.002155
grad_step = 000189, loss = 0.002155
grad_step = 000190, loss = 0.002152
grad_step = 000191, loss = 0.002149
grad_step = 000192, loss = 0.002146
grad_step = 000193, loss = 0.002143
grad_step = 000194, loss = 0.002141
grad_step = 000195, loss = 0.002139
grad_step = 000196, loss = 0.002138
grad_step = 000197, loss = 0.002136
grad_step = 000198, loss = 0.002135
grad_step = 000199, loss = 0.002133
grad_step = 000200, loss = 0.002132
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002130
grad_step = 000202, loss = 0.002129
grad_step = 000203, loss = 0.002127
grad_step = 000204, loss = 0.002126
grad_step = 000205, loss = 0.002125
grad_step = 000206, loss = 0.002124
grad_step = 000207, loss = 0.002124
grad_step = 000208, loss = 0.002128
grad_step = 000209, loss = 0.002139
grad_step = 000210, loss = 0.002180
grad_step = 000211, loss = 0.002222
grad_step = 000212, loss = 0.002324
grad_step = 000213, loss = 0.002174
grad_step = 000214, loss = 0.002129
grad_step = 000215, loss = 0.002214
grad_step = 000216, loss = 0.002149
grad_step = 000217, loss = 0.002123
grad_step = 000218, loss = 0.002173
grad_step = 000219, loss = 0.002133
grad_step = 000220, loss = 0.002106
grad_step = 000221, loss = 0.002135
grad_step = 000222, loss = 0.002136
grad_step = 000223, loss = 0.002122
grad_step = 000224, loss = 0.002097
grad_step = 000225, loss = 0.002098
grad_step = 000226, loss = 0.002116
grad_step = 000227, loss = 0.002123
grad_step = 000228, loss = 0.002122
grad_step = 000229, loss = 0.002098
grad_step = 000230, loss = 0.002084
grad_step = 000231, loss = 0.002087
grad_step = 000232, loss = 0.002095
grad_step = 000233, loss = 0.002096
grad_step = 000234, loss = 0.002082
grad_step = 000235, loss = 0.002075
grad_step = 000236, loss = 0.002078
grad_step = 000237, loss = 0.002080
grad_step = 000238, loss = 0.002076
grad_step = 000239, loss = 0.002068
grad_step = 000240, loss = 0.002065
grad_step = 000241, loss = 0.002068
grad_step = 000242, loss = 0.002069
grad_step = 000243, loss = 0.002066
grad_step = 000244, loss = 0.002060
grad_step = 000245, loss = 0.002056
grad_step = 000246, loss = 0.002055
grad_step = 000247, loss = 0.002057
grad_step = 000248, loss = 0.002062
grad_step = 000249, loss = 0.002066
grad_step = 000250, loss = 0.002065
grad_step = 000251, loss = 0.002060
grad_step = 000252, loss = 0.002053
grad_step = 000253, loss = 0.002050
grad_step = 000254, loss = 0.002050
grad_step = 000255, loss = 0.002045
grad_step = 000256, loss = 0.002040
grad_step = 000257, loss = 0.002036
grad_step = 000258, loss = 0.002036
grad_step = 000259, loss = 0.002039
grad_step = 000260, loss = 0.002042
grad_step = 000261, loss = 0.002043
grad_step = 000262, loss = 0.002040
grad_step = 000263, loss = 0.002038
grad_step = 000264, loss = 0.002035
grad_step = 000265, loss = 0.002035
grad_step = 000266, loss = 0.002036
grad_step = 000267, loss = 0.002039
grad_step = 000268, loss = 0.002044
grad_step = 000269, loss = 0.002052
grad_step = 000270, loss = 0.002059
grad_step = 000271, loss = 0.002044
grad_step = 000272, loss = 0.002028
grad_step = 000273, loss = 0.002023
grad_step = 000274, loss = 0.002026
grad_step = 000275, loss = 0.002031
grad_step = 000276, loss = 0.002024
grad_step = 000277, loss = 0.002015
grad_step = 000278, loss = 0.002011
grad_step = 000279, loss = 0.002018
grad_step = 000280, loss = 0.002029
grad_step = 000281, loss = 0.002031
grad_step = 000282, loss = 0.002032
grad_step = 000283, loss = 0.002040
grad_step = 000284, loss = 0.002079
grad_step = 000285, loss = 0.002101
grad_step = 000286, loss = 0.002131
grad_step = 000287, loss = 0.002049
grad_step = 000288, loss = 0.002003
grad_step = 000289, loss = 0.002039
grad_step = 000290, loss = 0.002051
grad_step = 000291, loss = 0.002016
grad_step = 000292, loss = 0.002010
grad_step = 000293, loss = 0.002031
grad_step = 000294, loss = 0.002021
grad_step = 000295, loss = 0.001997
grad_step = 000296, loss = 0.002010
grad_step = 000297, loss = 0.002029
grad_step = 000298, loss = 0.002014
grad_step = 000299, loss = 0.001995
grad_step = 000300, loss = 0.001988
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001994
grad_step = 000302, loss = 0.002006
grad_step = 000303, loss = 0.002012
grad_step = 000304, loss = 0.002015
grad_step = 000305, loss = 0.002006
grad_step = 000306, loss = 0.002000
grad_step = 000307, loss = 0.001992
grad_step = 000308, loss = 0.001990
grad_step = 000309, loss = 0.001985
grad_step = 000310, loss = 0.001982
grad_step = 000311, loss = 0.001979
grad_step = 000312, loss = 0.001979
grad_step = 000313, loss = 0.001981
grad_step = 000314, loss = 0.001983
grad_step = 000315, loss = 0.001984
grad_step = 000316, loss = 0.001976
grad_step = 000317, loss = 0.001969
grad_step = 000318, loss = 0.001966
grad_step = 000319, loss = 0.001967
grad_step = 000320, loss = 0.001970
grad_step = 000321, loss = 0.001971
grad_step = 000322, loss = 0.001971
grad_step = 000323, loss = 0.001967
grad_step = 000324, loss = 0.001964
grad_step = 000325, loss = 0.001961
grad_step = 000326, loss = 0.001961
grad_step = 000327, loss = 0.001961
grad_step = 000328, loss = 0.001962
grad_step = 000329, loss = 0.001962
grad_step = 000330, loss = 0.001962
grad_step = 000331, loss = 0.001960
grad_step = 000332, loss = 0.001959
grad_step = 000333, loss = 0.001958
grad_step = 000334, loss = 0.001959
grad_step = 000335, loss = 0.001960
grad_step = 000336, loss = 0.001964
grad_step = 000337, loss = 0.001964
grad_step = 000338, loss = 0.001965
grad_step = 000339, loss = 0.001957
grad_step = 000340, loss = 0.001951
grad_step = 000341, loss = 0.001945
grad_step = 000342, loss = 0.001941
grad_step = 000343, loss = 0.001939
grad_step = 000344, loss = 0.001938
grad_step = 000345, loss = 0.001938
grad_step = 000346, loss = 0.001938
grad_step = 000347, loss = 0.001938
grad_step = 000348, loss = 0.001938
grad_step = 000349, loss = 0.001939
grad_step = 000350, loss = 0.001942
grad_step = 000351, loss = 0.001949
grad_step = 000352, loss = 0.001969
grad_step = 000353, loss = 0.001989
grad_step = 000354, loss = 0.002040
grad_step = 000355, loss = 0.002018
grad_step = 000356, loss = 0.001995
grad_step = 000357, loss = 0.001950
grad_step = 000358, loss = 0.001961
grad_step = 000359, loss = 0.001971
grad_step = 000360, loss = 0.001943
grad_step = 000361, loss = 0.001918
grad_step = 000362, loss = 0.001931
grad_step = 000363, loss = 0.001952
grad_step = 000364, loss = 0.001934
grad_step = 000365, loss = 0.001910
grad_step = 000366, loss = 0.001907
grad_step = 000367, loss = 0.001924
grad_step = 000368, loss = 0.001938
grad_step = 000369, loss = 0.001944
grad_step = 000370, loss = 0.001934
grad_step = 000371, loss = 0.001931
grad_step = 000372, loss = 0.001927
grad_step = 000373, loss = 0.001937
grad_step = 000374, loss = 0.001934
grad_step = 000375, loss = 0.001929
grad_step = 000376, loss = 0.001907
grad_step = 000377, loss = 0.001893
grad_step = 000378, loss = 0.001887
grad_step = 000379, loss = 0.001886
grad_step = 000380, loss = 0.001889
grad_step = 000381, loss = 0.001889
grad_step = 000382, loss = 0.001891
grad_step = 000383, loss = 0.001886
grad_step = 000384, loss = 0.001879
grad_step = 000385, loss = 0.001872
grad_step = 000386, loss = 0.001871
grad_step = 000387, loss = 0.001873
grad_step = 000388, loss = 0.001876
grad_step = 000389, loss = 0.001877
grad_step = 000390, loss = 0.001875
grad_step = 000391, loss = 0.001873
grad_step = 000392, loss = 0.001871
grad_step = 000393, loss = 0.001869
grad_step = 000394, loss = 0.001866
grad_step = 000395, loss = 0.001862
grad_step = 000396, loss = 0.001858
grad_step = 000397, loss = 0.001856
grad_step = 000398, loss = 0.001854
grad_step = 000399, loss = 0.001853
grad_step = 000400, loss = 0.001851
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001849
grad_step = 000402, loss = 0.001848
grad_step = 000403, loss = 0.001845
grad_step = 000404, loss = 0.001844
grad_step = 000405, loss = 0.001843
grad_step = 000406, loss = 0.001842
grad_step = 000407, loss = 0.001842
grad_step = 000408, loss = 0.001844
grad_step = 000409, loss = 0.001848
grad_step = 000410, loss = 0.001853
grad_step = 000411, loss = 0.001868
grad_step = 000412, loss = 0.001886
grad_step = 000413, loss = 0.001930
grad_step = 000414, loss = 0.001953
grad_step = 000415, loss = 0.002002
grad_step = 000416, loss = 0.001925
grad_step = 000417, loss = 0.001860
grad_step = 000418, loss = 0.001832
grad_step = 000419, loss = 0.001864
grad_step = 000420, loss = 0.001884
grad_step = 000421, loss = 0.001849
grad_step = 000422, loss = 0.001841
grad_step = 000423, loss = 0.001855
grad_step = 000424, loss = 0.001848
grad_step = 000425, loss = 0.001824
grad_step = 000426, loss = 0.001819
grad_step = 000427, loss = 0.001837
grad_step = 000428, loss = 0.001841
grad_step = 000429, loss = 0.001830
grad_step = 000430, loss = 0.001819
grad_step = 000431, loss = 0.001815
grad_step = 000432, loss = 0.001827
grad_step = 000433, loss = 0.001822
grad_step = 000434, loss = 0.001813
grad_step = 000435, loss = 0.001799
grad_step = 000436, loss = 0.001800
grad_step = 000437, loss = 0.001809
grad_step = 000438, loss = 0.001807
grad_step = 000439, loss = 0.001801
grad_step = 000440, loss = 0.001794
grad_step = 000441, loss = 0.001797
grad_step = 000442, loss = 0.001804
grad_step = 000443, loss = 0.001804
grad_step = 000444, loss = 0.001800
grad_step = 000445, loss = 0.001798
grad_step = 000446, loss = 0.001799
grad_step = 000447, loss = 0.001809
grad_step = 000448, loss = 0.001807
grad_step = 000449, loss = 0.001804
grad_step = 000450, loss = 0.001788
grad_step = 000451, loss = 0.001777
grad_step = 000452, loss = 0.001772
grad_step = 000453, loss = 0.001772
grad_step = 000454, loss = 0.001774
grad_step = 000455, loss = 0.001776
grad_step = 000456, loss = 0.001779
grad_step = 000457, loss = 0.001783
grad_step = 000458, loss = 0.001793
grad_step = 000459, loss = 0.001800
grad_step = 000460, loss = 0.001818
grad_step = 000461, loss = 0.001823
grad_step = 000462, loss = 0.001843
grad_step = 000463, loss = 0.001824
grad_step = 000464, loss = 0.001809
grad_step = 000465, loss = 0.001771
grad_step = 000466, loss = 0.001751
grad_step = 000467, loss = 0.001752
grad_step = 000468, loss = 0.001766
grad_step = 000469, loss = 0.001780
grad_step = 000470, loss = 0.001770
grad_step = 000471, loss = 0.001755
grad_step = 000472, loss = 0.001739
grad_step = 000473, loss = 0.001738
grad_step = 000474, loss = 0.001748
grad_step = 000475, loss = 0.001754
grad_step = 000476, loss = 0.001761
grad_step = 000477, loss = 0.001757
grad_step = 000478, loss = 0.001757
grad_step = 000479, loss = 0.001750
grad_step = 000480, loss = 0.001745
grad_step = 000481, loss = 0.001735
grad_step = 000482, loss = 0.001727
grad_step = 000483, loss = 0.001721
grad_step = 000484, loss = 0.001719
grad_step = 000485, loss = 0.001720
grad_step = 000486, loss = 0.001720
grad_step = 000487, loss = 0.001717
grad_step = 000488, loss = 0.001713
grad_step = 000489, loss = 0.001709
grad_step = 000490, loss = 0.001706
grad_step = 000491, loss = 0.001705
grad_step = 000492, loss = 0.001705
grad_step = 000493, loss = 0.001704
grad_step = 000494, loss = 0.001703
grad_step = 000495, loss = 0.001700
grad_step = 000496, loss = 0.001697
grad_step = 000497, loss = 0.001695
grad_step = 000498, loss = 0.001694
grad_step = 000499, loss = 0.001694
grad_step = 000500, loss = 0.001698
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001710
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

  date_run                              2020-05-24 20:18:32.805575
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.228983
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-24 20:18:32.812337
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.118874
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-24 20:18:32.819389
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.146337
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-24 20:18:32.824482
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.806334
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
100%|██████████| 10/10 [00:02<00:00,  4.58it/s, avg_epoch_loss=5.28]
INFO:root:Epoch[0] Elapsed time 2.184 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.278164
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.27816424369812 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f34f0725320> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  8.65it/s, avg_epoch_loss=3.59e+3]
INFO:root:Epoch[0] Elapsed time 1.157 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=3590.403646
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 3590.4036458333335 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f34b812af28> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
 30%|███       | 3/10 [00:11<00:27,  3.94s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:22<00:15,  3.79s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:32<00:03,  3.67s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:35<00:00,  3.57s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 35.651 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.868827
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.868827199935913 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f34b808f588> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  5.85it/s, avg_epoch_loss=5.81]
INFO:root:Epoch[0] Elapsed time 1.709 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.810330
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.810329532623291 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f343c791e48> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
 10%|█         | 1/10 [02:07<19:08, 127.66s/it, avg_epoch_loss=0.412] 20%|██        | 2/10 [05:10<19:14, 144.28s/it, avg_epoch_loss=0.399] 30%|███       | 3/10 [08:38<19:03, 163.38s/it, avg_epoch_loss=0.39]  40%|████      | 4/10 [11:37<16:47, 167.98s/it, avg_epoch_loss=0.385] 50%|█████     | 5/10 [15:15<15:15, 183.02s/it, avg_epoch_loss=0.384] 60%|██████    | 6/10 [18:51<12:52, 193.04s/it, avg_epoch_loss=0.383] 70%|███████   | 7/10 [22:37<10:07, 202.66s/it, avg_epoch_loss=0.381] 80%|████████  | 8/10 [26:25<07:00, 210.47s/it, avg_epoch_loss=0.378] 90%|█████████ | 9/10 [29:41<03:26, 206.09s/it, avg_epoch_loss=0.376]100%|██████████| 10/10 [33:31<00:00, 213.17s/it, avg_epoch_loss=0.375]100%|██████████| 10/10 [33:31<00:00, 201.13s/it, avg_epoch_loss=0.375]
INFO:root:Epoch[0] Elapsed time 2011.356 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.374571
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3745713621377945 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f343c746128> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:02<00:00,  4.73it/s, avg_epoch_loss=415]
INFO:root:Epoch[0] Elapsed time 2.139 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=414.652022
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 414.65202175008733 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f34b8217fd0> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:00<00:00, 46.88it/s, avg_epoch_loss=5.12]
INFO:root:Epoch[0] Elapsed time 0.214 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.119356
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.119355821609497 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f34b83c4e80> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
0   2020-05-24 20:18:05.655545  ...    mean_absolute_error
1   2020-05-24 20:18:05.659063  ...     mean_squared_error
2   2020-05-24 20:18:05.661746  ...  median_absolute_error
3   2020-05-24 20:18:05.664302  ...               r2_score
4   2020-05-24 20:18:13.529428  ...    mean_absolute_error
5   2020-05-24 20:18:13.533154  ...     mean_squared_error
6   2020-05-24 20:18:13.536022  ...  median_absolute_error
7   2020-05-24 20:18:13.538725  ...               r2_score
8   2020-05-24 20:18:32.805575  ...    mean_absolute_error
9   2020-05-24 20:18:32.812337  ...     mean_squared_error
10  2020-05-24 20:18:32.819389  ...  median_absolute_error
11  2020-05-24 20:18:32.824482  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd277d65be0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd229ce0a20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd22a720d30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd229ce0a20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd277d65be0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd229ce0a20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd22a720d30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd229ce0a20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd277d65be0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd229ce0a20> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd22a720d30> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f2f88159080> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=7574d1f2518b0c3390757e1c0b9e9fd38fb2cc7a1ac7ece629da1b9d3546f183
  Stored in directory: /tmp/pip-ephem-wheel-cache-95mxy237/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f2f20e3ce48> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2850816/17464789 [===>..........................] - ETA: 0s
 9388032/17464789 [===============>..............] - ETA: 0s
16007168/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-24 20:54:35.662512: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-24 20:54:35.676354: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-24 20:54:35.676640: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5626d364bb40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-24 20:54:35.676708: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8966 - accuracy: 0.4850
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6820 - accuracy: 0.4990 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7382 - accuracy: 0.4953
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6321 - accuracy: 0.5023
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6636 - accuracy: 0.5002
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7280 - accuracy: 0.4960
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7039 - accuracy: 0.4976
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6762 - accuracy: 0.4994
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7126 - accuracy: 0.4970
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6758 - accuracy: 0.4994
11000/25000 [============>.................] - ETA: 3s - loss: 7.7070 - accuracy: 0.4974
12000/25000 [=============>................] - ETA: 2s - loss: 7.6973 - accuracy: 0.4980
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6843 - accuracy: 0.4988
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6984 - accuracy: 0.4979
15000/25000 [=================>............] - ETA: 2s - loss: 7.6922 - accuracy: 0.4983
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6858 - accuracy: 0.4988
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6717 - accuracy: 0.4997
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6473 - accuracy: 0.5013
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6452 - accuracy: 0.5014
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6433 - accuracy: 0.5015
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6429 - accuracy: 0.5015
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6580 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6538 - accuracy: 0.5008
25000/25000 [==============================] - 6s 253us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-24 20:54:47.894075
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-24 20:54:47.894075  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<138:19:11, 1.73kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:04<97:03:17, 2.47kB/s] .vector_cache/glove.6B.zip:   0%|          | 188k/862M [00:04<67:59:46, 3.52kB/s] .vector_cache/glove.6B.zip:   0%|          | 745k/862M [00:05<47:35:02, 5.03kB/s].vector_cache/glove.6B.zip:   0%|          | 2.96M/862M [00:05<33:13:37, 7.18kB/s].vector_cache/glove.6B.zip:   1%|          | 7.19M/862M [00:05<23:08:45, 10.3kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.0M/862M [00:05<16:06:42, 14.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.5M/862M [00:05<11:14:03, 20.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.2M/862M [00:05<7:48:45, 29.9kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.9M/862M [00:05<5:25:59, 42.7kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:05<3:46:45, 61.0kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.3M/862M [00:06<2:37:44, 87.1kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:06<1:49:45, 124kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 49.7M/862M [00:06<1:16:21, 177kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:07<54:15, 249kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:08<39:42, 338kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:09<30:03, 447kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:09<21:29, 624kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:09<15:10, 881kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:10<22:06, 604kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:11<17:08, 779kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.3M/862M [00:11<12:21, 1.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:12<11:15, 1.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:13<10:42, 1.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:13<08:13, 1.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:13<05:54, 2.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:14<24:22, 542kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:15<18:24, 718kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:15<13:07, 1.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:16<12:18, 1.07MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:17<09:57, 1.32MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:17<07:17, 1.80MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:18<08:11, 1.60MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:18<08:24, 1.55MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:19<06:32, 2.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:19<04:42, 2.76MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:20<12:35:41, 17.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:20<8:50:04, 24.5kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:21<6:10:38, 35.0kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:22<4:21:45, 49.4kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:22<3:05:56, 69.6kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:23<2:10:36, 99.0kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:23<1:31:22, 141kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:24<1:09:30, 185kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:24<49:58, 258kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:25<35:14, 364kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:26<27:36, 464kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:26<20:37, 620kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:26<14:44, 867kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:28<13:18, 957kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:28<11:55, 1.07MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:28<08:58, 1.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<06:52, 1.85MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<8:06:43, 26.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:30<5:40:36, 37.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<3:59:51, 52.6kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<2:51:07, 73.7kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<2:00:22, 105kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:32<1:24:10, 149kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<1:04:12, 195kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<46:19, 271kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<32:42, 382kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<25:32, 488kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<20:24, 611kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:36<14:54, 835kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<12:24, 1.00MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<09:58, 1.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:38<07:17, 1.70MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:40<07:57, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:40<08:04, 1.53MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<07:41, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:42<06:59, 1.75MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<06:08, 1.99MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:42<04:36, 2.65MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:44<06:04, 2.01MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<05:28, 2.23MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<04:04, 2.98MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<05:45, 2.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<05:15, 2.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:46<03:56, 3.07MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<05:37, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<05:10, 2.33MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:48<03:54, 3.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<05:34, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<06:21, 1.88MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:50<04:57, 2.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:50<03:37, 3.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<07:50, 1.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<06:43, 1.77MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:52<04:57, 2.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<06:15, 1.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:54<06:48, 1.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:54<05:21, 2.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<05:38, 2.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:56<05:10, 2.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:56<03:53, 3.02MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<05:27, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<05:07, 2.28MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:58<03:54, 2.99MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:59<05:14, 2.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<06:15, 1.86MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<05:01, 2.31MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:00<03:38, 3.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<14:58, 773kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:02<11:46, 983kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:02<08:29, 1.36MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<08:25, 1.37MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<08:27, 1.36MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<06:26, 1.78MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:04<04:40, 2.45MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<07:29, 1.53MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<06:30, 1.76MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:06<04:52, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<05:51, 1.94MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<06:38, 1.71MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:08<05:10, 2.20MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:08<03:47, 2.99MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<06:39, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<05:55, 1.91MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:10<04:23, 2.56MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<05:47, 1.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:12<04:31, 2.47MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<03:18, 3.37MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<14:27, 772kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<11:21, 982kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:14<08:11, 1.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<08:08, 1.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<08:09, 1.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:16<06:13, 1.78MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:16<04:32, 2.44MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<06:44, 1.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<05:45, 1.91MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:18<04:20, 2.53MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<05:23, 2.03MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<06:12, 1.76MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:20<04:57, 2.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<03:35, 3.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<14:11, 767kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<11:10, 973kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:21<08:03, 1.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<07:59, 1.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<07:59, 1.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<06:05, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:24<04:24, 2.44MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<07:15, 1.48MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<06:16, 1.71MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:25<04:38, 2.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<05:33, 1.92MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<06:16, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:27<04:59, 2.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<03:36, 2.95MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<13:54, 763kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:29<10:57, 967kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:29<07:57, 1.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<07:49, 1.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<06:39, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:31<04:53, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<05:41, 1.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<06:18, 1.66MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:33<05:00, 2.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<03:37, 2.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<15:07, 687kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:35<11:44, 885kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<08:26, 1.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<08:08, 1.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<08:00, 1.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:37<06:10, 1.67MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:38<04:47, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<6:28:27, 26.4kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:39<4:31:39, 37.7kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<3:11:16, 53.3kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<2:16:07, 74.9kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<1:35:44, 106kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<1:06:50, 152kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<57:35, 176kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<41:25, 244kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:43<29:12, 346kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<22:33, 446kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<16:53, 595kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:45<12:02, 833kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:47<10:35, 943kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<09:38, 1.04MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:47<07:17, 1.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<05:13, 1.90MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<15:39, 633kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<12:03, 822kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:49<08:39, 1.14MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<08:12, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<07:55, 1.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:51<06:00, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:51<04:19, 2.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<07:32, 1.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<06:23, 1.53MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:53<04:44, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<05:25, 1.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<05:57, 1.63MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:55<04:36, 2.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:55<03:23, 2.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<05:26, 1.77MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<04:52, 1.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:57<03:37, 2.65MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<04:38, 2.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<05:22, 1.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:59<04:12, 2.27MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:59<03:04, 3.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<05:43, 1.66MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<05:03, 1.88MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:01<03:47, 2.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<04:42, 2.00MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<05:23, 1.75MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<04:17, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<03:07, 3.00MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<13:31, 692kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<10:27, 894kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:05<07:33, 1.24MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<07:20, 1.27MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<07:12, 1.29MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:07<05:28, 1.70MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<03:56, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<06:38, 1.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<05:40, 1.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:09<04:12, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<04:54, 1.86MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<05:28, 1.67MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:11<04:20, 2.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<03:08, 2.89MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<13:12, 688kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<10:15, 886kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:13<07:21, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<07:06, 1.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<06:59, 1.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<05:23, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<03:52, 2.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<13:32, 661kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:17<10:27, 854kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:17<07:33, 1.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:19<07:11, 1.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:19<05:59, 1.48MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:19<04:25, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:21<05:01, 1.75MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:21<05:29, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:21<04:19, 2.03MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:21<03:07, 2.80MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:23<10:21, 843kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:23<08:14, 1.06MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:23<05:58, 1.46MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:25<06:02, 1.44MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:25<05:11, 1.67MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:25<03:51, 2.24MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:27<04:33, 1.88MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:27<04:06, 2.09MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:27<03:03, 2.80MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:29<04:05, 2.08MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:29<04:40, 1.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:29<03:39, 2.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:29<02:40, 3.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:31<05:26, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:31<04:44, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:31<03:33, 2.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:33<04:18, 1.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:33<04:52, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:33<03:52, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<02:48, 2.96MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:34<12:02, 691kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:35<09:20, 890kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:35<06:43, 1.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:36<06:28, 1.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:37<06:22, 1.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:37<04:50, 1.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:37<03:29, 2.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:38<05:32, 1.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:39<04:47, 1.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:39<03:34, 2.28MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:40<04:15, 1.91MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:41<04:47, 1.70MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:41<03:48, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<02:44, 2.94MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:42<10:32, 764kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:43<08:15, 972kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:43<05:57, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:44<05:53, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:45<05:53, 1.35MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:45<04:34, 1.74MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<03:16, 2.42MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:46<10:45, 735kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:47<08:21, 945kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:47<06:00, 1.31MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:48<05:57, 1.32MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:49<05:55, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:49<04:34, 1.71MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:49<03:16, 2.37MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<09:47, 793kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:51<07:14, 1.07MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:51<05:07, 1.50MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:52<14:34, 528kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:53<11:55, 645kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:53<08:45, 877kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<06:11, 1.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:54<12:29, 611kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:55<09:32, 798kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:55<06:51, 1.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:56<06:28, 1.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:57<06:12, 1.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:57<04:41, 1.61MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:57<03:23, 2.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:58<05:03, 1.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:58<04:22, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:59<03:13, 2.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:00<03:52, 1.91MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:00<04:23, 1.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:01<03:28, 2.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<02:31, 2.92MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:02<10:39, 690kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<08:17, 886kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:03<05:59, 1.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:04<05:45, 1.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:04<05:40, 1.28MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:05<04:19, 1.68MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<03:05, 2.34MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:06<10:09, 710kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:06<07:54, 910kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:07<05:43, 1.25MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:08<05:32, 1.29MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:08<04:40, 1.53MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<03:25, 2.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:10<03:56, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:10<04:19, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:11<03:25, 2.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:11<02:27, 2.84MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:12<09:13, 758kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<07:13, 967kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:13<05:14, 1.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:14<05:10, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:14<05:06, 1.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:15<03:56, 1.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<02:49, 2.43MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:16<16:35, 414kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<12:22, 554kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:16<08:49, 774kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:18<07:37, 892kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:18<06:03, 1.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:18<04:24, 1.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:20<04:36, 1.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:20<04:43, 1.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:20<03:40, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<02:54, 2.30MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:22<4:01:11, 27.6kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:22<2:48:28, 39.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:22<1:57:15, 56.3kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<1:33:32, 70.5kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:24<1:06:59, 98.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:24<47:08, 140kB/s]   .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:24<32:56, 199kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:26<25:11, 259kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:26<18:20, 355kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:26<12:58, 501kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:28<10:26, 619kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:28<08:40, 744kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:28<06:24, 1.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:28<04:32, 1.41MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:30<22:22, 286kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:30<16:20, 391kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:30<11:33, 550kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:32<09:27, 668kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:32<08:01, 787kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:32<05:57, 1.06MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:32<04:13, 1.48MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:34<09:37, 649kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:34<07:25, 840kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:34<05:21, 1.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:36<05:04, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:36<04:55, 1.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:36<03:44, 1.65MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:36<02:40, 2.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:38<06:23, 956kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:38<05:07, 1.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:38<03:42, 1.64MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:40<03:56, 1.53MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:40<03:23, 1.78MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<02:31, 2.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:42<03:07, 1.91MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:42<03:31, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:42<02:48, 2.13MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<02:01, 2.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:44<07:42, 766kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:44<06:04, 971kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:44<04:23, 1.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:46<04:19, 1.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:46<04:19, 1.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:46<03:20, 1.74MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<02:23, 2.41MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:48<07:49, 737kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:48<06:05, 945kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:48<04:22, 1.31MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:50<04:19, 1.31MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:50<04:18, 1.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:50<03:16, 1.73MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<02:22, 2.38MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:52<03:45, 1.50MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:52<03:15, 1.73MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:52<02:25, 2.31MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:54<02:53, 1.92MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:54<03:15, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:54<02:32, 2.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<01:51, 2.97MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:56<03:09, 1.74MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:56<02:49, 1.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<02:07, 2.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:58<02:39, 2.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:58<03:03, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:58<02:26, 2.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<01:45, 3.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:00<07:23, 723kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:00<05:44, 931kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<04:08, 1.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:02<04:03, 1.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:02<04:05, 1.29MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:02<03:09, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<02:15, 2.30MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:04<07:52, 661kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:04<06:04, 857kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<04:22, 1.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:06<04:11, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:06<04:04, 1.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:06<03:08, 1.63MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:06<02:14, 2.26MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:08<06:56, 730kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:08<05:24, 937kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:08<03:52, 1.30MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:10<03:49, 1.31MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:10<03:11, 1.56MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<02:21, 2.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:11<02:46, 1.78MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:12<03:03, 1.61MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:12<02:24, 2.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:12<01:44, 2.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:13<07:05, 685kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:14<05:30, 882kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:14<03:58, 1.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:15<03:47, 1.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:16<03:44, 1.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<02:49, 1.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<02:02, 2.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:17<03:12, 1.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:18<02:46, 1.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<02:03, 2.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:19<02:26, 1.91MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:20<02:42, 1.71MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:20<02:08, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:20<01:32, 2.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:21<12:02, 381kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:22<08:55, 513kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<06:19, 720kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:23<05:22, 840kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:24<04:46, 946kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:24<03:32, 1.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<02:32, 1.76MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:25<03:17, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:26<02:48, 1.58MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<02:04, 2.13MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:27<02:23, 1.83MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:28<02:36, 1.68MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:28<02:03, 2.12MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<01:28, 2.92MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:29<10:13, 421kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:29<07:37, 564kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<05:25, 787kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:31<04:41, 904kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:31<04:14, 998kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:32<03:12, 1.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<02:16, 1.84MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:33<06:02, 691kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:33<04:39, 892kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<03:21, 1.23MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:35<03:15, 1.26MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:35<03:11, 1.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:36<02:24, 1.69MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:36<01:44, 2.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:37<02:30, 1.61MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:37<02:11, 1.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<01:38, 2.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:39<02:00, 1.98MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:39<02:17, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<01:46, 2.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<01:17, 3.02MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:41<02:18, 1.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:41<02:02, 1.91MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<01:30, 2.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:43<01:54, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:43<01:43, 2.21MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<01:18, 2.91MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:45<01:46, 2.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:45<02:04, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<01:39, 2.26MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:46<01:11, 3.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:47<05:17, 696kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<04:06, 894kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<02:57, 1.24MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:49<02:49, 1.28MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:49<02:22, 1.52MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<01:45, 2.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<01:59, 1.79MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<02:10, 1.62MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<01:43, 2.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<01:13, 2.83MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:53<04:34, 758kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:53<03:34, 971kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<02:33, 1.34MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:55<02:32, 1.34MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:55<02:32, 1.34MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<01:57, 1.73MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:56<01:23, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:57<05:00, 665kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:57<03:52, 859kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<02:47, 1.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:59<02:38, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:59<02:34, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<01:58, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<01:31, 2.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:01<1:56:01, 27.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:01<1:20:45, 39.4kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<55:46, 56.3kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:03<46:48, 67.0kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:03<33:26, 93.7kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:03<23:28, 133kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:03<16:18, 189kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:05<12:25, 247kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:05<09:00, 339kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<06:20, 479kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:07<05:02, 595kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:07<04:11, 715kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:07<03:05, 966kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:07<02:09, 1.36MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:09<04:38, 630kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:09<03:34, 818kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<02:33, 1.13MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:11<02:23, 1.20MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:11<01:59, 1.43MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<01:27, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:13<01:36, 1.73MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:13<01:45, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<01:20, 2.06MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<00:58, 2.82MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:15<01:47, 1.51MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:15<01:33, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<01:08, 2.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:17<01:22, 1.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:17<01:32, 1.71MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:13, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<00:52, 2.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:19<03:22, 765kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<02:37, 979kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:53, 1.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<01:52, 1.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<01:51, 1.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<01:25, 1.76MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<01:01, 2.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:23<01:36, 1.51MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:23<01:24, 1.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<01:02, 2.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:25<01:13, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:25<01:24, 1.67MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<01:07, 2.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:25<00:47, 2.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:27<03:20, 689kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:27<02:35, 886kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<01:51, 1.22MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:29<01:45, 1.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:29<01:43, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<01:18, 1.70MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<00:56, 2.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:31<01:19, 1.63MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:31<01:10, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<00:52, 2.45MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:33<01:03, 1.98MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:33<00:58, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<00:43, 2.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:35<00:56, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:35<01:06, 1.83MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<00:51, 2.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:35<00:37, 3.17MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:37<01:10, 1.65MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:37<01:02, 1.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<00:46, 2.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:39<00:56, 2.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:39<01:04, 1.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<00:51, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:39<00:36, 3.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:41<02:21, 768kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<01:50, 982kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<01:18, 1.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:43<01:17, 1.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:43<01:16, 1.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:43<00:57, 1.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:43<00:40, 2.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:45<01:25, 1.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:45<01:10, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:45<00:51, 1.92MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:47<00:56, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:47<01:00, 1.59MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<00:46, 2.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:47<00:33, 2.80MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:49<00:53, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:49<00:47, 1.93MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:49<00:35, 2.58MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:51<00:43, 2.05MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:51<00:39, 2.23MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:51<00:29, 2.94MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:53<00:39, 2.13MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:53<00:45, 1.85MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:53<00:35, 2.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:53<00:25, 3.21MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:54<00:52, 1.53MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:55<00:45, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:55<00:33, 2.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<00:39, 1.94MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:57<00:44, 1.71MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:57<00:34, 2.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:57<00:24, 2.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:58<01:41, 702kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:59<01:19, 900kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<00:55, 1.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<00:52, 1.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:01<00:51, 1.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:01<00:38, 1.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<00:27, 2.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:02<00:44, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:03<00:38, 1.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<00:27, 2.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:31, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:05<00:35, 1.67MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:05<00:27, 2.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:05<00:19, 2.89MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<01:19, 688kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:07<01:01, 886kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:43, 1.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<00:40, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<00:39, 1.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:09<00:29, 1.67MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:09<00:20, 2.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<01:09, 671kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<00:53, 864kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<00:37, 1.19MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:34, 1.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<00:28, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:13<00:20, 2.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<00:21, 1.77MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<00:23, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:15<00:17, 2.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:15<00:12, 2.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<00:21, 1.58MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<00:18, 1.80MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:17<00:13, 2.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:18<00:15, 1.96MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:13, 2.14MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:09, 2.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<00:11, 2.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:14, 1.82MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:21<00:10, 2.28MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:21<00:07, 3.13MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:22<00:28, 772kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:22<00:21, 988kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:14, 1.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:12, 1.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:12, 1.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:09, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:25<00:05, 2.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:26<00:09, 1.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:26<00:07, 1.66MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<00:05, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:28<00:04, 1.88MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:28<00:05, 1.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:03, 2.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:29<00:01, 2.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:30<00:06, 766kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:30<00:04, 975kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:02, 1.35MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:32<00:00, 1.36MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 1.35MB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.19MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<207:38:07,  1.87s/it]  0%|          | 1051/400000 [00:01<144:57:58,  1.31s/it]  1%|          | 2124/400000 [00:02<101:12:23,  1.09it/s]  1%|          | 3184/400000 [00:02<70:39:32,  1.56it/s]   1%|          | 4220/400000 [00:02<49:20:07,  2.23it/s]  1%|▏         | 5330/400000 [00:02<34:26:27,  3.18it/s]  2%|▏         | 6360/400000 [00:02<24:02:55,  4.55it/s]  2%|▏         | 7294/400000 [00:02<16:47:52,  6.49it/s]  2%|▏         | 8308/400000 [00:02<11:43:52,  9.27it/s]  2%|▏         | 9273/400000 [00:02<8:11:42, 13.24it/s]   3%|▎         | 10323/400000 [00:02<5:43:27, 18.91it/s]  3%|▎         | 11375/400000 [00:02<3:59:57, 26.99it/s]  3%|▎         | 12381/400000 [00:03<2:47:43, 38.52it/s]  3%|▎         | 13384/400000 [00:03<1:57:18, 54.93it/s]  4%|▎         | 14380/400000 [00:03<1:22:05, 78.28it/s]  4%|▍         | 15370/400000 [00:03<57:31, 111.45it/s]   4%|▍         | 16373/400000 [00:03<40:20, 158.46it/s]  4%|▍         | 17388/400000 [00:03<28:21, 224.87it/s]  5%|▍         | 18440/400000 [00:03<19:58, 318.32it/s]  5%|▍         | 19454/400000 [00:03<14:08, 448.50it/s]  5%|▌         | 20534/400000 [00:03<10:02, 629.51it/s]  5%|▌         | 21603/400000 [00:03<07:11, 877.15it/s]  6%|▌         | 22643/400000 [00:04<05:12, 1208.08it/s]  6%|▌         | 23723/400000 [00:04<03:48, 1646.82it/s]  6%|▌         | 24768/400000 [00:04<02:50, 2200.99it/s]  6%|▋         | 25807/400000 [00:04<02:10, 2865.67it/s]  7%|▋         | 26826/400000 [00:04<01:43, 3612.06it/s]  7%|▋         | 27823/400000 [00:04<01:23, 4466.14it/s]  7%|▋         | 28868/400000 [00:04<01:08, 5392.44it/s]  7%|▋         | 29953/400000 [00:04<00:58, 6349.72it/s]  8%|▊         | 31075/400000 [00:04<00:50, 7299.20it/s]  8%|▊         | 32133/400000 [00:05<00:47, 7760.51it/s]  8%|▊         | 33206/400000 [00:05<00:43, 8462.83it/s]  9%|▊         | 34283/400000 [00:05<00:40, 9042.45it/s]  9%|▉         | 35329/400000 [00:05<00:38, 9424.93it/s]  9%|▉         | 36375/400000 [00:05<00:38, 9565.40it/s]  9%|▉         | 37423/400000 [00:05<00:36, 9819.85it/s] 10%|▉         | 38458/400000 [00:05<00:36, 9922.99it/s] 10%|▉         | 39543/400000 [00:05<00:35, 10181.31it/s] 10%|█         | 40607/400000 [00:05<00:34, 10313.15it/s] 10%|█         | 41695/400000 [00:05<00:34, 10473.54it/s] 11%|█         | 42757/400000 [00:06<00:34, 10425.81it/s] 11%|█         | 43810/400000 [00:06<00:35, 10113.41it/s] 11%|█         | 44831/400000 [00:06<00:35, 10076.09it/s] 11%|█▏        | 45868/400000 [00:06<00:34, 10159.76it/s] 12%|█▏        | 46889/400000 [00:06<00:35, 9970.82it/s]  12%|█▏        | 47891/400000 [00:06<00:36, 9547.75it/s] 12%|█▏        | 48853/400000 [00:06<00:37, 9281.76it/s] 12%|█▏        | 49806/400000 [00:06<00:37, 9354.15it/s] 13%|█▎        | 50785/400000 [00:06<00:36, 9480.53it/s] 13%|█▎        | 51737/400000 [00:06<00:37, 9263.91it/s] 13%|█▎        | 52667/400000 [00:07<00:38, 9087.36it/s] 13%|█▎        | 53651/400000 [00:07<00:37, 9298.16it/s] 14%|█▎        | 54674/400000 [00:07<00:36, 9558.08it/s] 14%|█▍        | 55652/400000 [00:07<00:35, 9622.01it/s] 14%|█▍        | 56677/400000 [00:07<00:35, 9801.27it/s] 14%|█▍        | 57680/400000 [00:07<00:34, 9865.98it/s] 15%|█▍        | 58669/400000 [00:07<00:35, 9712.39it/s] 15%|█▍        | 59643/400000 [00:07<00:35, 9562.13it/s] 15%|█▌        | 60602/400000 [00:07<00:35, 9498.48it/s] 15%|█▌        | 61607/400000 [00:08<00:35, 9655.46it/s] 16%|█▌        | 62575/400000 [00:08<00:35, 9635.62it/s] 16%|█▌        | 63540/400000 [00:08<00:35, 9612.72it/s] 16%|█▌        | 64503/400000 [00:08<00:35, 9457.46it/s] 16%|█▋        | 65450/400000 [00:08<00:36, 9277.48it/s] 17%|█▋        | 66403/400000 [00:08<00:35, 9349.37it/s] 17%|█▋        | 67344/400000 [00:08<00:35, 9367.36it/s] 17%|█▋        | 68283/400000 [00:08<00:35, 9373.58it/s] 17%|█▋        | 69276/400000 [00:08<00:34, 9531.08it/s] 18%|█▊        | 70283/400000 [00:08<00:34, 9685.17it/s] 18%|█▊        | 71264/400000 [00:09<00:33, 9720.24it/s] 18%|█▊        | 72238/400000 [00:09<00:33, 9709.32it/s] 18%|█▊        | 73211/400000 [00:09<00:33, 9714.48it/s] 19%|█▊        | 74208/400000 [00:09<00:33, 9788.82it/s] 19%|█▉        | 75222/400000 [00:09<00:32, 9888.72it/s] 19%|█▉        | 76212/400000 [00:09<00:33, 9783.54it/s] 19%|█▉        | 77209/400000 [00:09<00:32, 9834.75it/s] 20%|█▉        | 78193/400000 [00:09<00:33, 9725.53it/s] 20%|█▉        | 79167/400000 [00:09<00:33, 9587.24it/s] 20%|██        | 80160/400000 [00:09<00:33, 9685.81it/s] 20%|██        | 81171/400000 [00:10<00:32, 9806.25it/s] 21%|██        | 82199/400000 [00:10<00:31, 9943.43it/s] 21%|██        | 83195/400000 [00:10<00:32, 9814.00it/s] 21%|██        | 84225/400000 [00:10<00:31, 9952.94it/s] 21%|██▏       | 85272/400000 [00:10<00:31, 10100.38it/s] 22%|██▏       | 86284/400000 [00:10<00:31, 10105.47it/s] 22%|██▏       | 87311/400000 [00:10<00:30, 10152.53it/s] 22%|██▏       | 88328/400000 [00:10<00:31, 10027.51it/s] 22%|██▏       | 89332/400000 [00:10<00:31, 9790.30it/s]  23%|██▎       | 90313/400000 [00:10<00:31, 9767.45it/s] 23%|██▎       | 91302/400000 [00:11<00:31, 9801.33it/s] 23%|██▎       | 92328/400000 [00:11<00:30, 9933.39it/s] 23%|██▎       | 93323/400000 [00:11<00:31, 9857.26it/s] 24%|██▎       | 94323/400000 [00:11<00:30, 9898.28it/s] 24%|██▍       | 95314/400000 [00:11<00:30, 9882.39it/s] 24%|██▍       | 96303/400000 [00:11<00:31, 9625.92it/s] 24%|██▍       | 97268/400000 [00:11<00:31, 9528.89it/s] 25%|██▍       | 98226/400000 [00:11<00:31, 9542.57it/s] 25%|██▍       | 99239/400000 [00:11<00:30, 9711.08it/s] 25%|██▌       | 100299/400000 [00:11<00:30, 9960.05it/s] 25%|██▌       | 101298/400000 [00:12<00:30, 9906.61it/s] 26%|██▌       | 102291/400000 [00:12<00:30, 9791.39it/s] 26%|██▌       | 103272/400000 [00:12<00:30, 9749.01it/s] 26%|██▌       | 104300/400000 [00:12<00:29, 9902.22it/s] 26%|██▋       | 105292/400000 [00:12<00:30, 9732.10it/s] 27%|██▋       | 106315/400000 [00:12<00:29, 9874.82it/s] 27%|██▋       | 107314/400000 [00:12<00:29, 9907.83it/s] 27%|██▋       | 108309/400000 [00:12<00:29, 9919.16it/s] 27%|██▋       | 109302/400000 [00:12<00:29, 9778.70it/s] 28%|██▊       | 110281/400000 [00:12<00:30, 9642.12it/s] 28%|██▊       | 111320/400000 [00:13<00:29, 9852.30it/s] 28%|██▊       | 112308/400000 [00:13<00:30, 9513.01it/s] 28%|██▊       | 113264/400000 [00:13<00:30, 9435.53it/s] 29%|██▊       | 114303/400000 [00:13<00:29, 9700.15it/s] 29%|██▉       | 115277/400000 [00:13<00:29, 9678.73it/s] 29%|██▉       | 116285/400000 [00:13<00:28, 9794.75it/s] 29%|██▉       | 117267/400000 [00:13<00:29, 9631.38it/s] 30%|██▉       | 118233/400000 [00:13<00:29, 9587.42it/s] 30%|██▉       | 119244/400000 [00:13<00:28, 9737.60it/s] 30%|███       | 120220/400000 [00:14<00:28, 9671.48it/s] 30%|███       | 121217/400000 [00:14<00:28, 9758.26it/s] 31%|███       | 122206/400000 [00:14<00:28, 9795.38it/s] 31%|███       | 123232/400000 [00:14<00:27, 9928.74it/s] 31%|███       | 124305/400000 [00:14<00:27, 10154.92it/s] 31%|███▏      | 125323/400000 [00:14<00:27, 10153.72it/s] 32%|███▏      | 126340/400000 [00:14<00:27, 10056.21it/s] 32%|███▏      | 127347/400000 [00:14<00:27, 9848.96it/s]  32%|███▏      | 128334/400000 [00:14<00:27, 9707.80it/s] 32%|███▏      | 129307/400000 [00:14<00:28, 9657.16it/s] 33%|███▎      | 130274/400000 [00:15<00:28, 9550.36it/s] 33%|███▎      | 131274/400000 [00:15<00:27, 9679.16it/s] 33%|███▎      | 132244/400000 [00:15<00:27, 9595.55it/s] 33%|███▎      | 133220/400000 [00:15<00:27, 9641.41it/s] 34%|███▎      | 134248/400000 [00:15<00:27, 9822.29it/s] 34%|███▍      | 135232/400000 [00:15<00:27, 9754.52it/s] 34%|███▍      | 136229/400000 [00:15<00:26, 9816.64it/s] 34%|███▍      | 137212/400000 [00:15<00:27, 9611.98it/s] 35%|███▍      | 138175/400000 [00:15<00:27, 9498.64it/s] 35%|███▍      | 139187/400000 [00:15<00:26, 9673.93it/s] 35%|███▌      | 140232/400000 [00:16<00:26, 9893.05it/s] 35%|███▌      | 141272/400000 [00:16<00:25, 10038.55it/s] 36%|███▌      | 142288/400000 [00:16<00:25, 10072.83it/s] 36%|███▌      | 143345/400000 [00:16<00:25, 10215.16it/s] 36%|███▌      | 144416/400000 [00:16<00:24, 10358.62it/s] 36%|███▋      | 145454/400000 [00:16<00:24, 10307.68it/s] 37%|███▋      | 146486/400000 [00:16<00:25, 9991.60it/s]  37%|███▋      | 147489/400000 [00:16<00:25, 9718.70it/s] 37%|███▋      | 148465/400000 [00:16<00:26, 9652.05it/s] 37%|███▋      | 149437/400000 [00:16<00:25, 9669.82it/s] 38%|███▊      | 150406/400000 [00:17<00:26, 9392.02it/s] 38%|███▊      | 151349/400000 [00:17<00:26, 9375.36it/s] 38%|███▊      | 152330/400000 [00:17<00:26, 9499.99it/s] 38%|███▊      | 153342/400000 [00:17<00:25, 9676.61it/s] 39%|███▊      | 154320/400000 [00:17<00:25, 9705.17it/s] 39%|███▉      | 155343/400000 [00:17<00:24, 9854.47it/s] 39%|███▉      | 156331/400000 [00:17<00:24, 9849.80it/s] 39%|███▉      | 157363/400000 [00:17<00:24, 9984.15it/s] 40%|███▉      | 158365/400000 [00:17<00:24, 9992.13it/s] 40%|███▉      | 159376/400000 [00:17<00:24, 10025.25it/s] 40%|████      | 160380/400000 [00:18<00:24, 9839.68it/s]  40%|████      | 161402/400000 [00:18<00:23, 9949.04it/s] 41%|████      | 162423/400000 [00:18<00:23, 10023.56it/s] 41%|████      | 163469/400000 [00:18<00:23, 10148.78it/s] 41%|████      | 164485/400000 [00:18<00:23, 9937.13it/s]  41%|████▏     | 165566/400000 [00:18<00:23, 10183.30it/s] 42%|████▏     | 166588/400000 [00:18<00:22, 10159.30it/s] 42%|████▏     | 167606/400000 [00:18<00:23, 10084.71it/s] 42%|████▏     | 168616/400000 [00:18<00:22, 10084.91it/s] 42%|████▏     | 169626/400000 [00:19<00:22, 10020.67it/s] 43%|████▎     | 170629/400000 [00:19<00:22, 9992.61it/s]  43%|████▎     | 171629/400000 [00:19<00:23, 9922.87it/s] 43%|████▎     | 172622/400000 [00:19<00:22, 9916.74it/s] 43%|████▎     | 173615/400000 [00:19<00:23, 9791.35it/s] 44%|████▎     | 174595/400000 [00:19<00:23, 9720.74it/s] 44%|████▍     | 175568/400000 [00:19<00:23, 9618.65it/s] 44%|████▍     | 176542/400000 [00:19<00:23, 9653.79it/s] 44%|████▍     | 177562/400000 [00:19<00:22, 9810.99it/s] 45%|████▍     | 178559/400000 [00:19<00:22, 9855.69it/s] 45%|████▍     | 179571/400000 [00:20<00:22, 9930.61it/s] 45%|████▌     | 180565/400000 [00:20<00:22, 9760.64it/s] 45%|████▌     | 181587/400000 [00:20<00:22, 9892.48it/s] 46%|████▌     | 182578/400000 [00:20<00:22, 9634.65it/s] 46%|████▌     | 183544/400000 [00:20<00:22, 9521.57it/s] 46%|████▌     | 184499/400000 [00:20<00:22, 9459.89it/s] 46%|████▋     | 185549/400000 [00:20<00:21, 9749.42it/s] 47%|████▋     | 186544/400000 [00:20<00:21, 9806.35it/s] 47%|████▋     | 187527/400000 [00:20<00:21, 9772.51it/s] 47%|████▋     | 188563/400000 [00:20<00:21, 9938.69it/s] 47%|████▋     | 189559/400000 [00:21<00:21, 9893.21it/s] 48%|████▊     | 190618/400000 [00:21<00:20, 10091.17it/s] 48%|████▊     | 191641/400000 [00:21<00:20, 10129.64it/s] 48%|████▊     | 192656/400000 [00:21<00:20, 9950.37it/s]  48%|████▊     | 193653/400000 [00:21<00:20, 9941.81it/s] 49%|████▊     | 194666/400000 [00:21<00:20, 9995.35it/s] 49%|████▉     | 195717/400000 [00:21<00:20, 10143.02it/s] 49%|████▉     | 196733/400000 [00:21<00:20, 10136.90it/s] 49%|████▉     | 197748/400000 [00:21<00:20, 9972.73it/s]  50%|████▉     | 198747/400000 [00:21<00:20, 9737.11it/s] 50%|████▉     | 199723/400000 [00:22<00:21, 9401.60it/s] 50%|█████     | 200668/400000 [00:22<00:21, 9283.56it/s] 50%|█████     | 201603/400000 [00:22<00:21, 9302.45it/s] 51%|█████     | 202536/400000 [00:22<00:21, 9263.43it/s] 51%|█████     | 203487/400000 [00:22<00:21, 9334.84it/s] 51%|█████     | 204461/400000 [00:22<00:20, 9451.08it/s] 51%|█████▏    | 205464/400000 [00:22<00:20, 9616.78it/s] 52%|█████▏    | 206428/400000 [00:22<00:20, 9604.67it/s] 52%|█████▏    | 207390/400000 [00:22<00:20, 9557.50it/s] 52%|█████▏    | 208373/400000 [00:23<00:19, 9637.01it/s] 52%|█████▏    | 209359/400000 [00:23<00:19, 9699.67it/s] 53%|█████▎    | 210366/400000 [00:23<00:19, 9806.30it/s] 53%|█████▎    | 211348/400000 [00:23<00:19, 9545.42it/s] 53%|█████▎    | 212305/400000 [00:23<00:20, 9328.04it/s] 53%|█████▎    | 213258/400000 [00:23<00:19, 9385.34it/s] 54%|█████▎    | 214199/400000 [00:23<00:19, 9363.16it/s] 54%|█████▍    | 215137/400000 [00:23<00:19, 9308.68it/s] 54%|█████▍    | 216069/400000 [00:23<00:19, 9287.38it/s] 54%|█████▍    | 216999/400000 [00:23<00:19, 9226.21it/s] 55%|█████▍    | 218003/400000 [00:24<00:19, 9454.74it/s] 55%|█████▍    | 218979/400000 [00:24<00:18, 9542.51it/s] 55%|█████▍    | 219941/400000 [00:24<00:18, 9564.79it/s] 55%|█████▌    | 220920/400000 [00:24<00:18, 9630.66it/s] 55%|█████▌    | 221887/400000 [00:24<00:18, 9640.68it/s] 56%|█████▌    | 222902/400000 [00:24<00:18, 9786.79it/s] 56%|█████▌    | 223882/400000 [00:24<00:18, 9686.30it/s] 56%|█████▌    | 224852/400000 [00:24<00:18, 9652.98it/s] 56%|█████▋    | 225818/400000 [00:24<00:18, 9632.23it/s] 57%|█████▋    | 226782/400000 [00:24<00:18, 9457.19it/s] 57%|█████▋    | 227729/400000 [00:25<00:18, 9322.56it/s] 57%|█████▋    | 228688/400000 [00:25<00:18, 9399.98it/s] 57%|█████▋    | 229679/400000 [00:25<00:17, 9544.84it/s] 58%|█████▊    | 230660/400000 [00:25<00:17, 9621.53it/s] 58%|█████▊    | 231624/400000 [00:25<00:18, 9347.22it/s] 58%|█████▊    | 232562/400000 [00:25<00:18, 9287.19it/s] 58%|█████▊    | 233537/400000 [00:25<00:17, 9420.24it/s] 59%|█████▊    | 234498/400000 [00:25<00:17, 9474.99it/s] 59%|█████▉    | 235474/400000 [00:25<00:17, 9557.56it/s] 59%|█████▉    | 236473/400000 [00:25<00:16, 9683.28it/s] 59%|█████▉    | 237443/400000 [00:26<00:16, 9625.35it/s] 60%|█████▉    | 238427/400000 [00:26<00:16, 9687.71it/s] 60%|█████▉    | 239434/400000 [00:26<00:16, 9798.69it/s] 60%|██████    | 240415/400000 [00:26<00:16, 9715.50it/s] 60%|██████    | 241388/400000 [00:26<00:16, 9700.26it/s] 61%|██████    | 242359/400000 [00:26<00:16, 9531.82it/s] 61%|██████    | 243314/400000 [00:26<00:16, 9382.79it/s] 61%|██████    | 244320/400000 [00:26<00:16, 9574.16it/s] 61%|██████▏   | 245280/400000 [00:26<00:16, 9416.41it/s] 62%|██████▏   | 246383/400000 [00:26<00:15, 9847.62it/s] 62%|██████▏   | 247438/400000 [00:27<00:15, 10047.05it/s] 62%|██████▏   | 248449/400000 [00:27<00:15, 9913.63it/s]  62%|██████▏   | 249445/400000 [00:27<00:15, 9867.19it/s] 63%|██████▎   | 250492/400000 [00:27<00:14, 10037.66it/s] 63%|██████▎   | 251499/400000 [00:27<00:14, 9992.38it/s]  63%|██████▎   | 252501/400000 [00:27<00:14, 9884.15it/s] 63%|██████▎   | 253492/400000 [00:27<00:14, 9836.36it/s] 64%|██████▎   | 254477/400000 [00:27<00:14, 9763.46it/s] 64%|██████▍   | 255480/400000 [00:27<00:14, 9841.12it/s] 64%|██████▍   | 256507/400000 [00:27<00:14, 9965.50it/s] 64%|██████▍   | 257524/400000 [00:28<00:14, 10024.33it/s] 65%|██████▍   | 258528/400000 [00:28<00:14, 9934.40it/s]  65%|██████▍   | 259523/400000 [00:28<00:14, 9631.51it/s] 65%|██████▌   | 260489/400000 [00:28<00:14, 9557.87it/s] 65%|██████▌   | 261447/400000 [00:28<00:14, 9450.06it/s] 66%|██████▌   | 262394/400000 [00:28<00:14, 9311.22it/s] 66%|██████▌   | 263327/400000 [00:28<00:14, 9311.18it/s] 66%|██████▌   | 264260/400000 [00:28<00:14, 9221.14it/s] 66%|██████▋   | 265184/400000 [00:28<00:14, 9089.44it/s] 67%|██████▋   | 266158/400000 [00:29<00:14, 9272.74it/s] 67%|██████▋   | 267087/400000 [00:29<00:14, 9143.44it/s] 67%|██████▋   | 268081/400000 [00:29<00:14, 9366.97it/s] 67%|██████▋   | 269084/400000 [00:29<00:13, 9553.40it/s] 68%|██████▊   | 270042/400000 [00:29<00:13, 9476.93it/s] 68%|██████▊   | 270992/400000 [00:29<00:13, 9393.53it/s] 68%|██████▊   | 272028/400000 [00:29<00:13, 9663.03it/s] 68%|██████▊   | 272998/400000 [00:29<00:13, 9601.17it/s] 69%|██████▊   | 274018/400000 [00:29<00:12, 9771.25it/s] 69%|██████▊   | 274998/400000 [00:29<00:12, 9744.64it/s] 69%|██████▉   | 276015/400000 [00:30<00:12, 9866.70it/s] 69%|██████▉   | 277023/400000 [00:30<00:12, 9929.22it/s] 70%|██████▉   | 278031/400000 [00:30<00:12, 9973.24it/s] 70%|██████▉   | 279038/400000 [00:30<00:12, 10000.64it/s] 70%|███████   | 280039/400000 [00:30<00:12, 9707.53it/s]  70%|███████   | 281013/400000 [00:30<00:12, 9589.45it/s] 70%|███████   | 281974/400000 [00:30<00:12, 9579.44it/s] 71%|███████   | 282953/400000 [00:30<00:12, 9640.33it/s] 71%|███████   | 284040/400000 [00:30<00:11, 9976.63it/s] 71%|███████▏  | 285042/400000 [00:30<00:11, 9964.63it/s] 72%|███████▏  | 286042/400000 [00:31<00:11, 9841.19it/s] 72%|███████▏  | 287031/400000 [00:31<00:11, 9855.62it/s] 72%|███████▏  | 288019/400000 [00:31<00:11, 9726.37it/s] 72%|███████▏  | 288994/400000 [00:31<00:11, 9730.85it/s] 72%|███████▏  | 289969/400000 [00:31<00:11, 9545.81it/s] 73%|███████▎  | 290926/400000 [00:31<00:11, 9275.22it/s] 73%|███████▎  | 291857/400000 [00:31<00:11, 9100.57it/s] 73%|███████▎  | 292770/400000 [00:31<00:12, 8893.47it/s] 73%|███████▎  | 293663/400000 [00:31<00:12, 8791.56it/s] 74%|███████▎  | 294578/400000 [00:32<00:11, 8895.19it/s] 74%|███████▍  | 295590/400000 [00:32<00:11, 9228.04it/s] 74%|███████▍  | 296566/400000 [00:32<00:11, 9381.25it/s] 74%|███████▍  | 297508/400000 [00:32<00:10, 9327.91it/s] 75%|███████▍  | 298485/400000 [00:32<00:10, 9454.98it/s] 75%|███████▍  | 299433/400000 [00:32<00:10, 9399.26it/s] 75%|███████▌  | 300481/400000 [00:32<00:10, 9699.08it/s] 75%|███████▌  | 301473/400000 [00:32<00:10, 9763.30it/s] 76%|███████▌  | 302452/400000 [00:32<00:10, 9749.98it/s] 76%|███████▌  | 303480/400000 [00:32<00:09, 9902.58it/s] 76%|███████▌  | 304473/400000 [00:33<00:09, 9882.93it/s] 76%|███████▋  | 305499/400000 [00:33<00:09, 9991.21it/s] 77%|███████▋  | 306627/400000 [00:33<00:09, 10344.51it/s] 77%|███████▋  | 307721/400000 [00:33<00:08, 10516.20it/s] 77%|███████▋  | 308871/400000 [00:33<00:08, 10792.30it/s] 77%|███████▋  | 309955/400000 [00:33<00:08, 10509.23it/s] 78%|███████▊  | 311011/400000 [00:33<00:08, 10501.52it/s] 78%|███████▊  | 312065/400000 [00:33<00:08, 10406.29it/s] 78%|███████▊  | 313109/400000 [00:33<00:08, 10232.62it/s] 79%|███████▊  | 314156/400000 [00:33<00:08, 10302.08it/s] 79%|███████▉  | 315188/400000 [00:34<00:08, 10306.17it/s] 79%|███████▉  | 316312/400000 [00:34<00:07, 10568.59it/s] 79%|███████▉  | 317398/400000 [00:34<00:07, 10654.31it/s] 80%|███████▉  | 318466/400000 [00:34<00:07, 10638.59it/s] 80%|███████▉  | 319532/400000 [00:34<00:07, 10610.28it/s] 80%|████████  | 320594/400000 [00:34<00:07, 10483.60it/s] 80%|████████  | 321667/400000 [00:34<00:07, 10553.90it/s] 81%|████████  | 322727/400000 [00:34<00:07, 10566.34it/s] 81%|████████  | 323843/400000 [00:34<00:07, 10737.09it/s] 81%|████████  | 324918/400000 [00:34<00:07, 10574.40it/s] 81%|████████▏ | 325977/400000 [00:35<00:07, 10441.54it/s] 82%|████████▏ | 327023/400000 [00:35<00:07, 10225.17it/s] 82%|████████▏ | 328057/400000 [00:35<00:07, 10258.79it/s] 82%|████████▏ | 329085/400000 [00:35<00:06, 10246.66it/s] 83%|████████▎ | 330111/400000 [00:35<00:06, 10012.32it/s] 83%|████████▎ | 331115/400000 [00:35<00:07, 9805.02it/s]  83%|████████▎ | 332098/400000 [00:35<00:07, 9686.80it/s] 83%|████████▎ | 333069/400000 [00:35<00:06, 9605.28it/s] 84%|████████▎ | 334032/400000 [00:35<00:06, 9485.52it/s] 84%|████████▎ | 334982/400000 [00:35<00:06, 9441.55it/s] 84%|████████▍ | 335928/400000 [00:36<00:06, 9267.02it/s] 84%|████████▍ | 336975/400000 [00:36<00:06, 9597.40it/s] 85%|████████▍ | 338019/400000 [00:36<00:06, 9834.68it/s] 85%|████████▍ | 339022/400000 [00:36<00:06, 9890.78it/s] 85%|████████▌ | 340036/400000 [00:36<00:06, 9961.18it/s] 85%|████████▌ | 341035/400000 [00:36<00:05, 9923.72it/s] 86%|████████▌ | 342129/400000 [00:36<00:05, 10206.27it/s] 86%|████████▌ | 343153/400000 [00:36<00:05, 9874.30it/s]  86%|████████▌ | 344152/400000 [00:36<00:05, 9907.91it/s] 86%|████████▋ | 345189/400000 [00:37<00:05, 10041.46it/s] 87%|████████▋ | 346196/400000 [00:37<00:05, 9988.42it/s]  87%|████████▋ | 347197/400000 [00:37<00:05, 9888.17it/s] 87%|████████▋ | 348188/400000 [00:37<00:05, 9790.29it/s] 87%|████████▋ | 349225/400000 [00:37<00:05, 9955.50it/s] 88%|████████▊ | 350232/400000 [00:37<00:04, 9988.54it/s] 88%|████████▊ | 351233/400000 [00:37<00:04, 9947.20it/s] 88%|████████▊ | 352229/400000 [00:37<00:04, 9917.04it/s] 88%|████████▊ | 353222/400000 [00:37<00:04, 9897.47it/s] 89%|████████▊ | 354213/400000 [00:37<00:04, 9852.66it/s] 89%|████████▉ | 355215/400000 [00:38<00:04, 9901.65it/s] 89%|████████▉ | 356206/400000 [00:38<00:04, 9529.88it/s] 89%|████████▉ | 357244/400000 [00:38<00:04, 9769.61it/s] 90%|████████▉ | 358225/400000 [00:38<00:04, 9464.82it/s] 90%|████████▉ | 359177/400000 [00:38<00:04, 9359.34it/s] 90%|█████████ | 360175/400000 [00:38<00:04, 9535.83it/s] 90%|█████████ | 361132/400000 [00:38<00:04, 9380.47it/s] 91%|█████████ | 362125/400000 [00:38<00:03, 9537.63it/s] 91%|█████████ | 363082/400000 [00:38<00:03, 9492.56it/s] 91%|█████████ | 364034/400000 [00:38<00:03, 9388.48it/s] 91%|█████████ | 364976/400000 [00:39<00:03, 9396.29it/s] 91%|█████████▏| 365949/400000 [00:39<00:03, 9491.74it/s] 92%|█████████▏| 366926/400000 [00:39<00:03, 9570.93it/s] 92%|█████████▏| 367904/400000 [00:39<00:03, 9632.22it/s] 92%|█████████▏| 368933/400000 [00:39<00:03, 9819.35it/s] 92%|█████████▏| 369936/400000 [00:39<00:03, 9881.28it/s] 93%|█████████▎| 370926/400000 [00:39<00:03, 9640.64it/s] 93%|█████████▎| 371957/400000 [00:39<00:02, 9831.65it/s] 93%|█████████▎| 372943/400000 [00:39<00:02, 9830.47it/s] 93%|█████████▎| 373945/400000 [00:39<00:02, 9886.20it/s] 94%|█████████▎| 374935/400000 [00:40<00:02, 9703.84it/s] 94%|█████████▍| 375907/400000 [00:40<00:02, 9522.50it/s] 94%|█████████▍| 376905/400000 [00:40<00:02, 9653.41it/s] 94%|█████████▍| 377937/400000 [00:40<00:02, 9839.94it/s] 95%|█████████▍| 378924/400000 [00:40<00:02, 9715.44it/s] 95%|█████████▍| 379898/400000 [00:40<00:02, 9558.05it/s] 95%|█████████▌| 380869/400000 [00:40<00:01, 9600.78it/s] 95%|█████████▌| 381831/400000 [00:40<00:01, 9605.10it/s] 96%|█████████▌| 382793/400000 [00:40<00:01, 9419.25it/s] 96%|█████████▌| 383737/400000 [00:41<00:01, 9352.53it/s] 96%|█████████▌| 384674/400000 [00:41<00:01, 9338.57it/s] 96%|█████████▋| 385700/400000 [00:41<00:01, 9594.75it/s] 97%|█████████▋| 386748/400000 [00:41<00:01, 9842.38it/s] 97%|█████████▋| 387816/400000 [00:41<00:01, 10077.64it/s] 97%|█████████▋| 388855/400000 [00:41<00:01, 10168.51it/s] 97%|█████████▋| 389875/400000 [00:41<00:00, 10134.22it/s] 98%|█████████▊| 390891/400000 [00:41<00:00, 9728.09it/s]  98%|█████████▊| 391943/400000 [00:41<00:00, 9950.77it/s] 98%|█████████▊| 393008/400000 [00:41<00:00, 10149.30it/s] 99%|█████████▊| 394028/400000 [00:42<00:00, 9724.28it/s]  99%|█████████▉| 395035/400000 [00:42<00:00, 9824.45it/s] 99%|█████████▉| 396023/400000 [00:42<00:00, 9818.24it/s] 99%|█████████▉| 397064/400000 [00:42<00:00, 9986.41it/s]100%|█████████▉| 398066/400000 [00:42<00:00, 9861.06it/s]100%|█████████▉| 399055/400000 [00:42<00:00, 9543.03it/s]100%|█████████▉| 399999/400000 [00:42<00:00, 9377.10it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f77a9456ac8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011253955111824221 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011232887821452674 	 Accuracy: 51

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
2020-05-24 21:03:50.601718: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-24 21:03:50.605929: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-24 21:03:50.606090: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5586807fdff0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-24 21:03:50.606101: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f7754d16080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8046 - accuracy: 0.4910
 2000/25000 [=>............................] - ETA: 7s - loss: 7.9656 - accuracy: 0.4805 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8557 - accuracy: 0.4877
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7701 - accuracy: 0.4933
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7310 - accuracy: 0.4958
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7561 - accuracy: 0.4942
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7389 - accuracy: 0.4953
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7644 - accuracy: 0.4936
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7348 - accuracy: 0.4956
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7019 - accuracy: 0.4977
11000/25000 [============>.................] - ETA: 3s - loss: 7.6750 - accuracy: 0.4995
12000/25000 [=============>................] - ETA: 3s - loss: 7.6538 - accuracy: 0.5008
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6643 - accuracy: 0.5002
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6491 - accuracy: 0.5011
15000/25000 [=================>............] - ETA: 2s - loss: 7.6584 - accuracy: 0.5005
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6705 - accuracy: 0.4997
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6756 - accuracy: 0.4994
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6734 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6529 - accuracy: 0.5009
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6544 - accuracy: 0.5008
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6528 - accuracy: 0.5009
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6610 - accuracy: 0.5004
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6551 - accuracy: 0.5008
25000/25000 [==============================] - 6s 256us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f7709b3c320> <class 'mlmodels.model_tch.transformer_sentence.Model'>

  ############ Dataloader setup  ############################# 

  {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': True}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}} [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/AllNLI/s1.train.gz' 

  


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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f7754d16080> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 5s 5s/step - loss: 1.5448 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.4455 - val_crf_viterbi_accuracy: 0.0267

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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.py", line 139, in fit
    train_data       = SentencesDataset(train_reader.get_examples(train_fname),  model=model.model)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/sentence_transformers/readers/NLIDataReader.py", line 21, in get_examples
    mode="rt", encoding="utf-8").readlines()
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/gzip.py", line 53, in open
    binary_file = GzipFile(filename, gz_mode, compresslevel)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/gzip.py", line 163, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/AllNLI/s1.train.gz'
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

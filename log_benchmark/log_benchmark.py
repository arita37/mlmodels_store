
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f37e6ef5eb8> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-25 20:18:27.926220
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-25 20:18:27.929288
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-25 20:18:27.932300
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-25 20:18:27.935185
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f37f2cc02e8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352859.9688
Epoch 2/10

1/1 [==============================] - 0s 92ms/step - loss: 250238.0312
Epoch 3/10

1/1 [==============================] - 0s 91ms/step - loss: 148236.0469
Epoch 4/10

1/1 [==============================] - 0s 90ms/step - loss: 80631.5078
Epoch 5/10

1/1 [==============================] - 0s 95ms/step - loss: 44259.3750
Epoch 6/10

1/1 [==============================] - 0s 89ms/step - loss: 25648.1328
Epoch 7/10

1/1 [==============================] - 0s 91ms/step - loss: 16043.5596
Epoch 8/10

1/1 [==============================] - 0s 94ms/step - loss: 10794.7588
Epoch 9/10

1/1 [==============================] - 0s 98ms/step - loss: 7716.1401
Epoch 10/10

1/1 [==============================] - 0s 90ms/step - loss: 5834.6377

  #### Inference Need return ypred, ytrue ######################### 
[[ 3.4807369e-01  7.0968761e+00  9.8163433e+00  9.8725281e+00
   9.6487694e+00  7.9522080e+00  9.5665131e+00  7.5985565e+00
   7.9104466e+00  7.5136485e+00  6.6121707e+00  9.2397404e+00
   1.1109816e+01  1.1522207e+01  8.8634567e+00  9.8188334e+00
   8.8400517e+00  9.3962059e+00  1.0192362e+01  9.5746508e+00
   1.0910850e+01  8.3433352e+00  6.6287351e+00  8.6368666e+00
   1.1676565e+01  8.2532988e+00  7.8157649e+00  8.9705029e+00
   1.0330533e+01  8.5204754e+00  1.0869669e+01  1.0632700e+01
   9.5518780e+00  8.8458290e+00  1.0079676e+01  9.6356688e+00
   8.5868492e+00  8.7898836e+00  9.7475004e+00  8.6441698e+00
   7.4090261e+00  1.0330508e+01  6.8947320e+00  9.3779411e+00
   1.0031142e+01  6.2470384e+00  9.7650204e+00  7.2923975e+00
   8.9929314e+00  1.0481928e+01  8.8985691e+00  9.0489655e+00
   1.0147659e+01  9.6603336e+00  8.3992157e+00  9.8259325e+00
   9.3074703e+00  7.2462049e+00  7.8446884e+00  9.5850058e+00
   4.5965025e-01  1.5404487e+00  1.0530620e+00 -2.2403083e+00
   5.0853205e-01  4.7911730e-01 -6.0364348e-01 -7.8110617e-01
   2.5232549e+00  9.0010679e-01  3.9698821e-02  1.0977843e+00
  -1.6862662e+00  2.0857978e+00 -9.3941039e-01 -1.6005546e+00
  -3.8141128e-01 -1.4161903e-01  5.2095222e-01  8.8220358e-01
   5.3036684e-01 -1.5614381e+00 -2.6870131e-02  1.4975955e+00
   1.8291993e+00  2.7373028e-01  3.0673587e-01  4.8298883e-01
   6.7728734e-01 -9.0550661e-01  1.3260660e+00 -8.3076990e-01
   7.9187751e-01 -1.1248473e+00  1.0646974e+00 -9.5810622e-01
  -9.0138137e-01 -9.4005418e-01 -1.3103007e+00 -1.3765364e+00
   1.0130315e+00  6.9154185e-01 -1.8400017e+00 -1.7578337e+00
   1.7031077e+00  2.0261067e-01 -6.7706668e-01 -7.2702557e-01
   1.5239058e+00  1.4120734e+00 -1.6269155e+00 -4.6944606e-01
  -8.4459162e-01  1.4604053e-01  1.3963679e+00  4.4961935e-01
  -1.2189186e+00  4.2638630e-01  2.0260148e+00  1.7815456e-01
   3.9986169e-01  2.1655855e+00 -1.4808601e+00 -1.0057495e+00
   1.2629069e+00  1.5007629e+00  3.9942527e-01  1.0107540e+00
   1.3589486e+00  5.8341563e-02 -1.4760357e+00 -1.4357905e+00
  -1.4631222e+00  4.9028993e-02  7.7701735e-01  5.7045507e-01
   7.2474253e-01 -1.1620578e+00 -7.2214150e-01  1.5898483e+00
  -1.4202694e+00 -6.4241284e-01  1.8302048e+00  1.7124101e+00
  -6.9395602e-03  2.4703014e-01 -7.6971859e-02  1.1508164e+00
   6.3643003e-01 -4.0313292e-01  1.2408057e+00 -6.3865137e-01
  -9.7451210e-01  6.9019270e-01 -4.9186823e-01 -1.0279335e+00
   1.3765861e+00 -4.7373545e-01  7.8920943e-01  2.2341893e+00
   8.4480560e-01 -5.0043070e-01  2.4667063e-01 -4.5720553e-01
  -4.5057791e-01  5.6611359e-01 -1.5497601e+00 -1.2504214e-01
  -1.1204863e+00  7.4839377e-01  8.7716997e-01 -1.0191340e+00
   9.4786680e-01 -8.0127764e-01 -4.5571744e-02 -1.2212696e+00
   6.9681883e-02 -4.8077521e-01  1.2585952e+00 -4.4126950e-02
   1.9883507e-01  8.7168331e+00  7.4964733e+00  1.0493902e+01
   7.1176386e+00  1.0595031e+01  6.8805103e+00  9.8580856e+00
   9.4565439e+00  1.0814429e+01  1.0055159e+01  1.0262559e+01
   8.9584084e+00  8.9745388e+00  1.0149610e+01  8.2900801e+00
   7.5644240e+00  1.0466083e+01  9.6171656e+00  9.7473412e+00
   1.0178815e+01  1.1385778e+01  7.3418407e+00  7.8759298e+00
   9.1655903e+00  8.1204147e+00  6.9061050e+00  1.0890546e+01
   8.2318850e+00  1.0401828e+01  8.8237782e+00  9.6439915e+00
   8.6979799e+00  8.5656538e+00  9.3459110e+00  7.9096050e+00
   8.6663084e+00  8.6660728e+00  9.3589039e+00  8.0813303e+00
   9.4394426e+00  1.0431445e+01  8.3313179e+00  8.8167706e+00
   1.0747479e+01  9.3978128e+00  8.4136257e+00  1.0854970e+01
   9.7223921e+00  8.3810921e+00  9.6081343e+00  9.2505865e+00
   8.1750431e+00  1.0655632e+01  7.8708787e+00  8.9860334e+00
   6.9792452e+00  8.5335960e+00  7.8291206e+00  1.0139994e+01
   6.4574295e-01  1.2768264e+00  2.8043571e+00  9.7664547e-01
   2.2670922e+00  2.7319062e-01  5.1402926e-01  3.7038794e+00
   3.2251924e-01  2.8403020e-01  1.4224200e+00  1.0431757e+00
   2.6524124e+00  8.4505272e-01  4.2461753e-01  7.7061862e-01
   1.7418785e+00  3.3707476e-01  1.4276081e-01  2.3658564e+00
   1.5606140e+00  8.2080770e-01  1.0844004e-01  1.8751627e+00
   7.8428936e-01  2.2034764e+00  3.6336339e-01  7.1633178e-01
   2.5294530e-01  4.0184474e-01  3.2240691e+00  2.5896633e-01
   1.0276014e+00  1.1369722e+00  3.0766428e-01  2.0138106e+00
   9.7759289e-01  1.7202204e+00  2.6723835e+00  2.0067549e-01
   3.6850041e-01  1.2145797e+00  1.7690182e-01  2.5739720e+00
   1.7691160e+00  5.3464699e-01  6.5736914e-01  7.9583770e-01
   1.2577232e+00  1.7380410e-01  2.5852218e+00  1.6505203e+00
   3.0276141e+00  4.3091279e-01  2.9675424e-01  1.3369995e-01
   1.3770776e+00  1.9121900e+00  5.3814280e-01  2.1343217e+00
   4.2761779e-01  1.4323850e+00  1.8739322e+00  1.4004164e+00
   2.8567309e+00  1.0608554e+00  1.8545861e+00  1.3575516e+00
   1.7574180e+00  2.0159924e-01  1.3654881e+00  5.7198149e-01
   6.5139091e-01  2.0271444e-01  6.5128577e-01  8.4535265e-01
   2.4183602e+00  2.9378240e+00  1.8011928e+00  3.1016107e+00
   4.5313072e-01  4.6189284e-01  7.4873942e-01  3.2461333e-01
   7.7810264e-01  7.9117978e-01  2.2741288e-01  2.9509072e+00
   7.7755368e-01  9.9220872e-01  8.2671595e-01  1.6253324e+00
   7.7274686e-01  9.6307367e-01  2.7937031e+00  1.6114981e+00
   2.0271792e+00  6.1273068e-01  3.6462629e-01  7.0981568e-01
   3.6114802e+00  2.2327548e-01  2.0224471e+00  1.2702465e-01
   1.1163572e+00  5.1754677e-01  2.6523972e+00  2.5075448e-01
   1.3733222e+00  2.0993173e-01  1.7007612e+00  1.9512331e-01
   1.9070292e-01  9.9302548e-01  1.4363283e-01  1.1479352e+00
   5.7908076e-01  5.1124007e-01  3.2389059e+00  6.6253412e-01
   7.2566643e+00 -7.4917135e+00 -1.0114813e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-25 20:18:37.880970
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.9995
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-25 20:18:37.885075
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8667.37
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-25 20:18:37.888172
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.0137
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-25 20:18:37.891320
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -775.225
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139877732278568
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139876638990856
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139876638991360
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139876638991864
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139876638992368
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139876638992872

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f37d2880e10> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.473629
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.444086
grad_step = 000002, loss = 0.419813
grad_step = 000003, loss = 0.395544
grad_step = 000004, loss = 0.373240
grad_step = 000005, loss = 0.356394
grad_step = 000006, loss = 0.339358
grad_step = 000007, loss = 0.323842
grad_step = 000008, loss = 0.312189
grad_step = 000009, loss = 0.302569
grad_step = 000010, loss = 0.291072
grad_step = 000011, loss = 0.278890
grad_step = 000012, loss = 0.268096
grad_step = 000013, loss = 0.259709
grad_step = 000014, loss = 0.252579
grad_step = 000015, loss = 0.244715
grad_step = 000016, loss = 0.236529
grad_step = 000017, loss = 0.228589
grad_step = 000018, loss = 0.220384
grad_step = 000019, loss = 0.211806
grad_step = 000020, loss = 0.203853
grad_step = 000021, loss = 0.197230
grad_step = 000022, loss = 0.191071
grad_step = 000023, loss = 0.183973
grad_step = 000024, loss = 0.176326
grad_step = 000025, loss = 0.169429
grad_step = 000026, loss = 0.163276
grad_step = 000027, loss = 0.157050
grad_step = 000028, loss = 0.150684
grad_step = 000029, loss = 0.144597
grad_step = 000030, loss = 0.138844
grad_step = 000031, loss = 0.133096
grad_step = 000032, loss = 0.127431
grad_step = 000033, loss = 0.122139
grad_step = 000034, loss = 0.117065
grad_step = 000035, loss = 0.111994
grad_step = 000036, loss = 0.107022
grad_step = 000037, loss = 0.102282
grad_step = 000038, loss = 0.097676
grad_step = 000039, loss = 0.093096
grad_step = 000040, loss = 0.088729
grad_step = 000041, loss = 0.084673
grad_step = 000042, loss = 0.080681
grad_step = 000043, loss = 0.076696
grad_step = 000044, loss = 0.072985
grad_step = 000045, loss = 0.069491
grad_step = 000046, loss = 0.066051
grad_step = 000047, loss = 0.062739
grad_step = 000048, loss = 0.059567
grad_step = 000049, loss = 0.056496
grad_step = 000050, loss = 0.053547
grad_step = 000051, loss = 0.050699
grad_step = 000052, loss = 0.047963
grad_step = 000053, loss = 0.045367
grad_step = 000054, loss = 0.042868
grad_step = 000055, loss = 0.040419
grad_step = 000056, loss = 0.038093
grad_step = 000057, loss = 0.035889
grad_step = 000058, loss = 0.033734
grad_step = 000059, loss = 0.031683
grad_step = 000060, loss = 0.029716
grad_step = 000061, loss = 0.027823
grad_step = 000062, loss = 0.026021
grad_step = 000063, loss = 0.024298
grad_step = 000064, loss = 0.022665
grad_step = 000065, loss = 0.021097
grad_step = 000066, loss = 0.019595
grad_step = 000067, loss = 0.018200
grad_step = 000068, loss = 0.016869
grad_step = 000069, loss = 0.015608
grad_step = 000070, loss = 0.014423
grad_step = 000071, loss = 0.013317
grad_step = 000072, loss = 0.012285
grad_step = 000073, loss = 0.011321
grad_step = 000074, loss = 0.010438
grad_step = 000075, loss = 0.009614
grad_step = 000076, loss = 0.008854
grad_step = 000077, loss = 0.008160
grad_step = 000078, loss = 0.007518
grad_step = 000079, loss = 0.006932
grad_step = 000080, loss = 0.006399
grad_step = 000081, loss = 0.005916
grad_step = 000082, loss = 0.005472
grad_step = 000083, loss = 0.005079
grad_step = 000084, loss = 0.004722
grad_step = 000085, loss = 0.004403
grad_step = 000086, loss = 0.004121
grad_step = 000087, loss = 0.003870
grad_step = 000088, loss = 0.003647
grad_step = 000089, loss = 0.003453
grad_step = 000090, loss = 0.003280
grad_step = 000091, loss = 0.003130
grad_step = 000092, loss = 0.003000
grad_step = 000093, loss = 0.002884
grad_step = 000094, loss = 0.002785
grad_step = 000095, loss = 0.002700
grad_step = 000096, loss = 0.002625
grad_step = 000097, loss = 0.002561
grad_step = 000098, loss = 0.002505
grad_step = 000099, loss = 0.002458
grad_step = 000100, loss = 0.002418
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002383
grad_step = 000102, loss = 0.002354
grad_step = 000103, loss = 0.002329
grad_step = 000104, loss = 0.002307
grad_step = 000105, loss = 0.002289
grad_step = 000106, loss = 0.002273
grad_step = 000107, loss = 0.002260
grad_step = 000108, loss = 0.002248
grad_step = 000109, loss = 0.002238
grad_step = 000110, loss = 0.002229
grad_step = 000111, loss = 0.002221
grad_step = 000112, loss = 0.002213
grad_step = 000113, loss = 0.002207
grad_step = 000114, loss = 0.002201
grad_step = 000115, loss = 0.002196
grad_step = 000116, loss = 0.002192
grad_step = 000117, loss = 0.002191
grad_step = 000118, loss = 0.002195
grad_step = 000119, loss = 0.002205
grad_step = 000120, loss = 0.002224
grad_step = 000121, loss = 0.002224
grad_step = 000122, loss = 0.002204
grad_step = 000123, loss = 0.002165
grad_step = 000124, loss = 0.002150
grad_step = 000125, loss = 0.002164
grad_step = 000126, loss = 0.002177
grad_step = 000127, loss = 0.002169
grad_step = 000128, loss = 0.002144
grad_step = 000129, loss = 0.002131
grad_step = 000130, loss = 0.002139
grad_step = 000131, loss = 0.002147
grad_step = 000132, loss = 0.002139
grad_step = 000133, loss = 0.002122
grad_step = 000134, loss = 0.002116
grad_step = 000135, loss = 0.002121
grad_step = 000136, loss = 0.002125
grad_step = 000137, loss = 0.002119
grad_step = 000138, loss = 0.002108
grad_step = 000139, loss = 0.002102
grad_step = 000140, loss = 0.002103
grad_step = 000141, loss = 0.002106
grad_step = 000142, loss = 0.002104
grad_step = 000143, loss = 0.002097
grad_step = 000144, loss = 0.002091
grad_step = 000145, loss = 0.002088
grad_step = 000146, loss = 0.002089
grad_step = 000147, loss = 0.002089
grad_step = 000148, loss = 0.002088
grad_step = 000149, loss = 0.002084
grad_step = 000150, loss = 0.002079
grad_step = 000151, loss = 0.002075
grad_step = 000152, loss = 0.002072
grad_step = 000153, loss = 0.002070
grad_step = 000154, loss = 0.002070
grad_step = 000155, loss = 0.002069
grad_step = 000156, loss = 0.002068
grad_step = 000157, loss = 0.002067
grad_step = 000158, loss = 0.002066
grad_step = 000159, loss = 0.002065
grad_step = 000160, loss = 0.002064
grad_step = 000161, loss = 0.002062
grad_step = 000162, loss = 0.002061
grad_step = 000163, loss = 0.002060
grad_step = 000164, loss = 0.002059
grad_step = 000165, loss = 0.002057
grad_step = 000166, loss = 0.002057
grad_step = 000167, loss = 0.002057
grad_step = 000168, loss = 0.002058
grad_step = 000169, loss = 0.002058
grad_step = 000170, loss = 0.002059
grad_step = 000171, loss = 0.002057
grad_step = 000172, loss = 0.002056
grad_step = 000173, loss = 0.002051
grad_step = 000174, loss = 0.002046
grad_step = 000175, loss = 0.002038
grad_step = 000176, loss = 0.002031
grad_step = 000177, loss = 0.002024
grad_step = 000178, loss = 0.002018
grad_step = 000179, loss = 0.002014
grad_step = 000180, loss = 0.002010
grad_step = 000181, loss = 0.002008
grad_step = 000182, loss = 0.002006
grad_step = 000183, loss = 0.002004
grad_step = 000184, loss = 0.002003
grad_step = 000185, loss = 0.002003
grad_step = 000186, loss = 0.002006
grad_step = 000187, loss = 0.002015
grad_step = 000188, loss = 0.002035
grad_step = 000189, loss = 0.002081
grad_step = 000190, loss = 0.002146
grad_step = 000191, loss = 0.002244
grad_step = 000192, loss = 0.002226
grad_step = 000193, loss = 0.002130
grad_step = 000194, loss = 0.001998
grad_step = 000195, loss = 0.001991
grad_step = 000196, loss = 0.002079
grad_step = 000197, loss = 0.002106
grad_step = 000198, loss = 0.002042
grad_step = 000199, loss = 0.001968
grad_step = 000200, loss = 0.001989
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002046
grad_step = 000202, loss = 0.002031
grad_step = 000203, loss = 0.001973
grad_step = 000204, loss = 0.001955
grad_step = 000205, loss = 0.001989
grad_step = 000206, loss = 0.002007
grad_step = 000207, loss = 0.001974
grad_step = 000208, loss = 0.001944
grad_step = 000209, loss = 0.001952
grad_step = 000210, loss = 0.001972
grad_step = 000211, loss = 0.001967
grad_step = 000212, loss = 0.001941
grad_step = 000213, loss = 0.001930
grad_step = 000214, loss = 0.001940
grad_step = 000215, loss = 0.001948
grad_step = 000216, loss = 0.001939
grad_step = 000217, loss = 0.001923
grad_step = 000218, loss = 0.001916
grad_step = 000219, loss = 0.001921
grad_step = 000220, loss = 0.001925
grad_step = 000221, loss = 0.001921
grad_step = 000222, loss = 0.001910
grad_step = 000223, loss = 0.001902
grad_step = 000224, loss = 0.001900
grad_step = 000225, loss = 0.001902
grad_step = 000226, loss = 0.001903
grad_step = 000227, loss = 0.001899
grad_step = 000228, loss = 0.001893
grad_step = 000229, loss = 0.001886
grad_step = 000230, loss = 0.001881
grad_step = 000231, loss = 0.001878
grad_step = 000232, loss = 0.001877
grad_step = 000233, loss = 0.001876
grad_step = 000234, loss = 0.001876
grad_step = 000235, loss = 0.001876
grad_step = 000236, loss = 0.001875
grad_step = 000237, loss = 0.001875
grad_step = 000238, loss = 0.001874
grad_step = 000239, loss = 0.001875
grad_step = 000240, loss = 0.001876
grad_step = 000241, loss = 0.001879
grad_step = 000242, loss = 0.001883
grad_step = 000243, loss = 0.001892
grad_step = 000244, loss = 0.001902
grad_step = 000245, loss = 0.001917
grad_step = 000246, loss = 0.001931
grad_step = 000247, loss = 0.001946
grad_step = 000248, loss = 0.001949
grad_step = 000249, loss = 0.001943
grad_step = 000250, loss = 0.001914
grad_step = 000251, loss = 0.001876
grad_step = 000252, loss = 0.001840
grad_step = 000253, loss = 0.001818
grad_step = 000254, loss = 0.001812
grad_step = 000255, loss = 0.001819
grad_step = 000256, loss = 0.001834
grad_step = 000257, loss = 0.001850
grad_step = 000258, loss = 0.001863
grad_step = 000259, loss = 0.001869
grad_step = 000260, loss = 0.001870
grad_step = 000261, loss = 0.001860
grad_step = 000262, loss = 0.001845
grad_step = 000263, loss = 0.001826
grad_step = 000264, loss = 0.001808
grad_step = 000265, loss = 0.001793
grad_step = 000266, loss = 0.001781
grad_step = 000267, loss = 0.001773
grad_step = 000268, loss = 0.001769
grad_step = 000269, loss = 0.001768
grad_step = 000270, loss = 0.001770
grad_step = 000271, loss = 0.001773
grad_step = 000272, loss = 0.001781
grad_step = 000273, loss = 0.001798
grad_step = 000274, loss = 0.001829
grad_step = 000275, loss = 0.001885
grad_step = 000276, loss = 0.001972
grad_step = 000277, loss = 0.002086
grad_step = 000278, loss = 0.002154
grad_step = 000279, loss = 0.002107
grad_step = 000280, loss = 0.001926
grad_step = 000281, loss = 0.001786
grad_step = 000282, loss = 0.001785
grad_step = 000283, loss = 0.001863
grad_step = 000284, loss = 0.001910
grad_step = 000285, loss = 0.001876
grad_step = 000286, loss = 0.001795
grad_step = 000287, loss = 0.001738
grad_step = 000288, loss = 0.001746
grad_step = 000289, loss = 0.001802
grad_step = 000290, loss = 0.001822
grad_step = 000291, loss = 0.001772
grad_step = 000292, loss = 0.001710
grad_step = 000293, loss = 0.001706
grad_step = 000294, loss = 0.001739
grad_step = 000295, loss = 0.001756
grad_step = 000296, loss = 0.001742
grad_step = 000297, loss = 0.001718
grad_step = 000298, loss = 0.001702
grad_step = 000299, loss = 0.001694
grad_step = 000300, loss = 0.001692
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001699
grad_step = 000302, loss = 0.001709
grad_step = 000303, loss = 0.001707
grad_step = 000304, loss = 0.001687
grad_step = 000305, loss = 0.001665
grad_step = 000306, loss = 0.001655
grad_step = 000307, loss = 0.001656
grad_step = 000308, loss = 0.001660
grad_step = 000309, loss = 0.001662
grad_step = 000310, loss = 0.001665
grad_step = 000311, loss = 0.001673
grad_step = 000312, loss = 0.001687
grad_step = 000313, loss = 0.001699
grad_step = 000314, loss = 0.001717
grad_step = 000315, loss = 0.001729
grad_step = 000316, loss = 0.001751
grad_step = 000317, loss = 0.001757
grad_step = 000318, loss = 0.001752
grad_step = 000319, loss = 0.001720
grad_step = 000320, loss = 0.001677
grad_step = 000321, loss = 0.001636
grad_step = 000322, loss = 0.001611
grad_step = 000323, loss = 0.001606
grad_step = 000324, loss = 0.001614
grad_step = 000325, loss = 0.001627
grad_step = 000326, loss = 0.001643
grad_step = 000327, loss = 0.001666
grad_step = 000328, loss = 0.001690
grad_step = 000329, loss = 0.001718
grad_step = 000330, loss = 0.001678
grad_step = 000331, loss = 0.001637
grad_step = 000332, loss = 0.001607
grad_step = 000333, loss = 0.001608
grad_step = 000334, loss = 0.001616
grad_step = 000335, loss = 0.001596
grad_step = 000336, loss = 0.001583
grad_step = 000337, loss = 0.001593
grad_step = 000338, loss = 0.001613
grad_step = 000339, loss = 0.001625
grad_step = 000340, loss = 0.001610
grad_step = 000341, loss = 0.001590
grad_step = 000342, loss = 0.001577
grad_step = 000343, loss = 0.001575
grad_step = 000344, loss = 0.001583
grad_step = 000345, loss = 0.001591
grad_step = 000346, loss = 0.001595
grad_step = 000347, loss = 0.001583
grad_step = 000348, loss = 0.001570
grad_step = 000349, loss = 0.001557
grad_step = 000350, loss = 0.001552
grad_step = 000351, loss = 0.001553
grad_step = 000352, loss = 0.001557
grad_step = 000353, loss = 0.001561
grad_step = 000354, loss = 0.001558
grad_step = 000355, loss = 0.001552
grad_step = 000356, loss = 0.001545
grad_step = 000357, loss = 0.001542
grad_step = 000358, loss = 0.001541
grad_step = 000359, loss = 0.001544
grad_step = 000360, loss = 0.001548
grad_step = 000361, loss = 0.001551
grad_step = 000362, loss = 0.001556
grad_step = 000363, loss = 0.001567
grad_step = 000364, loss = 0.001590
grad_step = 000365, loss = 0.001639
grad_step = 000366, loss = 0.001715
grad_step = 000367, loss = 0.001839
grad_step = 000368, loss = 0.001907
grad_step = 000369, loss = 0.001884
grad_step = 000370, loss = 0.001714
grad_step = 000371, loss = 0.001586
grad_step = 000372, loss = 0.001628
grad_step = 000373, loss = 0.001677
grad_step = 000374, loss = 0.001650
grad_step = 000375, loss = 0.001561
grad_step = 000376, loss = 0.001563
grad_step = 000377, loss = 0.001626
grad_step = 000378, loss = 0.001571
grad_step = 000379, loss = 0.001550
grad_step = 000380, loss = 0.001588
grad_step = 000381, loss = 0.001544
grad_step = 000382, loss = 0.001514
grad_step = 000383, loss = 0.001551
grad_step = 000384, loss = 0.001551
grad_step = 000385, loss = 0.001515
grad_step = 000386, loss = 0.001509
grad_step = 000387, loss = 0.001522
grad_step = 000388, loss = 0.001532
grad_step = 000389, loss = 0.001515
grad_step = 000390, loss = 0.001513
grad_step = 000391, loss = 0.001516
grad_step = 000392, loss = 0.001510
grad_step = 000393, loss = 0.001497
grad_step = 000394, loss = 0.001493
grad_step = 000395, loss = 0.001502
grad_step = 000396, loss = 0.001508
grad_step = 000397, loss = 0.001504
grad_step = 000398, loss = 0.001492
grad_step = 000399, loss = 0.001487
grad_step = 000400, loss = 0.001492
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001500
grad_step = 000402, loss = 0.001503
grad_step = 000403, loss = 0.001498
grad_step = 000404, loss = 0.001492
grad_step = 000405, loss = 0.001484
grad_step = 000406, loss = 0.001479
grad_step = 000407, loss = 0.001472
grad_step = 000408, loss = 0.001467
grad_step = 000409, loss = 0.001464
grad_step = 000410, loss = 0.001465
grad_step = 000411, loss = 0.001466
grad_step = 000412, loss = 0.001464
grad_step = 000413, loss = 0.001462
grad_step = 000414, loss = 0.001460
grad_step = 000415, loss = 0.001461
grad_step = 000416, loss = 0.001462
grad_step = 000417, loss = 0.001462
grad_step = 000418, loss = 0.001462
grad_step = 000419, loss = 0.001461
grad_step = 000420, loss = 0.001461
grad_step = 000421, loss = 0.001461
grad_step = 000422, loss = 0.001461
grad_step = 000423, loss = 0.001460
grad_step = 000424, loss = 0.001459
grad_step = 000425, loss = 0.001458
grad_step = 000426, loss = 0.001457
grad_step = 000427, loss = 0.001459
grad_step = 000428, loss = 0.001462
grad_step = 000429, loss = 0.001466
grad_step = 000430, loss = 0.001472
grad_step = 000431, loss = 0.001479
grad_step = 000432, loss = 0.001486
grad_step = 000433, loss = 0.001496
grad_step = 000434, loss = 0.001503
grad_step = 000435, loss = 0.001510
grad_step = 000436, loss = 0.001507
grad_step = 000437, loss = 0.001499
grad_step = 000438, loss = 0.001482
grad_step = 000439, loss = 0.001466
grad_step = 000440, loss = 0.001453
grad_step = 000441, loss = 0.001454
grad_step = 000442, loss = 0.001465
grad_step = 000443, loss = 0.001484
grad_step = 000444, loss = 0.001489
grad_step = 000445, loss = 0.001479
grad_step = 000446, loss = 0.001449
grad_step = 000447, loss = 0.001417
grad_step = 000448, loss = 0.001400
grad_step = 000449, loss = 0.001405
grad_step = 000450, loss = 0.001421
grad_step = 000451, loss = 0.001432
grad_step = 000452, loss = 0.001428
grad_step = 000453, loss = 0.001411
grad_step = 000454, loss = 0.001393
grad_step = 000455, loss = 0.001384
grad_step = 000456, loss = 0.001386
grad_step = 000457, loss = 0.001394
grad_step = 000458, loss = 0.001397
grad_step = 000459, loss = 0.001393
grad_step = 000460, loss = 0.001384
grad_step = 000461, loss = 0.001374
grad_step = 000462, loss = 0.001370
grad_step = 000463, loss = 0.001370
grad_step = 000464, loss = 0.001373
grad_step = 000465, loss = 0.001376
grad_step = 000466, loss = 0.001377
grad_step = 000467, loss = 0.001375
grad_step = 000468, loss = 0.001377
grad_step = 000469, loss = 0.001386
grad_step = 000470, loss = 0.001413
grad_step = 000471, loss = 0.001464
grad_step = 000472, loss = 0.001576
grad_step = 000473, loss = 0.001720
grad_step = 000474, loss = 0.001951
grad_step = 000475, loss = 0.002025
grad_step = 000476, loss = 0.001960
grad_step = 000477, loss = 0.001737
grad_step = 000478, loss = 0.001550
grad_step = 000479, loss = 0.001537
grad_step = 000480, loss = 0.001504
grad_step = 000481, loss = 0.001552
grad_step = 000482, loss = 0.001611
grad_step = 000483, loss = 0.001451
grad_step = 000484, loss = 0.001439
grad_step = 000485, loss = 0.001556
grad_step = 000486, loss = 0.001439
grad_step = 000487, loss = 0.001343
grad_step = 000488, loss = 0.001417
grad_step = 000489, loss = 0.001445
grad_step = 000490, loss = 0.001387
grad_step = 000491, loss = 0.001345
grad_step = 000492, loss = 0.001372
grad_step = 000493, loss = 0.001385
grad_step = 000494, loss = 0.001357
grad_step = 000495, loss = 0.001356
grad_step = 000496, loss = 0.001363
grad_step = 000497, loss = 0.001335
grad_step = 000498, loss = 0.001334
grad_step = 000499, loss = 0.001354
grad_step = 000500, loss = 0.001333
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001312
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

  date_run                              2020-05-25 20:18:55.828642
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.275373
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-25 20:18:55.834182
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.181843
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-25 20:18:55.841061
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.163311
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-25 20:18:55.846458
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.76317
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
100%|██████████| 10/10 [00:02<00:00,  3.80it/s, avg_epoch_loss=5.28]
INFO:root:Epoch[0] Elapsed time 2.636 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.278419
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.27841854095459 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f37eeb40e48> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  7.91it/s, avg_epoch_loss=3.59e+3]
INFO:root:Epoch[0] Elapsed time 1.266 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=3590.403646
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 3590.4036458333335 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f3790269f28> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
 30%|███       | 3/10 [00:13<00:30,  4.34s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:24<00:16,  4.17s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:35<00:04,  4.05s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:39<00:00,  3.94s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 39.375 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.860164
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.860164165496826 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f37901cd588> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  5.17it/s, avg_epoch_loss=5.84]
INFO:root:Epoch[0] Elapsed time 1.937 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.836530
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.836530065536499 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f371d0b6e48> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
 10%|█         | 1/10 [01:55<17:19, 115.45s/it, avg_epoch_loss=0.412] 20%|██        | 2/10 [04:44<17:31, 131.39s/it, avg_epoch_loss=0.399] 30%|███       | 3/10 [07:41<16:57, 145.30s/it, avg_epoch_loss=0.39]  40%|████      | 4/10 [10:30<15:14, 152.35s/it, avg_epoch_loss=0.385] 50%|█████     | 5/10 [13:30<13:23, 160.65s/it, avg_epoch_loss=0.384] 60%|██████    | 6/10 [16:47<11:25, 171.42s/it, avg_epoch_loss=0.383] 70%|███████   | 7/10 [20:08<09:01, 180.49s/it, avg_epoch_loss=0.381] 80%|████████  | 8/10 [23:45<06:22, 191.43s/it, avg_epoch_loss=0.378] 90%|█████████ | 9/10 [27:20<03:18, 198.40s/it, avg_epoch_loss=0.376]100%|██████████| 10/10 [30:35<00:00, 197.54s/it, avg_epoch_loss=0.375]100%|██████████| 10/10 [30:36<00:00, 183.60s/it, avg_epoch_loss=0.375]
INFO:root:Epoch[0] Elapsed time 1836.027 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.374571
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3745713621377945 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f371d06a128> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:02<00:00,  4.80it/s, avg_epoch_loss=415]
INFO:root:Epoch[0] Elapsed time 2.102 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=414.652022
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 414.65202175008733 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f37d294ff98> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:00<00:00, 46.24it/s, avg_epoch_loss=5.18]
INFO:root:Epoch[0] Elapsed time 0.217 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.177984
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.1779844760894775 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f37d2a6d940> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
0   2020-05-25 20:18:27.926220  ...    mean_absolute_error
1   2020-05-25 20:18:27.929288  ...     mean_squared_error
2   2020-05-25 20:18:27.932300  ...  median_absolute_error
3   2020-05-25 20:18:27.935185  ...               r2_score
4   2020-05-25 20:18:37.880970  ...    mean_absolute_error
5   2020-05-25 20:18:37.885075  ...     mean_squared_error
6   2020-05-25 20:18:37.888172  ...  median_absolute_error
7   2020-05-25 20:18:37.891320  ...               r2_score
8   2020-05-25 20:18:55.828642  ...    mean_absolute_error
9   2020-05-25 20:18:55.834182  ...     mean_squared_error
10  2020-05-25 20:18:55.841061  ...  median_absolute_error
11  2020-05-25 20:18:55.846458  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f40f4ac2be0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f40a6a3ca20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f40a747dd30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f40a6a3ca20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f40f4ac2be0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f40a6a3ca20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f40a747dd30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f40a6a3ca20> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f40f4ac2be0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f40a6a3ca20> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f40a747dd30> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fda1c8e6080> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=e86eb6ae7cd4166ed0c1ee12b5bd3c990fc8a426db5359a529153eb105b0d511
  Stored in directory: /tmp/pip-ephem-wheel-cache-lid5xrpo/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fd9b46e25f8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
  335872/17464789 [..............................] - ETA: 2s
  770048/17464789 [>.............................] - ETA: 2s
 1204224/17464789 [=>............................] - ETA: 2s
 1671168/17464789 [=>............................] - ETA: 2s
 2129920/17464789 [==>...........................] - ETA: 1s
 2596864/17464789 [===>..........................] - ETA: 1s
 3047424/17464789 [====>.........................] - ETA: 1s
 3522560/17464789 [=====>........................] - ETA: 1s
 4071424/17464789 [=====>........................] - ETA: 1s
 4653056/17464789 [======>.......................] - ETA: 1s
 5292032/17464789 [========>.....................] - ETA: 1s
 6012928/17464789 [=========>....................] - ETA: 1s
 6791168/17464789 [==========>...................] - ETA: 1s
 7593984/17464789 [============>.................] - ETA: 0s
 8445952/17464789 [=============>................] - ETA: 0s
 9437184/17464789 [===============>..............] - ETA: 0s
10428416/17464789 [================>.............] - ETA: 0s
11493376/17464789 [==================>...........] - ETA: 0s
12673024/17464789 [====================>.........] - ETA: 0s
13787136/17464789 [======================>.......] - ETA: 0s
14958592/17464789 [========================>.....] - ETA: 0s
16244736/17464789 [==========================>...] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-25 20:52:05.627240: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-25 20:52:05.639164: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-25 20:52:05.639348: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bb281a6990 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-25 20:52:05.639365: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6820 - accuracy: 0.4990 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.8200 - accuracy: 0.4900
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7970 - accuracy: 0.4915
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7740 - accuracy: 0.4930
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7970 - accuracy: 0.4915
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7981 - accuracy: 0.4914
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7490 - accuracy: 0.4946
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7399 - accuracy: 0.4952
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7310 - accuracy: 0.4958
11000/25000 [============>.................] - ETA: 3s - loss: 7.7405 - accuracy: 0.4952
12000/25000 [=============>................] - ETA: 3s - loss: 7.7676 - accuracy: 0.4934
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7622 - accuracy: 0.4938
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7400 - accuracy: 0.4952
15000/25000 [=================>............] - ETA: 2s - loss: 7.7556 - accuracy: 0.4942
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7625 - accuracy: 0.4938
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7397 - accuracy: 0.4952
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7152 - accuracy: 0.4968
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7013 - accuracy: 0.4977
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6996 - accuracy: 0.4979
21000/25000 [========================>.....] - ETA: 0s - loss: 7.7002 - accuracy: 0.4978
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6820 - accuracy: 0.4990
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6706 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6628 - accuracy: 0.5002
25000/25000 [==============================] - 7s 281us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-25 20:52:19.181512
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-25 20:52:19.181512  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<143:56:53, 1.66kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:05<100:59:29, 2.37kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:05<70:44:07, 3.38kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:05<49:29:23, 4.83kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:05<34:32:09, 6.91kB/s].vector_cache/glove.6B.zip:   1%|          | 8.90M/862M [00:05<24:01:42, 9.86kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:05<16:45:22, 14.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.1M/862M [00:05<11:39:50, 20.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.0M/862M [00:05<8:07:42, 28.7kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.0M/862M [00:05<5:39:27, 41.1kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:06<3:56:42, 58.6kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.7M/862M [00:06<2:44:46, 83.7kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:06<1:54:58, 119kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 43.5M/862M [00:06<1:20:02, 170kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.7M/862M [00:06<55:56, 243kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 51.8M/862M [00:06<38:59, 346kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:07<30:35, 441kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:09<23:13, 578kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:09<18:30, 725kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:09<13:24, 1.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:09<09:32, 1.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:11<15:48, 845kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:11<12:31, 1.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:11<09:02, 1.47MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:13<09:18, 1.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:13<09:12, 1.44MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:13<07:06, 1.87MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:13<05:06, 2.59MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:15<1:39:23, 133kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:15<1:10:52, 186kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:15<49:51, 264kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:17<37:53, 347kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:17<29:13, 450kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:17<21:00, 625kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:17<14:51, 881kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:19<15:55, 822kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:19<12:28, 1.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:19<08:59, 1.45MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:20<09:20, 1.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:21<09:10, 1.42MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:21<06:58, 1.86MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:21<05:03, 2.56MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:22<08:58, 1.44MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:23<07:37, 1.70MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.5M/862M [00:23<05:39, 2.28MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:24<06:58, 1.85MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:25<07:29, 1.72MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:25<05:48, 2.22MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:25<04:13, 3.04MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:26<09:50, 1.30MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:26<08:11, 1.56MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:27<06:00, 2.13MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:28<07:11, 1.77MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:28<07:31, 1.69MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:29<05:54, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<04:15, 2.97MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<12:16:03, 17.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<8:36:17, 24.5kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:31<6:00:56, 35.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<4:14:54, 49.4kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<3:00:56, 69.6kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:32<2:07:04, 99.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:33<1:28:51, 141kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<1:08:52, 182kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:34<49:27, 253kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<34:50, 359kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<27:15, 457kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<21:36, 577kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:36<15:38, 796kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:36<11:05, 1.12MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<13:44, 902kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<10:52, 1.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:38<07:51, 1.57MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<08:23, 1.47MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<08:22, 1.47MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<06:28, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<04:39, 2.64MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<11:52:00, 17.2kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<8:19:25, 24.5kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:42<5:49:07, 35.0kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<4:06:32, 49.4kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<2:55:00, 69.6kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<2:02:59, 99.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:44<1:25:54, 141kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<1:15:29, 161kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<54:03, 224kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:46<38:03, 318kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<29:23, 410kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<23:02, 523kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:48<16:38, 723kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<11:44, 1.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<21:33, 556kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<16:19, 733kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:50<11:40, 1.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<10:56, 1.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<10:04, 1.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:52<07:39, 1.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:54<07:15, 1.63MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:54<06:16, 1.89MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:54<04:38, 2.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:56<06:01, 1.95MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:56<05:24, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:56<04:02, 2.91MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<05:36, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<05:06, 2.29MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:58<03:52, 3.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<05:27, 2.13MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<05:01, 2.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [01:00<03:46, 3.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:02<05:23, 2.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:02<04:54, 2.35MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:02<03:43, 3.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:04<05:19, 2.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:04<04:53, 2.35MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:04<03:39, 3.13MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:06<05:17, 2.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:06<04:51, 2.35MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:06<03:37, 3.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:08<05:14, 2.17MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:08<05:58, 1.90MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:08<04:40, 2.43MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<03:23, 3.34MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:10<11:31, 980kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:10<09:14, 1.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:10<06:41, 1.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<07:18, 1.54MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:12<06:03, 1.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:12<04:27, 2.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<03:18, 3.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<1:08:52, 162kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:14<50:27, 221kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:14<35:44, 312kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:14<25:09, 442kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<21:20, 520kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:16<16:05, 689kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:16<11:31, 960kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<10:36, 1.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<08:31, 1.29MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:18<06:13, 1.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<06:56, 1.58MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<07:04, 1.55MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:20<05:29, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<05:36, 1.94MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<06:07, 1.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:22<04:46, 2.28MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:22<03:27, 3.13MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<09:22, 1.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<07:39, 1.41MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<05:37, 1.92MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<06:27, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<06:41, 1.61MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<05:08, 2.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:26<03:43, 2.88MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:27<08:59, 1.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<07:22, 1.45MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:27<05:25, 1.96MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<06:16, 1.69MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<05:27, 1.94MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:29<04:04, 2.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<05:21, 1.97MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<04:49, 2.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<03:37, 2.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<05:01, 2.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<04:33, 2.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:33<03:24, 3.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<04:51, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<04:27, 2.33MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<03:22, 3.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<04:48, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<04:24, 2.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:37<03:20, 3.09MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<04:45, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<04:22, 2.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:39<03:16, 3.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<04:43, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<04:20, 2.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:41<03:17, 3.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:43<04:42, 2.15MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<04:19, 2.34MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:43<03:16, 3.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:45<04:40, 2.16MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:45<05:18, 1.89MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<04:13, 2.38MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:47<04:33, 2.19MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:47<04:12, 2.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:47<03:09, 3.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<04:32, 2.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<04:11, 2.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:49<03:10, 3.12MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<04:33, 2.17MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<05:13, 1.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:51<04:06, 2.39MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:51<02:58, 3.29MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<14:03, 696kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<10:50, 902kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:53<07:47, 1.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<07:43, 1.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<06:14, 1.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:55<04:36, 2.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:57<05:29, 1.76MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:57<05:48, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<04:32, 2.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:59<04:43, 2.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:59<05:19, 1.80MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<04:08, 2.31MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:59<03:16, 2.92MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:01<04:29, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<04:08, 2.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:01<03:06, 3.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:03<04:19, 2.18MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<04:51, 1.94MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<03:52, 2.43MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:04<04:14, 2.21MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<04:57, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<03:57, 2.37MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:05<02:52, 3.24MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<31:37, 294kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<23:03, 403kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:07<16:20, 568kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:08<13:33, 682kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:09<11:26, 808kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<08:24, 1.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:09<05:58, 1.54MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<10:42, 856kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<08:26, 1.09MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:11<06:07, 1.49MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<06:23, 1.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<06:20, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<04:53, 1.86MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:13<03:30, 2.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:14<8:45:10, 17.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:14<6:08:17, 24.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<4:17:15, 35.0kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:16<3:01:28, 49.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:16<2:08:48, 69.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<1:30:29, 98.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:18<1:04:24, 138kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<45:58, 193kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:18<32:17, 275kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:20<24:34, 359kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<18:05, 488kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:20<12:51, 684kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<11:02, 794kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<08:36, 1.02MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:22<06:13, 1.40MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<06:24, 1.36MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<06:14, 1.39MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:24<04:48, 1.81MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:26<04:44, 1.82MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:26<04:11, 2.05MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:26<03:09, 2.73MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:28<04:12, 2.04MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:28<03:47, 2.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:28<02:50, 3.00MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:30<04:00, 2.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:30<04:31, 1.88MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<03:31, 2.41MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:30<02:33, 3.29MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:32<06:27, 1.30MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:32<05:22, 1.56MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:32<03:56, 2.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:34<04:42, 1.77MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<04:08, 2.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:34<03:06, 2.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:36<04:07, 2.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<04:33, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<03:34, 2.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:36<02:35, 3.16MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:38<1:00:53, 135kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<43:26, 189kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:38<30:31, 268kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<23:11, 351kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<17:52, 455kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:40<12:50, 632kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:40<09:04, 891kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:42<09:45, 828kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:42<07:39, 1.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:42<05:32, 1.45MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:44<05:44, 1.39MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:44<04:41, 1.71MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:44<03:30, 2.28MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:44<02:31, 3.14MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:46<55:08, 144kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:46<40:13, 197kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<28:26, 279kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:46<19:57, 395kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:48<16:56, 464kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:48<12:38, 622kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:48<09:01, 869kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:50<08:07, 960kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<06:20, 1.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<04:34, 1.70MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:50<03:20, 2.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:52<08:55, 867kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:52<07:01, 1.10MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:52<05:06, 1.51MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:54<05:20, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:54<05:22, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:54<04:05, 1.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:54<02:59, 2.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:55<04:30, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:56<03:56, 1.93MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:56<02:54, 2.59MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:57<03:48, 1.98MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:58<04:11, 1.80MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<03:18, 2.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:59<03:30, 2.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:00<03:05, 2.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:00<02:18, 3.22MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<01:44, 4.26MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:01<45:15, 163kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:01<33:09, 223kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:02<23:29, 314kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<16:27, 446kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:03<15:42, 466kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:03<11:43, 624kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:04<08:20, 873kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:05<07:29, 968kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:05<06:39, 1.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:06<04:57, 1.46MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:06<03:32, 2.03MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:07<07:21, 976kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:07<05:53, 1.22MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<04:15, 1.68MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:09<04:39, 1.53MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<03:58, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:09<02:57, 2.40MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:11<03:43, 1.89MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:11<03:19, 2.12MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<02:29, 2.81MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:13<03:23, 2.06MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:13<02:57, 2.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<02:12, 3.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<01:39, 4.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:15<52:42, 131kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:15<37:33, 184kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<26:20, 261kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<19:58, 342kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<14:39, 466kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<10:23, 655kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:19<08:50, 766kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:19<07:33, 896kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:19<05:37, 1.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<03:58, 1.69MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:21<6:31:26, 17.1kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:21<4:34:25, 24.4kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<3:11:31, 34.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:23<2:14:54, 49.2kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:23<1:35:43, 69.3kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:23<1:07:13, 98.5kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<46:46, 140kB/s]   .vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:25<58:03, 113kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<41:09, 159kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:25<28:56, 226kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:25<20:12, 322kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:27<52:52, 123kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:27<38:16, 170kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:27<27:01, 240kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:27<18:51, 341kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:29<20:37, 312kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:29<15:05, 426kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<10:40, 599kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:31<08:55, 712kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:31<07:35, 837kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<05:35, 1.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:31<03:58, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:33<35:47, 176kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:33<25:40, 245kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:33<18:03, 347kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:35<14:01, 444kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:35<11:04, 562kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<08:02, 771kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:37<06:35, 935kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:37<05:51, 1.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:37<04:24, 1.39MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:37<03:08, 1.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:39<45:43, 133kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:39<32:36, 186kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<22:53, 265kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:41<17:19, 347kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:41<13:20, 451kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:41<09:34, 626kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:41<06:45, 882kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:43<07:01, 847kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:43<05:31, 1.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:43<03:58, 1.49MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:45<04:09, 1.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:45<03:30, 1.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<02:35, 2.26MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:46<03:10, 1.83MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:47<03:23, 1.71MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:47<02:37, 2.21MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:47<01:53, 3.04MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:48<05:37, 1.02MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:49<04:31, 1.27MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:49<03:16, 1.74MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:50<03:37, 1.57MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:51<03:40, 1.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:51<02:51, 1.98MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:52<02:53, 1.94MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:52<02:31, 2.22MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:53<01:54, 2.93MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:54<02:35, 2.13MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:54<02:56, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:55<02:19, 2.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:56<02:30, 2.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:56<02:18, 2.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<01:43, 3.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:58<02:28, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:58<02:17, 2.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<01:42, 3.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:00<02:26, 2.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:00<02:47, 1.91MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<02:10, 2.44MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:01<01:34, 3.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:02<04:38, 1.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:02<03:41, 1.42MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<02:40, 1.96MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:03<01:57, 2.66MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:04<29:31, 176kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:04<21:42, 239kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:04<15:24, 336kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:05<10:55, 472kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:06<08:46, 584kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:06<06:40, 766kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:06<04:46, 1.07MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:08<04:28, 1.13MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:08<03:38, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<02:39, 1.89MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:10<03:01, 1.65MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:10<03:07, 1.60MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<02:23, 2.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:10<01:43, 2.86MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:12<05:01, 978kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:12<04:01, 1.22MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<02:54, 1.68MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:14<03:10, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:14<02:42, 1.79MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:14<02:00, 2.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:16<02:31, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<02:15, 2.12MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<01:41, 2.81MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:18<02:17, 2.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:18<02:04, 2.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<01:34, 2.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:20<02:11, 2.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:20<02:00, 2.31MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<01:29, 3.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:22<02:07, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:22<02:24, 1.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:22<01:55, 2.38MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<01:22, 3.28MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:24<4:22:25, 17.2kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:24<3:03:54, 24.5kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<2:08:05, 34.9kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:26<1:29:59, 49.3kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:26<1:03:54, 69.4kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<44:47, 98.8kB/s]  .vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:26<31:05, 141kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:28<25:41, 170kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:28<18:24, 237kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<12:55, 336kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:30<09:58, 431kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:30<07:51, 548kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:30<05:40, 756kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:30<03:59, 1.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:32<05:09, 822kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:32<04:02, 1.05MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:32<02:55, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:34<02:59, 1.39MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:34<02:56, 1.42MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:34<02:15, 1.83MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:36<02:13, 1.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:36<01:58, 2.07MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<01:28, 2.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:37<01:58, 2.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:38<02:13, 1.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:38<01:45, 2.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:38<01:16, 3.12MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:39<29:13, 136kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:40<20:49, 190kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<14:34, 270kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:41<11:01, 353kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:42<08:05, 480kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<05:43, 674kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:43<04:52, 784kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:44<04:11, 912kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:44<03:07, 1.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<02:11, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:45<32:37, 115kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:45<23:10, 162kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<16:12, 230kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:47<12:05, 305kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:47<09:14, 399kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:48<06:38, 553kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<04:37, 783kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:49<16:19, 222kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:49<11:46, 307kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<08:15, 434kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:51<06:33, 541kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:51<05:18, 668kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<03:52, 910kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:53<03:14, 1.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:53<02:37, 1.32MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<01:54, 1.80MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:55<02:07, 1.61MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:55<02:10, 1.56MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:55<01:41, 2.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:56<01:12, 2.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:57<24:44, 135kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:57<17:37, 189kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:57<12:19, 269kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:59<09:18, 352kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:59<07:10, 456kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:59<05:09, 633kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<03:35, 893kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:01<3:07:19, 17.1kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:01<2:11:11, 24.4kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<1:31:10, 34.8kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:03<1:03:49, 49.2kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:03<44:54, 69.7kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<31:13, 99.4kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:05<22:20, 137kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:05<16:15, 189kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:05<11:27, 267kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<07:56, 379kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:07<08:04, 371kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:07<05:57, 503kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<04:11, 707kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:09<03:35, 816kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:09<03:06, 943kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:09<02:17, 1.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:09<01:37, 1.78MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:11<03:08, 911kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:11<02:29, 1.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<01:47, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:13<01:53, 1.47MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:13<01:54, 1.47MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:13<01:27, 1.90MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:15<01:26, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:15<01:17, 2.11MB/s].vector_cache/glove.6B.zip:  81%|████████  | 701M/862M [05:15<00:57, 2.80MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:17<01:17, 2.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:17<01:27, 1.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<01:08, 2.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<00:49, 3.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:19<01:27, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:19<01:17, 2.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<00:57, 2.66MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:21<01:14, 2.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:21<01:23, 1.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<01:04, 2.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<00:46, 3.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:23<01:44, 1.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:23<01:28, 1.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<01:04, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:25<01:18, 1.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:25<01:23, 1.71MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:25<01:05, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<00:46, 2.97MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:27<13:47, 168kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:27<09:51, 234kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<06:52, 332kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:28<05:16, 426kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:29<03:54, 574kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<02:45, 802kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:30<02:24, 903kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:31<02:07, 1.02MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:31<01:34, 1.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:31<01:06, 1.91MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:32<01:42, 1.24MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:33<01:24, 1.50MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:01, 2.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:34<01:10, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:34<01:14, 1.65MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:35<00:57, 2.10MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:35<00:40, 2.91MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:36<1:54:49, 17.2kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:36<1:20:18, 24.5kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:37<55:31, 34.9kB/s]  .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:38<38:35, 49.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:38<27:06, 70.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<18:45, 99.7kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:40<13:19, 138kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:40<09:40, 189kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:40<06:48, 267kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<04:40, 380kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:42<04:29, 393kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:42<03:19, 530kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<02:20, 743kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:44<01:59, 850kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:44<01:33, 1.08MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:44<01:07, 1.48MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:46<01:09, 1.41MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:46<00:58, 1.67MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:46<00:42, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:48<00:51, 1.83MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:48<00:54, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:48<00:41, 2.21MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<00:29, 3.04MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:50<01:19, 1.13MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:50<01:04, 1.38MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<00:46, 1.87MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:52<00:52, 1.64MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:52<00:53, 1.59MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<00:41, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:54<00:41, 1.97MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:54<00:36, 2.20MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:54<00:27, 2.93MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:56<00:36, 2.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:56<00:40, 1.89MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<00:32, 2.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:58<00:33, 2.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:58<00:38, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:58<00:30, 2.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<00:21, 3.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:00<00:49, 1.39MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:00<00:40, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:00<00:29, 2.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:02<00:35, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 798M/862M [06:02<00:37, 1.72MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:02<00:28, 2.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:20, 3.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:04<00:41, 1.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:04<00:34, 1.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:24, 2.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<00:17, 3.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:06<05:50, 161kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:06<04:09, 225kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:06<02:51, 319kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:08<02:07, 412kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:08<01:39, 526kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:08<01:11, 723kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:47, 1.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:10<47:06, 17.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:10<32:47, 24.4kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:10<22:14, 34.8kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:12<15:00, 49.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:12<10:29, 69.8kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<07:05, 99.4kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:14<04:52, 137kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:14<03:26, 192kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<02:19, 273kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:16<01:41, 357kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:16<01:13, 485kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<00:49, 683kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:18<00:40, 791kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:18<00:34, 919kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:18<00:25, 1.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:20, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:20<00:16, 1.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<00:11, 2.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:13, 1.81MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:22<00:13, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:22<00:10, 2.20MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:06, 3.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:23<00:16, 1.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:24<00:13, 1.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:08, 2.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:25<00:09, 1.72MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:26<00:09, 1.64MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:26<00:06, 2.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<00:03, 2.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<11:02, 17.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<07:28, 24.5kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:28<04:34, 35.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<02:27, 49.9kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<02:25, 50.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<01:36, 71.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:30<00:52, 101kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<00:22, 140kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<00:15, 192kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:32<00:08, 271kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 385kB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.20MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<70:31:32,  1.58it/s]  0%|          | 871/400000 [00:00<49:15:51,  2.25it/s]  0%|          | 1660/400000 [00:00<34:25:16,  3.21it/s]  1%|          | 2493/400000 [00:00<24:02:54,  4.59it/s]  1%|          | 3323/400000 [00:01<16:48:09,  6.56it/s]  1%|          | 4193/400000 [00:01<11:44:23,  9.37it/s]  1%|▏         | 5061/400000 [00:01<8:12:13, 13.37it/s]   1%|▏         | 5929/400000 [00:01<5:44:01, 19.09it/s]  2%|▏         | 6809/400000 [00:01<4:00:30, 27.25it/s]  2%|▏         | 7694/400000 [00:01<2:48:11, 38.87it/s]  2%|▏         | 8559/400000 [00:01<1:57:42, 55.43it/s]  2%|▏         | 9438/400000 [00:01<1:22:25, 78.97it/s]  3%|▎         | 10305/400000 [00:01<57:47, 112.37it/s]  3%|▎         | 11175/400000 [00:01<40:35, 159.65it/s]  3%|▎         | 12041/400000 [00:02<28:34, 226.28it/s]  3%|▎         | 12904/400000 [00:02<20:11, 319.61it/s]  3%|▎         | 13773/400000 [00:02<14:19, 449.49it/s]  4%|▎         | 14635/400000 [00:02<10:13, 627.64it/s]  4%|▍         | 15510/400000 [00:02<07:22, 869.87it/s]  4%|▍         | 16385/400000 [00:02<05:21, 1191.76it/s]  4%|▍         | 17249/400000 [00:02<03:58, 1606.89it/s]  5%|▍         | 18111/400000 [00:02<03:00, 2115.08it/s]  5%|▍         | 18968/400000 [00:02<02:19, 2732.49it/s]  5%|▍         | 19833/400000 [00:02<01:50, 3437.90it/s]  5%|▌         | 20705/400000 [00:03<01:30, 4201.12it/s]  5%|▌         | 21580/400000 [00:03<01:16, 4976.63it/s]  6%|▌         | 22444/400000 [00:03<01:06, 5700.94it/s]  6%|▌         | 23308/400000 [00:03<00:59, 6326.29it/s]  6%|▌         | 24183/400000 [00:03<00:54, 6899.49it/s]  6%|▋         | 25055/400000 [00:03<00:50, 7359.87it/s]  6%|▋         | 25926/400000 [00:03<00:48, 7716.73it/s]  7%|▋         | 26800/400000 [00:03<00:46, 7997.38it/s]  7%|▋         | 27670/400000 [00:03<00:46, 8061.75it/s]  7%|▋         | 28527/400000 [00:03<00:45, 8205.52it/s]  7%|▋         | 29405/400000 [00:04<00:44, 8368.68it/s]  8%|▊         | 30278/400000 [00:04<00:43, 8473.89it/s]  8%|▊         | 31144/400000 [00:04<00:43, 8509.42it/s]  8%|▊         | 32008/400000 [00:04<00:44, 8315.60it/s]  8%|▊         | 32850/400000 [00:04<00:44, 8290.05it/s]  8%|▊         | 33723/400000 [00:04<00:43, 8416.12it/s]  9%|▊         | 34585/400000 [00:04<00:43, 8473.68it/s]  9%|▉         | 35443/400000 [00:04<00:42, 8502.70it/s]  9%|▉         | 36296/400000 [00:04<00:42, 8503.82it/s]  9%|▉         | 37159/400000 [00:04<00:42, 8538.70it/s] 10%|▉         | 38028/400000 [00:05<00:42, 8581.74it/s] 10%|▉         | 38914/400000 [00:05<00:41, 8661.32it/s] 10%|▉         | 39790/400000 [00:05<00:41, 8687.91it/s] 10%|█         | 40660/400000 [00:05<00:41, 8682.63it/s] 10%|█         | 41542/400000 [00:05<00:41, 8723.25it/s] 11%|█         | 42415/400000 [00:05<00:41, 8699.78it/s] 11%|█         | 43298/400000 [00:05<00:40, 8736.08it/s] 11%|█         | 44172/400000 [00:05<00:40, 8700.78it/s] 11%|█▏        | 45043/400000 [00:05<00:40, 8679.83it/s] 11%|█▏        | 45912/400000 [00:05<00:41, 8630.07it/s] 12%|█▏        | 46780/400000 [00:06<00:40, 8640.04it/s] 12%|█▏        | 47650/400000 [00:06<00:40, 8656.82it/s] 12%|█▏        | 48524/400000 [00:06<00:40, 8680.63it/s] 12%|█▏        | 49393/400000 [00:06<00:40, 8635.85it/s] 13%|█▎        | 50270/400000 [00:06<00:40, 8672.77it/s] 13%|█▎        | 51165/400000 [00:06<00:39, 8753.43it/s] 13%|█▎        | 52041/400000 [00:06<00:39, 8725.25it/s] 13%|█▎        | 52919/400000 [00:06<00:39, 8740.45it/s] 13%|█▎        | 53794/400000 [00:06<00:39, 8706.31it/s] 14%|█▎        | 54665/400000 [00:06<00:39, 8663.74it/s] 14%|█▍        | 55537/400000 [00:07<00:39, 8680.13it/s] 14%|█▍        | 56406/400000 [00:07<00:39, 8662.24it/s] 14%|█▍        | 57277/400000 [00:07<00:39, 8675.66it/s] 15%|█▍        | 58145/400000 [00:07<00:39, 8642.31it/s] 15%|█▍        | 59010/400000 [00:07<00:40, 8408.74it/s] 15%|█▍        | 59853/400000 [00:07<00:42, 7941.12it/s] 15%|█▌        | 60679/400000 [00:07<00:42, 8032.20it/s] 15%|█▌        | 61534/400000 [00:07<00:41, 8179.44it/s] 16%|█▌        | 62389/400000 [00:07<00:40, 8285.78it/s] 16%|█▌        | 63262/400000 [00:08<00:40, 8413.25it/s] 16%|█▌        | 64144/400000 [00:08<00:39, 8530.29it/s] 16%|█▋        | 65032/400000 [00:08<00:38, 8632.03it/s] 16%|█▋        | 65907/400000 [00:08<00:38, 8665.98it/s] 17%|█▋        | 66775/400000 [00:08<00:38, 8668.98it/s] 17%|█▋        | 67644/400000 [00:08<00:38, 8672.94it/s] 17%|█▋        | 68520/400000 [00:08<00:38, 8696.44it/s] 17%|█▋        | 69391/400000 [00:08<00:38, 8669.28it/s] 18%|█▊        | 70259/400000 [00:08<00:38, 8605.34it/s] 18%|█▊        | 71120/400000 [00:08<00:38, 8587.00it/s] 18%|█▊        | 71997/400000 [00:09<00:37, 8640.68it/s] 18%|█▊        | 72876/400000 [00:09<00:37, 8682.63it/s] 18%|█▊        | 73754/400000 [00:09<00:37, 8709.20it/s] 19%|█▊        | 74626/400000 [00:09<00:37, 8606.60it/s] 19%|█▉        | 75488/400000 [00:09<00:38, 8462.21it/s] 19%|█▉        | 76336/400000 [00:09<00:38, 8411.35it/s] 19%|█▉        | 77217/400000 [00:09<00:37, 8526.12it/s] 20%|█▉        | 78093/400000 [00:09<00:37, 8593.17it/s] 20%|█▉        | 78970/400000 [00:09<00:37, 8642.57it/s] 20%|█▉        | 79835/400000 [00:09<00:37, 8610.82it/s] 20%|██        | 80697/400000 [00:10<00:37, 8464.21it/s] 20%|██        | 81570/400000 [00:10<00:37, 8542.07it/s] 21%|██        | 82425/400000 [00:10<00:37, 8539.05it/s] 21%|██        | 83286/400000 [00:10<00:36, 8560.19it/s] 21%|██        | 84146/400000 [00:10<00:36, 8571.63it/s] 21%|██▏       | 85006/400000 [00:10<00:36, 8577.76it/s] 21%|██▏       | 85864/400000 [00:10<00:36, 8569.00it/s] 22%|██▏       | 86735/400000 [00:10<00:36, 8609.99it/s] 22%|██▏       | 87605/400000 [00:10<00:36, 8636.14it/s] 22%|██▏       | 88469/400000 [00:10<00:36, 8608.49it/s] 22%|██▏       | 89330/400000 [00:11<00:36, 8520.31it/s] 23%|██▎       | 90183/400000 [00:11<00:37, 8325.59it/s] 23%|██▎       | 91050/400000 [00:11<00:36, 8424.14it/s] 23%|██▎       | 91915/400000 [00:11<00:36, 8489.78it/s] 23%|██▎       | 92786/400000 [00:11<00:35, 8554.40it/s] 23%|██▎       | 93657/400000 [00:11<00:35, 8599.45it/s] 24%|██▎       | 94532/400000 [00:11<00:35, 8641.76it/s] 24%|██▍       | 95429/400000 [00:11<00:34, 8736.83it/s] 24%|██▍       | 96311/400000 [00:11<00:34, 8759.20it/s] 24%|██▍       | 97188/400000 [00:11<00:34, 8762.32it/s] 25%|██▍       | 98065/400000 [00:12<00:34, 8715.02it/s] 25%|██▍       | 98937/400000 [00:12<00:34, 8688.37it/s] 25%|██▍       | 99816/400000 [00:12<00:34, 8718.46it/s] 25%|██▌       | 100689/400000 [00:12<00:34, 8721.59it/s] 25%|██▌       | 101568/400000 [00:12<00:34, 8739.34it/s] 26%|██▌       | 102443/400000 [00:12<00:34, 8649.41it/s] 26%|██▌       | 103323/400000 [00:12<00:34, 8693.47it/s] 26%|██▌       | 104193/400000 [00:12<00:35, 8219.58it/s] 26%|██▋       | 105038/400000 [00:12<00:35, 8285.50it/s] 26%|██▋       | 105899/400000 [00:12<00:35, 8378.09it/s] 27%|██▋       | 106740/400000 [00:13<00:35, 8335.63it/s] 27%|██▋       | 107598/400000 [00:13<00:34, 8404.45it/s] 27%|██▋       | 108495/400000 [00:13<00:34, 8564.76it/s] 27%|██▋       | 109362/400000 [00:13<00:33, 8593.84it/s] 28%|██▊       | 110223/400000 [00:13<00:33, 8591.76it/s] 28%|██▊       | 111084/400000 [00:13<00:33, 8582.70it/s] 28%|██▊       | 111946/400000 [00:13<00:33, 8592.41it/s] 28%|██▊       | 112815/400000 [00:13<00:33, 8618.80it/s] 28%|██▊       | 113680/400000 [00:13<00:33, 8627.00it/s] 29%|██▊       | 114543/400000 [00:13<00:33, 8616.08it/s] 29%|██▉       | 115405/400000 [00:14<00:34, 8282.59it/s] 29%|██▉       | 116242/400000 [00:14<00:34, 8307.05it/s] 29%|██▉       | 117130/400000 [00:14<00:33, 8470.54it/s] 29%|██▉       | 117980/400000 [00:14<00:33, 8355.10it/s] 30%|██▉       | 118818/400000 [00:14<00:33, 8330.29it/s] 30%|██▉       | 119685/400000 [00:14<00:33, 8428.59it/s] 30%|███       | 120544/400000 [00:14<00:32, 8474.24it/s] 30%|███       | 121414/400000 [00:14<00:32, 8540.37it/s] 31%|███       | 122292/400000 [00:14<00:32, 8608.20it/s] 31%|███       | 123156/400000 [00:15<00:32, 8615.02it/s] 31%|███       | 124018/400000 [00:15<00:32, 8594.47it/s] 31%|███       | 124886/400000 [00:15<00:31, 8619.85it/s] 31%|███▏      | 125756/400000 [00:15<00:31, 8641.87it/s] 32%|███▏      | 126633/400000 [00:15<00:31, 8677.53it/s] 32%|███▏      | 127520/400000 [00:15<00:31, 8733.43it/s] 32%|███▏      | 128394/400000 [00:15<00:31, 8727.78it/s] 32%|███▏      | 129282/400000 [00:15<00:30, 8772.52it/s] 33%|███▎      | 130160/400000 [00:15<00:30, 8731.81it/s] 33%|███▎      | 131034/400000 [00:15<00:30, 8730.44it/s] 33%|███▎      | 131908/400000 [00:16<00:30, 8705.33it/s] 33%|███▎      | 132779/400000 [00:16<00:31, 8509.40it/s] 33%|███▎      | 133631/400000 [00:16<00:31, 8475.69it/s] 34%|███▎      | 134483/400000 [00:16<00:31, 8488.89it/s] 34%|███▍      | 135350/400000 [00:16<00:30, 8539.53it/s] 34%|███▍      | 136205/400000 [00:16<00:30, 8510.69it/s] 34%|███▍      | 137057/400000 [00:16<00:31, 8471.11it/s] 34%|███▍      | 137924/400000 [00:16<00:30, 8528.15it/s] 35%|███▍      | 138778/400000 [00:16<00:30, 8524.37it/s] 35%|███▍      | 139633/400000 [00:16<00:30, 8532.02it/s] 35%|███▌      | 140487/400000 [00:17<00:30, 8496.47it/s] 35%|███▌      | 141340/400000 [00:17<00:30, 8504.31it/s] 36%|███▌      | 142221/400000 [00:17<00:30, 8591.46it/s] 36%|███▌      | 143090/400000 [00:17<00:29, 8618.52it/s] 36%|███▌      | 143955/400000 [00:17<00:29, 8626.47it/s] 36%|███▌      | 144827/400000 [00:17<00:29, 8651.77it/s] 36%|███▋      | 145693/400000 [00:17<00:29, 8608.64it/s] 37%|███▋      | 146555/400000 [00:17<00:29, 8482.15it/s] 37%|███▋      | 147404/400000 [00:17<00:29, 8442.35it/s] 37%|███▋      | 148278/400000 [00:17<00:29, 8528.58it/s] 37%|███▋      | 149150/400000 [00:18<00:29, 8582.58it/s] 38%|███▊      | 150011/400000 [00:18<00:29, 8585.99it/s] 38%|███▊      | 150870/400000 [00:18<00:29, 8578.24it/s] 38%|███▊      | 151732/400000 [00:18<00:28, 8588.28it/s] 38%|███▊      | 152591/400000 [00:18<00:28, 8564.10it/s] 38%|███▊      | 153450/400000 [00:18<00:28, 8569.16it/s] 39%|███▊      | 154308/400000 [00:18<00:28, 8549.53it/s] 39%|███▉      | 155178/400000 [00:18<00:28, 8593.13it/s] 39%|███▉      | 156047/400000 [00:18<00:28, 8619.13it/s] 39%|███▉      | 156926/400000 [00:18<00:28, 8667.64it/s] 39%|███▉      | 157813/400000 [00:19<00:27, 8725.44it/s] 40%|███▉      | 158686/400000 [00:19<00:27, 8708.08it/s] 40%|███▉      | 159559/400000 [00:19<00:27, 8713.90it/s] 40%|████      | 160432/400000 [00:19<00:27, 8717.31it/s] 40%|████      | 161304/400000 [00:19<00:27, 8566.94it/s] 41%|████      | 162162/400000 [00:19<00:27, 8540.49it/s] 41%|████      | 163017/400000 [00:19<00:27, 8540.40it/s] 41%|████      | 163894/400000 [00:19<00:27, 8607.11it/s] 41%|████      | 164777/400000 [00:19<00:27, 8672.20it/s] 41%|████▏     | 165655/400000 [00:19<00:26, 8703.17it/s] 42%|████▏     | 166540/400000 [00:20<00:26, 8743.80it/s] 42%|████▏     | 167415/400000 [00:20<00:26, 8716.72it/s] 42%|████▏     | 168287/400000 [00:20<00:26, 8688.01it/s] 42%|████▏     | 169169/400000 [00:20<00:26, 8726.54it/s] 43%|████▎     | 170059/400000 [00:20<00:26, 8776.06it/s] 43%|████▎     | 170948/400000 [00:20<00:25, 8809.75it/s] 43%|████▎     | 171830/400000 [00:20<00:25, 8809.65it/s] 43%|████▎     | 172712/400000 [00:20<00:25, 8753.71it/s] 43%|████▎     | 173608/400000 [00:20<00:25, 8812.18it/s] 44%|████▎     | 174494/400000 [00:20<00:25, 8825.09it/s] 44%|████▍     | 175377/400000 [00:21<00:25, 8771.86it/s] 44%|████▍     | 176255/400000 [00:21<00:26, 8598.56it/s] 44%|████▍     | 177135/400000 [00:21<00:25, 8657.02it/s] 45%|████▍     | 178011/400000 [00:21<00:25, 8685.17it/s] 45%|████▍     | 178881/400000 [00:21<00:25, 8671.82it/s] 45%|████▍     | 179765/400000 [00:21<00:25, 8719.09it/s] 45%|████▌     | 180638/400000 [00:21<00:25, 8501.19it/s] 45%|████▌     | 181524/400000 [00:21<00:25, 8605.01it/s] 46%|████▌     | 182395/400000 [00:21<00:25, 8635.32it/s] 46%|████▌     | 183260/400000 [00:21<00:25, 8632.39it/s] 46%|████▌     | 184140/400000 [00:22<00:24, 8679.71it/s] 46%|████▋     | 185009/400000 [00:22<00:24, 8670.39it/s] 46%|████▋     | 185877/400000 [00:22<00:24, 8654.97it/s] 47%|████▋     | 186758/400000 [00:22<00:24, 8700.00it/s] 47%|████▋     | 187635/400000 [00:22<00:24, 8720.61it/s] 47%|████▋     | 188518/400000 [00:22<00:24, 8751.06it/s] 47%|████▋     | 189394/400000 [00:22<00:24, 8678.96it/s] 48%|████▊     | 190263/400000 [00:22<00:24, 8645.11it/s] 48%|████▊     | 191132/400000 [00:22<00:24, 8656.14it/s] 48%|████▊     | 191998/400000 [00:22<00:24, 8649.18it/s] 48%|████▊     | 192889/400000 [00:23<00:23, 8724.89it/s] 48%|████▊     | 193762/400000 [00:23<00:23, 8686.83it/s] 49%|████▊     | 194631/400000 [00:23<00:23, 8687.59it/s] 49%|████▉     | 195500/400000 [00:23<00:23, 8605.26it/s] 49%|████▉     | 196374/400000 [00:23<00:23, 8645.07it/s] 49%|████▉     | 197256/400000 [00:23<00:23, 8696.66it/s] 50%|████▉     | 198128/400000 [00:23<00:23, 8703.06it/s] 50%|████▉     | 199000/400000 [00:23<00:23, 8705.92it/s] 50%|████▉     | 199883/400000 [00:23<00:22, 8742.05it/s] 50%|█████     | 200758/400000 [00:23<00:22, 8740.03it/s] 50%|█████     | 201656/400000 [00:24<00:22, 8808.26it/s] 51%|█████     | 202538/400000 [00:24<00:22, 8767.56it/s] 51%|█████     | 203415/400000 [00:24<00:22, 8723.34it/s] 51%|█████     | 204288/400000 [00:24<00:22, 8629.46it/s] 51%|█████▏    | 205152/400000 [00:24<00:23, 8424.04it/s] 52%|█████▏    | 206013/400000 [00:24<00:22, 8476.50it/s] 52%|█████▏    | 206882/400000 [00:24<00:22, 8537.18it/s] 52%|█████▏    | 207755/400000 [00:24<00:22, 8593.75it/s] 52%|█████▏    | 208620/400000 [00:24<00:22, 8609.70it/s] 52%|█████▏    | 209489/400000 [00:24<00:22, 8631.14it/s] 53%|█████▎    | 210353/400000 [00:25<00:22, 8365.62it/s] 53%|█████▎    | 211192/400000 [00:25<00:22, 8326.19it/s] 53%|█████▎    | 212027/400000 [00:25<00:23, 7975.05it/s] 53%|█████▎    | 212906/400000 [00:25<00:22, 8203.09it/s] 53%|█████▎    | 213782/400000 [00:25<00:22, 8360.38it/s] 54%|█████▎    | 214645/400000 [00:25<00:21, 8439.47it/s] 54%|█████▍    | 215492/400000 [00:25<00:22, 8361.25it/s] 54%|█████▍    | 216367/400000 [00:25<00:21, 8472.16it/s] 54%|█████▍    | 217234/400000 [00:25<00:21, 8528.60it/s] 55%|█████▍    | 218095/400000 [00:26<00:21, 8550.20it/s] 55%|█████▍    | 218952/400000 [00:26<00:21, 8504.38it/s] 55%|█████▍    | 219819/400000 [00:26<00:21, 8550.77it/s] 55%|█████▌    | 220699/400000 [00:26<00:20, 8622.47it/s] 55%|█████▌    | 221576/400000 [00:26<00:20, 8665.67it/s] 56%|█████▌    | 222458/400000 [00:26<00:20, 8709.75it/s] 56%|█████▌    | 223348/400000 [00:26<00:20, 8763.36it/s] 56%|█████▌    | 224225/400000 [00:26<00:20, 8677.98it/s] 56%|█████▋    | 225094/400000 [00:26<00:20, 8640.69it/s] 56%|█████▋    | 225961/400000 [00:26<00:20, 8648.21it/s] 57%|█████▋    | 226850/400000 [00:27<00:19, 8717.21it/s] 57%|█████▋    | 227730/400000 [00:27<00:19, 8741.67it/s] 57%|█████▋    | 228605/400000 [00:27<00:19, 8741.84it/s] 57%|█████▋    | 229501/400000 [00:27<00:19, 8805.41it/s] 58%|█████▊    | 230391/400000 [00:27<00:19, 8831.70it/s] 58%|█████▊    | 231275/400000 [00:27<00:19, 8822.78it/s] 58%|█████▊    | 232158/400000 [00:27<00:19, 8783.91it/s] 58%|█████▊    | 233037/400000 [00:27<00:19, 8770.25it/s] 58%|█████▊    | 233915/400000 [00:27<00:19, 8733.34it/s] 59%|█████▊    | 234789/400000 [00:27<00:19, 8689.02it/s] 59%|█████▉    | 235659/400000 [00:28<00:18, 8678.29it/s] 59%|█████▉    | 236541/400000 [00:28<00:18, 8719.79it/s] 59%|█████▉    | 237414/400000 [00:28<00:18, 8706.06it/s] 60%|█████▉    | 238285/400000 [00:28<00:18, 8698.47it/s] 60%|█████▉    | 239159/400000 [00:28<00:18, 8710.13it/s] 60%|██████    | 240043/400000 [00:28<00:18, 8746.58it/s] 60%|██████    | 240918/400000 [00:28<00:18, 8573.82it/s] 60%|██████    | 241789/400000 [00:28<00:18, 8613.31it/s] 61%|██████    | 242659/400000 [00:28<00:18, 8638.83it/s] 61%|██████    | 243547/400000 [00:28<00:17, 8709.43it/s] 61%|██████    | 244435/400000 [00:29<00:17, 8759.58it/s] 61%|██████▏   | 245336/400000 [00:29<00:17, 8831.50it/s] 62%|██████▏   | 246220/400000 [00:29<00:17, 8806.52it/s] 62%|██████▏   | 247101/400000 [00:29<00:17, 8806.82it/s] 62%|██████▏   | 247982/400000 [00:29<00:17, 8570.41it/s] 62%|██████▏   | 248841/400000 [00:29<00:17, 8542.97it/s] 62%|██████▏   | 249697/400000 [00:29<00:17, 8460.36it/s] 63%|██████▎   | 250563/400000 [00:29<00:17, 8516.95it/s] 63%|██████▎   | 251439/400000 [00:29<00:17, 8586.14it/s] 63%|██████▎   | 252299/400000 [00:29<00:17, 8581.65it/s] 63%|██████▎   | 253171/400000 [00:30<00:17, 8621.17it/s] 64%|██████▎   | 254034/400000 [00:30<00:16, 8613.54it/s] 64%|██████▎   | 254905/400000 [00:30<00:16, 8641.52it/s] 64%|██████▍   | 255770/400000 [00:30<00:16, 8605.13it/s] 64%|██████▍   | 256646/400000 [00:30<00:16, 8648.68it/s] 64%|██████▍   | 257512/400000 [00:30<00:16, 8479.32it/s] 65%|██████▍   | 258391/400000 [00:30<00:16, 8570.10it/s] 65%|██████▍   | 259249/400000 [00:30<00:16, 8473.08it/s] 65%|██████▌   | 260098/400000 [00:30<00:16, 8465.66it/s] 65%|██████▌   | 260982/400000 [00:30<00:16, 8572.87it/s] 65%|██████▌   | 261842/400000 [00:31<00:16, 8578.21it/s] 66%|██████▌   | 262701/400000 [00:31<00:16, 8373.60it/s] 66%|██████▌   | 263587/400000 [00:31<00:16, 8511.30it/s] 66%|██████▌   | 264442/400000 [00:31<00:15, 8520.35it/s] 66%|██████▋   | 265311/400000 [00:31<00:15, 8568.36it/s] 67%|██████▋   | 266201/400000 [00:31<00:15, 8663.89it/s] 67%|██████▋   | 267090/400000 [00:31<00:15, 8729.38it/s] 67%|██████▋   | 267964/400000 [00:31<00:15, 8717.71it/s] 67%|██████▋   | 268837/400000 [00:31<00:15, 8596.28it/s] 67%|██████▋   | 269720/400000 [00:31<00:15, 8662.60it/s] 68%|██████▊   | 270608/400000 [00:32<00:14, 8723.96it/s] 68%|██████▊   | 271499/400000 [00:32<00:14, 8778.35it/s] 68%|██████▊   | 272378/400000 [00:32<00:14, 8663.98it/s] 68%|██████▊   | 273246/400000 [00:32<00:14, 8661.18it/s] 69%|██████▊   | 274140/400000 [00:32<00:14, 8740.53it/s] 69%|██████▉   | 275024/400000 [00:32<00:14, 8768.06it/s] 69%|██████▉   | 275906/400000 [00:32<00:14, 8782.45it/s] 69%|██████▉   | 276785/400000 [00:32<00:14, 8480.25it/s] 69%|██████▉   | 277636/400000 [00:32<00:14, 8396.59it/s] 70%|██████▉   | 278499/400000 [00:33<00:14, 8464.50it/s] 70%|██████▉   | 279377/400000 [00:33<00:14, 8553.86it/s] 70%|███████   | 280251/400000 [00:33<00:13, 8606.81it/s] 70%|███████   | 281133/400000 [00:33<00:13, 8669.36it/s] 71%|███████   | 282001/400000 [00:33<00:13, 8642.22it/s] 71%|███████   | 282866/400000 [00:33<00:13, 8639.36it/s] 71%|███████   | 283731/400000 [00:33<00:13, 8607.53it/s] 71%|███████   | 284593/400000 [00:33<00:13, 8587.82it/s] 71%|███████▏  | 285460/400000 [00:33<00:13, 8610.68it/s] 72%|███████▏  | 286322/400000 [00:33<00:13, 8591.77it/s] 72%|███████▏  | 287210/400000 [00:34<00:13, 8675.13it/s] 72%|███████▏  | 288093/400000 [00:34<00:12, 8720.35it/s] 72%|███████▏  | 288981/400000 [00:34<00:12, 8765.65it/s] 72%|███████▏  | 289862/400000 [00:34<00:12, 8778.73it/s] 73%|███████▎  | 290741/400000 [00:34<00:12, 8647.99it/s] 73%|███████▎  | 291607/400000 [00:34<00:12, 8447.52it/s] 73%|███████▎  | 292475/400000 [00:34<00:12, 8515.77it/s] 73%|███████▎  | 293339/400000 [00:34<00:12, 8550.52it/s] 74%|███████▎  | 294201/400000 [00:34<00:12, 8568.92it/s] 74%|███████▍  | 295059/400000 [00:34<00:12, 8501.21it/s] 74%|███████▍  | 295932/400000 [00:35<00:12, 8566.73it/s] 74%|███████▍  | 296803/400000 [00:35<00:11, 8607.16it/s] 74%|███████▍  | 297686/400000 [00:35<00:11, 8671.98it/s] 75%|███████▍  | 298554/400000 [00:35<00:11, 8660.43it/s] 75%|███████▍  | 299423/400000 [00:35<00:11, 8668.50it/s] 75%|███████▌  | 300308/400000 [00:35<00:11, 8719.64it/s] 75%|███████▌  | 301198/400000 [00:35<00:11, 8770.98it/s] 76%|███████▌  | 302079/400000 [00:35<00:11, 8782.35it/s] 76%|███████▌  | 302969/400000 [00:35<00:11, 8815.41it/s] 76%|███████▌  | 303851/400000 [00:35<00:11, 8737.29it/s] 76%|███████▌  | 304725/400000 [00:36<00:10, 8705.79it/s] 76%|███████▋  | 305596/400000 [00:36<00:11, 8516.97it/s] 77%|███████▋  | 306452/400000 [00:36<00:10, 8527.42it/s] 77%|███████▋  | 307325/400000 [00:36<00:10, 8586.40it/s] 77%|███████▋  | 308185/400000 [00:36<00:10, 8444.55it/s] 77%|███████▋  | 309039/400000 [00:36<00:10, 8472.68it/s] 77%|███████▋  | 309909/400000 [00:36<00:10, 8538.01it/s] 78%|███████▊  | 310764/400000 [00:36<00:10, 8513.09it/s] 78%|███████▊  | 311619/400000 [00:36<00:10, 8524.07it/s] 78%|███████▊  | 312483/400000 [00:36<00:10, 8556.70it/s] 78%|███████▊  | 313347/400000 [00:37<00:10, 8580.48it/s] 79%|███████▊  | 314218/400000 [00:37<00:09, 8616.23it/s] 79%|███████▉  | 315096/400000 [00:37<00:09, 8663.92it/s] 79%|███████▉  | 315965/400000 [00:37<00:09, 8670.99it/s] 79%|███████▉  | 316833/400000 [00:37<00:09, 8496.44it/s] 79%|███████▉  | 317684/400000 [00:37<00:09, 8462.84it/s] 80%|███████▉  | 318557/400000 [00:37<00:09, 8539.56it/s] 80%|███████▉  | 319428/400000 [00:37<00:09, 8587.64it/s] 80%|████████  | 320288/400000 [00:37<00:09, 8558.77it/s] 80%|████████  | 321145/400000 [00:37<00:09, 8513.35it/s] 81%|████████  | 322005/400000 [00:38<00:09, 8537.87it/s] 81%|████████  | 322864/400000 [00:38<00:09, 8552.04it/s] 81%|████████  | 323732/400000 [00:38<00:08, 8588.86it/s] 81%|████████  | 324592/400000 [00:38<00:08, 8535.57it/s] 81%|████████▏ | 325451/400000 [00:38<00:08, 8550.32it/s] 82%|████████▏ | 326323/400000 [00:38<00:08, 8598.74it/s] 82%|████████▏ | 327184/400000 [00:38<00:08, 8527.13it/s] 82%|████████▏ | 328061/400000 [00:38<00:08, 8597.02it/s] 82%|████████▏ | 328933/400000 [00:38<00:08, 8632.91it/s] 82%|████████▏ | 329797/400000 [00:38<00:08, 8438.16it/s] 83%|████████▎ | 330662/400000 [00:39<00:08, 8498.78it/s] 83%|████████▎ | 331520/400000 [00:39<00:08, 8521.11it/s] 83%|████████▎ | 332380/400000 [00:39<00:07, 8543.05it/s] 83%|████████▎ | 333247/400000 [00:39<00:07, 8580.26it/s] 84%|████████▎ | 334106/400000 [00:39<00:07, 8547.97it/s] 84%|████████▎ | 334962/400000 [00:39<00:07, 8515.87it/s] 84%|████████▍ | 335826/400000 [00:39<00:07, 8552.40it/s] 84%|████████▍ | 336712/400000 [00:39<00:07, 8640.95it/s] 84%|████████▍ | 337601/400000 [00:39<00:07, 8713.32it/s] 85%|████████▍ | 338473/400000 [00:39<00:07, 8711.47it/s] 85%|████████▍ | 339348/400000 [00:40<00:06, 8721.22it/s] 85%|████████▌ | 340238/400000 [00:40<00:06, 8772.15it/s] 85%|████████▌ | 341116/400000 [00:40<00:06, 8760.63it/s] 86%|████████▌ | 342009/400000 [00:40<00:06, 8808.59it/s] 86%|████████▌ | 342891/400000 [00:40<00:06, 8806.50it/s] 86%|████████▌ | 343772/400000 [00:40<00:06, 8437.81it/s] 86%|████████▌ | 344643/400000 [00:40<00:06, 8515.32it/s] 86%|████████▋ | 345537/400000 [00:40<00:06, 8637.54it/s] 87%|████████▋ | 346408/400000 [00:40<00:06, 8657.00it/s] 87%|████████▋ | 347276/400000 [00:40<00:06, 8661.95it/s] 87%|████████▋ | 348146/400000 [00:41<00:05, 8672.21it/s] 87%|████████▋ | 349014/400000 [00:41<00:05, 8641.71it/s] 87%|████████▋ | 349879/400000 [00:41<00:05, 8566.10it/s] 88%|████████▊ | 350763/400000 [00:41<00:05, 8644.82it/s] 88%|████████▊ | 351638/400000 [00:41<00:05, 8674.58it/s] 88%|████████▊ | 352525/400000 [00:41<00:05, 8731.30it/s] 88%|████████▊ | 353408/400000 [00:41<00:05, 8760.57it/s] 89%|████████▊ | 354291/400000 [00:41<00:05, 8779.56it/s] 89%|████████▉ | 355170/400000 [00:41<00:05, 8751.04it/s] 89%|████████▉ | 356046/400000 [00:42<00:05, 8689.08it/s] 89%|████████▉ | 356916/400000 [00:42<00:04, 8639.88it/s] 89%|████████▉ | 357784/400000 [00:42<00:04, 8650.64it/s] 90%|████████▉ | 358681/400000 [00:42<00:04, 8743.75it/s] 90%|████████▉ | 359569/400000 [00:42<00:04, 8781.88it/s] 90%|█████████ | 360448/400000 [00:42<00:04, 8764.31it/s] 90%|█████████ | 361325/400000 [00:42<00:04, 8574.03it/s] 91%|█████████ | 362184/400000 [00:42<00:04, 8527.10it/s] 91%|█████████ | 363063/400000 [00:42<00:04, 8603.48it/s] 91%|█████████ | 363928/400000 [00:42<00:04, 8616.87it/s] 91%|█████████ | 364791/400000 [00:43<00:04, 8499.94it/s] 91%|█████████▏| 365642/400000 [00:43<00:04, 8406.63it/s] 92%|█████████▏| 366515/400000 [00:43<00:03, 8499.07it/s] 92%|█████████▏| 367389/400000 [00:43<00:03, 8567.86it/s] 92%|█████████▏| 368253/400000 [00:43<00:03, 8588.33it/s] 92%|█████████▏| 369113/400000 [00:43<00:03, 8568.18it/s] 92%|█████████▏| 369971/400000 [00:43<00:03, 8564.86it/s] 93%|█████████▎| 370848/400000 [00:43<00:03, 8625.17it/s] 93%|█████████▎| 371717/400000 [00:43<00:03, 8643.04it/s] 93%|█████████▎| 372582/400000 [00:43<00:03, 8641.72it/s] 93%|█████████▎| 373451/400000 [00:44<00:03, 8654.51it/s] 94%|█████████▎| 374317/400000 [00:44<00:02, 8618.09it/s] 94%|█████████▍| 375179/400000 [00:44<00:02, 8572.48it/s] 94%|█████████▍| 376041/400000 [00:44<00:02, 8584.28it/s] 94%|█████████▍| 376900/400000 [00:44<00:02, 8560.89it/s] 94%|█████████▍| 377757/400000 [00:44<00:02, 8373.29it/s] 95%|█████████▍| 378608/400000 [00:44<00:02, 8412.40it/s] 95%|█████████▍| 379479/400000 [00:44<00:02, 8498.94it/s] 95%|█████████▌| 380361/400000 [00:44<00:02, 8591.50it/s] 95%|█████████▌| 381235/400000 [00:44<00:02, 8633.70it/s] 96%|█████████▌| 382111/400000 [00:45<00:02, 8671.07it/s] 96%|█████████▌| 382979/400000 [00:45<00:01, 8649.71it/s] 96%|█████████▌| 383845/400000 [00:45<00:01, 8650.13it/s] 96%|█████████▌| 384715/400000 [00:45<00:01, 8664.35it/s] 96%|█████████▋| 385591/400000 [00:45<00:01, 8692.60it/s] 97%|█████████▋| 386468/400000 [00:45<00:01, 8715.19it/s] 97%|█████████▋| 387340/400000 [00:45<00:01, 8699.40it/s] 97%|█████████▋| 388211/400000 [00:45<00:01, 8700.04it/s] 97%|█████████▋| 389092/400000 [00:45<00:01, 8730.40it/s] 97%|█████████▋| 389966/400000 [00:45<00:01, 8727.76it/s] 98%|█████████▊| 390839/400000 [00:46<00:01, 8719.11it/s] 98%|█████████▊| 391711/400000 [00:46<00:00, 8658.52it/s] 98%|█████████▊| 392586/400000 [00:46<00:00, 8684.32it/s] 98%|█████████▊| 393467/400000 [00:46<00:00, 8719.03it/s] 99%|█████████▊| 394340/400000 [00:46<00:00, 8693.60it/s] 99%|█████████▉| 395218/400000 [00:46<00:00, 8719.09it/s] 99%|█████████▉| 396090/400000 [00:46<00:00, 8677.61it/s] 99%|█████████▉| 396972/400000 [00:46<00:00, 8718.24it/s] 99%|█████████▉| 397850/400000 [00:46<00:00, 8734.35it/s]100%|█████████▉| 398730/400000 [00:46<00:00, 8753.83it/s]100%|█████████▉| 399606/400000 [00:47<00:00, 8730.80it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8493.69it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f993ef52978> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010981444892965162 	 Accuracy: 56
Train Epoch: 1 	 Loss: 0.011304781987116886 	 Accuracy: 54

  model saves at 54% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15840 out of table with 15817 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15840 out of table with 15817 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-25 21:01:31.338320: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-25 21:01:31.342209: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-25 21:01:31.342331: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b21f8740f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-25 21:01:31.342343: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f98ea80b0f0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.4980 - accuracy: 0.5110
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6206 - accuracy: 0.5030 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6206 - accuracy: 0.5030
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6015 - accuracy: 0.5042
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5961 - accuracy: 0.5046
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6002 - accuracy: 0.5043
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6403 - accuracy: 0.5017
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6628 - accuracy: 0.5002
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6428 - accuracy: 0.5016
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6482 - accuracy: 0.5012
11000/25000 [============>.................] - ETA: 3s - loss: 7.6527 - accuracy: 0.5009
12000/25000 [=============>................] - ETA: 3s - loss: 7.6577 - accuracy: 0.5006
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6725 - accuracy: 0.4996
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6732 - accuracy: 0.4996
15000/25000 [=================>............] - ETA: 2s - loss: 7.6768 - accuracy: 0.4993
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6829 - accuracy: 0.4989
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6720 - accuracy: 0.4996
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6717 - accuracy: 0.4997
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6586 - accuracy: 0.5005
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6551 - accuracy: 0.5008
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6557 - accuracy: 0.5007
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6687 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6813 - accuracy: 0.4990
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6826 - accuracy: 0.4990
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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f98ac5a5860> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f98ea80b0f0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 3s 3s/step - loss: 1.2736 - crf_viterbi_accuracy: 0.3333 - val_loss: 1.3015 - val_crf_viterbi_accuracy: 0.2800

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

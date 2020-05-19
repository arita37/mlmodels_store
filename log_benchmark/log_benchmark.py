
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '73f54da32a5da4768415eb9105ad096255137679', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/73f54da32a5da4768415eb9105ad096255137679

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/73f54da32a5da4768415eb9105ad096255137679

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fa6bb444e80> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-19 08:14:24.468998
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-19 08:14:24.472751
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-19 08:14:24.476211
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-19 08:14:24.479772
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fa6bb1c8a58> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356205.4375
Epoch 2/10

1/1 [==============================] - 0s 114ms/step - loss: 280919.7500
Epoch 3/10

1/1 [==============================] - 0s 110ms/step - loss: 193960.7969
Epoch 4/10

1/1 [==============================] - 0s 110ms/step - loss: 117953.8125
Epoch 5/10

1/1 [==============================] - 0s 109ms/step - loss: 68366.2188
Epoch 6/10

1/1 [==============================] - 0s 101ms/step - loss: 40797.9805
Epoch 7/10

1/1 [==============================] - 0s 107ms/step - loss: 25387.6348
Epoch 8/10

1/1 [==============================] - 0s 101ms/step - loss: 16715.9980
Epoch 9/10

1/1 [==============================] - 0s 99ms/step - loss: 11806.6445
Epoch 10/10

1/1 [==============================] - 0s 103ms/step - loss: 8850.9336

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.30077913e-01  7.61766243e+00  5.19895601e+00  5.31935501e+00
   7.50051212e+00  7.25887012e+00  7.53937769e+00  7.34339046e+00
   6.10836458e+00  6.80659389e+00  6.47840548e+00  6.85901117e+00
   4.55206728e+00  8.28784084e+00  6.13435888e+00  6.14340591e+00
   7.11636639e+00  6.59264421e+00  7.80150795e+00  6.63936758e+00
   6.20499277e+00  7.12220764e+00  6.37816477e+00  5.47230673e+00
   6.85406160e+00  6.96719837e+00  6.11612368e+00  6.36430025e+00
   6.51712275e+00  6.48813581e+00  7.81557274e+00  8.15991497e+00
   5.14056587e+00  6.68512678e+00  7.68457651e+00  7.04089260e+00
   7.11744308e+00  8.13958740e+00  7.86168051e+00  6.85369205e+00
   7.39719534e+00  6.28740454e+00  7.02028942e+00  6.15390301e+00
   7.07239485e+00  6.54129887e+00  7.26904058e+00  6.40060091e+00
   6.05745173e+00  7.31857681e+00  6.87500668e+00  7.67347193e+00
   5.80395222e+00  6.31024790e+00  5.85214472e+00  7.70157719e+00
   7.26420498e+00  5.96045876e+00  7.33417988e+00  5.50056314e+00
   1.20347345e+00 -7.47853994e-01 -1.20037723e+00 -6.44025445e-01
  -7.93910265e-01  3.75807524e-01 -9.95898426e-01 -4.33488578e-01
   2.50163972e-02  1.24163055e+00  8.54119658e-03 -4.37383592e-01
  -6.59193039e-01  4.86371636e-01 -9.94662642e-01  3.60006988e-01
  -1.79346323e-01  2.10508317e-01  6.40808225e-01 -6.21665835e-01
   1.02076566e+00  9.07991409e-01  1.66188610e+00  2.10277438e-01
  -3.58055413e-01  2.30604291e-01 -1.70483446e+00 -1.32862782e+00
   1.93519294e-01 -8.93563092e-01 -1.33367801e+00  1.41864169e+00
  -1.62195832e-01 -2.49184072e-01  5.47192037e-01  2.64444768e-01
   1.48345542e+00 -4.56800759e-01  4.46122944e-01  1.91772354e+00
   1.02283669e+00  2.31832623e-01 -8.77430081e-01  8.25912356e-01
   1.22606301e+00  2.77262837e-01 -7.29849517e-01  4.77295309e-01
   3.05084765e-01  1.98305392e+00  1.25401056e+00 -3.38312328e-01
   5.20275414e-01  3.08932275e-01  5.98136067e-01  3.73522997e-01
   7.78388306e-02  1.77169526e+00  6.43648326e-01 -1.72907162e+00
  -2.00515890e+00 -7.41804481e-01 -8.52877319e-01  8.53468537e-01
  -9.49333012e-01 -2.53361851e-01  5.03421724e-02 -2.05686510e-01
   5.97984910e-01  9.90384221e-01  2.85141885e-01 -3.54231268e-01
   9.53196883e-01  1.16303539e+00  3.94055486e-01  1.85213852e+00
   1.95855647e-03  4.24930871e-01  8.42292786e-01  1.17425549e+00
  -1.04705453e+00  1.49677634e+00  1.29174316e+00  1.07917655e+00
   2.31869549e-01 -1.65182829e+00 -7.32374012e-01 -4.65548038e-03
  -1.31772256e+00  5.83773792e-01 -1.58893788e+00 -6.30043626e-01
  -6.68281019e-01 -2.37118199e-01  6.96740985e-01  5.76719999e-01
   2.26341105e+00  5.74176550e-01 -1.16482759e+00  3.12546551e-01
  -8.41585398e-02 -1.35116041e+00 -8.12011719e-01 -4.47642326e-01
  -1.79634988e-01  1.09147334e+00  4.36634779e-01 -7.57138848e-01
   1.58444390e-01 -3.79294544e-01  9.57014143e-01 -1.54474831e+00
  -4.91650611e-01  2.75437057e-01 -6.53291702e-01  9.57976520e-01
   7.22727180e-02 -2.46224210e-01  9.29888427e-01  1.27696311e+00
   8.28626752e-02  5.69465351e+00  6.40227222e+00  6.99754381e+00
   7.59873343e+00  6.19169760e+00  7.67849493e+00  5.30731773e+00
   7.08438206e+00  7.26083422e+00  7.89637041e+00  7.83428335e+00
   8.05146027e+00  7.34668827e+00  9.01473808e+00  7.21382952e+00
   8.71019650e+00  8.19252682e+00  6.59498882e+00  7.60528326e+00
   5.50736856e+00  6.75447845e+00  7.97216034e+00  7.28828955e+00
   7.74917603e+00  7.99899864e+00  6.33963919e+00  7.16124344e+00
   7.84190416e+00  5.83876657e+00  7.56703186e+00  7.05590773e+00
   7.21574450e+00  7.91071987e+00  7.62324238e+00  8.32148743e+00
   7.65332413e+00  7.47185183e+00  8.04419708e+00  6.55430984e+00
   7.25340223e+00  6.50463390e+00  6.82784700e+00  7.14154196e+00
   6.88833714e+00  6.43527746e+00  6.58507538e+00  6.74297953e+00
   7.92425394e+00  7.39762497e+00  7.38855314e+00  6.97887039e+00
   8.13994980e+00  7.53864574e+00  8.47870827e+00  7.17568207e+00
   8.27776718e+00  6.49896908e+00  7.99853849e+00  8.03791142e+00
   5.72487652e-01  4.59745765e-01  2.12113357e+00  1.86287308e+00
   2.56235838e+00  1.62547040e+00  1.96250379e-01  8.09683084e-01
   4.40233767e-01  9.49084342e-01  1.01504886e+00  1.63727999e+00
   1.62656486e+00  2.21213484e+00  5.38005888e-01  1.13566685e+00
   1.97654378e+00  4.81599391e-01  8.66363108e-01  3.66299212e-01
   8.24728847e-01  2.76800847e+00  1.05444705e+00  8.31298172e-01
   1.69004405e+00  7.23809600e-01  1.14940965e+00  1.31271601e+00
   1.04966664e+00  5.16673803e-01  3.01679087e+00  2.31777406e+00
   1.65777385e+00  1.36188459e+00  1.30017877e+00  5.28831065e-01
   2.19684315e+00  3.42970490e-01  1.23284602e+00  1.25348473e+00
   8.43774915e-01  2.20396042e+00  2.53776860e+00  6.86795473e-01
   1.03391945e+00  1.55920744e-01  1.20707798e+00  3.57455373e-01
   6.14704430e-01  7.20451832e-01  3.36854815e-01  1.33046150e-01
   8.94689679e-01  1.91482592e+00  2.69344664e+00  2.20528984e+00
   1.57490683e+00  5.90033770e-01  2.02996111e+00  1.80511761e+00
   6.78151250e-01  2.34536791e+00  1.49360871e+00  8.68382454e-01
   2.71003914e+00  7.22354531e-01  4.35158134e-01  2.73232818e-01
   1.11463070e-01  3.62777591e-01  2.02753735e+00  1.82398653e+00
   1.70804274e+00  4.55496311e-01  1.19503474e+00  7.01920390e-01
   1.05815220e+00  2.99949312e+00  6.01786733e-01  2.19073832e-01
   2.01793909e-01  4.32076693e-01  1.12815285e+00  1.08715999e+00
   1.87765777e-01  4.56873834e-01  1.25494611e+00  9.97998178e-01
   1.53751206e+00  9.25404727e-01  3.80032301e-01  2.41764355e+00
   9.60635662e-01  1.36648524e+00  1.74244463e+00  1.14554346e+00
   1.91797137e+00  2.24015808e+00  7.34442949e-01  1.73772573e+00
   1.86999881e+00  1.10431850e-01  1.01870096e+00  1.11722374e+00
   1.45572877e+00  1.47584808e+00  1.59211707e+00  1.73118591e+00
   1.20677960e+00  7.01036811e-01  7.57946968e-01  9.50671017e-01
   2.51848745e+00  5.65362811e-01  6.13295555e-01  1.16513944e+00
   5.88365138e-01  2.19789076e+00  1.22344375e-01  5.60499907e-01
   6.00401592e+00 -7.32963085e+00 -7.80993557e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-19 08:14:34.619709
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.4481
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-19 08:14:34.623847
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9128.18
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-19 08:14:34.627867
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.1651
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-19 08:14:34.631153
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -816.493
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140353740912064
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140352648143368
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140352648143872
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140352648144376
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140352648144880
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140352648145384

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fa6a6dcfe10> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.494447
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.461861
grad_step = 000002, loss = 0.435492
grad_step = 000003, loss = 0.408104
grad_step = 000004, loss = 0.381600
grad_step = 000005, loss = 0.361561
grad_step = 000006, loss = 0.348978
grad_step = 000007, loss = 0.332440
grad_step = 000008, loss = 0.319544
grad_step = 000009, loss = 0.306794
grad_step = 000010, loss = 0.293070
grad_step = 000011, loss = 0.281539
grad_step = 000012, loss = 0.271933
grad_step = 000013, loss = 0.261983
grad_step = 000014, loss = 0.250993
grad_step = 000015, loss = 0.240063
grad_step = 000016, loss = 0.229796
grad_step = 000017, loss = 0.219388
grad_step = 000018, loss = 0.209634
grad_step = 000019, loss = 0.200949
grad_step = 000020, loss = 0.192236
grad_step = 000021, loss = 0.183087
grad_step = 000022, loss = 0.173835
grad_step = 000023, loss = 0.164898
grad_step = 000024, loss = 0.156606
grad_step = 000025, loss = 0.148957
grad_step = 000026, loss = 0.141511
grad_step = 000027, loss = 0.133850
grad_step = 000028, loss = 0.126055
grad_step = 000029, loss = 0.118771
grad_step = 000030, loss = 0.112375
grad_step = 000031, loss = 0.106279
grad_step = 000032, loss = 0.099960
grad_step = 000033, loss = 0.093562
grad_step = 000034, loss = 0.087477
grad_step = 000035, loss = 0.082072
grad_step = 000036, loss = 0.077049
grad_step = 000037, loss = 0.071998
grad_step = 000038, loss = 0.067003
grad_step = 000039, loss = 0.062272
grad_step = 000040, loss = 0.057915
grad_step = 000041, loss = 0.053781
grad_step = 000042, loss = 0.049854
grad_step = 000043, loss = 0.046125
grad_step = 000044, loss = 0.042608
grad_step = 000045, loss = 0.039258
grad_step = 000046, loss = 0.036110
grad_step = 000047, loss = 0.033227
grad_step = 000048, loss = 0.030548
grad_step = 000049, loss = 0.027995
grad_step = 000050, loss = 0.025600
grad_step = 000051, loss = 0.023447
grad_step = 000052, loss = 0.021487
grad_step = 000053, loss = 0.019620
grad_step = 000054, loss = 0.017878
grad_step = 000055, loss = 0.016322
grad_step = 000056, loss = 0.014930
grad_step = 000057, loss = 0.013626
grad_step = 000058, loss = 0.012430
grad_step = 000059, loss = 0.011346
grad_step = 000060, loss = 0.010359
grad_step = 000061, loss = 0.009464
grad_step = 000062, loss = 0.008666
grad_step = 000063, loss = 0.007937
grad_step = 000064, loss = 0.007255
grad_step = 000065, loss = 0.006650
grad_step = 000066, loss = 0.006119
grad_step = 000067, loss = 0.005628
grad_step = 000068, loss = 0.005170
grad_step = 000069, loss = 0.004771
grad_step = 000070, loss = 0.004416
grad_step = 000071, loss = 0.004092
grad_step = 000072, loss = 0.003801
grad_step = 000073, loss = 0.003543
grad_step = 000074, loss = 0.003312
grad_step = 000075, loss = 0.003110
grad_step = 000076, loss = 0.002932
grad_step = 000077, loss = 0.002770
grad_step = 000078, loss = 0.002631
grad_step = 000079, loss = 0.002516
grad_step = 000080, loss = 0.002414
grad_step = 000081, loss = 0.002322
grad_step = 000082, loss = 0.002246
grad_step = 000083, loss = 0.002183
grad_step = 000084, loss = 0.002135
grad_step = 000085, loss = 0.002104
grad_step = 000086, loss = 0.002123
grad_step = 000087, loss = 0.002055
grad_step = 000088, loss = 0.002013
grad_step = 000089, loss = 0.001971
grad_step = 000090, loss = 0.001957
grad_step = 000091, loss = 0.001968
grad_step = 000092, loss = 0.001958
grad_step = 000093, loss = 0.001943
grad_step = 000094, loss = 0.001918
grad_step = 000095, loss = 0.001891
grad_step = 000096, loss = 0.001873
grad_step = 000097, loss = 0.001873
grad_step = 000098, loss = 0.001883
grad_step = 000099, loss = 0.001895
grad_step = 000100, loss = 0.001905
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.001877
grad_step = 000102, loss = 0.001849
grad_step = 000103, loss = 0.001822
grad_step = 000104, loss = 0.001809
grad_step = 000105, loss = 0.001813
grad_step = 000106, loss = 0.001828
grad_step = 000107, loss = 0.001849
grad_step = 000108, loss = 0.001851
grad_step = 000109, loss = 0.001850
grad_step = 000110, loss = 0.001807
grad_step = 000111, loss = 0.001773
grad_step = 000112, loss = 0.001758
grad_step = 000113, loss = 0.001768
grad_step = 000114, loss = 0.001793
grad_step = 000115, loss = 0.001792
grad_step = 000116, loss = 0.001785
grad_step = 000117, loss = 0.001751
grad_step = 000118, loss = 0.001728
grad_step = 000119, loss = 0.001723
grad_step = 000120, loss = 0.001731
grad_step = 000121, loss = 0.001745
grad_step = 000122, loss = 0.001743
grad_step = 000123, loss = 0.001738
grad_step = 000124, loss = 0.001718
grad_step = 000125, loss = 0.001702
grad_step = 000126, loss = 0.001692
grad_step = 000127, loss = 0.001690
grad_step = 000128, loss = 0.001694
grad_step = 000129, loss = 0.001703
grad_step = 000130, loss = 0.001719
grad_step = 000131, loss = 0.001728
grad_step = 000132, loss = 0.001743
grad_step = 000133, loss = 0.001732
grad_step = 000134, loss = 0.001715
grad_step = 000135, loss = 0.001688
grad_step = 000136, loss = 0.001676
grad_step = 000137, loss = 0.001677
grad_step = 000138, loss = 0.001682
grad_step = 000139, loss = 0.001685
grad_step = 000140, loss = 0.001673
grad_step = 000141, loss = 0.001663
grad_step = 000142, loss = 0.001653
grad_step = 000143, loss = 0.001650
grad_step = 000144, loss = 0.001652
grad_step = 000145, loss = 0.001657
grad_step = 000146, loss = 0.001659
grad_step = 000147, loss = 0.001659
grad_step = 000148, loss = 0.001650
grad_step = 000149, loss = 0.001639
grad_step = 000150, loss = 0.001628
grad_step = 000151, loss = 0.001620
grad_step = 000152, loss = 0.001621
grad_step = 000153, loss = 0.001630
grad_step = 000154, loss = 0.001656
grad_step = 000155, loss = 0.001704
grad_step = 000156, loss = 0.001803
grad_step = 000157, loss = 0.001884
grad_step = 000158, loss = 0.001937
grad_step = 000159, loss = 0.001767
grad_step = 000160, loss = 0.001639
grad_step = 000161, loss = 0.001668
grad_step = 000162, loss = 0.001711
grad_step = 000163, loss = 0.001675
grad_step = 000164, loss = 0.001622
grad_step = 000165, loss = 0.001671
grad_step = 000166, loss = 0.001744
grad_step = 000167, loss = 0.001679
grad_step = 000168, loss = 0.001600
grad_step = 000169, loss = 0.001584
grad_step = 000170, loss = 0.001636
grad_step = 000171, loss = 0.001678
grad_step = 000172, loss = 0.001626
grad_step = 000173, loss = 0.001572
grad_step = 000174, loss = 0.001569
grad_step = 000175, loss = 0.001604
grad_step = 000176, loss = 0.001619
grad_step = 000177, loss = 0.001586
grad_step = 000178, loss = 0.001556
grad_step = 000179, loss = 0.001561
grad_step = 000180, loss = 0.001578
grad_step = 000181, loss = 0.001582
grad_step = 000182, loss = 0.001561
grad_step = 000183, loss = 0.001546
grad_step = 000184, loss = 0.001547
grad_step = 000185, loss = 0.001556
grad_step = 000186, loss = 0.001559
grad_step = 000187, loss = 0.001548
grad_step = 000188, loss = 0.001534
grad_step = 000189, loss = 0.001526
grad_step = 000190, loss = 0.001527
grad_step = 000191, loss = 0.001533
grad_step = 000192, loss = 0.001535
grad_step = 000193, loss = 0.001534
grad_step = 000194, loss = 0.001528
grad_step = 000195, loss = 0.001523
grad_step = 000196, loss = 0.001520
grad_step = 000197, loss = 0.001526
grad_step = 000198, loss = 0.001543
grad_step = 000199, loss = 0.001581
grad_step = 000200, loss = 0.001597
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001614
grad_step = 000202, loss = 0.001564
grad_step = 000203, loss = 0.001506
grad_step = 000204, loss = 0.001486
grad_step = 000205, loss = 0.001517
grad_step = 000206, loss = 0.001567
grad_step = 000207, loss = 0.001572
grad_step = 000208, loss = 0.001533
grad_step = 000209, loss = 0.001481
grad_step = 000210, loss = 0.001481
grad_step = 000211, loss = 0.001517
grad_step = 000212, loss = 0.001539
grad_step = 000213, loss = 0.001531
grad_step = 000214, loss = 0.001506
grad_step = 000215, loss = 0.001514
grad_step = 000216, loss = 0.001572
grad_step = 000217, loss = 0.001636
grad_step = 000218, loss = 0.001680
grad_step = 000219, loss = 0.001612
grad_step = 000220, loss = 0.001521
grad_step = 000221, loss = 0.001450
grad_step = 000222, loss = 0.001471
grad_step = 000223, loss = 0.001540
grad_step = 000224, loss = 0.001542
grad_step = 000225, loss = 0.001501
grad_step = 000226, loss = 0.001434
grad_step = 000227, loss = 0.001419
grad_step = 000228, loss = 0.001455
grad_step = 000229, loss = 0.001490
grad_step = 000230, loss = 0.001509
grad_step = 000231, loss = 0.001468
grad_step = 000232, loss = 0.001424
grad_step = 000233, loss = 0.001396
grad_step = 000234, loss = 0.001401
grad_step = 000235, loss = 0.001423
grad_step = 000236, loss = 0.001430
grad_step = 000237, loss = 0.001419
grad_step = 000238, loss = 0.001392
grad_step = 000239, loss = 0.001377
grad_step = 000240, loss = 0.001386
grad_step = 000241, loss = 0.001436
grad_step = 000242, loss = 0.001594
grad_step = 000243, loss = 0.001873
grad_step = 000244, loss = 0.001536
grad_step = 000245, loss = 0.001643
grad_step = 000246, loss = 0.001472
grad_step = 000247, loss = 0.001507
grad_step = 000248, loss = 0.001494
grad_step = 000249, loss = 0.001503
grad_step = 000250, loss = 0.001496
grad_step = 000251, loss = 0.001468
grad_step = 000252, loss = 0.001441
grad_step = 000253, loss = 0.001429
grad_step = 000254, loss = 0.001457
grad_step = 000255, loss = 0.001442
grad_step = 000256, loss = 0.001427
grad_step = 000257, loss = 0.001386
grad_step = 000258, loss = 0.001405
grad_step = 000259, loss = 0.001386
grad_step = 000260, loss = 0.001418
grad_step = 000261, loss = 0.001371
grad_step = 000262, loss = 0.001398
grad_step = 000263, loss = 0.001354
grad_step = 000264, loss = 0.001385
grad_step = 000265, loss = 0.001356
grad_step = 000266, loss = 0.001379
grad_step = 000267, loss = 0.001359
grad_step = 000268, loss = 0.001362
grad_step = 000269, loss = 0.001352
grad_step = 000270, loss = 0.001344
grad_step = 000271, loss = 0.001350
grad_step = 000272, loss = 0.001339
grad_step = 000273, loss = 0.001351
grad_step = 000274, loss = 0.001338
grad_step = 000275, loss = 0.001348
grad_step = 000276, loss = 0.001339
grad_step = 000277, loss = 0.001341
grad_step = 000278, loss = 0.001337
grad_step = 000279, loss = 0.001331
grad_step = 000280, loss = 0.001334
grad_step = 000281, loss = 0.001325
grad_step = 000282, loss = 0.001327
grad_step = 000283, loss = 0.001321
grad_step = 000284, loss = 0.001321
grad_step = 000285, loss = 0.001319
grad_step = 000286, loss = 0.001315
grad_step = 000287, loss = 0.001316
grad_step = 000288, loss = 0.001312
grad_step = 000289, loss = 0.001313
grad_step = 000290, loss = 0.001311
grad_step = 000291, loss = 0.001310
grad_step = 000292, loss = 0.001310
grad_step = 000293, loss = 0.001309
grad_step = 000294, loss = 0.001313
grad_step = 000295, loss = 0.001316
grad_step = 000296, loss = 0.001328
grad_step = 000297, loss = 0.001348
grad_step = 000298, loss = 0.001392
grad_step = 000299, loss = 0.001446
grad_step = 000300, loss = 0.001549
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001561
grad_step = 000302, loss = 0.001538
grad_step = 000303, loss = 0.001397
grad_step = 000304, loss = 0.001306
grad_step = 000305, loss = 0.001339
grad_step = 000306, loss = 0.001418
grad_step = 000307, loss = 0.001436
grad_step = 000308, loss = 0.001357
grad_step = 000309, loss = 0.001298
grad_step = 000310, loss = 0.001297
grad_step = 000311, loss = 0.001336
grad_step = 000312, loss = 0.001389
grad_step = 000313, loss = 0.001398
grad_step = 000314, loss = 0.001384
grad_step = 000315, loss = 0.001339
grad_step = 000316, loss = 0.001304
grad_step = 000317, loss = 0.001297
grad_step = 000318, loss = 0.001310
grad_step = 000319, loss = 0.001321
grad_step = 000320, loss = 0.001317
grad_step = 000321, loss = 0.001308
grad_step = 000322, loss = 0.001299
grad_step = 000323, loss = 0.001295
grad_step = 000324, loss = 0.001294
grad_step = 000325, loss = 0.001289
grad_step = 000326, loss = 0.001283
grad_step = 000327, loss = 0.001280
grad_step = 000328, loss = 0.001281
grad_step = 000329, loss = 0.001285
grad_step = 000330, loss = 0.001285
grad_step = 000331, loss = 0.001280
grad_step = 000332, loss = 0.001271
grad_step = 000333, loss = 0.001264
grad_step = 000334, loss = 0.001261
grad_step = 000335, loss = 0.001264
grad_step = 000336, loss = 0.001268
grad_step = 000337, loss = 0.001269
grad_step = 000338, loss = 0.001267
grad_step = 000339, loss = 0.001261
grad_step = 000340, loss = 0.001256
grad_step = 000341, loss = 0.001252
grad_step = 000342, loss = 0.001251
grad_step = 000343, loss = 0.001252
grad_step = 000344, loss = 0.001253
grad_step = 000345, loss = 0.001253
grad_step = 000346, loss = 0.001252
grad_step = 000347, loss = 0.001249
grad_step = 000348, loss = 0.001246
grad_step = 000349, loss = 0.001244
grad_step = 000350, loss = 0.001243
grad_step = 000351, loss = 0.001242
grad_step = 000352, loss = 0.001242
grad_step = 000353, loss = 0.001242
grad_step = 000354, loss = 0.001243
grad_step = 000355, loss = 0.001243
grad_step = 000356, loss = 0.001245
grad_step = 000357, loss = 0.001246
grad_step = 000358, loss = 0.001249
grad_step = 000359, loss = 0.001254
grad_step = 000360, loss = 0.001263
grad_step = 000361, loss = 0.001275
grad_step = 000362, loss = 0.001296
grad_step = 000363, loss = 0.001319
grad_step = 000364, loss = 0.001352
grad_step = 000365, loss = 0.001373
grad_step = 000366, loss = 0.001391
grad_step = 000367, loss = 0.001383
grad_step = 000368, loss = 0.001348
grad_step = 000369, loss = 0.001300
grad_step = 000370, loss = 0.001251
grad_step = 000371, loss = 0.001228
grad_step = 000372, loss = 0.001235
grad_step = 000373, loss = 0.001262
grad_step = 000374, loss = 0.001291
grad_step = 000375, loss = 0.001319
grad_step = 000376, loss = 0.001318
grad_step = 000377, loss = 0.001305
grad_step = 000378, loss = 0.001270
grad_step = 000379, loss = 0.001238
grad_step = 000380, loss = 0.001219
grad_step = 000381, loss = 0.001219
grad_step = 000382, loss = 0.001229
grad_step = 000383, loss = 0.001241
grad_step = 000384, loss = 0.001243
grad_step = 000385, loss = 0.001236
grad_step = 000386, loss = 0.001225
grad_step = 000387, loss = 0.001215
grad_step = 000388, loss = 0.001212
grad_step = 000389, loss = 0.001212
grad_step = 000390, loss = 0.001214
grad_step = 000391, loss = 0.001213
grad_step = 000392, loss = 0.001209
grad_step = 000393, loss = 0.001205
grad_step = 000394, loss = 0.001202
grad_step = 000395, loss = 0.001202
grad_step = 000396, loss = 0.001204
grad_step = 000397, loss = 0.001208
grad_step = 000398, loss = 0.001212
grad_step = 000399, loss = 0.001216
grad_step = 000400, loss = 0.001216
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001215
grad_step = 000402, loss = 0.001211
grad_step = 000403, loss = 0.001206
grad_step = 000404, loss = 0.001201
grad_step = 000405, loss = 0.001196
grad_step = 000406, loss = 0.001191
grad_step = 000407, loss = 0.001188
grad_step = 000408, loss = 0.001185
grad_step = 000409, loss = 0.001182
grad_step = 000410, loss = 0.001181
grad_step = 000411, loss = 0.001179
grad_step = 000412, loss = 0.001178
grad_step = 000413, loss = 0.001177
grad_step = 000414, loss = 0.001176
grad_step = 000415, loss = 0.001175
grad_step = 000416, loss = 0.001174
grad_step = 000417, loss = 0.001174
grad_step = 000418, loss = 0.001173
grad_step = 000419, loss = 0.001173
grad_step = 000420, loss = 0.001173
grad_step = 000421, loss = 0.001175
grad_step = 000422, loss = 0.001179
grad_step = 000423, loss = 0.001188
grad_step = 000424, loss = 0.001202
grad_step = 000425, loss = 0.001231
grad_step = 000426, loss = 0.001270
grad_step = 000427, loss = 0.001344
grad_step = 000428, loss = 0.001403
grad_step = 000429, loss = 0.001463
grad_step = 000430, loss = 0.001425
grad_step = 000431, loss = 0.001325
grad_step = 000432, loss = 0.001205
grad_step = 000433, loss = 0.001161
grad_step = 000434, loss = 0.001205
grad_step = 000435, loss = 0.001271
grad_step = 000436, loss = 0.001303
grad_step = 000437, loss = 0.001252
grad_step = 000438, loss = 0.001201
grad_step = 000439, loss = 0.001184
grad_step = 000440, loss = 0.001211
grad_step = 000441, loss = 0.001249
grad_step = 000442, loss = 0.001246
grad_step = 000443, loss = 0.001212
grad_step = 000444, loss = 0.001170
grad_step = 000445, loss = 0.001155
grad_step = 000446, loss = 0.001168
grad_step = 000447, loss = 0.001179
grad_step = 000448, loss = 0.001174
grad_step = 000449, loss = 0.001151
grad_step = 000450, loss = 0.001136
grad_step = 000451, loss = 0.001137
grad_step = 000452, loss = 0.001149
grad_step = 000453, loss = 0.001160
grad_step = 000454, loss = 0.001158
grad_step = 000455, loss = 0.001151
grad_step = 000456, loss = 0.001144
grad_step = 000457, loss = 0.001145
grad_step = 000458, loss = 0.001153
grad_step = 000459, loss = 0.001162
grad_step = 000460, loss = 0.001166
grad_step = 000461, loss = 0.001164
grad_step = 000462, loss = 0.001161
grad_step = 000463, loss = 0.001159
grad_step = 000464, loss = 0.001159
grad_step = 000465, loss = 0.001158
grad_step = 000466, loss = 0.001155
grad_step = 000467, loss = 0.001147
grad_step = 000468, loss = 0.001136
grad_step = 000469, loss = 0.001124
grad_step = 000470, loss = 0.001116
grad_step = 000471, loss = 0.001113
grad_step = 000472, loss = 0.001113
grad_step = 000473, loss = 0.001113
grad_step = 000474, loss = 0.001114
grad_step = 000475, loss = 0.001114
grad_step = 000476, loss = 0.001114
grad_step = 000477, loss = 0.001115
grad_step = 000478, loss = 0.001116
grad_step = 000479, loss = 0.001118
grad_step = 000480, loss = 0.001119
grad_step = 000481, loss = 0.001121
grad_step = 000482, loss = 0.001121
grad_step = 000483, loss = 0.001121
grad_step = 000484, loss = 0.001120
grad_step = 000485, loss = 0.001119
grad_step = 000486, loss = 0.001117
grad_step = 000487, loss = 0.001115
grad_step = 000488, loss = 0.001112
grad_step = 000489, loss = 0.001110
grad_step = 000490, loss = 0.001107
grad_step = 000491, loss = 0.001103
grad_step = 000492, loss = 0.001100
grad_step = 000493, loss = 0.001096
grad_step = 000494, loss = 0.001093
grad_step = 000495, loss = 0.001090
grad_step = 000496, loss = 0.001087
grad_step = 000497, loss = 0.001085
grad_step = 000498, loss = 0.001083
grad_step = 000499, loss = 0.001081
grad_step = 000500, loss = 0.001080
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001079
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

  date_run                              2020-05-19 08:14:59.379466
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.29628
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-19 08:14:59.387068
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.276673
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-19 08:14:59.396131
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.147455
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-19 08:14:59.402564
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -3.20414
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
100%|██████████| 10/10 [00:03<00:00,  3.17it/s, avg_epoch_loss=5.23]
INFO:root:Epoch[0] Elapsed time 3.154 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.225453
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.225452899932861 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fa6c720e320> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  6.32it/s, avg_epoch_loss=3.59e+3]
INFO:root:Epoch[0] Elapsed time 1.584 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=3590.403646
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 3590.4036458333335 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fa60e437588> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
 30%|███       | 3/10 [00:13<00:32,  4.60s/it, avg_epoch_loss=6.94] 60%|██████    | 6/10 [00:25<00:17,  4.41s/it, avg_epoch_loss=6.91] 90%|█████████ | 9/10 [00:38<00:04,  4.32s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:42<00:00,  4.20s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 42.035 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.868105
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.868105125427246 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fa5f272f1d0> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:02<00:00,  4.49it/s, avg_epoch_loss=5.78]
INFO:root:Epoch[0] Elapsed time 2.227 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.781131
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.7811308860778805 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fa5f15ece10> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
 10%|█         | 1/10 [02:11<19:44, 131.60s/it, avg_epoch_loss=0.412] 20%|██        | 2/10 [05:18<19:46, 148.27s/it, avg_epoch_loss=0.399] 30%|███       | 3/10 [08:42<19:15, 165.03s/it, avg_epoch_loss=0.39]  40%|████      | 4/10 [12:08<17:43, 177.21s/it, avg_epoch_loss=0.385] 50%|█████     | 5/10 [15:23<15:12, 182.52s/it, avg_epoch_loss=0.384] 60%|██████    | 6/10 [18:34<12:20, 185.18s/it, avg_epoch_loss=0.383] 70%|███████   | 7/10 [22:12<09:44, 194.91s/it, avg_epoch_loss=0.381] 80%|████████  | 8/10 [25:55<06:46, 203.26s/it, avg_epoch_loss=0.378] 90%|█████████ | 9/10 [29:19<03:23, 203.58s/it, avg_epoch_loss=0.376]100%|██████████| 10/10 [33:03<00:00, 209.82s/it, avg_epoch_loss=0.375]100%|██████████| 10/10 [33:03<00:00, 198.39s/it, avg_epoch_loss=0.375]
INFO:root:Epoch[0] Elapsed time 1983.924 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.374571
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3745713621377945 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fa5f1627ac8> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  5.09it/s, avg_epoch_loss=415]
INFO:root:Epoch[0] Elapsed time 1.985 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=414.652022
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 414.65202175008733 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fa664272a58> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:00<00:00, 39.63it/s, avg_epoch_loss=5.18]
INFO:root:Epoch[0] Elapsed time 0.254 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.176167
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.176167440414429 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7fa664208c18> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
0   2020-05-19 08:14:24.468998  ...    mean_absolute_error
1   2020-05-19 08:14:24.472751  ...     mean_squared_error
2   2020-05-19 08:14:24.476211  ...  median_absolute_error
3   2020-05-19 08:14:24.479772  ...               r2_score
4   2020-05-19 08:14:34.619709  ...    mean_absolute_error
5   2020-05-19 08:14:34.623847  ...     mean_squared_error
6   2020-05-19 08:14:34.627867  ...  median_absolute_error
7   2020-05-19 08:14:34.631153  ...               r2_score
8   2020-05-19 08:14:59.379466  ...    mean_absolute_error
9   2020-05-19 08:14:59.387068  ...     mean_squared_error
10  2020-05-19 08:14:59.396131  ...  median_absolute_error
11  2020-05-19 08:14:59.402564  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe070296ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe01faad8d0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe022c51d30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe01faad8d0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe070296ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe01faad8d0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe022c51d30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe01faad8d0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe070296ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe01faad8d0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fe022c51d30> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f1bba2fb080> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=9b41ac7a41f6084b63ba77a6b6809ebb39ca679a7c6ae61850cb1efd29f370a3
  Stored in directory: /tmp/pip-ephem-wheel-cache-dnciord3/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1b5a026c88> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3563520/17464789 [=====>........................] - ETA: 0s
11714560/17464789 [===================>..........] - ETA: 0s
16506880/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-19 08:50:42.283345: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-19 08:50:42.297826: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-19 08:50:42.298071: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558413a927c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-19 08:50:42.298092: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 15s - loss: 7.0993 - accuracy: 0.5370
 2000/25000 [=>............................] - ETA: 11s - loss: 7.1990 - accuracy: 0.5305
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.3600 - accuracy: 0.5200 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.4635 - accuracy: 0.5132
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5072 - accuracy: 0.5104
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.5414 - accuracy: 0.5082
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5659 - accuracy: 0.5066
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6130 - accuracy: 0.5035
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6223 - accuracy: 0.5029
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6222 - accuracy: 0.5029
11000/25000 [============>.................] - ETA: 4s - loss: 7.6485 - accuracy: 0.5012
12000/25000 [=============>................] - ETA: 4s - loss: 7.6794 - accuracy: 0.4992
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6761 - accuracy: 0.4994
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7039 - accuracy: 0.4976
15000/25000 [=================>............] - ETA: 3s - loss: 7.7106 - accuracy: 0.4971
16000/25000 [==================>...........] - ETA: 3s - loss: 7.7011 - accuracy: 0.4978
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6874 - accuracy: 0.4986
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6879 - accuracy: 0.4986
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6908 - accuracy: 0.4984
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6827 - accuracy: 0.4990
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6827 - accuracy: 0.4990
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6778 - accuracy: 0.4993
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6800 - accuracy: 0.4991
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
25000/25000 [==============================] - 10s 401us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-19 08:51:00.146894
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-19 08:51:00.146894  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<42:08:53, 5.68kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<29:44:03, 8.05kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<20:51:55, 11.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<14:36:30, 16.4kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<10:11:48, 23.4kB/s].vector_cache/glove.6B.zip:   1%|          | 8.91M/862M [00:02<7:05:43, 33.4kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:02<4:56:43, 47.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.2M/862M [00:02<3:26:44, 68.1kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.8M/862M [00:02<2:24:03, 97.2kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.8M/862M [00:02<1:40:19, 139kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 30.4M/862M [00:02<1:10:02, 198kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.2M/862M [00:02<48:55, 282kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 36.2M/862M [00:02<34:30, 399kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.5M/862M [00:02<24:07, 567kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.1M/862M [00:03<16:56, 805kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.8M/862M [00:03<11:54, 1.14MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:03<08:45, 1.54MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<08:01, 1.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<07:33, 1.78MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:05<05:41, 2.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<04:07, 3.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:07<2:42:05, 82.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<1:56:09, 115kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:07<1:21:48, 163kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<57:15, 232kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:09<48:14, 276kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<35:07, 378kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:09<24:53, 533kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:11<20:27, 647kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:11<16:59, 778kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<12:33, 1.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:13<10:53, 1.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:13<08:57, 1.47MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:13<06:32, 2.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:15<07:39, 1.71MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:15<06:40, 1.96MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:15<04:56, 2.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:17<06:33, 1.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:17<07:15, 1.79MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:17<05:38, 2.30MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:17<04:05, 3.17MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:19<11:06, 1.17MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:19<09:05, 1.42MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:19<06:40, 1.94MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:21<07:45, 1.66MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:21<08:02, 1.60MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:21<06:11, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:21<04:27, 2.87MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:23<15:19, 836kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:23<12:01, 1.06MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:23<08:41, 1.47MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:25<09:03, 1.41MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:25<07:38, 1.67MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:25<05:39, 2.25MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<06:52, 1.84MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<07:24, 1.71MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<05:48, 2.18MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<06:06, 2.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<05:35, 2.25MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<04:11, 2.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<05:49, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<06:36, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<05:15, 2.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<05:41, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<05:15, 2.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<03:57, 3.15MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<05:40, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:35<06:31, 1.90MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<05:06, 2.43MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:36<07:09, 1.73MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<06:13, 1.98MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<05:38, 2.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<04:11, 2.93MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:49, 2.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<06:37, 1.85MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<05:09, 2.37MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:39<03:47, 3.22MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<07:20, 1.66MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<06:24, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<04:44, 2.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<06:07, 1.98MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<05:31, 2.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<04:10, 2.90MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<05:46, 2.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<06:32, 1.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<05:06, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<03:42, 3.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<11:31, 1.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<09:20, 1.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<06:48, 1.76MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:31, 1.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<06:28, 1.84MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<04:50, 2.46MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:10, 1.92MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<06:44, 1.76MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<05:19, 2.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:39, 2.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<06:22, 1.85MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<05:03, 2.32MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<03:39, 3.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<20:03, 584kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<15:13, 769kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<10:56, 1.07MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<10:21, 1.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:25, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<06:08, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<07:02, 1.64MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:07, 1.89MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<04:31, 2.55MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:54, 1.95MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:28, 1.78MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<05:01, 2.28MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<03:38, 3.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<10:24, 1.10MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<08:28, 1.35MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<06:12, 1.84MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<07:00, 1.62MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<07:13, 1.57MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:32, 2.05MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<04:00, 2.82MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<09:20, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:41, 1.47MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<05:39, 1.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<06:35, 1.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<06:53, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:24, 2.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<03:54, 2.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<1:22:26, 135kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<58:48, 190kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<41:21, 269kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<31:27, 353kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<23:07, 479kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<16:23, 675kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<14:04, 784kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<12:06, 911kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<09:01, 1.22MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<08:03, 1.36MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<06:44, 1.62MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:57, 2.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<06:01, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<06:25, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<04:58, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<03:36, 3.01MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<08:51, 1.22MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<07:18, 1.48MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<05:22, 2.01MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<06:16, 1.72MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<06:34, 1.63MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<05:04, 2.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<03:40, 2.92MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<10:58, 973kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<08:47, 1.22MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:24, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<06:57, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<07:01, 1.51MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<05:26, 1.94MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<05:30, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<04:55, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<03:42, 2.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<05:02, 2.08MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<04:34, 2.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<03:27, 3.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<04:52, 2.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<05:32, 1.88MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<04:24, 2.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<04:44, 2.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<04:23, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<03:20, 3.09MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<04:42, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<05:22, 1.91MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<04:17, 2.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:07, 3.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<1:15:08, 136kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<53:37, 190kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<37:40, 270kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<28:37, 354kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:40<21:03, 481kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<14:57, 675kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:42<12:39, 794kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<10:44, 936kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<07:58, 1.26MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<07:13, 1.38MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<06:11, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<04:33, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<05:19, 1.86MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<05:44, 1.73MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<04:31, 2.19MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<03:15, 3.02MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<9:29:29, 17.3kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<6:39:24, 24.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<4:39:03, 35.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<3:14:34, 50.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<2:29:26, 65.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<1:48:02, 90.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<1:16:23, 128kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<53:30, 182kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<40:09, 242kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:52<29:20, 330kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<20:49, 465kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<16:23, 588kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<12:43, 756kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<09:09, 1.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<08:15, 1.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:56<08:06, 1.18MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<06:10, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:26, 2.14MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<06:41, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<05:39, 1.68MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:11, 2.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<05:06, 1.85MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<04:31, 2.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<03:21, 2.80MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<04:34, 2.04MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<05:05, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<03:58, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<02:52, 3.23MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<13:26, 691kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<10:22, 895kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<07:28, 1.24MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<07:22, 1.25MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<05:57, 1.55MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:23, 2.09MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<05:14, 1.75MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<05:31, 1.66MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<04:16, 2.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:06, 2.93MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<06:33, 1.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<05:31, 1.64MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:05, 2.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<04:57, 1.82MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<04:23, 2.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<03:17, 2.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<04:24, 2.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<03:59, 2.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:00, 2.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<04:12, 2.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<04:44, 1.87MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<03:42, 2.38MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<02:40, 3.28MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<13:40, 644kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:18<10:27, 841kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<07:31, 1.17MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<07:17, 1.20MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<05:59, 1.46MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:23, 1.98MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<05:07, 1.69MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<05:20, 1.62MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<04:10, 2.07MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<04:18, 2.00MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<03:53, 2.21MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<02:56, 2.91MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<04:02, 2.11MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<04:34, 1.86MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<03:38, 2.34MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<02:38, 3.20MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<1:02:20, 136kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<44:27, 190kB/s]  .vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<31:13, 270kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<23:43, 354kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<18:20, 458kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<13:14, 633kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<09:18, 894kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<8:05:45, 17.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<5:40:31, 24.4kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<3:57:51, 34.9kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<2:47:39, 49.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<1:58:59, 69.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<1:23:32, 98.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<58:16, 141kB/s]   .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<47:33, 172kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<34:05, 240kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<24:00, 340kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<18:38, 436kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<13:51, 585kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<09:52, 819kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<08:46, 917kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<07:47, 1.03MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<05:51, 1.37MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<05:22, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:41<04:34, 1.74MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:23, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<04:13, 1.87MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<04:33, 1.73MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<03:32, 2.23MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:33, 3.07MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<07:18, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<05:47, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<04:12, 1.86MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:02, 2.56MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<33:24, 233kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<24:58, 311kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<17:47, 436kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<12:31, 617kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<11:55, 646kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<09:08, 843kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<06:34, 1.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<06:21, 1.20MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<06:00, 1.27MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:32, 1.68MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:16, 2.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<05:25, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<04:34, 1.65MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:21, 2.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<04:06, 1.83MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<03:38, 2.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<02:42, 2.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<03:39, 2.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<03:18, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<02:29, 2.97MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<03:28, 2.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<03:10, 2.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<02:24, 3.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<03:24, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<03:51, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<03:04, 2.37MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<03:18, 2.18MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<03:02, 2.37MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<02:18, 3.12MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<03:17, 2.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<03:02, 2.36MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<02:16, 3.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<03:15, 2.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:07<03:00, 2.36MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<02:16, 3.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:14, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:09<03:41, 1.90MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:09<02:56, 2.38MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:10, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<02:55, 2.38MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<02:12, 3.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<03:09, 2.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:13<03:39, 1.88MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<02:54, 2.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:06, 3.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<26:06, 261kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:15<18:58, 359kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<13:24, 506kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<10:55, 618kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<08:20, 809kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<05:58, 1.12MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<05:44, 1.16MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:41, 1.42MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<03:26, 1.93MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<03:56, 1.67MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<04:06, 1.61MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<03:09, 2.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:17, 2.86MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<04:37, 1.42MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:53, 1.68MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<02:53, 2.26MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<03:31, 1.84MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<03:48, 1.70MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<02:57, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:07, 3.02MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<14:55, 429kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<11:06, 577kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<07:54, 807kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<06:59, 907kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<05:31, 1.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:00, 1.57MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<04:17, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<04:17, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:15, 1.92MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:21, 2.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<04:29, 1.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<03:46, 1.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<02:46, 2.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:22, 1.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:37, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:48, 2.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:01, 3.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<05:56, 1.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<04:46, 1.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:29, 1.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<03:47, 1.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<03:50, 1.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:58, 2.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<03:02, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<02:44, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:02, 2.89MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<02:47, 2.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<02:32, 2.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<01:55, 3.03MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<02:42, 2.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<03:04, 1.88MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<02:26, 2.36MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<01:45, 3.24MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<1:07:53, 84.2kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<48:03, 119kB/s]   .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<33:37, 169kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<24:43, 228kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<17:51, 316kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<12:33, 447kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<10:03, 555kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<08:10, 683kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<05:56, 936kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<04:12, 1.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<05:43, 963kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<04:34, 1.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:19, 1.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<03:35, 1.52MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<03:03, 1.77MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:16, 2.38MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<02:51, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<02:32, 2.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<01:54, 2.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<02:34, 2.05MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<02:52, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:15, 2.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<01:37, 3.23MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<06:45, 776kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<05:15, 996kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:47, 1.37MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<03:51, 1.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<03:44, 1.38MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<02:52, 1.79MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:04<02:49, 1.81MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<02:29, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<01:51, 2.72MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:28, 2.03MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:06<02:45, 1.83MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<02:08, 2.34MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:33, 3.21MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<04:10, 1.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<03:25, 1.44MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:29, 1.98MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:53, 1.70MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<03:01, 1.62MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<02:21, 2.07MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:41, 2.87MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<4:40:43, 17.2kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<3:16:44, 24.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<2:17:05, 35.0kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<1:36:21, 49.4kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<1:08:22, 69.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<47:56, 98.9kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<33:21, 141kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<25:54, 181kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<18:36, 252kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<13:04, 356kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<10:09, 455kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<08:02, 574kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<05:49, 791kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<04:05, 1.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<05:43, 796kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<04:23, 1.03MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<03:09, 1.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:15, 1.99MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<12:52, 348kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<09:55, 451kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<07:09, 624kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<05:40, 779kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<04:24, 999kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<03:10, 1.38MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<03:13, 1.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<04:26, 979kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<03:06, 1.38MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<05:07, 834kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<04:02, 1.06MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:53, 1.47MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<03:00, 1.40MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:31, 1.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:51, 2.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<02:16, 1.82MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<02:00, 2.06MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:30, 2.73MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<02:00, 2.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<01:49, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:22, 2.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<01:53, 2.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<02:08, 1.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:41, 2.35MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<01:48, 2.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<01:40, 2.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:15, 3.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<01:46, 2.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<02:02, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<01:35, 2.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:09, 3.28MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<02:21, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<02:02, 1.86MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:30, 2.49MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<01:55, 1.94MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<01:43, 2.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:17, 2.85MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<01:45, 2.08MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<01:58, 1.85MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:32, 2.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:06, 3.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<02:40, 1.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<02:10, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:37, 2.20MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:09, 3.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:49<15:00, 235kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:49<11:12, 314kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<07:58, 440kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<05:33, 623kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<07:09, 482kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<05:21, 644kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:48, 899kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<03:25, 987kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<03:06, 1.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<02:20, 1.44MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:39, 2.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<13:11, 252kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<09:30, 349kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<06:41, 492kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<04:40, 696kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<09:24, 345kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<07:14, 448kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:57<05:11, 623kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<03:37, 880kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<04:38, 685kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<03:30, 903kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:30, 1.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:47, 1.74MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<15:32, 200kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<11:10, 278kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<07:49, 394kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<06:09, 494kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<04:34, 664kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:14, 928kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:56, 1.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:21, 1.26MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:42, 1.73MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:51, 1.56MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:35, 1.82MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:10, 2.44MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:29, 1.91MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:37, 1.75MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:16, 2.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<00:54, 3.06MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<2:40:22, 17.3kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<1:52:17, 24.6kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<1:17:55, 35.1kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<54:27, 49.6kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<38:37, 69.8kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<27:01, 99.3kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<18:39, 142kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<14:58, 176kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<10:40, 246kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<07:27, 349kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<05:11, 494kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<17:31, 146kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<12:30, 205kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<08:43, 290kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<06:36, 377kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<04:52, 511kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<03:26, 717kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<02:56, 824kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<02:18, 1.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:39, 1.45MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:41, 1.39MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:39, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:15, 1.85MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:54, 2.56MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:51, 1.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:31, 1.49MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:06, 2.02MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:17, 1.73MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:19, 1.67MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:02, 2.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<01:03, 2.03MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<00:57, 2.23MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:42, 2.99MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<00:58, 2.13MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:53, 2.32MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:40, 3.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<00:56, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<00:51, 2.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:38, 3.08MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<00:54, 2.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<00:49, 2.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:37, 3.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<00:52, 2.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<00:48, 2.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<00:35, 3.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<00:50, 2.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<00:57, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:45, 2.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:31, 3.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<1:40:43, 17.3kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<1:10:24, 24.6kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<48:37, 35.2kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<32:59, 50.2kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<24:13, 67.7kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<18:16, 89.7kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<13:08, 124kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<09:14, 176kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<06:18, 251kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<05:47, 271kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<04:10, 375kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<02:54, 529kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<02:00, 749kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<05:11, 289kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<03:55, 381kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<02:48, 530kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<01:54, 752kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<11:48, 122kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<08:22, 171kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<05:47, 242kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<04:16, 320kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<03:15, 418kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<02:19, 582kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:35, 823kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<02:05, 622kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:35, 814kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:07, 1.13MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:03, 1.17MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:51, 1.43MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:36, 1.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:41, 1.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:36, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:26, 2.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:33, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:29, 2.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:22, 2.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:29, 2.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:32, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:25, 2.34MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:26, 2.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:24, 2.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:17, 3.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:24, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:22, 2.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:16, 3.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:22, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:20, 2.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:15, 3.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:20, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:23, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:18, 2.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:12, 3.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:29, 1.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:24, 1.66MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:17, 2.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:20, 1.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:21, 1.71MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:16, 2.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:11, 3.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:21, 1.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:17, 1.80MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:12, 2.43MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:14, 1.93MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:15, 1.77MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:12, 2.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:11, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:13, 1.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:09, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:06, 3.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:17, 1.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:13, 1.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:09, 1.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:05, 2.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<01:08, 233kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:48, 322kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:30, 456kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:20, 565kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:16, 697kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:24<00:11, 955kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:07, 1.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:06, 1.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:04, 1.47MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:02, 2.00MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:02, 1.72MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 1.96MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 2.63MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<159:43:08,  1.44s/it]  0%|          | 768/400000 [00:01<111:35:35,  1.01s/it]  0%|          | 1518/400000 [00:01<77:58:22,  1.42it/s]  1%|          | 2224/400000 [00:01<54:29:20,  2.03it/s]  1%|          | 2852/400000 [00:01<38:05:14,  2.90it/s]  1%|          | 3576/400000 [00:01<26:37:01,  4.14it/s]  1%|          | 4342/400000 [00:02<18:36:01,  5.91it/s]  1%|▏         | 5117/400000 [00:02<12:59:56,  8.44it/s]  1%|▏         | 5872/400000 [00:02<9:05:10, 12.05it/s]   2%|▏         | 6622/400000 [00:02<6:21:09, 17.20it/s]  2%|▏         | 7358/400000 [00:02<4:26:34, 24.55it/s]  2%|▏         | 8153/400000 [00:02<3:06:28, 35.02it/s]  2%|▏         | 8897/400000 [00:02<2:10:32, 49.93it/s]  2%|▏         | 9651/400000 [00:02<1:31:28, 71.13it/s]  3%|▎         | 10403/400000 [00:02<1:04:09, 101.20it/s]  3%|▎         | 11151/400000 [00:02<45:06, 143.70it/s]    3%|▎         | 11928/400000 [00:03<31:45, 203.67it/s]  3%|▎         | 12720/400000 [00:03<22:25, 287.78it/s]  3%|▎         | 13482/400000 [00:03<15:57, 403.72it/s]  4%|▎         | 14282/400000 [00:03<11:23, 564.52it/s]  4%|▍         | 15037/400000 [00:03<08:13, 779.45it/s]  4%|▍         | 15784/400000 [00:03<06:00, 1065.76it/s]  4%|▍         | 16542/400000 [00:03<04:27, 1435.98it/s]  4%|▍         | 17301/400000 [00:03<03:21, 1897.32it/s]  5%|▍         | 18060/400000 [00:03<02:36, 2447.93it/s]  5%|▍         | 18812/400000 [00:03<02:04, 3062.01it/s]  5%|▍         | 19560/400000 [00:04<01:42, 3706.75it/s]  5%|▌         | 20323/400000 [00:04<01:26, 4382.73it/s]  5%|▌         | 21072/400000 [00:04<01:15, 5004.47it/s]  5%|▌         | 21821/400000 [00:04<01:08, 5524.79it/s]  6%|▌         | 22585/400000 [00:04<01:02, 6023.72it/s]  6%|▌         | 23334/400000 [00:04<01:00, 6263.41it/s]  6%|▌         | 24066/400000 [00:04<00:57, 6538.32it/s]  6%|▌         | 24841/400000 [00:04<00:54, 6860.00it/s]  6%|▋         | 25628/400000 [00:04<00:52, 7132.50it/s]  7%|▋         | 26385/400000 [00:04<00:52, 7167.07it/s]  7%|▋         | 27133/400000 [00:05<00:52, 7114.97it/s]  7%|▋         | 27866/400000 [00:05<00:52, 7073.34it/s]  7%|▋         | 28589/400000 [00:05<00:54, 6842.25it/s]  7%|▋         | 29287/400000 [00:05<00:53, 6881.24it/s]  7%|▋         | 29996/400000 [00:05<00:53, 6940.05it/s]  8%|▊         | 30735/400000 [00:05<00:52, 7067.05it/s]  8%|▊         | 31463/400000 [00:05<00:51, 7129.13it/s]  8%|▊         | 32185/400000 [00:05<00:51, 7154.16it/s]  8%|▊         | 32903/400000 [00:05<00:52, 6986.11it/s]  8%|▊         | 33608/400000 [00:06<00:52, 7003.20it/s]  9%|▊         | 34336/400000 [00:06<00:51, 7083.63it/s]  9%|▉         | 35046/400000 [00:06<00:51, 7082.09it/s]  9%|▉         | 35778/400000 [00:06<00:50, 7150.87it/s]  9%|▉         | 36512/400000 [00:06<00:50, 7204.27it/s]  9%|▉         | 37266/400000 [00:06<00:49, 7299.88it/s] 10%|▉         | 38057/400000 [00:06<00:48, 7469.42it/s] 10%|▉         | 38810/400000 [00:06<00:48, 7487.40it/s] 10%|▉         | 39560/400000 [00:06<00:48, 7417.65it/s] 10%|█         | 40339/400000 [00:06<00:47, 7524.80it/s] 10%|█         | 41093/400000 [00:07<00:49, 7313.18it/s] 10%|█         | 41853/400000 [00:07<00:48, 7395.79it/s] 11%|█         | 42613/400000 [00:07<00:48, 7295.94it/s] 11%|█         | 43345/400000 [00:07<00:49, 7268.46it/s] 11%|█         | 44083/400000 [00:07<00:48, 7300.48it/s] 11%|█         | 44840/400000 [00:07<00:48, 7379.22it/s] 11%|█▏        | 45630/400000 [00:07<00:47, 7526.49it/s] 12%|█▏        | 46398/400000 [00:07<00:46, 7570.01it/s] 12%|█▏        | 47156/400000 [00:07<00:46, 7550.48it/s] 12%|█▏        | 47912/400000 [00:07<00:47, 7460.75it/s] 12%|█▏        | 48659/400000 [00:08<00:47, 7433.42it/s] 12%|█▏        | 49413/400000 [00:08<00:46, 7463.60it/s] 13%|█▎        | 50170/400000 [00:08<00:46, 7495.01it/s] 13%|█▎        | 50949/400000 [00:08<00:46, 7580.37it/s] 13%|█▎        | 51708/400000 [00:08<00:46, 7549.76it/s] 13%|█▎        | 52464/400000 [00:08<00:46, 7475.89it/s] 13%|█▎        | 53212/400000 [00:08<00:50, 6927.43it/s] 13%|█▎        | 53969/400000 [00:08<00:48, 7106.73it/s] 14%|█▎        | 54744/400000 [00:08<00:47, 7286.38it/s] 14%|█▍        | 55529/400000 [00:08<00:46, 7445.54it/s] 14%|█▍        | 56306/400000 [00:09<00:45, 7538.98it/s] 14%|█▍        | 57121/400000 [00:09<00:44, 7711.24it/s] 14%|█▍        | 57896/400000 [00:09<00:44, 7630.13it/s] 15%|█▍        | 58693/400000 [00:09<00:44, 7726.87it/s] 15%|█▍        | 59468/400000 [00:09<00:44, 7689.42it/s] 15%|█▌        | 60239/400000 [00:09<00:45, 7521.62it/s] 15%|█▌        | 60994/400000 [00:09<00:45, 7424.69it/s] 15%|█▌        | 61739/400000 [00:09<00:45, 7422.96it/s] 16%|█▌        | 62483/400000 [00:09<00:45, 7364.43it/s] 16%|█▌        | 63221/400000 [00:10<00:47, 7113.73it/s] 16%|█▌        | 63935/400000 [00:10<00:47, 7025.40it/s] 16%|█▌        | 64640/400000 [00:10<00:48, 6972.82it/s] 16%|█▋        | 65363/400000 [00:10<00:47, 7047.94it/s] 17%|█▋        | 66070/400000 [00:10<00:47, 7008.32it/s] 17%|█▋        | 66850/400000 [00:10<00:46, 7225.57it/s] 17%|█▋        | 67575/400000 [00:10<00:46, 7211.48it/s] 17%|█▋        | 68354/400000 [00:10<00:44, 7375.51it/s] 17%|█▋        | 69130/400000 [00:10<00:44, 7484.35it/s] 17%|█▋        | 69912/400000 [00:10<00:43, 7579.48it/s] 18%|█▊        | 70721/400000 [00:11<00:42, 7724.94it/s] 18%|█▊        | 71496/400000 [00:11<00:43, 7485.75it/s] 18%|█▊        | 72326/400000 [00:11<00:42, 7712.43it/s] 18%|█▊        | 73151/400000 [00:11<00:41, 7865.63it/s] 18%|█▊        | 73941/400000 [00:11<00:43, 7540.52it/s] 19%|█▊        | 74796/400000 [00:11<00:41, 7815.90it/s] 19%|█▉        | 75591/400000 [00:11<00:41, 7853.48it/s] 19%|█▉        | 76381/400000 [00:11<00:41, 7815.21it/s] 19%|█▉        | 77241/400000 [00:11<00:40, 8034.25it/s] 20%|█▉        | 78049/400000 [00:11<00:40, 7921.77it/s] 20%|█▉        | 78845/400000 [00:12<00:40, 7901.75it/s] 20%|█▉        | 79638/400000 [00:12<00:41, 7764.83it/s] 20%|██        | 80417/400000 [00:12<00:41, 7636.37it/s] 20%|██        | 81227/400000 [00:12<00:41, 7768.15it/s] 21%|██        | 82035/400000 [00:12<00:40, 7857.26it/s] 21%|██        | 82834/400000 [00:12<00:40, 7894.94it/s] 21%|██        | 83625/400000 [00:12<00:41, 7670.55it/s] 21%|██        | 84396/400000 [00:12<00:41, 7681.32it/s] 21%|██▏       | 85166/400000 [00:12<00:41, 7574.78it/s] 21%|██▏       | 85925/400000 [00:12<00:41, 7576.54it/s] 22%|██▏       | 86757/400000 [00:13<00:40, 7784.89it/s] 22%|██▏       | 87538/400000 [00:13<00:41, 7447.30it/s] 22%|██▏       | 88288/400000 [00:13<00:41, 7444.94it/s] 22%|██▏       | 89036/400000 [00:13<00:42, 7339.92it/s] 22%|██▏       | 89777/400000 [00:13<00:42, 7358.16it/s] 23%|██▎       | 90537/400000 [00:13<00:41, 7428.69it/s] 23%|██▎       | 91282/400000 [00:13<00:42, 7231.05it/s] 23%|██▎       | 92008/400000 [00:13<00:42, 7235.23it/s] 23%|██▎       | 92734/400000 [00:13<00:43, 7104.82it/s] 23%|██▎       | 93447/400000 [00:14<00:43, 7067.37it/s] 24%|██▎       | 94164/400000 [00:14<00:43, 7095.77it/s] 24%|██▎       | 94943/400000 [00:14<00:41, 7288.72it/s] 24%|██▍       | 95782/400000 [00:14<00:40, 7587.49it/s] 24%|██▍       | 96618/400000 [00:14<00:38, 7801.95it/s] 24%|██▍       | 97405/400000 [00:14<00:38, 7818.78it/s] 25%|██▍       | 98285/400000 [00:14<00:37, 8087.63it/s] 25%|██▍       | 99099/400000 [00:14<00:37, 7951.41it/s] 25%|██▍       | 99976/400000 [00:14<00:36, 8177.35it/s] 25%|██▌       | 100826/400000 [00:14<00:36, 8268.29it/s] 25%|██▌       | 101659/400000 [00:15<00:36, 8284.98it/s] 26%|██▌       | 102525/400000 [00:15<00:35, 8393.61it/s] 26%|██▌       | 103367/400000 [00:15<00:36, 8222.46it/s] 26%|██▌       | 104192/400000 [00:15<00:37, 7956.88it/s] 26%|██▌       | 104992/400000 [00:15<00:38, 7584.22it/s] 26%|██▋       | 105786/400000 [00:15<00:38, 7685.32it/s] 27%|██▋       | 106601/400000 [00:15<00:37, 7817.81it/s] 27%|██▋       | 107387/400000 [00:15<00:37, 7710.90it/s] 27%|██▋       | 108162/400000 [00:15<00:38, 7570.15it/s] 27%|██▋       | 108922/400000 [00:15<00:39, 7327.51it/s] 27%|██▋       | 109746/400000 [00:16<00:38, 7578.60it/s] 28%|██▊       | 110509/400000 [00:16<00:38, 7550.24it/s] 28%|██▊       | 111340/400000 [00:16<00:37, 7762.15it/s] 28%|██▊       | 112175/400000 [00:16<00:36, 7929.33it/s] 28%|██▊       | 113016/400000 [00:16<00:35, 8064.28it/s] 28%|██▊       | 113835/400000 [00:16<00:35, 8100.09it/s] 29%|██▊       | 114656/400000 [00:16<00:35, 8131.43it/s] 29%|██▉       | 115471/400000 [00:16<00:35, 8058.56it/s] 29%|██▉       | 116306/400000 [00:16<00:34, 8141.89it/s] 29%|██▉       | 117122/400000 [00:17<00:35, 7974.54it/s] 29%|██▉       | 117921/400000 [00:17<00:36, 7719.87it/s] 30%|██▉       | 118696/400000 [00:17<00:36, 7724.45it/s] 30%|██▉       | 119471/400000 [00:17<00:37, 7538.13it/s] 30%|███       | 120230/400000 [00:17<00:37, 7552.04it/s] 30%|███       | 121035/400000 [00:17<00:36, 7692.78it/s] 30%|███       | 121823/400000 [00:17<00:35, 7746.79it/s] 31%|███       | 122600/400000 [00:17<00:36, 7690.85it/s] 31%|███       | 123415/400000 [00:17<00:35, 7820.53it/s] 31%|███       | 124266/400000 [00:17<00:34, 8013.49it/s] 31%|███▏      | 125070/400000 [00:18<00:34, 7976.78it/s] 31%|███▏      | 125870/400000 [00:18<00:34, 7920.50it/s] 32%|███▏      | 126664/400000 [00:18<00:35, 7762.72it/s] 32%|███▏      | 127519/400000 [00:18<00:34, 7982.41it/s] 32%|███▏      | 128338/400000 [00:18<00:33, 8043.23it/s] 32%|███▏      | 129253/400000 [00:18<00:32, 8345.48it/s] 33%|███▎      | 130101/400000 [00:18<00:32, 8382.90it/s] 33%|███▎      | 130943/400000 [00:18<00:33, 7921.60it/s] 33%|███▎      | 131743/400000 [00:18<00:34, 7762.70it/s] 33%|███▎      | 132526/400000 [00:18<00:35, 7622.84it/s] 33%|███▎      | 133293/400000 [00:19<00:35, 7546.44it/s] 34%|███▎      | 134052/400000 [00:19<00:36, 7374.91it/s] 34%|███▎      | 134793/400000 [00:19<00:36, 7310.23it/s] 34%|███▍      | 135527/400000 [00:19<00:36, 7313.42it/s] 34%|███▍      | 136341/400000 [00:19<00:34, 7542.68it/s] 34%|███▍      | 137123/400000 [00:19<00:34, 7621.72it/s] 34%|███▍      | 137888/400000 [00:19<00:34, 7609.26it/s] 35%|███▍      | 138654/400000 [00:19<00:34, 7621.74it/s] 35%|███▍      | 139418/400000 [00:19<00:34, 7615.85it/s] 35%|███▌      | 140181/400000 [00:20<00:34, 7544.15it/s] 35%|███▌      | 140961/400000 [00:20<00:34, 7615.45it/s] 35%|███▌      | 141724/400000 [00:20<00:33, 7610.87it/s] 36%|███▌      | 142486/400000 [00:20<00:34, 7495.89it/s] 36%|███▌      | 143237/400000 [00:20<00:34, 7426.29it/s] 36%|███▌      | 143990/400000 [00:20<00:34, 7455.12it/s] 36%|███▌      | 144781/400000 [00:20<00:33, 7583.18it/s] 36%|███▋      | 145541/400000 [00:20<00:34, 7474.91it/s] 37%|███▋      | 146290/400000 [00:20<00:34, 7455.81it/s] 37%|███▋      | 147155/400000 [00:20<00:32, 7776.51it/s] 37%|███▋      | 148006/400000 [00:21<00:31, 7979.81it/s] 37%|███▋      | 148812/400000 [00:21<00:31, 8001.77it/s] 37%|███▋      | 149616/400000 [00:21<00:31, 8003.63it/s] 38%|███▊      | 150419/400000 [00:21<00:31, 7930.80it/s] 38%|███▊      | 151214/400000 [00:21<00:32, 7591.28it/s] 38%|███▊      | 151978/400000 [00:21<00:33, 7398.10it/s] 38%|███▊      | 152757/400000 [00:21<00:32, 7509.29it/s] 38%|███▊      | 153513/400000 [00:21<00:32, 7523.24it/s] 39%|███▊      | 154268/400000 [00:21<00:33, 7309.29it/s] 39%|███▉      | 155102/400000 [00:21<00:32, 7590.14it/s] 39%|███▉      | 155866/400000 [00:22<00:32, 7423.54it/s] 39%|███▉      | 156613/400000 [00:22<00:32, 7407.59it/s] 39%|███▉      | 157391/400000 [00:22<00:32, 7512.84it/s] 40%|███▉      | 158173/400000 [00:22<00:31, 7601.53it/s] 40%|███▉      | 159040/400000 [00:22<00:30, 7891.82it/s] 40%|███▉      | 159852/400000 [00:22<00:30, 7957.90it/s] 40%|████      | 160667/400000 [00:22<00:29, 8012.96it/s] 40%|████      | 161482/400000 [00:22<00:29, 8051.68it/s] 41%|████      | 162289/400000 [00:22<00:29, 7946.07it/s] 41%|████      | 163095/400000 [00:22<00:29, 7979.01it/s] 41%|████      | 163917/400000 [00:23<00:29, 8049.42it/s] 41%|████      | 164789/400000 [00:23<00:28, 8238.47it/s] 41%|████▏     | 165615/400000 [00:23<00:28, 8164.88it/s] 42%|████▏     | 166433/400000 [00:23<00:29, 8008.68it/s] 42%|████▏     | 167236/400000 [00:23<00:29, 8012.21it/s] 42%|████▏     | 168039/400000 [00:23<00:29, 7919.54it/s] 42%|████▏     | 168832/400000 [00:23<00:29, 7895.98it/s] 42%|████▏     | 169623/400000 [00:23<00:30, 7447.05it/s] 43%|████▎     | 170384/400000 [00:23<00:30, 7491.40it/s] 43%|████▎     | 171240/400000 [00:24<00:29, 7782.46it/s] 43%|████▎     | 172072/400000 [00:24<00:28, 7935.16it/s] 43%|████▎     | 172872/400000 [00:24<00:28, 7951.61it/s] 43%|████▎     | 173671/400000 [00:24<00:28, 7946.22it/s] 44%|████▎     | 174515/400000 [00:24<00:27, 8086.36it/s] 44%|████▍     | 175326/400000 [00:24<00:28, 8002.77it/s] 44%|████▍     | 176129/400000 [00:24<00:28, 7895.35it/s] 44%|████▍     | 176965/400000 [00:24<00:27, 8027.42it/s] 44%|████▍     | 177778/400000 [00:24<00:27, 8057.25it/s] 45%|████▍     | 178591/400000 [00:24<00:27, 8077.33it/s] 45%|████▍     | 179462/400000 [00:25<00:26, 8255.08it/s] 45%|████▌     | 180290/400000 [00:25<00:27, 8119.83it/s] 45%|████▌     | 181104/400000 [00:25<00:27, 7962.90it/s] 45%|████▌     | 181903/400000 [00:25<00:28, 7560.94it/s] 46%|████▌     | 182665/400000 [00:25<00:29, 7467.73it/s] 46%|████▌     | 183416/400000 [00:25<00:29, 7390.14it/s] 46%|████▌     | 184188/400000 [00:25<00:28, 7483.07it/s] 46%|████▌     | 184939/400000 [00:25<00:28, 7442.95it/s] 46%|████▋     | 185686/400000 [00:25<00:29, 7341.88it/s] 47%|████▋     | 186462/400000 [00:25<00:28, 7460.46it/s] 47%|████▋     | 187258/400000 [00:26<00:27, 7600.71it/s] 47%|████▋     | 188064/400000 [00:26<00:27, 7730.33it/s] 47%|████▋     | 188842/400000 [00:26<00:27, 7743.24it/s] 47%|████▋     | 189618/400000 [00:26<00:27, 7670.59it/s] 48%|████▊     | 190387/400000 [00:26<00:27, 7588.41it/s] 48%|████▊     | 191147/400000 [00:26<00:27, 7591.76it/s] 48%|████▊     | 191918/400000 [00:26<00:27, 7625.64it/s] 48%|████▊     | 192682/400000 [00:26<00:27, 7451.73it/s] 48%|████▊     | 193447/400000 [00:26<00:27, 7508.86it/s] 49%|████▊     | 194199/400000 [00:26<00:28, 7259.42it/s] 49%|████▊     | 194971/400000 [00:27<00:27, 7390.82it/s] 49%|████▉     | 195713/400000 [00:27<00:28, 7152.41it/s] 49%|████▉     | 196439/400000 [00:27<00:28, 7183.02it/s] 49%|████▉     | 197160/400000 [00:27<00:28, 7153.69it/s] 49%|████▉     | 197955/400000 [00:27<00:27, 7373.55it/s] 50%|████▉     | 198744/400000 [00:27<00:26, 7520.75it/s] 50%|████▉     | 199499/400000 [00:27<00:27, 7204.22it/s] 50%|█████     | 200289/400000 [00:27<00:26, 7397.88it/s] 50%|█████     | 201034/400000 [00:27<00:27, 7161.33it/s] 50%|█████     | 201780/400000 [00:28<00:27, 7246.94it/s] 51%|█████     | 202577/400000 [00:28<00:26, 7449.47it/s] 51%|█████     | 203342/400000 [00:28<00:26, 7506.88it/s] 51%|█████     | 204098/400000 [00:28<00:26, 7521.65it/s] 51%|█████     | 204853/400000 [00:28<00:26, 7413.60it/s] 51%|█████▏    | 205629/400000 [00:28<00:25, 7514.14it/s] 52%|█████▏    | 206423/400000 [00:28<00:25, 7635.80it/s] 52%|█████▏    | 207189/400000 [00:28<00:25, 7449.49it/s] 52%|█████▏    | 207937/400000 [00:28<00:25, 7451.43it/s] 52%|█████▏    | 208728/400000 [00:28<00:25, 7581.70it/s] 52%|█████▏    | 209530/400000 [00:29<00:24, 7706.32it/s] 53%|█████▎    | 210303/400000 [00:29<00:24, 7706.97it/s] 53%|█████▎    | 211075/400000 [00:29<00:24, 7675.20it/s] 53%|█████▎    | 211865/400000 [00:29<00:24, 7739.11it/s] 53%|█████▎    | 212640/400000 [00:29<00:24, 7519.10it/s] 53%|█████▎    | 213394/400000 [00:29<00:24, 7519.63it/s] 54%|█████▎    | 214176/400000 [00:29<00:24, 7605.01it/s] 54%|█████▎    | 214938/400000 [00:29<00:24, 7605.74it/s] 54%|█████▍    | 215700/400000 [00:29<00:24, 7543.47it/s] 54%|█████▍    | 216456/400000 [00:29<00:24, 7428.02it/s] 54%|█████▍    | 217215/400000 [00:30<00:24, 7473.16it/s] 54%|█████▍    | 217985/400000 [00:30<00:24, 7537.55it/s] 55%|█████▍    | 218745/400000 [00:30<00:23, 7554.43it/s] 55%|█████▍    | 219516/400000 [00:30<00:23, 7598.95it/s] 55%|█████▌    | 220277/400000 [00:30<00:24, 7348.72it/s] 55%|█████▌    | 221029/400000 [00:30<00:24, 7397.01it/s] 55%|█████▌    | 221782/400000 [00:30<00:23, 7436.10it/s] 56%|█████▌    | 222629/400000 [00:30<00:22, 7717.83it/s] 56%|█████▌    | 223405/400000 [00:30<00:22, 7679.59it/s] 56%|█████▌    | 224221/400000 [00:30<00:22, 7817.33it/s] 56%|█████▋    | 225006/400000 [00:31<00:22, 7803.10it/s] 56%|█████▋    | 225788/400000 [00:31<00:22, 7591.27it/s] 57%|█████▋    | 226601/400000 [00:31<00:22, 7744.75it/s] 57%|█████▋    | 227379/400000 [00:31<00:22, 7753.42it/s] 57%|█████▋    | 228157/400000 [00:31<00:22, 7739.95it/s] 57%|█████▋    | 228933/400000 [00:31<00:22, 7658.09it/s] 57%|█████▋    | 229742/400000 [00:31<00:21, 7780.63it/s] 58%|█████▊    | 230522/400000 [00:31<00:22, 7658.74it/s] 58%|█████▊    | 231290/400000 [00:31<00:22, 7625.13it/s] 58%|█████▊    | 232087/400000 [00:32<00:21, 7722.10it/s] 58%|█████▊    | 232861/400000 [00:32<00:21, 7722.23it/s] 58%|█████▊    | 233634/400000 [00:32<00:21, 7724.42it/s] 59%|█████▊    | 234421/400000 [00:32<00:21, 7765.73it/s] 59%|█████▉    | 235225/400000 [00:32<00:21, 7843.81it/s] 59%|█████▉    | 236010/400000 [00:32<00:21, 7601.23it/s] 59%|█████▉    | 236773/400000 [00:32<00:21, 7473.90it/s] 59%|█████▉    | 237528/400000 [00:32<00:21, 7495.91it/s] 60%|█████▉    | 238287/400000 [00:32<00:21, 7521.23it/s] 60%|█████▉    | 239041/400000 [00:32<00:21, 7508.75it/s] 60%|█████▉    | 239793/400000 [00:33<00:21, 7429.05it/s] 60%|██████    | 240588/400000 [00:33<00:21, 7576.19it/s] 60%|██████    | 241391/400000 [00:33<00:20, 7706.42it/s] 61%|██████    | 242209/400000 [00:33<00:20, 7841.56it/s] 61%|██████    | 242995/400000 [00:33<00:20, 7666.37it/s] 61%|██████    | 243764/400000 [00:33<00:20, 7631.49it/s] 61%|██████    | 244529/400000 [00:33<00:20, 7525.09it/s] 61%|██████▏   | 245283/400000 [00:33<00:21, 7363.14it/s] 62%|██████▏   | 246022/400000 [00:33<00:21, 7219.81it/s] 62%|██████▏   | 246750/400000 [00:33<00:21, 7237.37it/s] 62%|██████▏   | 247484/400000 [00:34<00:20, 7266.22it/s] 62%|██████▏   | 248250/400000 [00:34<00:20, 7379.99it/s] 62%|██████▏   | 249012/400000 [00:34<00:20, 7449.92it/s] 62%|██████▏   | 249791/400000 [00:34<00:19, 7546.90it/s] 63%|██████▎   | 250551/400000 [00:34<00:19, 7560.76it/s] 63%|██████▎   | 251317/400000 [00:34<00:19, 7588.86it/s] 63%|██████▎   | 252077/400000 [00:34<00:19, 7469.49it/s] 63%|██████▎   | 252863/400000 [00:34<00:19, 7582.33it/s] 63%|██████▎   | 253623/400000 [00:34<00:19, 7560.21it/s] 64%|██████▎   | 254408/400000 [00:34<00:19, 7642.97it/s] 64%|██████▍   | 255173/400000 [00:35<00:18, 7643.91it/s] 64%|██████▍   | 255949/400000 [00:35<00:18, 7676.70it/s] 64%|██████▍   | 256771/400000 [00:35<00:18, 7831.90it/s] 64%|██████▍   | 257574/400000 [00:35<00:18, 7889.04it/s] 65%|██████▍   | 258364/400000 [00:35<00:18, 7698.75it/s] 65%|██████▍   | 259136/400000 [00:35<00:18, 7597.84it/s] 65%|██████▍   | 259898/400000 [00:35<00:18, 7585.83it/s] 65%|██████▌   | 260717/400000 [00:35<00:17, 7755.65it/s] 65%|██████▌   | 261589/400000 [00:35<00:17, 8019.31it/s] 66%|██████▌   | 262435/400000 [00:35<00:16, 8145.36it/s] 66%|██████▌   | 263253/400000 [00:36<00:17, 7979.63it/s] 66%|██████▌   | 264136/400000 [00:36<00:16, 8216.62it/s] 66%|██████▌   | 264962/400000 [00:36<00:16, 8212.73it/s] 66%|██████▋   | 265797/400000 [00:36<00:16, 8253.19it/s] 67%|██████▋   | 266625/400000 [00:36<00:16, 8214.95it/s] 67%|██████▋   | 267448/400000 [00:36<00:16, 8138.46it/s] 67%|██████▋   | 268276/400000 [00:36<00:16, 8178.98it/s] 67%|██████▋   | 269095/400000 [00:36<00:16, 8070.81it/s] 67%|██████▋   | 269926/400000 [00:36<00:15, 8139.59it/s] 68%|██████▊   | 270804/400000 [00:36<00:15, 8320.38it/s] 68%|██████▊   | 271638/400000 [00:37<00:15, 8241.75it/s] 68%|██████▊   | 272464/400000 [00:37<00:16, 7932.61it/s] 68%|██████▊   | 273264/400000 [00:37<00:15, 7952.02it/s] 69%|██████▊   | 274062/400000 [00:37<00:15, 7902.94it/s] 69%|██████▊   | 274855/400000 [00:37<00:15, 7899.63it/s] 69%|██████▉   | 275647/400000 [00:37<00:15, 7774.43it/s] 69%|██████▉   | 276427/400000 [00:37<00:15, 7782.07it/s] 69%|██████▉   | 277243/400000 [00:37<00:15, 7889.85it/s] 70%|██████▉   | 278033/400000 [00:37<00:15, 7852.38it/s] 70%|██████▉   | 278833/400000 [00:38<00:15, 7893.04it/s] 70%|██████▉   | 279623/400000 [00:38<00:15, 7587.63it/s] 70%|███████   | 280385/400000 [00:38<00:16, 7269.90it/s] 70%|███████   | 281117/400000 [00:38<00:16, 7251.72it/s] 70%|███████   | 281903/400000 [00:38<00:15, 7422.62it/s] 71%|███████   | 282667/400000 [00:38<00:15, 7485.99it/s] 71%|███████   | 283419/400000 [00:38<00:15, 7389.52it/s] 71%|███████   | 284160/400000 [00:38<00:15, 7258.86it/s] 71%|███████   | 284909/400000 [00:38<00:15, 7325.73it/s] 71%|███████▏  | 285644/400000 [00:38<00:15, 7260.89it/s] 72%|███████▏  | 286444/400000 [00:39<00:15, 7467.35it/s] 72%|███████▏  | 287194/400000 [00:39<00:15, 7299.10it/s] 72%|███████▏  | 288021/400000 [00:39<00:14, 7565.18it/s] 72%|███████▏  | 288782/400000 [00:39<00:14, 7563.99it/s] 72%|███████▏  | 289559/400000 [00:39<00:14, 7621.59it/s] 73%|███████▎  | 290338/400000 [00:39<00:14, 7668.94it/s] 73%|███████▎  | 291107/400000 [00:39<00:14, 7606.64it/s] 73%|███████▎  | 291869/400000 [00:39<00:14, 7531.83it/s] 73%|███████▎  | 292646/400000 [00:39<00:14, 7600.92it/s] 73%|███████▎  | 293407/400000 [00:39<00:14, 7550.35it/s] 74%|███████▎  | 294194/400000 [00:40<00:13, 7642.54it/s] 74%|███████▎  | 294960/400000 [00:40<00:14, 7352.42it/s] 74%|███████▍  | 295776/400000 [00:40<00:13, 7576.51it/s] 74%|███████▍  | 296538/400000 [00:40<00:13, 7484.00it/s] 74%|███████▍  | 297290/400000 [00:40<00:13, 7369.05it/s] 75%|███████▍  | 298030/400000 [00:40<00:14, 7237.34it/s] 75%|███████▍  | 298756/400000 [00:40<00:14, 7203.76it/s] 75%|███████▍  | 299539/400000 [00:40<00:13, 7379.01it/s] 75%|███████▌  | 300354/400000 [00:40<00:13, 7592.39it/s] 75%|███████▌  | 301117/400000 [00:41<00:13, 7557.99it/s] 75%|███████▌  | 301882/400000 [00:41<00:12, 7584.63it/s] 76%|███████▌  | 302643/400000 [00:41<00:12, 7515.82it/s] 76%|███████▌  | 303415/400000 [00:41<00:12, 7572.64it/s] 76%|███████▌  | 304174/400000 [00:41<00:13, 7303.71it/s] 76%|███████▌  | 304952/400000 [00:41<00:12, 7436.77it/s] 76%|███████▋  | 305717/400000 [00:41<00:12, 7498.52it/s] 77%|███████▋  | 306469/400000 [00:41<00:12, 7478.19it/s] 77%|███████▋  | 307310/400000 [00:41<00:11, 7735.24it/s] 77%|███████▋  | 308127/400000 [00:41<00:11, 7859.95it/s] 77%|███████▋  | 309004/400000 [00:42<00:11, 8111.15it/s] 77%|███████▋  | 309857/400000 [00:42<00:10, 8229.80it/s] 78%|███████▊  | 310684/400000 [00:42<00:11, 8087.35it/s] 78%|███████▊  | 311566/400000 [00:42<00:10, 8293.29it/s] 78%|███████▊  | 312444/400000 [00:42<00:10, 8432.22it/s] 78%|███████▊  | 313315/400000 [00:42<00:10, 8510.58it/s] 79%|███████▊  | 314176/400000 [00:42<00:10, 8538.71it/s] 79%|███████▉  | 315032/400000 [00:42<00:10, 8367.44it/s] 79%|███████▉  | 315871/400000 [00:42<00:10, 7999.01it/s] 79%|███████▉  | 316676/400000 [00:42<00:10, 7650.82it/s] 79%|███████▉  | 317448/400000 [00:43<00:10, 7574.95it/s] 80%|███████▉  | 318218/400000 [00:43<00:10, 7609.95it/s] 80%|███████▉  | 319054/400000 [00:43<00:10, 7819.78it/s] 80%|███████▉  | 319925/400000 [00:43<00:09, 8064.77it/s] 80%|████████  | 320783/400000 [00:43<00:09, 8212.13it/s] 80%|████████  | 321631/400000 [00:43<00:09, 8290.41it/s] 81%|████████  | 322463/400000 [00:43<00:09, 8146.29it/s] 81%|████████  | 323281/400000 [00:43<00:09, 8045.30it/s] 81%|████████  | 324088/400000 [00:43<00:09, 7988.44it/s] 81%|████████  | 324889/400000 [00:44<00:09, 7934.17it/s] 81%|████████▏ | 325702/400000 [00:44<00:09, 7989.35it/s] 82%|████████▏ | 326502/400000 [00:44<00:09, 7972.76it/s] 82%|████████▏ | 327300/400000 [00:44<00:09, 7862.49it/s] 82%|████████▏ | 328088/400000 [00:44<00:09, 7733.63it/s] 82%|████████▏ | 328863/400000 [00:44<00:09, 7404.96it/s] 82%|████████▏ | 329647/400000 [00:44<00:09, 7529.72it/s] 83%|████████▎ | 330404/400000 [00:44<00:09, 7261.86it/s] 83%|████████▎ | 331152/400000 [00:44<00:09, 7325.77it/s] 83%|████████▎ | 331888/400000 [00:44<00:09, 7326.70it/s] 83%|████████▎ | 332623/400000 [00:45<00:09, 7155.45it/s] 83%|████████▎ | 333420/400000 [00:45<00:09, 7380.60it/s] 84%|████████▎ | 334162/400000 [00:45<00:08, 7338.50it/s] 84%|████████▎ | 334899/400000 [00:45<00:08, 7305.77it/s] 84%|████████▍ | 335633/400000 [00:45<00:08, 7312.98it/s] 84%|████████▍ | 336374/400000 [00:45<00:08, 7340.32it/s] 84%|████████▍ | 337109/400000 [00:45<00:08, 7269.24it/s] 84%|████████▍ | 337837/400000 [00:45<00:08, 7245.84it/s] 85%|████████▍ | 338628/400000 [00:45<00:08, 7432.55it/s] 85%|████████▍ | 339435/400000 [00:45<00:07, 7611.02it/s] 85%|████████▌ | 340265/400000 [00:46<00:07, 7804.05it/s] 85%|████████▌ | 341049/400000 [00:46<00:07, 7757.15it/s] 85%|████████▌ | 341883/400000 [00:46<00:07, 7922.04it/s] 86%|████████▌ | 342715/400000 [00:46<00:07, 8034.82it/s] 86%|████████▌ | 343614/400000 [00:46<00:06, 8295.47it/s] 86%|████████▌ | 344484/400000 [00:46<00:06, 8410.56it/s] 86%|████████▋ | 345359/400000 [00:46<00:06, 8509.17it/s] 87%|████████▋ | 346213/400000 [00:46<00:06, 8515.36it/s] 87%|████████▋ | 347067/400000 [00:46<00:06, 8272.06it/s] 87%|████████▋ | 347897/400000 [00:46<00:06, 8177.84it/s] 87%|████████▋ | 348717/400000 [00:47<00:06, 8050.62it/s] 87%|████████▋ | 349542/400000 [00:47<00:06, 8107.68it/s] 88%|████████▊ | 350355/400000 [00:47<00:06, 8057.94it/s] 88%|████████▊ | 351162/400000 [00:47<00:06, 8002.93it/s] 88%|████████▊ | 351964/400000 [00:47<00:06, 7710.42it/s] 88%|████████▊ | 352776/400000 [00:47<00:06, 7828.88it/s] 88%|████████▊ | 353562/400000 [00:47<00:05, 7773.91it/s] 89%|████████▊ | 354342/400000 [00:47<00:06, 7578.55it/s] 89%|████████▉ | 355103/400000 [00:47<00:05, 7549.48it/s] 89%|████████▉ | 355890/400000 [00:48<00:05, 7642.00it/s] 89%|████████▉ | 356672/400000 [00:48<00:05, 7694.47it/s] 89%|████████▉ | 357443/400000 [00:48<00:05, 7628.66it/s] 90%|████████▉ | 358207/400000 [00:48<00:05, 7166.71it/s] 90%|████████▉ | 358955/400000 [00:48<00:05, 7256.61it/s] 90%|████████▉ | 359712/400000 [00:48<00:05, 7347.37it/s] 90%|█████████ | 360451/400000 [00:48<00:05, 7187.94it/s] 90%|█████████ | 361201/400000 [00:48<00:05, 7278.34it/s] 90%|█████████ | 361933/400000 [00:48<00:05, 7288.86it/s] 91%|█████████ | 362664/400000 [00:48<00:05, 7116.65it/s] 91%|█████████ | 363378/400000 [00:49<00:05, 7102.62it/s] 91%|█████████ | 364144/400000 [00:49<00:04, 7258.46it/s] 91%|█████████ | 364872/400000 [00:49<00:04, 7170.95it/s] 91%|█████████▏| 365595/400000 [00:49<00:04, 7186.52it/s] 92%|█████████▏| 366327/400000 [00:49<00:04, 7223.29it/s] 92%|█████████▏| 367067/400000 [00:49<00:04, 7271.38it/s] 92%|█████████▏| 367831/400000 [00:49<00:04, 7376.15it/s] 92%|█████████▏| 368570/400000 [00:49<00:04, 7326.12it/s] 92%|█████████▏| 369304/400000 [00:49<00:04, 7226.44it/s] 93%|█████████▎| 370074/400000 [00:49<00:04, 7360.68it/s] 93%|█████████▎| 370835/400000 [00:50<00:03, 7431.52it/s] 93%|█████████▎| 371602/400000 [00:50<00:03, 7499.90it/s] 93%|█████████▎| 372378/400000 [00:50<00:03, 7574.17it/s] 93%|█████████▎| 373137/400000 [00:50<00:03, 7350.39it/s] 93%|█████████▎| 373875/400000 [00:50<00:03, 7319.45it/s] 94%|█████████▎| 374609/400000 [00:50<00:03, 7312.66it/s] 94%|█████████▍| 375342/400000 [00:50<00:03, 7178.57it/s] 94%|█████████▍| 376095/400000 [00:50<00:03, 7277.78it/s] 94%|█████████▍| 376824/400000 [00:50<00:03, 7246.94it/s] 94%|█████████▍| 377603/400000 [00:50<00:03, 7399.42it/s] 95%|█████████▍| 378345/400000 [00:51<00:03, 7049.72it/s] 95%|█████████▍| 379125/400000 [00:51<00:02, 7258.59it/s] 95%|█████████▍| 379903/400000 [00:51<00:02, 7405.88it/s] 95%|█████████▌| 380690/400000 [00:51<00:02, 7538.84it/s] 95%|█████████▌| 381448/400000 [00:51<00:02, 7528.64it/s] 96%|█████████▌| 382241/400000 [00:51<00:02, 7642.19it/s] 96%|█████████▌| 383040/400000 [00:51<00:02, 7741.29it/s] 96%|█████████▌| 383816/400000 [00:51<00:02, 7631.17it/s] 96%|█████████▌| 384581/400000 [00:51<00:02, 7555.78it/s] 96%|█████████▋| 385410/400000 [00:52<00:01, 7761.17it/s] 97%|█████████▋| 386216/400000 [00:52<00:01, 7847.74it/s] 97%|█████████▋| 387003/400000 [00:52<00:01, 7705.63it/s] 97%|█████████▋| 387776/400000 [00:52<00:01, 7709.64it/s] 97%|█████████▋| 388549/400000 [00:52<00:01, 7684.94it/s] 97%|█████████▋| 389380/400000 [00:52<00:01, 7859.96it/s] 98%|█████████▊| 390193/400000 [00:52<00:01, 7938.69it/s] 98%|█████████▊| 391007/400000 [00:52<00:01, 7995.71it/s] 98%|█████████▊| 391808/400000 [00:52<00:01, 7906.37it/s] 98%|█████████▊| 392600/400000 [00:52<00:00, 7701.97it/s] 98%|█████████▊| 393440/400000 [00:53<00:00, 7898.35it/s] 99%|█████████▊| 394247/400000 [00:53<00:00, 7948.65it/s] 99%|█████████▉| 395044/400000 [00:53<00:00, 7712.81it/s] 99%|█████████▉| 395819/400000 [00:53<00:00, 7509.39it/s] 99%|█████████▉| 396574/400000 [00:53<00:00, 7305.38it/s] 99%|█████████▉| 397308/400000 [00:53<00:00, 7294.84it/s]100%|█████████▉| 398040/400000 [00:53<00:00, 7275.25it/s]100%|█████████▉| 398804/400000 [00:53<00:00, 7379.17it/s]100%|█████████▉| 399560/400000 [00:53<00:00, 7430.59it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7416.38it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f634d291ba8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011071185697301774 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011093313279359238 	 Accuracy: 51

  model saves at 51% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15713 out of table with 15694 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15713 out of table with 15694 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-19 09:00:17.259376: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-19 09:00:17.263237: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-19 09:00:17.263400: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e229b16130 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-19 09:00:17.263416: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f62f8b51048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7510 - accuracy: 0.4945
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6513 - accuracy: 0.5010 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6245 - accuracy: 0.5027
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6636 - accuracy: 0.5002
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7177 - accuracy: 0.4967
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6841 - accuracy: 0.4989
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6264 - accuracy: 0.5026
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6291 - accuracy: 0.5024
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6145 - accuracy: 0.5034
11000/25000 [============>.................] - ETA: 4s - loss: 7.6415 - accuracy: 0.5016
12000/25000 [=============>................] - ETA: 4s - loss: 7.6321 - accuracy: 0.5023
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6265 - accuracy: 0.5026
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6349 - accuracy: 0.5021
15000/25000 [=================>............] - ETA: 3s - loss: 7.6124 - accuracy: 0.5035
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6254 - accuracy: 0.5027
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6269 - accuracy: 0.5026
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6240 - accuracy: 0.5028
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6327 - accuracy: 0.5022
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6344 - accuracy: 0.5021
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6265 - accuracy: 0.5026
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6401 - accuracy: 0.5017
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6420 - accuracy: 0.5016
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6436 - accuracy: 0.5015
25000/25000 [==============================] - 10s 399us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f62cdce6ef0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f62cdb786d8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.2068 - crf_viterbi_accuracy: 0.0267 - val_loss: 1.1123 - val_crf_viterbi_accuracy: 0.0000e+00

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

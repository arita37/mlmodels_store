
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/b4834e37b5f52c474468dc44c2de9ffdc4143259', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'b4834e37b5f52c474468dc44c2de9ffdc4143259', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/b4834e37b5f52c474468dc44c2de9ffdc4143259

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/b4834e37b5f52c474468dc44c2de9ffdc4143259

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fcee1348fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 14:13:20.022116
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 14:13:20.027208
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 14:13:20.030918
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 14:13:20.034838
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fceed112470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353426.4688
Epoch 2/10

1/1 [==============================] - 0s 105ms/step - loss: 256274.4688
Epoch 3/10

1/1 [==============================] - 0s 112ms/step - loss: 153036.2031
Epoch 4/10

1/1 [==============================] - 0s 112ms/step - loss: 85960.3828
Epoch 5/10

1/1 [==============================] - 0s 138ms/step - loss: 50614.3984
Epoch 6/10

1/1 [==============================] - 0s 117ms/step - loss: 32065.7656
Epoch 7/10

1/1 [==============================] - 0s 102ms/step - loss: 21717.9062
Epoch 8/10

1/1 [==============================] - 0s 105ms/step - loss: 15552.6074
Epoch 9/10

1/1 [==============================] - 0s 100ms/step - loss: 11603.0430
Epoch 10/10

1/1 [==============================] - 0s 103ms/step - loss: 8981.6826

  #### Inference Need return ypred, ytrue ######################### 
[[-8.4836382e-01  6.3430727e-01  8.3959758e-01 -1.0312684e+00
   4.8586303e-01  4.7816417e-01 -4.8587799e-01  8.9146173e-01
   8.0075318e-01  1.7039886e-01 -1.2758432e-01  3.6820844e-01
   6.7859679e-01  6.5052485e-01  3.7294599e-01  6.1303908e-01
   9.4143903e-01 -1.7123380e+00  3.4720302e-02 -3.7268251e-02
  -5.2165782e-01  9.1534412e-01 -2.0365779e+00  3.9250973e-01
  -5.2281797e-02 -1.1878173e+00 -6.9092298e-01 -8.2503581e-01
   5.7254672e-01 -1.8331895e+00 -2.2693060e-01  7.6290822e-01
  -4.6464121e-01  3.8956362e-01  5.0344360e-01  1.2104570e+00
   6.2441301e-01  4.6099859e-01  6.6252214e-01  1.4205148e+00
  -1.3849328e+00  6.0905874e-02 -1.0260532e+00  8.2412827e-01
  -4.3148494e-01  1.2209654e-02 -1.4692229e+00 -1.0396030e+00
  -2.5244504e-01 -8.9384818e-01 -1.2246116e+00  7.6841533e-02
  -8.1772053e-01  3.1518209e-01  1.3888986e+00 -4.2778760e-01
   8.7164474e-01 -5.9562999e-01  4.5999205e-01 -7.8833514e-01
   1.0048178e+00 -1.2441514e+00 -2.7322966e-01 -6.5427184e-02
  -1.3748956e-01 -5.5838335e-01 -1.7772412e+00 -8.2609332e-01
  -7.6462692e-01  7.2867298e-01 -7.8194968e-02 -3.8222122e-01
  -4.4278234e-01  2.7710915e-02  1.3165104e-01 -6.0151625e-01
  -1.0985423e+00 -6.2472031e-02 -1.1659493e+00 -4.4046280e-01
  -1.5840169e+00 -2.2867739e-01  1.5757944e-01  2.3770690e-02
   1.5326695e+00  1.5596642e+00 -5.3139067e-01  9.0984249e-01
   9.3955886e-01 -9.3932211e-01  5.3947347e-01 -2.1443026e+00
  -1.7144754e+00  9.3882215e-01  9.4711566e-01  1.2897073e+00
   1.0334718e+00 -1.0418845e+00 -1.4119442e+00 -7.4801522e-01
   2.2882539e-01 -1.5721650e-01 -8.1164920e-01  2.4578735e-02
   3.6398390e-01 -3.8350141e-01  4.4987756e-01  6.5350497e-01
  -1.7504990e-03 -2.0402288e-01 -8.4292114e-02  1.2224414e+00
   4.6594214e-01  2.4558836e-01  1.0252137e+00 -2.9749191e-01
   1.4906075e+00  6.2652659e-01 -2.8666258e-02 -1.0217757e+00
  -1.1347751e-01  6.5984178e+00  5.8648324e+00  5.9334035e+00
   7.5900807e+00  7.2387576e+00  6.6431770e+00  7.1639938e+00
   5.4852571e+00  6.1020083e+00  6.0229197e+00  5.6995773e+00
   4.7519288e+00  6.5050721e+00  5.6457558e+00  7.5717726e+00
   7.7094522e+00  8.1718636e+00  6.3038087e+00  7.1350703e+00
   8.2468538e+00  6.8637056e+00  6.6401591e+00  6.9696040e+00
   7.9530687e+00  7.5837474e+00  5.9267006e+00  5.4716988e+00
   4.9398518e+00  6.6277242e+00  5.8156633e+00  6.0311394e+00
   7.5091720e+00  6.4724045e+00  6.9042101e+00  7.6698046e+00
   5.5823002e+00  6.2589250e+00  6.8810172e+00  7.2863631e+00
   5.8774776e+00  7.2464671e+00  5.7017169e+00  7.1586051e+00
   6.5723772e+00  6.4796414e+00  7.5869761e+00  5.5608978e+00
   7.3534474e+00  7.4062872e+00  7.0179653e+00  7.1554561e+00
   6.9998736e+00  7.0166149e+00  6.7625031e+00  6.5490236e+00
   7.3991385e+00  7.2022710e+00  6.0984302e+00  7.6739926e+00
   4.1979289e-01  2.0600410e+00  1.1899350e+00  1.5038282e-01
   4.7985625e-01  8.6100304e-01  1.0695997e+00  5.0321859e-01
   1.3106260e+00  9.1999757e-01  3.5676116e-01  5.9513789e-01
   9.2526573e-01  2.1582313e+00  1.5389285e+00  1.8235197e+00
   1.3289845e+00  7.9234439e-01  8.7068880e-01  1.8419729e+00
   4.4356942e-01  3.5000223e-01  1.8088937e+00  1.3405919e-01
   2.0605721e+00  2.9644299e-01  1.1089939e+00  9.4908446e-01
   9.2712492e-01  2.2915816e+00  8.5101128e-01  1.1313956e+00
   7.6705259e-01  1.1540518e+00  9.6524382e-01  2.2568083e+00
   4.8750937e-01  2.1437917e+00  3.0048239e-01  3.1550324e-01
   3.1231153e-01  6.8028110e-01  5.9768373e-01  1.0003438e+00
   8.7378228e-01  2.6204081e+00  1.0079926e+00  1.1755056e+00
   1.7061955e+00  7.9650450e-01  6.4564538e-01  2.1298537e+00
   1.4070787e+00  2.3433404e+00  9.7109342e-01  1.6384648e+00
   1.8668360e-01  1.6924558e+00  2.1669240e+00  1.7779464e-01
   8.8792062e-01  1.1482084e+00  2.0471506e+00  2.2509136e+00
   1.7413712e+00  5.5243146e-01  5.4746109e-01  1.8004693e+00
   7.3227602e-01  3.9677328e-01  1.9426872e+00  1.0138264e+00
   2.2460904e+00  2.3253825e+00  5.8937460e-01  1.0669568e+00
   1.0410109e+00  1.2669177e+00  1.7478935e+00  4.6651173e-01
   5.9817725e-01  7.7465516e-01  3.2000482e-01  4.8173440e-01
   5.8538562e-01  1.1471599e+00  2.6764231e+00  1.0979898e+00
   1.8924837e+00  1.9255037e+00  1.4469509e+00  3.8867652e-01
   4.0927213e-01  9.0401721e-01  5.4896414e-01  1.1541725e+00
   1.4150797e+00  2.3668075e+00  1.8275456e+00  6.7218673e-01
   6.2668151e-01  1.4607282e+00  1.1821592e+00  1.7340704e+00
   1.6486592e+00  1.6655889e+00  2.4708569e-01  2.0429583e+00
   1.6197143e+00  1.0842415e+00  6.3039386e-01  1.8794385e+00
   1.0392706e+00  1.5447203e+00  1.3515465e+00  2.5528409e+00
   1.4570504e+00  4.4926262e-01  2.0821514e+00  1.8205395e+00
   5.5624247e-02  7.3178792e+00  7.2999091e+00  7.4817729e+00
   8.2924080e+00  7.5162110e+00  7.6837907e+00  7.8021646e+00
   5.6547241e+00  6.1287618e+00  7.0738449e+00  6.4624257e+00
   8.4823351e+00  6.8684015e+00  6.7796321e+00  8.2771435e+00
   7.2320023e+00  7.1022224e+00  7.3201284e+00  8.8829565e+00
   5.9290357e+00  7.3708363e+00  7.5713506e+00  7.6739163e+00
   7.4984198e+00  6.9967170e+00  6.0550642e+00  6.8558722e+00
   5.4597583e+00  7.0668550e+00  8.1006422e+00  5.9779415e+00
   7.9148145e+00  7.8047533e+00  6.4820700e+00  6.9640255e+00
   7.3946548e+00  6.3825397e+00  7.0738811e+00  6.3847942e+00
   7.5052309e+00  7.5808454e+00  5.9190307e+00  6.4251399e+00
   6.3231936e+00  8.9867601e+00  8.0475655e+00  8.0902996e+00
   7.6791096e+00  6.4082251e+00  7.3998890e+00  7.1573629e+00
   7.9102259e+00  8.2237215e+00  6.2794003e+00  6.2533922e+00
   6.2583737e+00  6.4939089e+00  7.3390727e+00  7.3645172e+00
  -6.4123931e+00 -5.9279084e+00  6.4094286e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 14:13:29.237530
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.0809
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 14:13:29.242334
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9060.49
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 14:13:29.246534
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.3574
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 14:13:29.250564
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -810.431
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140526176109568
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140524948660856
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140524948661360
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140524948661864
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140524948662368
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140524948662872

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fcee1348240> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.772424
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.734846
grad_step = 000002, loss = 0.704351
grad_step = 000003, loss = 0.674048
grad_step = 000004, loss = 0.641218
grad_step = 000005, loss = 0.607372
grad_step = 000006, loss = 0.577300
grad_step = 000007, loss = 0.557950
grad_step = 000008, loss = 0.557967
grad_step = 000009, loss = 0.547508
grad_step = 000010, loss = 0.525826
grad_step = 000011, loss = 0.506799
grad_step = 000012, loss = 0.494226
grad_step = 000013, loss = 0.485044
grad_step = 000014, loss = 0.475761
grad_step = 000015, loss = 0.464597
grad_step = 000016, loss = 0.451262
grad_step = 000017, loss = 0.436593
grad_step = 000018, loss = 0.422255
grad_step = 000019, loss = 0.410058
grad_step = 000020, loss = 0.400500
grad_step = 000021, loss = 0.391439
grad_step = 000022, loss = 0.380279
grad_step = 000023, loss = 0.367205
grad_step = 000024, loss = 0.354581
grad_step = 000025, loss = 0.343789
grad_step = 000026, loss = 0.334169
grad_step = 000027, loss = 0.324434
grad_step = 000028, loss = 0.313995
grad_step = 000029, loss = 0.303068
grad_step = 000030, loss = 0.292231
grad_step = 000031, loss = 0.281932
grad_step = 000032, loss = 0.272347
grad_step = 000033, loss = 0.263230
grad_step = 000034, loss = 0.253947
grad_step = 000035, loss = 0.244316
grad_step = 000036, loss = 0.234841
grad_step = 000037, loss = 0.225883
grad_step = 000038, loss = 0.217266
grad_step = 000039, loss = 0.208798
grad_step = 000040, loss = 0.200459
grad_step = 000041, loss = 0.192255
grad_step = 000042, loss = 0.184282
grad_step = 000043, loss = 0.175522
grad_step = 000044, loss = 0.166270
grad_step = 000045, loss = 0.157079
grad_step = 000046, loss = 0.148739
grad_step = 000047, loss = 0.141899
grad_step = 000048, loss = 0.135188
grad_step = 000049, loss = 0.128129
grad_step = 000050, loss = 0.121326
grad_step = 000051, loss = 0.115286
grad_step = 000052, loss = 0.109170
grad_step = 000053, loss = 0.102774
grad_step = 000054, loss = 0.096703
grad_step = 000055, loss = 0.091122
grad_step = 000056, loss = 0.086108
grad_step = 000057, loss = 0.080828
grad_step = 000058, loss = 0.075572
grad_step = 000059, loss = 0.070779
grad_step = 000060, loss = 0.066308
grad_step = 000061, loss = 0.062080
grad_step = 000062, loss = 0.057879
grad_step = 000063, loss = 0.053972
grad_step = 000064, loss = 0.050169
grad_step = 000065, loss = 0.046583
grad_step = 000066, loss = 0.043295
grad_step = 000067, loss = 0.040260
grad_step = 000068, loss = 0.037239
grad_step = 000069, loss = 0.034331
grad_step = 000070, loss = 0.031735
grad_step = 000071, loss = 0.029370
grad_step = 000072, loss = 0.027068
grad_step = 000073, loss = 0.024945
grad_step = 000074, loss = 0.022981
grad_step = 000075, loss = 0.021091
grad_step = 000076, loss = 0.019376
grad_step = 000077, loss = 0.017859
grad_step = 000078, loss = 0.016374
grad_step = 000079, loss = 0.014996
grad_step = 000080, loss = 0.013776
grad_step = 000081, loss = 0.012645
grad_step = 000082, loss = 0.011592
grad_step = 000083, loss = 0.010645
grad_step = 000084, loss = 0.009746
grad_step = 000085, loss = 0.008944
grad_step = 000086, loss = 0.008240
grad_step = 000087, loss = 0.007561
grad_step = 000088, loss = 0.006959
grad_step = 000089, loss = 0.006433
grad_step = 000090, loss = 0.005949
grad_step = 000091, loss = 0.005525
grad_step = 000092, loss = 0.005139
grad_step = 000093, loss = 0.004786
grad_step = 000094, loss = 0.004488
grad_step = 000095, loss = 0.004217
grad_step = 000096, loss = 0.003971
grad_step = 000097, loss = 0.003757
grad_step = 000098, loss = 0.003561
grad_step = 000099, loss = 0.003393
grad_step = 000100, loss = 0.003241
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003097
grad_step = 000102, loss = 0.002978
grad_step = 000103, loss = 0.002870
grad_step = 000104, loss = 0.002772
grad_step = 000105, loss = 0.002684
grad_step = 000106, loss = 0.002605
grad_step = 000107, loss = 0.002539
grad_step = 000108, loss = 0.002479
grad_step = 000109, loss = 0.002428
grad_step = 000110, loss = 0.002395
grad_step = 000111, loss = 0.002374
grad_step = 000112, loss = 0.002365
grad_step = 000113, loss = 0.002306
grad_step = 000114, loss = 0.002240
grad_step = 000115, loss = 0.002207
grad_step = 000116, loss = 0.002207
grad_step = 000117, loss = 0.002205
grad_step = 000118, loss = 0.002166
grad_step = 000119, loss = 0.002127
grad_step = 000120, loss = 0.002112
grad_step = 000121, loss = 0.002117
grad_step = 000122, loss = 0.002119
grad_step = 000123, loss = 0.002100
grad_step = 000124, loss = 0.002076
grad_step = 000125, loss = 0.002055
grad_step = 000126, loss = 0.002047
grad_step = 000127, loss = 0.002052
grad_step = 000128, loss = 0.002059
grad_step = 000129, loss = 0.002076
grad_step = 000130, loss = 0.002067
grad_step = 000131, loss = 0.002050
grad_step = 000132, loss = 0.002014
grad_step = 000133, loss = 0.001994
grad_step = 000134, loss = 0.001996
grad_step = 000135, loss = 0.002017
grad_step = 000136, loss = 0.002065
grad_step = 000137, loss = 0.002093
grad_step = 000138, loss = 0.002070
grad_step = 000139, loss = 0.001981
grad_step = 000140, loss = 0.001962
grad_step = 000141, loss = 0.001992
grad_step = 000142, loss = 0.002031
grad_step = 000143, loss = 0.001986
grad_step = 000144, loss = 0.001940
grad_step = 000145, loss = 0.001940
grad_step = 000146, loss = 0.001964
grad_step = 000147, loss = 0.001991
grad_step = 000148, loss = 0.001984
grad_step = 000149, loss = 0.001944
grad_step = 000150, loss = 0.001910
grad_step = 000151, loss = 0.001919
grad_step = 000152, loss = 0.001949
grad_step = 000153, loss = 0.001947
grad_step = 000154, loss = 0.001918
grad_step = 000155, loss = 0.001895
grad_step = 000156, loss = 0.001897
grad_step = 000157, loss = 0.001909
grad_step = 000158, loss = 0.001921
grad_step = 000159, loss = 0.001914
grad_step = 000160, loss = 0.001895
grad_step = 000161, loss = 0.001883
grad_step = 000162, loss = 0.001888
grad_step = 000163, loss = 0.001904
grad_step = 000164, loss = 0.001906
grad_step = 000165, loss = 0.001892
grad_step = 000166, loss = 0.001873
grad_step = 000167, loss = 0.001874
grad_step = 000168, loss = 0.001882
grad_step = 000169, loss = 0.001877
grad_step = 000170, loss = 0.001866
grad_step = 000171, loss = 0.001858
grad_step = 000172, loss = 0.001861
grad_step = 000173, loss = 0.001864
grad_step = 000174, loss = 0.001859
grad_step = 000175, loss = 0.001854
grad_step = 000176, loss = 0.001853
grad_step = 000177, loss = 0.001856
grad_step = 000178, loss = 0.001852
grad_step = 000179, loss = 0.001846
grad_step = 000180, loss = 0.001843
grad_step = 000181, loss = 0.001843
grad_step = 000182, loss = 0.001841
grad_step = 000183, loss = 0.001836
grad_step = 000184, loss = 0.001833
grad_step = 000185, loss = 0.001832
grad_step = 000186, loss = 0.001831
grad_step = 000187, loss = 0.001829
grad_step = 000188, loss = 0.001825
grad_step = 000189, loss = 0.001822
grad_step = 000190, loss = 0.001822
grad_step = 000191, loss = 0.001821
grad_step = 000192, loss = 0.001820
grad_step = 000193, loss = 0.001818
grad_step = 000194, loss = 0.001816
grad_step = 000195, loss = 0.001816
grad_step = 000196, loss = 0.001819
grad_step = 000197, loss = 0.001831
grad_step = 000198, loss = 0.001859
grad_step = 000199, loss = 0.001920
grad_step = 000200, loss = 0.002053
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002274
grad_step = 000202, loss = 0.002415
grad_step = 000203, loss = 0.002204
grad_step = 000204, loss = 0.001856
grad_step = 000205, loss = 0.001882
grad_step = 000206, loss = 0.002130
grad_step = 000207, loss = 0.002047
grad_step = 000208, loss = 0.001819
grad_step = 000209, loss = 0.001880
grad_step = 000210, loss = 0.002009
grad_step = 000211, loss = 0.001880
grad_step = 000212, loss = 0.001787
grad_step = 000213, loss = 0.001889
grad_step = 000214, loss = 0.001898
grad_step = 000215, loss = 0.001793
grad_step = 000216, loss = 0.001822
grad_step = 000217, loss = 0.001874
grad_step = 000218, loss = 0.001798
grad_step = 000219, loss = 0.001776
grad_step = 000220, loss = 0.001830
grad_step = 000221, loss = 0.001810
grad_step = 000222, loss = 0.001766
grad_step = 000223, loss = 0.001793
grad_step = 000224, loss = 0.001802
grad_step = 000225, loss = 0.001762
grad_step = 000226, loss = 0.001760
grad_step = 000227, loss = 0.001783
grad_step = 000228, loss = 0.001766
grad_step = 000229, loss = 0.001747
grad_step = 000230, loss = 0.001764
grad_step = 000231, loss = 0.001765
grad_step = 000232, loss = 0.001746
grad_step = 000233, loss = 0.001746
grad_step = 000234, loss = 0.001756
grad_step = 000235, loss = 0.001747
grad_step = 000236, loss = 0.001736
grad_step = 000237, loss = 0.001741
grad_step = 000238, loss = 0.001744
grad_step = 000239, loss = 0.001732
grad_step = 000240, loss = 0.001728
grad_step = 000241, loss = 0.001732
grad_step = 000242, loss = 0.001731
grad_step = 000243, loss = 0.001725
grad_step = 000244, loss = 0.001729
grad_step = 000245, loss = 0.001736
grad_step = 000246, loss = 0.001747
grad_step = 000247, loss = 0.001759
grad_step = 000248, loss = 0.001803
grad_step = 000249, loss = 0.001831
grad_step = 000250, loss = 0.001882
grad_step = 000251, loss = 0.001831
grad_step = 000252, loss = 0.001765
grad_step = 000253, loss = 0.001702
grad_step = 000254, loss = 0.001735
grad_step = 000255, loss = 0.001799
grad_step = 000256, loss = 0.001756
grad_step = 000257, loss = 0.001700
grad_step = 000258, loss = 0.001698
grad_step = 000259, loss = 0.001741
grad_step = 000260, loss = 0.001744
grad_step = 000261, loss = 0.001687
grad_step = 000262, loss = 0.001683
grad_step = 000263, loss = 0.001718
grad_step = 000264, loss = 0.001700
grad_step = 000265, loss = 0.001670
grad_step = 000266, loss = 0.001671
grad_step = 000267, loss = 0.001686
grad_step = 000268, loss = 0.001684
grad_step = 000269, loss = 0.001661
grad_step = 000270, loss = 0.001657
grad_step = 000271, loss = 0.001667
grad_step = 000272, loss = 0.001663
grad_step = 000273, loss = 0.001651
grad_step = 000274, loss = 0.001645
grad_step = 000275, loss = 0.001646
grad_step = 000276, loss = 0.001646
grad_step = 000277, loss = 0.001640
grad_step = 000278, loss = 0.001636
grad_step = 000279, loss = 0.001634
grad_step = 000280, loss = 0.001630
grad_step = 000281, loss = 0.001626
grad_step = 000282, loss = 0.001623
grad_step = 000283, loss = 0.001622
grad_step = 000284, loss = 0.001623
grad_step = 000285, loss = 0.001620
grad_step = 000286, loss = 0.001613
grad_step = 000287, loss = 0.001607
grad_step = 000288, loss = 0.001603
grad_step = 000289, loss = 0.001602
grad_step = 000290, loss = 0.001598
grad_step = 000291, loss = 0.001594
grad_step = 000292, loss = 0.001591
grad_step = 000293, loss = 0.001591
grad_step = 000294, loss = 0.001596
grad_step = 000295, loss = 0.001603
grad_step = 000296, loss = 0.001622
grad_step = 000297, loss = 0.001655
grad_step = 000298, loss = 0.001764
grad_step = 000299, loss = 0.001913
grad_step = 000300, loss = 0.002205
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.002087
grad_step = 000302, loss = 0.001874
grad_step = 000303, loss = 0.001679
grad_step = 000304, loss = 0.001624
grad_step = 000305, loss = 0.001706
grad_step = 000306, loss = 0.001803
grad_step = 000307, loss = 0.001747
grad_step = 000308, loss = 0.001589
grad_step = 000309, loss = 0.001590
grad_step = 000310, loss = 0.001707
grad_step = 000311, loss = 0.001709
grad_step = 000312, loss = 0.001579
grad_step = 000313, loss = 0.001566
grad_step = 000314, loss = 0.001642
grad_step = 000315, loss = 0.001632
grad_step = 000316, loss = 0.001571
grad_step = 000317, loss = 0.001576
grad_step = 000318, loss = 0.001600
grad_step = 000319, loss = 0.001565
grad_step = 000320, loss = 0.001532
grad_step = 000321, loss = 0.001561
grad_step = 000322, loss = 0.001601
grad_step = 000323, loss = 0.001573
grad_step = 000324, loss = 0.001548
grad_step = 000325, loss = 0.001557
grad_step = 000326, loss = 0.001553
grad_step = 000327, loss = 0.001521
grad_step = 000328, loss = 0.001507
grad_step = 000329, loss = 0.001527
grad_step = 000330, loss = 0.001537
grad_step = 000331, loss = 0.001527
grad_step = 000332, loss = 0.001526
grad_step = 000333, loss = 0.001541
grad_step = 000334, loss = 0.001541
grad_step = 000335, loss = 0.001532
grad_step = 000336, loss = 0.001526
grad_step = 000337, loss = 0.001530
grad_step = 000338, loss = 0.001523
grad_step = 000339, loss = 0.001508
grad_step = 000340, loss = 0.001498
grad_step = 000341, loss = 0.001496
grad_step = 000342, loss = 0.001488
grad_step = 000343, loss = 0.001478
grad_step = 000344, loss = 0.001476
grad_step = 000345, loss = 0.001478
grad_step = 000346, loss = 0.001480
grad_step = 000347, loss = 0.001475
grad_step = 000348, loss = 0.001472
grad_step = 000349, loss = 0.001474
grad_step = 000350, loss = 0.001483
grad_step = 000351, loss = 0.001492
grad_step = 000352, loss = 0.001507
grad_step = 000353, loss = 0.001532
grad_step = 000354, loss = 0.001611
grad_step = 000355, loss = 0.001697
grad_step = 000356, loss = 0.001859
grad_step = 000357, loss = 0.001826
grad_step = 000358, loss = 0.001752
grad_step = 000359, loss = 0.001588
grad_step = 000360, loss = 0.001484
grad_step = 000361, loss = 0.001499
grad_step = 000362, loss = 0.001585
grad_step = 000363, loss = 0.001617
grad_step = 000364, loss = 0.001529
grad_step = 000365, loss = 0.001446
grad_step = 000366, loss = 0.001490
grad_step = 000367, loss = 0.001549
grad_step = 000368, loss = 0.001520
grad_step = 000369, loss = 0.001458
grad_step = 000370, loss = 0.001451
grad_step = 000371, loss = 0.001477
grad_step = 000372, loss = 0.001487
grad_step = 000373, loss = 0.001480
grad_step = 000374, loss = 0.001460
grad_step = 000375, loss = 0.001442
grad_step = 000376, loss = 0.001431
grad_step = 000377, loss = 0.001444
grad_step = 000378, loss = 0.001468
grad_step = 000379, loss = 0.001461
grad_step = 000380, loss = 0.001434
grad_step = 000381, loss = 0.001414
grad_step = 000382, loss = 0.001418
grad_step = 000383, loss = 0.001429
grad_step = 000384, loss = 0.001428
grad_step = 000385, loss = 0.001423
grad_step = 000386, loss = 0.001420
grad_step = 000387, loss = 0.001413
grad_step = 000388, loss = 0.001402
grad_step = 000389, loss = 0.001398
grad_step = 000390, loss = 0.001403
grad_step = 000391, loss = 0.001407
grad_step = 000392, loss = 0.001406
grad_step = 000393, loss = 0.001405
grad_step = 000394, loss = 0.001403
grad_step = 000395, loss = 0.001399
grad_step = 000396, loss = 0.001392
grad_step = 000397, loss = 0.001387
grad_step = 000398, loss = 0.001383
grad_step = 000399, loss = 0.001382
grad_step = 000400, loss = 0.001380
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001379
grad_step = 000402, loss = 0.001379
grad_step = 000403, loss = 0.001380
grad_step = 000404, loss = 0.001383
grad_step = 000405, loss = 0.001386
grad_step = 000406, loss = 0.001390
grad_step = 000407, loss = 0.001398
grad_step = 000408, loss = 0.001411
grad_step = 000409, loss = 0.001442
grad_step = 000410, loss = 0.001468
grad_step = 000411, loss = 0.001523
grad_step = 000412, loss = 0.001550
grad_step = 000413, loss = 0.001603
grad_step = 000414, loss = 0.001591
grad_step = 000415, loss = 0.001562
grad_step = 000416, loss = 0.001467
grad_step = 000417, loss = 0.001381
grad_step = 000418, loss = 0.001354
grad_step = 000419, loss = 0.001388
grad_step = 000420, loss = 0.001435
grad_step = 000421, loss = 0.001445
grad_step = 000422, loss = 0.001411
grad_step = 000423, loss = 0.001363
grad_step = 000424, loss = 0.001342
grad_step = 000425, loss = 0.001356
grad_step = 000426, loss = 0.001382
grad_step = 000427, loss = 0.001395
grad_step = 000428, loss = 0.001388
grad_step = 000429, loss = 0.001367
grad_step = 000430, loss = 0.001346
grad_step = 000431, loss = 0.001333
grad_step = 000432, loss = 0.001332
grad_step = 000433, loss = 0.001341
grad_step = 000434, loss = 0.001351
grad_step = 000435, loss = 0.001357
grad_step = 000436, loss = 0.001354
grad_step = 000437, loss = 0.001344
grad_step = 000438, loss = 0.001332
grad_step = 000439, loss = 0.001322
grad_step = 000440, loss = 0.001317
grad_step = 000441, loss = 0.001315
grad_step = 000442, loss = 0.001317
grad_step = 000443, loss = 0.001320
grad_step = 000444, loss = 0.001325
grad_step = 000445, loss = 0.001330
grad_step = 000446, loss = 0.001333
grad_step = 000447, loss = 0.001336
grad_step = 000448, loss = 0.001338
grad_step = 000449, loss = 0.001343
grad_step = 000450, loss = 0.001348
grad_step = 000451, loss = 0.001358
grad_step = 000452, loss = 0.001358
grad_step = 000453, loss = 0.001360
grad_step = 000454, loss = 0.001352
grad_step = 000455, loss = 0.001347
grad_step = 000456, loss = 0.001335
grad_step = 000457, loss = 0.001324
grad_step = 000458, loss = 0.001311
grad_step = 000459, loss = 0.001301
grad_step = 000460, loss = 0.001292
grad_step = 000461, loss = 0.001287
grad_step = 000462, loss = 0.001285
grad_step = 000463, loss = 0.001284
grad_step = 000464, loss = 0.001281
grad_step = 000465, loss = 0.001277
grad_step = 000466, loss = 0.001274
grad_step = 000467, loss = 0.001272
grad_step = 000468, loss = 0.001272
grad_step = 000469, loss = 0.001272
grad_step = 000470, loss = 0.001271
grad_step = 000471, loss = 0.001269
grad_step = 000472, loss = 0.001269
grad_step = 000473, loss = 0.001270
grad_step = 000474, loss = 0.001277
grad_step = 000475, loss = 0.001290
grad_step = 000476, loss = 0.001321
grad_step = 000477, loss = 0.001382
grad_step = 000478, loss = 0.001539
grad_step = 000479, loss = 0.001743
grad_step = 000480, loss = 0.002121
grad_step = 000481, loss = 0.002082
grad_step = 000482, loss = 0.001814
grad_step = 000483, loss = 0.001414
grad_step = 000484, loss = 0.001346
grad_step = 000485, loss = 0.001553
grad_step = 000486, loss = 0.001610
grad_step = 000487, loss = 0.001480
grad_step = 000488, loss = 0.001345
grad_step = 000489, loss = 0.001360
grad_step = 000490, loss = 0.001456
grad_step = 000491, loss = 0.001431
grad_step = 000492, loss = 0.001310
grad_step = 000493, loss = 0.001295
grad_step = 000494, loss = 0.001386
grad_step = 000495, loss = 0.001388
grad_step = 000496, loss = 0.001288
grad_step = 000497, loss = 0.001253
grad_step = 000498, loss = 0.001326
grad_step = 000499, loss = 0.001350
grad_step = 000500, loss = 0.001276
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001229
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

  date_run                              2020-05-15 14:13:54.212133
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.259286
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 14:13:54.218363
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.168688
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 14:13:54.224758
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.152345
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 14:13:54.230753
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.56328
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
0   2020-05-15 14:13:20.022116  ...    mean_absolute_error
1   2020-05-15 14:13:20.027208  ...     mean_squared_error
2   2020-05-15 14:13:20.030918  ...  median_absolute_error
3   2020-05-15 14:13:20.034838  ...               r2_score
4   2020-05-15 14:13:29.237530  ...    mean_absolute_error
5   2020-05-15 14:13:29.242334  ...     mean_squared_error
6   2020-05-15 14:13:29.246534  ...  median_absolute_error
7   2020-05-15 14:13:29.250564  ...               r2_score
8   2020-05-15 14:13:54.212133  ...    mean_absolute_error
9   2020-05-15 14:13:54.218363  ...     mean_squared_error
10  2020-05-15 14:13:54.224758  ...  median_absolute_error
11  2020-05-15 14:13:54.230753  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5499535be0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 42%|████▏     | 4202496/9912422 [00:00<00:00, 41917066.85it/s]9920512it [00:00, 32240916.46it/s]                             
0it [00:00, ?it/s]32768it [00:00, 717805.59it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 484995.35it/s]1654784it [00:00, 12210670.38it/s]                         
0it [00:00, ?it/s]8192it [00:00, 229300.34it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f544beeee80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f544b5200b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f544beeee80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5499540ba8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5448cb04e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5499540ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f544beeee80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5499540ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f5448cb04e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f54994f8f28> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fa2883c51d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=218a5bd5bedae19b8ef36b45177af60ab64be044d3b8608d0a25d94a642b8dcd
  Stored in directory: /tmp/pip-ephem-wheel-cache-ia2biw_2/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fa27e74b048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1761280/17464789 [==>...........................] - ETA: 0s
 7487488/17464789 [===========>..................] - ETA: 0s
13238272/17464789 [=====================>........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 14:15:23.452267: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 14:15:23.456667: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-15 14:15:23.456879: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564fd5969030 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 14:15:23.456894: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6206 - accuracy: 0.5030
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6666 - accuracy: 0.5000
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5797 - accuracy: 0.5057 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.5363 - accuracy: 0.5085
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5133 - accuracy: 0.5100
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5848 - accuracy: 0.5053
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6294 - accuracy: 0.5024
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6015 - accuracy: 0.5042
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6274 - accuracy: 0.5026
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6252 - accuracy: 0.5027
11000/25000 [============>.................] - ETA: 4s - loss: 7.6150 - accuracy: 0.5034
12000/25000 [=============>................] - ETA: 4s - loss: 7.6091 - accuracy: 0.5038
13000/25000 [==============>...............] - ETA: 4s - loss: 7.5994 - accuracy: 0.5044
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5998 - accuracy: 0.5044
15000/25000 [=================>............] - ETA: 3s - loss: 7.5869 - accuracy: 0.5052
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6168 - accuracy: 0.5033
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6143 - accuracy: 0.5034
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6096 - accuracy: 0.5037
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6327 - accuracy: 0.5022
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6628 - accuracy: 0.5002
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6659 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6806 - accuracy: 0.4991
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6566 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 10s 401us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 14:15:40.954781
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 14:15:40.954781  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:51:37, 10.5kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:14:14, 14.7kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:01<11:25:22, 21.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 868k/862M [00:01<8:00:17, 29.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.44M/862M [00:01<5:35:23, 42.7kB/s].vector_cache/glove.6B.zip:   1%|          | 6.79M/862M [00:01<3:53:58, 60.9kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.5M/862M [00:01<2:42:59, 87.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.7M/862M [00:01<1:53:37, 124kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.3M/862M [00:01<1:19:11, 177kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.2M/862M [00:01<55:17, 253kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 28.7M/862M [00:01<38:35, 360kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.9M/862M [00:02<26:59, 512kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.4M/862M [00:02<18:52, 728kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.6M/862M [00:02<13:15, 1.03MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.2M/862M [00:02<09:18, 1.46MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.2M/862M [00:02<06:35, 2.05MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<05:10, 2.61MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<05:31, 2.43MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<06:04, 2.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:04<04:44, 2.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:06<05:39, 2.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<05:43, 2.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.5M/862M [00:06<04:21, 3.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:06<03:13, 4.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<21:37, 615kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<16:43, 795kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:08<12:05, 1.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<11:11, 1.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<09:14, 1.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<06:46, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:12<07:42, 1.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:12<08:05, 1.63MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<06:20, 2.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:12<04:33, 2.87MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<12:39:02, 17.3kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<8:52:26, 24.6kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:14<6:12:15, 35.1kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<4:22:55, 49.5kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<3:06:40, 69.8kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.6M/862M [00:16<2:11:12, 99.2kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<1:33:35, 138kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<1:06:48, 194kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:18<47:00, 275kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<35:50, 360kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<27:44, 465kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<19:57, 645kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.0M/862M [00:20<14:06, 910kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<16:30, 776kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<12:51, 996kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<09:19, 1.37MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.0M/862M [00:24<09:28, 1.35MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<07:54, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:24<05:48, 2.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<07:04, 1.79MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:15, 2.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:42, 2.69MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:16, 2.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:58, 1.81MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:26, 2.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<03:57, 3.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<10:54, 1.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<08:54, 1.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:30, 1.92MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:28, 1.67MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:30, 1.92MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<04:49, 2.58MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:19, 1.96MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:41, 2.18MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:17, 2.88MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:55, 2.08MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:24, 2.28MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:05, 3.01MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:45, 2.13MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:17, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:00, 3.05MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:41, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:29, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:11, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:34, 2.17MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<05:09, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<03:55, 3.09MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<05:33, 2.17MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:08, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<03:50, 3.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<05:32, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:06, 2.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<03:52, 3.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:32, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<05:05, 2.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<03:51, 3.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<05:30, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:04, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:47, 3.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:27, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:15, 1.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<04:59, 2.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:22, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<04:59, 2.35MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<03:45, 3.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:20, 2.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:08, 1.90MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<04:47, 2.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:30, 3.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<07:48, 1.48MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:40, 1.73MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<04:57, 2.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<06:09, 1.87MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<06:40, 1.72MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:11, 2.21MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<03:45, 3.05MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<14:52, 770kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<11:35, 987kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<08:23, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<08:31, 1.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:07, 1.60MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<05:13, 2.17MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<06:19, 1.79MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:23, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<04:00, 2.82MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<02:57, 3.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<44:31, 253kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<33:36, 335kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<24:05, 466kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<16:55, 660kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<1:31:18, 122kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<1:04:51, 172kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<45:32, 245kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<31:58, 347kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<49:53, 223kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<36:02, 308kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<25:27, 435kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<20:22, 542kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<15:23, 717kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<10:59, 1.00MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<10:15, 1.07MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<08:18, 1.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<06:05, 1.80MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<04:26, 2.45MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<13:42, 795kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<11:23, 957kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<08:24, 1.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<06:02, 1.79MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<11:26, 947kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<12:51, 843kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<10:05, 1.07MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<07:22, 1.47MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<05:24, 1.99MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<14:04, 765kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<14:41, 733kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<11:29, 936kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:19, 1.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:21<06:00, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<18:30, 578kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<17:45, 602kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<13:37, 784kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<09:47, 1.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<07:01, 1.51MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<12:20, 861kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<13:01, 816kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<10:15, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<07:27, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<05:22, 1.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<24:38, 428kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<21:34, 489kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<16:11, 652kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<11:35, 907kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<08:17, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<51:44, 203kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<40:31, 259kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<29:14, 358kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<20:53, 501kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<14:45, 707kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:29<10:35, 983kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<1:56:27, 89.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<1:25:47, 121kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<1:00:55, 171kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<42:47, 243kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<30:11, 343kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<24:55, 415kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<21:41, 477kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<16:15, 636kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<11:35, 890kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<08:15, 1.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<20:56, 491kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<18:36, 553kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<13:52, 740kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<10:02, 1.02MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<07:12, 1.42MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<09:19, 1.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<10:23, 982kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<08:07, 1.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<06:10, 1.65MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<04:28, 2.27MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<07:16, 1.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<09:15, 1.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<07:19, 1.38MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:49, 1.74MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<04:13, 2.39MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<08:01, 1.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<08:55, 1.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<06:56, 1.45MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:12, 1.93MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<03:52, 2.58MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<08:19, 1.20MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<08:56, 1.12MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<07:00, 1.43MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<05:08, 1.94MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:48, 2.61MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<08:30, 1.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<12:06, 820kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<09:54, 1.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<07:18, 1.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<05:22, 1.84MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<07:03, 1.40MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<11:00, 896kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<09:10, 1.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<06:47, 1.45MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<04:58, 1.97MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<06:53, 1.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<10:12, 960kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<08:28, 1.15MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<06:17, 1.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:39, 2.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<06:48, 1.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<10:06, 962kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<08:24, 1.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<06:15, 1.55MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<04:37, 2.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<06:46, 1.43MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<07:14, 1.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:43, 1.69MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:13, 2.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<03:08, 3.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<11:26, 838kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<13:14, 724kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<10:35, 905kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<07:42, 1.24MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<05:37, 1.70MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<07:30, 1.27MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<07:51, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<06:07, 1.55MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:32, 2.09MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<03:24, 2.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<14:50, 637kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<15:34, 607kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<12:11, 775kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<08:52, 1.06MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<06:24, 1.47MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<07:47, 1.20MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<10:34, 887kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<08:43, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:23, 1.46MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<04:42, 1.98MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:31, 2.64MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<13:43, 678kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<14:43, 633kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<11:37, 801kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:25, 1.10MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<06:08, 1.51MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:30, 2.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<13:20, 692kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<11:52, 778kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<08:48, 1.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:24, 1.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<04:45, 1.93MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<07:35, 1.21MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<10:22, 883kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<08:33, 1.07MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<06:20, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<04:37, 1.97MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<06:41, 1.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<09:38, 943kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<07:49, 1.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:48, 1.56MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:14, 2.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:17, 2.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<12:43, 710kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<13:52, 651kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<10:59, 822kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<08:00, 1.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<05:49, 1.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<07:33, 1.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<07:37, 1.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<05:51, 1.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<04:18, 2.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:15, 2.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<06:10, 1.44MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<08:42, 1.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<07:14, 1.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:23, 1.65MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<03:57, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<06:32, 1.35MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<09:24, 939kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<07:43, 1.14MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<05:43, 1.54MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<04:12, 2.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:44, 1.30MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:54, 1.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<05:22, 1.63MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<03:57, 2.21MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<02:57, 2.94MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<48:23, 180kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<38:08, 228kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<27:45, 313kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<19:38, 441kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<13:54, 621kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<14:08, 610kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<13:45, 627kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<10:37, 810kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<07:40, 1.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:33, 1.54MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<09:35, 892kB/s] .vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<10:34, 809kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<08:21, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<06:06, 1.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<04:25, 1.92MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<09:36, 882kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<10:33, 803kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<08:17, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<06:01, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<04:20, 1.94MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<09:38, 873kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<10:12, 825kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<07:52, 1.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<05:45, 1.45MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<04:08, 2.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<11:50, 705kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<11:42, 713kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<08:52, 939kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<06:26, 1.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:30<04:36, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<24:49, 333kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<20:33, 402kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<15:02, 550kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<10:41, 771kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:32<07:39, 1.07MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<09:02, 908kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<09:12, 890kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<07:09, 1.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<05:11, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<05:39, 1.44MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<06:37, 1.23MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<05:11, 1.57MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<03:47, 2.14MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:36<02:49, 2.86MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<07:33, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<07:56, 1.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<06:14, 1.29MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:31, 1.78MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:08, 1.55MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:55, 1.35MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<04:43, 1.69MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<03:28, 2.29MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:46, 1.66MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<05:30, 1.44MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:20, 1.82MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:42<03:12, 2.46MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:42<02:21, 3.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<45:45, 172kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<34:05, 230kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<24:21, 322kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<17:06, 457kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<14:56, 521kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<12:28, 624kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<09:07, 853kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<06:28, 1.20MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<07:17, 1.06MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<07:00, 1.10MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<05:23, 1.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<03:53, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<05:43, 1.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<06:14, 1.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:55, 1.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<03:34, 2.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<05:05, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<05:31, 1.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<04:18, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<03:06, 2.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:39, 1.61MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:25, 1.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<04:15, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:11, 2.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:58, 1.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<06:51, 1.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:49, 1.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:19, 1.72MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:20, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<04:22, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<03:24, 2.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<02:28, 2.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<20:42, 353kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<15:49, 461kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<11:23, 640kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<07:59, 905kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<1:29:26, 80.9kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<1:03:50, 113kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<44:51, 161kB/s]  .vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<31:19, 229kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<26:50, 267kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<20:23, 351kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<14:38, 488kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<10:17, 691kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<15:26, 460kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<12:19, 576kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<08:59, 788kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<06:20, 1.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<16:57, 415kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<13:26, 523kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<09:43, 721kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<06:51, 1.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<08:37, 807kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<07:32, 922kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<05:38, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<04:01, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<18:29, 373kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<14:24, 478kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<10:26, 659kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<07:20, 930kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<16:03, 425kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<12:13, 558kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<08:44, 778kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<07:21, 919kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<06:28, 1.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<04:51, 1.39MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<04:28, 1.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<04:26, 1.50MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:26, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<03:28, 1.90MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<03:44, 1.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:55, 2.25MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<03:06, 2.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<03:28, 1.89MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<02:44, 2.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<02:58, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<03:18, 1.96MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<02:35, 2.49MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<01:52, 3.42MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<06:06, 1.05MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<05:31, 1.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:10, 1.53MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<03:56, 1.61MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<04:00, 1.58MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<03:03, 2.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<02:13, 2.83MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<04:40, 1.34MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<04:35, 1.36MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<03:31, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:27, 1.80MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<03:38, 1.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:50, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<02:59, 2.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:15, 1.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<02:34, 2.38MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:47, 2.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<03:08, 1.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<02:29, 2.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<02:43, 2.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<03:04, 1.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<02:26, 2.45MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:40, 2.22MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 507M/862M [03:39<03:02, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<02:24, 2.45MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:38, 2.22MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:51, 2.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:37, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<01:57, 2.97MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:37, 2.21MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:55, 1.98MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<02:17, 2.53MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<01:40, 3.44MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:19, 1.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:10, 1.37MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:11, 1.79MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:08, 1.80MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:18, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:35, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:43, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:57, 1.89MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:17, 2.43MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<01:40, 3.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<04:08, 1.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:58, 1.39MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:02, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:00, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:08, 1.74MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:27, 2.21MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:35, 2.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:52, 1.87MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:14, 2.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<01:37, 3.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<04:39, 1.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<04:18, 1.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:13, 1.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:19, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 547M/862M [03:59<03:39, 1.44MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:35, 1.46MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:45, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:45, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:00, 1.72MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:21, 2.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:28, 2.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<02:41, 1.90MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:07, 2.40MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:18, 2.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:36, 1.93MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:01, 2.48MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:29, 3.36MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:10, 1.57MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<03:11, 1.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:26, 2.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:30, 1.95MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:43, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<02:05, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<01:31, 3.17MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<03:11, 1.52MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<03:10, 1.52MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:27, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:29, 1.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:40, 1.78MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:06, 2.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<02:13, 2.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:29, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<01:56, 2.41MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:23, 3.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<10:03, 460kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<07:55, 584kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<05:43, 806kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<04:01, 1.13MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<05:28, 831kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<04:41, 970kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:29, 1.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<03:09, 1.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<03:05, 1.45MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:22, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<02:22, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<02:33, 1.73MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<01:59, 2.20MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:05, 2.08MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:19, 1.87MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<01:48, 2.40MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:19, 3.26MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:41, 1.60MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:41, 1.59MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:05, 2.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:08, 1.97MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<02:19, 1.81MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<01:49, 2.29MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:56, 2.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<02:10, 1.90MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<01:41, 2.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:13, 3.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:53, 1.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:49, 1.44MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:10, 1.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:09, 1.86MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:18, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:48, 2.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:53, 2.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:06, 1.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:38, 2.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:46, 2.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:01, 1.92MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:35, 2.41MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:43, 2.20MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:57, 1.94MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:31, 2.47MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:06, 3.38MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<03:07, 1.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:55, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:11, 1.70MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:34, 2.34MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:36, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<02:31, 1.45MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:56, 1.88MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:55, 1.86MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<02:03, 1.75MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:36, 2.22MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:41, 2.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<01:52, 1.88MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:28, 2.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:35, 2.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:48, 1.92MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:25, 2.42MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:32, 2.20MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:44, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:21, 2.50MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:00, 3.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:46, 1.86MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:54, 1.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:27, 2.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:03, 3.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:51, 1.14MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:38, 1.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<01:59, 1.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:53, 1.68MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:57, 1.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:31, 2.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:33, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:42, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:20, 2.31MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<01:25, 2.14MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<01:35, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:14, 2.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<00:53, 3.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<02:58, 1.00MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:38, 1.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:59, 1.49MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<01:50, 1.58MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:51, 1.56MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:25, 2.04MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:01, 2.78MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:52, 1.52MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:52, 1.52MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:26, 1.96MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:26, 1.92MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:32, 1.80MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:12, 2.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:16, 2.12MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:25, 1.90MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:07, 2.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:12, 2.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:20, 1.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:02, 2.51MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<00:44, 3.44MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<03:16, 783kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<02:46, 923kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:02, 1.25MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:16<01:26, 1.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:38, 945kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<02:20, 1.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:45, 1.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:36, 1.52MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:34, 1.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:13, 1.98MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:13, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:19, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:01, 2.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:43, 3.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<07:37, 301kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<05:46, 396kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<04:07, 551kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<03:11, 697kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<02:40, 832kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:57, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:41, 1.27MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:36, 1.34MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:13, 1.74MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:10, 1.77MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:13, 1.71MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:56, 2.19MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:40, 3.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<39:46, 50.8kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<28:11, 71.5kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<19:42, 102kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<13:45, 142kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<09:59, 195kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<07:02, 275kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<04:53, 389kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<04:10, 451kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<03:50, 490kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<02:53, 649kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:02, 904kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:26, 1.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<04:39, 390kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<04:04, 445kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<03:01, 595kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<02:08, 834kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:29, 1.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<03:45, 464kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<03:20, 521kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<02:28, 700kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<01:44, 981kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:15, 1.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:35, 1.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:44, 957kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:21, 1.23MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:58, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:42, 2.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:14, 1.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:26, 1.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:08, 1.39MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:49, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:35, 2.60MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<02:25, 634kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<02:14, 687kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:40, 910kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<01:11, 1.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:51, 1.74MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:20, 1.10MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:27, 1.01MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:08, 1.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:49, 1.74MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:52, 1.58MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<01:06, 1.26MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:52, 1.59MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:38, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:44, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:59, 1.34MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:48, 1.64MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:35, 2.21MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:41, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:54, 1.39MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:43, 1.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:31, 2.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:40, 1.76MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:51, 1.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:41, 1.69MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:30, 2.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:37, 1.80MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:47, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:38, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:27, 2.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<00:20, 3.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:53, 1.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:58, 1.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:45, 1.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:32, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:37, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:45, 1.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:36, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<00:25, 2.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:31, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:39, 1.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:32, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:23, 2.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:28, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:36, 1.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:28, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:20, 2.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<00:14, 3.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:43, 1.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:44, 1.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:34, 1.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:24, 1.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:26, 1.57MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:32, 1.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:25, 1.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:18, 2.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:21, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:26, 1.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:21, 1.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:15, 2.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:19, 1.76MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:23, 1.42MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:19, 1.74MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:13, 2.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:16, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:21, 1.39MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:17, 1.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:11, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:14, 1.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:18, 1.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:14, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:10, 2.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:11, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:15, 1.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:12, 1.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:05, 3.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:16, 1.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:16, 1.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:12, 1.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:08, 1.83MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:05, 2.49MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:13, 981kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:13, 960kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:10, 1.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.69MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:06, 1.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:07, 1.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:03, 2.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 1.69MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:03, 1.35MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.71MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 2.33MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:00, 3.10MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.01MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 955kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 1.22MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 703/400000 [00:00<00:56, 7027.63it/s]  0%|          | 1458/400000 [00:00<00:55, 7175.25it/s]  1%|          | 2206/400000 [00:00<00:54, 7262.47it/s]  1%|          | 2955/400000 [00:00<00:54, 7328.56it/s]  1%|          | 3723/400000 [00:00<00:53, 7430.30it/s]  1%|          | 4491/400000 [00:00<00:52, 7502.88it/s]  1%|▏         | 5259/400000 [00:00<00:52, 7554.88it/s]  2%|▏         | 6028/400000 [00:00<00:51, 7592.50it/s]  2%|▏         | 6748/400000 [00:00<00:53, 7378.95it/s]  2%|▏         | 7469/400000 [00:01<00:53, 7325.49it/s]  2%|▏         | 8226/400000 [00:01<00:52, 7395.06it/s]  2%|▏         | 8954/400000 [00:01<00:53, 7358.85it/s]  2%|▏         | 9707/400000 [00:01<00:52, 7408.44it/s]  3%|▎         | 10468/400000 [00:01<00:52, 7465.65it/s]  3%|▎         | 11222/400000 [00:01<00:51, 7485.13it/s]  3%|▎         | 11975/400000 [00:01<00:51, 7496.44it/s]  3%|▎         | 12723/400000 [00:01<00:51, 7478.93it/s]  3%|▎         | 13470/400000 [00:01<00:51, 7465.61it/s]  4%|▎         | 14216/400000 [00:01<00:52, 7391.21it/s]  4%|▎         | 14971/400000 [00:02<00:51, 7436.73it/s]  4%|▍         | 15746/400000 [00:02<00:51, 7527.46it/s]  4%|▍         | 16500/400000 [00:02<00:50, 7529.59it/s]  4%|▍         | 17274/400000 [00:02<00:50, 7590.12it/s]  5%|▍         | 18036/400000 [00:02<00:50, 7598.19it/s]  5%|▍         | 18796/400000 [00:02<00:50, 7576.95it/s]  5%|▍         | 19565/400000 [00:02<00:49, 7609.91it/s]  5%|▌         | 20334/400000 [00:02<00:49, 7632.88it/s]  5%|▌         | 21107/400000 [00:02<00:49, 7659.40it/s]  5%|▌         | 21874/400000 [00:02<00:49, 7570.34it/s]  6%|▌         | 22632/400000 [00:03<00:50, 7507.99it/s]  6%|▌         | 23384/400000 [00:03<00:50, 7486.92it/s]  6%|▌         | 24136/400000 [00:03<00:50, 7494.98it/s]  6%|▌         | 24886/400000 [00:03<00:50, 7375.87it/s]  6%|▋         | 25645/400000 [00:03<00:50, 7436.27it/s]  7%|▋         | 26392/400000 [00:03<00:50, 7443.94it/s]  7%|▋         | 27172/400000 [00:03<00:49, 7546.57it/s]  7%|▋         | 27950/400000 [00:03<00:48, 7613.58it/s]  7%|▋         | 28728/400000 [00:03<00:48, 7659.85it/s]  7%|▋         | 29495/400000 [00:03<00:48, 7593.81it/s]  8%|▊         | 30265/400000 [00:04<00:48, 7622.64it/s]  8%|▊         | 31040/400000 [00:04<00:48, 7657.77it/s]  8%|▊         | 31807/400000 [00:04<00:48, 7646.67it/s]  8%|▊         | 32573/400000 [00:04<00:48, 7647.95it/s]  8%|▊         | 33345/400000 [00:04<00:47, 7668.83it/s]  9%|▊         | 34112/400000 [00:04<00:47, 7638.95it/s]  9%|▊         | 34907/400000 [00:04<00:47, 7729.32it/s]  9%|▉         | 35681/400000 [00:04<00:47, 7674.39it/s]  9%|▉         | 36449/400000 [00:04<00:47, 7589.43it/s]  9%|▉         | 37209/400000 [00:04<00:47, 7561.78it/s]  9%|▉         | 37966/400000 [00:05<00:47, 7542.49it/s] 10%|▉         | 38721/400000 [00:05<00:47, 7531.15it/s] 10%|▉         | 39492/400000 [00:05<00:47, 7583.30it/s] 10%|█         | 40280/400000 [00:05<00:46, 7668.89it/s] 10%|█         | 41048/400000 [00:05<00:47, 7522.40it/s] 10%|█         | 41802/400000 [00:05<00:47, 7518.86it/s] 11%|█         | 42589/400000 [00:05<00:46, 7620.01it/s] 11%|█         | 43356/400000 [00:05<00:46, 7632.37it/s] 11%|█         | 44120/400000 [00:05<00:47, 7520.16it/s] 11%|█         | 44873/400000 [00:05<00:47, 7522.48it/s] 11%|█▏        | 45626/400000 [00:06<00:47, 7423.99it/s] 12%|█▏        | 46370/400000 [00:06<00:47, 7422.45it/s] 12%|█▏        | 47133/400000 [00:06<00:47, 7481.40it/s] 12%|█▏        | 47904/400000 [00:06<00:46, 7547.45it/s] 12%|█▏        | 48660/400000 [00:06<00:46, 7546.94it/s] 12%|█▏        | 49416/400000 [00:06<00:47, 7429.14it/s] 13%|█▎        | 50160/400000 [00:06<00:47, 7424.59it/s] 13%|█▎        | 50903/400000 [00:06<00:47, 7350.58it/s] 13%|█▎        | 51639/400000 [00:06<00:47, 7323.50it/s] 13%|█▎        | 52397/400000 [00:06<00:46, 7396.29it/s] 13%|█▎        | 53138/400000 [00:07<00:47, 7338.74it/s] 13%|█▎        | 53893/400000 [00:07<00:46, 7399.28it/s] 14%|█▎        | 54634/400000 [00:07<00:46, 7402.46it/s] 14%|█▍        | 55407/400000 [00:07<00:45, 7495.99it/s] 14%|█▍        | 56191/400000 [00:07<00:45, 7593.80it/s] 14%|█▍        | 56952/400000 [00:07<00:45, 7523.13it/s] 14%|█▍        | 57705/400000 [00:07<00:45, 7497.00it/s] 15%|█▍        | 58464/400000 [00:07<00:45, 7523.45it/s] 15%|█▍        | 59217/400000 [00:07<00:45, 7518.47it/s] 15%|█▍        | 59989/400000 [00:07<00:44, 7577.21it/s] 15%|█▌        | 60747/400000 [00:08<00:45, 7435.69it/s] 15%|█▌        | 61515/400000 [00:08<00:45, 7505.80it/s] 16%|█▌        | 62276/400000 [00:08<00:44, 7534.48it/s] 16%|█▌        | 63035/400000 [00:08<00:44, 7550.66it/s] 16%|█▌        | 63791/400000 [00:08<00:44, 7519.85it/s] 16%|█▌        | 64544/400000 [00:08<00:45, 7449.28it/s] 16%|█▋        | 65299/400000 [00:08<00:44, 7477.86it/s] 17%|█▋        | 66060/400000 [00:08<00:44, 7516.39it/s] 17%|█▋        | 66824/400000 [00:08<00:44, 7549.98it/s] 17%|█▋        | 67580/400000 [00:08<00:44, 7527.04it/s] 17%|█▋        | 68333/400000 [00:09<00:44, 7493.28it/s] 17%|█▋        | 69088/400000 [00:09<00:44, 7508.84it/s] 17%|█▋        | 69840/400000 [00:09<00:43, 7510.64it/s] 18%|█▊        | 70592/400000 [00:09<00:43, 7513.07it/s] 18%|█▊        | 71382/400000 [00:09<00:43, 7622.38it/s] 18%|█▊        | 72145/400000 [00:09<00:43, 7578.54it/s] 18%|█▊        | 72914/400000 [00:09<00:42, 7609.94it/s] 18%|█▊        | 73676/400000 [00:09<00:43, 7572.65it/s] 19%|█▊        | 74434/400000 [00:09<00:43, 7482.60it/s] 19%|█▉        | 75183/400000 [00:10<00:43, 7440.06it/s] 19%|█▉        | 75928/400000 [00:10<00:44, 7353.82it/s] 19%|█▉        | 76667/400000 [00:10<00:43, 7362.77it/s] 19%|█▉        | 77411/400000 [00:10<00:43, 7385.25it/s] 20%|█▉        | 78151/400000 [00:10<00:43, 7387.40it/s] 20%|█▉        | 78902/400000 [00:10<00:43, 7423.19it/s] 20%|█▉        | 79645/400000 [00:10<00:43, 7381.57it/s] 20%|██        | 80405/400000 [00:10<00:42, 7442.70it/s] 20%|██        | 81169/400000 [00:10<00:42, 7499.46it/s] 20%|██        | 81920/400000 [00:10<00:42, 7481.59it/s] 21%|██        | 82669/400000 [00:11<00:42, 7445.93it/s] 21%|██        | 83414/400000 [00:11<00:44, 7157.82it/s] 21%|██        | 84162/400000 [00:11<00:43, 7250.52it/s] 21%|██        | 84901/400000 [00:11<00:43, 7291.51it/s] 21%|██▏       | 85632/400000 [00:11<00:43, 7235.46it/s] 22%|██▏       | 86395/400000 [00:11<00:42, 7348.65it/s] 22%|██▏       | 87132/400000 [00:11<00:42, 7336.29it/s] 22%|██▏       | 87875/400000 [00:11<00:42, 7363.73it/s] 22%|██▏       | 88613/400000 [00:11<00:42, 7346.94it/s] 22%|██▏       | 89386/400000 [00:11<00:41, 7455.62it/s] 23%|██▎       | 90149/400000 [00:12<00:41, 7505.97it/s] 23%|██▎       | 90903/400000 [00:12<00:41, 7514.87it/s] 23%|██▎       | 91657/400000 [00:12<00:40, 7520.57it/s] 23%|██▎       | 92446/400000 [00:12<00:40, 7627.67it/s] 23%|██▎       | 93219/400000 [00:12<00:40, 7655.86it/s] 23%|██▎       | 93989/400000 [00:12<00:39, 7666.45it/s] 24%|██▎       | 94756/400000 [00:12<00:39, 7639.75it/s] 24%|██▍       | 95521/400000 [00:12<00:39, 7637.12it/s] 24%|██▍       | 96285/400000 [00:12<00:40, 7505.20it/s] 24%|██▍       | 97040/400000 [00:12<00:40, 7517.22it/s] 24%|██▍       | 97810/400000 [00:13<00:39, 7568.94it/s] 25%|██▍       | 98568/400000 [00:13<00:40, 7531.36it/s] 25%|██▍       | 99322/400000 [00:13<00:39, 7517.13it/s] 25%|██▌       | 100074/400000 [00:13<00:40, 7475.26it/s] 25%|██▌       | 100822/400000 [00:13<00:40, 7346.49it/s] 25%|██▌       | 101558/400000 [00:13<00:41, 7256.58it/s] 26%|██▌       | 102285/400000 [00:13<00:41, 7242.69it/s] 26%|██▌       | 103048/400000 [00:13<00:40, 7353.30it/s] 26%|██▌       | 103796/400000 [00:13<00:40, 7390.19it/s] 26%|██▌       | 104536/400000 [00:13<00:40, 7327.18it/s] 26%|██▋       | 105270/400000 [00:14<00:40, 7315.31it/s] 27%|██▋       | 106002/400000 [00:14<00:40, 7211.73it/s] 27%|██▋       | 106728/400000 [00:14<00:40, 7225.13it/s] 27%|██▋       | 107461/400000 [00:14<00:40, 7255.52it/s] 27%|██▋       | 108191/400000 [00:14<00:40, 7268.41it/s] 27%|██▋       | 108964/400000 [00:14<00:39, 7400.65it/s] 27%|██▋       | 109705/400000 [00:14<00:39, 7308.70it/s] 28%|██▊       | 110448/400000 [00:14<00:39, 7343.04it/s] 28%|██▊       | 111183/400000 [00:14<00:39, 7262.17it/s] 28%|██▊       | 111921/400000 [00:14<00:39, 7294.28it/s] 28%|██▊       | 112662/400000 [00:15<00:39, 7328.43it/s] 28%|██▊       | 113396/400000 [00:15<00:39, 7242.57it/s] 29%|██▊       | 114131/400000 [00:15<00:39, 7273.51it/s] 29%|██▊       | 114874/400000 [00:15<00:38, 7319.73it/s] 29%|██▉       | 115627/400000 [00:15<00:38, 7381.55it/s] 29%|██▉       | 116379/400000 [00:15<00:38, 7421.45it/s] 29%|██▉       | 117122/400000 [00:15<00:38, 7369.68it/s] 29%|██▉       | 117862/400000 [00:15<00:38, 7376.10it/s] 30%|██▉       | 118600/400000 [00:15<00:38, 7234.66it/s] 30%|██▉       | 119341/400000 [00:15<00:38, 7284.36it/s] 30%|███       | 120079/400000 [00:16<00:38, 7310.76it/s] 30%|███       | 120811/400000 [00:16<00:38, 7253.07it/s] 30%|███       | 121553/400000 [00:16<00:38, 7301.31it/s] 31%|███       | 122293/400000 [00:16<00:37, 7330.23it/s] 31%|███       | 123027/400000 [00:16<00:38, 7227.91it/s] 31%|███       | 123753/400000 [00:16<00:38, 7235.52it/s] 31%|███       | 124477/400000 [00:16<00:38, 7207.66it/s] 31%|███▏      | 125199/400000 [00:16<00:38, 7191.92it/s] 31%|███▏      | 125919/400000 [00:16<00:38, 7129.62it/s] 32%|███▏      | 126669/400000 [00:16<00:37, 7235.19it/s] 32%|███▏      | 127403/400000 [00:17<00:37, 7265.13it/s] 32%|███▏      | 128130/400000 [00:17<00:37, 7218.15it/s] 32%|███▏      | 128854/400000 [00:17<00:37, 7222.01it/s] 32%|███▏      | 129577/400000 [00:17<00:37, 7210.26it/s] 33%|███▎      | 130347/400000 [00:17<00:36, 7349.31it/s] 33%|███▎      | 131086/400000 [00:17<00:36, 7360.39it/s] 33%|███▎      | 131823/400000 [00:17<00:36, 7341.41it/s] 33%|███▎      | 132564/400000 [00:17<00:36, 7361.27it/s] 33%|███▎      | 133301/400000 [00:17<00:36, 7310.79it/s] 34%|███▎      | 134036/400000 [00:18<00:36, 7322.48it/s] 34%|███▎      | 134769/400000 [00:18<00:36, 7301.15it/s] 34%|███▍      | 135500/400000 [00:18<00:36, 7283.81it/s] 34%|███▍      | 136230/400000 [00:18<00:36, 7286.64it/s] 34%|███▍      | 136967/400000 [00:18<00:35, 7310.97it/s] 34%|███▍      | 137702/400000 [00:18<00:35, 7321.60it/s] 35%|███▍      | 138435/400000 [00:18<00:35, 7271.54it/s] 35%|███▍      | 139173/400000 [00:18<00:35, 7302.26it/s] 35%|███▍      | 139904/400000 [00:18<00:35, 7234.36it/s] 35%|███▌      | 140629/400000 [00:18<00:35, 7238.77it/s] 35%|███▌      | 141379/400000 [00:19<00:35, 7313.11it/s] 36%|███▌      | 142117/400000 [00:19<00:35, 7332.46it/s] 36%|███▌      | 142851/400000 [00:19<00:35, 7275.63it/s] 36%|███▌      | 143589/400000 [00:19<00:35, 7305.54it/s] 36%|███▌      | 144320/400000 [00:19<00:35, 7247.93it/s] 36%|███▋      | 145075/400000 [00:19<00:34, 7332.77it/s] 36%|███▋      | 145833/400000 [00:19<00:34, 7404.07it/s] 37%|███▋      | 146574/400000 [00:19<00:34, 7403.08it/s] 37%|███▋      | 147353/400000 [00:19<00:33, 7511.81it/s] 37%|███▋      | 148123/400000 [00:19<00:33, 7564.55it/s] 37%|███▋      | 148881/400000 [00:20<00:33, 7566.87it/s] 37%|███▋      | 149656/400000 [00:20<00:32, 7618.86it/s] 38%|███▊      | 150419/400000 [00:20<00:33, 7491.41it/s] 38%|███▊      | 151169/400000 [00:20<00:33, 7410.89it/s] 38%|███▊      | 151911/400000 [00:20<00:33, 7410.18it/s] 38%|███▊      | 152664/400000 [00:20<00:33, 7445.21it/s] 38%|███▊      | 153409/400000 [00:20<00:33, 7431.88it/s] 39%|███▊      | 154153/400000 [00:20<00:33, 7360.42it/s] 39%|███▊      | 154905/400000 [00:20<00:33, 7405.66it/s] 39%|███▉      | 155646/400000 [00:20<00:33, 7348.35it/s] 39%|███▉      | 156382/400000 [00:21<00:33, 7232.56it/s] 39%|███▉      | 157106/400000 [00:21<00:33, 7210.29it/s] 39%|███▉      | 157836/400000 [00:21<00:33, 7235.28it/s] 40%|███▉      | 158573/400000 [00:21<00:33, 7271.36it/s] 40%|███▉      | 159305/400000 [00:21<00:33, 7284.79it/s] 40%|████      | 160055/400000 [00:21<00:32, 7347.80it/s] 40%|████      | 160791/400000 [00:21<00:32, 7323.71it/s] 40%|████      | 161524/400000 [00:21<00:32, 7249.03it/s] 41%|████      | 162250/400000 [00:21<00:33, 7162.08it/s] 41%|████      | 162988/400000 [00:21<00:32, 7225.31it/s] 41%|████      | 163725/400000 [00:22<00:32, 7266.98it/s] 41%|████      | 164480/400000 [00:22<00:32, 7346.65it/s] 41%|████▏     | 165219/400000 [00:22<00:31, 7358.50it/s] 41%|████▏     | 165965/400000 [00:22<00:31, 7387.74it/s] 42%|████▏     | 166705/400000 [00:22<00:31, 7351.53it/s] 42%|████▏     | 167441/400000 [00:22<00:31, 7294.29it/s] 42%|████▏     | 168171/400000 [00:22<00:31, 7294.99it/s] 42%|████▏     | 168901/400000 [00:22<00:31, 7242.83it/s] 42%|████▏     | 169626/400000 [00:22<00:31, 7235.23it/s] 43%|████▎     | 170350/400000 [00:22<00:32, 7151.69it/s] 43%|████▎     | 171094/400000 [00:23<00:31, 7234.99it/s] 43%|████▎     | 171846/400000 [00:23<00:31, 7316.01it/s] 43%|████▎     | 172589/400000 [00:23<00:30, 7348.90it/s] 43%|████▎     | 173325/400000 [00:23<00:30, 7317.97it/s] 44%|████▎     | 174058/400000 [00:23<00:31, 7270.89it/s] 44%|████▎     | 174800/400000 [00:23<00:30, 7313.24it/s] 44%|████▍     | 175532/400000 [00:23<00:30, 7240.97it/s] 44%|████▍     | 176257/400000 [00:23<00:30, 7241.44it/s] 44%|████▍     | 176982/400000 [00:23<00:31, 7156.80it/s] 44%|████▍     | 177722/400000 [00:23<00:30, 7227.70it/s] 45%|████▍     | 178446/400000 [00:24<00:30, 7196.84it/s] 45%|████▍     | 179204/400000 [00:24<00:30, 7306.14it/s] 45%|████▍     | 179967/400000 [00:24<00:29, 7397.23it/s] 45%|████▌     | 180708/400000 [00:24<00:29, 7389.87it/s] 45%|████▌     | 181463/400000 [00:24<00:29, 7435.45it/s] 46%|████▌     | 182230/400000 [00:24<00:29, 7503.42it/s] 46%|████▌     | 182981/400000 [00:24<00:28, 7495.86it/s] 46%|████▌     | 183746/400000 [00:24<00:28, 7539.60it/s] 46%|████▌     | 184501/400000 [00:24<00:28, 7463.61it/s] 46%|████▋     | 185249/400000 [00:24<00:28, 7467.19it/s] 47%|████▋     | 186002/400000 [00:25<00:28, 7483.63it/s] 47%|████▋     | 186751/400000 [00:25<00:28, 7476.24it/s] 47%|████▋     | 187499/400000 [00:25<00:28, 7396.08it/s] 47%|████▋     | 188239/400000 [00:25<00:28, 7362.58it/s] 47%|████▋     | 188976/400000 [00:25<00:29, 7237.35it/s] 47%|████▋     | 189716/400000 [00:25<00:28, 7281.77it/s] 48%|████▊     | 190451/400000 [00:25<00:28, 7299.61it/s] 48%|████▊     | 191193/400000 [00:25<00:28, 7335.20it/s] 48%|████▊     | 191927/400000 [00:25<00:28, 7221.70it/s] 48%|████▊     | 192669/400000 [00:25<00:28, 7279.54it/s] 48%|████▊     | 193398/400000 [00:26<00:28, 7156.40it/s] 49%|████▊     | 194115/400000 [00:26<00:28, 7100.82it/s] 49%|████▊     | 194870/400000 [00:26<00:28, 7228.28it/s] 49%|████▉     | 195597/400000 [00:26<00:28, 7238.72it/s] 49%|████▉     | 196322/400000 [00:26<00:28, 7213.78it/s] 49%|████▉     | 197056/400000 [00:26<00:27, 7250.64it/s] 49%|████▉     | 197798/400000 [00:26<00:27, 7299.57it/s] 50%|████▉     | 198529/400000 [00:26<00:27, 7288.40it/s] 50%|████▉     | 199259/400000 [00:26<00:28, 7162.66it/s] 50%|████▉     | 199976/400000 [00:27<00:28, 7095.05it/s] 50%|█████     | 200722/400000 [00:27<00:27, 7200.26it/s] 50%|█████     | 201477/400000 [00:27<00:27, 7299.71it/s] 51%|█████     | 202230/400000 [00:27<00:26, 7366.66it/s] 51%|█████     | 202968/400000 [00:27<00:27, 7262.45it/s] 51%|█████     | 203713/400000 [00:27<00:26, 7315.75it/s] 51%|█████     | 204449/400000 [00:27<00:26, 7328.23it/s] 51%|█████▏    | 205195/400000 [00:27<00:26, 7365.60it/s] 51%|█████▏    | 205932/400000 [00:27<00:26, 7366.42it/s] 52%|█████▏    | 206669/400000 [00:27<00:26, 7195.55it/s] 52%|█████▏    | 207404/400000 [00:28<00:26, 7238.27it/s] 52%|█████▏    | 208129/400000 [00:28<00:26, 7193.15it/s] 52%|█████▏    | 208870/400000 [00:28<00:26, 7256.38it/s] 52%|█████▏    | 209620/400000 [00:28<00:25, 7327.77it/s] 53%|█████▎    | 210354/400000 [00:28<00:26, 7265.99it/s] 53%|█████▎    | 211096/400000 [00:28<00:25, 7311.20it/s] 53%|█████▎    | 211828/400000 [00:28<00:25, 7300.08it/s] 53%|█████▎    | 212572/400000 [00:28<00:25, 7340.99it/s] 53%|█████▎    | 213327/400000 [00:28<00:25, 7400.28it/s] 54%|█████▎    | 214068/400000 [00:28<00:25, 7364.98it/s] 54%|█████▎    | 214835/400000 [00:29<00:24, 7451.52it/s] 54%|█████▍    | 215583/400000 [00:29<00:24, 7458.39it/s] 54%|█████▍    | 216334/400000 [00:29<00:24, 7471.36it/s] 54%|█████▍    | 217082/400000 [00:29<00:24, 7385.57it/s] 54%|█████▍    | 217821/400000 [00:29<00:24, 7319.67it/s] 55%|█████▍    | 218554/400000 [00:29<00:24, 7265.50it/s] 55%|█████▍    | 219300/400000 [00:29<00:24, 7321.80it/s] 55%|█████▌    | 220033/400000 [00:29<00:24, 7297.08it/s] 55%|█████▌    | 220774/400000 [00:29<00:24, 7328.49it/s] 55%|█████▌    | 221508/400000 [00:29<00:24, 7276.69it/s] 56%|█████▌    | 222236/400000 [00:30<00:24, 7268.33it/s] 56%|█████▌    | 222980/400000 [00:30<00:24, 7318.51it/s] 56%|█████▌    | 223718/400000 [00:30<00:24, 7335.00it/s] 56%|█████▌    | 224456/400000 [00:30<00:23, 7346.56it/s] 56%|█████▋    | 225191/400000 [00:30<00:23, 7321.00it/s] 56%|█████▋    | 225924/400000 [00:30<00:23, 7318.72it/s] 57%|█████▋    | 226669/400000 [00:30<00:23, 7357.18it/s] 57%|█████▋    | 227405/400000 [00:30<00:23, 7328.51it/s] 57%|█████▋    | 228138/400000 [00:30<00:23, 7227.13it/s] 57%|█████▋    | 228862/400000 [00:30<00:23, 7178.32it/s] 57%|█████▋    | 229581/400000 [00:31<00:23, 7139.47it/s] 58%|█████▊    | 230306/400000 [00:31<00:23, 7169.89it/s] 58%|█████▊    | 231052/400000 [00:31<00:23, 7253.06it/s] 58%|█████▊    | 231830/400000 [00:31<00:22, 7403.21it/s] 58%|█████▊    | 232583/400000 [00:31<00:22, 7438.32it/s] 58%|█████▊    | 233328/400000 [00:31<00:22, 7434.47it/s] 59%|█████▊    | 234072/400000 [00:31<00:22, 7389.18it/s] 59%|█████▊    | 234812/400000 [00:31<00:22, 7391.79it/s] 59%|█████▉    | 235560/400000 [00:31<00:22, 7415.57it/s] 59%|█████▉    | 236302/400000 [00:31<00:22, 7409.31it/s] 59%|█████▉    | 237044/400000 [00:32<00:22, 7313.66it/s] 59%|█████▉    | 237776/400000 [00:32<00:22, 7275.86it/s] 60%|█████▉    | 238504/400000 [00:32<00:22, 7254.45it/s] 60%|█████▉    | 239230/400000 [00:32<00:22, 7216.58it/s] 60%|█████▉    | 239955/400000 [00:32<00:22, 7225.00it/s] 60%|██████    | 240678/400000 [00:32<00:22, 7172.65it/s] 60%|██████    | 241409/400000 [00:32<00:21, 7212.38it/s] 61%|██████    | 242156/400000 [00:32<00:21, 7286.10it/s] 61%|██████    | 242913/400000 [00:32<00:21, 7366.85it/s] 61%|██████    | 243673/400000 [00:32<00:21, 7434.59it/s] 61%|██████    | 244417/400000 [00:33<00:21, 7291.13it/s] 61%|██████▏   | 245159/400000 [00:33<00:21, 7328.91it/s] 61%|██████▏   | 245894/400000 [00:33<00:21, 7334.13it/s] 62%|██████▏   | 246628/400000 [00:33<00:20, 7322.49it/s] 62%|██████▏   | 247361/400000 [00:33<00:20, 7311.30it/s] 62%|██████▏   | 248093/400000 [00:33<00:20, 7238.03it/s] 62%|██████▏   | 248818/400000 [00:33<00:21, 7166.04it/s] 62%|██████▏   | 249572/400000 [00:33<00:20, 7273.70it/s] 63%|██████▎   | 250327/400000 [00:33<00:20, 7352.66it/s] 63%|██████▎   | 251071/400000 [00:33<00:20, 7376.96it/s] 63%|██████▎   | 251810/400000 [00:34<00:20, 7345.24it/s] 63%|██████▎   | 252559/400000 [00:34<00:19, 7386.27it/s] 63%|██████▎   | 253298/400000 [00:34<00:20, 7318.86it/s] 64%|██████▎   | 254037/400000 [00:34<00:19, 7338.13it/s] 64%|██████▎   | 254785/400000 [00:34<00:19, 7378.34it/s] 64%|██████▍   | 255524/400000 [00:34<00:19, 7338.07it/s] 64%|██████▍   | 256269/400000 [00:34<00:19, 7369.09it/s] 64%|██████▍   | 257007/400000 [00:34<00:19, 7331.14it/s] 64%|██████▍   | 257741/400000 [00:34<00:19, 7264.76it/s] 65%|██████▍   | 258470/400000 [00:35<00:19, 7272.22it/s] 65%|██████▍   | 259198/400000 [00:35<00:19, 7171.92it/s] 65%|██████▍   | 259920/400000 [00:35<00:19, 7185.34it/s] 65%|██████▌   | 260639/400000 [00:35<00:19, 7032.41it/s] 65%|██████▌   | 261355/400000 [00:35<00:19, 7069.93it/s] 66%|██████▌   | 262063/400000 [00:35<00:19, 7056.07it/s] 66%|██████▌   | 262783/400000 [00:35<00:19, 7097.06it/s] 66%|██████▌   | 263516/400000 [00:35<00:19, 7163.62it/s] 66%|██████▌   | 264233/400000 [00:35<00:19, 7141.30it/s] 66%|██████▌   | 264948/400000 [00:35<00:18, 7132.17it/s] 66%|██████▋   | 265669/400000 [00:36<00:18, 7154.66it/s] 67%|██████▋   | 266385/400000 [00:36<00:18, 7051.90it/s] 67%|██████▋   | 267091/400000 [00:36<00:19, 6948.34it/s] 67%|██████▋   | 267824/400000 [00:36<00:18, 7056.58it/s] 67%|██████▋   | 268559/400000 [00:36<00:18, 7138.67it/s] 67%|██████▋   | 269274/400000 [00:36<00:18, 7101.19it/s] 67%|██████▋   | 269985/400000 [00:36<00:19, 6842.06it/s] 68%|██████▊   | 270743/400000 [00:36<00:18, 7047.25it/s] 68%|██████▊   | 271452/400000 [00:36<00:18, 7023.30it/s] 68%|██████▊   | 272157/400000 [00:36<00:19, 6695.88it/s] 68%|██████▊   | 272874/400000 [00:37<00:18, 6829.33it/s] 68%|██████▊   | 273573/400000 [00:37<00:18, 6876.36it/s] 69%|██████▊   | 274267/400000 [00:37<00:18, 6852.94it/s] 69%|██████▊   | 274955/400000 [00:37<00:18, 6725.23it/s] 69%|██████▉   | 275674/400000 [00:37<00:18, 6858.12it/s] 69%|██████▉   | 276405/400000 [00:37<00:17, 6985.06it/s] 69%|██████▉   | 277116/400000 [00:37<00:17, 7019.26it/s] 69%|██████▉   | 277857/400000 [00:37<00:17, 7131.71it/s] 70%|██████▉   | 278572/400000 [00:37<00:17, 7085.03it/s] 70%|██████▉   | 279314/400000 [00:37<00:16, 7182.24it/s] 70%|███████   | 280034/400000 [00:38<00:17, 6784.95it/s] 70%|███████   | 280747/400000 [00:38<00:17, 6884.14it/s] 70%|███████   | 281452/400000 [00:38<00:17, 6932.26it/s] 71%|███████   | 282203/400000 [00:38<00:16, 7095.97it/s] 71%|███████   | 282946/400000 [00:38<00:16, 7192.19it/s] 71%|███████   | 283669/400000 [00:38<00:16, 7201.63it/s] 71%|███████   | 284391/400000 [00:38<00:16, 6942.90it/s] 71%|███████▏  | 285130/400000 [00:38<00:16, 7068.82it/s] 71%|███████▏  | 285848/400000 [00:38<00:16, 7099.61it/s] 72%|███████▏  | 286560/400000 [00:39<00:16, 6969.59it/s] 72%|███████▏  | 287284/400000 [00:39<00:15, 7048.38it/s] 72%|███████▏  | 288024/400000 [00:39<00:15, 7148.33it/s] 72%|███████▏  | 288769/400000 [00:39<00:15, 7234.21it/s] 72%|███████▏  | 289531/400000 [00:39<00:15, 7344.92it/s] 73%|███████▎  | 290280/400000 [00:39<00:14, 7387.21it/s] 73%|███████▎  | 291020/400000 [00:39<00:15, 7174.13it/s] 73%|███████▎  | 291740/400000 [00:39<00:15, 7007.38it/s] 73%|███████▎  | 292484/400000 [00:39<00:15, 7130.31it/s] 73%|███████▎  | 293200/400000 [00:39<00:15, 6928.84it/s] 73%|███████▎  | 293896/400000 [00:40<00:15, 6935.77it/s] 74%|███████▎  | 294592/400000 [00:40<00:15, 6843.14it/s] 74%|███████▍  | 295278/400000 [00:40<00:16, 6495.68it/s] 74%|███████▍  | 295992/400000 [00:40<00:15, 6674.65it/s] 74%|███████▍  | 296681/400000 [00:40<00:15, 6736.72it/s] 74%|███████▍  | 297434/400000 [00:40<00:14, 6956.19it/s] 75%|███████▍  | 298134/400000 [00:40<00:14, 6964.12it/s] 75%|███████▍  | 298860/400000 [00:40<00:14, 7047.93it/s] 75%|███████▍  | 299591/400000 [00:40<00:14, 7122.53it/s] 75%|███████▌  | 300339/400000 [00:40<00:13, 7225.95it/s] 75%|███████▌  | 301077/400000 [00:41<00:13, 7268.88it/s] 75%|███████▌  | 301806/400000 [00:41<00:13, 7084.34it/s] 76%|███████▌  | 302517/400000 [00:41<00:14, 6902.99it/s] 76%|███████▌  | 303237/400000 [00:41<00:13, 6984.19it/s] 76%|███████▌  | 303938/400000 [00:41<00:14, 6728.39it/s] 76%|███████▌  | 304669/400000 [00:41<00:13, 6890.96it/s] 76%|███████▋  | 305362/400000 [00:41<00:13, 6879.08it/s] 77%|███████▋  | 306082/400000 [00:41<00:13, 6970.53it/s] 77%|███████▋  | 306781/400000 [00:41<00:13, 6764.31it/s] 77%|███████▋  | 307498/400000 [00:42<00:13, 6877.98it/s] 77%|███████▋  | 308189/400000 [00:42<00:13, 6775.91it/s] 77%|███████▋  | 308875/400000 [00:42<00:13, 6798.92it/s] 77%|███████▋  | 309600/400000 [00:42<00:13, 6924.31it/s] 78%|███████▊  | 310333/400000 [00:42<00:12, 7039.65it/s] 78%|███████▊  | 311059/400000 [00:42<00:12, 7102.33it/s] 78%|███████▊  | 311771/400000 [00:42<00:12, 6913.05it/s] 78%|███████▊  | 312482/400000 [00:42<00:12, 6970.17it/s] 78%|███████▊  | 313201/400000 [00:42<00:12, 7033.50it/s] 78%|███████▊  | 313906/400000 [00:42<00:12, 6676.11it/s] 79%|███████▊  | 314608/400000 [00:43<00:12, 6773.34it/s] 79%|███████▉  | 315289/400000 [00:43<00:12, 6745.60it/s] 79%|███████▉  | 315967/400000 [00:43<00:12, 6582.87it/s] 79%|███████▉  | 316687/400000 [00:43<00:12, 6755.49it/s] 79%|███████▉  | 317383/400000 [00:43<00:12, 6814.18it/s] 80%|███████▉  | 318094/400000 [00:43<00:11, 6900.25it/s] 80%|███████▉  | 318786/400000 [00:43<00:11, 6883.57it/s] 80%|███████▉  | 319476/400000 [00:43<00:11, 6816.16it/s] 80%|████████  | 320192/400000 [00:43<00:11, 6915.49it/s] 80%|████████  | 320915/400000 [00:43<00:11, 7006.01it/s] 80%|████████  | 321642/400000 [00:44<00:11, 7080.63it/s] 81%|████████  | 322376/400000 [00:44<00:10, 7155.21it/s] 81%|████████  | 323093/400000 [00:44<00:11, 6951.69it/s] 81%|████████  | 323822/400000 [00:44<00:10, 7046.25it/s] 81%|████████  | 324548/400000 [00:44<00:10, 7108.93it/s] 81%|████████▏ | 325304/400000 [00:44<00:10, 7235.59it/s] 82%|████████▏ | 326029/400000 [00:44<00:10, 6954.99it/s] 82%|████████▏ | 326737/400000 [00:44<00:10, 6991.39it/s] 82%|████████▏ | 327478/400000 [00:44<00:10, 7111.01it/s] 82%|████████▏ | 328192/400000 [00:44<00:10, 6996.72it/s] 82%|████████▏ | 328935/400000 [00:45<00:09, 7120.86it/s] 82%|████████▏ | 329649/400000 [00:45<00:09, 7064.51it/s] 83%|████████▎ | 330357/400000 [00:45<00:10, 6918.43it/s] 83%|████████▎ | 331051/400000 [00:45<00:10, 6893.42it/s] 83%|████████▎ | 331742/400000 [00:45<00:10, 6751.03it/s] 83%|████████▎ | 332486/400000 [00:45<00:09, 6942.98it/s] 83%|████████▎ | 333196/400000 [00:45<00:09, 6987.26it/s] 83%|████████▎ | 333897/400000 [00:45<00:09, 6968.10it/s] 84%|████████▎ | 334600/400000 [00:45<00:09, 6985.33it/s] 84%|████████▍ | 335362/400000 [00:46<00:09, 7163.79it/s] 84%|████████▍ | 336127/400000 [00:46<00:08, 7302.67it/s] 84%|████████▍ | 336884/400000 [00:46<00:08, 7379.68it/s] 84%|████████▍ | 337631/400000 [00:46<00:08, 7404.60it/s] 85%|████████▍ | 338397/400000 [00:46<00:08, 7479.06it/s] 85%|████████▍ | 339146/400000 [00:46<00:08, 7468.10it/s] 85%|████████▍ | 339916/400000 [00:46<00:07, 7535.89it/s] 85%|████████▌ | 340671/400000 [00:46<00:07, 7510.95it/s] 85%|████████▌ | 341423/400000 [00:46<00:07, 7472.11it/s] 86%|████████▌ | 342171/400000 [00:46<00:07, 7463.66it/s] 86%|████████▌ | 342927/400000 [00:47<00:07, 7490.62it/s] 86%|████████▌ | 343680/400000 [00:47<00:07, 7501.35it/s] 86%|████████▌ | 344431/400000 [00:47<00:07, 7422.34it/s] 86%|████████▋ | 345174/400000 [00:47<00:07, 7382.18it/s] 86%|████████▋ | 345913/400000 [00:47<00:07, 7143.78it/s] 87%|████████▋ | 346630/400000 [00:47<00:07, 6997.90it/s] 87%|████████▋ | 347338/400000 [00:47<00:07, 7021.92it/s] 87%|████████▋ | 348065/400000 [00:47<00:07, 7093.86it/s] 87%|████████▋ | 348776/400000 [00:47<00:07, 6831.90it/s] 87%|████████▋ | 349463/400000 [00:47<00:07, 6783.06it/s] 88%|████████▊ | 350160/400000 [00:48<00:07, 6837.50it/s] 88%|████████▊ | 350846/400000 [00:48<00:07, 6792.02it/s] 88%|████████▊ | 351563/400000 [00:48<00:07, 6899.95it/s] 88%|████████▊ | 352290/400000 [00:48<00:06, 7005.86it/s] 88%|████████▊ | 352992/400000 [00:48<00:06, 6903.85it/s] 88%|████████▊ | 353684/400000 [00:48<00:06, 6906.44it/s] 89%|████████▊ | 354376/400000 [00:48<00:06, 6899.39it/s] 89%|████████▉ | 355124/400000 [00:48<00:06, 7062.46it/s] 89%|████████▉ | 355851/400000 [00:48<00:06, 7123.41it/s] 89%|████████▉ | 356607/400000 [00:48<00:05, 7246.41it/s] 89%|████████▉ | 357363/400000 [00:49<00:05, 7336.71it/s] 90%|████████▉ | 358098/400000 [00:49<00:05, 7287.75it/s] 90%|████████▉ | 358828/400000 [00:49<00:05, 7019.98it/s] 90%|████████▉ | 359550/400000 [00:49<00:05, 7077.48it/s] 90%|█████████ | 360303/400000 [00:49<00:05, 7207.40it/s] 90%|█████████ | 361073/400000 [00:49<00:05, 7348.25it/s] 90%|█████████ | 361864/400000 [00:49<00:05, 7506.31it/s] 91%|█████████ | 362626/400000 [00:49<00:04, 7537.38it/s] 91%|█████████ | 363382/400000 [00:49<00:04, 7528.04it/s] 91%|█████████ | 364136/400000 [00:49<00:04, 7490.51it/s] 91%|█████████ | 364892/400000 [00:50<00:04, 7510.29it/s] 91%|█████████▏| 365644/400000 [00:50<00:04, 7466.02it/s] 92%|█████████▏| 366392/400000 [00:50<00:04, 7135.03it/s] 92%|█████████▏| 367129/400000 [00:50<00:04, 7201.89it/s] 92%|█████████▏| 367892/400000 [00:50<00:04, 7323.99it/s] 92%|█████████▏| 368667/400000 [00:50<00:04, 7445.09it/s] 92%|█████████▏| 369440/400000 [00:50<00:04, 7525.82it/s] 93%|█████████▎| 370195/400000 [00:50<00:04, 7287.23it/s] 93%|█████████▎| 370927/400000 [00:50<00:04, 7055.36it/s] 93%|█████████▎| 371672/400000 [00:51<00:03, 7167.81it/s] 93%|█████████▎| 372419/400000 [00:51<00:03, 7254.76it/s] 93%|█████████▎| 373185/400000 [00:51<00:03, 7370.48it/s] 93%|█████████▎| 373925/400000 [00:51<00:03, 7321.36it/s] 94%|█████████▎| 374659/400000 [00:51<00:03, 7186.65it/s] 94%|█████████▍| 375380/400000 [00:51<00:03, 7022.16it/s] 94%|█████████▍| 376143/400000 [00:51<00:03, 7192.64it/s] 94%|█████████▍| 376914/400000 [00:51<00:03, 7340.27it/s] 94%|█████████▍| 377651/400000 [00:51<00:03, 7341.62it/s] 95%|█████████▍| 378428/400000 [00:51<00:02, 7463.96it/s] 95%|█████████▍| 379198/400000 [00:52<00:02, 7531.97it/s] 95%|█████████▍| 379953/400000 [00:52<00:02, 7082.46it/s] 95%|█████████▌| 380675/400000 [00:52<00:02, 7120.48it/s] 95%|█████████▌| 381397/400000 [00:52<00:02, 7147.41it/s] 96%|█████████▌| 382157/400000 [00:52<00:02, 7276.69it/s] 96%|█████████▌| 382915/400000 [00:52<00:02, 7364.61it/s] 96%|█████████▌| 383657/400000 [00:52<00:02, 7380.50it/s] 96%|█████████▌| 384401/400000 [00:52<00:02, 7398.10it/s] 96%|█████████▋| 385142/400000 [00:52<00:02, 7353.50it/s] 96%|█████████▋| 385889/400000 [00:52<00:01, 7387.66it/s] 97%|█████████▋| 386642/400000 [00:53<00:01, 7428.30it/s] 97%|█████████▋| 387386/400000 [00:53<00:01, 7356.02it/s] 97%|█████████▋| 388158/400000 [00:53<00:01, 7461.27it/s] 97%|█████████▋| 388911/400000 [00:53<00:01, 7480.46it/s] 97%|█████████▋| 389685/400000 [00:53<00:01, 7556.35it/s] 98%|█████████▊| 390451/400000 [00:53<00:01, 7584.64it/s] 98%|█████████▊| 391219/400000 [00:53<00:01, 7610.01it/s] 98%|█████████▊| 391983/400000 [00:53<00:01, 7618.67it/s] 98%|█████████▊| 392746/400000 [00:53<00:00, 7607.39it/s] 98%|█████████▊| 393507/400000 [00:53<00:00, 7596.53it/s] 99%|█████████▊| 394278/400000 [00:54<00:00, 7628.93it/s] 99%|█████████▉| 395042/400000 [00:54<00:00, 7629.98it/s] 99%|█████████▉| 395817/400000 [00:54<00:00, 7664.77it/s] 99%|█████████▉| 396584/400000 [00:54<00:00, 7625.04it/s] 99%|█████████▉| 397362/400000 [00:54<00:00, 7669.26it/s]100%|█████████▉| 398130/400000 [00:54<00:00, 7242.42it/s]100%|█████████▉| 398860/400000 [00:54<00:00, 7181.06it/s]100%|█████████▉| 399635/400000 [00:54<00:00, 7341.01it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7293.16it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9fb77c1940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010988315543391674 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.010982921888995729 	 Accuracy: 69

  model saves at 69% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15767 out of table with 15764 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15767 out of table with 15764 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-15 14:24:59.753481: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 14:24:59.758190: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-15 14:24:59.758365: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558e5a0802a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 14:24:59.758381: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f9fc3332fd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.9580 - accuracy: 0.4810
 2000/25000 [=>............................] - ETA: 10s - loss: 7.9426 - accuracy: 0.4820
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.8455 - accuracy: 0.4883 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7280 - accuracy: 0.4960
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7556 - accuracy: 0.4942
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6666 - accuracy: 0.5000
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6360 - accuracy: 0.5020
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6264 - accuracy: 0.5026
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6377 - accuracy: 0.5019
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6314 - accuracy: 0.5023
11000/25000 [============>.................] - ETA: 4s - loss: 7.6318 - accuracy: 0.5023
12000/25000 [=============>................] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6407 - accuracy: 0.5017
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6590 - accuracy: 0.5005
15000/25000 [=================>............] - ETA: 3s - loss: 7.6544 - accuracy: 0.5008
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6639 - accuracy: 0.5002
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6649 - accuracy: 0.5001
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6590 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6469 - accuracy: 0.5013
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6640 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6609 - accuracy: 0.5004
25000/25000 [==============================] - 10s 404us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f9f0e7da128> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f9f17615128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.9250 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.8877 - val_crf_viterbi_accuracy: 0.0000e+00

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

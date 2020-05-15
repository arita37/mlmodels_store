
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/0ca7fc10154e30acfd3477806bcaa34404fe1bf2', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '0ca7fc10154e30acfd3477806bcaa34404fe1bf2', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/0ca7fc10154e30acfd3477806bcaa34404fe1bf2

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/0ca7fc10154e30acfd3477806bcaa34404fe1bf2

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f9f35063fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 15:12:59.832511
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 15:12:59.837601
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 15:12:59.841234
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 15:12:59.844976
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f9f4107b438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355399.3125
Epoch 2/10

1/1 [==============================] - 0s 106ms/step - loss: 257494.2656
Epoch 3/10

1/1 [==============================] - 0s 104ms/step - loss: 156720.2656
Epoch 4/10

1/1 [==============================] - 0s 108ms/step - loss: 82225.4219
Epoch 5/10

1/1 [==============================] - 0s 106ms/step - loss: 40692.8281
Epoch 6/10

1/1 [==============================] - 0s 103ms/step - loss: 22391.1543
Epoch 7/10

1/1 [==============================] - 0s 99ms/step - loss: 13697.6357
Epoch 8/10

1/1 [==============================] - 0s 100ms/step - loss: 9296.5303
Epoch 9/10

1/1 [==============================] - 0s 125ms/step - loss: 6661.8491
Epoch 10/10

1/1 [==============================] - 0s 107ms/step - loss: 5149.8184

  #### Inference Need return ypred, ytrue ######################### 
[[-0.62159616 -1.2613461   1.0828412  -0.16214734 -0.25727925 -0.3268865
  -0.7525205   0.01214024 -1.6948042  -0.02559346 -0.05844241 -1.9420784
   0.2016995   1.6650333   0.29536834  1.2276546  -0.5506559   0.9161229
   0.8412631  -0.21353291  0.7728565   1.3991387   0.6437938   0.15777528
  -0.43324375  0.88916373  0.5772954   1.2939112   0.11196673  2.380519
  -1.3142525   1.9035661  -0.4062686  -0.18577117 -0.62878114 -0.95691466
  -0.09198005 -0.07929935 -2.179842    0.8168336  -1.0469749  -0.6882022
  -0.817616   -0.2740737  -1.4000571   0.7182264   1.4012839   0.87983775
  -1.5706997   1.9263538  -0.91194624  0.77206516  0.599591   -0.14178526
   0.60361636  0.5138583  -1.5115521  -0.8489189  -0.42335778 -0.1955117
  -0.41772863 10.426621    8.492324   10.522854    9.061161    8.942974
   8.908843    9.842452    9.39253    10.9417515   9.898133    9.355937
   8.774825    7.150401   11.924468    9.476305    7.5553308  11.745784
  10.454825    9.653383    8.975653    9.442671    9.5088825   9.882614
   8.504377   10.249215   11.02453     8.534419   11.183042   10.765637
   8.779231    9.965712   11.381302    9.892659    9.120437    9.270625
   8.489628    9.372645    8.682155   10.667248    8.525905    9.037128
  10.513187   10.503865    8.694788    7.3406477   7.834657    9.357558
   9.422187   11.870086    8.9573765   9.098986    9.045309    9.965692
   9.819075    7.58506     9.58429    11.090393    9.177477   11.394159
   0.5513779   1.1155112  -1.6226327   1.2387816  -0.2534778  -0.26456323
   1.4287015  -0.8563129  -0.20919749 -1.3051095   0.41187295  0.23915344
   0.20432563 -2.519392   -0.23317315  0.4188594   0.3916844  -0.97563756
   1.2293234   0.6447221  -0.27998152  1.8968265  -0.8353595  -0.77941304
   0.7929544  -0.409006   -1.4340279   0.6539054   0.5009163  -2.0672383
  -0.60374653 -0.6276996   0.8754148  -0.22807471  1.7258174  -0.6356812
  -1.0165678   0.55875736 -1.4050046  -0.45063818  0.10669595 -1.6980581
   1.9270753  -0.10051546 -0.308321   -1.5712225   1.2503821  -0.37021983
  -0.32779962 -2.3220694  -0.3940698   0.61273205 -1.2215196  -0.05609486
   1.7281814   0.4143662  -1.8936303  -0.44637606 -0.11633968  0.7997473
   0.78056     0.16921532  0.72105896  0.71388674  0.47873354  2.356659
   0.98464584  0.6420636   0.9850878   1.8522063   0.5073215   2.663837
   0.11420363  1.0366206   2.281488    0.8078604   0.08058023  0.12200832
   0.4689327   2.170577    0.5421304   0.49176538  0.2511779   1.1686418
   1.2427864   1.4021364   1.9094172   0.41327304  1.1259538   2.1973157
   0.5605204   1.0591583   3.1703825   0.19832587  1.7894692   0.33737206
   2.1895998   0.7405584   1.4848914   2.0734      0.9696847   0.34552187
   0.05143088  0.08370495  0.9301254   1.3669913   1.16879     0.50704587
   2.9727921   0.56370735  1.3785549   0.22606993  2.0848217   1.3503065
   1.2772272   0.10527921  1.6921084   0.28559357  1.3904448   0.05807728
   0.06583905  9.636108    8.303475    7.7849627  11.440416    9.511639
  10.326843    9.6541      9.18793     9.186969   11.320154    9.951439
  11.863343    8.872465    9.894373   10.229911   11.776322    8.94631
   9.685887    9.7095     10.043728    7.871305   11.248047    9.101879
   9.565411    8.379219    7.832373   10.506959    8.935614   10.597683
   9.504858   10.9609995  10.343534   10.013999    8.163033   10.395066
   9.060513   10.738792   10.576979    8.928156    7.8621707  10.1834345
   9.283228   10.106415    9.271233    7.3066187  10.983577    9.254635
   9.77579     9.872384   11.434198   10.624352    9.6416855   9.174097
   8.166924    9.462707    9.8014345   9.539348    8.114683   10.173021
   1.0414251   2.0705237   0.5469568   1.7781788   0.8242861   0.10788077
   0.78101844  0.5018229   0.22902709  0.17295325  0.26795965  1.8213475
   2.7073402   2.417696    0.465981    1.2264445   0.26269555  2.5512123
   0.65148413  0.2505139   2.1968284   0.5685616   0.6026734   0.40088218
   2.2875848   1.942286    0.43939936  0.59859836  0.2297349   1.4324186
   0.2483955   1.5964918   0.1696406   1.0991412   0.6031338   0.48873723
   1.2627254   0.26359552  2.1221585   0.28334677  0.41574413  2.277789
   0.5724653   0.39409578  1.1715313   0.32295     0.33535218  0.43670332
   2.4103384   0.5799805   1.3014864   2.1397853   0.45015556  0.5832179
   0.18134964  0.790646    0.7391841   2.20739     0.7421341   0.5134923
  -6.6850038   6.790142   -6.5438275 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 15:13:09.664553
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    92.737
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 15:13:09.669237
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8625.46
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 15:13:09.673623
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.1169
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 15:13:09.677444
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -771.471
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140321426171384
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140320198730304
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140320198730808
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140320198731312
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140320198731816
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140320198732320

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f9f2ea465f8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.517630
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.483416
grad_step = 000002, loss = 0.452370
grad_step = 000003, loss = 0.417147
grad_step = 000004, loss = 0.376964
grad_step = 000005, loss = 0.339599
grad_step = 000006, loss = 0.314907
grad_step = 000007, loss = 0.300958
grad_step = 000008, loss = 0.287857
grad_step = 000009, loss = 0.268052
grad_step = 000010, loss = 0.249744
grad_step = 000011, loss = 0.237458
grad_step = 000012, loss = 0.229672
grad_step = 000013, loss = 0.222943
grad_step = 000014, loss = 0.214539
grad_step = 000015, loss = 0.203585
grad_step = 000016, loss = 0.191833
grad_step = 000017, loss = 0.181176
grad_step = 000018, loss = 0.172059
grad_step = 000019, loss = 0.163651
grad_step = 000020, loss = 0.154840
grad_step = 000021, loss = 0.145568
grad_step = 000022, loss = 0.136942
grad_step = 000023, loss = 0.129598
grad_step = 000024, loss = 0.122712
grad_step = 000025, loss = 0.115763
grad_step = 000026, loss = 0.108944
grad_step = 000027, loss = 0.102288
grad_step = 000028, loss = 0.095953
grad_step = 000029, loss = 0.090148
grad_step = 000030, loss = 0.084640
grad_step = 000031, loss = 0.079255
grad_step = 000032, loss = 0.074015
grad_step = 000033, loss = 0.069301
grad_step = 000034, loss = 0.065030
grad_step = 000035, loss = 0.060637
grad_step = 000036, loss = 0.056325
grad_step = 000037, loss = 0.052535
grad_step = 000038, loss = 0.049083
grad_step = 000039, loss = 0.045664
grad_step = 000040, loss = 0.042351
grad_step = 000041, loss = 0.039291
grad_step = 000042, loss = 0.036499
grad_step = 000043, loss = 0.034007
grad_step = 000044, loss = 0.031708
grad_step = 000045, loss = 0.029413
grad_step = 000046, loss = 0.027246
grad_step = 000047, loss = 0.025282
grad_step = 000048, loss = 0.023397
grad_step = 000049, loss = 0.021613
grad_step = 000050, loss = 0.019984
grad_step = 000051, loss = 0.018445
grad_step = 000052, loss = 0.017018
grad_step = 000053, loss = 0.015721
grad_step = 000054, loss = 0.014499
grad_step = 000055, loss = 0.013383
grad_step = 000056, loss = 0.012347
grad_step = 000057, loss = 0.011353
grad_step = 000058, loss = 0.010452
grad_step = 000059, loss = 0.009625
grad_step = 000060, loss = 0.008856
grad_step = 000061, loss = 0.008174
grad_step = 000062, loss = 0.007534
grad_step = 000063, loss = 0.006933
grad_step = 000064, loss = 0.006398
grad_step = 000065, loss = 0.005914
grad_step = 000066, loss = 0.005479
grad_step = 000067, loss = 0.005076
grad_step = 000068, loss = 0.004707
grad_step = 000069, loss = 0.004388
grad_step = 000070, loss = 0.004096
grad_step = 000071, loss = 0.003840
grad_step = 000072, loss = 0.003618
grad_step = 000073, loss = 0.003410
grad_step = 000074, loss = 0.003228
grad_step = 000075, loss = 0.003070
grad_step = 000076, loss = 0.002935
grad_step = 000077, loss = 0.002815
grad_step = 000078, loss = 0.002707
grad_step = 000079, loss = 0.002621
grad_step = 000080, loss = 0.002546
grad_step = 000081, loss = 0.002482
grad_step = 000082, loss = 0.002432
grad_step = 000083, loss = 0.002387
grad_step = 000084, loss = 0.002350
grad_step = 000085, loss = 0.002320
grad_step = 000086, loss = 0.002297
grad_step = 000087, loss = 0.002275
grad_step = 000088, loss = 0.002257
grad_step = 000089, loss = 0.002245
grad_step = 000090, loss = 0.002233
grad_step = 000091, loss = 0.002224
grad_step = 000092, loss = 0.002216
grad_step = 000093, loss = 0.002208
grad_step = 000094, loss = 0.002201
grad_step = 000095, loss = 0.002196
grad_step = 000096, loss = 0.002191
grad_step = 000097, loss = 0.002185
grad_step = 000098, loss = 0.002180
grad_step = 000099, loss = 0.002175
grad_step = 000100, loss = 0.002170
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002165
grad_step = 000102, loss = 0.002159
grad_step = 000103, loss = 0.002154
grad_step = 000104, loss = 0.002149
grad_step = 000105, loss = 0.002144
grad_step = 000106, loss = 0.002138
grad_step = 000107, loss = 0.002133
grad_step = 000108, loss = 0.002127
grad_step = 000109, loss = 0.002121
grad_step = 000110, loss = 0.002115
grad_step = 000111, loss = 0.002109
grad_step = 000112, loss = 0.002103
grad_step = 000113, loss = 0.002097
grad_step = 000114, loss = 0.002091
grad_step = 000115, loss = 0.002084
grad_step = 000116, loss = 0.002077
grad_step = 000117, loss = 0.002071
grad_step = 000118, loss = 0.002064
grad_step = 000119, loss = 0.002058
grad_step = 000120, loss = 0.002051
grad_step = 000121, loss = 0.002044
grad_step = 000122, loss = 0.002038
grad_step = 000123, loss = 0.002031
grad_step = 000124, loss = 0.002024
grad_step = 000125, loss = 0.002018
grad_step = 000126, loss = 0.002011
grad_step = 000127, loss = 0.002005
grad_step = 000128, loss = 0.001998
grad_step = 000129, loss = 0.001991
grad_step = 000130, loss = 0.001985
grad_step = 000131, loss = 0.001979
grad_step = 000132, loss = 0.001972
grad_step = 000133, loss = 0.001966
grad_step = 000134, loss = 0.001959
grad_step = 000135, loss = 0.001953
grad_step = 000136, loss = 0.001947
grad_step = 000137, loss = 0.001941
grad_step = 000138, loss = 0.001935
grad_step = 000139, loss = 0.001929
grad_step = 000140, loss = 0.001926
grad_step = 000141, loss = 0.001925
grad_step = 000142, loss = 0.001923
grad_step = 000143, loss = 0.001914
grad_step = 000144, loss = 0.001903
grad_step = 000145, loss = 0.001896
grad_step = 000146, loss = 0.001893
grad_step = 000147, loss = 0.001893
grad_step = 000148, loss = 0.001891
grad_step = 000149, loss = 0.001888
grad_step = 000150, loss = 0.001877
grad_step = 000151, loss = 0.001869
grad_step = 000152, loss = 0.001862
grad_step = 000153, loss = 0.001859
grad_step = 000154, loss = 0.001858
grad_step = 000155, loss = 0.001859
grad_step = 000156, loss = 0.001864
grad_step = 000157, loss = 0.001856
grad_step = 000158, loss = 0.001850
grad_step = 000159, loss = 0.001836
grad_step = 000160, loss = 0.001828
grad_step = 000161, loss = 0.001828
grad_step = 000162, loss = 0.001827
grad_step = 000163, loss = 0.001822
grad_step = 000164, loss = 0.001812
grad_step = 000165, loss = 0.001805
grad_step = 000166, loss = 0.001803
grad_step = 000167, loss = 0.001802
grad_step = 000168, loss = 0.001802
grad_step = 000169, loss = 0.001797
grad_step = 000170, loss = 0.001794
grad_step = 000171, loss = 0.001783
grad_step = 000172, loss = 0.001774
grad_step = 000173, loss = 0.001768
grad_step = 000174, loss = 0.001765
grad_step = 000175, loss = 0.001761
grad_step = 000176, loss = 0.001756
grad_step = 000177, loss = 0.001751
grad_step = 000178, loss = 0.001746
grad_step = 000179, loss = 0.001739
grad_step = 000180, loss = 0.001733
grad_step = 000181, loss = 0.001728
grad_step = 000182, loss = 0.001725
grad_step = 000183, loss = 0.001721
grad_step = 000184, loss = 0.001716
grad_step = 000185, loss = 0.001715
grad_step = 000186, loss = 0.001726
grad_step = 000187, loss = 0.001745
grad_step = 000188, loss = 0.001827
grad_step = 000189, loss = 0.001890
grad_step = 000190, loss = 0.001859
grad_step = 000191, loss = 0.001744
grad_step = 000192, loss = 0.001740
grad_step = 000193, loss = 0.001777
grad_step = 000194, loss = 0.001758
grad_step = 000195, loss = 0.001732
grad_step = 000196, loss = 0.001703
grad_step = 000197, loss = 0.001710
grad_step = 000198, loss = 0.001739
grad_step = 000199, loss = 0.001695
grad_step = 000200, loss = 0.001662
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001708
grad_step = 000202, loss = 0.001707
grad_step = 000203, loss = 0.001655
grad_step = 000204, loss = 0.001654
grad_step = 000205, loss = 0.001676
grad_step = 000206, loss = 0.001668
grad_step = 000207, loss = 0.001644
grad_step = 000208, loss = 0.001647
grad_step = 000209, loss = 0.001651
grad_step = 000210, loss = 0.001640
grad_step = 000211, loss = 0.001626
grad_step = 000212, loss = 0.001628
grad_step = 000213, loss = 0.001642
grad_step = 000214, loss = 0.001637
grad_step = 000215, loss = 0.001622
grad_step = 000216, loss = 0.001624
grad_step = 000217, loss = 0.001630
grad_step = 000218, loss = 0.001615
grad_step = 000219, loss = 0.001612
grad_step = 000220, loss = 0.001611
grad_step = 000221, loss = 0.001602
grad_step = 000222, loss = 0.001607
grad_step = 000223, loss = 0.001607
grad_step = 000224, loss = 0.001596
grad_step = 000225, loss = 0.001594
grad_step = 000226, loss = 0.001600
grad_step = 000227, loss = 0.001598
grad_step = 000228, loss = 0.001605
grad_step = 000229, loss = 0.001639
grad_step = 000230, loss = 0.001727
grad_step = 000231, loss = 0.001865
grad_step = 000232, loss = 0.002074
grad_step = 000233, loss = 0.001945
grad_step = 000234, loss = 0.001768
grad_step = 000235, loss = 0.001691
grad_step = 000236, loss = 0.001753
grad_step = 000237, loss = 0.001773
grad_step = 000238, loss = 0.001719
grad_step = 000239, loss = 0.001640
grad_step = 000240, loss = 0.001732
grad_step = 000241, loss = 0.001744
grad_step = 000242, loss = 0.001595
grad_step = 000243, loss = 0.001681
grad_step = 000244, loss = 0.001752
grad_step = 000245, loss = 0.001663
grad_step = 000246, loss = 0.001655
grad_step = 000247, loss = 0.001703
grad_step = 000248, loss = 0.001688
grad_step = 000249, loss = 0.001660
grad_step = 000250, loss = 0.001659
grad_step = 000251, loss = 0.001649
grad_step = 000252, loss = 0.001617
grad_step = 000253, loss = 0.001674
grad_step = 000254, loss = 0.001621
grad_step = 000255, loss = 0.001594
grad_step = 000256, loss = 0.001683
grad_step = 000257, loss = 0.001580
grad_step = 000258, loss = 0.001600
grad_step = 000259, loss = 0.001606
grad_step = 000260, loss = 0.001601
grad_step = 000261, loss = 0.001583
grad_step = 000262, loss = 0.001572
grad_step = 000263, loss = 0.001600
grad_step = 000264, loss = 0.001557
grad_step = 000265, loss = 0.001575
grad_step = 000266, loss = 0.001561
grad_step = 000267, loss = 0.001553
grad_step = 000268, loss = 0.001565
grad_step = 000269, loss = 0.001548
grad_step = 000270, loss = 0.001554
grad_step = 000271, loss = 0.001550
grad_step = 000272, loss = 0.001537
grad_step = 000273, loss = 0.001552
grad_step = 000274, loss = 0.001545
grad_step = 000275, loss = 0.001529
grad_step = 000276, loss = 0.001540
grad_step = 000277, loss = 0.001534
grad_step = 000278, loss = 0.001529
grad_step = 000279, loss = 0.001533
grad_step = 000280, loss = 0.001525
grad_step = 000281, loss = 0.001523
grad_step = 000282, loss = 0.001526
grad_step = 000283, loss = 0.001518
grad_step = 000284, loss = 0.001519
grad_step = 000285, loss = 0.001521
grad_step = 000286, loss = 0.001514
grad_step = 000287, loss = 0.001512
grad_step = 000288, loss = 0.001513
grad_step = 000289, loss = 0.001509
grad_step = 000290, loss = 0.001509
grad_step = 000291, loss = 0.001508
grad_step = 000292, loss = 0.001503
grad_step = 000293, loss = 0.001503
grad_step = 000294, loss = 0.001502
grad_step = 000295, loss = 0.001499
grad_step = 000296, loss = 0.001498
grad_step = 000297, loss = 0.001497
grad_step = 000298, loss = 0.001494
grad_step = 000299, loss = 0.001493
grad_step = 000300, loss = 0.001493
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001490
grad_step = 000302, loss = 0.001489
grad_step = 000303, loss = 0.001488
grad_step = 000304, loss = 0.001486
grad_step = 000305, loss = 0.001485
grad_step = 000306, loss = 0.001484
grad_step = 000307, loss = 0.001483
grad_step = 000308, loss = 0.001482
grad_step = 000309, loss = 0.001483
grad_step = 000310, loss = 0.001485
grad_step = 000311, loss = 0.001489
grad_step = 000312, loss = 0.001500
grad_step = 000313, loss = 0.001522
grad_step = 000314, loss = 0.001559
grad_step = 000315, loss = 0.001608
grad_step = 000316, loss = 0.001638
grad_step = 000317, loss = 0.001624
grad_step = 000318, loss = 0.001552
grad_step = 000319, loss = 0.001526
grad_step = 000320, loss = 0.001549
grad_step = 000321, loss = 0.001506
grad_step = 000322, loss = 0.001472
grad_step = 000323, loss = 0.001495
grad_step = 000324, loss = 0.001530
grad_step = 000325, loss = 0.001564
grad_step = 000326, loss = 0.001522
grad_step = 000327, loss = 0.001494
grad_step = 000328, loss = 0.001477
grad_step = 000329, loss = 0.001475
grad_step = 000330, loss = 0.001483
grad_step = 000331, loss = 0.001502
grad_step = 000332, loss = 0.001515
grad_step = 000333, loss = 0.001483
grad_step = 000334, loss = 0.001459
grad_step = 000335, loss = 0.001454
grad_step = 000336, loss = 0.001455
grad_step = 000337, loss = 0.001467
grad_step = 000338, loss = 0.001458
grad_step = 000339, loss = 0.001442
grad_step = 000340, loss = 0.001438
grad_step = 000341, loss = 0.001445
grad_step = 000342, loss = 0.001445
grad_step = 000343, loss = 0.001449
grad_step = 000344, loss = 0.001459
grad_step = 000345, loss = 0.001461
grad_step = 000346, loss = 0.001459
grad_step = 000347, loss = 0.001459
grad_step = 000348, loss = 0.001458
grad_step = 000349, loss = 0.001449
grad_step = 000350, loss = 0.001446
grad_step = 000351, loss = 0.001448
grad_step = 000352, loss = 0.001456
grad_step = 000353, loss = 0.001467
grad_step = 000354, loss = 0.001496
grad_step = 000355, loss = 0.001525
grad_step = 000356, loss = 0.001564
grad_step = 000357, loss = 0.001567
grad_step = 000358, loss = 0.001553
grad_step = 000359, loss = 0.001488
grad_step = 000360, loss = 0.001437
grad_step = 000361, loss = 0.001413
grad_step = 000362, loss = 0.001425
grad_step = 000363, loss = 0.001456
grad_step = 000364, loss = 0.001468
grad_step = 000365, loss = 0.001460
grad_step = 000366, loss = 0.001429
grad_step = 000367, loss = 0.001407
grad_step = 000368, loss = 0.001403
grad_step = 000369, loss = 0.001416
grad_step = 000370, loss = 0.001436
grad_step = 000371, loss = 0.001443
grad_step = 000372, loss = 0.001446
grad_step = 000373, loss = 0.001434
grad_step = 000374, loss = 0.001421
grad_step = 000375, loss = 0.001405
grad_step = 000376, loss = 0.001396
grad_step = 000377, loss = 0.001393
grad_step = 000378, loss = 0.001399
grad_step = 000379, loss = 0.001412
grad_step = 000380, loss = 0.001431
grad_step = 000381, loss = 0.001457
grad_step = 000382, loss = 0.001494
grad_step = 000383, loss = 0.001529
grad_step = 000384, loss = 0.001558
grad_step = 000385, loss = 0.001531
grad_step = 000386, loss = 0.001488
grad_step = 000387, loss = 0.001452
grad_step = 000388, loss = 0.001478
grad_step = 000389, loss = 0.001505
grad_step = 000390, loss = 0.001495
grad_step = 000391, loss = 0.001415
grad_step = 000392, loss = 0.001373
grad_step = 000393, loss = 0.001396
grad_step = 000394, loss = 0.001427
grad_step = 000395, loss = 0.001420
grad_step = 000396, loss = 0.001392
grad_step = 000397, loss = 0.001391
grad_step = 000398, loss = 0.001402
grad_step = 000399, loss = 0.001393
grad_step = 000400, loss = 0.001369
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001364
grad_step = 000402, loss = 0.001383
grad_step = 000403, loss = 0.001395
grad_step = 000404, loss = 0.001391
grad_step = 000405, loss = 0.001388
grad_step = 000406, loss = 0.001406
grad_step = 000407, loss = 0.001421
grad_step = 000408, loss = 0.001430
grad_step = 000409, loss = 0.001407
grad_step = 000410, loss = 0.001387
grad_step = 000411, loss = 0.001372
grad_step = 000412, loss = 0.001365
grad_step = 000413, loss = 0.001357
grad_step = 000414, loss = 0.001354
grad_step = 000415, loss = 0.001361
grad_step = 000416, loss = 0.001371
grad_step = 000417, loss = 0.001376
grad_step = 000418, loss = 0.001368
grad_step = 000419, loss = 0.001358
grad_step = 000420, loss = 0.001350
grad_step = 000421, loss = 0.001346
grad_step = 000422, loss = 0.001343
grad_step = 000423, loss = 0.001341
grad_step = 000424, loss = 0.001340
grad_step = 000425, loss = 0.001342
grad_step = 000426, loss = 0.001348
grad_step = 000427, loss = 0.001355
grad_step = 000428, loss = 0.001365
grad_step = 000429, loss = 0.001372
grad_step = 000430, loss = 0.001387
grad_step = 000431, loss = 0.001395
grad_step = 000432, loss = 0.001411
grad_step = 000433, loss = 0.001410
grad_step = 000434, loss = 0.001412
grad_step = 000435, loss = 0.001388
grad_step = 000436, loss = 0.001365
grad_step = 000437, loss = 0.001340
grad_step = 000438, loss = 0.001328
grad_step = 000439, loss = 0.001330
grad_step = 000440, loss = 0.001340
grad_step = 000441, loss = 0.001350
grad_step = 000442, loss = 0.001351
grad_step = 000443, loss = 0.001350
grad_step = 000444, loss = 0.001341
grad_step = 000445, loss = 0.001332
grad_step = 000446, loss = 0.001323
grad_step = 000447, loss = 0.001318
grad_step = 000448, loss = 0.001316
grad_step = 000449, loss = 0.001317
grad_step = 000450, loss = 0.001321
grad_step = 000451, loss = 0.001328
grad_step = 000452, loss = 0.001339
grad_step = 000453, loss = 0.001354
grad_step = 000454, loss = 0.001380
grad_step = 000455, loss = 0.001403
grad_step = 000456, loss = 0.001444
grad_step = 000457, loss = 0.001454
grad_step = 000458, loss = 0.001469
grad_step = 000459, loss = 0.001420
grad_step = 000460, loss = 0.001370
grad_step = 000461, loss = 0.001319
grad_step = 000462, loss = 0.001309
grad_step = 000463, loss = 0.001335
grad_step = 000464, loss = 0.001358
grad_step = 000465, loss = 0.001357
grad_step = 000466, loss = 0.001327
grad_step = 000467, loss = 0.001304
grad_step = 000468, loss = 0.001302
grad_step = 000469, loss = 0.001315
grad_step = 000470, loss = 0.001332
grad_step = 000471, loss = 0.001336
grad_step = 000472, loss = 0.001334
grad_step = 000473, loss = 0.001320
grad_step = 000474, loss = 0.001313
grad_step = 000475, loss = 0.001311
grad_step = 000476, loss = 0.001317
grad_step = 000477, loss = 0.001331
grad_step = 000478, loss = 0.001352
grad_step = 000479, loss = 0.001377
grad_step = 000480, loss = 0.001407
grad_step = 000481, loss = 0.001431
grad_step = 000482, loss = 0.001448
grad_step = 000483, loss = 0.001435
grad_step = 000484, loss = 0.001422
grad_step = 000485, loss = 0.001382
grad_step = 000486, loss = 0.001370
grad_step = 000487, loss = 0.001355
grad_step = 000488, loss = 0.001350
grad_step = 000489, loss = 0.001330
grad_step = 000490, loss = 0.001309
grad_step = 000491, loss = 0.001299
grad_step = 000492, loss = 0.001304
grad_step = 000493, loss = 0.001320
grad_step = 000494, loss = 0.001322
grad_step = 000495, loss = 0.001311
grad_step = 000496, loss = 0.001289
grad_step = 000497, loss = 0.001276
grad_step = 000498, loss = 0.001278
grad_step = 000499, loss = 0.001291
grad_step = 000500, loss = 0.001304
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001303
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

  date_run                              2020-05-15 15:13:35.053118
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.228234
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 15:13:35.060510
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.123573
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 15:13:35.071223
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.135132
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 15:13:35.077305
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.877734
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
0   2020-05-15 15:12:59.832511  ...    mean_absolute_error
1   2020-05-15 15:12:59.837601  ...     mean_squared_error
2   2020-05-15 15:12:59.841234  ...  median_absolute_error
3   2020-05-15 15:12:59.844976  ...               r2_score
4   2020-05-15 15:13:09.664553  ...    mean_absolute_error
5   2020-05-15 15:13:09.669237  ...     mean_squared_error
6   2020-05-15 15:13:09.673623  ...  median_absolute_error
7   2020-05-15 15:13:09.677444  ...               r2_score
8   2020-05-15 15:13:35.053118  ...    mean_absolute_error
9   2020-05-15 15:13:35.060510  ...     mean_squared_error
10  2020-05-15 15:13:35.071223  ...  median_absolute_error
11  2020-05-15 15:13:35.077305  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc45c0ee898> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 34%|███▍      | 3416064/9912422 [00:00<00:00, 34158636.56it/s]9920512it [00:00, 32086572.93it/s]                             
0it [00:00, ?it/s]32768it [00:00, 1577835.67it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1015460.26it/s]1654784it [00:00, 12463353.63it/s]                           
0it [00:00, ?it/s]8192it [00:00, 219008.07it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc40ea9ee10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc45c0a6e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc40ea9ee10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc45c0a6e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc40b85f470> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc45c0a6e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc40ea9ee10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc45c0a6e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc40b85f470> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fc45c0a6e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f16b962a1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=3a0e245e2b92526968b6ce284b2487d2324724503c25019fc0eaf86da7de8a22
  Stored in directory: /tmp/pip-ephem-wheel-cache-nz5xb5cs/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f16af795080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1187840/17464789 [=>............................] - ETA: 0s
 4702208/17464789 [=======>......................] - ETA: 0s
11730944/17464789 [===================>..........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 15:15:04.796194: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 15:15:04.801072: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 15:15:04.801252: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c6ee26b8f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 15:15:04.801270: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.7586 - accuracy: 0.4940
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7203 - accuracy: 0.4965
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.8046 - accuracy: 0.4910 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7970 - accuracy: 0.4915
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8445 - accuracy: 0.4884
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.8506 - accuracy: 0.4880
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7959 - accuracy: 0.4916
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.8008 - accuracy: 0.4913
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.8012 - accuracy: 0.4912
10000/25000 [===========>..................] - ETA: 5s - loss: 7.8108 - accuracy: 0.4906
11000/25000 [============>.................] - ETA: 4s - loss: 7.7949 - accuracy: 0.4916
12000/25000 [=============>................] - ETA: 4s - loss: 7.7561 - accuracy: 0.4942
13000/25000 [==============>...............] - ETA: 4s - loss: 7.7339 - accuracy: 0.4956
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7159 - accuracy: 0.4968
15000/25000 [=================>............] - ETA: 3s - loss: 7.6881 - accuracy: 0.4986
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6772 - accuracy: 0.4993
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6756 - accuracy: 0.4994
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6709 - accuracy: 0.4997
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6626 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6590 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6659 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6680 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6800 - accuracy: 0.4991
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6826 - accuracy: 0.4990
25000/25000 [==============================] - 10s 406us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 15:15:22.960785
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 15:15:22.960785  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:33:27, 11.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:19:28, 15.6kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:46:51, 22.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:33:17, 31.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:16:30, 45.2kB/s].vector_cache/glove.6B.zip:   1%|          | 9.27M/862M [00:01<3:40:12, 64.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.6M/862M [00:01<2:33:15, 92.2kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:01<1:47:01, 131kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.3M/862M [00:01<1:14:30, 188kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:02<52:21, 266kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.0M/862M [00:02<36:32, 379kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.7M/862M [00:02<25:31, 540kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.5M/862M [00:02<17:53, 766kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:02<12:33, 1.09MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.6M/862M [00:02<08:51, 1.53MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:02<06:33, 2.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<06:29, 2.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:05<06:22, 2.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:05<04:55, 2.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<05:59, 2.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<05:36, 2.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:07<04:16, 3.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<05:58, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:09<06:55, 1.92MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:09<05:31, 2.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<06:00, 2.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<05:34, 2.37MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:11<04:12, 3.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<06:00, 2.19MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<05:33, 2.37MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.6M/862M [00:13<04:13, 3.11MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<06:02, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<05:34, 2.35MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:15<04:13, 3.09MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<06:01, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<05:33, 2.34MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:17<04:12, 3.09MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<06:00, 2.16MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<05:31, 2.34MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.0M/862M [00:18<04:11, 3.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<06:00, 2.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<06:51, 1.88MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:20<05:21, 2.40MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.3M/862M [00:21<03:55, 3.27MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<09:16, 1.38MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<07:49, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<05:47, 2.21MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<07:01, 1.81MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<06:13, 2.05MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:24<04:40, 2.72MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:15, 2.02MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:40, 2.23MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<04:17, 2.95MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:59, 2.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:29, 2.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<04:06, 3.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:51, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:23, 2.33MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:05, 3.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:48, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:20, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:02, 3.07MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:45, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:18, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:01, 3.08MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:43, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:33, 1.88MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:13, 2.36MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:37, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:12, 2.36MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<03:56, 3.10MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:37, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:26, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:03, 2.41MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:40<03:40, 3.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<10:06, 1.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<08:18, 1.46MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<06:04, 1.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<07:04, 1.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<07:19, 1.65MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:44, 2.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<04:08, 2.90MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<11:34:25, 17.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<8:07:06, 24.6kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<5:40:32, 35.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<4:00:27, 49.6kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<2:49:15, 70.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<1:58:40, 100kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<1:22:54, 143kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<1:45:42, 112kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<1:16:24, 155kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<54:01, 219kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<39:34, 298kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<28:54, 408kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<20:29, 574kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<17:02, 688kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<14:19, 818kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<10:32, 1.11MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<07:29, 1.56MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<14:38, 795kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<11:27, 1.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<08:16, 1.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<08:29, 1.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<07:09, 1.62MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:17, 2.18MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<06:25, 1.79MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:28, 2.10MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<04:05, 2.81MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<03:00, 3.81MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<45:18, 253kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<34:02, 336kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<24:19, 470kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<17:05, 666kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<23:30, 484kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<17:37, 645kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<12:36, 900kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<11:26, 988kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<10:20, 1.09MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:43, 1.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<05:33, 2.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:55, 1.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<07:24, 1.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<05:27, 2.05MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:24, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:46, 1.65MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:14, 2.12MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<03:48, 2.92MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<09:16, 1.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<07:38, 1.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<05:34, 1.98MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:29, 1.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:40, 1.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<04:11, 2.62MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:33, 1.97MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<04:59, 2.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:43, 2.93MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:11, 2.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:33, 2.39MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<03:29, 3.11MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<02:34, 4.22MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<45:27, 238kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<32:54, 329kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<23:15, 464kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<18:46, 573kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<15:21, 700kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<11:17, 952kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<07:59, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<10:19:30, 17.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<7:14:31, 24.6kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<5:03:42, 35.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<3:34:20, 49.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<2:31:01, 70.3kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<1:45:43, 100kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<1:16:15, 138kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<54:25, 194kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<38:16, 275kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<29:10, 359kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<21:29, 487kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<15:16, 684kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<13:07, 794kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<10:15, 1.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<07:25, 1.40MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<07:38, 1.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<06:24, 1.61MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:41, 2.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<05:43, 1.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<06:08, 1.67MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:45, 2.16MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<03:26, 2.97MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<10:13, 998kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<08:12, 1.24MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:59, 1.70MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<06:33, 1.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:37, 1.80MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:08, 2.44MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<05:15, 1.91MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:42, 2.14MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:32, 2.83MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<04:50, 2.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:26, 1.84MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:19, 2.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<03:06, 3.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<9:36:51, 17.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<6:44:33, 24.5kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<4:42:42, 35.0kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<3:19:29, 49.5kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<2:20:34, 70.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<1:38:23, 99.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<1:10:56, 138kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<50:37, 193kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<35:35, 274kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<27:07, 359kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<19:58, 487kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<14:11, 683kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<12:11, 792kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<09:30, 1.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<06:51, 1.41MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<07:03, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:55, 1.62MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<04:22, 2.18MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<05:18, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:41, 2.03MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:30, 2.70MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:41, 2.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:15, 2.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:12, 2.94MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:27, 2.11MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:04, 2.30MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:05, 3.03MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:21, 2.13MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:58, 1.88MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:53, 2.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<02:50, 3.25MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<06:05, 1.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:14, 1.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:51, 2.39MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:50, 1.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:19, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:15, 2.81MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:25, 2.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:01, 2.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:01, 3.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:15, 2.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:54, 2.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<02:57, 3.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:11, 2.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:50, 2.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<02:54, 3.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:08, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:48, 2.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<02:53, 3.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<04:06, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<03:47, 2.33MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<02:50, 3.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:04, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<03:44, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<02:50, 3.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:02, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:20<03:42, 2.35MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<02:48, 3.09MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<03:59, 2.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<03:41, 2.34MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:47, 3.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<03:58, 2.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<03:39, 2.34MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<02:44, 3.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<03:54, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:33, 1.87MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:37, 2.34MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<02:37, 3.20MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<1:01:57, 136kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<44:13, 190kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<31:02, 270kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<23:35, 354kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<18:13, 458kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<13:07, 635kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<10:29, 790kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<08:11, 1.01MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<05:53, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<06:02, 1.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<05:54, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:30, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<03:14, 2.52MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<06:47, 1.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<05:35, 1.46MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<04:06, 1.97MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<04:45, 1.70MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:09, 1.94MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:05, 2.61MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:03, 1.97MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:40, 2.18MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<02:44, 2.91MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:46, 2.10MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:27, 2.29MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<02:37, 3.02MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:40, 2.14MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<04:10, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<03:17, 2.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<02:22, 3.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<13:43, 569kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<10:25, 749kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<07:28, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<07:01, 1.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:42, 1.35MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<04:10, 1.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:43, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:06, 1.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:03, 2.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:55, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:31, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<02:39, 2.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:37, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:18, 2.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:30, 2.99MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:31, 2.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:59, 1.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:06, 2.39MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<02:15, 3.28MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<06:50, 1.08MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<05:34, 1.33MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<04:04, 1.80MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:33, 1.61MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:42, 1.56MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:36, 2.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<02:36, 2.80MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<07:06, 1.02MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<05:42, 1.27MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:10, 1.73MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:35, 1.57MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:56, 1.82MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:54, 2.46MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:42, 1.92MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:19, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:28, 2.87MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:24, 2.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:49, 1.84MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:02, 2.32MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<02:11, 3.20MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<52:03, 134kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<37:08, 188kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<26:04, 267kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<19:46, 350kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<15:15, 453kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<10:57, 630kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<07:43, 888kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<08:39, 791kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<06:46, 1.01MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<04:52, 1.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:59, 1.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:11, 1.61MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:06, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:44, 1.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:59, 1.68MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:08, 2.13MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:17<02:15, 2.95MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<6:25:32, 17.2kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<4:30:17, 24.6kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<3:08:38, 35.0kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<2:12:52, 49.5kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<1:33:34, 70.2kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<1:05:23, 100kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<47:04, 138kB/s]  .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<34:15, 190kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<24:13, 268kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<16:55, 381kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<16:38, 387kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<12:18, 523kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<08:44, 732kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<07:34, 840kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<05:57, 1.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:17, 1.48MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<04:29, 1.40MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<04:25, 1.42MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:23, 1.86MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:25, 2.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<09:10, 679kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<07:04, 880kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<05:05, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:59, 1.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<04:45, 1.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:38, 1.69MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<02:35, 2.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<45:52, 133kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<32:42, 186kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<22:57, 264kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<17:23, 347kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<13:23, 450kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<09:39, 622kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<07:40, 776kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<05:59, 994kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:19, 1.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:23, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:41, 1.60MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:43, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:15, 1.79MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:46, 2.10MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:05, 2.76MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<01:31, 3.77MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<24:16, 237kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<18:09, 317kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<12:58, 442kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<09:56, 572kB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<07:31, 754kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<05:23, 1.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<05:03, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<04:06, 1.36MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<03:00, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:24, 1.63MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:57, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:10, 2.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:49, 1.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:32, 2.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<01:54, 2.86MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:36, 2.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:17, 2.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<01:42, 3.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:54<01:15, 4.22MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<21:00, 254kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<15:14, 350kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<10:44, 494kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<08:43, 605kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<06:34, 802kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<04:42, 1.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<03:20, 1.56MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<21:56, 237kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<16:24, 317kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<11:43, 442kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<08:57, 573kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<06:47, 755kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<04:51, 1.05MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<04:34, 1.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:42, 1.36MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:42, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:03, 1.63MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:39, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:58, 2.51MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:32, 1.94MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:16, 2.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:42, 2.86MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:20, 2.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:03, 2.37MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:32, 3.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<01:07, 4.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<18:52, 254kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<13:36, 352kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<09:34, 497kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<06:43, 703kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<23:44, 199kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<17:34, 269kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<12:30, 376kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<09:24, 494kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<07:03, 658kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<05:02, 918kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<04:34, 1.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<04:08, 1.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:06, 1.47MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:18<02:12, 2.05MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<05:11, 871kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<04:06, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:58, 1.51MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<03:06, 1.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:37, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:56, 2.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<02:23, 1.83MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:34, 1.70MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<01:58, 2.20MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:26, 2.99MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:34, 1.67MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:15, 1.91MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:39, 2.57MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:09, 1.97MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<02:23, 1.78MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:51, 2.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:20, 3.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<04:04, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:13, 1.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:19, 1.78MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<01:40, 2.46MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<17:40, 232kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<13:12, 311kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<09:23, 435kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<06:34, 617kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<06:49, 591kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<05:11, 776kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<03:41, 1.08MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:29, 1.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:51, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:05, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:22, 1.65MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:03, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:31, 2.53MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:58, 1.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:45, 2.17MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:19, 2.87MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:48, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:38, 2.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:13, 3.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:44, 2.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:35, 2.32MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:43<01:11, 3.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:41, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:55, 1.89MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:30, 2.40MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:05, 3.29MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:56, 1.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:25, 1.46MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:46, 1.98MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:03, 1.70MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:09, 1.62MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:40, 2.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:42, 2.00MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:29, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:08, 2.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<00:49, 4.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<09:01, 371kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<06:59, 479kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<05:03, 660kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<04:00, 818kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<03:08, 1.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:15, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<02:18, 1.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:56, 1.65MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:25, 2.22MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:44, 1.81MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:29, 2.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:05, 2.84MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<00:48, 3.83MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<13:02, 236kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<09:25, 326kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<06:36, 460kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<05:17, 569kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<04:00, 750kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:51, 1.04MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:40, 1.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:10, 1.35MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:34, 1.84MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:46, 1.62MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:31, 1.87MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:07, 2.53MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:26, 1.94MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:35, 1.77MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:14, 2.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<00:53, 3.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<2:38:13, 17.3kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<1:50:46, 24.6kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<1:16:52, 35.2kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<53:43, 49.6kB/s]  .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<38:06, 69.9kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<26:39, 99.4kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<18:24, 142kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<14:44, 176kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<10:33, 245kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<07:22, 348kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<05:41, 445kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<04:13, 597kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<02:59, 834kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<02:38, 929kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<02:06, 1.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:30, 1.60MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:36, 1.48MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:21, 1.75MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:00, 2.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:14, 1.87MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:06, 2.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:49, 2.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:06, 2.05MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:59, 2.25MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:44, 3.00MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:01, 2.12MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:56, 2.31MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<00:42, 3.05MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:59, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:54, 2.33MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<00:40, 3.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:57, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:52, 2.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:39, 3.07MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:55, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<01:02, 1.89MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:49, 2.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:52, 2.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:48, 2.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:36, 3.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:50, 2.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:46, 2.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:35, 3.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:49, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:45, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:33, 3.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:47, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:43, 2.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:32, 3.07MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:45, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:41, 2.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:31, 3.07MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:43, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:38, 2.44MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:29, 3.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:44<00:21, 4.29MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<05:55, 253kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<04:26, 337kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<03:09, 470kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<02:21, 605kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:47, 794kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:16, 1.10MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<01:11, 1.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:57, 1.41MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:41, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:46, 1.66MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:40, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:29, 2.56MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:37, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:33, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:24, 2.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:33, 2.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:30, 2.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:22, 3.03MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:30, 2.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:27, 2.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:20, 3.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:28, 2.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:26, 2.33MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:19, 3.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:26, 2.15MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:24, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:17, 3.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:24, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:22, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:16, 3.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:22, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:20, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:15, 3.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:20, 2.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:18, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:13, 3.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:18, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:17, 2.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:10<00:12, 3.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:16, 2.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:15, 2.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:11, 3.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:14, 2.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:13, 2.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:09, 3.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:13, 2.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:11, 2.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:08, 3.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:10, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:07, 3.07MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:10, 1.89MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:04, 3.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<15:21, 17.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<10:30, 24.7kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<06:37, 35.2kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<03:58, 49.7kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<02:46, 70.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<01:49, 99.5kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<01:03, 142kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:40, 189kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:27, 262kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:15, 371kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:07, 471kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:05, 630kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:01, 879kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 721/400000 [00:00<00:55, 7208.96it/s]  0%|          | 1480/400000 [00:00<00:54, 7318.10it/s]  1%|          | 2206/400000 [00:00<00:54, 7298.47it/s]  1%|          | 2932/400000 [00:00<00:54, 7283.96it/s]  1%|          | 3681/400000 [00:00<00:53, 7344.33it/s]  1%|          | 4393/400000 [00:00<00:54, 7273.34it/s]  1%|▏         | 5102/400000 [00:00<00:54, 7215.06it/s]  1%|▏         | 5815/400000 [00:00<00:54, 7187.34it/s]  2%|▏         | 6509/400000 [00:00<00:55, 7109.14it/s]  2%|▏         | 7240/400000 [00:01<00:54, 7166.34it/s]  2%|▏         | 7941/400000 [00:01<00:55, 7118.53it/s]  2%|▏         | 8645/400000 [00:01<00:55, 7092.73it/s]  2%|▏         | 9357/400000 [00:01<00:55, 7099.00it/s]  3%|▎         | 10061/400000 [00:01<00:57, 6755.32it/s]  3%|▎         | 10764/400000 [00:01<00:56, 6834.91it/s]  3%|▎         | 11487/400000 [00:01<00:55, 6945.57it/s]  3%|▎         | 12226/400000 [00:01<00:54, 7069.66it/s]  3%|▎         | 12954/400000 [00:01<00:54, 7130.78it/s]  3%|▎         | 13668/400000 [00:01<00:54, 7133.37it/s]  4%|▎         | 14392/400000 [00:02<00:53, 7162.59it/s]  4%|▍         | 15124/400000 [00:02<00:53, 7208.73it/s]  4%|▍         | 15868/400000 [00:02<00:52, 7272.83it/s]  4%|▍         | 16600/400000 [00:02<00:52, 7286.07it/s]  4%|▍         | 17341/400000 [00:02<00:52, 7320.93it/s]  5%|▍         | 18096/400000 [00:02<00:51, 7387.11it/s]  5%|▍         | 18835/400000 [00:02<00:52, 7277.92it/s]  5%|▍         | 19564/400000 [00:02<00:52, 7247.52it/s]  5%|▌         | 20290/400000 [00:02<00:53, 7113.11it/s]  5%|▌         | 21003/400000 [00:02<00:54, 6929.21it/s]  5%|▌         | 21735/400000 [00:03<00:53, 7040.73it/s]  6%|▌         | 22441/400000 [00:03<00:54, 6924.85it/s]  6%|▌         | 23178/400000 [00:03<00:53, 7050.86it/s]  6%|▌         | 23885/400000 [00:03<00:54, 6964.53it/s]  6%|▌         | 24586/400000 [00:03<00:53, 6975.62it/s]  6%|▋         | 25295/400000 [00:03<00:53, 7007.43it/s]  6%|▋         | 25997/400000 [00:03<00:53, 7003.96it/s]  7%|▋         | 26737/400000 [00:03<00:52, 7117.98it/s]  7%|▋         | 27450/400000 [00:03<00:52, 7056.51it/s]  7%|▋         | 28157/400000 [00:03<00:53, 7007.61it/s]  7%|▋         | 28859/400000 [00:04<00:53, 6993.39it/s]  7%|▋         | 29559/400000 [00:04<00:54, 6790.21it/s]  8%|▊         | 30282/400000 [00:04<00:53, 6914.72it/s]  8%|▊         | 30976/400000 [00:04<00:54, 6792.30it/s]  8%|▊         | 31675/400000 [00:04<00:53, 6849.99it/s]  8%|▊         | 32391/400000 [00:04<00:52, 6937.25it/s]  8%|▊         | 33119/400000 [00:04<00:52, 7035.51it/s]  8%|▊         | 33853/400000 [00:04<00:51, 7123.35it/s]  9%|▊         | 34578/400000 [00:04<00:51, 7158.35it/s]  9%|▉         | 35295/400000 [00:04<00:51, 7134.47it/s]  9%|▉         | 36010/400000 [00:05<00:51, 7114.24it/s]  9%|▉         | 36771/400000 [00:05<00:50, 7254.99it/s]  9%|▉         | 37511/400000 [00:05<00:49, 7296.03it/s] 10%|▉         | 38242/400000 [00:05<00:50, 7210.55it/s] 10%|▉         | 38964/400000 [00:05<00:50, 7184.01it/s] 10%|▉         | 39689/400000 [00:05<00:50, 7201.49it/s] 10%|█         | 40424/400000 [00:05<00:49, 7244.33it/s] 10%|█         | 41173/400000 [00:05<00:49, 7314.52it/s] 10%|█         | 41905/400000 [00:05<00:49, 7284.29it/s] 11%|█         | 42634/400000 [00:05<00:49, 7237.89it/s] 11%|█         | 43359/400000 [00:06<00:50, 7053.72it/s] 11%|█         | 44079/400000 [00:06<00:50, 7096.14it/s] 11%|█         | 44793/400000 [00:06<00:49, 7107.41it/s] 11%|█▏        | 45505/400000 [00:06<00:49, 7110.22it/s] 12%|█▏        | 46247/400000 [00:06<00:49, 7198.94it/s] 12%|█▏        | 46995/400000 [00:06<00:48, 7278.28it/s] 12%|█▏        | 47745/400000 [00:06<00:47, 7342.35it/s] 12%|█▏        | 48503/400000 [00:06<00:47, 7409.57it/s] 12%|█▏        | 49245/400000 [00:06<00:47, 7316.99it/s] 12%|█▏        | 49978/400000 [00:07<00:48, 7276.41it/s] 13%|█▎        | 50724/400000 [00:07<00:47, 7329.59it/s] 13%|█▎        | 51458/400000 [00:07<00:49, 6994.17it/s] 13%|█▎        | 52161/400000 [00:07<00:50, 6904.47it/s] 13%|█▎        | 52859/400000 [00:07<00:50, 6924.54it/s] 13%|█▎        | 53595/400000 [00:07<00:49, 7048.48it/s] 14%|█▎        | 54332/400000 [00:07<00:48, 7140.60it/s] 14%|█▍        | 55052/400000 [00:07<00:48, 7158.17it/s] 14%|█▍        | 55772/400000 [00:07<00:48, 7161.41it/s] 14%|█▍        | 56498/400000 [00:07<00:47, 7189.46it/s] 14%|█▍        | 57223/400000 [00:08<00:47, 7205.42it/s] 14%|█▍        | 57944/400000 [00:08<00:47, 7195.50it/s] 15%|█▍        | 58691/400000 [00:08<00:46, 7273.47it/s] 15%|█▍        | 59419/400000 [00:08<00:47, 7194.25it/s] 15%|█▌        | 60139/400000 [00:08<00:47, 7167.66it/s] 15%|█▌        | 60857/400000 [00:08<00:47, 7155.48it/s] 15%|█▌        | 61587/400000 [00:08<00:47, 7197.22it/s] 16%|█▌        | 62307/400000 [00:08<00:46, 7189.05it/s] 16%|█▌        | 63030/400000 [00:08<00:46, 7199.90it/s] 16%|█▌        | 63765/400000 [00:08<00:46, 7242.77it/s] 16%|█▌        | 64490/400000 [00:09<00:47, 7054.36it/s] 16%|█▋        | 65202/400000 [00:09<00:47, 7073.65it/s] 16%|█▋        | 65931/400000 [00:09<00:46, 7134.82it/s] 17%|█▋        | 66665/400000 [00:09<00:46, 7194.93it/s] 17%|█▋        | 67394/400000 [00:09<00:46, 7221.31it/s] 17%|█▋        | 68135/400000 [00:09<00:45, 7276.60it/s] 17%|█▋        | 68877/400000 [00:09<00:45, 7317.19it/s] 17%|█▋        | 69627/400000 [00:09<00:44, 7370.85it/s] 18%|█▊        | 70365/400000 [00:09<00:44, 7344.43it/s] 18%|█▊        | 71100/400000 [00:09<00:45, 7298.33it/s] 18%|█▊        | 71831/400000 [00:10<00:46, 7126.25it/s] 18%|█▊        | 72545/400000 [00:10<00:46, 7112.96it/s] 18%|█▊        | 73271/400000 [00:10<00:45, 7156.25it/s] 18%|█▊        | 73988/400000 [00:10<00:46, 7076.07it/s] 19%|█▊        | 74727/400000 [00:10<00:45, 7166.76it/s] 19%|█▉        | 75448/400000 [00:10<00:45, 7177.34it/s] 19%|█▉        | 76172/400000 [00:10<00:45, 7195.47it/s] 19%|█▉        | 76896/400000 [00:10<00:44, 7204.22it/s] 19%|█▉        | 77617/400000 [00:10<00:45, 7148.12it/s] 20%|█▉        | 78333/400000 [00:10<00:45, 7011.52it/s] 20%|█▉        | 79044/400000 [00:11<00:45, 7040.07it/s] 20%|█▉        | 79749/400000 [00:11<00:46, 6954.93it/s] 20%|██        | 80477/400000 [00:11<00:45, 7047.56it/s] 20%|██        | 81220/400000 [00:11<00:44, 7154.17it/s] 20%|██        | 81937/400000 [00:11<00:44, 7078.94it/s] 21%|██        | 82646/400000 [00:11<00:45, 6926.83it/s] 21%|██        | 83388/400000 [00:11<00:44, 7065.58it/s] 21%|██        | 84135/400000 [00:11<00:43, 7180.14it/s] 21%|██        | 84864/400000 [00:11<00:43, 7212.17it/s] 21%|██▏       | 85587/400000 [00:11<00:43, 7202.11it/s] 22%|██▏       | 86321/400000 [00:12<00:43, 7241.39it/s] 22%|██▏       | 87046/400000 [00:12<00:43, 7213.88it/s] 22%|██▏       | 87768/400000 [00:12<00:43, 7147.45it/s] 22%|██▏       | 88484/400000 [00:12<00:44, 7048.40it/s] 22%|██▏       | 89217/400000 [00:12<00:43, 7127.96it/s] 22%|██▏       | 89986/400000 [00:12<00:42, 7286.26it/s] 23%|██▎       | 90727/400000 [00:12<00:42, 7320.74it/s] 23%|██▎       | 91484/400000 [00:12<00:41, 7393.58it/s] 23%|██▎       | 92225/400000 [00:12<00:41, 7389.84it/s] 23%|██▎       | 92965/400000 [00:12<00:41, 7330.16it/s] 23%|██▎       | 93706/400000 [00:13<00:41, 7353.73it/s] 24%|██▎       | 94443/400000 [00:13<00:41, 7357.51it/s] 24%|██▍       | 95180/400000 [00:13<00:42, 7138.40it/s] 24%|██▍       | 95912/400000 [00:13<00:42, 7191.67it/s] 24%|██▍       | 96640/400000 [00:13<00:42, 7215.22it/s] 24%|██▍       | 97386/400000 [00:13<00:41, 7286.54it/s] 25%|██▍       | 98145/400000 [00:13<00:40, 7373.07it/s] 25%|██▍       | 98884/400000 [00:13<00:40, 7352.04it/s] 25%|██▍       | 99625/400000 [00:13<00:40, 7368.13it/s] 25%|██▌       | 100363/400000 [00:14<00:40, 7346.09it/s] 25%|██▌       | 101098/400000 [00:14<00:41, 7272.56it/s] 25%|██▌       | 101826/400000 [00:14<00:41, 7209.56it/s] 26%|██▌       | 102578/400000 [00:14<00:40, 7299.63it/s] 26%|██▌       | 103309/400000 [00:14<00:41, 7211.84it/s] 26%|██▌       | 104044/400000 [00:14<00:40, 7250.39it/s] 26%|██▌       | 104784/400000 [00:14<00:40, 7292.70it/s] 26%|██▋       | 105563/400000 [00:14<00:39, 7433.08it/s] 27%|██▋       | 106328/400000 [00:14<00:39, 7495.79it/s] 27%|██▋       | 107079/400000 [00:14<00:39, 7485.33it/s] 27%|██▋       | 107829/400000 [00:15<00:39, 7464.53it/s] 27%|██▋       | 108576/400000 [00:15<00:39, 7407.16it/s] 27%|██▋       | 109364/400000 [00:15<00:38, 7541.93it/s] 28%|██▊       | 110141/400000 [00:15<00:38, 7607.93it/s] 28%|██▊       | 110903/400000 [00:15<00:39, 7352.59it/s] 28%|██▊       | 111641/400000 [00:15<00:39, 7336.99it/s] 28%|██▊       | 112377/400000 [00:15<00:39, 7289.00it/s] 28%|██▊       | 113111/400000 [00:15<00:39, 7302.86it/s] 28%|██▊       | 113843/400000 [00:15<00:40, 7146.84it/s] 29%|██▊       | 114602/400000 [00:15<00:39, 7273.59it/s] 29%|██▉       | 115331/400000 [00:16<00:39, 7271.43it/s] 29%|██▉       | 116060/400000 [00:16<00:40, 7040.67it/s] 29%|██▉       | 116804/400000 [00:16<00:39, 7155.34it/s] 29%|██▉       | 117556/400000 [00:16<00:38, 7259.50it/s] 30%|██▉       | 118311/400000 [00:16<00:38, 7342.22it/s] 30%|██▉       | 119062/400000 [00:16<00:38, 7391.05it/s] 30%|██▉       | 119806/400000 [00:16<00:37, 7403.39it/s] 30%|███       | 120558/400000 [00:16<00:37, 7437.69it/s] 30%|███       | 121315/400000 [00:16<00:37, 7476.29it/s] 31%|███       | 122073/400000 [00:16<00:37, 7504.92it/s] 31%|███       | 122824/400000 [00:17<00:37, 7463.49it/s] 31%|███       | 123583/400000 [00:17<00:36, 7499.05it/s] 31%|███       | 124339/400000 [00:17<00:36, 7515.60it/s] 31%|███▏      | 125102/400000 [00:17<00:36, 7547.93it/s] 31%|███▏      | 125858/400000 [00:17<00:36, 7550.16it/s] 32%|███▏      | 126614/400000 [00:17<00:36, 7515.85it/s] 32%|███▏      | 127366/400000 [00:17<00:36, 7494.75it/s] 32%|███▏      | 128116/400000 [00:17<00:36, 7379.13it/s] 32%|███▏      | 128865/400000 [00:17<00:36, 7410.89it/s] 32%|███▏      | 129607/400000 [00:17<00:36, 7399.82it/s] 33%|███▎      | 130348/400000 [00:18<00:36, 7397.34it/s] 33%|███▎      | 131090/400000 [00:18<00:36, 7403.03it/s] 33%|███▎      | 131846/400000 [00:18<00:35, 7448.93it/s] 33%|███▎      | 132601/400000 [00:18<00:35, 7476.02it/s] 33%|███▎      | 133350/400000 [00:18<00:35, 7478.49it/s] 34%|███▎      | 134098/400000 [00:18<00:35, 7428.14it/s] 34%|███▎      | 134843/400000 [00:18<00:35, 7433.02it/s] 34%|███▍      | 135587/400000 [00:18<00:35, 7420.34it/s] 34%|███▍      | 136355/400000 [00:18<00:35, 7494.09it/s] 34%|███▍      | 137109/400000 [00:18<00:35, 7506.51it/s] 34%|███▍      | 137860/400000 [00:19<00:35, 7480.68it/s] 35%|███▍      | 138609/400000 [00:19<00:37, 7061.68it/s] 35%|███▍      | 139321/400000 [00:19<00:37, 7042.20it/s] 35%|███▌      | 140067/400000 [00:19<00:36, 7160.64it/s] 35%|███▌      | 140786/400000 [00:19<00:36, 7087.55it/s] 35%|███▌      | 141497/400000 [00:19<00:37, 6945.37it/s] 36%|███▌      | 142252/400000 [00:19<00:36, 7115.15it/s] 36%|███▌      | 143013/400000 [00:19<00:35, 7256.27it/s] 36%|███▌      | 143760/400000 [00:19<00:35, 7316.98it/s] 36%|███▌      | 144517/400000 [00:20<00:34, 7391.00it/s] 36%|███▋      | 145258/400000 [00:20<00:34, 7321.15it/s] 36%|███▋      | 145992/400000 [00:20<00:34, 7267.36it/s] 37%|███▋      | 146720/400000 [00:20<00:34, 7255.39it/s] 37%|███▋      | 147447/400000 [00:20<00:35, 7166.49it/s] 37%|███▋      | 148170/400000 [00:20<00:35, 7184.34it/s] 37%|███▋      | 148909/400000 [00:20<00:34, 7242.25it/s] 37%|███▋      | 149636/400000 [00:20<00:34, 7249.17it/s] 38%|███▊      | 150390/400000 [00:20<00:34, 7332.86it/s] 38%|███▊      | 151147/400000 [00:20<00:33, 7402.04it/s] 38%|███▊      | 151902/400000 [00:21<00:33, 7443.45it/s] 38%|███▊      | 152668/400000 [00:21<00:32, 7506.75it/s] 38%|███▊      | 153420/400000 [00:21<00:33, 7442.11it/s] 39%|███▊      | 154182/400000 [00:21<00:32, 7492.82it/s] 39%|███▊      | 154932/400000 [00:21<00:32, 7480.92it/s] 39%|███▉      | 155681/400000 [00:21<00:33, 7395.16it/s] 39%|███▉      | 156424/400000 [00:21<00:32, 7404.41it/s] 39%|███▉      | 157165/400000 [00:21<00:32, 7403.65it/s] 39%|███▉      | 157927/400000 [00:21<00:32, 7464.94it/s] 40%|███▉      | 158702/400000 [00:21<00:31, 7545.94it/s] 40%|███▉      | 159457/400000 [00:22<00:32, 7478.14it/s] 40%|████      | 160215/400000 [00:22<00:31, 7504.18it/s] 40%|████      | 160967/400000 [00:22<00:31, 7506.77it/s] 40%|████      | 161718/400000 [00:22<00:32, 7411.99it/s] 41%|████      | 162460/400000 [00:22<00:32, 7361.81it/s] 41%|████      | 163214/400000 [00:22<00:31, 7413.49it/s] 41%|████      | 163989/400000 [00:22<00:31, 7511.17it/s] 41%|████      | 164741/400000 [00:22<00:31, 7512.88it/s] 41%|████▏     | 165493/400000 [00:22<00:31, 7460.17it/s] 42%|████▏     | 166245/400000 [00:22<00:31, 7476.74it/s] 42%|████▏     | 167002/400000 [00:23<00:31, 7503.38it/s] 42%|████▏     | 167753/400000 [00:23<00:31, 7479.11it/s] 42%|████▏     | 168502/400000 [00:23<00:31, 7441.80it/s] 42%|████▏     | 169264/400000 [00:23<00:30, 7494.15it/s] 43%|████▎     | 170019/400000 [00:23<00:30, 7506.35it/s] 43%|████▎     | 170796/400000 [00:23<00:30, 7580.92it/s] 43%|████▎     | 171555/400000 [00:23<00:30, 7476.82it/s] 43%|████▎     | 172304/400000 [00:23<00:30, 7363.27it/s] 43%|████▎     | 173055/400000 [00:23<00:30, 7403.93it/s] 43%|████▎     | 173796/400000 [00:23<00:30, 7391.34it/s] 44%|████▎     | 174562/400000 [00:24<00:30, 7468.45it/s] 44%|████▍     | 175310/400000 [00:24<00:30, 7421.47it/s] 44%|████▍     | 176053/400000 [00:24<00:30, 7391.08it/s] 44%|████▍     | 176793/400000 [00:24<00:30, 7383.16it/s] 44%|████▍     | 177558/400000 [00:24<00:29, 7459.23it/s] 45%|████▍     | 178315/400000 [00:24<00:29, 7492.10it/s] 45%|████▍     | 179065/400000 [00:24<00:29, 7490.07it/s] 45%|████▍     | 179815/400000 [00:24<00:29, 7480.86it/s] 45%|████▌     | 180572/400000 [00:24<00:29, 7507.06it/s] 45%|████▌     | 181338/400000 [00:24<00:28, 7549.81it/s] 46%|████▌     | 182112/400000 [00:25<00:28, 7604.83it/s] 46%|████▌     | 182877/400000 [00:25<00:28, 7616.42it/s] 46%|████▌     | 183639/400000 [00:25<00:28, 7532.43it/s] 46%|████▌     | 184405/400000 [00:25<00:28, 7568.92it/s] 46%|████▋     | 185163/400000 [00:25<00:28, 7568.33it/s] 46%|████▋     | 185921/400000 [00:25<00:28, 7550.66it/s] 47%|████▋     | 186677/400000 [00:25<00:28, 7506.18it/s] 47%|████▋     | 187428/400000 [00:25<00:28, 7421.72it/s] 47%|████▋     | 188190/400000 [00:25<00:28, 7479.35it/s] 47%|████▋     | 188939/400000 [00:25<00:28, 7310.82it/s] 47%|████▋     | 189707/400000 [00:26<00:28, 7415.93it/s] 48%|████▊     | 190475/400000 [00:26<00:27, 7491.60it/s] 48%|████▊     | 191226/400000 [00:26<00:28, 7376.02it/s] 48%|████▊     | 191978/400000 [00:26<00:28, 7416.81it/s] 48%|████▊     | 192721/400000 [00:26<00:28, 7312.61it/s] 48%|████▊     | 193454/400000 [00:26<00:28, 7312.69it/s] 49%|████▊     | 194203/400000 [00:26<00:27, 7361.45it/s] 49%|████▊     | 194940/400000 [00:26<00:27, 7359.60it/s] 49%|████▉     | 195677/400000 [00:26<00:27, 7345.19it/s] 49%|████▉     | 196422/400000 [00:26<00:27, 7373.54it/s] 49%|████▉     | 197160/400000 [00:27<00:27, 7323.86it/s] 49%|████▉     | 197897/400000 [00:27<00:27, 7337.08it/s] 50%|████▉     | 198658/400000 [00:27<00:27, 7415.58it/s] 50%|████▉     | 199427/400000 [00:27<00:26, 7494.58it/s] 50%|█████     | 200193/400000 [00:27<00:26, 7541.63it/s] 50%|█████     | 200965/400000 [00:27<00:26, 7593.74it/s] 50%|█████     | 201725/400000 [00:27<00:26, 7544.95it/s] 51%|█████     | 202480/400000 [00:27<00:26, 7496.94it/s] 51%|█████     | 203236/400000 [00:27<00:26, 7513.70it/s] 51%|█████     | 203988/400000 [00:28<00:26, 7485.16it/s] 51%|█████     | 204738/400000 [00:28<00:26, 7488.20it/s] 51%|█████▏    | 205487/400000 [00:28<00:26, 7403.14it/s] 52%|█████▏    | 206228/400000 [00:28<00:26, 7241.72it/s] 52%|█████▏    | 206988/400000 [00:28<00:26, 7344.99it/s] 52%|█████▏    | 207724/400000 [00:28<00:26, 7225.13it/s] 52%|█████▏    | 208480/400000 [00:28<00:26, 7322.37it/s] 52%|█████▏    | 209215/400000 [00:28<00:26, 7329.29it/s] 52%|█████▏    | 209949/400000 [00:28<00:26, 7249.92it/s] 53%|█████▎    | 210694/400000 [00:28<00:25, 7305.35it/s] 53%|█████▎    | 211426/400000 [00:29<00:25, 7298.57it/s] 53%|█████▎    | 212198/400000 [00:29<00:25, 7417.62it/s] 53%|█████▎    | 212947/400000 [00:29<00:25, 7437.88it/s] 53%|█████▎    | 213692/400000 [00:29<00:25, 7335.16it/s] 54%|█████▎    | 214440/400000 [00:29<00:25, 7377.85it/s] 54%|█████▍    | 215195/400000 [00:29<00:24, 7425.75it/s] 54%|█████▍    | 215939/400000 [00:29<00:24, 7417.69it/s] 54%|█████▍    | 216682/400000 [00:29<00:24, 7416.78it/s] 54%|█████▍    | 217424/400000 [00:29<00:24, 7386.56it/s] 55%|█████▍    | 218185/400000 [00:29<00:24, 7452.18it/s] 55%|█████▍    | 218944/400000 [00:30<00:24, 7491.03it/s] 55%|█████▍    | 219694/400000 [00:30<00:24, 7462.40it/s] 55%|█████▌    | 220444/400000 [00:30<00:24, 7471.63it/s] 55%|█████▌    | 221192/400000 [00:30<00:24, 7415.22it/s] 55%|█████▌    | 221934/400000 [00:30<00:25, 7105.90it/s] 56%|█████▌    | 222683/400000 [00:30<00:24, 7215.80it/s] 56%|█████▌    | 223416/400000 [00:30<00:24, 7248.91it/s] 56%|█████▌    | 224154/400000 [00:30<00:24, 7286.47it/s] 56%|█████▌    | 224891/400000 [00:30<00:23, 7309.52it/s] 56%|█████▋    | 225642/400000 [00:30<00:23, 7367.62it/s] 57%|█████▋    | 226428/400000 [00:31<00:23, 7507.94it/s] 57%|█████▋    | 227184/400000 [00:31<00:22, 7522.38it/s] 57%|█████▋    | 227953/400000 [00:31<00:22, 7571.42it/s] 57%|█████▋    | 228711/400000 [00:31<00:22, 7520.01it/s] 57%|█████▋    | 229464/400000 [00:31<00:22, 7494.93it/s] 58%|█████▊    | 230219/400000 [00:31<00:22, 7511.18it/s] 58%|█████▊    | 230971/400000 [00:31<00:22, 7377.60it/s] 58%|█████▊    | 231710/400000 [00:31<00:22, 7351.04it/s] 58%|█████▊    | 232446/400000 [00:31<00:22, 7298.73it/s] 58%|█████▊    | 233190/400000 [00:31<00:22, 7339.89it/s] 58%|█████▊    | 233937/400000 [00:32<00:22, 7378.38it/s] 59%|█████▊    | 234682/400000 [00:32<00:22, 7397.87it/s] 59%|█████▉    | 235425/400000 [00:32<00:22, 7405.67it/s] 59%|█████▉    | 236168/400000 [00:32<00:22, 7411.83it/s] 59%|█████▉    | 236926/400000 [00:32<00:21, 7460.94it/s] 59%|█████▉    | 237687/400000 [00:32<00:21, 7505.02it/s] 60%|█████▉    | 238457/400000 [00:32<00:21, 7560.13it/s] 60%|█████▉    | 239223/400000 [00:32<00:21, 7588.74it/s] 60%|█████▉    | 239983/400000 [00:32<00:21, 7525.13it/s] 60%|██████    | 240736/400000 [00:32<00:21, 7478.56it/s] 60%|██████    | 241523/400000 [00:33<00:20, 7589.40it/s] 61%|██████    | 242283/400000 [00:33<00:20, 7591.34it/s] 61%|██████    | 243043/400000 [00:33<00:20, 7586.75it/s] 61%|██████    | 243805/400000 [00:33<00:20, 7594.79it/s] 61%|██████    | 244565/400000 [00:33<00:20, 7552.90it/s] 61%|██████▏   | 245321/400000 [00:33<00:20, 7516.06it/s] 62%|██████▏   | 246073/400000 [00:33<00:20, 7420.08it/s] 62%|██████▏   | 246816/400000 [00:33<00:21, 7203.52it/s] 62%|██████▏   | 247558/400000 [00:33<00:20, 7263.84it/s] 62%|██████▏   | 248296/400000 [00:33<00:20, 7296.12it/s] 62%|██████▏   | 249027/400000 [00:34<00:20, 7238.83it/s] 62%|██████▏   | 249752/400000 [00:34<00:20, 7239.37it/s] 63%|██████▎   | 250516/400000 [00:34<00:20, 7354.67it/s] 63%|██████▎   | 251253/400000 [00:34<00:20, 7348.22it/s] 63%|██████▎   | 251989/400000 [00:34<00:20, 7267.47it/s] 63%|██████▎   | 252748/400000 [00:34<00:20, 7360.19it/s] 63%|██████▎   | 253525/400000 [00:34<00:19, 7476.17it/s] 64%|██████▎   | 254274/400000 [00:34<00:19, 7479.74it/s] 64%|██████▍   | 255044/400000 [00:34<00:19, 7542.86it/s] 64%|██████▍   | 255807/400000 [00:34<00:19, 7566.48it/s] 64%|██████▍   | 256565/400000 [00:35<00:18, 7564.72it/s] 64%|██████▍   | 257322/400000 [00:35<00:19, 7507.77it/s] 65%|██████▍   | 258090/400000 [00:35<00:18, 7555.21it/s] 65%|██████▍   | 258846/400000 [00:35<00:18, 7541.83it/s] 65%|██████▍   | 259601/400000 [00:35<00:19, 7361.17it/s] 65%|██████▌   | 260339/400000 [00:35<00:19, 7288.31it/s] 65%|██████▌   | 261069/400000 [00:35<00:19, 7252.85it/s] 65%|██████▌   | 261810/400000 [00:35<00:18, 7296.30it/s] 66%|██████▌   | 262541/400000 [00:35<00:19, 7106.80it/s] 66%|██████▌   | 263290/400000 [00:36<00:18, 7217.24it/s] 66%|██████▌   | 264034/400000 [00:36<00:18, 7280.30it/s] 66%|██████▌   | 264764/400000 [00:36<00:18, 7159.55it/s] 66%|██████▋   | 265511/400000 [00:36<00:18, 7248.63it/s] 67%|██████▋   | 266265/400000 [00:36<00:18, 7330.75it/s] 67%|██████▋   | 267011/400000 [00:36<00:18, 7365.33it/s] 67%|██████▋   | 267755/400000 [00:36<00:17, 7385.49it/s] 67%|██████▋   | 268498/400000 [00:36<00:17, 7396.95it/s] 67%|██████▋   | 269239/400000 [00:36<00:17, 7387.92it/s] 67%|██████▋   | 269985/400000 [00:36<00:17, 7406.95it/s] 68%|██████▊   | 270726/400000 [00:37<00:17, 7400.54it/s] 68%|██████▊   | 271467/400000 [00:37<00:18, 6916.29it/s] 68%|██████▊   | 272180/400000 [00:37<00:18, 6978.39it/s] 68%|██████▊   | 272915/400000 [00:37<00:17, 7085.53it/s] 68%|██████▊   | 273643/400000 [00:37<00:17, 7141.40it/s] 69%|██████▊   | 274387/400000 [00:37<00:17, 7226.06it/s] 69%|██████▉   | 275129/400000 [00:37<00:17, 7280.92it/s] 69%|██████▉   | 275860/400000 [00:37<00:17, 7287.71it/s] 69%|██████▉   | 276605/400000 [00:37<00:16, 7331.79it/s] 69%|██████▉   | 277340/400000 [00:37<00:16, 7247.32it/s] 70%|██████▉   | 278077/400000 [00:38<00:16, 7282.74it/s] 70%|██████▉   | 278832/400000 [00:38<00:16, 7359.23it/s] 70%|██████▉   | 279580/400000 [00:38<00:16, 7394.71it/s] 70%|███████   | 280320/400000 [00:38<00:16, 7342.31it/s] 70%|███████   | 281055/400000 [00:38<00:16, 7341.95it/s] 70%|███████   | 281797/400000 [00:38<00:16, 7363.54it/s] 71%|███████   | 282540/400000 [00:38<00:15, 7382.45it/s] 71%|███████   | 283280/400000 [00:38<00:15, 7386.77it/s] 71%|███████   | 284019/400000 [00:38<00:15, 7350.56it/s] 71%|███████   | 284755/400000 [00:38<00:15, 7347.88it/s] 71%|███████▏  | 285499/400000 [00:39<00:15, 7373.93it/s] 72%|███████▏  | 286250/400000 [00:39<00:15, 7412.49it/s] 72%|███████▏  | 286996/400000 [00:39<00:15, 7425.25it/s] 72%|███████▏  | 287765/400000 [00:39<00:14, 7502.40it/s] 72%|███████▏  | 288523/400000 [00:39<00:14, 7524.08it/s] 72%|███████▏  | 289276/400000 [00:39<00:14, 7502.52it/s] 73%|███████▎  | 290027/400000 [00:39<00:14, 7384.94it/s] 73%|███████▎  | 290767/400000 [00:39<00:14, 7347.72it/s] 73%|███████▎  | 291503/400000 [00:39<00:15, 7209.27it/s] 73%|███████▎  | 292225/400000 [00:39<00:15, 7100.03it/s] 73%|███████▎  | 292936/400000 [00:40<00:15, 7034.85it/s] 73%|███████▎  | 293676/400000 [00:40<00:14, 7140.49it/s] 74%|███████▎  | 294433/400000 [00:40<00:14, 7263.41it/s] 74%|███████▍  | 295161/400000 [00:40<00:14, 7250.17it/s] 74%|███████▍  | 295912/400000 [00:40<00:14, 7325.99it/s] 74%|███████▍  | 296654/400000 [00:40<00:14, 7352.54it/s] 74%|███████▍  | 297399/400000 [00:40<00:13, 7381.34it/s] 75%|███████▍  | 298138/400000 [00:40<00:13, 7347.45it/s] 75%|███████▍  | 298885/400000 [00:40<00:13, 7381.41it/s] 75%|███████▍  | 299624/400000 [00:40<00:13, 7175.15it/s] 75%|███████▌  | 300343/400000 [00:41<00:14, 7081.59it/s] 75%|███████▌  | 301100/400000 [00:41<00:13, 7220.52it/s] 75%|███████▌  | 301837/400000 [00:41<00:13, 7262.85it/s] 76%|███████▌  | 302574/400000 [00:41<00:13, 7293.48it/s] 76%|███████▌  | 303305/400000 [00:41<00:13, 7129.59it/s] 76%|███████▌  | 304058/400000 [00:41<00:13, 7243.65it/s] 76%|███████▌  | 304804/400000 [00:41<00:13, 7306.61it/s] 76%|███████▋  | 305536/400000 [00:41<00:12, 7276.29it/s] 77%|███████▋  | 306287/400000 [00:41<00:12, 7341.69it/s] 77%|███████▋  | 307050/400000 [00:42<00:12, 7421.33it/s] 77%|███████▋  | 307807/400000 [00:42<00:12, 7463.12it/s] 77%|███████▋  | 308554/400000 [00:42<00:12, 7393.37it/s] 77%|███████▋  | 309294/400000 [00:42<00:12, 7284.50it/s] 78%|███████▊  | 310024/400000 [00:42<00:12, 7269.17it/s] 78%|███████▊  | 310752/400000 [00:42<00:12, 7189.85it/s] 78%|███████▊  | 311494/400000 [00:42<00:12, 7257.35it/s] 78%|███████▊  | 312241/400000 [00:42<00:11, 7316.76it/s] 78%|███████▊  | 312982/400000 [00:42<00:11, 7342.68it/s] 78%|███████▊  | 313717/400000 [00:42<00:11, 7282.09it/s] 79%|███████▊  | 314459/400000 [00:43<00:11, 7322.59it/s] 79%|███████▉  | 315192/400000 [00:43<00:12, 7043.00it/s] 79%|███████▉  | 315899/400000 [00:43<00:11, 7046.79it/s] 79%|███████▉  | 316658/400000 [00:43<00:11, 7198.85it/s] 79%|███████▉  | 317412/400000 [00:43<00:11, 7296.60it/s] 80%|███████▉  | 318163/400000 [00:43<00:11, 7358.21it/s] 80%|███████▉  | 318901/400000 [00:43<00:11, 7357.88it/s] 80%|███████▉  | 319640/400000 [00:43<00:10, 7366.52it/s] 80%|████████  | 320407/400000 [00:43<00:10, 7453.32it/s] 80%|████████  | 321154/400000 [00:43<00:10, 7451.97it/s] 80%|████████  | 321920/400000 [00:44<00:10, 7511.56it/s] 81%|████████  | 322672/400000 [00:44<00:10, 7512.80it/s] 81%|████████  | 323424/400000 [00:44<00:10, 7434.08it/s] 81%|████████  | 324168/400000 [00:44<00:10, 7427.60it/s] 81%|████████  | 324912/400000 [00:44<00:10, 7423.70it/s] 81%|████████▏ | 325655/400000 [00:44<00:10, 7336.27it/s] 82%|████████▏ | 326390/400000 [00:44<00:10, 7328.60it/s] 82%|████████▏ | 327141/400000 [00:44<00:09, 7380.87it/s] 82%|████████▏ | 327890/400000 [00:44<00:09, 7412.85it/s] 82%|████████▏ | 328643/400000 [00:44<00:09, 7445.57it/s] 82%|████████▏ | 329388/400000 [00:45<00:09, 7429.33it/s] 83%|████████▎ | 330132/400000 [00:45<00:09, 7405.18it/s] 83%|████████▎ | 330877/400000 [00:45<00:09, 7416.67it/s] 83%|████████▎ | 331636/400000 [00:45<00:09, 7466.35it/s] 83%|████████▎ | 332383/400000 [00:45<00:09, 7460.48it/s] 83%|████████▎ | 333144/400000 [00:45<00:08, 7503.48it/s] 83%|████████▎ | 333904/400000 [00:45<00:08, 7531.17it/s] 84%|████████▎ | 334662/400000 [00:45<00:08, 7541.63it/s] 84%|████████▍ | 335417/400000 [00:45<00:08, 7504.27it/s] 84%|████████▍ | 336168/400000 [00:45<00:08, 7386.50it/s] 84%|████████▍ | 336920/400000 [00:46<00:08, 7424.06it/s] 84%|████████▍ | 337663/400000 [00:46<00:08, 7364.09it/s] 85%|████████▍ | 338403/400000 [00:46<00:08, 7374.45it/s] 85%|████████▍ | 339169/400000 [00:46<00:08, 7456.42it/s] 85%|████████▍ | 339930/400000 [00:46<00:08, 7501.78it/s] 85%|████████▌ | 340681/400000 [00:46<00:07, 7477.03it/s] 85%|████████▌ | 341439/400000 [00:46<00:07, 7505.06it/s] 86%|████████▌ | 342190/400000 [00:46<00:07, 7498.64it/s] 86%|████████▌ | 342941/400000 [00:46<00:07, 7495.35it/s] 86%|████████▌ | 343707/400000 [00:46<00:07, 7543.90it/s] 86%|████████▌ | 344462/400000 [00:47<00:07, 7495.75it/s] 86%|████████▋ | 345215/400000 [00:47<00:07, 7504.32it/s] 86%|████████▋ | 345966/400000 [00:47<00:07, 7404.25it/s] 87%|████████▋ | 346737/400000 [00:47<00:07, 7492.35it/s] 87%|████████▋ | 347509/400000 [00:47<00:06, 7556.95it/s] 87%|████████▋ | 348266/400000 [00:47<00:06, 7545.63it/s] 87%|████████▋ | 349021/400000 [00:47<00:06, 7449.64it/s] 87%|████████▋ | 349767/400000 [00:47<00:06, 7359.40it/s] 88%|████████▊ | 350525/400000 [00:47<00:06, 7420.28it/s] 88%|████████▊ | 351292/400000 [00:47<00:06, 7492.92it/s] 88%|████████▊ | 352042/400000 [00:48<00:06, 7473.41it/s] 88%|████████▊ | 352790/400000 [00:48<00:06, 7447.30it/s] 88%|████████▊ | 353536/400000 [00:48<00:06, 7424.37it/s] 89%|████████▊ | 354279/400000 [00:48<00:06, 7408.03it/s] 89%|████████▉ | 355020/400000 [00:48<00:06, 7328.33it/s] 89%|████████▉ | 355754/400000 [00:48<00:06, 7310.35it/s] 89%|████████▉ | 356517/400000 [00:48<00:05, 7402.61it/s] 89%|████████▉ | 357269/400000 [00:48<00:05, 7434.08it/s] 90%|████████▉ | 358013/400000 [00:48<00:05, 7419.94it/s] 90%|████████▉ | 358760/400000 [00:48<00:05, 7434.29it/s] 90%|████████▉ | 359504/400000 [00:49<00:05, 7318.57it/s] 90%|█████████ | 360237/400000 [00:49<00:05, 7309.22it/s] 90%|█████████ | 360972/400000 [00:49<00:05, 7319.72it/s] 90%|█████████ | 361705/400000 [00:49<00:05, 7262.50it/s] 91%|█████████ | 362452/400000 [00:49<00:05, 7322.86it/s] 91%|█████████ | 363203/400000 [00:49<00:05, 7338.75it/s] 91%|█████████ | 363950/400000 [00:49<00:04, 7375.56it/s] 91%|█████████ | 364691/400000 [00:49<00:04, 7383.51it/s] 91%|█████████▏| 365447/400000 [00:49<00:04, 7433.83it/s] 92%|█████████▏| 366195/400000 [00:50<00:04, 7444.42it/s] 92%|█████████▏| 366948/400000 [00:50<00:04, 7468.90it/s] 92%|█████████▏| 367696/400000 [00:50<00:04, 7419.69it/s] 92%|█████████▏| 368439/400000 [00:50<00:04, 7391.31it/s] 92%|█████████▏| 369183/400000 [00:50<00:04, 7404.30it/s] 92%|█████████▏| 369924/400000 [00:50<00:04, 7389.77it/s] 93%|█████████▎| 370673/400000 [00:50<00:03, 7416.98it/s] 93%|█████████▎| 371415/400000 [00:50<00:03, 7371.66it/s] 93%|█████████▎| 372159/400000 [00:50<00:03, 7391.91it/s] 93%|█████████▎| 372912/400000 [00:50<00:03, 7430.70it/s] 93%|█████████▎| 373675/400000 [00:51<00:03, 7487.90it/s] 94%|█████████▎| 374424/400000 [00:51<00:03, 7486.15it/s] 94%|█████████▍| 375173/400000 [00:51<00:03, 7425.22it/s] 94%|█████████▍| 375932/400000 [00:51<00:03, 7473.58it/s] 94%|█████████▍| 376705/400000 [00:51<00:03, 7546.27it/s] 94%|█████████▍| 377461/400000 [00:51<00:02, 7548.47it/s] 95%|█████████▍| 378225/400000 [00:51<00:02, 7573.69it/s] 95%|█████████▍| 378983/400000 [00:51<00:02, 7530.04it/s] 95%|█████████▍| 379737/400000 [00:51<00:02, 7530.71it/s] 95%|█████████▌| 380491/400000 [00:51<00:02, 7526.22it/s] 95%|█████████▌| 381244/400000 [00:52<00:02, 7330.78it/s] 95%|█████████▌| 381982/400000 [00:52<00:02, 7342.55it/s] 96%|█████████▌| 382718/400000 [00:52<00:02, 7253.08it/s] 96%|█████████▌| 383445/400000 [00:52<00:02, 7211.48it/s] 96%|█████████▌| 384175/400000 [00:52<00:02, 7235.26it/s] 96%|█████████▌| 384899/400000 [00:52<00:02, 7221.64it/s] 96%|█████████▋| 385640/400000 [00:52<00:01, 7275.71it/s] 97%|█████████▋| 386368/400000 [00:52<00:01, 7027.19it/s] 97%|█████████▋| 387103/400000 [00:52<00:01, 7120.97it/s] 97%|█████████▋| 387850/400000 [00:52<00:01, 7222.06it/s] 97%|█████████▋| 388574/400000 [00:53<00:01, 7157.10it/s] 97%|█████████▋| 389299/400000 [00:53<00:01, 7184.43it/s] 98%|█████████▊| 390039/400000 [00:53<00:01, 7245.70it/s] 98%|█████████▊| 390782/400000 [00:53<00:01, 7297.66it/s] 98%|█████████▊| 391515/400000 [00:53<00:01, 7304.10it/s] 98%|█████████▊| 392256/400000 [00:53<00:01, 7333.49it/s] 98%|█████████▊| 392990/400000 [00:53<00:00, 7242.46it/s] 98%|█████████▊| 393715/400000 [00:53<00:00, 7160.15it/s] 99%|█████████▊| 394453/400000 [00:53<00:00, 7221.74it/s] 99%|█████████▉| 395212/400000 [00:53<00:00, 7327.61it/s] 99%|█████████▉| 395946/400000 [00:54<00:00, 7283.27it/s] 99%|█████████▉| 396687/400000 [00:54<00:00, 7319.50it/s] 99%|█████████▉| 397420/400000 [00:54<00:00, 7294.94it/s]100%|█████████▉| 398150/400000 [00:54<00:00, 7287.15it/s]100%|█████████▉| 398922/400000 [00:54<00:00, 7411.77it/s]100%|█████████▉| 399674/400000 [00:54<00:00, 7440.42it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7323.93it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ff5a2982cf8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010944861796110315 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.010866710772881141 	 Accuracy: 70

  model saves at 70% accuracy 

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
2020-05-15 15:24:48.348719: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 15:24:48.353024: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 15:24:48.353239: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558f6b110180 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 15:24:48.353260: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ff54beba400> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.7586 - accuracy: 0.4940
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7356 - accuracy: 0.4955
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.8046 - accuracy: 0.4910 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.8046 - accuracy: 0.4910
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7678 - accuracy: 0.4934
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.7637 - accuracy: 0.4937
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7236 - accuracy: 0.4963
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.7337 - accuracy: 0.4956
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7228 - accuracy: 0.4963
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7157 - accuracy: 0.4968
11000/25000 [============>.................] - ETA: 4s - loss: 7.7029 - accuracy: 0.4976
12000/25000 [=============>................] - ETA: 4s - loss: 7.6807 - accuracy: 0.4991
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6938 - accuracy: 0.4982
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6951 - accuracy: 0.4981
15000/25000 [=================>............] - ETA: 3s - loss: 7.7004 - accuracy: 0.4978
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6839 - accuracy: 0.4989
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6675 - accuracy: 0.4999
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6598 - accuracy: 0.5004
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6553 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6436 - accuracy: 0.5015
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6764 - accuracy: 0.4994
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6720 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6730 - accuracy: 0.4996
25000/25000 [==============================] - 10s 400us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7ff4f5b01668> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7ff508343e48> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.5750 - crf_viterbi_accuracy: 0.3467 - val_loss: 1.5355 - val_crf_viterbi_accuracy: 0.3333

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


  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'd580c5017e28eefaf82dbb63ddf4270e71792c2b', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/d580c5017e28eefaf82dbb63ddf4270e71792c2b

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fb70ac93fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 17:13:01.326288
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 17:13:01.331177
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 17:13:01.335174
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 17:13:01.339980
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fb716cab400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355433.1250
Epoch 2/10

1/1 [==============================] - 0s 106ms/step - loss: 276765.4375
Epoch 3/10

1/1 [==============================] - 0s 97ms/step - loss: 192206.6250
Epoch 4/10

1/1 [==============================] - 0s 97ms/step - loss: 124473.3672
Epoch 5/10

1/1 [==============================] - 0s 96ms/step - loss: 79069.5859
Epoch 6/10

1/1 [==============================] - 0s 100ms/step - loss: 50942.9180
Epoch 7/10

1/1 [==============================] - 0s 113ms/step - loss: 34166.5117
Epoch 8/10

1/1 [==============================] - 0s 103ms/step - loss: 24123.4199
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 17872.5957
Epoch 10/10

1/1 [==============================] - 0s 96ms/step - loss: 13790.9326

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.68217033  0.27114058  0.02602345 -0.918713   -1.1672931  -0.27316663
   1.1170614   0.25132787 -0.35910094 -0.5907615   0.21321887  0.18684152
  -0.10654432 -1.2705979  -0.9637631   0.21358046  0.17216182 -0.47075418
   0.17438608 -0.4262602  -0.878497   -1.0912964  -0.91050315 -0.4428791
  -0.7866163  -0.21293063  1.0740838   0.23866805  1.1702807  -0.34341633
   1.0719445  -0.20437968 -1.546855    0.19180405  0.37809008  0.684863
  -0.13000175 -1.0292637   0.29500479 -0.14367035 -0.6157889  -1.0705421
   0.16957119 -0.27502826 -0.7699132   1.1928707   0.04490834  0.7857742
  -1.2645292  -0.02392747 -0.5670134   0.3019446   1.2246282  -0.12465587
   0.44267476  0.69507277  0.548021   -1.4105659   0.68650377 -0.39339957
   0.22289705 -1.2400931   1.7271132   0.40482023 -0.50942576  0.04354686
  -0.25353956  0.25667003 -0.29849696  0.45897985  0.51468956  0.27977028
   1.1599636  -1.0274342  -0.837445    0.7913212   0.39444926  0.03348184
   1.2199755  -0.8763717  -0.49183884  0.2514019   0.5782951  -0.7734045
   0.77511024 -0.7708995   1.432004    1.10186    -0.34677398  0.27465
  -0.13675243  0.65441024 -0.41181445  0.51102763 -0.59859395  0.34180626
  -0.4490949  -0.69426066  0.62936306  0.54416245  0.39746934 -1.413316
   0.5396718   0.56362927 -1.3864306  -1.2840918   0.09482497  0.8152295
   0.37244427  1.088503   -0.16623795 -0.1646334  -1.0609586   0.67208946
  -0.626256    0.48236212  0.50169766  0.7660049   0.5562238  -0.665023
   0.13693684  4.9302697   6.0402317   4.064987    5.380899    4.6628013
   5.0214376   5.8605337   4.354829    6.0937696   6.2014065   3.4016104
   4.2445097   3.5995266   4.9826527   4.9170585   6.0684967   5.4329886
   4.3130674   5.21123     5.008833    5.533416    6.353035    5.6777105
   4.9613433   4.9166117   5.4633794   5.8123217   5.189755    5.022853
   4.6272564   6.1203575   4.8515406   5.383339    5.8785324   6.7445736
   4.196188    4.9349146   5.520171    5.2004666   5.1526046   5.05725
   4.604912    4.1328535   4.3027406   6.3106008   5.691536    5.052994
   5.0506563   5.2897787   5.5501714   4.3154163   5.628541    4.5471377
   5.2233872   4.2134256   5.3410797   5.262135    5.8111043   4.533545
   0.6681651   0.2893821   1.4867676   1.4459713   0.24608487  1.331863
   1.3411237   0.8009604   0.6404246   1.0400674   1.3902371   1.2776974
   2.047522    0.930326    2.4576087   0.5065442   2.4963603   1.5986645
   1.5112737   0.5209551   0.6568515   1.6929921   0.77583134  2.2001367
   2.458841    0.91242045  0.6446165   0.3739105   0.29787362  1.3661894
   2.4018831   1.8696513   1.5913444   2.1513224   0.90081024  0.8839723
   2.1424217   1.3555262   1.231437    0.8696284   0.39521086  0.6404793
   0.54168683  1.5937737   1.4353099   0.3648187   1.0916476   0.65969497
   0.74408764  0.58758473  1.389327    1.7726388   1.7152698   0.41182637
   0.72935474  0.69696105  0.50478333  1.0735652   0.726906    1.0813607
   1.6645677   0.24927032  1.4199116   0.21225363  1.3101729   1.7318605
   1.404089    0.23483121  1.6219258   0.27145314  1.5538967   1.185317
   0.2059257   1.0743811   2.1223645   0.4094857   1.1570249   0.5586334
   2.013029    1.6926203   0.72732866  0.18262291  1.3729725   2.2056003
   1.151827    0.39296418  1.5109963   0.5788646   2.1233683   1.171239
   2.159686    1.1744574   1.6914954   0.7507791   1.3570405   2.2812276
   1.3721621   1.2285278   0.8065534   1.3515025   1.2833147   2.3689995
   0.56633687  2.05476     1.5974996   1.302836    0.82892215  2.2489867
   0.3792231   0.7652333   1.5013154   1.059635    0.20992434  1.8006194
   1.2854125   1.0812788   1.1227942   0.869643    0.31536484  0.30481327
   0.03759474  6.021926    5.377343    4.9338393   6.3842072   5.5589714
   5.126961    5.5277257   4.7530203   4.7697616   5.8490734   6.685057
   5.773577    7.0806785   6.1343117   5.8153963   5.76828     6.278883
   6.4738007   5.720655    4.798238    5.7478623   6.5032988   4.924022
   4.8746667   6.0135345   6.143876    5.410243    6.1459684   5.1541014
   5.4469347   6.273699    5.8285875   6.0330033   5.437943    6.702633
   4.9789143   5.5637083   5.554339    6.64007     5.7116303   4.9997973
   4.9961157   4.7295895   5.8542476   5.783031    6.1155386   6.25715
   6.622087    5.9310346   5.6748323   4.328045    6.2312913   5.3434267
   5.553434    6.3185134   6.685869    5.2888484   5.448883    5.9574194
  -7.9248724  -3.226128    2.9374027 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 17:13:10.774627
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.8512
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 17:13:10.779666
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9399.17
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 17:13:10.783382
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.5649
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 17:13:10.786979
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -840.762
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140423796920784
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140422838034952
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140422838035456
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140422838035960
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140422838036464
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140422838036968

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fb7046775c0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.651651
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.615898
grad_step = 000002, loss = 0.587769
grad_step = 000003, loss = 0.557786
grad_step = 000004, loss = 0.525123
grad_step = 000005, loss = 0.494050
grad_step = 000006, loss = 0.468955
grad_step = 000007, loss = 0.444781
grad_step = 000008, loss = 0.417675
grad_step = 000009, loss = 0.398770
grad_step = 000010, loss = 0.386478
grad_step = 000011, loss = 0.366758
grad_step = 000012, loss = 0.344757
grad_step = 000013, loss = 0.326810
grad_step = 000014, loss = 0.311506
grad_step = 000015, loss = 0.298413
grad_step = 000016, loss = 0.285756
grad_step = 000017, loss = 0.270457
grad_step = 000018, loss = 0.253806
grad_step = 000019, loss = 0.239278
grad_step = 000020, loss = 0.227750
grad_step = 000021, loss = 0.217602
grad_step = 000022, loss = 0.206694
grad_step = 000023, loss = 0.194885
grad_step = 000024, loss = 0.183862
grad_step = 000025, loss = 0.174690
grad_step = 000026, loss = 0.166627
grad_step = 000027, loss = 0.158169
grad_step = 000028, loss = 0.148914
grad_step = 000029, loss = 0.139679
grad_step = 000030, loss = 0.131392
grad_step = 000031, loss = 0.124063
grad_step = 000032, loss = 0.116618
grad_step = 000033, loss = 0.108730
grad_step = 000034, loss = 0.101206
grad_step = 000035, loss = 0.094708
grad_step = 000036, loss = 0.088821
grad_step = 000037, loss = 0.083048
grad_step = 000038, loss = 0.077333
grad_step = 000039, loss = 0.071722
grad_step = 000040, loss = 0.066461
grad_step = 000041, loss = 0.061631
grad_step = 000042, loss = 0.057004
grad_step = 000043, loss = 0.052518
grad_step = 000044, loss = 0.048268
grad_step = 000045, loss = 0.044334
grad_step = 000046, loss = 0.040761
grad_step = 000047, loss = 0.037432
grad_step = 000048, loss = 0.034299
grad_step = 000049, loss = 0.031366
grad_step = 000050, loss = 0.028657
grad_step = 000051, loss = 0.026172
grad_step = 000052, loss = 0.023929
grad_step = 000053, loss = 0.021841
grad_step = 000054, loss = 0.019827
grad_step = 000055, loss = 0.018027
grad_step = 000056, loss = 0.016433
grad_step = 000057, loss = 0.014956
grad_step = 000058, loss = 0.013531
grad_step = 000059, loss = 0.012245
grad_step = 000060, loss = 0.011134
grad_step = 000061, loss = 0.010136
grad_step = 000062, loss = 0.009223
grad_step = 000063, loss = 0.008380
grad_step = 000064, loss = 0.007654
grad_step = 000065, loss = 0.006989
grad_step = 000066, loss = 0.006370
grad_step = 000067, loss = 0.005815
grad_step = 000068, loss = 0.005331
grad_step = 000069, loss = 0.004903
grad_step = 000070, loss = 0.004527
grad_step = 000071, loss = 0.004193
grad_step = 000072, loss = 0.003885
grad_step = 000073, loss = 0.003624
grad_step = 000074, loss = 0.003394
grad_step = 000075, loss = 0.003194
grad_step = 000076, loss = 0.003013
grad_step = 000077, loss = 0.002864
grad_step = 000078, loss = 0.002741
grad_step = 000079, loss = 0.002632
grad_step = 000080, loss = 0.002534
grad_step = 000081, loss = 0.002459
grad_step = 000082, loss = 0.002398
grad_step = 000083, loss = 0.002346
grad_step = 000084, loss = 0.002303
grad_step = 000085, loss = 0.002271
grad_step = 000086, loss = 0.002245
grad_step = 000087, loss = 0.002222
grad_step = 000088, loss = 0.002202
grad_step = 000089, loss = 0.002186
grad_step = 000090, loss = 0.002171
grad_step = 000091, loss = 0.002161
grad_step = 000092, loss = 0.002152
grad_step = 000093, loss = 0.002143
grad_step = 000094, loss = 0.002136
grad_step = 000095, loss = 0.002128
grad_step = 000096, loss = 0.002119
grad_step = 000097, loss = 0.002111
grad_step = 000098, loss = 0.002103
grad_step = 000099, loss = 0.002095
grad_step = 000100, loss = 0.002086
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002077
grad_step = 000102, loss = 0.002068
grad_step = 000103, loss = 0.002059
grad_step = 000104, loss = 0.002050
grad_step = 000105, loss = 0.002043
grad_step = 000106, loss = 0.002037
grad_step = 000107, loss = 0.002035
grad_step = 000108, loss = 0.002036
grad_step = 000109, loss = 0.002036
grad_step = 000110, loss = 0.002033
grad_step = 000111, loss = 0.002023
grad_step = 000112, loss = 0.002006
grad_step = 000113, loss = 0.001990
grad_step = 000114, loss = 0.001980
grad_step = 000115, loss = 0.001973
grad_step = 000116, loss = 0.001971
grad_step = 000117, loss = 0.001973
grad_step = 000118, loss = 0.001976
grad_step = 000119, loss = 0.001981
grad_step = 000120, loss = 0.001986
grad_step = 000121, loss = 0.001992
grad_step = 000122, loss = 0.001992
grad_step = 000123, loss = 0.001987
grad_step = 000124, loss = 0.001974
grad_step = 000125, loss = 0.001958
grad_step = 000126, loss = 0.001940
grad_step = 000127, loss = 0.001927
grad_step = 000128, loss = 0.001919
grad_step = 000129, loss = 0.001916
grad_step = 000130, loss = 0.001916
grad_step = 000131, loss = 0.001920
grad_step = 000132, loss = 0.001928
grad_step = 000133, loss = 0.001942
grad_step = 000134, loss = 0.001967
grad_step = 000135, loss = 0.001999
grad_step = 000136, loss = 0.002044
grad_step = 000137, loss = 0.002059
grad_step = 000138, loss = 0.002047
grad_step = 000139, loss = 0.001981
grad_step = 000140, loss = 0.001912
grad_step = 000141, loss = 0.001884
grad_step = 000142, loss = 0.001907
grad_step = 000143, loss = 0.001949
grad_step = 000144, loss = 0.001971
grad_step = 000145, loss = 0.001963
grad_step = 000146, loss = 0.001922
grad_step = 000147, loss = 0.001884
grad_step = 000148, loss = 0.001869
grad_step = 000149, loss = 0.001881
grad_step = 000150, loss = 0.001905
grad_step = 000151, loss = 0.001919
grad_step = 000152, loss = 0.001919
grad_step = 000153, loss = 0.001900
grad_step = 000154, loss = 0.001877
grad_step = 000155, loss = 0.001860
grad_step = 000156, loss = 0.001856
grad_step = 000157, loss = 0.001863
grad_step = 000158, loss = 0.001873
grad_step = 000159, loss = 0.001881
grad_step = 000160, loss = 0.001883
grad_step = 000161, loss = 0.001882
grad_step = 000162, loss = 0.001873
grad_step = 000163, loss = 0.001864
grad_step = 000164, loss = 0.001854
grad_step = 000165, loss = 0.001846
grad_step = 000166, loss = 0.001840
grad_step = 000167, loss = 0.001837
grad_step = 000168, loss = 0.001836
grad_step = 000169, loss = 0.001837
grad_step = 000170, loss = 0.001839
grad_step = 000171, loss = 0.001843
grad_step = 000172, loss = 0.001851
grad_step = 000173, loss = 0.001863
grad_step = 000174, loss = 0.001889
grad_step = 000175, loss = 0.001925
grad_step = 000176, loss = 0.001986
grad_step = 000177, loss = 0.002042
grad_step = 000178, loss = 0.002089
grad_step = 000179, loss = 0.002055
grad_step = 000180, loss = 0.001961
grad_step = 000181, loss = 0.001855
grad_step = 000182, loss = 0.001816
grad_step = 000183, loss = 0.001855
grad_step = 000184, loss = 0.001919
grad_step = 000185, loss = 0.001956
grad_step = 000186, loss = 0.001931
grad_step = 000187, loss = 0.001884
grad_step = 000188, loss = 0.001827
grad_step = 000189, loss = 0.001805
grad_step = 000190, loss = 0.001825
grad_step = 000191, loss = 0.001855
grad_step = 000192, loss = 0.001869
grad_step = 000193, loss = 0.001853
grad_step = 000194, loss = 0.001825
grad_step = 000195, loss = 0.001801
grad_step = 000196, loss = 0.001797
grad_step = 000197, loss = 0.001810
grad_step = 000198, loss = 0.001825
grad_step = 000199, loss = 0.001830
grad_step = 000200, loss = 0.001824
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001814
grad_step = 000202, loss = 0.001798
grad_step = 000203, loss = 0.001787
grad_step = 000204, loss = 0.001784
grad_step = 000205, loss = 0.001786
grad_step = 000206, loss = 0.001790
grad_step = 000207, loss = 0.001795
grad_step = 000208, loss = 0.001798
grad_step = 000209, loss = 0.001798
grad_step = 000210, loss = 0.001796
grad_step = 000211, loss = 0.001791
grad_step = 000212, loss = 0.001787
grad_step = 000213, loss = 0.001782
grad_step = 000214, loss = 0.001778
grad_step = 000215, loss = 0.001774
grad_step = 000216, loss = 0.001771
grad_step = 000217, loss = 0.001769
grad_step = 000218, loss = 0.001766
grad_step = 000219, loss = 0.001765
grad_step = 000220, loss = 0.001764
grad_step = 000221, loss = 0.001764
grad_step = 000222, loss = 0.001764
grad_step = 000223, loss = 0.001767
grad_step = 000224, loss = 0.001774
grad_step = 000225, loss = 0.001789
grad_step = 000226, loss = 0.001819
grad_step = 000227, loss = 0.001876
grad_step = 000228, loss = 0.001989
grad_step = 000229, loss = 0.002152
grad_step = 000230, loss = 0.002377
grad_step = 000231, loss = 0.002432
grad_step = 000232, loss = 0.002247
grad_step = 000233, loss = 0.001885
grad_step = 000234, loss = 0.001750
grad_step = 000235, loss = 0.001924
grad_step = 000236, loss = 0.002099
grad_step = 000237, loss = 0.002030
grad_step = 000238, loss = 0.001803
grad_step = 000239, loss = 0.001750
grad_step = 000240, loss = 0.001889
grad_step = 000241, loss = 0.001967
grad_step = 000242, loss = 0.001887
grad_step = 000243, loss = 0.001749
grad_step = 000244, loss = 0.001757
grad_step = 000245, loss = 0.001862
grad_step = 000246, loss = 0.001873
grad_step = 000247, loss = 0.001790
grad_step = 000248, loss = 0.001726
grad_step = 000249, loss = 0.001759
grad_step = 000250, loss = 0.001810
grad_step = 000251, loss = 0.001792
grad_step = 000252, loss = 0.001736
grad_step = 000253, loss = 0.001720
grad_step = 000254, loss = 0.001756
grad_step = 000255, loss = 0.001774
grad_step = 000256, loss = 0.001746
grad_step = 000257, loss = 0.001714
grad_step = 000258, loss = 0.001718
grad_step = 000259, loss = 0.001739
grad_step = 000260, loss = 0.001745
grad_step = 000261, loss = 0.001728
grad_step = 000262, loss = 0.001707
grad_step = 000263, loss = 0.001704
grad_step = 000264, loss = 0.001717
grad_step = 000265, loss = 0.001723
grad_step = 000266, loss = 0.001713
grad_step = 000267, loss = 0.001700
grad_step = 000268, loss = 0.001696
grad_step = 000269, loss = 0.001699
grad_step = 000270, loss = 0.001705
grad_step = 000271, loss = 0.001703
grad_step = 000272, loss = 0.001694
grad_step = 000273, loss = 0.001687
grad_step = 000274, loss = 0.001686
grad_step = 000275, loss = 0.001688
grad_step = 000276, loss = 0.001689
grad_step = 000277, loss = 0.001688
grad_step = 000278, loss = 0.001684
grad_step = 000279, loss = 0.001679
grad_step = 000280, loss = 0.001675
grad_step = 000281, loss = 0.001674
grad_step = 000282, loss = 0.001674
grad_step = 000283, loss = 0.001674
grad_step = 000284, loss = 0.001673
grad_step = 000285, loss = 0.001671
grad_step = 000286, loss = 0.001668
grad_step = 000287, loss = 0.001665
grad_step = 000288, loss = 0.001662
grad_step = 000289, loss = 0.001660
grad_step = 000290, loss = 0.001659
grad_step = 000291, loss = 0.001657
grad_step = 000292, loss = 0.001657
grad_step = 000293, loss = 0.001656
grad_step = 000294, loss = 0.001655
grad_step = 000295, loss = 0.001655
grad_step = 000296, loss = 0.001655
grad_step = 000297, loss = 0.001655
grad_step = 000298, loss = 0.001657
grad_step = 000299, loss = 0.001660
grad_step = 000300, loss = 0.001664
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001673
grad_step = 000302, loss = 0.001688
grad_step = 000303, loss = 0.001714
grad_step = 000304, loss = 0.001754
grad_step = 000305, loss = 0.001822
grad_step = 000306, loss = 0.001912
grad_step = 000307, loss = 0.002034
grad_step = 000308, loss = 0.002123
grad_step = 000309, loss = 0.002147
grad_step = 000310, loss = 0.002013
grad_step = 000311, loss = 0.001808
grad_step = 000312, loss = 0.001649
grad_step = 000313, loss = 0.001644
grad_step = 000314, loss = 0.001753
grad_step = 000315, loss = 0.001852
grad_step = 000316, loss = 0.001856
grad_step = 000317, loss = 0.001750
grad_step = 000318, loss = 0.001646
grad_step = 000319, loss = 0.001617
grad_step = 000320, loss = 0.001667
grad_step = 000321, loss = 0.001733
grad_step = 000322, loss = 0.001744
grad_step = 000323, loss = 0.001702
grad_step = 000324, loss = 0.001637
grad_step = 000325, loss = 0.001606
grad_step = 000326, loss = 0.001623
grad_step = 000327, loss = 0.001658
grad_step = 000328, loss = 0.001673
grad_step = 000329, loss = 0.001654
grad_step = 000330, loss = 0.001619
grad_step = 000331, loss = 0.001598
grad_step = 000332, loss = 0.001601
grad_step = 000333, loss = 0.001620
grad_step = 000334, loss = 0.001634
grad_step = 000335, loss = 0.001635
grad_step = 000336, loss = 0.001622
grad_step = 000337, loss = 0.001603
grad_step = 000338, loss = 0.001589
grad_step = 000339, loss = 0.001584
grad_step = 000340, loss = 0.001589
grad_step = 000341, loss = 0.001596
grad_step = 000342, loss = 0.001601
grad_step = 000343, loss = 0.001601
grad_step = 000344, loss = 0.001597
grad_step = 000345, loss = 0.001589
grad_step = 000346, loss = 0.001580
grad_step = 000347, loss = 0.001574
grad_step = 000348, loss = 0.001570
grad_step = 000349, loss = 0.001568
grad_step = 000350, loss = 0.001568
grad_step = 000351, loss = 0.001569
grad_step = 000352, loss = 0.001570
grad_step = 000353, loss = 0.001572
grad_step = 000354, loss = 0.001575
grad_step = 000355, loss = 0.001578
grad_step = 000356, loss = 0.001581
grad_step = 000357, loss = 0.001587
grad_step = 000358, loss = 0.001594
grad_step = 000359, loss = 0.001604
grad_step = 000360, loss = 0.001619
grad_step = 000361, loss = 0.001642
grad_step = 000362, loss = 0.001673
grad_step = 000363, loss = 0.001718
grad_step = 000364, loss = 0.001771
grad_step = 000365, loss = 0.001835
grad_step = 000366, loss = 0.001882
grad_step = 000367, loss = 0.001904
grad_step = 000368, loss = 0.001865
grad_step = 000369, loss = 0.001776
grad_step = 000370, loss = 0.001658
grad_step = 000371, loss = 0.001566
grad_step = 000372, loss = 0.001531
grad_step = 000373, loss = 0.001554
grad_step = 000374, loss = 0.001610
grad_step = 000375, loss = 0.001667
grad_step = 000376, loss = 0.001703
grad_step = 000377, loss = 0.001693
grad_step = 000378, loss = 0.001654
grad_step = 000379, loss = 0.001596
grad_step = 000380, loss = 0.001544
grad_step = 000381, loss = 0.001517
grad_step = 000382, loss = 0.001519
grad_step = 000383, loss = 0.001540
grad_step = 000384, loss = 0.001568
grad_step = 000385, loss = 0.001592
grad_step = 000386, loss = 0.001603
grad_step = 000387, loss = 0.001603
grad_step = 000388, loss = 0.001589
grad_step = 000389, loss = 0.001568
grad_step = 000390, loss = 0.001543
grad_step = 000391, loss = 0.001521
grad_step = 000392, loss = 0.001505
grad_step = 000393, loss = 0.001496
grad_step = 000394, loss = 0.001493
grad_step = 000395, loss = 0.001496
grad_step = 000396, loss = 0.001501
grad_step = 000397, loss = 0.001510
grad_step = 000398, loss = 0.001520
grad_step = 000399, loss = 0.001533
grad_step = 000400, loss = 0.001550
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001570
grad_step = 000402, loss = 0.001598
grad_step = 000403, loss = 0.001628
grad_step = 000404, loss = 0.001666
grad_step = 000405, loss = 0.001701
grad_step = 000406, loss = 0.001731
grad_step = 000407, loss = 0.001737
grad_step = 000408, loss = 0.001714
grad_step = 000409, loss = 0.001655
grad_step = 000410, loss = 0.001577
grad_step = 000411, loss = 0.001506
grad_step = 000412, loss = 0.001467
grad_step = 000413, loss = 0.001466
grad_step = 000414, loss = 0.001494
grad_step = 000415, loss = 0.001536
grad_step = 000416, loss = 0.001576
grad_step = 000417, loss = 0.001613
grad_step = 000418, loss = 0.001632
grad_step = 000419, loss = 0.001634
grad_step = 000420, loss = 0.001608
grad_step = 000421, loss = 0.001565
grad_step = 000422, loss = 0.001514
grad_step = 000423, loss = 0.001471
grad_step = 000424, loss = 0.001445
grad_step = 000425, loss = 0.001439
grad_step = 000426, loss = 0.001448
grad_step = 000427, loss = 0.001467
grad_step = 000428, loss = 0.001489
grad_step = 000429, loss = 0.001514
grad_step = 000430, loss = 0.001540
grad_step = 000431, loss = 0.001562
grad_step = 000432, loss = 0.001583
grad_step = 000433, loss = 0.001590
grad_step = 000434, loss = 0.001586
grad_step = 000435, loss = 0.001564
grad_step = 000436, loss = 0.001529
grad_step = 000437, loss = 0.001488
grad_step = 000438, loss = 0.001450
grad_step = 000439, loss = 0.001423
grad_step = 000440, loss = 0.001409
grad_step = 000441, loss = 0.001409
grad_step = 000442, loss = 0.001418
grad_step = 000443, loss = 0.001433
grad_step = 000444, loss = 0.001452
grad_step = 000445, loss = 0.001474
grad_step = 000446, loss = 0.001500
grad_step = 000447, loss = 0.001535
grad_step = 000448, loss = 0.001576
grad_step = 000449, loss = 0.001620
grad_step = 000450, loss = 0.001654
grad_step = 000451, loss = 0.001667
grad_step = 000452, loss = 0.001644
grad_step = 000453, loss = 0.001585
grad_step = 000454, loss = 0.001503
grad_step = 000455, loss = 0.001427
grad_step = 000456, loss = 0.001381
grad_step = 000457, loss = 0.001374
grad_step = 000458, loss = 0.001397
grad_step = 000459, loss = 0.001435
grad_step = 000460, loss = 0.001476
grad_step = 000461, loss = 0.001514
grad_step = 000462, loss = 0.001550
grad_step = 000463, loss = 0.001571
grad_step = 000464, loss = 0.001580
grad_step = 000465, loss = 0.001557
grad_step = 000466, loss = 0.001514
grad_step = 000467, loss = 0.001455
grad_step = 000468, loss = 0.001398
grad_step = 000469, loss = 0.001357
grad_step = 000470, loss = 0.001338
grad_step = 000471, loss = 0.001342
grad_step = 000472, loss = 0.001363
grad_step = 000473, loss = 0.001389
grad_step = 000474, loss = 0.001412
grad_step = 000475, loss = 0.001428
grad_step = 000476, loss = 0.001439
grad_step = 000477, loss = 0.001454
grad_step = 000478, loss = 0.001464
grad_step = 000479, loss = 0.001470
grad_step = 000480, loss = 0.001457
grad_step = 000481, loss = 0.001429
grad_step = 000482, loss = 0.001390
grad_step = 000483, loss = 0.001353
grad_step = 000484, loss = 0.001325
grad_step = 000485, loss = 0.001303
grad_step = 000486, loss = 0.001293
grad_step = 000487, loss = 0.001298
grad_step = 000488, loss = 0.001316
grad_step = 000489, loss = 0.001341
grad_step = 000490, loss = 0.001369
grad_step = 000491, loss = 0.001408
grad_step = 000492, loss = 0.001476
grad_step = 000493, loss = 0.001579
grad_step = 000494, loss = 0.001727
grad_step = 000495, loss = 0.001830
grad_step = 000496, loss = 0.001840
grad_step = 000497, loss = 0.001672
grad_step = 000498, loss = 0.001449
grad_step = 000499, loss = 0.001323
grad_step = 000500, loss = 0.001331
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001413
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

  date_run                              2020-05-15 17:13:33.793040
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.252571
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 17:13:33.799938
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.170944
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 17:13:33.808372
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.141735
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 17:13:33.814672
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.59755
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
0   2020-05-15 17:13:01.326288  ...    mean_absolute_error
1   2020-05-15 17:13:01.331177  ...     mean_squared_error
2   2020-05-15 17:13:01.335174  ...  median_absolute_error
3   2020-05-15 17:13:01.339980  ...               r2_score
4   2020-05-15 17:13:10.774627  ...    mean_absolute_error
5   2020-05-15 17:13:10.779666  ...     mean_squared_error
6   2020-05-15 17:13:10.783382  ...  median_absolute_error
7   2020-05-15 17:13:10.786979  ...               r2_score
8   2020-05-15 17:13:33.793040  ...    mean_absolute_error
9   2020-05-15 17:13:33.799938  ...     mean_squared_error
10  2020-05-15 17:13:33.808372  ...  median_absolute_error
11  2020-05-15 17:13:33.814672  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f35f8484a58> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 304886.01it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 394666.23it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 545872.01it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 770911.60it/s] 75%|███████▌  | 7446528/9912422 [00:00<00:02, 1089401.72it/s]9920512it [00:01, 9867337.17it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 138790.22it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 300243.70it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 387492.58it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 536111.94it/s]1654784it [00:00, 2691552.92it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 50623.05it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f35aae33e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f35a7c7f0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f35aae33e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f35a7c7f048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f35f843ceb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f35a7c7f0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f35aae33e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f35a7c7f048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f35f843ceb8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f35f8484a58> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f5ed00431d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=f34a9e39a475616e360b917ba6c43107b74f029394926216cb1d56e79c7d60ee
  Stored in directory: /tmp/pip-ephem-wheel-cache-ple8iel4/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f5e67e3e780> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 47s
   57344/17464789 [..............................] - ETA: 41s
  106496/17464789 [..............................] - ETA: 33s
  229376/17464789 [..............................] - ETA: 20s
  458752/17464789 [..............................] - ETA: 12s
  942080/17464789 [>.............................] - ETA: 7s 
 1916928/17464789 [==>...........................] - ETA: 3s
 3850240/17464789 [=====>........................] - ETA: 1s
 6750208/17464789 [==========>...................] - ETA: 0s
 7749632/17464789 [============>.................] - ETA: 0s
 9699328/17464789 [===============>..............] - ETA: 0s
11517952/17464789 [==================>...........] - ETA: 0s
14319616/17464789 [=======================>......] - ETA: 0s
16990208/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 17:15:06.866247: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 17:15:06.870872: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-15 17:15:06.871175: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55dc1af1d1c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 17:15:06.871194: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.7126 - accuracy: 0.4970
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6666 - accuracy: 0.5000
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.5491 - accuracy: 0.5077 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5938 - accuracy: 0.5048
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6268 - accuracy: 0.5026
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6053 - accuracy: 0.5040
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6360 - accuracy: 0.5020
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6360 - accuracy: 0.5020
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6394 - accuracy: 0.5018
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6145 - accuracy: 0.5034
11000/25000 [============>.................] - ETA: 4s - loss: 7.5941 - accuracy: 0.5047
12000/25000 [=============>................] - ETA: 4s - loss: 7.6027 - accuracy: 0.5042
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6395 - accuracy: 0.5018
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6655 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 3s - loss: 7.6901 - accuracy: 0.4985
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6963 - accuracy: 0.4981
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6774 - accuracy: 0.4993
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6586 - accuracy: 0.5005
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6778 - accuracy: 0.4993
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6866 - accuracy: 0.4987
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6737 - accuracy: 0.4995
25000/25000 [==============================] - 9s 378us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 17:15:23.889715
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 17:15:23.889715  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<9:47:01, 24.5kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:00<6:51:50, 34.9kB/s] .vector_cache/glove.6B.zip:   1%|          | 6.46M/862M [00:00<4:46:20, 49.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:00<3:19:07, 71.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.6M/862M [00:00<2:18:37, 102kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 25.5M/862M [00:00<1:36:10, 145kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.9M/862M [00:00<1:06:52, 207kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.4M/862M [00:01<46:26, 295kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 48.3M/862M [00:01<32:12, 421kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:01<23:02, 586kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:03<18:01, 745kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:03<13:25, 1.00MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:05<11:24, 1.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:05<08:28, 1.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:07<08:13, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:07<06:13, 2.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:09<06:40, 1.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:09<05:36, 2.35MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:11<05:55, 2.22MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:11<04:34, 2.87MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:13<05:28, 2.39MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:13<04:17, 3.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:15<05:16, 2.46MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:15<04:08, 3.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:17<05:06, 2.53MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:17<04:02, 3.20MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:19<05:04, 2.53MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:19<04:32, 2.84MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:21<05:05, 2.51MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:21<04:02, 3.16MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:23<05:01, 2.53MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:23<03:58, 3.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<05:00, 2.53MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:25<03:57, 3.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<04:58, 2.53MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:27<03:56, 3.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:27<03:04, 4.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:28<02:29, 5.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<16:47:57, 12.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<11:46:09, 17.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<8:13:45, 25.2kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<5:45:58, 35.9kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<4:02:59, 50.9kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<2:50:52, 72.3kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<2:00:54, 102kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<1:25:16, 144kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<1:01:18, 199kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:38<43:19, 282kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<32:17, 376kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:40<23:03, 527kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<18:17, 661kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<14:03, 859kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:42<09:54, 1.21MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<25:27, 472kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<18:14, 658kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<14:48, 807kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<10:57, 1.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:46<07:58, 1.49MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<12:35:02, 15.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<8:48:22, 22.4kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<6:10:07, 31.9kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<4:19:16, 45.5kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<3:02:34, 64.3kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<2:08:11, 91.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<1:31:11, 128kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<1:04:10, 182kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<46:45, 248kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<33:12, 349kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<25:00, 461kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<17:54, 643kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<14:24, 795kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<10:29, 1.09MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<09:14, 1.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<06:51, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<06:43, 1.68MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<05:10, 2.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<05:27, 2.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<04:37, 2.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:54, 2.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<03:49, 2.91MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<04:37, 2.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<03:37, 3.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:25, 2.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<03:29, 3.16MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<04:17, 2.55MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<03:28, 3.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<04:12, 2.60MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<03:20, 3.26MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<04:09, 2.60MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:24, 3.17MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<04:06, 2.62MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<03:13, 3.33MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:21<04:06, 2.60MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<03:15, 3.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<04:08, 2.57MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<03:17, 3.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 228M/862M [01:25<04:08, 2.55MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<03:18, 3.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<04:03, 2.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<03:14, 3.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:29<03:59, 2.61MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<03:17, 3.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<03:57, 2.62MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<03:10, 3.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<03:55, 2.62MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<03:10, 3.24MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<03:53, 2.63MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<03:05, 3.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<03:55, 2.58MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<03:06, 3.26MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<03:56, 2.56MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:30, 2.87MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<03:57, 2.53MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:41<03:08, 3.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<03:55, 2.54MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<03:05, 3.21MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<03:53, 2.53MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<03:04, 3.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<03:52, 2.53MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<03:03, 3.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<03:50, 2.53MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<03:02, 3.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:48, 2.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<03:00, 3.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<03:46, 2.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<02:59, 3.20MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<03:44, 2.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<02:58, 3.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<03:43, 2.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<02:55, 3.23MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:39, 2.57MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<02:52, 3.27MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<03:36, 2.58MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<02:53, 3.22MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<03:32, 2.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<02:57, 3.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:30, 2.62MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<02:50, 3.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<03:28, 2.63MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<02:45, 3.30MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:30, 2.58MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<02:46, 3.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<03:27, 2.59MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<02:43, 3.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:25, 2.60MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<02:44, 3.25MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<03:25, 2.57MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<02:43, 3.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:25, 2.56MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<02:45, 3.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:20, 2.60MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<02:39, 3.27MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<03:21, 2.57MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<02:40, 3.22MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:20, 2.56MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<02:39, 3.22MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:20, 2.54MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<02:56, 2.87MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:20, 2.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<02:38, 3.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:17, 2.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<02:36, 3.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:16, 2.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<02:34, 3.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:14, 2.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:34, 3.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<03:09, 2.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<02:28, 3.29MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:08, 2.57MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<02:29, 3.23MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:04, 2.60MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:31, 3.17MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:02, 2.61MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:29, 3.18MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:01, 2.60MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:25, 3.24MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:01, 2.58MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:24, 3.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:00, 2.56MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:23, 3.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<03:00, 2.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<02:25, 3.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:55, 2.59MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:19, 3.27MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<02:56, 2.56MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:21, 3.19MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<02:52, 2.60MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:16, 3.27MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:52, 2.56MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<02:14, 3.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<02:51, 2.56MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<02:16, 3.22MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<02:50, 2.56MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<02:14, 3.22MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<02:49, 2.55MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<02:14, 3.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:44, 2.59MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:04<02:09, 3.29MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:43, 2.59MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:06<02:10, 3.24MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<02:42, 2.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<02:09, 3.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<02:41, 2.56MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:10<02:08, 3.21MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:40, 2.55MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<02:05, 3.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:38, 2.57MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:05, 3.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:16<02:37, 2.55MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:16<02:06, 3.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:33, 2.59MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<02:01, 3.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<02:31, 2.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:20<01:59, 3.27MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:31, 2.56MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:22<02:00, 3.22MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:36, 2.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:38, 2.41MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<02:02, 3.10MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<02:31, 2.49MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<01:58, 3.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:26, 2.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<01:56, 3.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:24, 2.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<01:54, 3.21MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:23, 2.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<08:35, 706kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:34<06:07, 987kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<05:21, 1.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<03:59, 1.50MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<03:46, 1.57MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<02:51, 2.07MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:57, 1.98MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<02:15, 2.59MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:34, 2.24MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:01, 2.86MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<02:21, 2.42MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<01:51, 3.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:16, 2.48MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<02:01, 2.80MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<02:14, 2.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<01:46, 3.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:11, 2.51MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<01:44, 3.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:09, 2.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:42, 3.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:07, 2.54MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<02:05, 2.58MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:54<01:29, 3.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<04:53, 1.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<03:35, 1.48MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<03:25, 1.53MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:47, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:43, 1.90MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<02:05, 2.47MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:20, 2.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<01:48, 2.82MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:07, 2.37MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<01:39, 3.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:01, 2.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:35, 3.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:57, 2.50MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:34, 3.12MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:52, 2.57MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:10<01:27, 3.28MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:51, 2.57MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<01:27, 3.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:50, 2.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<01:27, 3.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:49, 2.54MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:25, 3.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:46, 2.57MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:23, 3.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<01:44, 2.59MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:22, 3.26MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<01:43, 2.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:31, 2.88MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:43, 2.53MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:21, 3.18MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<01:40, 2.54MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<01:22, 3.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<01:37, 2.59MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<01:17, 3.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:37, 2.56MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:29<01:15, 3.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:34, 2.58MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:31<01:24, 2.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:34, 2.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:33<01:14, 3.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:32, 2.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<01:12, 3.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:30, 2.55MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<01:12, 3.19MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<01:29, 2.55MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:39<01:10, 3.20MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:27, 2.54MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<01:09, 3.20MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:26, 2.54MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<01:08, 3.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:24, 2.54MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:06, 3.23MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<01:22, 2.57MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:04, 3.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:21, 2.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:03, 3.22MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:51<01:19, 2.54MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:03, 3.20MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:16, 2.58MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<01:00, 3.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:15, 2.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<00:59, 3.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:13, 2.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<00:57, 3.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:11, 2.59MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<00:55, 3.32MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:10, 2.58MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<00:55, 3.24MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:09, 2.55MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<00:54, 3.22MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:08, 2.54MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<00:54, 3.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:06, 2.55MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:07<00:52, 3.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:04, 2.55MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:09<00:51, 3.21MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:03, 2.54MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<00:50, 3.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:01, 2.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<00:48, 3.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:00, 2.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:47, 3.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:58, 2.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<00:46, 3.21MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:57, 2.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:44, 3.20MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<00:55, 2.54MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<00:43, 3.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:53, 2.56MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<00:41, 3.23MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:51, 2.55MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:40, 3.20MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:50, 2.55MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:39, 3.20MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:48, 2.55MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:43, 2.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:47, 2.52MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:38, 3.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:44, 2.58MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:40, 2.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:44, 2.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:34, 3.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:42, 2.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:33, 3.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:40, 2.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:31, 3.25MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:38, 2.56MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:30, 3.24MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:37, 2.55MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:29, 3.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:35, 2.54MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:27, 3.24MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:46<00:33, 2.56MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:26, 3.23MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:48<00:32, 2.55MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:25, 3.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:30, 2.59MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:24, 3.18MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:28, 2.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:52<00:22, 3.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:27, 2.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:20, 3.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:25, 2.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:19, 3.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:23, 2.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:18, 3.30MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:22, 2.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:00<00:17, 3.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:20, 2.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:16, 3.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:19, 2.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:04<00:15, 3.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:17, 2.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:18, 2.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:06<00:12, 3.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:32, 1.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:24, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:21, 1.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:16, 2.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:15, 2.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:12, 2.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:12, 2.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:09, 2.90MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:10, 2.43MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:08, 2.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:08, 2.52MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:07, 2.80MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:06, 2.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:04, 3.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:04, 2.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:03, 3.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:03, 2.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:02, 3.19MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 2.53MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:00, 3.20MB/s].vector_cache/glove.6B.zip: 862MB [06:26, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 661/400000 [00:00<01:00, 6602.14it/s]  0%|          | 1318/400000 [00:00<01:00, 6590.21it/s]  1%|          | 2012/400000 [00:00<00:59, 6690.12it/s]  1%|          | 2808/400000 [00:00<00:56, 7026.34it/s]  1%|          | 3648/400000 [00:00<00:53, 7388.17it/s]  1%|          | 4496/400000 [00:00<00:51, 7682.99it/s]  1%|▏         | 5321/400000 [00:00<00:50, 7842.65it/s]  2%|▏         | 6077/400000 [00:00<00:50, 7755.13it/s]  2%|▏         | 6854/400000 [00:00<00:50, 7759.17it/s]  2%|▏         | 7691/400000 [00:01<00:49, 7932.46it/s]  2%|▏         | 8521/400000 [00:01<00:48, 8038.36it/s]  2%|▏         | 9379/400000 [00:01<00:47, 8192.96it/s]  3%|▎         | 10241/400000 [00:01<00:46, 8315.99it/s]  3%|▎         | 11069/400000 [00:01<00:48, 8064.64it/s]  3%|▎         | 11911/400000 [00:01<00:47, 8167.20it/s]  3%|▎         | 12728/400000 [00:01<00:48, 8017.49it/s]  3%|▎         | 13561/400000 [00:01<00:47, 8108.38it/s]  4%|▎         | 14394/400000 [00:01<00:47, 8171.31it/s]  4%|▍         | 15220/400000 [00:01<00:46, 8195.46it/s]  4%|▍         | 16069/400000 [00:02<00:46, 8281.25it/s]  4%|▍         | 16902/400000 [00:02<00:46, 8295.49it/s]  4%|▍         | 17770/400000 [00:02<00:45, 8406.29it/s]  5%|▍         | 18612/400000 [00:02<00:47, 7993.72it/s]  5%|▍         | 19417/400000 [00:02<00:49, 7672.21it/s]  5%|▌         | 20197/400000 [00:02<00:49, 7708.99it/s]  5%|▌         | 21047/400000 [00:02<00:47, 7928.25it/s]  5%|▌         | 21893/400000 [00:02<00:46, 8080.54it/s]  6%|▌         | 22705/400000 [00:02<00:49, 7662.21it/s]  6%|▌         | 23479/400000 [00:02<00:50, 7431.19it/s]  6%|▌         | 24334/400000 [00:03<00:48, 7734.18it/s]  6%|▋         | 25181/400000 [00:03<00:47, 7938.11it/s]  7%|▋         | 26025/400000 [00:03<00:46, 8081.24it/s]  7%|▋         | 26839/400000 [00:03<00:46, 7963.99it/s]  7%|▋         | 27640/400000 [00:03<00:47, 7781.48it/s]  7%|▋         | 28422/400000 [00:03<00:50, 7424.48it/s]  7%|▋         | 29171/400000 [00:03<00:51, 7198.00it/s]  8%|▊         | 30016/400000 [00:03<00:49, 7531.96it/s]  8%|▊         | 30844/400000 [00:03<00:47, 7739.89it/s]  8%|▊         | 31665/400000 [00:04<00:46, 7873.53it/s]  8%|▊         | 32458/400000 [00:04<00:48, 7503.81it/s]  8%|▊         | 33216/400000 [00:04<00:50, 7208.32it/s]  8%|▊         | 33945/400000 [00:04<00:52, 6969.99it/s]  9%|▊         | 34702/400000 [00:04<00:51, 7139.41it/s]  9%|▉         | 35549/400000 [00:04<00:48, 7492.31it/s]  9%|▉         | 36393/400000 [00:04<00:46, 7752.53it/s]  9%|▉         | 37246/400000 [00:04<00:45, 7968.58it/s] 10%|▉         | 38054/400000 [00:04<00:45, 7998.51it/s] 10%|▉         | 38872/400000 [00:04<00:44, 8050.00it/s] 10%|▉         | 39681/400000 [00:05<00:47, 7608.13it/s] 10%|█         | 40450/400000 [00:05<00:49, 7285.77it/s] 10%|█         | 41187/400000 [00:05<00:50, 7052.83it/s] 10%|█         | 41900/400000 [00:05<00:52, 6858.20it/s] 11%|█         | 42593/400000 [00:05<00:52, 6843.48it/s] 11%|█         | 43313/400000 [00:05<00:51, 6946.03it/s] 11%|█         | 44012/400000 [00:05<00:52, 6735.51it/s] 11%|█         | 44796/400000 [00:05<00:50, 7031.83it/s] 11%|█▏        | 45506/400000 [00:05<00:51, 6901.20it/s] 12%|█▏        | 46201/400000 [00:06<00:51, 6895.28it/s] 12%|█▏        | 47061/400000 [00:06<00:48, 7329.52it/s] 12%|█▏        | 47933/400000 [00:06<00:45, 7697.04it/s] 12%|█▏        | 48776/400000 [00:06<00:44, 7901.46it/s] 12%|█▏        | 49576/400000 [00:06<00:44, 7818.86it/s] 13%|█▎        | 50365/400000 [00:06<00:47, 7418.60it/s] 13%|█▎        | 51116/400000 [00:06<00:48, 7224.21it/s] 13%|█▎        | 51846/400000 [00:06<00:48, 7213.01it/s] 13%|█▎        | 52717/400000 [00:06<00:45, 7604.85it/s] 13%|█▎        | 53537/400000 [00:06<00:44, 7774.10it/s] 14%|█▎        | 54360/400000 [00:07<00:43, 7903.36it/s] 14%|█▍        | 55157/400000 [00:07<00:43, 7909.08it/s] 14%|█▍        | 55988/400000 [00:07<00:42, 8022.79it/s] 14%|█▍        | 56794/400000 [00:07<00:45, 7591.25it/s] 14%|█▍        | 57561/400000 [00:07<00:47, 7254.06it/s] 15%|█▍        | 58295/400000 [00:07<00:47, 7119.53it/s] 15%|█▍        | 59105/400000 [00:07<00:46, 7385.96it/s] 15%|█▍        | 59954/400000 [00:07<00:44, 7684.02it/s] 15%|█▌        | 60811/400000 [00:07<00:42, 7928.00it/s] 15%|█▌        | 61637/400000 [00:08<00:42, 8024.73it/s] 16%|█▌        | 62466/400000 [00:08<00:41, 8100.00it/s] 16%|█▌        | 63281/400000 [00:08<00:43, 7679.83it/s] 16%|█▌        | 64057/400000 [00:08<00:45, 7383.49it/s] 16%|█▌        | 64803/400000 [00:08<00:47, 7069.74it/s] 16%|█▋        | 65519/400000 [00:08<00:48, 6950.15it/s] 17%|█▋        | 66352/400000 [00:08<00:45, 7312.15it/s] 17%|█▋        | 67135/400000 [00:08<00:44, 7459.95it/s] 17%|█▋        | 67953/400000 [00:08<00:43, 7661.53it/s] 17%|█▋        | 68738/400000 [00:08<00:42, 7716.78it/s] 17%|█▋        | 69574/400000 [00:09<00:41, 7897.01it/s] 18%|█▊        | 70419/400000 [00:09<00:40, 8052.13it/s] 18%|█▊        | 71228/400000 [00:09<00:40, 8045.18it/s] 18%|█▊        | 72040/400000 [00:09<00:40, 8066.84it/s] 18%|█▊        | 72878/400000 [00:09<00:40, 8157.89it/s] 18%|█▊        | 73696/400000 [00:09<00:40, 8128.87it/s] 19%|█▊        | 74555/400000 [00:09<00:39, 8260.15it/s] 19%|█▉        | 75392/400000 [00:09<00:39, 8291.39it/s] 19%|█▉        | 76223/400000 [00:09<00:39, 8221.66it/s] 19%|█▉        | 77046/400000 [00:10<00:40, 7964.67it/s] 19%|█▉        | 77865/400000 [00:10<00:40, 8029.56it/s] 20%|█▉        | 78701/400000 [00:10<00:39, 8125.10it/s] 20%|█▉        | 79566/400000 [00:10<00:38, 8273.08it/s] 20%|██        | 80402/400000 [00:10<00:38, 8298.34it/s] 20%|██        | 81234/400000 [00:10<00:38, 8189.89it/s] 21%|██        | 82055/400000 [00:10<00:40, 7779.43it/s] 21%|██        | 82871/400000 [00:10<00:40, 7889.36it/s] 21%|██        | 83664/400000 [00:10<00:40, 7807.05it/s] 21%|██        | 84526/400000 [00:10<00:39, 8032.65it/s] 21%|██▏       | 85343/400000 [00:11<00:38, 8072.25it/s] 22%|██▏       | 86193/400000 [00:11<00:38, 8195.64it/s] 22%|██▏       | 87039/400000 [00:11<00:37, 8267.15it/s] 22%|██▏       | 87895/400000 [00:11<00:37, 8349.92it/s] 22%|██▏       | 88740/400000 [00:11<00:37, 8377.76it/s] 22%|██▏       | 89579/400000 [00:11<00:37, 8346.47it/s] 23%|██▎       | 90415/400000 [00:11<00:37, 8197.25it/s] 23%|██▎       | 91254/400000 [00:11<00:37, 8252.15it/s] 23%|██▎       | 92081/400000 [00:11<00:37, 8247.68it/s] 23%|██▎       | 92907/400000 [00:11<00:37, 8249.56it/s] 23%|██▎       | 93733/400000 [00:12<00:37, 8242.70it/s] 24%|██▎       | 94558/400000 [00:12<00:39, 7707.82it/s] 24%|██▍       | 95337/400000 [00:12<00:40, 7462.65it/s] 24%|██▍       | 96185/400000 [00:12<00:39, 7739.50it/s] 24%|██▍       | 96993/400000 [00:12<00:38, 7837.62it/s] 24%|██▍       | 97794/400000 [00:12<00:38, 7888.42it/s] 25%|██▍       | 98629/400000 [00:12<00:37, 8020.96it/s] 25%|██▍       | 99454/400000 [00:12<00:37, 8087.47it/s] 25%|██▌       | 100291/400000 [00:12<00:36, 8170.11it/s] 25%|██▌       | 101135/400000 [00:12<00:36, 8247.48it/s] 25%|██▌       | 101962/400000 [00:13<00:36, 8201.94it/s] 26%|██▌       | 102824/400000 [00:13<00:35, 8322.45it/s] 26%|██▌       | 103658/400000 [00:13<00:35, 8286.43it/s] 26%|██▌       | 104502/400000 [00:13<00:35, 8331.53it/s] 26%|██▋       | 105336/400000 [00:13<00:36, 8048.97it/s] 27%|██▋       | 106144/400000 [00:13<00:38, 7566.31it/s] 27%|██▋       | 106909/400000 [00:13<00:40, 7274.74it/s] 27%|██▋       | 107645/400000 [00:13<00:41, 7090.70it/s] 27%|██▋       | 108421/400000 [00:13<00:40, 7277.72it/s] 27%|██▋       | 109201/400000 [00:14<00:39, 7425.23it/s] 27%|██▋       | 109962/400000 [00:14<00:38, 7479.10it/s] 28%|██▊       | 110818/400000 [00:14<00:37, 7773.11it/s] 28%|██▊       | 111677/400000 [00:14<00:36, 8000.33it/s] 28%|██▊       | 112522/400000 [00:14<00:35, 8129.58it/s] 28%|██▊       | 113340/400000 [00:14<00:35, 8058.48it/s] 29%|██▊       | 114150/400000 [00:14<00:35, 8034.21it/s] 29%|██▊       | 114966/400000 [00:14<00:35, 8069.86it/s] 29%|██▉       | 115775/400000 [00:14<00:35, 8018.61it/s] 29%|██▉       | 116620/400000 [00:14<00:34, 8141.25it/s] 29%|██▉       | 117482/400000 [00:15<00:34, 8276.88it/s] 30%|██▉       | 118312/400000 [00:15<00:34, 8101.32it/s] 30%|██▉       | 119124/400000 [00:15<00:36, 7596.60it/s] 30%|██▉       | 119892/400000 [00:15<00:38, 7222.18it/s] 30%|███       | 120624/400000 [00:15<00:39, 6990.73it/s] 30%|███       | 121332/400000 [00:15<00:40, 6879.84it/s] 31%|███       | 122167/400000 [00:15<00:38, 7262.98it/s] 31%|███       | 123014/400000 [00:15<00:36, 7585.92it/s] 31%|███       | 123859/400000 [00:15<00:35, 7825.58it/s] 31%|███       | 124710/400000 [00:16<00:34, 8018.99it/s] 31%|███▏      | 125543/400000 [00:16<00:33, 8108.46it/s] 32%|███▏      | 126399/400000 [00:16<00:33, 8238.30it/s] 32%|███▏      | 127239/400000 [00:16<00:32, 8285.14it/s] 32%|███▏      | 128077/400000 [00:16<00:32, 8312.49it/s] 32%|███▏      | 128915/400000 [00:16<00:32, 8330.89it/s] 32%|███▏      | 129754/400000 [00:16<00:32, 8347.55it/s] 33%|███▎      | 130590/400000 [00:16<00:32, 8289.40it/s] 33%|███▎      | 131440/400000 [00:16<00:32, 8350.48it/s] 33%|███▎      | 132280/400000 [00:16<00:32, 8362.85it/s] 33%|███▎      | 133123/400000 [00:17<00:31, 8380.02it/s] 33%|███▎      | 133962/400000 [00:17<00:32, 8283.96it/s] 34%|███▎      | 134791/400000 [00:17<00:33, 7894.63it/s] 34%|███▍      | 135585/400000 [00:17<00:35, 7513.85it/s] 34%|███▍      | 136344/400000 [00:17<00:36, 7224.73it/s] 34%|███▍      | 137074/400000 [00:17<00:37, 7057.42it/s] 34%|███▍      | 137786/400000 [00:17<00:37, 6911.54it/s] 35%|███▍      | 138638/400000 [00:17<00:35, 7324.89it/s] 35%|███▍      | 139399/400000 [00:17<00:35, 7406.19it/s] 35%|███▌      | 140228/400000 [00:17<00:33, 7650.68it/s] 35%|███▌      | 141071/400000 [00:18<00:32, 7867.09it/s] 35%|███▌      | 141905/400000 [00:18<00:32, 8000.65it/s] 36%|███▌      | 142749/400000 [00:18<00:31, 8127.15it/s] 36%|███▌      | 143577/400000 [00:18<00:31, 8171.16it/s] 36%|███▌      | 144398/400000 [00:18<00:31, 7987.86it/s] 36%|███▋      | 145200/400000 [00:18<00:33, 7569.45it/s] 36%|███▋      | 145964/400000 [00:18<00:33, 7512.95it/s] 37%|███▋      | 146799/400000 [00:18<00:32, 7745.83it/s] 37%|███▋      | 147637/400000 [00:18<00:31, 7923.72it/s] 37%|███▋      | 148469/400000 [00:19<00:31, 8036.79it/s] 37%|███▋      | 149277/400000 [00:19<00:31, 8012.19it/s] 38%|███▊      | 150081/400000 [00:19<00:31, 7938.89it/s] 38%|███▊      | 150925/400000 [00:19<00:30, 8081.14it/s] 38%|███▊      | 151777/400000 [00:19<00:30, 8206.92it/s] 38%|███▊      | 152626/400000 [00:19<00:29, 8287.93it/s] 38%|███▊      | 153457/400000 [00:19<00:29, 8254.38it/s] 39%|███▊      | 154306/400000 [00:19<00:29, 8321.74it/s] 39%|███▉      | 155156/400000 [00:19<00:29, 8373.26it/s] 39%|███▉      | 156025/400000 [00:19<00:28, 8465.67it/s] 39%|███▉      | 156892/400000 [00:20<00:28, 8525.65it/s] 39%|███▉      | 157746/400000 [00:20<00:28, 8522.19it/s] 40%|███▉      | 158599/400000 [00:20<00:29, 8250.49it/s] 40%|███▉      | 159427/400000 [00:20<00:30, 7856.74it/s] 40%|████      | 160282/400000 [00:20<00:29, 8051.15it/s] 40%|████      | 161121/400000 [00:20<00:29, 8149.09it/s] 40%|████      | 161954/400000 [00:20<00:29, 8200.51it/s] 41%|████      | 162777/400000 [00:20<00:28, 8199.06it/s] 41%|████      | 163631/400000 [00:20<00:28, 8297.28it/s] 41%|████      | 164494/400000 [00:20<00:28, 8391.63it/s] 41%|████▏     | 165335/400000 [00:21<00:27, 8390.98it/s] 42%|████▏     | 166198/400000 [00:21<00:27, 8460.31it/s] 42%|████▏     | 167045/400000 [00:21<00:27, 8429.51it/s] 42%|████▏     | 167889/400000 [00:21<00:28, 8173.00it/s] 42%|████▏     | 168709/400000 [00:21<00:30, 7672.05it/s] 42%|████▏     | 169484/400000 [00:21<00:29, 7688.22it/s] 43%|████▎     | 170321/400000 [00:21<00:29, 7878.31it/s] 43%|████▎     | 171114/400000 [00:21<00:29, 7857.08it/s] 43%|████▎     | 171931/400000 [00:21<00:28, 7946.04it/s] 43%|████▎     | 172741/400000 [00:21<00:28, 7989.67it/s] 43%|████▎     | 173542/400000 [00:22<00:28, 7916.78it/s] 44%|████▎     | 174394/400000 [00:22<00:27, 8086.26it/s] 44%|████▍     | 175234/400000 [00:22<00:27, 8176.33it/s] 44%|████▍     | 176083/400000 [00:22<00:27, 8265.80it/s] 44%|████▍     | 176941/400000 [00:22<00:26, 8356.38it/s] 44%|████▍     | 177807/400000 [00:22<00:26, 8444.92it/s] 45%|████▍     | 178653/400000 [00:22<00:26, 8402.06it/s] 45%|████▍     | 179494/400000 [00:22<00:26, 8312.11it/s] 45%|████▌     | 180326/400000 [00:22<00:28, 7812.64it/s] 45%|████▌     | 181114/400000 [00:23<00:29, 7487.27it/s] 45%|████▌     | 181871/400000 [00:23<00:29, 7361.38it/s] 46%|████▌     | 182705/400000 [00:23<00:28, 7629.90it/s] 46%|████▌     | 183475/400000 [00:23<00:29, 7464.08it/s] 46%|████▌     | 184336/400000 [00:23<00:27, 7774.05it/s] 46%|████▋     | 185153/400000 [00:23<00:27, 7886.74it/s] 46%|████▋     | 185974/400000 [00:23<00:26, 7980.47it/s] 47%|████▋     | 186793/400000 [00:23<00:26, 8040.61it/s] 47%|████▋     | 187605/400000 [00:23<00:26, 8064.10it/s] 47%|████▋     | 188414/400000 [00:23<00:26, 8019.26it/s] 47%|████▋     | 189231/400000 [00:24<00:26, 8061.19it/s] 48%|████▊     | 190039/400000 [00:24<00:26, 7939.50it/s] 48%|████▊     | 190835/400000 [00:24<00:26, 7906.80it/s] 48%|████▊     | 191627/400000 [00:24<00:26, 7831.02it/s] 48%|████▊     | 192439/400000 [00:24<00:26, 7915.40it/s] 48%|████▊     | 193267/400000 [00:24<00:25, 8018.98it/s] 49%|████▊     | 194124/400000 [00:24<00:25, 8171.15it/s] 49%|████▊     | 194969/400000 [00:24<00:24, 8251.70it/s] 49%|████▉     | 195796/400000 [00:24<00:24, 8203.95it/s] 49%|████▉     | 196618/400000 [00:24<00:26, 7725.11it/s] 49%|████▉     | 197397/400000 [00:25<00:27, 7406.20it/s] 50%|████▉     | 198145/400000 [00:25<00:28, 7055.73it/s] 50%|████▉     | 198947/400000 [00:25<00:27, 7306.92it/s] 50%|████▉     | 199798/400000 [00:25<00:26, 7629.94it/s] 50%|█████     | 200657/400000 [00:25<00:25, 7893.61it/s] 50%|█████     | 201520/400000 [00:25<00:24, 8100.47it/s] 51%|█████     | 202384/400000 [00:25<00:23, 8254.38it/s] 51%|█████     | 203219/400000 [00:25<00:23, 8282.01it/s] 51%|█████     | 204085/400000 [00:25<00:23, 8390.19it/s] 51%|█████     | 204934/400000 [00:26<00:23, 8417.06it/s] 51%|█████▏    | 205779/400000 [00:26<00:24, 7882.10it/s] 52%|█████▏    | 206576/400000 [00:26<00:25, 7664.15it/s] 52%|█████▏    | 207350/400000 [00:26<00:25, 7676.64it/s] 52%|█████▏    | 208207/400000 [00:26<00:24, 7922.42it/s] 52%|█████▏    | 209060/400000 [00:26<00:23, 8093.34it/s] 52%|█████▏    | 209885/400000 [00:26<00:23, 8139.12it/s] 53%|█████▎    | 210710/400000 [00:26<00:23, 8171.03it/s] 53%|█████▎    | 211530/400000 [00:26<00:23, 7932.88it/s] 53%|█████▎    | 212389/400000 [00:26<00:23, 8118.93it/s] 53%|█████▎    | 213251/400000 [00:27<00:22, 8261.60it/s] 54%|█████▎    | 214113/400000 [00:27<00:22, 8365.38it/s] 54%|█████▎    | 214990/400000 [00:27<00:21, 8482.27it/s] 54%|█████▍    | 215841/400000 [00:27<00:21, 8449.64it/s] 54%|█████▍    | 216688/400000 [00:27<00:22, 8204.60it/s] 54%|█████▍    | 217512/400000 [00:27<00:23, 7705.30it/s] 55%|█████▍    | 218291/400000 [00:27<00:24, 7404.96it/s] 55%|█████▍    | 219040/400000 [00:27<00:25, 7150.13it/s] 55%|█████▍    | 219885/400000 [00:27<00:24, 7494.98it/s] 55%|█████▌    | 220701/400000 [00:28<00:23, 7681.36it/s] 55%|█████▌    | 221532/400000 [00:28<00:22, 7858.97it/s] 56%|█████▌    | 222392/400000 [00:28<00:22, 8064.52it/s] 56%|█████▌    | 223242/400000 [00:28<00:21, 8189.94it/s] 56%|█████▌    | 224066/400000 [00:28<00:21, 8204.31it/s] 56%|█████▌    | 224890/400000 [00:28<00:21, 7982.51it/s] 56%|█████▋    | 225735/400000 [00:28<00:21, 8117.16it/s] 57%|█████▋    | 226595/400000 [00:28<00:21, 8254.91it/s] 57%|█████▋    | 227424/400000 [00:28<00:21, 8203.63it/s] 57%|█████▋    | 228277/400000 [00:28<00:20, 8298.38it/s] 57%|█████▋    | 229149/400000 [00:29<00:20, 8418.51it/s] 58%|█████▊    | 230006/400000 [00:29<00:20, 8462.18it/s] 58%|█████▊    | 230880/400000 [00:29<00:19, 8543.58it/s] 58%|█████▊    | 231736/400000 [00:29<00:20, 8228.50it/s] 58%|█████▊    | 232575/400000 [00:29<00:20, 8273.79it/s] 58%|█████▊    | 233405/400000 [00:29<00:20, 8184.09it/s] 59%|█████▊    | 234226/400000 [00:29<00:21, 7728.54it/s] 59%|█████▉    | 235040/400000 [00:29<00:21, 7847.49it/s] 59%|█████▉    | 235830/400000 [00:29<00:21, 7770.29it/s] 59%|█████▉    | 236671/400000 [00:29<00:20, 7950.01it/s] 59%|█████▉    | 237497/400000 [00:30<00:20, 8039.54it/s] 60%|█████▉    | 238341/400000 [00:30<00:19, 8155.50it/s] 60%|█████▉    | 239206/400000 [00:30<00:19, 8295.31it/s] 60%|██████    | 240043/400000 [00:30<00:19, 8317.30it/s] 60%|██████    | 240877/400000 [00:30<00:19, 8225.21it/s] 60%|██████    | 241728/400000 [00:30<00:19, 8306.32it/s] 61%|██████    | 242583/400000 [00:30<00:18, 8377.08it/s] 61%|██████    | 243434/400000 [00:30<00:18, 8414.08it/s] 61%|██████    | 244277/400000 [00:30<00:18, 8221.60it/s] 61%|██████▏   | 245101/400000 [00:30<00:18, 8226.83it/s] 61%|██████▏   | 245963/400000 [00:31<00:18, 8339.44it/s] 62%|██████▏   | 246799/400000 [00:31<00:19, 7891.01it/s] 62%|██████▏   | 247594/400000 [00:31<00:19, 7705.35it/s] 62%|██████▏   | 248370/400000 [00:31<00:20, 7453.95it/s] 62%|██████▏   | 249121/400000 [00:31<00:20, 7201.93it/s] 62%|██████▏   | 249847/400000 [00:31<00:21, 6968.89it/s] 63%|██████▎   | 250550/400000 [00:31<00:21, 6853.81it/s] 63%|██████▎   | 251240/400000 [00:31<00:21, 6775.58it/s] 63%|██████▎   | 251921/400000 [00:31<00:21, 6759.09it/s] 63%|██████▎   | 252600/400000 [00:32<00:21, 6715.07it/s] 63%|██████▎   | 253363/400000 [00:32<00:21, 6965.53it/s] 64%|██████▎   | 254064/400000 [00:32<00:21, 6927.68it/s] 64%|██████▎   | 254760/400000 [00:32<00:21, 6853.25it/s] 64%|██████▍   | 255448/400000 [00:32<00:21, 6785.62it/s] 64%|██████▍   | 256129/400000 [00:32<00:21, 6761.82it/s] 64%|██████▍   | 256931/400000 [00:32<00:20, 7094.71it/s] 64%|██████▍   | 257764/400000 [00:32<00:19, 7423.34it/s] 65%|██████▍   | 258616/400000 [00:32<00:18, 7719.51it/s] 65%|██████▍   | 259446/400000 [00:32<00:17, 7883.32it/s] 65%|██████▌   | 260299/400000 [00:33<00:17, 8066.36it/s] 65%|██████▌   | 261112/400000 [00:33<00:17, 7980.24it/s] 65%|██████▌   | 261944/400000 [00:33<00:17, 8077.22it/s] 66%|██████▌   | 262768/400000 [00:33<00:16, 8124.68it/s] 66%|██████▌   | 263583/400000 [00:33<00:17, 7956.91it/s] 66%|██████▌   | 264425/400000 [00:33<00:16, 8089.40it/s] 66%|██████▋   | 265237/400000 [00:33<00:16, 7953.93it/s] 67%|██████▋   | 266073/400000 [00:33<00:16, 8065.62it/s] 67%|██████▋   | 266917/400000 [00:33<00:16, 8174.18it/s] 67%|██████▋   | 267768/400000 [00:34<00:15, 8271.68it/s] 67%|██████▋   | 268597/400000 [00:34<00:16, 8193.32it/s] 67%|██████▋   | 269466/400000 [00:34<00:15, 8335.62it/s] 68%|██████▊   | 270326/400000 [00:34<00:15, 8410.54it/s] 68%|██████▊   | 271169/400000 [00:34<00:15, 8344.11it/s] 68%|██████▊   | 272005/400000 [00:34<00:15, 8274.70it/s] 68%|██████▊   | 272859/400000 [00:34<00:15, 8351.00it/s] 68%|██████▊   | 273704/400000 [00:34<00:15, 8378.73it/s] 69%|██████▊   | 274543/400000 [00:34<00:15, 8325.15it/s] 69%|██████▉   | 275376/400000 [00:34<00:15, 8252.40it/s] 69%|██████▉   | 276202/400000 [00:35<00:15, 8073.57it/s] 69%|██████▉   | 277058/400000 [00:35<00:14, 8213.44it/s] 69%|██████▉   | 277920/400000 [00:35<00:14, 8324.94it/s] 70%|██████▉   | 278780/400000 [00:35<00:14, 8404.67it/s] 70%|██████▉   | 279622/400000 [00:35<00:14, 8207.52it/s] 70%|███████   | 280453/400000 [00:35<00:14, 8235.84it/s] 70%|███████   | 281278/400000 [00:35<00:14, 8023.67it/s] 71%|███████   | 282092/400000 [00:35<00:14, 8058.03it/s] 71%|███████   | 282927/400000 [00:35<00:14, 8143.10it/s] 71%|███████   | 283766/400000 [00:35<00:14, 8213.60it/s] 71%|███████   | 284599/400000 [00:36<00:13, 8247.75it/s] 71%|███████▏  | 285425/400000 [00:36<00:14, 8118.19it/s] 72%|███████▏  | 286286/400000 [00:36<00:13, 8258.31it/s] 72%|███████▏  | 287128/400000 [00:36<00:13, 8303.63it/s] 72%|███████▏  | 287960/400000 [00:36<00:13, 8307.81it/s] 72%|███████▏  | 288792/400000 [00:36<00:13, 8223.22it/s] 72%|███████▏  | 289615/400000 [00:36<00:13, 7977.65it/s] 73%|███████▎  | 290455/400000 [00:36<00:13, 8098.91it/s] 73%|███████▎  | 291320/400000 [00:36<00:13, 8254.51it/s] 73%|███████▎  | 292152/400000 [00:36<00:13, 8272.59it/s] 73%|███████▎  | 292981/400000 [00:37<00:12, 8265.09it/s] 73%|███████▎  | 293829/400000 [00:37<00:12, 8326.99it/s] 74%|███████▎  | 294686/400000 [00:37<00:12, 8397.90it/s] 74%|███████▍  | 295527/400000 [00:37<00:12, 8201.21it/s] 74%|███████▍  | 296349/400000 [00:37<00:12, 8201.04it/s] 74%|███████▍  | 297171/400000 [00:37<00:13, 7606.99it/s] 74%|███████▍  | 297942/400000 [00:37<00:13, 7301.82it/s] 75%|███████▍  | 298682/400000 [00:37<00:14, 7086.05it/s] 75%|███████▍  | 299423/400000 [00:37<00:14, 7178.90it/s] 75%|███████▌  | 300201/400000 [00:38<00:13, 7347.48it/s] 75%|███████▌  | 301028/400000 [00:38<00:13, 7599.64it/s] 75%|███████▌  | 301800/400000 [00:38<00:12, 7632.37it/s] 76%|███████▌  | 302568/400000 [00:38<00:13, 7426.15it/s] 76%|███████▌  | 303421/400000 [00:38<00:12, 7724.91it/s] 76%|███████▌  | 304232/400000 [00:38<00:12, 7836.38it/s] 76%|███████▋  | 305021/400000 [00:38<00:12, 7445.90it/s] 76%|███████▋  | 305773/400000 [00:38<00:13, 7233.77it/s] 77%|███████▋  | 306503/400000 [00:38<00:13, 7023.17it/s] 77%|███████▋  | 307349/400000 [00:38<00:12, 7393.41it/s] 77%|███████▋  | 308098/400000 [00:39<00:12, 7293.30it/s] 77%|███████▋  | 308835/400000 [00:39<00:12, 7132.01it/s] 77%|███████▋  | 309554/400000 [00:39<00:12, 6963.17it/s] 78%|███████▊  | 310383/400000 [00:39<00:12, 7313.44it/s] 78%|███████▊  | 311246/400000 [00:39<00:11, 7662.24it/s] 78%|███████▊  | 312087/400000 [00:39<00:11, 7870.06it/s] 78%|███████▊  | 312953/400000 [00:39<00:10, 8090.78it/s] 78%|███████▊  | 313770/400000 [00:39<00:10, 7890.04it/s] 79%|███████▊  | 314566/400000 [00:39<00:10, 7782.75it/s] 79%|███████▉  | 315378/400000 [00:40<00:10, 7880.37it/s] 79%|███████▉  | 316224/400000 [00:40<00:10, 8045.12it/s] 79%|███████▉  | 317084/400000 [00:40<00:10, 8201.95it/s] 79%|███████▉  | 317923/400000 [00:40<00:09, 8255.38it/s] 80%|███████▉  | 318771/400000 [00:40<00:09, 8320.36it/s] 80%|███████▉  | 319626/400000 [00:40<00:09, 8385.82it/s] 80%|████████  | 320466/400000 [00:40<00:09, 8360.75it/s] 80%|████████  | 321328/400000 [00:40<00:09, 8434.74it/s] 81%|████████  | 322173/400000 [00:40<00:09, 8315.73it/s] 81%|████████  | 323006/400000 [00:40<00:09, 8021.94it/s] 81%|████████  | 323812/400000 [00:41<00:10, 7581.71it/s] 81%|████████  | 324578/400000 [00:41<00:10, 7295.99it/s] 81%|████████▏ | 325315/400000 [00:41<00:10, 7214.87it/s] 82%|████████▏ | 326166/400000 [00:41<00:09, 7557.58it/s] 82%|████████▏ | 326930/400000 [00:41<00:09, 7419.31it/s] 82%|████████▏ | 327754/400000 [00:41<00:09, 7646.53it/s] 82%|████████▏ | 328525/400000 [00:41<00:09, 7597.38it/s] 82%|████████▏ | 329344/400000 [00:41<00:09, 7765.64it/s] 83%|████████▎ | 330178/400000 [00:41<00:08, 7929.10it/s] 83%|████████▎ | 331015/400000 [00:41<00:08, 8054.73it/s] 83%|████████▎ | 331858/400000 [00:42<00:08, 8161.85it/s] 83%|████████▎ | 332720/400000 [00:42<00:08, 8293.09it/s] 83%|████████▎ | 333588/400000 [00:42<00:07, 8403.29it/s] 84%|████████▎ | 334456/400000 [00:42<00:07, 8482.02it/s] 84%|████████▍ | 335306/400000 [00:42<00:07, 8466.13it/s] 84%|████████▍ | 336154/400000 [00:42<00:08, 7872.07it/s] 84%|████████▍ | 336951/400000 [00:42<00:08, 7819.48it/s] 84%|████████▍ | 337810/400000 [00:42<00:07, 8033.24it/s] 85%|████████▍ | 338667/400000 [00:42<00:07, 8185.05it/s] 85%|████████▍ | 339535/400000 [00:43<00:07, 8325.85it/s] 85%|████████▌ | 340372/400000 [00:43<00:07, 7980.00it/s] 85%|████████▌ | 341176/400000 [00:43<00:07, 7633.95it/s] 85%|████████▌ | 341947/400000 [00:43<00:07, 7345.95it/s] 86%|████████▌ | 342690/400000 [00:43<00:08, 7159.50it/s] 86%|████████▌ | 343517/400000 [00:43<00:07, 7451.90it/s] 86%|████████▌ | 344349/400000 [00:43<00:07, 7692.50it/s] 86%|████████▋ | 345155/400000 [00:43<00:07, 7798.85it/s] 87%|████████▋ | 346006/400000 [00:43<00:06, 7998.92it/s] 87%|████████▋ | 346841/400000 [00:43<00:06, 8099.41it/s] 87%|████████▋ | 347679/400000 [00:44<00:06, 8178.62it/s] 87%|████████▋ | 348512/400000 [00:44<00:06, 8221.00it/s] 87%|████████▋ | 349359/400000 [00:44<00:06, 8293.36it/s] 88%|████████▊ | 350218/400000 [00:44<00:05, 8378.79it/s] 88%|████████▊ | 351058/400000 [00:44<00:05, 8323.29it/s] 88%|████████▊ | 351914/400000 [00:44<00:05, 8391.48it/s] 88%|████████▊ | 352754/400000 [00:44<00:05, 8099.05it/s] 88%|████████▊ | 353567/400000 [00:44<00:05, 7908.11it/s] 89%|████████▊ | 354413/400000 [00:44<00:05, 8065.63it/s] 89%|████████▉ | 355223/400000 [00:44<00:05, 8035.66it/s] 89%|████████▉ | 356029/400000 [00:45<00:05, 7980.75it/s] 89%|████████▉ | 356829/400000 [00:45<00:05, 7826.56it/s] 89%|████████▉ | 357680/400000 [00:45<00:05, 8019.40it/s] 90%|████████▉ | 358517/400000 [00:45<00:05, 8118.30it/s] 90%|████████▉ | 359331/400000 [00:45<00:05, 8103.84it/s] 90%|█████████ | 360149/400000 [00:45<00:04, 8124.68it/s] 90%|█████████ | 361007/400000 [00:45<00:04, 8254.16it/s] 90%|█████████ | 361871/400000 [00:45<00:04, 8364.85it/s] 91%|█████████ | 362739/400000 [00:45<00:04, 8456.25it/s] 91%|█████████ | 363606/400000 [00:46<00:04, 8518.44it/s] 91%|█████████ | 364459/400000 [00:46<00:04, 8494.92it/s] 91%|█████████▏| 365325/400000 [00:46<00:04, 8543.28it/s] 92%|█████████▏| 366180/400000 [00:46<00:03, 8532.83it/s] 92%|█████████▏| 367034/400000 [00:46<00:04, 8097.88it/s] 92%|█████████▏| 367860/400000 [00:46<00:03, 8144.71it/s] 92%|█████████▏| 368678/400000 [00:46<00:03, 8133.77it/s] 92%|█████████▏| 369544/400000 [00:46<00:03, 8282.95it/s] 93%|█████████▎| 370396/400000 [00:46<00:03, 8350.72it/s] 93%|█████████▎| 371233/400000 [00:46<00:03, 8326.78it/s] 93%|█████████▎| 372080/400000 [00:47<00:03, 8367.37it/s] 93%|█████████▎| 372918/400000 [00:47<00:03, 7791.95it/s] 93%|█████████▎| 373706/400000 [00:47<00:03, 7459.59it/s] 94%|█████████▎| 374517/400000 [00:47<00:03, 7642.44it/s] 94%|█████████▍| 375298/400000 [00:47<00:03, 7691.91it/s] 94%|█████████▍| 376073/400000 [00:47<00:03, 7487.15it/s] 94%|█████████▍| 376827/400000 [00:47<00:03, 7200.29it/s] 94%|█████████▍| 377553/400000 [00:47<00:03, 7143.29it/s] 95%|█████████▍| 378386/400000 [00:47<00:02, 7461.41it/s] 95%|█████████▍| 379146/400000 [00:47<00:02, 7501.22it/s] 95%|█████████▍| 379914/400000 [00:48<00:02, 7552.57it/s] 95%|█████████▌| 380740/400000 [00:48<00:02, 7749.28it/s] 95%|█████████▌| 381550/400000 [00:48<00:02, 7849.18it/s] 96%|█████████▌| 382402/400000 [00:48<00:02, 8038.27it/s] 96%|█████████▌| 383242/400000 [00:48<00:02, 8141.23it/s] 96%|█████████▌| 384080/400000 [00:48<00:01, 8210.37it/s] 96%|█████████▌| 384903/400000 [00:48<00:01, 8215.11it/s] 96%|█████████▋| 385756/400000 [00:48<00:01, 8305.14it/s] 97%|█████████▋| 386591/400000 [00:48<00:01, 8317.47it/s] 97%|█████████▋| 387456/400000 [00:48<00:01, 8412.60it/s] 97%|█████████▋| 388321/400000 [00:49<00:01, 8482.09it/s] 97%|█████████▋| 389170/400000 [00:49<00:01, 8422.46it/s] 98%|█████████▊| 390013/400000 [00:49<00:01, 8363.10it/s] 98%|█████████▊| 390876/400000 [00:49<00:01, 8440.70it/s] 98%|█████████▊| 391721/400000 [00:49<00:00, 8422.72it/s] 98%|█████████▊| 392564/400000 [00:49<00:00, 8111.38it/s] 98%|█████████▊| 393378/400000 [00:49<00:00, 8021.65it/s] 99%|█████████▊| 394234/400000 [00:49<00:00, 8170.24it/s] 99%|█████████▉| 395088/400000 [00:49<00:00, 8277.65it/s] 99%|█████████▉| 395950/400000 [00:50<00:00, 8375.51it/s] 99%|█████████▉| 396812/400000 [00:50<00:00, 8445.11it/s] 99%|█████████▉| 397658/400000 [00:50<00:00, 8055.39it/s]100%|█████████▉| 398469/400000 [00:50<00:00, 7958.89it/s]100%|█████████▉| 399269/400000 [00:50<00:00, 7934.35it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7915.65it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f190bb23940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010907605397002038 	 Accuracy: 55
Train Epoch: 1 	 Loss: 0.010737552092625545 	 Accuracy: 63

  model saves at 63% accuracy 

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
2020-05-15 17:24:28.035465: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 17:24:28.039660: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-15 17:24:28.039784: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562eceb79a80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 17:24:28.039800: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1917694fd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6820 - accuracy: 0.4990
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6590 - accuracy: 0.5005
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6820 - accuracy: 0.4990 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7510 - accuracy: 0.4945
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5900 - accuracy: 0.5050
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6491 - accuracy: 0.5011
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6149 - accuracy: 0.5034
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6377 - accuracy: 0.5019
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6436 - accuracy: 0.5015
11000/25000 [============>.................] - ETA: 4s - loss: 7.6541 - accuracy: 0.5008
12000/25000 [=============>................] - ETA: 3s - loss: 7.6628 - accuracy: 0.5002
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6985 - accuracy: 0.4979
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7071 - accuracy: 0.4974
15000/25000 [=================>............] - ETA: 3s - loss: 7.7208 - accuracy: 0.4965
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7241 - accuracy: 0.4963
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7298 - accuracy: 0.4959
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7126 - accuracy: 0.4970
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7167 - accuracy: 0.4967
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7096 - accuracy: 0.4972
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7163 - accuracy: 0.4968
22000/25000 [=========================>....] - ETA: 0s - loss: 7.7064 - accuracy: 0.4974
23000/25000 [==========================>...] - ETA: 0s - loss: 7.7020 - accuracy: 0.4977
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6852 - accuracy: 0.4988
25000/25000 [==============================] - 9s 356us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f18708aa5f8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f1888215dd8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 975ms/step - loss: 1.4621 - crf_viterbi_accuracy: 0.0267 - val_loss: 1.4835 - val_crf_viterbi_accuracy: 0.6533

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

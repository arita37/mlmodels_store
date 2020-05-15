
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f5599994fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 18:13:39.978595
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 18:13:39.982503
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 18:13:39.985731
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 18:13:39.988906
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f55a59ac438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 357864.1562
Epoch 2/10

1/1 [==============================] - 0s 108ms/step - loss: 290921.8125
Epoch 3/10

1/1 [==============================] - 0s 102ms/step - loss: 192756.2969
Epoch 4/10

1/1 [==============================] - 0s 103ms/step - loss: 120280.4531
Epoch 5/10

1/1 [==============================] - 0s 106ms/step - loss: 71994.9297
Epoch 6/10

1/1 [==============================] - 0s 103ms/step - loss: 43948.4570
Epoch 7/10

1/1 [==============================] - 0s 101ms/step - loss: 27972.6836
Epoch 8/10

1/1 [==============================] - 0s 105ms/step - loss: 18473.6855
Epoch 9/10

1/1 [==============================] - 0s 104ms/step - loss: 12846.5938
Epoch 10/10

1/1 [==============================] - 0s 98ms/step - loss: 9518.7783

  #### Inference Need return ypred, ytrue ######################### 
[[ 6.16304278e-02 -2.77450740e-01  3.25712025e-01 -2.87454367e-01
  -1.00495219e+00 -1.22847706e-01 -2.56300867e-01 -1.97906911e-01
  -1.18753314e-02  7.45432913e-01 -2.78728455e-02  3.74709755e-01
  -1.97903442e+00  4.12142396e-01  6.45475030e-01 -5.41469269e-02
  -3.65994632e-01 -6.29230380e-01  1.05924428e-01 -1.31496775e+00
   1.14892578e+00  3.44984978e-02  1.20224047e+00 -2.48682089e-02
  -6.58291757e-01  1.05816483e-01 -5.18198192e-01 -1.27081275e-01
  -1.93784922e-01 -4.65245128e-01  5.82500935e-01 -1.43669724e-01
  -1.34760022e-01 -1.27382493e+00  6.92409158e-01 -7.07275271e-02
   1.20867753e+00  1.24563754e-01 -1.71776652e+00 -2.94271529e-01
  -4.01103169e-01 -5.64982831e-01  5.39536774e-01  2.42216259e-01
  -3.28230262e-01 -1.72501755e+00 -1.22458327e+00 -2.00643539e-02
   1.81174338e-01 -5.53907156e-01 -1.16464555e-01 -7.69173265e-01
  -2.33625442e-01 -3.66811395e-01  3.58068645e-02 -4.62747753e-01
  -4.40948099e-01  5.87567091e-01  1.10670137e+00 -8.91790390e-01
  -8.29526335e-02  6.29718924e+00  5.80557966e+00  6.84399986e+00
   5.68817091e+00  7.61720657e+00  5.89270449e+00  8.11536598e+00
   7.02635384e+00  6.54765892e+00  7.92972755e+00  7.48440313e+00
   5.32632494e+00  5.21985245e+00  6.16096783e+00  8.07394791e+00
   8.03369999e+00  6.80908298e+00  7.65878868e+00  6.59473133e+00
   6.99083328e+00  8.02033901e+00  7.26012611e+00  6.16697311e+00
   5.44214535e+00  5.61342621e+00  6.25517845e+00  7.33624554e+00
   6.87563944e+00  7.42736006e+00  5.74701977e+00  6.34514380e+00
   6.46124935e+00  5.43902922e+00  7.21204138e+00  6.71213388e+00
   6.54261732e+00  5.72809219e+00  7.24748421e+00  6.58885384e+00
   6.64818382e+00  7.18350267e+00  5.00404644e+00  7.65551853e+00
   6.98338509e+00  6.62122011e+00  7.52871418e+00  8.75878525e+00
   7.49904680e+00  6.69419336e+00  7.15867043e+00  7.40641499e+00
   6.20139742e+00  7.09030724e+00  6.56486893e+00  6.25450706e+00
   5.48935795e+00  7.37295389e+00  6.10610104e+00  6.92036104e+00
  -1.03214771e-01 -1.16609693e-01 -8.93180668e-02  3.59624922e-02
  -1.41902238e-01 -3.93107533e-01 -9.68435764e-01  6.24068320e-01
  -4.39642549e-01  7.93987393e-01 -8.58923793e-02 -5.95578969e-01
   4.64531600e-01 -9.44389880e-01 -4.35761362e-01 -1.43099457e-01
  -1.74502134e-02 -8.88512075e-01  5.20274818e-01 -9.00910914e-01
  -1.56622612e+00  5.30745387e-02 -4.87380624e-02 -6.09124243e-01
   8.36402297e-01 -5.39967775e-01 -1.46388078e+00  3.78227532e-01
  -6.46446347e-01 -4.46272492e-02 -4.41028029e-02 -9.25846100e-01
  -9.85971749e-01 -1.04227734e+00 -1.02021551e+00  9.57218409e-02
  -3.68259072e-01  2.47611284e-01  9.29693043e-01 -8.88067245e-01
  -8.69065642e-01  6.31927848e-01  1.50752068e-03  5.81887722e-01
   3.80470216e-01  4.20291573e-01 -6.23459101e-01  1.31702209e+00
  -3.72519672e-01 -3.51630777e-01  2.75125325e-01  1.46981359e-01
   1.77738070e+00  2.72064537e-01  3.37773174e-01 -8.59401107e-01
  -3.40269446e-01 -9.43053365e-02 -1.64481670e-01 -1.07297413e-01
   1.06176472e+00  1.79872704e+00  1.70457602e-01  2.44374323e+00
   6.32071018e-01  1.60428953e+00  4.68669236e-01  1.64776611e+00
   5.62898397e-01  9.86266971e-01  4.08432126e-01  1.06084752e+00
   1.16690397e+00  1.46890604e+00  8.05041432e-01  2.99013972e-01
   1.10018802e+00  2.85442924e+00  1.48314023e+00  2.36294985e-01
   4.39717591e-01  1.00959229e+00  5.78114033e-01  2.03701019e+00
   2.40644264e+00  2.40026855e+00  1.20447969e+00  5.46659827e-01
   7.30201423e-01  1.24309111e+00  1.39548612e+00  3.60173285e-01
   2.73782873e+00  4.75265861e-01  1.65717161e+00  1.53728628e+00
   4.90420818e-01  5.10134876e-01  1.24424386e+00  3.40008318e-01
   1.75468993e+00  5.20874143e-01  1.20221412e+00  6.94955230e-01
   6.66204453e-01  4.34750199e-01  1.17591095e+00  8.72793198e-01
   6.02704644e-01  1.57983506e+00  4.82795238e-01  5.17798960e-01
   5.08841693e-01  1.39894342e+00  1.61710846e+00  4.48969603e-01
   5.71544170e-01  7.77575910e-01  1.78344405e+00  2.44319057e+00
   6.04373217e-02  7.00100183e+00  7.14327431e+00  6.62737751e+00
   6.35801840e+00  7.77487469e+00  7.10296011e+00  7.09741259e+00
   7.79710388e+00  6.56987047e+00  7.11297512e+00  7.34979343e+00
   8.25371170e+00  6.38844347e+00  7.20698690e+00  7.75810623e+00
   5.20241547e+00  6.76216030e+00  7.70901728e+00  7.02704287e+00
   6.81801033e+00  6.54545164e+00  6.85572386e+00  8.08399200e+00
   7.01358986e+00  7.83077860e+00  6.86753845e+00  7.63382339e+00
   6.20107794e+00  7.54649925e+00  6.31401920e+00  7.13851929e+00
   6.61881924e+00  8.01205635e+00  5.89955235e+00  6.21619749e+00
   7.69938278e+00  7.01224327e+00  7.47156858e+00  7.06977987e+00
   7.64719296e+00  8.07264996e+00  8.29625988e+00  7.70226955e+00
   6.59469271e+00  6.86519337e+00  6.90817261e+00  6.19735909e+00
   7.64512110e+00  8.16280651e+00  6.08874893e+00  5.46966457e+00
   7.12880707e+00  6.23363161e+00  6.89272118e+00  5.80728674e+00
   5.85765362e+00  7.01606369e+00  7.81528616e+00  6.89683247e+00
   1.11259425e+00  7.87565112e-01  3.06832671e-01  1.43514502e+00
   8.38518858e-01  2.71457458e+00  1.01241720e+00  2.18164682e-01
   1.27663064e+00  6.04375601e-01  4.60605979e-01  5.35376608e-01
   2.72876930e+00  1.59991670e+00  1.27423620e+00  1.02615571e+00
   5.06340146e-01  6.10592008e-01  4.05985236e-01  2.63592720e-01
   7.64266133e-01  1.16273034e+00  8.22586060e-01  3.72026682e-01
   4.57467437e-01  4.33382034e-01  4.79214907e-01  6.69885457e-01
   1.65830886e+00  1.87363756e+00  1.95016623e-01  8.67434859e-01
   1.64270294e+00  1.13750064e+00  1.30642509e+00  8.18014920e-01
   8.39820862e-01  2.69563937e+00  1.00271070e+00  5.23803473e-01
   1.52983212e+00  1.60623288e+00  1.85320520e+00  4.41963315e-01
   1.19835603e+00  1.05521929e+00  4.83318508e-01  1.66131747e+00
   1.15381265e+00  3.94367039e-01  1.96316719e-01  1.88033998e+00
   2.26412416e+00  7.59921908e-01  8.76956224e-01  8.81764591e-01
   1.04158950e+00  1.43233001e+00  2.10305429e+00  3.50618422e-01
  -7.93389893e+00  6.08671808e+00 -3.22765255e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 18:13:49.268937
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.5773
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 18:13:49.272752
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9153.89
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 18:13:49.276236
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    95.369
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 18:13:49.283368
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -818.796
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140005285945920
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140004327199296
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140004327199800
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140004327200304
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140004327200808
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140004327201312

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f5593078fd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.642279
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.607414
grad_step = 000002, loss = 0.582492
grad_step = 000003, loss = 0.555261
grad_step = 000004, loss = 0.524602
grad_step = 000005, loss = 0.495441
grad_step = 000006, loss = 0.474424
grad_step = 000007, loss = 0.459124
grad_step = 000008, loss = 0.442934
grad_step = 000009, loss = 0.423841
grad_step = 000010, loss = 0.408146
grad_step = 000011, loss = 0.395539
grad_step = 000012, loss = 0.383641
grad_step = 000013, loss = 0.370806
grad_step = 000014, loss = 0.356650
grad_step = 000015, loss = 0.341377
grad_step = 000016, loss = 0.325877
grad_step = 000017, loss = 0.311751
grad_step = 000018, loss = 0.299887
grad_step = 000019, loss = 0.287173
grad_step = 000020, loss = 0.272980
grad_step = 000021, loss = 0.259307
grad_step = 000022, loss = 0.246808
grad_step = 000023, loss = 0.234730
grad_step = 000024, loss = 0.222514
grad_step = 000025, loss = 0.210159
grad_step = 000026, loss = 0.197967
grad_step = 000027, loss = 0.186586
grad_step = 000028, loss = 0.175537
grad_step = 000029, loss = 0.163965
grad_step = 000030, loss = 0.152195
grad_step = 000031, loss = 0.141220
grad_step = 000032, loss = 0.130952
grad_step = 000033, loss = 0.121144
grad_step = 000034, loss = 0.111962
grad_step = 000035, loss = 0.103868
grad_step = 000036, loss = 0.096597
grad_step = 000037, loss = 0.089196
grad_step = 000038, loss = 0.081704
grad_step = 000039, loss = 0.074846
grad_step = 000040, loss = 0.068600
grad_step = 000041, loss = 0.062607
grad_step = 000042, loss = 0.057032
grad_step = 000043, loss = 0.052148
grad_step = 000044, loss = 0.047713
grad_step = 000045, loss = 0.043453
grad_step = 000046, loss = 0.039510
grad_step = 000047, loss = 0.035915
grad_step = 000048, loss = 0.032555
grad_step = 000049, loss = 0.029499
grad_step = 000050, loss = 0.026829
grad_step = 000051, loss = 0.024395
grad_step = 000052, loss = 0.022130
grad_step = 000053, loss = 0.020038
grad_step = 000054, loss = 0.018084
grad_step = 000055, loss = 0.016325
grad_step = 000056, loss = 0.014819
grad_step = 000057, loss = 0.013422
grad_step = 000058, loss = 0.012116
grad_step = 000059, loss = 0.010971
grad_step = 000060, loss = 0.009938
grad_step = 000061, loss = 0.008970
grad_step = 000062, loss = 0.008129
grad_step = 000063, loss = 0.007392
grad_step = 000064, loss = 0.006723
grad_step = 000065, loss = 0.006151
grad_step = 000066, loss = 0.005629
grad_step = 000067, loss = 0.005134
grad_step = 000068, loss = 0.004714
grad_step = 000069, loss = 0.004356
grad_step = 000070, loss = 0.004033
grad_step = 000071, loss = 0.003771
grad_step = 000072, loss = 0.003541
grad_step = 000073, loss = 0.003322
grad_step = 000074, loss = 0.003144
grad_step = 000075, loss = 0.003003
grad_step = 000076, loss = 0.002902
grad_step = 000077, loss = 0.002863
grad_step = 000078, loss = 0.002887
grad_step = 000079, loss = 0.002882
grad_step = 000080, loss = 0.002795
grad_step = 000081, loss = 0.002559
grad_step = 000082, loss = 0.002363
grad_step = 000083, loss = 0.002329
grad_step = 000084, loss = 0.002406
grad_step = 000085, loss = 0.002453
grad_step = 000086, loss = 0.002363
grad_step = 000087, loss = 0.002230
grad_step = 000088, loss = 0.002179
grad_step = 000089, loss = 0.002222
grad_step = 000090, loss = 0.002273
grad_step = 000091, loss = 0.002247
grad_step = 000092, loss = 0.002167
grad_step = 000093, loss = 0.002101
grad_step = 000094, loss = 0.002095
grad_step = 000095, loss = 0.002126
grad_step = 000096, loss = 0.002142
grad_step = 000097, loss = 0.002122
grad_step = 000098, loss = 0.002070
grad_step = 000099, loss = 0.002024
grad_step = 000100, loss = 0.002002
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002006
grad_step = 000102, loss = 0.002022
grad_step = 000103, loss = 0.002035
grad_step = 000104, loss = 0.002038
grad_step = 000105, loss = 0.002024
grad_step = 000106, loss = 0.002003
grad_step = 000107, loss = 0.001975
grad_step = 000108, loss = 0.001950
grad_step = 000109, loss = 0.001928
grad_step = 000110, loss = 0.001910
grad_step = 000111, loss = 0.001897
grad_step = 000112, loss = 0.001887
grad_step = 000113, loss = 0.001878
grad_step = 000114, loss = 0.001871
grad_step = 000115, loss = 0.001864
grad_step = 000116, loss = 0.001858
grad_step = 000117, loss = 0.001854
grad_step = 000118, loss = 0.001855
grad_step = 000119, loss = 0.001869
grad_step = 000120, loss = 0.001923
grad_step = 000121, loss = 0.002105
grad_step = 000122, loss = 0.002603
grad_step = 000123, loss = 0.003700
grad_step = 000124, loss = 0.004104
grad_step = 000125, loss = 0.002878
grad_step = 000126, loss = 0.001843
grad_step = 000127, loss = 0.002942
grad_step = 000128, loss = 0.003104
grad_step = 000129, loss = 0.001877
grad_step = 000130, loss = 0.002455
grad_step = 000131, loss = 0.002773
grad_step = 000132, loss = 0.001854
grad_step = 000133, loss = 0.002362
grad_step = 000134, loss = 0.002480
grad_step = 000135, loss = 0.001822
grad_step = 000136, loss = 0.002312
grad_step = 000137, loss = 0.002202
grad_step = 000138, loss = 0.001823
grad_step = 000139, loss = 0.002256
grad_step = 000140, loss = 0.002007
grad_step = 000141, loss = 0.001850
grad_step = 000142, loss = 0.002157
grad_step = 000143, loss = 0.001869
grad_step = 000144, loss = 0.001881
grad_step = 000145, loss = 0.002047
grad_step = 000146, loss = 0.001797
grad_step = 000147, loss = 0.001891
grad_step = 000148, loss = 0.001943
grad_step = 000149, loss = 0.001768
grad_step = 000150, loss = 0.001882
grad_step = 000151, loss = 0.001865
grad_step = 000152, loss = 0.001760
grad_step = 000153, loss = 0.001858
grad_step = 000154, loss = 0.001813
grad_step = 000155, loss = 0.001756
grad_step = 000156, loss = 0.001831
grad_step = 000157, loss = 0.001782
grad_step = 000158, loss = 0.001751
grad_step = 000159, loss = 0.001804
grad_step = 000160, loss = 0.001762
grad_step = 000161, loss = 0.001744
grad_step = 000162, loss = 0.001781
grad_step = 000163, loss = 0.001749
grad_step = 000164, loss = 0.001734
grad_step = 000165, loss = 0.001761
grad_step = 000166, loss = 0.001740
grad_step = 000167, loss = 0.001725
grad_step = 000168, loss = 0.001744
grad_step = 000169, loss = 0.001733
grad_step = 000170, loss = 0.001716
grad_step = 000171, loss = 0.001728
grad_step = 000172, loss = 0.001726
grad_step = 000173, loss = 0.001710
grad_step = 000174, loss = 0.001713
grad_step = 000175, loss = 0.001717
grad_step = 000176, loss = 0.001706
grad_step = 000177, loss = 0.001701
grad_step = 000178, loss = 0.001714
grad_step = 000179, loss = 0.001713
grad_step = 000180, loss = 0.001696
grad_step = 000181, loss = 0.001695
grad_step = 000182, loss = 0.001703
grad_step = 000183, loss = 0.001695
grad_step = 000184, loss = 0.001684
grad_step = 000185, loss = 0.001687
grad_step = 000186, loss = 0.001690
grad_step = 000187, loss = 0.001683
grad_step = 000188, loss = 0.001675
grad_step = 000189, loss = 0.001676
grad_step = 000190, loss = 0.001678
grad_step = 000191, loss = 0.001673
grad_step = 000192, loss = 0.001666
grad_step = 000193, loss = 0.001665
grad_step = 000194, loss = 0.001665
grad_step = 000195, loss = 0.001663
grad_step = 000196, loss = 0.001659
grad_step = 000197, loss = 0.001656
grad_step = 000198, loss = 0.001652
grad_step = 000199, loss = 0.001649
grad_step = 000200, loss = 0.001647
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001647
grad_step = 000202, loss = 0.001647
grad_step = 000203, loss = 0.001650
grad_step = 000204, loss = 0.001661
grad_step = 000205, loss = 0.001691
grad_step = 000206, loss = 0.001762
grad_step = 000207, loss = 0.001889
grad_step = 000208, loss = 0.002070
grad_step = 000209, loss = 0.002171
grad_step = 000210, loss = 0.002104
grad_step = 000211, loss = 0.001838
grad_step = 000212, loss = 0.001639
grad_step = 000213, loss = 0.001683
grad_step = 000214, loss = 0.001843
grad_step = 000215, loss = 0.001876
grad_step = 000216, loss = 0.001730
grad_step = 000217, loss = 0.001625
grad_step = 000218, loss = 0.001679
grad_step = 000219, loss = 0.001772
grad_step = 000220, loss = 0.001757
grad_step = 000221, loss = 0.001659
grad_step = 000222, loss = 0.001616
grad_step = 000223, loss = 0.001664
grad_step = 000224, loss = 0.001711
grad_step = 000225, loss = 0.001691
grad_step = 000226, loss = 0.001631
grad_step = 000227, loss = 0.001607
grad_step = 000228, loss = 0.001633
grad_step = 000229, loss = 0.001664
grad_step = 000230, loss = 0.001662
grad_step = 000231, loss = 0.001629
grad_step = 000232, loss = 0.001600
grad_step = 000233, loss = 0.001598
grad_step = 000234, loss = 0.001615
grad_step = 000235, loss = 0.001632
grad_step = 000236, loss = 0.001631
grad_step = 000237, loss = 0.001616
grad_step = 000238, loss = 0.001596
grad_step = 000239, loss = 0.001582
grad_step = 000240, loss = 0.001580
grad_step = 000241, loss = 0.001586
grad_step = 000242, loss = 0.001597
grad_step = 000243, loss = 0.001610
grad_step = 000244, loss = 0.001625
grad_step = 000245, loss = 0.001645
grad_step = 000246, loss = 0.001678
grad_step = 000247, loss = 0.001729
grad_step = 000248, loss = 0.001812
grad_step = 000249, loss = 0.001916
grad_step = 000250, loss = 0.002025
grad_step = 000251, loss = 0.002059
grad_step = 000252, loss = 0.001978
grad_step = 000253, loss = 0.001782
grad_step = 000254, loss = 0.001608
grad_step = 000255, loss = 0.001573
grad_step = 000256, loss = 0.001661
grad_step = 000257, loss = 0.001749
grad_step = 000258, loss = 0.001734
grad_step = 000259, loss = 0.001640
grad_step = 000260, loss = 0.001570
grad_step = 000261, loss = 0.001588
grad_step = 000262, loss = 0.001650
grad_step = 000263, loss = 0.001675
grad_step = 000264, loss = 0.001638
grad_step = 000265, loss = 0.001573
grad_step = 000266, loss = 0.001544
grad_step = 000267, loss = 0.001564
grad_step = 000268, loss = 0.001602
grad_step = 000269, loss = 0.001623
grad_step = 000270, loss = 0.001605
grad_step = 000271, loss = 0.001570
grad_step = 000272, loss = 0.001541
grad_step = 000273, loss = 0.001534
grad_step = 000274, loss = 0.001547
grad_step = 000275, loss = 0.001567
grad_step = 000276, loss = 0.001582
grad_step = 000277, loss = 0.001586
grad_step = 000278, loss = 0.001582
grad_step = 000279, loss = 0.001573
grad_step = 000280, loss = 0.001566
grad_step = 000281, loss = 0.001566
grad_step = 000282, loss = 0.001579
grad_step = 000283, loss = 0.001607
grad_step = 000284, loss = 0.001661
grad_step = 000285, loss = 0.001729
grad_step = 000286, loss = 0.001834
grad_step = 000287, loss = 0.001921
grad_step = 000288, loss = 0.002017
grad_step = 000289, loss = 0.002069
grad_step = 000290, loss = 0.002053
grad_step = 000291, loss = 0.001912
grad_step = 000292, loss = 0.001714
grad_step = 000293, loss = 0.001566
grad_step = 000294, loss = 0.001575
grad_step = 000295, loss = 0.001688
grad_step = 000296, loss = 0.001749
grad_step = 000297, loss = 0.001686
grad_step = 000298, loss = 0.001563
grad_step = 000299, loss = 0.001521
grad_step = 000300, loss = 0.001582
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001646
grad_step = 000302, loss = 0.001634
grad_step = 000303, loss = 0.001556
grad_step = 000304, loss = 0.001504
grad_step = 000305, loss = 0.001522
grad_step = 000306, loss = 0.001572
grad_step = 000307, loss = 0.001592
grad_step = 000308, loss = 0.001559
grad_step = 000309, loss = 0.001510
grad_step = 000310, loss = 0.001490
grad_step = 000311, loss = 0.001506
grad_step = 000312, loss = 0.001536
grad_step = 000313, loss = 0.001548
grad_step = 000314, loss = 0.001537
grad_step = 000315, loss = 0.001512
grad_step = 000316, loss = 0.001492
grad_step = 000317, loss = 0.001484
grad_step = 000318, loss = 0.001486
grad_step = 000319, loss = 0.001492
grad_step = 000320, loss = 0.001499
grad_step = 000321, loss = 0.001507
grad_step = 000322, loss = 0.001518
grad_step = 000323, loss = 0.001531
grad_step = 000324, loss = 0.001546
grad_step = 000325, loss = 0.001563
grad_step = 000326, loss = 0.001581
grad_step = 000327, loss = 0.001611
grad_step = 000328, loss = 0.001666
grad_step = 000329, loss = 0.001741
grad_step = 000330, loss = 0.001814
grad_step = 000331, loss = 0.001874
grad_step = 000332, loss = 0.001859
grad_step = 000333, loss = 0.001770
grad_step = 000334, loss = 0.001617
grad_step = 000335, loss = 0.001488
grad_step = 000336, loss = 0.001445
grad_step = 000337, loss = 0.001487
grad_step = 000338, loss = 0.001555
grad_step = 000339, loss = 0.001581
grad_step = 000340, loss = 0.001552
grad_step = 000341, loss = 0.001486
grad_step = 000342, loss = 0.001429
grad_step = 000343, loss = 0.001403
grad_step = 000344, loss = 0.001411
grad_step = 000345, loss = 0.001435
grad_step = 000346, loss = 0.001456
grad_step = 000347, loss = 0.001460
grad_step = 000348, loss = 0.001438
grad_step = 000349, loss = 0.001400
grad_step = 000350, loss = 0.001357
grad_step = 000351, loss = 0.001326
grad_step = 000352, loss = 0.001308
grad_step = 000353, loss = 0.001301
grad_step = 000354, loss = 0.001301
grad_step = 000355, loss = 0.001302
grad_step = 000356, loss = 0.001311
grad_step = 000357, loss = 0.001336
grad_step = 000358, loss = 0.001400
grad_step = 000359, loss = 0.001549
grad_step = 000360, loss = 0.001792
grad_step = 000361, loss = 0.002077
grad_step = 000362, loss = 0.002219
grad_step = 000363, loss = 0.001957
grad_step = 000364, loss = 0.001456
grad_step = 000365, loss = 0.001187
grad_step = 000366, loss = 0.001343
grad_step = 000367, loss = 0.001606
grad_step = 000368, loss = 0.001568
grad_step = 000369, loss = 0.001290
grad_step = 000370, loss = 0.001135
grad_step = 000371, loss = 0.001248
grad_step = 000372, loss = 0.001390
grad_step = 000373, loss = 0.001335
grad_step = 000374, loss = 0.001181
grad_step = 000375, loss = 0.001088
grad_step = 000376, loss = 0.001131
grad_step = 000377, loss = 0.001211
grad_step = 000378, loss = 0.001195
grad_step = 000379, loss = 0.001124
grad_step = 000380, loss = 0.001049
grad_step = 000381, loss = 0.001036
grad_step = 000382, loss = 0.001066
grad_step = 000383, loss = 0.001075
grad_step = 000384, loss = 0.001047
grad_step = 000385, loss = 0.000996
grad_step = 000386, loss = 0.000964
grad_step = 000387, loss = 0.000964
grad_step = 000388, loss = 0.000979
grad_step = 000389, loss = 0.000989
grad_step = 000390, loss = 0.000968
grad_step = 000391, loss = 0.000930
grad_step = 000392, loss = 0.000892
grad_step = 000393, loss = 0.000877
grad_step = 000394, loss = 0.000883
grad_step = 000395, loss = 0.000894
grad_step = 000396, loss = 0.000903
grad_step = 000397, loss = 0.000921
grad_step = 000398, loss = 0.000942
grad_step = 000399, loss = 0.000966
grad_step = 000400, loss = 0.000955
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000919
grad_step = 000402, loss = 0.000860
grad_step = 000403, loss = 0.000807
grad_step = 000404, loss = 0.000780
grad_step = 000405, loss = 0.000786
grad_step = 000406, loss = 0.000805
grad_step = 000407, loss = 0.000823
grad_step = 000408, loss = 0.000852
grad_step = 000409, loss = 0.000873
grad_step = 000410, loss = 0.000883
grad_step = 000411, loss = 0.000850
grad_step = 000412, loss = 0.000805
grad_step = 000413, loss = 0.000756
grad_step = 000414, loss = 0.000713
grad_step = 000415, loss = 0.000700
grad_step = 000416, loss = 0.000719
grad_step = 000417, loss = 0.000739
grad_step = 000418, loss = 0.000748
grad_step = 000419, loss = 0.000753
grad_step = 000420, loss = 0.000742
grad_step = 000421, loss = 0.000715
grad_step = 000422, loss = 0.000680
grad_step = 000423, loss = 0.000659
grad_step = 000424, loss = 0.000644
grad_step = 000425, loss = 0.000636
grad_step = 000426, loss = 0.000641
grad_step = 000427, loss = 0.000654
grad_step = 000428, loss = 0.000671
grad_step = 000429, loss = 0.000693
grad_step = 000430, loss = 0.000733
grad_step = 000431, loss = 0.000759
grad_step = 000432, loss = 0.000773
grad_step = 000433, loss = 0.000743
grad_step = 000434, loss = 0.000706
grad_step = 000435, loss = 0.000649
grad_step = 000436, loss = 0.000593
grad_step = 000437, loss = 0.000576
grad_step = 000438, loss = 0.000592
grad_step = 000439, loss = 0.000606
grad_step = 000440, loss = 0.000614
grad_step = 000441, loss = 0.000631
grad_step = 000442, loss = 0.000642
grad_step = 000443, loss = 0.000632
grad_step = 000444, loss = 0.000607
grad_step = 000445, loss = 0.000588
grad_step = 000446, loss = 0.000570
grad_step = 000447, loss = 0.000545
grad_step = 000448, loss = 0.000529
grad_step = 000449, loss = 0.000527
grad_step = 000450, loss = 0.000526
grad_step = 000451, loss = 0.000519
grad_step = 000452, loss = 0.000514
grad_step = 000453, loss = 0.000517
grad_step = 000454, loss = 0.000526
grad_step = 000455, loss = 0.000528
grad_step = 000456, loss = 0.000535
grad_step = 000457, loss = 0.000549
grad_step = 000458, loss = 0.000576
grad_step = 000459, loss = 0.000611
grad_step = 000460, loss = 0.000661
grad_step = 000461, loss = 0.000723
grad_step = 000462, loss = 0.000773
grad_step = 000463, loss = 0.000771
grad_step = 000464, loss = 0.000736
grad_step = 000465, loss = 0.000672
grad_step = 000466, loss = 0.000575
grad_step = 000467, loss = 0.000497
grad_step = 000468, loss = 0.000474
grad_step = 000469, loss = 0.000485
grad_step = 000470, loss = 0.000512
grad_step = 000471, loss = 0.000552
grad_step = 000472, loss = 0.000579
grad_step = 000473, loss = 0.000577
grad_step = 000474, loss = 0.000551
grad_step = 000475, loss = 0.000516
grad_step = 000476, loss = 0.000480
grad_step = 000477, loss = 0.000451
grad_step = 000478, loss = 0.000439
grad_step = 000479, loss = 0.000445
grad_step = 000480, loss = 0.000456
grad_step = 000481, loss = 0.000467
grad_step = 000482, loss = 0.000479
grad_step = 000483, loss = 0.000488
grad_step = 000484, loss = 0.000486
grad_step = 000485, loss = 0.000479
grad_step = 000486, loss = 0.000470
grad_step = 000487, loss = 0.000458
grad_step = 000488, loss = 0.000443
grad_step = 000489, loss = 0.000431
grad_step = 000490, loss = 0.000424
grad_step = 000491, loss = 0.000418
grad_step = 000492, loss = 0.000412
grad_step = 000493, loss = 0.000409
grad_step = 000494, loss = 0.000408
grad_step = 000495, loss = 0.000407
grad_step = 000496, loss = 0.000407
grad_step = 000497, loss = 0.000407
grad_step = 000498, loss = 0.000411
grad_step = 000499, loss = 0.000417
grad_step = 000500, loss = 0.000428
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000447
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

  date_run                              2020-05-15 18:14:09.125918
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.22805
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 18:14:09.132252
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.140374
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 18:14:09.141150
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.122731
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 18:14:09.146900
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.13303
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
0   2020-05-15 18:13:39.978595  ...    mean_absolute_error
1   2020-05-15 18:13:39.982503  ...     mean_squared_error
2   2020-05-15 18:13:39.985731  ...  median_absolute_error
3   2020-05-15 18:13:39.988906  ...               r2_score
4   2020-05-15 18:13:49.268937  ...    mean_absolute_error
5   2020-05-15 18:13:49.272752  ...     mean_squared_error
6   2020-05-15 18:13:49.276236  ...  median_absolute_error
7   2020-05-15 18:13:49.283368  ...               r2_score
8   2020-05-15 18:14:09.125918  ...    mean_absolute_error
9   2020-05-15 18:14:09.132252  ...     mean_squared_error
10  2020-05-15 18:14:09.141150  ...  median_absolute_error
11  2020-05-15 18:14:09.146900  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa34d3e4fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▊      | 3825664/9912422 [00:00<00:00, 38250383.95it/s]9920512it [00:00, 33274138.68it/s]                             
0it [00:00, ?it/s]32768it [00:00, 555897.37it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 161325.07it/s]1654784it [00:00, 10859112.88it/s]                         
0it [00:00, ?it/s]8192it [00:00, 191095.52it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2ffde6e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2ff4170b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2ffde6e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2ff32c0f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2fcba84e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2fcb91c50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2ffde6e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2ff2ea710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2fcba84e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2ff417128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f26a6364ac8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=6d5d4b5a505d4ff1bc0329027d1b92f80555e3254de61487b1da99f82abe755a
  Stored in directory: /tmp/pip-ephem-wheel-cache-5v39aasm/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f2652d45b00> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1179648/17464789 [=>............................] - ETA: 0s
 7356416/17464789 [===========>..................] - ETA: 0s
 9388032/17464789 [===============>..............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 18:15:36.504298: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 18:15:36.508381: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095110000 Hz
2020-05-15 18:15:36.508511: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56380887e0e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 18:15:36.508525: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6973 - accuracy: 0.4980
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6973 - accuracy: 0.4980 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7331 - accuracy: 0.4957
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6053 - accuracy: 0.5040
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6605 - accuracy: 0.5004
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7101 - accuracy: 0.4972
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7411 - accuracy: 0.4951
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6915 - accuracy: 0.4984
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7348 - accuracy: 0.4956
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7050 - accuracy: 0.4975
11000/25000 [============>.................] - ETA: 3s - loss: 7.7098 - accuracy: 0.4972
12000/25000 [=============>................] - ETA: 3s - loss: 7.7267 - accuracy: 0.4961
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7197 - accuracy: 0.4965
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7312 - accuracy: 0.4958
15000/25000 [=================>............] - ETA: 2s - loss: 7.7096 - accuracy: 0.4972
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7011 - accuracy: 0.4978
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6874 - accuracy: 0.4986
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6675 - accuracy: 0.4999
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6626 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6597 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6586 - accuracy: 0.5005
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6520 - accuracy: 0.5010
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6600 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6692 - accuracy: 0.4998
25000/25000 [==============================] - 8s 301us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 18:15:50.859489
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 18:15:50.859489  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<50:59:24, 4.70kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<35:55:23, 6.67kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:02<25:12:00, 9.50kB/s] .vector_cache/glove.6B.zip:   0%|          | 877k/862M [00:02<17:38:27, 13.6kB/s].vector_cache/glove.6B.zip:   0%|          | 3.51M/862M [00:02<12:18:52, 19.4kB/s].vector_cache/glove.6B.zip:   1%|          | 8.85M/862M [00:02<8:34:04, 27.7kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:02<5:58:35, 39.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.8M/862M [00:02<4:09:27, 56.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.4M/862M [00:02<2:53:32, 80.6kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.3M/862M [00:02<2:01:12, 115kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 31.9M/862M [00:02<1:24:20, 164kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.9M/862M [00:03<58:58, 234kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 40.6M/862M [00:03<41:04, 333kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.6M/862M [00:03<28:47, 474kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.0M/862M [00:03<20:05, 674kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.7M/862M [00:03<14:45, 917kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:03<10:58, 1.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:03<07:44, 1.74MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:05<51:55, 259kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<39:04, 344kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:06<27:55, 480kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:06<19:40, 680kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:07<21:21, 626kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<16:18, 819kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:08<11:42, 1.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:09<11:16, 1.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<09:15, 1.44MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:10<06:45, 1.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:11<07:48, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<06:50, 1.93MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:12<05:07, 2.58MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:13<06:38, 1.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:13<06:02, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:13<04:30, 2.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:15<06:10, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:15<07:06, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:15<05:38, 2.31MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<04:04, 3.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:17<50:21, 259kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:17<36:34, 356kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:17<25:50, 502kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:19<21:04, 615kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:19<16:05, 805kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.1M/862M [00:19<11:34, 1.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:21<11:06, 1.16MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:21<10:33, 1.22MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:21<08:02, 1.60MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:21<05:46, 2.22MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:23<1:35:24, 134kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:23<1:08:04, 188kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:23<47:50, 267kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:25<36:22, 350kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:25<26:47, 476kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:25<19:02, 668kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<16:14, 781kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<12:40, 1.00MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<09:11, 1.38MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<09:20, 1.35MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<09:06, 1.38MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<07:01, 1.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<05:02, 2.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<1:23:31, 150kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<59:33, 210kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<41:54, 299kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<29:26, 424kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<1:04:20, 194kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<47:34, 262kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<33:55, 367kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<23:48, 521kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<1:45:09, 118kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<1:14:51, 166kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<52:36, 235kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<39:38, 311kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<29:11, 422kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<21:06, 583kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<14:52, 824kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<1:37:55, 125kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<1:09:48, 176kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<49:01, 250kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<37:04, 329kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<28:28, 428kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<20:26, 596kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<14:26, 841kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<15:26, 785kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<12:03, 1.00MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<08:44, 1.38MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<08:54, 1.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<08:41, 1.39MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<06:36, 1.82MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<04:50, 2.48MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<07:18, 1.64MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<06:22, 1.88MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<04:43, 2.53MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<06:05, 1.96MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<05:31, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<04:10, 2.85MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:40, 2.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<06:23, 1.85MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<05:00, 2.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<03:41, 3.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:43, 1.75MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<05:55, 1.99MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<04:26, 2.64MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<05:50, 2.01MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:55<06:28, 1.81MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<05:04, 2.31MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<03:39, 3.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<15:45, 739kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<12:14, 951kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:57<08:48, 1.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<08:48, 1.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<08:37, 1.34MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:59<06:31, 1.77MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<04:46, 2.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:57, 1.66MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:01, 1.91MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<04:28, 2.57MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:47, 1.98MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:31, 1.75MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:03<05:10, 2.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<03:44, 3.03MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<1:23:42, 136kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<59:44, 190kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<41:58, 270kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<31:56, 354kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<24:45, 457kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<17:53, 631kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<12:36, 892kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<49:23, 227kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<35:33, 316kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<25:07, 446kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<17:40, 632kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<33:27, 334kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<24:34, 454kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<17:26, 639kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<14:43, 754kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<12:39, 876kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<09:21, 1.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<06:41, 1.65MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<10:00, 1.10MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<08:09, 1.35MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:56, 1.85MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<06:42, 1.63MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<07:01, 1.56MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<05:28, 2.00MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<03:56, 2.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<42:12, 258kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<30:39, 355kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<21:41, 501kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<17:39, 613kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<13:32, 799kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<09:43, 1.11MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<09:08, 1.18MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<08:40, 1.24MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:33, 1.64MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<04:47, 2.24MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<06:27, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<05:27, 1.96MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:01, 2.65MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<02:58, 3.58MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<28:47, 369kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<21:17, 498kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<15:09, 699kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<12:59, 812kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<11:19, 931kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<08:28, 1.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<06:00, 1.74MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<24:37, 426kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<18:18, 572kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<13:03, 800kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:32<11:32, 903kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<10:11, 1.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<07:36, 1.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<05:25, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:34<12:23, 835kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<09:43, 1.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<07:00, 1.47MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:36<07:18, 1.41MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<06:09, 1.67MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<04:34, 2.24MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<05:33, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<06:03, 1.68MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<04:45, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<03:27, 2.94MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<1:14:55, 135kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<53:17, 190kB/s]  .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<37:29, 270kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<26:16, 383kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<38:02, 265kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<28:47, 350kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<20:39, 487kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<14:30, 689kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<45:54, 218kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<33:00, 303kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<23:16, 428kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<16:22, 607kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<30:12, 329kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<22:09, 448kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<15:41, 631kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<13:13, 746kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:48<11:21, 868kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<08:27, 1.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:59, 1.63MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<1:06:59, 146kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<47:53, 204kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<33:41, 290kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<25:45, 378kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<20:05, 484kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<14:33, 667kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<10:15, 942kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<1:08:53, 140kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<49:12, 196kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<34:36, 278kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<26:21, 364kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<20:29, 468kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<14:50, 645kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<10:26, 912kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<41:37, 229kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<30:07, 316kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<21:15, 446kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<16:59, 556kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<13:54, 680kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<10:08, 931kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<07:13, 1.30MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<08:20, 1.13MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<06:48, 1.38MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:59, 1.87MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<05:39, 1.65MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<04:56, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<03:41, 2.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<04:43, 1.96MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<05:17, 1.75MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<04:06, 2.25MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<03:00, 3.06MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<06:04, 1.51MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<05:02, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<03:53, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<02:49, 3.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<35:52, 254kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<26:02, 350kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<18:22, 494kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<14:56, 605kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<12:18, 735kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<09:04, 995kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<06:24, 1.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<39:24, 228kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<28:29, 315kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<20:07, 444kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<16:05, 553kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<13:09, 676kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<09:36, 925kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<06:54, 1.28MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<07:08, 1.24MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<06:52, 1.28MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<05:12, 1.69MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<03:45, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<06:46, 1.30MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<06:36, 1.33MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<05:05, 1.72MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<03:39, 2.38MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<27:22, 318kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<21:01, 414kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<15:09, 573kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<10:39, 809kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<30:54, 279kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:23<23:28, 368kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<16:47, 513kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<11:51, 724kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:25<11:20, 755kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<09:45, 878kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:25<07:12, 1.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:07, 1.66MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<09:08, 929kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<08:12, 1.04MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<06:07, 1.38MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:22, 1.93MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<07:23, 1.14MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<06:59, 1.20MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:29<05:17, 1.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:47, 2.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<08:51, 943kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<07:58, 1.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<06:01, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:17, 1.93MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<22:57, 361kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:33<17:50, 464kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<12:51, 644kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<09:08, 903kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<08:45, 939kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:35<07:52, 1.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<05:53, 1.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:14, 1.93MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<06:15, 1.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:37<06:06, 1.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<04:39, 1.75MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<03:20, 2.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<08:36, 939kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:39<07:41, 1.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<05:43, 1.41MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:07, 1.95MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<05:58, 1.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<05:49, 1.37MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<04:29, 1.78MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:13, 2.46MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<32:28, 245kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<24:20, 326kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<17:25, 455kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<12:13, 645kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<36:54, 213kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<27:25, 287kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<19:31, 403kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<13:42, 571kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<13:07, 595kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<10:50, 719kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<07:58, 977kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:38, 1.37MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<09:48, 789kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<08:30, 909kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<06:21, 1.21MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:30, 1.70MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<21:36, 355kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<16:49, 456kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<12:05, 633kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<08:32, 892kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<09:16, 819kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<08:07, 936kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<06:01, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:18, 1.75MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<06:10, 1.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<05:59, 1.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:32, 1.66MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<03:15, 2.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<05:36, 1.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<05:31, 1.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<04:16, 1.74MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:04, 2.41MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<13:26, 550kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<11:02, 669kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<08:05, 912kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:42, 1.28MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<10:50, 675kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<09:08, 800kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<06:45, 1.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<04:47, 1.52MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<08:20, 870kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<07:27, 973kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<05:35, 1.29MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:59, 1.80MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<14:05, 510kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<11:23, 630kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<08:17, 864kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<05:52, 1.21MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<07:02, 1.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<06:30, 1.09MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<04:55, 1.44MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:31, 2.00MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<13:32, 520kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<10:58, 642kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<07:59, 879kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<05:43, 1.22MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<05:51, 1.19MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<05:39, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:20, 1.60MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<03:06, 2.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<13:01, 530kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<10:23, 664kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<07:38, 902kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<05:25, 1.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<06:38, 1.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<06:10, 1.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<04:40, 1.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<03:20, 2.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<11:37, 582kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<09:34, 707kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<06:59, 965kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:01, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<05:12, 1.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<05:07, 1.31MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:18<03:53, 1.72MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:47, 2.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<05:38, 1.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<05:21, 1.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<04:06, 1.61MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:55, 2.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<11:01, 594kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<09:10, 714kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:22<06:46, 966kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<04:47, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<13:16, 489kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<10:43, 604kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<07:48, 829kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<05:32, 1.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<06:22, 1.01MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<05:49, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<04:22, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<03:07, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<16:17, 390kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<12:45, 498kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<09:14, 685kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<06:30, 966kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:30<21:38, 290kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<16:28, 381kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<11:47, 531kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<08:17, 751kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<08:36, 722kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<07:20, 845kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<05:27, 1.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:51, 1.59MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<18:08, 339kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<13:57, 440kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<10:04, 608kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<07:04, 860kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<20:10, 301kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<15:20, 396kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<11:02, 549kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<07:44, 776kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<28:18, 212kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<21:04, 285kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<15:00, 399kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<10:29, 567kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<11:41, 508kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<09:23, 632kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<06:51, 863kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<04:50, 1.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<33:49, 173kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<24:54, 235kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<17:41, 331kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<12:22, 470kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<11:36, 499kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<09:18, 623kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<06:47, 852kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<04:46, 1.20MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<1:08:53, 83.2kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<49:18, 116kB/s]   .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<34:43, 165kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<24:09, 234kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<1:37:52, 57.8kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<1:09:32, 81.4kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<48:47, 116kB/s]   .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<34:02, 165kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<26:12, 213kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<19:28, 287kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<13:53, 402kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<09:41, 570kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<26:42, 207kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<19:48, 279kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<14:06, 390kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<09:51, 554kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<45:53, 119kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:54<33:13, 164kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<23:28, 232kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<16:20, 330kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:56<39:24, 137kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:56<28:39, 188kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<20:17, 265kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<14:07, 377kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<25:22, 210kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:58<18:49, 282kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<13:25, 395kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<09:22, 560kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<39:54, 132kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:00<28:58, 181kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<20:30, 255kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<14:47, 353kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<11:18, 458kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:02<08:58, 577kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<06:31, 791kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:34, 1.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<30:06, 170kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<22:09, 231kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:04<15:42, 325kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<11:02, 460kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<09:10, 550kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<07:26, 678kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<05:26, 924kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:50, 1.30MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<59:37, 83.4kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<42:40, 116kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:08<30:00, 165kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<21:01, 235kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<15:57, 307kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<12:10, 403kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<08:42, 562kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<06:07, 793kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<06:17, 768kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<05:21, 903kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:56, 1.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:48, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<04:20, 1.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<03:56, 1.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:58, 1.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<02:50, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<03:01, 1.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:20, 2.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:40, 2.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<04:04, 1.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:45, 1.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:50, 1.62MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:01, 2.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<1:06:57, 68.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<47:48, 95.4kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<33:37, 135kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<23:20, 193kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:21<22:20, 201kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<16:30, 272kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<11:43, 382kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<08:51, 500kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<07:07, 620kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<05:13, 845kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<03:40, 1.19MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<09:24, 463kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<07:24, 588kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<05:20, 813kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<03:46, 1.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<04:20, 986kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<03:57, 1.08MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:57, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<02:05, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<04:52, 864kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<04:18, 978kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<03:11, 1.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:16, 1.83MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<03:34, 1.16MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<03:23, 1.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<02:35, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:50, 2.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<11:06, 367kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<08:38, 472kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<06:12, 654kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<04:22, 922kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<04:41, 855kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<04:07, 970kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<03:03, 1.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:11, 1.81MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<02:49, 1.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<02:38, 1.49MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:00, 1.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<02:03, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<02:17, 1.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<01:46, 2.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:17, 2.95MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<02:15, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<02:23, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<01:50, 2.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:18, 2.85MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<03:27, 1.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<03:12, 1.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<02:24, 1.54MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:42, 2.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<03:41, 995kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:45<03:22, 1.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<02:32, 1.44MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:47, 2.01MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<09:04, 397kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<07:04, 508kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<05:05, 703kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:35, 987kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:49<03:45, 939kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:49<03:22, 1.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:49<02:32, 1.38MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:48, 1.92MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<10:41, 324kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<08:12, 421kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<05:52, 586kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<04:07, 827kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<04:32, 747kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<03:52, 876kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<02:50, 1.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:00, 1.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<03:13, 1.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:55<02:57, 1.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<02:12, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:35, 2.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:15, 1.45MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:57<02:16, 1.43MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<01:43, 1.88MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:15, 2.56MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:52, 1.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<01:59, 1.60MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<01:31, 2.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:05, 2.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:28, 1.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:01<02:09, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:01<01:36, 1.92MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<01:41, 1.81MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:03<01:45, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<01:21, 2.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:26, 2.08MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:33, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:05<01:13, 2.41MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<01:19, 2.19MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:27, 2.00MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:09, 2.51MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:16, 2.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:24, 2.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:05, 2.60MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<00:47, 3.55MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<02:42, 1.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<02:25, 1.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:49, 1.51MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:18, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:59, 1.36MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:53, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:25, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:00, 2.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<03:04, 857kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<02:38, 996kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:56, 1.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:22, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<03:07, 820kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<02:40, 960kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:57, 1.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:16<01:23, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<02:25, 1.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<02:09, 1.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:37, 1.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<01:08, 2.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<29:09, 83.4kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<20:50, 117kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<14:36, 165kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<10:24, 227kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<07:43, 306kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<05:28, 428kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<04:08, 555kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<03:19, 690kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<02:25, 942kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<02:01, 1.10MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:50, 1.21MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<01:23, 1.60MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:28<01:18, 1.65MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<01:18, 1.65MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<01:00, 2.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<01:02, 2.01MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<01:06, 1.87MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:52, 2.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<00:55, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<01:01, 1.96MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<00:48, 2.47MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<00:52, 2.22MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<00:59, 1.98MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:34<00:45, 2.54MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<00:32, 3.49MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<02:51, 660kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<02:21, 798kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<01:43, 1.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:38<01:28, 1.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<01:22, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:38<01:01, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<00:43, 2.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:40<01:17, 1.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<01:33, 1.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<01:15, 1.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:54, 1.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:02, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:02, 1.61MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<00:47, 2.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<00:48, 1.98MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<01:10, 1.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<00:56, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:41, 2.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<00:51, 1.81MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<00:52, 1.74MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:41, 2.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<00:42, 2.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<00:45, 1.92MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<00:35, 2.43MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<00:38, 2.20MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<00:42, 1.98MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:50<00:33, 2.49MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<00:35, 2.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<00:39, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:52<00:31, 2.52MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:33, 2.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:54<00:37, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:54<00:29, 2.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:31, 2.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<00:35, 1.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:27, 2.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:19, 3.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:44, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:58<00:43, 1.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:33, 1.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:32, 1.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<00:34, 1.82MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<00:26, 2.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:18, 3.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:36, 1.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:36, 1.62MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:27, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:27, 1.99MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:29, 1.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:04<00:23, 2.35MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:23, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:26, 1.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:20, 2.46MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:21, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:23, 2.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<00:18, 2.53MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:09<00:19, 2.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:21, 2.00MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:16, 2.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:11<00:17, 2.25MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:19, 1.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:14, 2.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:10, 3.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:50, 687kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:41, 827kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:29, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:20, 1.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:24, 1.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:22, 1.32MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:16, 1.73MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:15, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:15, 1.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:11, 2.18MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:10, 2.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:11, 1.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:08, 2.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:08, 2.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:09, 1.93MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:06, 2.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:06, 2.21MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:06, 1.98MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:05, 2.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:04, 2.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:04, 1.99MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:03, 2.51MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:02, 2.24MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:03, 1.51MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 1.84MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 2.47MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 2.06MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 1.90MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 2.41MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 876/400000 [00:00<00:45, 8753.73it/s]  0%|          | 1752/400000 [00:00<00:45, 8753.41it/s]  1%|          | 2622/400000 [00:00<00:45, 8735.75it/s]  1%|          | 3499/400000 [00:00<00:45, 8744.67it/s]  1%|          | 4364/400000 [00:00<00:45, 8713.30it/s]  1%|▏         | 5243/400000 [00:00<00:45, 8734.61it/s]  2%|▏         | 6134/400000 [00:00<00:44, 8784.93it/s]  2%|▏         | 7012/400000 [00:00<00:44, 8782.03it/s]  2%|▏         | 7891/400000 [00:00<00:44, 8783.90it/s]  2%|▏         | 8775/400000 [00:01<00:44, 8799.53it/s]  2%|▏         | 9654/400000 [00:01<00:44, 8796.31it/s]  3%|▎         | 10542/400000 [00:01<00:44, 8820.79it/s]  3%|▎         | 11432/400000 [00:01<00:43, 8843.93it/s]  3%|▎         | 12313/400000 [00:01<00:43, 8831.28it/s]  3%|▎         | 13198/400000 [00:01<00:43, 8835.77it/s]  4%|▎         | 14083/400000 [00:01<00:43, 8838.24it/s]  4%|▎         | 14974/400000 [00:01<00:43, 8858.17it/s]  4%|▍         | 15861/400000 [00:01<00:43, 8858.76it/s]  4%|▍         | 16746/400000 [00:01<00:43, 8848.91it/s]  4%|▍         | 17630/400000 [00:02<00:43, 8755.91it/s]  5%|▍         | 18506/400000 [00:02<00:43, 8747.16it/s]  5%|▍         | 19382/400000 [00:02<00:43, 8749.75it/s]  5%|▌         | 20257/400000 [00:02<00:43, 8704.65it/s]  5%|▌         | 21137/400000 [00:02<00:43, 8731.70it/s]  6%|▌         | 22011/400000 [00:02<00:43, 8701.00it/s]  6%|▌         | 22883/400000 [00:02<00:43, 8704.99it/s]  6%|▌         | 23773/400000 [00:02<00:42, 8760.91it/s]  6%|▌         | 24650/400000 [00:02<00:44, 8485.26it/s]  6%|▋         | 25536/400000 [00:02<00:43, 8592.16it/s]  7%|▋         | 26427/400000 [00:03<00:43, 8684.54it/s]  7%|▋         | 27297/400000 [00:03<00:43, 8627.44it/s]  7%|▋         | 28185/400000 [00:03<00:42, 8701.51it/s]  7%|▋         | 29057/400000 [00:03<00:43, 8619.19it/s]  7%|▋         | 29942/400000 [00:03<00:42, 8684.76it/s]  8%|▊         | 30821/400000 [00:03<00:42, 8715.35it/s]  8%|▊         | 31694/400000 [00:03<00:43, 8460.26it/s]  8%|▊         | 32543/400000 [00:03<00:43, 8416.64it/s]  8%|▊         | 33387/400000 [00:03<00:43, 8379.15it/s]  9%|▊         | 34261/400000 [00:03<00:43, 8481.70it/s]  9%|▉         | 35111/400000 [00:04<00:43, 8411.72it/s]  9%|▉         | 35991/400000 [00:04<00:42, 8523.72it/s]  9%|▉         | 36876/400000 [00:04<00:42, 8617.47it/s]  9%|▉         | 37752/400000 [00:04<00:41, 8656.96it/s] 10%|▉         | 38619/400000 [00:04<00:41, 8631.46it/s] 10%|▉         | 39503/400000 [00:04<00:41, 8692.22it/s] 10%|█         | 40385/400000 [00:04<00:41, 8727.48it/s] 10%|█         | 41273/400000 [00:04<00:40, 8772.66it/s] 11%|█         | 42157/400000 [00:04<00:40, 8790.98it/s] 11%|█         | 43037/400000 [00:04<00:41, 8668.77it/s] 11%|█         | 43905/400000 [00:05<00:41, 8559.76it/s] 11%|█         | 44789/400000 [00:05<00:41, 8641.15it/s] 11%|█▏        | 45668/400000 [00:05<00:40, 8682.67it/s] 12%|█▏        | 46551/400000 [00:05<00:40, 8726.25it/s] 12%|█▏        | 47432/400000 [00:05<00:40, 8749.30it/s] 12%|█▏        | 48317/400000 [00:05<00:40, 8777.11it/s] 12%|█▏        | 49195/400000 [00:05<00:40, 8763.50it/s] 13%|█▎        | 50072/400000 [00:05<00:40, 8706.98it/s] 13%|█▎        | 50953/400000 [00:05<00:39, 8735.52it/s] 13%|█▎        | 51832/400000 [00:05<00:39, 8750.73it/s] 13%|█▎        | 52720/400000 [00:06<00:39, 8786.28it/s] 13%|█▎        | 53611/400000 [00:06<00:39, 8821.39it/s] 14%|█▎        | 54496/400000 [00:06<00:39, 8829.51it/s] 14%|█▍        | 55385/400000 [00:06<00:38, 8845.44it/s] 14%|█▍        | 56274/400000 [00:06<00:38, 8856.80it/s] 14%|█▍        | 57160/400000 [00:06<00:38, 8836.59it/s] 15%|█▍        | 58050/400000 [00:06<00:38, 8853.65it/s] 15%|█▍        | 58940/400000 [00:06<00:38, 8865.39it/s] 15%|█▍        | 59827/400000 [00:06<00:38, 8810.17it/s] 15%|█▌        | 60709/400000 [00:06<00:38, 8802.75it/s] 15%|█▌        | 61590/400000 [00:07<00:39, 8536.03it/s] 16%|█▌        | 62480/400000 [00:07<00:39, 8640.39it/s] 16%|█▌        | 63364/400000 [00:07<00:38, 8697.90it/s] 16%|█▌        | 64251/400000 [00:07<00:38, 8748.78it/s] 16%|█▋        | 65139/400000 [00:07<00:38, 8785.06it/s] 17%|█▋        | 66034/400000 [00:07<00:37, 8833.17it/s] 17%|█▋        | 66924/400000 [00:07<00:37, 8852.07it/s] 17%|█▋        | 67818/400000 [00:07<00:37, 8876.77it/s] 17%|█▋        | 68707/400000 [00:07<00:37, 8877.75it/s] 17%|█▋        | 69595/400000 [00:07<00:37, 8863.27it/s] 18%|█▊        | 70484/400000 [00:08<00:37, 8870.38it/s] 18%|█▊        | 71377/400000 [00:08<00:36, 8887.44it/s] 18%|█▊        | 72266/400000 [00:08<00:37, 8836.99it/s] 18%|█▊        | 73150/400000 [00:08<00:37, 8810.04it/s] 19%|█▊        | 74042/400000 [00:08<00:36, 8840.21it/s] 19%|█▊        | 74927/400000 [00:08<00:37, 8761.55it/s] 19%|█▉        | 75804/400000 [00:08<00:37, 8716.11it/s] 19%|█▉        | 76683/400000 [00:08<00:37, 8736.66it/s] 19%|█▉        | 77557/400000 [00:08<00:37, 8649.96it/s] 20%|█▉        | 78448/400000 [00:08<00:36, 8723.84it/s] 20%|█▉        | 79341/400000 [00:09<00:36, 8784.29it/s] 20%|██        | 80227/400000 [00:09<00:36, 8805.64it/s] 20%|██        | 81113/400000 [00:09<00:36, 8820.90it/s] 21%|██        | 82001/400000 [00:09<00:35, 8838.02it/s] 21%|██        | 82885/400000 [00:09<00:36, 8717.10it/s] 21%|██        | 83765/400000 [00:09<00:36, 8739.75it/s] 21%|██        | 84645/400000 [00:09<00:36, 8756.28it/s] 21%|██▏       | 85525/400000 [00:09<00:35, 8767.57it/s] 22%|██▏       | 86418/400000 [00:09<00:35, 8813.62it/s] 22%|██▏       | 87302/400000 [00:09<00:35, 8820.83it/s] 22%|██▏       | 88185/400000 [00:10<00:35, 8709.67it/s] 22%|██▏       | 89057/400000 [00:10<00:35, 8682.10it/s] 22%|██▏       | 89948/400000 [00:10<00:35, 8748.54it/s] 23%|██▎       | 90824/400000 [00:10<00:35, 8601.59it/s] 23%|██▎       | 91712/400000 [00:10<00:35, 8681.98it/s] 23%|██▎       | 92601/400000 [00:10<00:35, 8741.18it/s] 23%|██▎       | 93486/400000 [00:10<00:34, 8772.19it/s] 24%|██▎       | 94372/400000 [00:10<00:34, 8795.66it/s] 24%|██▍       | 95256/400000 [00:10<00:34, 8806.52it/s] 24%|██▍       | 96146/400000 [00:11<00:34, 8831.80it/s] 24%|██▍       | 97030/400000 [00:11<00:34, 8779.35it/s] 24%|██▍       | 97917/400000 [00:11<00:34, 8806.16it/s] 25%|██▍       | 98818/400000 [00:11<00:33, 8864.92it/s] 25%|██▍       | 99705/400000 [00:11<00:33, 8856.27it/s] 25%|██▌       | 100595/400000 [00:11<00:33, 8866.93it/s] 25%|██▌       | 101482/400000 [00:11<00:33, 8831.27it/s] 26%|██▌       | 102373/400000 [00:11<00:33, 8851.92it/s] 26%|██▌       | 103259/400000 [00:11<00:33, 8836.15it/s] 26%|██▌       | 104143/400000 [00:11<00:33, 8797.14it/s] 26%|██▋       | 105023/400000 [00:12<00:33, 8704.06it/s] 26%|██▋       | 105894/400000 [00:12<00:33, 8676.58it/s] 27%|██▋       | 106786/400000 [00:12<00:33, 8747.91it/s] 27%|██▋       | 107662/400000 [00:12<00:33, 8739.64it/s] 27%|██▋       | 108556/400000 [00:12<00:33, 8796.03it/s] 27%|██▋       | 109444/400000 [00:12<00:32, 8819.07it/s] 28%|██▊       | 110327/400000 [00:12<00:32, 8820.98it/s] 28%|██▊       | 111217/400000 [00:12<00:32, 8842.36it/s] 28%|██▊       | 112103/400000 [00:12<00:32, 8844.34it/s] 28%|██▊       | 112989/400000 [00:12<00:32, 8843.35it/s] 28%|██▊       | 113874/400000 [00:13<00:32, 8826.28it/s] 29%|██▊       | 114760/400000 [00:13<00:32, 8835.56it/s] 29%|██▉       | 115644/400000 [00:13<00:32, 8748.81it/s] 29%|██▉       | 116520/400000 [00:13<00:32, 8658.69it/s] 29%|██▉       | 117408/400000 [00:13<00:32, 8723.41it/s] 30%|██▉       | 118281/400000 [00:13<00:32, 8708.41it/s] 30%|██▉       | 119171/400000 [00:13<00:32, 8762.80it/s] 30%|███       | 120055/400000 [00:13<00:31, 8784.50it/s] 30%|███       | 120934/400000 [00:13<00:31, 8751.19it/s] 30%|███       | 121817/400000 [00:13<00:31, 8773.96it/s] 31%|███       | 122695/400000 [00:14<00:31, 8751.62it/s] 31%|███       | 123571/400000 [00:14<00:31, 8735.94it/s] 31%|███       | 124448/400000 [00:14<00:31, 8745.38it/s] 31%|███▏      | 125331/400000 [00:14<00:31, 8768.85it/s] 32%|███▏      | 126219/400000 [00:14<00:31, 8799.04it/s] 32%|███▏      | 127100/400000 [00:14<00:31, 8800.91it/s] 32%|███▏      | 127984/400000 [00:14<00:30, 8810.21it/s] 32%|███▏      | 128871/400000 [00:14<00:30, 8826.20it/s] 32%|███▏      | 129760/400000 [00:14<00:30, 8843.52it/s] 33%|███▎      | 130646/400000 [00:14<00:30, 8847.04it/s] 33%|███▎      | 131531/400000 [00:15<00:31, 8650.56it/s] 33%|███▎      | 132398/400000 [00:15<00:30, 8652.00it/s] 33%|███▎      | 133280/400000 [00:15<00:30, 8698.95it/s] 34%|███▎      | 134151/400000 [00:15<00:30, 8626.25it/s] 34%|███▍      | 135038/400000 [00:15<00:30, 8695.46it/s] 34%|███▍      | 135913/400000 [00:15<00:30, 8711.65it/s] 34%|███▍      | 136792/400000 [00:15<00:30, 8733.27it/s] 34%|███▍      | 137668/400000 [00:15<00:30, 8740.72it/s] 35%|███▍      | 138543/400000 [00:15<00:29, 8725.77it/s] 35%|███▍      | 139425/400000 [00:15<00:29, 8751.48it/s] 35%|███▌      | 140318/400000 [00:16<00:29, 8801.69it/s] 35%|███▌      | 141199/400000 [00:16<00:29, 8693.04it/s] 36%|███▌      | 142086/400000 [00:16<00:29, 8744.18it/s] 36%|███▌      | 142961/400000 [00:16<00:29, 8722.90it/s] 36%|███▌      | 143849/400000 [00:16<00:29, 8767.41it/s] 36%|███▌      | 144742/400000 [00:16<00:28, 8813.73it/s] 36%|███▋      | 145633/400000 [00:16<00:28, 8840.09it/s] 37%|███▋      | 146525/400000 [00:16<00:28, 8862.96it/s] 37%|███▋      | 147416/400000 [00:16<00:28, 8875.47it/s] 37%|███▋      | 148304/400000 [00:16<00:28, 8873.92it/s] 37%|███▋      | 149194/400000 [00:17<00:28, 8879.84it/s] 38%|███▊      | 150083/400000 [00:17<00:29, 8563.33it/s] 38%|███▊      | 150956/400000 [00:17<00:28, 8612.57it/s] 38%|███▊      | 151839/400000 [00:17<00:28, 8673.92it/s] 38%|███▊      | 152727/400000 [00:17<00:28, 8733.20it/s] 38%|███▊      | 153617/400000 [00:17<00:28, 8780.75it/s] 39%|███▊      | 154509/400000 [00:17<00:27, 8819.58it/s] 39%|███▉      | 155394/400000 [00:17<00:27, 8826.43it/s] 39%|███▉      | 156284/400000 [00:17<00:27, 8847.11it/s] 39%|███▉      | 157174/400000 [00:17<00:27, 8861.46it/s] 40%|███▉      | 158061/400000 [00:18<00:27, 8854.73it/s] 40%|███▉      | 158949/400000 [00:18<00:27, 8862.27it/s] 40%|███▉      | 159840/400000 [00:18<00:27, 8875.26it/s] 40%|████      | 160728/400000 [00:18<00:27, 8755.95it/s] 40%|████      | 161616/400000 [00:18<00:27, 8790.83it/s] 41%|████      | 162509/400000 [00:18<00:26, 8832.01it/s] 41%|████      | 163400/400000 [00:18<00:26, 8854.01it/s] 41%|████      | 164294/400000 [00:18<00:26, 8877.52it/s] 41%|████▏     | 165182/400000 [00:18<00:26, 8853.36it/s] 42%|████▏     | 166068/400000 [00:18<00:26, 8840.44it/s] 42%|████▏     | 166953/400000 [00:19<00:26, 8841.35it/s] 42%|████▏     | 167844/400000 [00:19<00:26, 8859.18it/s] 42%|████▏     | 168730/400000 [00:19<00:26, 8736.13it/s] 42%|████▏     | 169614/400000 [00:19<00:26, 8766.71it/s] 43%|████▎     | 170505/400000 [00:19<00:26, 8808.41it/s] 43%|████▎     | 171397/400000 [00:19<00:25, 8839.03it/s] 43%|████▎     | 172284/400000 [00:19<00:25, 8845.94it/s] 43%|████▎     | 173175/400000 [00:19<00:25, 8862.31it/s] 44%|████▎     | 174063/400000 [00:19<00:25, 8865.03it/s] 44%|████▎     | 174959/400000 [00:19<00:25, 8892.00it/s] 44%|████▍     | 175855/400000 [00:20<00:25, 8911.65it/s] 44%|████▍     | 176747/400000 [00:20<00:25, 8906.90it/s] 44%|████▍     | 177638/400000 [00:20<00:25, 8880.27it/s] 45%|████▍     | 178527/400000 [00:20<00:25, 8772.32it/s] 45%|████▍     | 179411/400000 [00:20<00:25, 8791.99it/s] 45%|████▌     | 180291/400000 [00:20<00:25, 8780.92it/s] 45%|████▌     | 181170/400000 [00:20<00:25, 8577.67it/s] 46%|████▌     | 182059/400000 [00:20<00:25, 8667.55it/s] 46%|████▌     | 182942/400000 [00:20<00:24, 8713.26it/s] 46%|████▌     | 183834/400000 [00:20<00:24, 8773.27it/s] 46%|████▌     | 184723/400000 [00:21<00:24, 8807.16it/s] 46%|████▋     | 185605/400000 [00:21<00:24, 8809.95it/s] 47%|████▋     | 186496/400000 [00:21<00:24, 8838.14it/s] 47%|████▋     | 187381/400000 [00:21<00:24, 8805.61it/s] 47%|████▋     | 188268/400000 [00:21<00:23, 8822.38it/s] 47%|████▋     | 189158/400000 [00:21<00:23, 8844.21it/s] 48%|████▊     | 190044/400000 [00:21<00:23, 8848.86it/s] 48%|████▊     | 190929/400000 [00:21<00:23, 8846.55it/s] 48%|████▊     | 191814/400000 [00:21<00:23, 8843.73it/s] 48%|████▊     | 192703/400000 [00:21<00:23, 8855.46it/s] 48%|████▊     | 193589/400000 [00:22<00:23, 8852.32it/s] 49%|████▊     | 194475/400000 [00:22<00:23, 8841.51it/s] 49%|████▉     | 195360/400000 [00:22<00:23, 8545.93it/s] 49%|████▉     | 196243/400000 [00:22<00:23, 8628.51it/s] 49%|████▉     | 197124/400000 [00:22<00:23, 8680.72it/s] 50%|████▉     | 198017/400000 [00:22<00:23, 8753.62it/s] 50%|████▉     | 198907/400000 [00:22<00:22, 8794.89it/s] 50%|████▉     | 199788/400000 [00:22<00:23, 8576.55it/s] 50%|█████     | 200672/400000 [00:22<00:23, 8653.33it/s] 50%|█████     | 201546/400000 [00:23<00:22, 8678.10it/s] 51%|█████     | 202435/400000 [00:23<00:22, 8739.60it/s] 51%|█████     | 203326/400000 [00:23<00:22, 8787.87it/s] 51%|█████     | 204207/400000 [00:23<00:22, 8791.85it/s] 51%|█████▏    | 205087/400000 [00:23<00:22, 8704.66it/s] 51%|█████▏    | 205973/400000 [00:23<00:22, 8749.59it/s] 52%|█████▏    | 206860/400000 [00:23<00:21, 8784.05it/s] 52%|█████▏    | 207745/400000 [00:23<00:21, 8802.62it/s] 52%|█████▏    | 208635/400000 [00:23<00:21, 8829.02it/s] 52%|█████▏    | 209524/400000 [00:23<00:21, 8846.55it/s] 53%|█████▎    | 210409/400000 [00:24<00:21, 8792.58it/s] 53%|█████▎    | 211289/400000 [00:24<00:21, 8749.09it/s] 53%|█████▎    | 212165/400000 [00:24<00:21, 8599.74it/s] 53%|█████▎    | 213042/400000 [00:24<00:21, 8649.94it/s] 53%|█████▎    | 213909/400000 [00:24<00:21, 8654.55it/s] 54%|█████▎    | 214801/400000 [00:24<00:21, 8730.45it/s] 54%|█████▍    | 215684/400000 [00:24<00:21, 8757.26it/s] 54%|█████▍    | 216572/400000 [00:24<00:20, 8790.82it/s] 54%|█████▍    | 217463/400000 [00:24<00:20, 8819.37it/s] 55%|█████▍    | 218346/400000 [00:24<00:20, 8819.92it/s] 55%|█████▍    | 219236/400000 [00:25<00:20, 8841.29it/s] 55%|█████▌    | 220123/400000 [00:25<00:20, 8848.82it/s] 55%|█████▌    | 221012/400000 [00:25<00:20, 8858.12it/s] 55%|█████▌    | 221898/400000 [00:25<00:20, 8839.44it/s] 56%|█████▌    | 222782/400000 [00:25<00:20, 8835.50it/s] 56%|█████▌    | 223670/400000 [00:25<00:19, 8847.95it/s] 56%|█████▌    | 224562/400000 [00:25<00:19, 8868.34it/s] 56%|█████▋    | 225454/400000 [00:25<00:19, 8881.39it/s] 57%|█████▋    | 226343/400000 [00:25<00:19, 8871.94it/s] 57%|█████▋    | 227231/400000 [00:25<00:19, 8846.89it/s] 57%|█████▋    | 228116/400000 [00:26<00:19, 8844.64it/s] 57%|█████▋    | 229006/400000 [00:26<00:19, 8859.52it/s] 57%|█████▋    | 229895/400000 [00:26<00:19, 8866.82it/s] 58%|█████▊    | 230782/400000 [00:26<00:19, 8825.60it/s] 58%|█████▊    | 231665/400000 [00:26<00:19, 8814.57it/s] 58%|█████▊    | 232547/400000 [00:26<00:19, 8807.15it/s] 58%|█████▊    | 233428/400000 [00:26<00:18, 8791.75it/s] 59%|█████▊    | 234311/400000 [00:26<00:18, 8800.68it/s] 59%|█████▉    | 235192/400000 [00:26<00:18, 8777.24it/s] 59%|█████▉    | 236070/400000 [00:26<00:18, 8758.60it/s] 59%|█████▉    | 236946/400000 [00:27<00:18, 8732.99it/s] 59%|█████▉    | 237827/400000 [00:27<00:18, 8755.33it/s] 60%|█████▉    | 238703/400000 [00:27<00:18, 8748.02it/s] 60%|█████▉    | 239578/400000 [00:27<00:18, 8668.66it/s] 60%|██████    | 240464/400000 [00:27<00:18, 8724.14it/s] 60%|██████    | 241337/400000 [00:27<00:18, 8715.86it/s] 61%|██████    | 242219/400000 [00:27<00:18, 8744.67it/s] 61%|██████    | 243108/400000 [00:27<00:17, 8787.44it/s] 61%|██████    | 243995/400000 [00:27<00:17, 8810.47it/s] 61%|██████    | 244877/400000 [00:27<00:17, 8742.03it/s] 61%|██████▏   | 245752/400000 [00:28<00:17, 8701.51it/s] 62%|██████▏   | 246637/400000 [00:28<00:17, 8743.91it/s] 62%|██████▏   | 247529/400000 [00:28<00:17, 8795.16it/s] 62%|██████▏   | 248413/400000 [00:28<00:17, 8808.25it/s] 62%|██████▏   | 249308/400000 [00:28<00:17, 8849.31it/s] 63%|██████▎   | 250194/400000 [00:28<00:16, 8832.22it/s] 63%|██████▎   | 251087/400000 [00:28<00:16, 8859.63it/s] 63%|██████▎   | 251974/400000 [00:28<00:16, 8861.65it/s] 63%|██████▎   | 252862/400000 [00:28<00:16, 8864.41it/s] 63%|██████▎   | 253750/400000 [00:28<00:16, 8867.65it/s] 64%|██████▎   | 254637/400000 [00:29<00:16, 8855.40it/s] 64%|██████▍   | 255524/400000 [00:29<00:16, 8859.62it/s] 64%|██████▍   | 256414/400000 [00:29<00:16, 8869.32it/s] 64%|██████▍   | 257301/400000 [00:29<00:16, 8712.38it/s] 65%|██████▍   | 258186/400000 [00:29<00:16, 8750.18it/s] 65%|██████▍   | 259067/400000 [00:29<00:16, 8766.72it/s] 65%|██████▍   | 259963/400000 [00:29<00:15, 8821.36it/s] 65%|██████▌   | 260846/400000 [00:29<00:16, 8547.16it/s] 65%|██████▌   | 261733/400000 [00:29<00:16, 8640.85it/s] 66%|██████▌   | 262627/400000 [00:29<00:15, 8726.20it/s] 66%|██████▌   | 263502/400000 [00:30<00:15, 8585.94it/s] 66%|██████▌   | 264363/400000 [00:30<00:15, 8539.46it/s] 66%|██████▋   | 265241/400000 [00:30<00:15, 8608.30it/s] 67%|██████▋   | 266114/400000 [00:30<00:15, 8643.30it/s] 67%|██████▋   | 266979/400000 [00:30<00:15, 8538.22it/s] 67%|██████▋   | 267859/400000 [00:30<00:15, 8612.34it/s] 67%|██████▋   | 268743/400000 [00:30<00:15, 8677.66it/s] 67%|██████▋   | 269634/400000 [00:30<00:14, 8744.12it/s] 68%|██████▊   | 270519/400000 [00:30<00:14, 8774.27it/s] 68%|██████▊   | 271410/400000 [00:30<00:14, 8813.58it/s] 68%|██████▊   | 272292/400000 [00:31<00:14, 8749.01it/s] 68%|██████▊   | 273168/400000 [00:31<00:14, 8701.05it/s] 69%|██████▊   | 274039/400000 [00:31<00:14, 8699.53it/s] 69%|██████▊   | 274927/400000 [00:31<00:14, 8752.72it/s] 69%|██████▉   | 275815/400000 [00:31<00:14, 8788.87it/s] 69%|██████▉   | 276695/400000 [00:31<00:14, 8688.40it/s] 69%|██████▉   | 277586/400000 [00:31<00:13, 8752.14it/s] 70%|██████▉   | 278474/400000 [00:31<00:13, 8788.95it/s] 70%|██████▉   | 279363/400000 [00:31<00:13, 8817.37it/s] 70%|███████   | 280245/400000 [00:31<00:13, 8811.69it/s] 70%|███████   | 281127/400000 [00:32<00:13, 8804.64it/s] 71%|███████   | 282017/400000 [00:32<00:13, 8830.41it/s] 71%|███████   | 282901/400000 [00:32<00:13, 8802.90it/s] 71%|███████   | 283782/400000 [00:32<00:13, 8761.76it/s] 71%|███████   | 284670/400000 [00:32<00:13, 8794.11it/s] 71%|███████▏  | 285550/400000 [00:32<00:13, 8775.60it/s] 72%|███████▏  | 286438/400000 [00:32<00:12, 8804.02it/s] 72%|███████▏  | 287319/400000 [00:32<00:13, 8461.31it/s] 72%|███████▏  | 288209/400000 [00:32<00:13, 8586.77it/s] 72%|███████▏  | 289099/400000 [00:32<00:12, 8675.97it/s] 72%|███████▏  | 289969/400000 [00:33<00:12, 8603.08it/s] 73%|███████▎  | 290831/400000 [00:33<00:12, 8577.67it/s] 73%|███████▎  | 291694/400000 [00:33<00:12, 8593.13it/s] 73%|███████▎  | 292580/400000 [00:33<00:12, 8669.74it/s] 73%|███████▎  | 293448/400000 [00:33<00:12, 8656.21it/s] 74%|███████▎  | 294315/400000 [00:33<00:12, 8558.79it/s] 74%|███████▍  | 295192/400000 [00:33<00:12, 8618.57it/s] 74%|███████▍  | 296064/400000 [00:33<00:12, 8646.73it/s] 74%|███████▍  | 296930/400000 [00:33<00:11, 8616.26it/s] 74%|███████▍  | 297802/400000 [00:34<00:11, 8646.07it/s] 75%|███████▍  | 298686/400000 [00:34<00:11, 8701.02it/s] 75%|███████▍  | 299565/400000 [00:34<00:11, 8727.39it/s] 75%|███████▌  | 300438/400000 [00:34<00:11, 8627.44it/s] 75%|███████▌  | 301324/400000 [00:34<00:11, 8692.96it/s] 76%|███████▌  | 302206/400000 [00:34<00:11, 8729.20it/s] 76%|███████▌  | 303090/400000 [00:34<00:11, 8760.40it/s] 76%|███████▌  | 303976/400000 [00:34<00:10, 8789.13it/s] 76%|███████▌  | 304857/400000 [00:34<00:10, 8794.17it/s] 76%|███████▋  | 305744/400000 [00:34<00:10, 8814.10it/s] 77%|███████▋  | 306629/400000 [00:35<00:10, 8822.53it/s] 77%|███████▋  | 307513/400000 [00:35<00:10, 8825.18it/s] 77%|███████▋  | 308399/400000 [00:35<00:10, 8833.74it/s] 77%|███████▋  | 309287/400000 [00:35<00:10, 8846.47it/s] 78%|███████▊  | 310172/400000 [00:35<00:10, 8833.88it/s] 78%|███████▊  | 311059/400000 [00:35<00:10, 8843.84it/s] 78%|███████▊  | 311944/400000 [00:35<00:09, 8823.80it/s] 78%|███████▊  | 312827/400000 [00:35<00:09, 8825.42it/s] 78%|███████▊  | 313714/400000 [00:35<00:09, 8835.91it/s] 79%|███████▊  | 314604/400000 [00:35<00:09, 8853.63it/s] 79%|███████▉  | 315490/400000 [00:36<00:09, 8846.67it/s] 79%|███████▉  | 316377/400000 [00:36<00:09, 8853.14it/s] 79%|███████▉  | 317265/400000 [00:36<00:09, 8860.01it/s] 80%|███████▉  | 318152/400000 [00:36<00:09, 8737.20it/s] 80%|███████▉  | 319028/400000 [00:36<00:09, 8740.84it/s] 80%|███████▉  | 319905/400000 [00:36<00:09, 8747.33it/s] 80%|████████  | 320781/400000 [00:36<00:09, 8749.30it/s] 80%|████████  | 321667/400000 [00:36<00:08, 8781.59it/s] 81%|████████  | 322546/400000 [00:36<00:08, 8748.60it/s] 81%|████████  | 323429/400000 [00:36<00:08, 8772.76it/s] 81%|████████  | 324316/400000 [00:37<00:08, 8799.13it/s] 81%|████████▏ | 325206/400000 [00:37<00:08, 8827.73it/s] 82%|████████▏ | 326093/400000 [00:37<00:08, 8837.74it/s] 82%|████████▏ | 326977/400000 [00:37<00:08, 8699.69it/s] 82%|████████▏ | 327864/400000 [00:37<00:08, 8749.40it/s] 82%|████████▏ | 328740/400000 [00:37<00:08, 8457.32it/s] 82%|████████▏ | 329589/400000 [00:37<00:08, 8434.20it/s] 83%|████████▎ | 330477/400000 [00:37<00:08, 8561.48it/s] 83%|████████▎ | 331363/400000 [00:37<00:07, 8647.22it/s] 83%|████████▎ | 332251/400000 [00:37<00:07, 8714.42it/s] 83%|████████▎ | 333144/400000 [00:38<00:07, 8776.60it/s] 84%|████████▎ | 334029/400000 [00:38<00:07, 8796.33it/s] 84%|████████▎ | 334917/400000 [00:38<00:07, 8821.03it/s] 84%|████████▍ | 335800/400000 [00:38<00:07, 8805.05it/s] 84%|████████▍ | 336687/400000 [00:38<00:07, 8823.87it/s] 84%|████████▍ | 337582/400000 [00:38<00:07, 8860.93it/s] 85%|████████▍ | 338471/400000 [00:38<00:06, 8866.88it/s] 85%|████████▍ | 339358/400000 [00:38<00:06, 8863.57it/s] 85%|████████▌ | 340248/400000 [00:38<00:06, 8874.32it/s] 85%|████████▌ | 341136/400000 [00:38<00:06, 8808.86it/s] 86%|████████▌ | 342018/400000 [00:39<00:06, 8799.31it/s] 86%|████████▌ | 342903/400000 [00:39<00:06, 8814.26it/s] 86%|████████▌ | 343796/400000 [00:39<00:06, 8847.50it/s] 86%|████████▌ | 344688/400000 [00:39<00:06, 8867.25it/s] 86%|████████▋ | 345575/400000 [00:39<00:06, 8818.66it/s] 87%|████████▋ | 346464/400000 [00:39<00:06, 8838.32it/s] 87%|████████▋ | 347349/400000 [00:39<00:05, 8838.90it/s] 87%|████████▋ | 348239/400000 [00:39<00:05, 8854.36it/s] 87%|████████▋ | 349141/400000 [00:39<00:05, 8902.70it/s] 88%|████████▊ | 350032/400000 [00:39<00:05, 8891.45it/s] 88%|████████▊ | 350922/400000 [00:40<00:05, 8854.62it/s] 88%|████████▊ | 351811/400000 [00:40<00:05, 8863.43it/s] 88%|████████▊ | 352698/400000 [00:40<00:05, 8858.30it/s] 88%|████████▊ | 353584/400000 [00:40<00:05, 8851.85it/s] 89%|████████▊ | 354470/400000 [00:40<00:05, 8617.28it/s] 89%|████████▉ | 355353/400000 [00:40<00:05, 8677.48it/s] 89%|████████▉ | 356243/400000 [00:40<00:05, 8740.34it/s] 89%|████████▉ | 357131/400000 [00:40<00:04, 8780.07it/s] 90%|████████▉ | 358015/400000 [00:40<00:04, 8797.27it/s] 90%|████████▉ | 358905/400000 [00:40<00:04, 8826.80it/s] 90%|████████▉ | 359794/400000 [00:41<00:04, 8845.09it/s] 90%|█████████ | 360683/400000 [00:41<00:04, 8858.50it/s] 90%|█████████ | 361570/400000 [00:41<00:04, 8851.93it/s] 91%|█████████ | 362456/400000 [00:41<00:04, 8746.44it/s] 91%|█████████ | 363338/400000 [00:41<00:04, 8766.89it/s] 91%|█████████ | 364224/400000 [00:41<00:04, 8793.84it/s] 91%|█████████▏| 365114/400000 [00:41<00:03, 8822.73it/s] 92%|█████████▏| 366000/400000 [00:41<00:03, 8833.34it/s] 92%|█████████▏| 366893/400000 [00:41<00:03, 8860.76it/s] 92%|█████████▏| 367780/400000 [00:41<00:03, 8862.29it/s] 92%|█████████▏| 368667/400000 [00:42<00:03, 8800.86it/s] 92%|█████████▏| 369555/400000 [00:42<00:03, 8822.09it/s] 93%|█████████▎| 370438/400000 [00:42<00:03, 8807.48it/s] 93%|█████████▎| 371325/400000 [00:42<00:03, 8825.70it/s] 93%|█████████▎| 372216/400000 [00:42<00:03, 8847.93it/s] 93%|█████████▎| 373104/400000 [00:42<00:03, 8855.76it/s] 93%|█████████▎| 373994/400000 [00:42<00:02, 8868.69it/s] 94%|█████████▎| 374881/400000 [00:42<00:02, 8827.79it/s] 94%|█████████▍| 375770/400000 [00:42<00:02, 8844.91it/s] 94%|█████████▍| 376660/400000 [00:42<00:02, 8858.63it/s] 94%|█████████▍| 377550/400000 [00:43<00:02, 8869.67it/s] 95%|█████████▍| 378438/400000 [00:43<00:02, 8866.78it/s] 95%|█████████▍| 379325/400000 [00:43<00:02, 8679.81it/s] 95%|█████████▌| 380215/400000 [00:43<00:02, 8742.75it/s] 95%|█████████▌| 381091/400000 [00:43<00:02, 8675.34it/s] 95%|█████████▌| 381973/400000 [00:43<00:02, 8717.23it/s] 96%|█████████▌| 382864/400000 [00:43<00:01, 8772.38it/s] 96%|█████████▌| 383748/400000 [00:43<00:01, 8791.78it/s] 96%|█████████▌| 384635/400000 [00:43<00:01, 8813.61it/s] 96%|█████████▋| 385517/400000 [00:43<00:01, 8806.80it/s] 97%|█████████▋| 386404/400000 [00:44<00:01, 8823.07it/s] 97%|█████████▋| 387287/400000 [00:44<00:01, 8823.29it/s] 97%|█████████▋| 388170/400000 [00:44<00:01, 8807.10it/s] 97%|█████████▋| 389062/400000 [00:44<00:01, 8838.64it/s] 97%|█████████▋| 389946/400000 [00:44<00:01, 8818.84it/s] 98%|█████████▊| 390828/400000 [00:44<00:01, 8799.62it/s] 98%|█████████▊| 391709/400000 [00:44<00:00, 8782.17it/s] 98%|█████████▊| 392592/400000 [00:44<00:00, 8795.62it/s] 98%|█████████▊| 393472/400000 [00:44<00:00, 8783.48it/s] 99%|█████████▊| 394356/400000 [00:44<00:00, 8799.66it/s] 99%|█████████▉| 395242/400000 [00:45<00:00, 8815.67it/s] 99%|█████████▉| 396127/400000 [00:45<00:00, 8824.57it/s] 99%|█████████▉| 397010/400000 [00:45<00:00, 8818.30it/s] 99%|█████████▉| 397893/400000 [00:45<00:00, 8819.40it/s]100%|█████████▉| 398783/400000 [00:45<00:00, 8840.81it/s]100%|█████████▉| 399673/400000 [00:45<00:00, 8855.86it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8768.27it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f837a74e940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011567330155761456 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.011260444902656071 	 Accuracy: 51

  model saves at 51% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 16111 out of table with 16077 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 16111 out of table with 16077 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-15 18:25:04.515117: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 18:25:04.519100: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095110000 Hz
2020-05-15 18:25:04.519390: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563a32f0cf30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 18:25:04.519407: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f837d643588> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 8.1573 - accuracy: 0.4680
 2000/25000 [=>............................] - ETA: 8s - loss: 7.9886 - accuracy: 0.4790 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7331 - accuracy: 0.4957
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7050 - accuracy: 0.4975
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6794 - accuracy: 0.4992
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6579 - accuracy: 0.5006
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7356 - accuracy: 0.4955
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6942 - accuracy: 0.4982
11000/25000 [============>.................] - ETA: 3s - loss: 7.7001 - accuracy: 0.4978
12000/25000 [=============>................] - ETA: 3s - loss: 7.6909 - accuracy: 0.4984
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6749 - accuracy: 0.4995
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6820 - accuracy: 0.4990
15000/25000 [=================>............] - ETA: 2s - loss: 7.6840 - accuracy: 0.4989
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6781 - accuracy: 0.4992
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6630 - accuracy: 0.5002
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6624 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6553 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6559 - accuracy: 0.5007
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6674 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6778 - accuracy: 0.4993
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6726 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 8s 304us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f82d58ccb38> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f837a74ecf8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.7004 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.6685 - val_crf_viterbi_accuracy: 0.0267

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

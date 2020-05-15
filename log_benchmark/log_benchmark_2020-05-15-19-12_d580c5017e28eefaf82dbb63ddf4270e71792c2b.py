
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f732d846f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 19:12:49.037072
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 19:12:49.042056
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 19:12:49.046232
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 19:12:49.049772
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f7339610470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352205.9375
Epoch 2/10

1/1 [==============================] - 0s 91ms/step - loss: 248294.0469
Epoch 3/10

1/1 [==============================] - 0s 92ms/step - loss: 138106.2812
Epoch 4/10

1/1 [==============================] - 0s 93ms/step - loss: 72976.6719
Epoch 5/10

1/1 [==============================] - 0s 89ms/step - loss: 38643.4727
Epoch 6/10

1/1 [==============================] - 0s 87ms/step - loss: 22236.7793
Epoch 7/10

1/1 [==============================] - 0s 90ms/step - loss: 14046.7539
Epoch 8/10

1/1 [==============================] - 0s 88ms/step - loss: 9638.0703
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 7094.7432
Epoch 10/10

1/1 [==============================] - 0s 97ms/step - loss: 5559.2095

  #### Inference Need return ypred, ytrue ######################### 
[[ 3.38430732e-01  1.83908954e-01 -9.74345267e-01  6.06041670e-01
   8.91749799e-01  3.44542533e-01 -1.39910877e-02 -1.62852740e+00
  -1.21050537e+00 -1.93740413e-01  2.82902718e-01  9.66435492e-01
   5.63003600e-01 -1.00694013e+00 -4.83347505e-01  1.50143063e+00
  -2.00363219e-01  5.04163653e-02 -8.45755756e-01 -2.74769694e-01
   1.20043206e+00  1.12967515e+00  1.64709166e-02  2.83872038e-02
  -2.02952409e+00  1.09897010e-01  1.26675713e+00  6.64019942e-01
   8.09094787e-01  1.42294610e+00  1.94648996e-01 -1.26961648e+00
   4.06468034e-01 -6.96509004e-01  1.83851168e-01 -1.51378453e+00
  -1.09453046e+00  1.55058551e+00  1.49009168e-01 -9.15495694e-01
   5.13624132e-01  4.07546759e-01 -1.69693008e-02 -9.13904905e-01
   1.52214384e+00 -2.24794054e+00  4.60021436e-01 -4.67657447e-01
   6.72515631e-01 -1.79837301e-01  1.20976722e+00  9.97192621e-01
   9.85330701e-01 -5.89116991e-01  3.83469701e-01 -8.79151523e-01
   1.48478711e+00 -6.35103405e-01 -1.90275252e+00  8.28973591e-01
   4.20418352e-01  9.31057358e+00  8.04467773e+00  7.68647146e+00
   8.52054691e+00  9.57011986e+00  8.62407303e+00  1.12596369e+01
   7.75160933e+00  9.83133984e+00  7.15079451e+00  8.68596554e+00
   9.68488026e+00  8.20743847e+00  9.00714779e+00  9.86214256e+00
   1.08470592e+01  7.44013166e+00  8.83704948e+00  1.05855579e+01
   8.00698280e+00  9.22058678e+00  7.75034142e+00  8.44775486e+00
   7.99627447e+00  1.03046446e+01  8.53384209e+00  7.11949635e+00
   7.32004833e+00  1.00955362e+01  8.55289173e+00  9.96496677e+00
   1.04329395e+01  9.17099094e+00  1.01674719e+01  6.77318287e+00
   6.54493237e+00  1.06781292e+01  8.81937695e+00  1.00482149e+01
   8.17437935e+00  7.06586742e+00  9.54027462e+00  7.46014500e+00
   8.99034500e+00  9.23531246e+00  1.03258381e+01  8.44037914e+00
   8.98527908e+00  8.53520393e+00  9.03124809e+00  8.89013290e+00
   8.09197903e+00  8.66788960e+00  1.04153814e+01  9.70667076e+00
   9.99432087e+00  8.93203640e+00  9.09027195e+00  8.30671978e+00
  -4.64998454e-01 -6.29821941e-02 -2.03648901e+00 -1.02374387e+00
  -1.70402300e+00 -4.26381491e-02  4.27285194e-01 -5.11233687e-01
  -1.24447846e+00 -1.36200690e+00  1.57983029e+00  9.58078742e-01
  -7.97009289e-01  9.08083022e-02  6.01436257e-01  9.02919352e-01
  -1.52763307e+00  1.91448998e+00  2.26707983e+00 -5.65733612e-01
  -3.35231394e-01 -3.47028792e-01  2.37092093e-01  9.66681302e-01
   2.28556722e-01 -1.50226843e+00 -1.53195798e+00 -3.92998010e-01
  -7.90484697e-02  9.15892661e-01 -2.61274767e+00 -5.36262691e-01
  -9.08171356e-01 -9.09872949e-01  9.67396498e-02  4.85751450e-01
   1.45943916e+00 -1.82847452e+00  7.99240053e-01  1.63862097e+00
   1.10246336e+00  8.10446203e-01  1.36535430e+00 -7.55578279e-01
   1.25122011e+00  1.08423102e+00  1.07377112e+00  1.14381462e-02
  -1.27122283e-01 -4.96592730e-01 -1.36412680e-02 -8.77275050e-01
   1.91576511e-01 -1.07737589e+00 -1.26734412e+00  1.21429694e+00
   2.22609901e+00 -2.24848819e+00 -1.32817340e+00  1.64147508e+00
   6.68289900e-01  2.76429987e+00  2.84242153e-01  9.28320467e-01
   2.15088940e+00  1.62863851e-01  1.93327069e-01  7.18227684e-01
   1.16083586e+00  5.77400565e-01  1.49110317e+00  3.03419828e+00
   2.16647291e+00  5.00691533e-01  8.51945519e-01  1.17527223e+00
   2.35763407e+00  1.37639308e+00  8.47402632e-01  1.96522105e+00
   1.07412124e+00  6.87005520e-01  2.23458147e+00  1.18387628e+00
   2.34354734e+00  2.22768211e+00  5.47295570e-01  2.49726772e-01
   1.53781390e+00  2.44665086e-01  2.85169768e+00  2.87887764e+00
   7.94161677e-01  6.35715425e-01  1.39836335e+00  1.64103580e+00
   1.41021276e+00  2.06538141e-01  6.60545647e-01  5.32646775e-01
   9.92776871e-01  1.73842788e+00  8.96832287e-01  1.90761113e+00
   1.41651165e+00  4.68925893e-01  7.51852036e-01  6.02914572e-01
   6.45730495e-02  2.41149187e-01  2.89289904e+00  1.06337285e+00
   7.96498120e-01  2.79874849e+00  1.53230155e+00  4.64562297e-01
   3.73897374e-01  1.11807311e+00  6.33135915e-01  3.87943745e-01
   1.68573976e-01  1.07080917e+01  7.89426613e+00  1.00959406e+01
   7.06655788e+00  9.74776554e+00  8.63432026e+00  1.07512455e+01
   8.04427052e+00  1.05252161e+01  8.14288235e+00  9.10351849e+00
   7.86905813e+00  7.59410381e+00  7.34303474e+00  1.05393763e+01
   9.13388157e+00  8.18492126e+00  9.70632458e+00  9.76284409e+00
   9.60947514e+00  9.57041073e+00  1.11033545e+01  7.43896389e+00
   8.00306988e+00  1.07467022e+01  9.75025749e+00  9.77222347e+00
   8.12564468e+00  9.15819645e+00  8.60987854e+00  8.33759022e+00
   8.41792011e+00  1.02860975e+01  9.81769562e+00  9.55859852e+00
   1.05075979e+01  9.27953243e+00  1.09504633e+01  1.01914186e+01
   9.18384838e+00  9.11828136e+00  7.77984476e+00  1.02562733e+01
   8.12598419e+00  9.50988007e+00  9.64831257e+00  9.25713539e+00
   8.84950638e+00  1.04125471e+01  1.05629511e+01  9.39584160e+00
   1.07476387e+01  7.13696814e+00  1.02913876e+01  9.31381035e+00
   8.42315960e+00  8.28334236e+00  9.19672871e+00  1.03797064e+01
   2.86208463e+00  1.26021743e+00  1.74483538e-01  9.59666908e-01
   1.47090471e+00  1.24587202e+00  1.08583057e+00  2.33303666e-01
   1.95854878e+00  8.47854435e-01  8.38645637e-01  2.23226762e+00
   5.03851473e-01  2.00857687e+00  1.03623140e+00  2.41506886e+00
   1.40414715e+00  4.28389311e-02  2.11337328e+00  2.17970967e-01
   1.28860474e+00  9.07193005e-01  2.78868389e+00  1.44504309e+00
   1.75593567e+00  6.45143628e-01  6.06239438e-01  3.46971393e-01
   1.48774111e+00  8.49942684e-01  2.32895112e+00  1.49824762e+00
   1.30034351e+00  2.48648524e-01  2.78896713e+00  4.74773884e-01
   1.00552189e+00  1.46873593e-01  8.92829299e-01  4.33497071e-01
   6.62761331e-01  1.64491320e+00  3.57346296e-01  1.53003097e-01
   1.26222181e+00  5.39737225e-01  1.18482590e+00  7.23223686e-01
   1.35463881e+00  2.03527117e+00  1.29374480e+00  7.45442271e-01
   1.32967341e+00  1.03423715e+00  3.97297382e-01  1.83027005e+00
   3.91950488e-01  3.79307091e-01  9.54324007e-01  2.95237899e-01
  -1.28775053e+01  6.53152180e+00 -1.20234652e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 19:12:58.426161
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.6187
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 19:12:58.430266
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8608.99
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 19:12:58.433931
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.0457
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 19:12:58.437525
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -769.996
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140132319368640
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140129789248064
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140129789248568
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140129789249072
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140129789249576
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140129789250080

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f7319232e10> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.795709
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.750960
grad_step = 000002, loss = 0.714343
grad_step = 000003, loss = 0.674651
grad_step = 000004, loss = 0.625913
grad_step = 000005, loss = 0.569477
grad_step = 000006, loss = 0.512047
grad_step = 000007, loss = 0.459607
grad_step = 000008, loss = 0.433032
grad_step = 000009, loss = 0.430841
grad_step = 000010, loss = 0.413017
grad_step = 000011, loss = 0.380711
grad_step = 000012, loss = 0.355401
grad_step = 000013, loss = 0.341760
grad_step = 000014, loss = 0.333837
grad_step = 000015, loss = 0.325172
grad_step = 000016, loss = 0.313441
grad_step = 000017, loss = 0.299383
grad_step = 000018, loss = 0.285283
grad_step = 000019, loss = 0.272990
grad_step = 000020, loss = 0.262840
grad_step = 000021, loss = 0.253325
grad_step = 000022, loss = 0.242946
grad_step = 000023, loss = 0.231153
grad_step = 000024, loss = 0.219914
grad_step = 000025, loss = 0.210501
grad_step = 000026, loss = 0.202357
grad_step = 000027, loss = 0.194115
grad_step = 000028, loss = 0.185128
grad_step = 000029, loss = 0.175988
grad_step = 000030, loss = 0.167848
grad_step = 000031, loss = 0.161114
grad_step = 000032, loss = 0.154764
grad_step = 000033, loss = 0.147671
grad_step = 000034, loss = 0.140091
grad_step = 000035, loss = 0.133095
grad_step = 000036, loss = 0.127121
grad_step = 000037, loss = 0.121669
grad_step = 000038, loss = 0.116065
grad_step = 000039, loss = 0.110167
grad_step = 000040, loss = 0.104408
grad_step = 000041, loss = 0.099275
grad_step = 000042, loss = 0.094549
grad_step = 000043, loss = 0.089916
grad_step = 000044, loss = 0.085128
grad_step = 000045, loss = 0.080427
grad_step = 000046, loss = 0.076168
grad_step = 000047, loss = 0.072340
grad_step = 000048, loss = 0.068652
grad_step = 000049, loss = 0.064963
grad_step = 000050, loss = 0.061386
grad_step = 000051, loss = 0.058025
grad_step = 000052, loss = 0.054842
grad_step = 000053, loss = 0.051767
grad_step = 000054, loss = 0.048790
grad_step = 000055, loss = 0.045952
grad_step = 000056, loss = 0.043276
grad_step = 000057, loss = 0.040723
grad_step = 000058, loss = 0.038270
grad_step = 000059, loss = 0.035908
grad_step = 000060, loss = 0.033670
grad_step = 000061, loss = 0.031588
grad_step = 000062, loss = 0.029606
grad_step = 000063, loss = 0.027673
grad_step = 000064, loss = 0.025822
grad_step = 000065, loss = 0.024121
grad_step = 000066, loss = 0.022544
grad_step = 000067, loss = 0.021008
grad_step = 000068, loss = 0.019512
grad_step = 000069, loss = 0.018115
grad_step = 000070, loss = 0.016837
grad_step = 000071, loss = 0.015641
grad_step = 000072, loss = 0.014498
grad_step = 000073, loss = 0.013414
grad_step = 000074, loss = 0.012407
grad_step = 000075, loss = 0.011476
grad_step = 000076, loss = 0.010606
grad_step = 000077, loss = 0.009796
grad_step = 000078, loss = 0.009055
grad_step = 000079, loss = 0.008368
grad_step = 000080, loss = 0.007719
grad_step = 000081, loss = 0.007118
grad_step = 000082, loss = 0.006579
grad_step = 000083, loss = 0.006092
grad_step = 000084, loss = 0.005640
grad_step = 000085, loss = 0.005221
grad_step = 000086, loss = 0.004844
grad_step = 000087, loss = 0.004506
grad_step = 000088, loss = 0.004198
grad_step = 000089, loss = 0.003912
grad_step = 000090, loss = 0.003656
grad_step = 000091, loss = 0.003430
grad_step = 000092, loss = 0.003228
grad_step = 000093, loss = 0.003045
grad_step = 000094, loss = 0.002881
grad_step = 000095, loss = 0.002737
grad_step = 000096, loss = 0.002612
grad_step = 000097, loss = 0.002505
grad_step = 000098, loss = 0.002412
grad_step = 000099, loss = 0.002334
grad_step = 000100, loss = 0.002268
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002213
grad_step = 000102, loss = 0.002168
grad_step = 000103, loss = 0.002131
grad_step = 000104, loss = 0.002100
grad_step = 000105, loss = 0.002074
grad_step = 000106, loss = 0.002054
grad_step = 000107, loss = 0.002039
grad_step = 000108, loss = 0.002030
grad_step = 000109, loss = 0.002023
grad_step = 000110, loss = 0.002017
grad_step = 000111, loss = 0.002014
grad_step = 000112, loss = 0.002010
grad_step = 000113, loss = 0.002005
grad_step = 000114, loss = 0.002003
grad_step = 000115, loss = 0.002001
grad_step = 000116, loss = 0.002000
grad_step = 000117, loss = 0.002000
grad_step = 000118, loss = 0.002000
grad_step = 000119, loss = 0.002000
grad_step = 000120, loss = 0.001999
grad_step = 000121, loss = 0.001998
grad_step = 000122, loss = 0.001996
grad_step = 000123, loss = 0.001993
grad_step = 000124, loss = 0.001990
grad_step = 000125, loss = 0.001986
grad_step = 000126, loss = 0.001982
grad_step = 000127, loss = 0.001978
grad_step = 000128, loss = 0.001974
grad_step = 000129, loss = 0.001970
grad_step = 000130, loss = 0.001965
grad_step = 000131, loss = 0.001961
grad_step = 000132, loss = 0.001957
grad_step = 000133, loss = 0.001953
grad_step = 000134, loss = 0.001949
grad_step = 000135, loss = 0.001946
grad_step = 000136, loss = 0.001945
grad_step = 000137, loss = 0.001948
grad_step = 000138, loss = 0.001956
grad_step = 000139, loss = 0.001972
grad_step = 000140, loss = 0.001978
grad_step = 000141, loss = 0.001968
grad_step = 000142, loss = 0.001934
grad_step = 000143, loss = 0.001910
grad_step = 000144, loss = 0.001913
grad_step = 000145, loss = 0.001927
grad_step = 000146, loss = 0.001931
grad_step = 000147, loss = 0.001911
grad_step = 000148, loss = 0.001892
grad_step = 000149, loss = 0.001889
grad_step = 000150, loss = 0.001897
grad_step = 000151, loss = 0.001900
grad_step = 000152, loss = 0.001889
grad_step = 000153, loss = 0.001874
grad_step = 000154, loss = 0.001867
grad_step = 000155, loss = 0.001869
grad_step = 000156, loss = 0.001871
grad_step = 000157, loss = 0.001866
grad_step = 000158, loss = 0.001856
grad_step = 000159, loss = 0.001846
grad_step = 000160, loss = 0.001842
grad_step = 000161, loss = 0.001843
grad_step = 000162, loss = 0.001842
grad_step = 000163, loss = 0.001838
grad_step = 000164, loss = 0.001829
grad_step = 000165, loss = 0.001820
grad_step = 000166, loss = 0.001813
grad_step = 000167, loss = 0.001809
grad_step = 000168, loss = 0.001807
grad_step = 000169, loss = 0.001806
grad_step = 000170, loss = 0.001805
grad_step = 000171, loss = 0.001800
grad_step = 000172, loss = 0.001793
grad_step = 000173, loss = 0.001784
grad_step = 000174, loss = 0.001775
grad_step = 000175, loss = 0.001769
grad_step = 000176, loss = 0.001765
grad_step = 000177, loss = 0.001760
grad_step = 000178, loss = 0.001756
grad_step = 000179, loss = 0.001758
grad_step = 000180, loss = 0.001769
grad_step = 000181, loss = 0.001764
grad_step = 000182, loss = 0.001751
grad_step = 000183, loss = 0.001754
grad_step = 000184, loss = 0.001747
grad_step = 000185, loss = 0.001726
grad_step = 000186, loss = 0.001719
grad_step = 000187, loss = 0.001723
grad_step = 000188, loss = 0.001718
grad_step = 000189, loss = 0.001708
grad_step = 000190, loss = 0.001709
grad_step = 000191, loss = 0.001714
grad_step = 000192, loss = 0.001703
grad_step = 000193, loss = 0.001690
grad_step = 000194, loss = 0.001687
grad_step = 000195, loss = 0.001692
grad_step = 000196, loss = 0.001691
grad_step = 000197, loss = 0.001676
grad_step = 000198, loss = 0.001668
grad_step = 000199, loss = 0.001668
grad_step = 000200, loss = 0.001670
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001669
grad_step = 000202, loss = 0.001658
grad_step = 000203, loss = 0.001653
grad_step = 000204, loss = 0.001655
grad_step = 000205, loss = 0.001660
grad_step = 000206, loss = 0.001665
grad_step = 000207, loss = 0.001670
grad_step = 000208, loss = 0.001681
grad_step = 000209, loss = 0.001720
grad_step = 000210, loss = 0.001792
grad_step = 000211, loss = 0.001926
grad_step = 000212, loss = 0.001795
grad_step = 000213, loss = 0.001720
grad_step = 000214, loss = 0.001670
grad_step = 000215, loss = 0.001671
grad_step = 000216, loss = 0.001773
grad_step = 000217, loss = 0.001728
grad_step = 000218, loss = 0.001627
grad_step = 000219, loss = 0.001651
grad_step = 000220, loss = 0.001694
grad_step = 000221, loss = 0.001655
grad_step = 000222, loss = 0.001655
grad_step = 000223, loss = 0.001683
grad_step = 000224, loss = 0.001617
grad_step = 000225, loss = 0.001723
grad_step = 000226, loss = 0.001753
grad_step = 000227, loss = 0.001657
grad_step = 000228, loss = 0.001747
grad_step = 000229, loss = 0.001652
grad_step = 000230, loss = 0.001766
grad_step = 000231, loss = 0.001651
grad_step = 000232, loss = 0.001692
grad_step = 000233, loss = 0.001639
grad_step = 000234, loss = 0.001668
grad_step = 000235, loss = 0.001644
grad_step = 000236, loss = 0.001604
grad_step = 000237, loss = 0.001628
grad_step = 000238, loss = 0.001594
grad_step = 000239, loss = 0.001638
grad_step = 000240, loss = 0.001588
grad_step = 000241, loss = 0.001606
grad_step = 000242, loss = 0.001597
grad_step = 000243, loss = 0.001597
grad_step = 000244, loss = 0.001599
grad_step = 000245, loss = 0.001580
grad_step = 000246, loss = 0.001600
grad_step = 000247, loss = 0.001583
grad_step = 000248, loss = 0.001598
grad_step = 000249, loss = 0.001577
grad_step = 000250, loss = 0.001578
grad_step = 000251, loss = 0.001574
grad_step = 000252, loss = 0.001574
grad_step = 000253, loss = 0.001580
grad_step = 000254, loss = 0.001568
grad_step = 000255, loss = 0.001570
grad_step = 000256, loss = 0.001560
grad_step = 000257, loss = 0.001563
grad_step = 000258, loss = 0.001563
grad_step = 000259, loss = 0.001560
grad_step = 000260, loss = 0.001561
grad_step = 000261, loss = 0.001554
grad_step = 000262, loss = 0.001554
grad_step = 000263, loss = 0.001551
grad_step = 000264, loss = 0.001549
grad_step = 000265, loss = 0.001550
grad_step = 000266, loss = 0.001547
grad_step = 000267, loss = 0.001548
grad_step = 000268, loss = 0.001546
grad_step = 000269, loss = 0.001544
grad_step = 000270, loss = 0.001543
grad_step = 000271, loss = 0.001539
grad_step = 000272, loss = 0.001539
grad_step = 000273, loss = 0.001537
grad_step = 000274, loss = 0.001535
grad_step = 000275, loss = 0.001534
grad_step = 000276, loss = 0.001532
grad_step = 000277, loss = 0.001532
grad_step = 000278, loss = 0.001530
grad_step = 000279, loss = 0.001528
grad_step = 000280, loss = 0.001527
grad_step = 000281, loss = 0.001525
grad_step = 000282, loss = 0.001525
grad_step = 000283, loss = 0.001523
grad_step = 000284, loss = 0.001522
grad_step = 000285, loss = 0.001521
grad_step = 000286, loss = 0.001521
grad_step = 000287, loss = 0.001522
grad_step = 000288, loss = 0.001528
grad_step = 000289, loss = 0.001540
grad_step = 000290, loss = 0.001575
grad_step = 000291, loss = 0.001635
grad_step = 000292, loss = 0.001765
grad_step = 000293, loss = 0.001831
grad_step = 000294, loss = 0.001879
grad_step = 000295, loss = 0.001668
grad_step = 000296, loss = 0.001518
grad_step = 000297, loss = 0.001554
grad_step = 000298, loss = 0.001665
grad_step = 000299, loss = 0.001685
grad_step = 000300, loss = 0.001543
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001510
grad_step = 000302, loss = 0.001599
grad_step = 000303, loss = 0.001612
grad_step = 000304, loss = 0.001538
grad_step = 000305, loss = 0.001497
grad_step = 000306, loss = 0.001548
grad_step = 000307, loss = 0.001581
grad_step = 000308, loss = 0.001518
grad_step = 000309, loss = 0.001493
grad_step = 000310, loss = 0.001531
grad_step = 000311, loss = 0.001538
grad_step = 000312, loss = 0.001503
grad_step = 000313, loss = 0.001487
grad_step = 000314, loss = 0.001510
grad_step = 000315, loss = 0.001518
grad_step = 000316, loss = 0.001493
grad_step = 000317, loss = 0.001482
grad_step = 000318, loss = 0.001498
grad_step = 000319, loss = 0.001502
grad_step = 000320, loss = 0.001485
grad_step = 000321, loss = 0.001476
grad_step = 000322, loss = 0.001485
grad_step = 000323, loss = 0.001491
grad_step = 000324, loss = 0.001479
grad_step = 000325, loss = 0.001470
grad_step = 000326, loss = 0.001474
grad_step = 000327, loss = 0.001479
grad_step = 000328, loss = 0.001475
grad_step = 000329, loss = 0.001467
grad_step = 000330, loss = 0.001464
grad_step = 000331, loss = 0.001467
grad_step = 000332, loss = 0.001468
grad_step = 000333, loss = 0.001464
grad_step = 000334, loss = 0.001458
grad_step = 000335, loss = 0.001457
grad_step = 000336, loss = 0.001459
grad_step = 000337, loss = 0.001459
grad_step = 000338, loss = 0.001456
grad_step = 000339, loss = 0.001452
grad_step = 000340, loss = 0.001449
grad_step = 000341, loss = 0.001449
grad_step = 000342, loss = 0.001449
grad_step = 000343, loss = 0.001448
grad_step = 000344, loss = 0.001446
grad_step = 000345, loss = 0.001443
grad_step = 000346, loss = 0.001441
grad_step = 000347, loss = 0.001439
grad_step = 000348, loss = 0.001438
grad_step = 000349, loss = 0.001437
grad_step = 000350, loss = 0.001436
grad_step = 000351, loss = 0.001434
grad_step = 000352, loss = 0.001433
grad_step = 000353, loss = 0.001431
grad_step = 000354, loss = 0.001429
grad_step = 000355, loss = 0.001427
grad_step = 000356, loss = 0.001426
grad_step = 000357, loss = 0.001425
grad_step = 000358, loss = 0.001424
grad_step = 000359, loss = 0.001423
grad_step = 000360, loss = 0.001422
grad_step = 000361, loss = 0.001422
grad_step = 000362, loss = 0.001423
grad_step = 000363, loss = 0.001424
grad_step = 000364, loss = 0.001425
grad_step = 000365, loss = 0.001428
grad_step = 000366, loss = 0.001430
grad_step = 000367, loss = 0.001434
grad_step = 000368, loss = 0.001436
grad_step = 000369, loss = 0.001442
grad_step = 000370, loss = 0.001444
grad_step = 000371, loss = 0.001453
grad_step = 000372, loss = 0.001455
grad_step = 000373, loss = 0.001465
grad_step = 000374, loss = 0.001465
grad_step = 000375, loss = 0.001470
grad_step = 000376, loss = 0.001458
grad_step = 000377, loss = 0.001451
grad_step = 000378, loss = 0.001431
grad_step = 000379, loss = 0.001416
grad_step = 000380, loss = 0.001400
grad_step = 000381, loss = 0.001392
grad_step = 000382, loss = 0.001389
grad_step = 000383, loss = 0.001393
grad_step = 000384, loss = 0.001402
grad_step = 000385, loss = 0.001417
grad_step = 000386, loss = 0.001441
grad_step = 000387, loss = 0.001461
grad_step = 000388, loss = 0.001498
grad_step = 000389, loss = 0.001507
grad_step = 000390, loss = 0.001525
grad_step = 000391, loss = 0.001484
grad_step = 000392, loss = 0.001444
grad_step = 000393, loss = 0.001393
grad_step = 000394, loss = 0.001374
grad_step = 000395, loss = 0.001388
grad_step = 000396, loss = 0.001416
grad_step = 000397, loss = 0.001441
grad_step = 000398, loss = 0.001437
grad_step = 000399, loss = 0.001422
grad_step = 000400, loss = 0.001388
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001368
grad_step = 000402, loss = 0.001364
grad_step = 000403, loss = 0.001372
grad_step = 000404, loss = 0.001387
grad_step = 000405, loss = 0.001392
grad_step = 000406, loss = 0.001391
grad_step = 000407, loss = 0.001377
grad_step = 000408, loss = 0.001365
grad_step = 000409, loss = 0.001356
grad_step = 000410, loss = 0.001353
grad_step = 000411, loss = 0.001355
grad_step = 000412, loss = 0.001362
grad_step = 000413, loss = 0.001371
grad_step = 000414, loss = 0.001374
grad_step = 000415, loss = 0.001379
grad_step = 000416, loss = 0.001375
grad_step = 000417, loss = 0.001369
grad_step = 000418, loss = 0.001358
grad_step = 000419, loss = 0.001350
grad_step = 000420, loss = 0.001342
grad_step = 000421, loss = 0.001339
grad_step = 000422, loss = 0.001339
grad_step = 000423, loss = 0.001343
grad_step = 000424, loss = 0.001350
grad_step = 000425, loss = 0.001360
grad_step = 000426, loss = 0.001378
grad_step = 000427, loss = 0.001391
grad_step = 000428, loss = 0.001414
grad_step = 000429, loss = 0.001418
grad_step = 000430, loss = 0.001431
grad_step = 000431, loss = 0.001411
grad_step = 000432, loss = 0.001389
grad_step = 000433, loss = 0.001354
grad_step = 000434, loss = 0.001331
grad_step = 000435, loss = 0.001324
grad_step = 000436, loss = 0.001332
grad_step = 000437, loss = 0.001351
grad_step = 000438, loss = 0.001370
grad_step = 000439, loss = 0.001394
grad_step = 000440, loss = 0.001390
grad_step = 000441, loss = 0.001386
grad_step = 000442, loss = 0.001357
grad_step = 000443, loss = 0.001331
grad_step = 000444, loss = 0.001317
grad_step = 000445, loss = 0.001320
grad_step = 000446, loss = 0.001336
grad_step = 000447, loss = 0.001349
grad_step = 000448, loss = 0.001359
grad_step = 000449, loss = 0.001359
grad_step = 000450, loss = 0.001362
grad_step = 000451, loss = 0.001338
grad_step = 000452, loss = 0.001321
grad_step = 000453, loss = 0.001312
grad_step = 000454, loss = 0.001308
grad_step = 000455, loss = 0.001307
grad_step = 000456, loss = 0.001311
grad_step = 000457, loss = 0.001317
grad_step = 000458, loss = 0.001316
grad_step = 000459, loss = 0.001313
grad_step = 000460, loss = 0.001313
grad_step = 000461, loss = 0.001314
grad_step = 000462, loss = 0.001307
grad_step = 000463, loss = 0.001302
grad_step = 000464, loss = 0.001301
grad_step = 000465, loss = 0.001299
grad_step = 000466, loss = 0.001295
grad_step = 000467, loss = 0.001291
grad_step = 000468, loss = 0.001289
grad_step = 000469, loss = 0.001289
grad_step = 000470, loss = 0.001287
grad_step = 000471, loss = 0.001285
grad_step = 000472, loss = 0.001284
grad_step = 000473, loss = 0.001284
grad_step = 000474, loss = 0.001282
grad_step = 000475, loss = 0.001280
grad_step = 000476, loss = 0.001280
grad_step = 000477, loss = 0.001279
grad_step = 000478, loss = 0.001277
grad_step = 000479, loss = 0.001276
grad_step = 000480, loss = 0.001275
grad_step = 000481, loss = 0.001276
grad_step = 000482, loss = 0.001277
grad_step = 000483, loss = 0.001282
grad_step = 000484, loss = 0.001291
grad_step = 000485, loss = 0.001313
grad_step = 000486, loss = 0.001336
grad_step = 000487, loss = 0.001393
grad_step = 000488, loss = 0.001435
grad_step = 000489, loss = 0.001516
grad_step = 000490, loss = 0.001501
grad_step = 000491, loss = 0.001461
grad_step = 000492, loss = 0.001361
grad_step = 000493, loss = 0.001283
grad_step = 000494, loss = 0.001266
grad_step = 000495, loss = 0.001293
grad_step = 000496, loss = 0.001331
grad_step = 000497, loss = 0.001350
grad_step = 000498, loss = 0.001370
grad_step = 000499, loss = 0.001311
grad_step = 000500, loss = 0.001270
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001255
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

  date_run                              2020-05-15 19:13:20.415151
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.256813
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 19:13:20.421734
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.162503
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 19:13:20.429477
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.149333
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 19:13:20.435040
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -1.4693
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
0   2020-05-15 19:12:49.037072  ...    mean_absolute_error
1   2020-05-15 19:12:49.042056  ...     mean_squared_error
2   2020-05-15 19:12:49.046232  ...  median_absolute_error
3   2020-05-15 19:12:49.049772  ...               r2_score
4   2020-05-15 19:12:58.426161  ...    mean_absolute_error
5   2020-05-15 19:12:58.430266  ...     mean_squared_error
6   2020-05-15 19:12:58.433931  ...  median_absolute_error
7   2020-05-15 19:12:58.437525  ...               r2_score
8   2020-05-15 19:13:20.415151  ...    mean_absolute_error
9   2020-05-15 19:13:20.421734  ...     mean_squared_error
10  2020-05-15 19:13:20.429477  ...  median_absolute_error
11  2020-05-15 19:13:20.435040  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb3c1f2afd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 303041.80it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 392643.85it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 543389.34it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 767173.22it/s] 77%|███████▋  | 7602176/9912422 [00:00<00:02, 1085503.62it/s]9920512it [00:00, 10354029.00it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 139909.47it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 310178.26it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 402694.15it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 557450.04it/s]1654784it [00:00, 2803172.83it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 49911.45it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb37492be48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb373f5c0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb37492be48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb373eb4080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb37492be48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb3716d9ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb37492be48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb373e726a0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb37492be48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb373f5c048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc62d94b1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=64a468d499c3bc5d9a721cba7eaec87d1bb55286fee16c4fffce009590ab329a
  Stored in directory: /tmp/pip-ephem-wheel-cache-zkjmk83z/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc623cd1048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 46s
   57344/17464789 [..............................] - ETA: 39s
   90112/17464789 [..............................] - ETA: 37s
  180224/17464789 [..............................] - ETA: 25s
  385024/17464789 [..............................] - ETA: 14s
  802816/17464789 [>.............................] - ETA: 8s 
 1622016/17464789 [=>............................] - ETA: 4s
 3260416/17464789 [====>.........................] - ETA: 2s
 6258688/17464789 [=========>....................] - ETA: 1s
 9142272/17464789 [==============>...............] - ETA: 0s
12140544/17464789 [===================>..........] - ETA: 0s
15220736/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 19:14:51.792466: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 19:14:51.796395: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-15 19:14:51.796669: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b87a207670 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 19:14:51.796689: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.2680 - accuracy: 0.5260
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5133 - accuracy: 0.5100 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.4571 - accuracy: 0.5137
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5785 - accuracy: 0.5058
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.4949 - accuracy: 0.5112
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5414 - accuracy: 0.5082
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5352 - accuracy: 0.5086
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.4941 - accuracy: 0.5113
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5218 - accuracy: 0.5094
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5302 - accuracy: 0.5089
11000/25000 [============>.................] - ETA: 4s - loss: 7.5746 - accuracy: 0.5060
12000/25000 [=============>................] - ETA: 3s - loss: 7.5567 - accuracy: 0.5072
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5581 - accuracy: 0.5071
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5779 - accuracy: 0.5058
15000/25000 [=================>............] - ETA: 2s - loss: 7.5910 - accuracy: 0.5049
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5794 - accuracy: 0.5057
17000/25000 [===================>..........] - ETA: 2s - loss: 7.5936 - accuracy: 0.5048
18000/25000 [====================>.........] - ETA: 2s - loss: 7.5840 - accuracy: 0.5054
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6013 - accuracy: 0.5043
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6022 - accuracy: 0.5042
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6287 - accuracy: 0.5025
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6562 - accuracy: 0.5007
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6506 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 9s 347us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 19:15:07.427617
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 19:15:07.427617  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<11:54:10, 20.1kB/s].vector_cache/glove.6B.zip:   0%|          | 434k/862M [00:00<8:20:51, 28.7kB/s]  .vector_cache/glove.6B.zip:   1%|          | 6.35M/862M [00:00<5:48:16, 41.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.2M/862M [00:00<4:01:36, 58.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.1M/862M [00:00<2:47:48, 83.5kB/s].vector_cache/glove.6B.zip:   4%|▎         | 30.3M/862M [00:00<1:56:13, 119kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 41.1M/862M [00:01<1:20:20, 170kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.3M/862M [00:01<55:34, 243kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:01<40:54, 330kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:01<28:51, 466kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<11:03:13, 20.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<7:43:14, 29.0kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:05<5:27:06, 40.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:05<3:49:26, 58.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:05<2:41:05, 82.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<7:39:18, 29.0kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:08<5:21:46, 41.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.2M/862M [00:08<3:46:11, 58.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:10<2:39:35, 82.7kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.6M/862M [00:10<1:52:17, 117kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:12<1:20:23, 163kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.8M/862M [00:12<56:52, 231kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:12<40:25, 324kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:13<6:04:16, 36.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:15<4:15:31, 51.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:15<2:59:47, 72.4kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:17<2:07:13, 102kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:17<1:29:37, 144kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:19<1:04:32, 200kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:19<45:46, 281kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:19<32:45, 392kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:20<5:33:12, 38.6kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<3:53:49, 54.7kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:22<2:44:35, 77.6kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:24<1:56:35, 109kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:24<1:22:11, 155kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<59:18, 213kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<42:08, 300kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<31:23, 401kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<22:34, 557kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<16:31, 759kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<5:21:58, 38.9kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<3:45:55, 55.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<2:39:24, 78.2kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<1:52:49, 110kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:33<1:19:30, 156kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<57:24, 215kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<40:46, 302kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:35<29:07, 422kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<5:36:15, 36.6kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:36<3:54:17, 52.2kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<3:01:38, 67.3kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<2:08:04, 95.4kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<1:31:01, 134kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 134M/862M [00:40<1:04:15, 189kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<46:44, 259kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<33:14, 363kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<25:05, 479kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<18:09, 661kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<13:18, 900kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<5:24:19, 36.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:45<3:45:50, 52.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<4:19:22, 45.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<3:01:59, 65.4kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<2:08:41, 92.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<1:30:35, 131kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<1:05:00, 181kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:51<46:04, 255kB/s]  .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<33:58, 344kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:53<24:20, 480kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<17:35, 663kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<5:22:12, 36.2kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<3:45:52, 51.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<2:38:56, 72.9kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<1:52:24, 103kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<1:19:16, 145kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<56:59, 201kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<40:25, 283kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<29:59, 380kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<21:38, 526kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<15:40, 724kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<5:11:23, 36.4kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:03<3:36:50, 52.0kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<2:50:56, 66.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:05<2:00:28, 93.6kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<1:25:34, 131kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<1:00:23, 185kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<43:52, 254kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<31:15, 356kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<23:32, 470kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<17:01, 650kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<12:28, 885kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<4:58:12, 37.0kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:12<3:27:33, 52.8kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<3:33:13, 51.4kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:14<2:30:00, 73.1kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<1:46:02, 103kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<1:14:41, 146kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<53:45, 201kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<38:10, 283kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<28:18, 380kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<20:19, 529kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:20<14:49, 723kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<4:37:33, 38.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<3:14:34, 54.7kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<2:17:00, 77.7kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<1:36:56, 109kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<1:08:22, 155kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<49:14, 214kB/s]  .vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<34:57, 300kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<26:01, 401kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<18:45, 556kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<13:36, 764kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<4:46:22, 36.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<3:20:37, 51.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<2:21:10, 73.2kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<1:39:46, 103kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<1:10:16, 146kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<50:34, 202kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<35:52, 284kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:38<26:36, 381kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<19:09, 528kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:38<13:52, 727kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<4:37:33, 36.4kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:39<3:13:11, 51.9kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<2:27:47, 67.8kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<1:44:10, 96.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<1:13:58, 135kB/s] .vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<52:13, 190kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<37:56, 260kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<27:01, 365kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<20:24, 481kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<14:50, 661kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:47<10:55, 895kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<4:08:55, 39.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<2:54:25, 55.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<2:02:46, 79.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<1:26:51, 111kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<1:01:12, 157kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<44:08, 217kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<31:18, 305kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<23:18, 408kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<16:44, 567kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<12:11, 776kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<4:19:08, 36.5kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<3:01:25, 51.8kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<2:07:40, 73.5kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<1:30:11, 103kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<1:03:30, 147kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<45:41, 203kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<32:24, 285kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<24:01, 382kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<17:16, 531kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<12:31, 730kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<4:10:18, 36.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:06<2:54:06, 52.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<11:13:01, 13.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<7:50:53, 19.3kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<5:29:03, 27.4kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<3:50:31, 39.0kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<2:41:45, 55.3kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<1:53:31, 78.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<1:20:22, 110kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<56:37, 156kB/s]  .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<40:48, 216kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<28:57, 304kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:16<20:40, 424kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<4:02:55, 36.1kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:19<2:49:57, 51.2kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<1:59:33, 72.7kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<1:24:24, 102kB/s] .vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<59:26, 145kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<42:43, 200kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<30:18, 282kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<22:26, 378kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<16:07, 526kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<11:40, 723kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<3:52:37, 36.3kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<2:42:43, 51.5kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<1:54:41, 73.0kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<1:20:53, 103kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<56:58, 146kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<40:55, 201kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<29:04, 283kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<21:28, 380kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<15:31, 526kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<11:13, 724kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<3:41:56, 36.6kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<2:35:12, 52.0kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<1:49:11, 73.8kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<1:17:03, 104kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<54:13, 147kB/s]  .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<38:59, 203kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<27:38, 286kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<20:29, 383kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<14:40, 535kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:43<10:38, 735kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<3:34:18, 36.5kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<2:29:48, 51.7kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<1:45:22, 73.5kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<1:14:20, 103kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<52:20, 147kB/s]  .vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<37:36, 202kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<26:39, 285kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<19:44, 382kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<14:08, 532kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<10:16, 731kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<3:25:28, 36.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:53<2:22:36, 52.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<2:16:00, 54.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<1:35:43, 77.6kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<1:07:33, 109kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<47:36, 155kB/s]  .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<34:14, 213kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<24:14, 301kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<18:03, 401kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<12:59, 556kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:01<09:25, 763kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<3:17:50, 36.3kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<2:18:11, 51.5kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<1:37:22, 73.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<1:08:35, 103kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<48:11, 146kB/s]  .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<34:37, 202kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<24:29, 284kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<18:10, 380kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<13:02, 529kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:10<09:27, 727kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<3:08:09, 36.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<2:11:22, 51.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<1:32:20, 73.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:13<1:04:10, 105kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<55:47, 121kB/s]  .vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<39:17, 171kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<28:20, 235kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<20:06, 331kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<15:01, 439kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<10:48, 610kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:19<07:53, 831kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<2:56:51, 37.1kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:20<2:02:38, 53.0kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<1:40:33, 64.6kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<1:10:51, 91.5kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<50:05, 128kB/s]   .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<35:18, 182kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<25:32, 249kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<18:09, 349kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<13:36, 462kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<09:49, 638kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:28<07:10, 870kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<2:49:00, 37.0kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:29<1:57:10, 52.8kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<1:30:43, 68.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<1:03:56, 96.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:33<45:13, 135kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<31:50, 191kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<23:06, 261kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<16:24, 367kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<12:20, 484kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<08:52, 671kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<06:34, 902kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<2:31:35, 39.1kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<1:45:43, 55.5kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<1:14:24, 78.7kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<52:24, 111kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<37:03, 156kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<26:31, 216kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<18:49, 304kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<13:59, 404kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<10:33, 535kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:46<07:21, 760kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<14:55, 375kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<10:42, 521kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<08:18, 664kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<06:05, 905kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<05:05, 1.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<03:49, 1.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:52<02:58, 1.82MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<2:31:39, 35.7kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<1:45:34, 50.6kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<1:14:13, 71.8kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<52:09, 101kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<36:44, 143kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<26:14, 198kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<18:36, 279kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<13:41, 375kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<09:54, 517kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<07:37, 664kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<05:34, 905kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<04:40, 1.07MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<03:30, 1.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<03:13, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<02:26, 2.01MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<01:59, 2.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<2:17:06, 35.6kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<1:35:19, 50.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<1:06:57, 71.8kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:10<46:22, 102kB/s]   .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<38:41, 123kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<27:14, 174kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<19:35, 239kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<14:00, 334kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<09:46, 474kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<08:42, 529kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<06:26, 715kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<04:32, 1.00MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<04:27, 1.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<03:27, 1.31MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:18<02:25, 1.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<07:52, 568kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<05:44, 776kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:20<04:01, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<06:15, 704kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<04:35, 956kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:22<03:24, 1.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<2:09:21, 33.7kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:23<1:29:17, 48.1kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<1:08:22, 62.7kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<48:09, 88.9kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<33:51, 125kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:27<23:46, 177kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<17:07, 242kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<12:08, 341kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<09:02, 452kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<06:28, 628kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<05:09, 779kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<03:47, 1.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<03:15, 1.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<02:28, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<02:20, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:49, 2.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:37<01:27, 2.64MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<1:53:22, 33.8kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<1:18:28, 48.0kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<55:07, 68.2kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<38:31, 96.0kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<27:01, 136kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<19:14, 189kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<13:36, 266kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<09:56, 358kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<07:07, 498kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<05:27, 639kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<03:57, 879kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<03:18, 1.03MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<02:29, 1.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<02:15, 1.49MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<01:42, 1.95MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:52<01:21, 2.45MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<1:39:22, 33.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<1:08:34, 47.3kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:55<48:00, 67.3kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<33:31, 94.7kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<23:30, 134kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<16:39, 186kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [04:59<11:44, 263kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<08:35, 353kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<06:08, 493kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<04:41, 632kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<03:25, 862kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<02:49, 1.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<02:06, 1.37MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:54, 1.48MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:26, 1.96MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<01:08, 2.44MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<1:22:33, 33.7kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:08<56:26, 48.2kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<51:03, 53.2kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<35:52, 75.6kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:10<24:39, 108kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<19:28, 136kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<13:42, 192kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<09:48, 263kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<06:55, 370kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<05:09, 486kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<03:42, 674kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<02:55, 833kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:18<02:09, 1.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:20<01:51, 1.27MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<01:26, 1.65MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:20, 1.72MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<01:03, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<00:50, 2.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<1:07:15, 33.6kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:23<46:12, 48.0kB/s]  .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<32:47, 66.9kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<23:12, 94.3kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:25<15:56, 135kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<11:56, 178kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<08:31, 248kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<06:05, 337kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<04:21, 469kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<03:16, 607kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:31<02:23, 826kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<01:55, 997kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<01:25, 1.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:15, 1.46MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:58, 1.88MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:56, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:44, 2.40MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:46, 2.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:37, 2.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:39<00:31, 3.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<47:35, 35.0kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<32:10, 49.7kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<22:32, 70.6kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<15:24, 99.3kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<10:48, 141kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<07:29, 195kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<05:16, 274kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<03:46, 369kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<02:41, 513kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<02:01, 655kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<01:27, 903kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<01:10, 1.06MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:55, 1.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:47, 1.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:38, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:54<00:26, 2.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<01:35, 700kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<01:09, 949kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:56<00:50, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<33:21, 32.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:57<21:56, 45.9kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<18:09, 55.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<12:44, 78.5kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [05:59<08:21, 112kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<1:06:25, 14.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<45:57, 20.1kB/s]  .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<30:19, 28.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:03<20:59, 40.8kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<13:49, 57.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<09:32, 82.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<06:20, 115kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<04:26, 163kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<02:56, 224kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<02:04, 314kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<01:24, 420kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<01:01, 571kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:43, 727kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<00:31, 988kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:21, 1.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<14:54, 32.1kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<09:00, 45.6kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<06:13, 64.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<03:44, 91.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<02:32, 130kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<01:31, 180kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<01:02, 254kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:20<00:36, 361kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:30, 407kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:22<00:20, 565kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:11, 714kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:07, 978kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:03, 1.13MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:02, 1.49MB/s].vector_cache/glove.6B.zip: 862MB [06:26, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 758/400000 [00:00<00:52, 7570.41it/s]  0%|          | 1561/400000 [00:00<00:51, 7700.46it/s]  1%|          | 2334/400000 [00:00<00:51, 7707.59it/s]  1%|          | 3132/400000 [00:00<00:50, 7786.70it/s]  1%|          | 3928/400000 [00:00<00:50, 7835.31it/s]  1%|          | 4745/400000 [00:00<00:49, 7932.25it/s]  1%|▏         | 5550/400000 [00:00<00:49, 7964.73it/s]  2%|▏         | 6358/400000 [00:00<00:49, 7997.69it/s]  2%|▏         | 7160/400000 [00:00<00:49, 7984.34it/s]  2%|▏         | 7929/400000 [00:01<00:49, 7861.39it/s]  2%|▏         | 8695/400000 [00:01<00:50, 7772.99it/s]  2%|▏         | 9486/400000 [00:01<00:49, 7812.28it/s]  3%|▎         | 10258/400000 [00:01<00:50, 7663.29it/s]  3%|▎         | 11031/400000 [00:01<00:50, 7681.90it/s]  3%|▎         | 11795/400000 [00:01<00:51, 7579.88it/s]  3%|▎         | 12580/400000 [00:01<00:50, 7656.49it/s]  3%|▎         | 13382/400000 [00:01<00:49, 7761.92it/s]  4%|▎         | 14189/400000 [00:01<00:49, 7850.66it/s]  4%|▎         | 14974/400000 [00:01<00:50, 7653.24it/s]  4%|▍         | 15741/400000 [00:02<00:50, 7601.61it/s]  4%|▍         | 16514/400000 [00:02<00:50, 7637.60it/s]  4%|▍         | 17301/400000 [00:02<00:49, 7705.41it/s]  5%|▍         | 18122/400000 [00:02<00:48, 7849.32it/s]  5%|▍         | 18931/400000 [00:02<00:48, 7919.48it/s]  5%|▍         | 19724/400000 [00:02<00:48, 7851.38it/s]  5%|▌         | 20536/400000 [00:02<00:47, 7928.83it/s]  5%|▌         | 21368/400000 [00:02<00:47, 8041.43it/s]  6%|▌         | 22199/400000 [00:02<00:46, 8118.07it/s]  6%|▌         | 23042/400000 [00:02<00:45, 8208.31it/s]  6%|▌         | 23864/400000 [00:03<00:47, 8002.54it/s]  6%|▌         | 24666/400000 [00:03<00:47, 7852.79it/s]  6%|▋         | 25454/400000 [00:03<00:48, 7711.39it/s]  7%|▋         | 26230/400000 [00:03<00:48, 7725.81it/s]  7%|▋         | 27031/400000 [00:03<00:47, 7807.54it/s]  7%|▋         | 27813/400000 [00:03<00:48, 7699.65it/s]  7%|▋         | 28629/400000 [00:03<00:47, 7829.02it/s]  7%|▋         | 29461/400000 [00:03<00:46, 7968.40it/s]  8%|▊         | 30284/400000 [00:03<00:45, 8043.79it/s]  8%|▊         | 31113/400000 [00:03<00:45, 8113.92it/s]  8%|▊         | 31926/400000 [00:04<00:46, 7988.20it/s]  8%|▊         | 32726/400000 [00:04<00:46, 7929.54it/s]  8%|▊         | 33556/400000 [00:04<00:45, 8036.05it/s]  9%|▊         | 34395/400000 [00:04<00:44, 8136.87it/s]  9%|▉         | 35216/400000 [00:04<00:44, 8157.94it/s]  9%|▉         | 36033/400000 [00:04<00:45, 8068.61it/s]  9%|▉         | 36841/400000 [00:04<00:45, 8025.98it/s]  9%|▉         | 37645/400000 [00:04<00:45, 7949.83it/s] 10%|▉         | 38441/400000 [00:04<00:45, 7899.08it/s] 10%|▉         | 39232/400000 [00:04<00:46, 7834.00it/s] 10%|█         | 40016/400000 [00:05<00:46, 7746.45it/s] 10%|█         | 40829/400000 [00:05<00:45, 7857.41it/s] 10%|█         | 41631/400000 [00:05<00:45, 7904.95it/s] 11%|█         | 42423/400000 [00:05<00:45, 7862.30it/s] 11%|█         | 43220/400000 [00:05<00:45, 7892.94it/s] 11%|█         | 44012/400000 [00:05<00:45, 7898.72it/s] 11%|█         | 44808/400000 [00:05<00:44, 7916.31it/s] 11%|█▏        | 45600/400000 [00:05<00:45, 7818.10it/s] 12%|█▏        | 46383/400000 [00:05<00:46, 7557.82it/s] 12%|█▏        | 47162/400000 [00:05<00:46, 7624.89it/s] 12%|█▏        | 47927/400000 [00:06<00:46, 7628.19it/s] 12%|█▏        | 48695/400000 [00:06<00:45, 7642.25it/s] 12%|█▏        | 49474/400000 [00:06<00:45, 7685.14it/s] 13%|█▎        | 50260/400000 [00:06<00:45, 7735.60it/s] 13%|█▎        | 51065/400000 [00:06<00:44, 7825.00it/s] 13%|█▎        | 51849/400000 [00:06<00:44, 7767.02it/s] 13%|█▎        | 52627/400000 [00:06<00:44, 7738.52it/s] 13%|█▎        | 53457/400000 [00:06<00:43, 7898.23it/s] 14%|█▎        | 54275/400000 [00:06<00:43, 7979.57it/s] 14%|█▍        | 55091/400000 [00:07<00:42, 8031.63it/s] 14%|█▍        | 55895/400000 [00:07<00:43, 7994.33it/s] 14%|█▍        | 56715/400000 [00:07<00:42, 8052.70it/s] 14%|█▍        | 57528/400000 [00:07<00:42, 8073.63it/s] 15%|█▍        | 58364/400000 [00:07<00:41, 8155.18it/s] 15%|█▍        | 59214/400000 [00:07<00:41, 8254.55it/s] 15%|█▌        | 60041/400000 [00:07<00:41, 8135.75it/s] 15%|█▌        | 60856/400000 [00:07<00:42, 7914.51it/s] 15%|█▌        | 61650/400000 [00:07<00:43, 7804.97it/s] 16%|█▌        | 62433/400000 [00:07<00:43, 7768.14it/s] 16%|█▌        | 63212/400000 [00:08<00:43, 7704.14it/s] 16%|█▌        | 63984/400000 [00:08<00:44, 7516.72it/s] 16%|█▌        | 64738/400000 [00:08<00:44, 7514.43it/s] 16%|█▋        | 65491/400000 [00:08<00:44, 7517.37it/s] 17%|█▋        | 66244/400000 [00:08<00:44, 7457.02it/s] 17%|█▋        | 67013/400000 [00:08<00:44, 7525.24it/s] 17%|█▋        | 67779/400000 [00:08<00:43, 7563.96it/s] 17%|█▋        | 68609/400000 [00:08<00:42, 7769.07it/s] 17%|█▋        | 69451/400000 [00:08<00:41, 7951.61it/s] 18%|█▊        | 70286/400000 [00:08<00:40, 8065.28it/s] 18%|█▊        | 71104/400000 [00:09<00:40, 8097.20it/s] 18%|█▊        | 71916/400000 [00:09<00:40, 8033.93it/s] 18%|█▊        | 72744/400000 [00:09<00:40, 8105.79it/s] 18%|█▊        | 73579/400000 [00:09<00:39, 8176.50it/s] 19%|█▊        | 74418/400000 [00:09<00:39, 8236.75it/s] 19%|█▉        | 75252/400000 [00:09<00:39, 8267.05it/s] 19%|█▉        | 76080/400000 [00:09<00:39, 8120.21it/s] 19%|█▉        | 76907/400000 [00:09<00:39, 8163.59it/s] 19%|█▉        | 77725/400000 [00:09<00:39, 8111.19it/s] 20%|█▉        | 78537/400000 [00:09<00:40, 7974.73it/s] 20%|█▉        | 79336/400000 [00:10<00:40, 7936.32it/s] 20%|██        | 80154/400000 [00:10<00:39, 8006.98it/s] 20%|██        | 80956/400000 [00:10<00:40, 7958.17it/s] 20%|██        | 81770/400000 [00:10<00:39, 8010.60it/s] 21%|██        | 82572/400000 [00:10<00:39, 8001.08it/s] 21%|██        | 83417/400000 [00:10<00:38, 8128.50it/s] 21%|██        | 84231/400000 [00:10<00:38, 8101.56it/s] 21%|██▏       | 85079/400000 [00:10<00:38, 8211.47it/s] 21%|██▏       | 85901/400000 [00:10<00:38, 8213.79it/s] 22%|██▏       | 86735/400000 [00:10<00:37, 8248.93it/s] 22%|██▏       | 87561/400000 [00:11<00:38, 8205.20it/s] 22%|██▏       | 88382/400000 [00:11<00:39, 7926.96it/s] 22%|██▏       | 89182/400000 [00:11<00:39, 7946.53it/s] 23%|██▎       | 90010/400000 [00:11<00:38, 8043.67it/s] 23%|██▎       | 90818/400000 [00:11<00:38, 8053.29it/s] 23%|██▎       | 91625/400000 [00:11<00:38, 8001.88it/s] 23%|██▎       | 92426/400000 [00:11<00:39, 7842.51it/s] 23%|██▎       | 93212/400000 [00:11<00:39, 7788.10it/s] 24%|██▎       | 94026/400000 [00:11<00:38, 7888.42it/s] 24%|██▎       | 94826/400000 [00:11<00:38, 7918.98it/s] 24%|██▍       | 95637/400000 [00:12<00:38, 7974.59it/s] 24%|██▍       | 96436/400000 [00:12<00:38, 7872.13it/s] 24%|██▍       | 97239/400000 [00:12<00:38, 7917.06it/s] 25%|██▍       | 98044/400000 [00:12<00:37, 7955.87it/s] 25%|██▍       | 98854/400000 [00:12<00:37, 7998.18it/s] 25%|██▍       | 99656/400000 [00:12<00:37, 8003.70it/s] 25%|██▌       | 100457/400000 [00:12<00:37, 7904.06it/s] 25%|██▌       | 101248/400000 [00:12<00:38, 7848.47it/s] 26%|██▌       | 102062/400000 [00:12<00:37, 7932.56it/s] 26%|██▌       | 102878/400000 [00:13<00:37, 7997.41it/s] 26%|██▌       | 103679/400000 [00:13<00:37, 7946.88it/s] 26%|██▌       | 104475/400000 [00:13<00:37, 7847.00it/s] 26%|██▋       | 105261/400000 [00:13<00:37, 7836.02it/s] 27%|██▋       | 106062/400000 [00:13<00:37, 7885.31it/s] 27%|██▋       | 106866/400000 [00:13<00:36, 7930.17it/s] 27%|██▋       | 107673/400000 [00:13<00:36, 7970.30it/s] 27%|██▋       | 108471/400000 [00:13<00:36, 7949.46it/s] 27%|██▋       | 109280/400000 [00:13<00:36, 7987.85it/s] 28%|██▊       | 110079/400000 [00:13<00:36, 7943.34it/s] 28%|██▊       | 110885/400000 [00:14<00:36, 7975.69it/s] 28%|██▊       | 111684/400000 [00:14<00:36, 7979.79it/s] 28%|██▊       | 112483/400000 [00:14<00:36, 7780.90it/s] 28%|██▊       | 113263/400000 [00:14<00:37, 7595.17it/s] 29%|██▊       | 114025/400000 [00:14<00:37, 7547.56it/s] 29%|██▊       | 114816/400000 [00:14<00:37, 7650.13it/s] 29%|██▉       | 115608/400000 [00:14<00:36, 7727.25it/s] 29%|██▉       | 116398/400000 [00:14<00:36, 7775.65it/s] 29%|██▉       | 117239/400000 [00:14<00:35, 7953.07it/s] 30%|██▉       | 118036/400000 [00:14<00:35, 7875.39it/s] 30%|██▉       | 118825/400000 [00:15<00:35, 7833.65it/s] 30%|██▉       | 119610/400000 [00:15<00:36, 7754.10it/s] 30%|███       | 120387/400000 [00:15<00:36, 7697.65it/s] 30%|███       | 121174/400000 [00:15<00:35, 7747.57it/s] 30%|███       | 121989/400000 [00:15<00:35, 7863.56it/s] 31%|███       | 122796/400000 [00:15<00:34, 7923.93it/s] 31%|███       | 123604/400000 [00:15<00:34, 7968.30it/s] 31%|███       | 124402/400000 [00:15<00:34, 7954.71it/s] 31%|███▏      | 125251/400000 [00:15<00:33, 8107.51it/s] 32%|███▏      | 126063/400000 [00:15<00:34, 7970.44it/s] 32%|███▏      | 126866/400000 [00:16<00:34, 7986.39it/s] 32%|███▏      | 127683/400000 [00:16<00:33, 8037.92it/s] 32%|███▏      | 128488/400000 [00:16<00:33, 8013.77it/s] 32%|███▏      | 129325/400000 [00:16<00:33, 8117.37it/s] 33%|███▎      | 130155/400000 [00:16<00:33, 8170.12it/s] 33%|███▎      | 130974/400000 [00:16<00:32, 8174.16it/s] 33%|███▎      | 131796/400000 [00:16<00:32, 8187.57it/s] 33%|███▎      | 132616/400000 [00:16<00:32, 8145.91it/s] 33%|███▎      | 133450/400000 [00:16<00:32, 8202.96it/s] 34%|███▎      | 134282/400000 [00:16<00:32, 8235.84it/s] 34%|███▍      | 135106/400000 [00:17<00:32, 8127.27it/s] 34%|███▍      | 135938/400000 [00:17<00:32, 8182.91it/s] 34%|███▍      | 136757/400000 [00:17<00:32, 8159.48it/s] 34%|███▍      | 137587/400000 [00:17<00:31, 8200.45it/s] 35%|███▍      | 138408/400000 [00:17<00:31, 8197.17it/s] 35%|███▍      | 139242/400000 [00:17<00:31, 8237.89it/s] 35%|███▌      | 140066/400000 [00:17<00:31, 8158.53it/s] 35%|███▌      | 140883/400000 [00:17<00:31, 8097.98it/s] 35%|███▌      | 141720/400000 [00:17<00:31, 8175.78it/s] 36%|███▌      | 142561/400000 [00:17<00:31, 8243.72it/s] 36%|███▌      | 143391/400000 [00:18<00:31, 8259.10it/s] 36%|███▌      | 144222/400000 [00:18<00:30, 8273.09it/s] 36%|███▋      | 145050/400000 [00:18<00:31, 8117.26it/s] 36%|███▋      | 145863/400000 [00:18<00:32, 7830.33it/s] 37%|███▋      | 146667/400000 [00:18<00:32, 7890.51it/s] 37%|███▋      | 147510/400000 [00:18<00:31, 8043.92it/s] 37%|███▋      | 148351/400000 [00:18<00:30, 8147.79it/s] 37%|███▋      | 149175/400000 [00:18<00:30, 8174.62it/s] 37%|███▋      | 149997/400000 [00:18<00:30, 8188.02it/s] 38%|███▊      | 150839/400000 [00:18<00:30, 8253.39it/s] 38%|███▊      | 151666/400000 [00:19<00:30, 8234.75it/s] 38%|███▊      | 152491/400000 [00:19<00:30, 8208.78it/s] 38%|███▊      | 153313/400000 [00:19<00:31, 7923.90it/s] 39%|███▊      | 154138/400000 [00:19<00:30, 8016.50it/s] 39%|███▊      | 154968/400000 [00:19<00:30, 8096.94it/s] 39%|███▉      | 155799/400000 [00:19<00:29, 8157.10it/s] 39%|███▉      | 156624/400000 [00:19<00:29, 8183.64it/s] 39%|███▉      | 157444/400000 [00:19<00:30, 7967.67it/s] 40%|███▉      | 158277/400000 [00:19<00:29, 8070.41it/s] 40%|███▉      | 159086/400000 [00:20<00:29, 8064.60it/s] 40%|███▉      | 159921/400000 [00:20<00:29, 8145.88it/s] 40%|████      | 160737/400000 [00:20<00:29, 8000.96it/s] 40%|████      | 161539/400000 [00:20<00:30, 7732.38it/s] 41%|████      | 162334/400000 [00:20<00:30, 7796.19it/s] 41%|████      | 163126/400000 [00:20<00:30, 7831.08it/s] 41%|████      | 163929/400000 [00:20<00:29, 7887.04it/s] 41%|████      | 164754/400000 [00:20<00:29, 7989.57it/s] 41%|████▏     | 165555/400000 [00:20<00:29, 7952.79it/s] 42%|████▏     | 166364/400000 [00:20<00:29, 7991.28it/s] 42%|████▏     | 167172/400000 [00:21<00:29, 8017.46it/s] 42%|████▏     | 167984/400000 [00:21<00:28, 8047.70it/s] 42%|████▏     | 168790/400000 [00:21<00:28, 8049.70it/s] 42%|████▏     | 169596/400000 [00:21<00:28, 8013.75it/s] 43%|████▎     | 170398/400000 [00:21<00:28, 7979.07it/s] 43%|████▎     | 171197/400000 [00:21<00:29, 7839.99it/s] 43%|████▎     | 171982/400000 [00:21<00:29, 7759.57it/s] 43%|████▎     | 172759/400000 [00:21<00:29, 7734.67it/s] 43%|████▎     | 173533/400000 [00:21<00:29, 7696.32it/s] 44%|████▎     | 174304/400000 [00:21<00:30, 7515.13it/s] 44%|████▍     | 175083/400000 [00:22<00:29, 7594.01it/s] 44%|████▍     | 175859/400000 [00:22<00:29, 7641.31it/s] 44%|████▍     | 176628/400000 [00:22<00:29, 7653.59it/s] 44%|████▍     | 177407/400000 [00:22<00:28, 7692.54it/s] 45%|████▍     | 178216/400000 [00:22<00:28, 7804.96it/s] 45%|████▍     | 179040/400000 [00:22<00:27, 7930.51it/s] 45%|████▍     | 179840/400000 [00:22<00:27, 7950.39it/s] 45%|████▌     | 180636/400000 [00:22<00:27, 7944.63it/s] 45%|████▌     | 181431/400000 [00:22<00:27, 7920.93it/s] 46%|████▌     | 182245/400000 [00:22<00:27, 7982.74it/s] 46%|████▌     | 183071/400000 [00:23<00:26, 8063.90it/s] 46%|████▌     | 183881/400000 [00:23<00:26, 8071.25it/s] 46%|████▌     | 184689/400000 [00:23<00:27, 7843.78it/s] 46%|████▋     | 185476/400000 [00:23<00:27, 7749.38it/s] 47%|████▋     | 186253/400000 [00:23<00:27, 7750.89it/s] 47%|████▋     | 187030/400000 [00:23<00:27, 7671.85it/s] 47%|████▋     | 187799/400000 [00:23<00:28, 7568.21it/s] 47%|████▋     | 188571/400000 [00:23<00:27, 7612.49it/s] 47%|████▋     | 189334/400000 [00:23<00:27, 7616.08it/s] 48%|████▊     | 190112/400000 [00:23<00:27, 7664.33it/s] 48%|████▊     | 190928/400000 [00:24<00:26, 7805.27it/s] 48%|████▊     | 191755/400000 [00:24<00:26, 7936.49it/s] 48%|████▊     | 192550/400000 [00:24<00:26, 7910.61it/s] 48%|████▊     | 193342/400000 [00:24<00:26, 7889.60it/s] 49%|████▊     | 194154/400000 [00:24<00:25, 7957.17it/s] 49%|████▊     | 194957/400000 [00:24<00:25, 7976.81it/s] 49%|████▉     | 195756/400000 [00:24<00:25, 7872.82it/s] 49%|████▉     | 196574/400000 [00:24<00:25, 7960.79it/s] 49%|████▉     | 197379/400000 [00:24<00:25, 7986.69it/s] 50%|████▉     | 198192/400000 [00:24<00:25, 8027.48it/s] 50%|████▉     | 199010/400000 [00:25<00:24, 8072.44it/s] 50%|████▉     | 199818/400000 [00:25<00:24, 8027.28it/s] 50%|█████     | 200622/400000 [00:25<00:25, 7669.75it/s] 50%|█████     | 201393/400000 [00:25<00:26, 7508.19it/s] 51%|█████     | 202162/400000 [00:25<00:26, 7559.26it/s] 51%|█████     | 202947/400000 [00:25<00:25, 7642.55it/s] 51%|█████     | 203747/400000 [00:25<00:25, 7745.80it/s] 51%|█████     | 204546/400000 [00:25<00:25, 7816.58it/s] 51%|█████▏    | 205329/400000 [00:25<00:25, 7741.59it/s] 52%|█████▏    | 206105/400000 [00:26<00:25, 7689.05it/s] 52%|█████▏    | 206896/400000 [00:26<00:24, 7751.69it/s] 52%|█████▏    | 207689/400000 [00:26<00:24, 7804.30it/s] 52%|█████▏    | 208482/400000 [00:26<00:24, 7841.02it/s] 52%|█████▏    | 209294/400000 [00:26<00:24, 7921.19it/s] 53%|█████▎    | 210141/400000 [00:26<00:23, 8075.76it/s] 53%|█████▎    | 210985/400000 [00:26<00:23, 8181.23it/s] 53%|█████▎    | 211805/400000 [00:26<00:23, 8146.42it/s] 53%|█████▎    | 212637/400000 [00:26<00:22, 8196.61it/s] 53%|█████▎    | 213458/400000 [00:26<00:23, 7960.91it/s] 54%|█████▎    | 214257/400000 [00:27<00:23, 7929.64it/s] 54%|█████▍    | 215114/400000 [00:27<00:22, 8110.14it/s] 54%|█████▍    | 215940/400000 [00:27<00:22, 8151.40it/s] 54%|█████▍    | 216775/400000 [00:27<00:22, 8207.76it/s] 54%|█████▍    | 217597/400000 [00:27<00:22, 8139.45it/s] 55%|█████▍    | 218451/400000 [00:27<00:21, 8254.03it/s] 55%|█████▍    | 219282/400000 [00:27<00:21, 8270.02it/s] 55%|█████▌    | 220123/400000 [00:27<00:21, 8310.86it/s] 55%|█████▌    | 220973/400000 [00:27<00:21, 8364.20it/s] 55%|█████▌    | 221810/400000 [00:27<00:21, 8299.89it/s] 56%|█████▌    | 222641/400000 [00:28<00:21, 8264.82it/s] 56%|█████▌    | 223476/400000 [00:28<00:21, 8289.53it/s] 56%|█████▌    | 224311/400000 [00:28<00:21, 8306.74it/s] 56%|█████▋    | 225157/400000 [00:28<00:20, 8352.10it/s] 56%|█████▋    | 225993/400000 [00:28<00:21, 8122.14it/s] 57%|█████▋    | 226807/400000 [00:28<00:21, 8057.79it/s] 57%|█████▋    | 227614/400000 [00:28<00:21, 7950.35it/s] 57%|█████▋    | 228411/400000 [00:28<00:21, 7876.61it/s] 57%|█████▋    | 229200/400000 [00:28<00:21, 7799.52it/s] 57%|█████▋    | 229981/400000 [00:28<00:22, 7715.66it/s] 58%|█████▊    | 230808/400000 [00:29<00:21, 7873.89it/s] 58%|█████▊    | 231662/400000 [00:29<00:20, 8060.18it/s] 58%|█████▊    | 232471/400000 [00:29<00:21, 7945.30it/s] 58%|█████▊    | 233331/400000 [00:29<00:20, 8130.87it/s] 59%|█████▊    | 234152/400000 [00:29<00:20, 8154.08it/s] 59%|█████▊    | 234972/400000 [00:29<00:20, 8165.04it/s] 59%|█████▉    | 235803/400000 [00:29<00:20, 8207.08it/s] 59%|█████▉    | 236625/400000 [00:29<00:19, 8171.46it/s] 59%|█████▉    | 237443/400000 [00:29<00:19, 8150.21it/s] 60%|█████▉    | 238264/400000 [00:29<00:19, 8167.58it/s] 60%|█████▉    | 239103/400000 [00:30<00:19, 8230.63it/s] 60%|█████▉    | 239934/400000 [00:30<00:19, 8253.14it/s] 60%|██████    | 240760/400000 [00:30<00:19, 8159.54it/s] 60%|██████    | 241587/400000 [00:30<00:19, 8191.82it/s] 61%|██████    | 242412/400000 [00:30<00:19, 8207.50it/s] 61%|██████    | 243233/400000 [00:30<00:19, 8183.16it/s] 61%|██████    | 244052/400000 [00:30<00:19, 7872.74it/s] 61%|██████    | 244901/400000 [00:30<00:19, 8045.85it/s] 61%|██████▏   | 245756/400000 [00:30<00:18, 8188.91it/s] 62%|██████▏   | 246580/400000 [00:30<00:18, 8202.06it/s] 62%|██████▏   | 247430/400000 [00:31<00:18, 8288.75it/s] 62%|██████▏   | 248283/400000 [00:31<00:18, 8356.44it/s] 62%|██████▏   | 249126/400000 [00:31<00:18, 8377.80it/s] 62%|██████▏   | 249981/400000 [00:31<00:17, 8426.68it/s] 63%|██████▎   | 250825/400000 [00:31<00:17, 8321.90it/s] 63%|██████▎   | 251666/400000 [00:31<00:17, 8346.38it/s] 63%|██████▎   | 252502/400000 [00:31<00:17, 8280.77it/s] 63%|██████▎   | 253331/400000 [00:31<00:17, 8280.66it/s] 64%|██████▎   | 254160/400000 [00:31<00:17, 8230.72it/s] 64%|██████▎   | 254984/400000 [00:32<00:17, 8214.35it/s] 64%|██████▍   | 255817/400000 [00:32<00:17, 8246.89it/s] 64%|██████▍   | 256666/400000 [00:32<00:17, 8315.53it/s] 64%|██████▍   | 257513/400000 [00:32<00:17, 8359.02it/s] 65%|██████▍   | 258363/400000 [00:32<00:16, 8398.18it/s] 65%|██████▍   | 259204/400000 [00:32<00:16, 8362.46it/s] 65%|██████▌   | 260041/400000 [00:32<00:16, 8326.18it/s] 65%|██████▌   | 260874/400000 [00:32<00:16, 8326.55it/s] 65%|██████▌   | 261707/400000 [00:32<00:16, 8321.77it/s] 66%|██████▌   | 262542/400000 [00:32<00:16, 8327.61it/s] 66%|██████▌   | 263375/400000 [00:33<00:16, 8311.64it/s] 66%|██████▌   | 264207/400000 [00:33<00:16, 8204.68it/s] 66%|██████▋   | 265028/400000 [00:33<00:16, 8097.26it/s] 66%|██████▋   | 265874/400000 [00:33<00:16, 8202.28it/s] 67%|██████▋   | 266695/400000 [00:33<00:16, 8189.95it/s] 67%|██████▋   | 267538/400000 [00:33<00:16, 8258.76it/s] 67%|██████▋   | 268365/400000 [00:33<00:16, 8093.17it/s] 67%|██████▋   | 269199/400000 [00:33<00:16, 8163.93it/s] 68%|██████▊   | 270053/400000 [00:33<00:15, 8271.25it/s] 68%|██████▊   | 270906/400000 [00:33<00:15, 8345.87it/s] 68%|██████▊   | 271754/400000 [00:34<00:15, 8385.54it/s] 68%|██████▊   | 272594/400000 [00:34<00:15, 8210.03it/s] 68%|██████▊   | 273417/400000 [00:34<00:15, 8208.13it/s] 69%|██████▊   | 274262/400000 [00:34<00:15, 8277.28it/s] 69%|██████▉   | 275094/400000 [00:34<00:15, 8288.38it/s] 69%|██████▉   | 275940/400000 [00:34<00:14, 8338.05it/s] 69%|██████▉   | 276775/400000 [00:34<00:14, 8260.80it/s] 69%|██████▉   | 277623/400000 [00:34<00:14, 8324.00it/s] 70%|██████▉   | 278464/400000 [00:34<00:14, 8347.90it/s] 70%|██████▉   | 279318/400000 [00:34<00:14, 8404.35it/s] 70%|███████   | 280159/400000 [00:35<00:14, 8365.23it/s] 70%|███████   | 280996/400000 [00:35<00:14, 8280.65it/s] 70%|███████   | 281825/400000 [00:35<00:14, 8280.40it/s] 71%|███████   | 282654/400000 [00:35<00:14, 8270.24it/s] 71%|███████   | 283498/400000 [00:35<00:14, 8318.37it/s] 71%|███████   | 284356/400000 [00:35<00:13, 8393.86it/s] 71%|███████▏  | 285196/400000 [00:35<00:13, 8336.45it/s] 72%|███████▏  | 286030/400000 [00:35<00:13, 8170.38it/s] 72%|███████▏  | 286848/400000 [00:35<00:14, 7795.67it/s] 72%|███████▏  | 287632/400000 [00:35<00:14, 7715.09it/s] 72%|███████▏  | 288407/400000 [00:36<00:14, 7555.51it/s] 72%|███████▏  | 289166/400000 [00:36<00:14, 7494.65it/s] 72%|███████▏  | 289918/400000 [00:36<00:14, 7458.88it/s] 73%|███████▎  | 290666/400000 [00:36<00:14, 7430.92it/s] 73%|███████▎  | 291411/400000 [00:36<00:14, 7426.30it/s] 73%|███████▎  | 292156/400000 [00:36<00:14, 7430.34it/s] 73%|███████▎  | 292928/400000 [00:36<00:14, 7514.00it/s] 73%|███████▎  | 293744/400000 [00:36<00:13, 7695.94it/s] 74%|███████▎  | 294554/400000 [00:36<00:13, 7812.43it/s] 74%|███████▍  | 295361/400000 [00:36<00:13, 7887.08it/s] 74%|███████▍  | 296161/400000 [00:37<00:13, 7915.98it/s] 74%|███████▍  | 296962/400000 [00:37<00:12, 7943.50it/s] 74%|███████▍  | 297757/400000 [00:37<00:12, 7931.62it/s] 75%|███████▍  | 298569/400000 [00:37<00:12, 7986.09it/s] 75%|███████▍  | 299383/400000 [00:37<00:12, 8030.80it/s] 75%|███████▌  | 300199/400000 [00:37<00:12, 8063.76it/s] 75%|███████▌  | 301006/400000 [00:37<00:12, 8033.43it/s] 75%|███████▌  | 301810/400000 [00:37<00:12, 7880.42it/s] 76%|███████▌  | 302608/400000 [00:37<00:12, 7907.45it/s] 76%|███████▌  | 303436/400000 [00:37<00:12, 8013.50it/s] 76%|███████▌  | 304258/400000 [00:38<00:11, 8074.13it/s] 76%|███████▋  | 305067/400000 [00:38<00:11, 8030.33it/s] 76%|███████▋  | 305909/400000 [00:38<00:11, 8143.18it/s] 77%|███████▋  | 306738/400000 [00:38<00:11, 8049.16it/s] 77%|███████▋  | 307544/400000 [00:38<00:11, 7982.57it/s] 77%|███████▋  | 308343/400000 [00:38<00:11, 7953.31it/s] 77%|███████▋  | 309176/400000 [00:38<00:11, 8062.60it/s] 78%|███████▊  | 310030/400000 [00:38<00:10, 8199.04it/s] 78%|███████▊  | 310854/400000 [00:38<00:10, 8210.83it/s] 78%|███████▊  | 311703/400000 [00:39<00:10, 8291.10it/s] 78%|███████▊  | 312533/400000 [00:39<00:10, 8256.15it/s] 78%|███████▊  | 313371/400000 [00:39<00:10, 8292.22it/s] 79%|███████▊  | 314203/400000 [00:39<00:10, 8297.97it/s] 79%|███████▉  | 315053/400000 [00:39<00:10, 8354.57it/s] 79%|███████▉  | 315892/400000 [00:39<00:10, 8363.73it/s] 79%|███████▉  | 316729/400000 [00:39<00:10, 8245.39it/s] 79%|███████▉  | 317559/400000 [00:39<00:09, 8258.63it/s] 80%|███████▉  | 318405/400000 [00:39<00:09, 8314.81it/s] 80%|███████▉  | 319244/400000 [00:39<00:09, 8334.67it/s] 80%|████████  | 320084/400000 [00:40<00:09, 8351.55it/s] 80%|████████  | 320920/400000 [00:40<00:09, 8284.45it/s] 80%|████████  | 321749/400000 [00:40<00:09, 8245.77it/s] 81%|████████  | 322574/400000 [00:40<00:09, 8228.25it/s] 81%|████████  | 323397/400000 [00:40<00:09, 8203.62it/s] 81%|████████  | 324242/400000 [00:40<00:09, 8274.14it/s] 81%|████████▏ | 325071/400000 [00:40<00:09, 8276.18it/s] 81%|████████▏ | 325899/400000 [00:40<00:09, 8227.17it/s] 82%|████████▏ | 326722/400000 [00:40<00:08, 8204.76it/s] 82%|████████▏ | 327543/400000 [00:40<00:08, 8121.61it/s] 82%|████████▏ | 328356/400000 [00:41<00:08, 8028.82it/s] 82%|████████▏ | 329160/400000 [00:41<00:08, 8032.08it/s] 82%|████████▏ | 329972/400000 [00:41<00:08, 8057.45it/s] 83%|████████▎ | 330803/400000 [00:41<00:08, 8130.62it/s] 83%|████████▎ | 331637/400000 [00:41<00:08, 8191.36it/s] 83%|████████▎ | 332479/400000 [00:41<00:08, 8255.98it/s] 83%|████████▎ | 333315/400000 [00:41<00:08, 8285.70it/s] 84%|████████▎ | 334144/400000 [00:41<00:08, 8218.12it/s] 84%|████████▎ | 334989/400000 [00:41<00:07, 8284.90it/s] 84%|████████▍ | 335818/400000 [00:41<00:08, 7956.14it/s] 84%|████████▍ | 336617/400000 [00:42<00:08, 7750.75it/s] 84%|████████▍ | 337396/400000 [00:42<00:08, 7629.32it/s] 85%|████████▍ | 338162/400000 [00:42<00:08, 7470.70it/s] 85%|████████▍ | 338912/400000 [00:42<00:08, 7446.88it/s] 85%|████████▍ | 339704/400000 [00:42<00:07, 7582.53it/s] 85%|████████▌ | 340532/400000 [00:42<00:07, 7779.10it/s] 85%|████████▌ | 341364/400000 [00:42<00:07, 7933.37it/s] 86%|████████▌ | 342162/400000 [00:42<00:07, 7945.86it/s] 86%|████████▌ | 342999/400000 [00:42<00:07, 8068.34it/s] 86%|████████▌ | 343841/400000 [00:42<00:06, 8169.35it/s] 86%|████████▌ | 344681/400000 [00:43<00:06, 8236.47it/s] 86%|████████▋ | 345516/400000 [00:43<00:06, 8268.50it/s] 87%|████████▋ | 346344/400000 [00:43<00:06, 8207.33it/s] 87%|████████▋ | 347177/400000 [00:43<00:06, 8241.10it/s] 87%|████████▋ | 348002/400000 [00:43<00:06, 8144.60it/s] 87%|████████▋ | 348818/400000 [00:43<00:06, 8122.48it/s] 87%|████████▋ | 349631/400000 [00:43<00:06, 8079.29it/s] 88%|████████▊ | 350440/400000 [00:43<00:06, 7979.28it/s] 88%|████████▊ | 351239/400000 [00:43<00:06, 7902.86it/s] 88%|████████▊ | 352043/400000 [00:43<00:06, 7942.27it/s] 88%|████████▊ | 352856/400000 [00:44<00:05, 7996.52it/s] 88%|████████▊ | 353688/400000 [00:44<00:05, 8088.92it/s] 89%|████████▊ | 354498/400000 [00:44<00:05, 8085.35it/s] 89%|████████▉ | 355344/400000 [00:44<00:05, 8192.30it/s] 89%|████████▉ | 356184/400000 [00:44<00:05, 8253.36it/s] 89%|████████▉ | 357019/400000 [00:44<00:05, 8279.67it/s] 89%|████████▉ | 357849/400000 [00:44<00:05, 8284.25it/s] 90%|████████▉ | 358678/400000 [00:44<00:05, 8228.59it/s] 90%|████████▉ | 359521/400000 [00:44<00:04, 8286.77it/s] 90%|█████████ | 360354/400000 [00:44<00:04, 8297.73it/s] 90%|█████████ | 361184/400000 [00:45<00:04, 8243.40it/s] 91%|█████████ | 362017/400000 [00:45<00:04, 8265.85it/s] 91%|█████████ | 362844/400000 [00:45<00:04, 8189.37it/s] 91%|█████████ | 363664/400000 [00:45<00:04, 8056.91it/s] 91%|█████████ | 364471/400000 [00:45<00:04, 8026.02it/s] 91%|█████████▏| 365298/400000 [00:45<00:04, 8097.35it/s] 92%|█████████▏| 366126/400000 [00:45<00:04, 8150.05it/s] 92%|█████████▏| 366942/400000 [00:45<00:04, 8121.89it/s] 92%|█████████▏| 367762/400000 [00:45<00:03, 8143.89it/s] 92%|█████████▏| 368577/400000 [00:46<00:03, 8132.22it/s] 92%|█████████▏| 369391/400000 [00:46<00:03, 7813.06it/s] 93%|█████████▎| 370176/400000 [00:46<00:03, 7776.83it/s] 93%|█████████▎| 370993/400000 [00:46<00:03, 7888.48it/s] 93%|█████████▎| 371829/400000 [00:46<00:03, 8023.13it/s] 93%|█████████▎| 372668/400000 [00:46<00:03, 8129.58it/s] 93%|█████████▎| 373513/400000 [00:46<00:03, 8222.92it/s] 94%|█████████▎| 374340/400000 [00:46<00:03, 8235.53it/s] 94%|█████████▍| 375165/400000 [00:46<00:03, 8222.87it/s] 94%|█████████▍| 376002/400000 [00:46<00:02, 8265.99it/s] 94%|█████████▍| 376832/400000 [00:47<00:02, 8274.92it/s] 94%|█████████▍| 377660/400000 [00:47<00:02, 8170.33it/s] 95%|█████████▍| 378478/400000 [00:47<00:02, 7941.86it/s] 95%|█████████▍| 379308/400000 [00:47<00:02, 8045.06it/s] 95%|█████████▌| 380131/400000 [00:47<00:02, 8098.47it/s] 95%|█████████▌| 380952/400000 [00:47<00:02, 8129.15it/s] 95%|█████████▌| 381766/400000 [00:47<00:02, 8127.50it/s] 96%|█████████▌| 382580/400000 [00:47<00:02, 7964.10it/s] 96%|█████████▌| 383403/400000 [00:47<00:02, 8039.89it/s] 96%|█████████▌| 384215/400000 [00:47<00:01, 8063.60it/s] 96%|█████████▋| 385023/400000 [00:48<00:01, 8030.82it/s] 96%|█████████▋| 385827/400000 [00:48<00:01, 7898.23it/s] 97%|█████████▋| 386618/400000 [00:48<00:01, 7635.26it/s] 97%|█████████▋| 387385/400000 [00:48<00:01, 7479.91it/s] 97%|█████████▋| 388136/400000 [00:48<00:01, 7372.43it/s] 97%|█████████▋| 388885/400000 [00:48<00:01, 7406.09it/s] 97%|█████████▋| 389633/400000 [00:48<00:01, 7427.93it/s] 98%|█████████▊| 390377/400000 [00:48<00:01, 7373.70it/s] 98%|█████████▊| 391116/400000 [00:48<00:01, 7328.40it/s] 98%|█████████▊| 391850/400000 [00:48<00:01, 7319.47it/s] 98%|█████████▊| 392594/400000 [00:49<00:01, 7353.63it/s] 98%|█████████▊| 393330/400000 [00:49<00:00, 7353.12it/s] 99%|█████████▊| 394066/400000 [00:49<00:00, 7211.16it/s] 99%|█████████▊| 394836/400000 [00:49<00:00, 7348.22it/s] 99%|█████████▉| 395647/400000 [00:49<00:00, 7560.06it/s] 99%|█████████▉| 396447/400000 [00:49<00:00, 7686.85it/s] 99%|█████████▉| 397273/400000 [00:49<00:00, 7848.83it/s]100%|█████████▉| 398106/400000 [00:49<00:00, 7986.27it/s]100%|█████████▉| 398907/400000 [00:49<00:00, 7671.80it/s]100%|█████████▉| 399688/400000 [00:50<00:00, 7711.44it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7991.51it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fac645d6c88> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011367624153906694 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.011036617700072833 	 Accuracy: 64

  model saves at 64% accuracy 

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
2020-05-15 19:24:05.836180: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 19:24:05.840598: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-15 19:24:05.840741: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560e13291580 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 19:24:05.840758: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fac70152f98> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6206 - accuracy: 0.5030
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8890 - accuracy: 0.4855 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.9580 - accuracy: 0.4810
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8238 - accuracy: 0.4897
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7954 - accuracy: 0.4916
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7433 - accuracy: 0.4950
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7323 - accuracy: 0.4957
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7011 - accuracy: 0.4978
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6734 - accuracy: 0.4996
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6636 - accuracy: 0.5002
11000/25000 [============>.................] - ETA: 4s - loss: 7.6597 - accuracy: 0.5005
12000/25000 [=============>................] - ETA: 3s - loss: 7.6653 - accuracy: 0.5001
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6713 - accuracy: 0.4997
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6699 - accuracy: 0.4998
15000/25000 [=================>............] - ETA: 2s - loss: 7.6768 - accuracy: 0.4993
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7002 - accuracy: 0.4978
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7063 - accuracy: 0.4974
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7135 - accuracy: 0.4969
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6881 - accuracy: 0.4986
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6849 - accuracy: 0.4988
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6764 - accuracy: 0.4994
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 8s 336us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fabe177c0f0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fabe069f278> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 2.0944 - crf_viterbi_accuracy: 0.0267 - val_loss: 2.0616 - val_crf_viterbi_accuracy: 0.0000e+00

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

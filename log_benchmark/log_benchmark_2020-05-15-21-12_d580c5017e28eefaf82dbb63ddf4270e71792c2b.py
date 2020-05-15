
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f13a8168fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 21:12:54.211314
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 21:12:54.217818
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 21:12:54.222616
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 21:12:54.230316
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f13b4180400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354240.7500
Epoch 2/10

1/1 [==============================] - 0s 106ms/step - loss: 295383.9062
Epoch 3/10

1/1 [==============================] - 0s 94ms/step - loss: 224959.8594
Epoch 4/10

1/1 [==============================] - 0s 92ms/step - loss: 152916.6875
Epoch 5/10

1/1 [==============================] - 0s 96ms/step - loss: 95774.4062
Epoch 6/10

1/1 [==============================] - 0s 97ms/step - loss: 58660.8203
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 37598.9023
Epoch 8/10

1/1 [==============================] - 0s 98ms/step - loss: 25171.7324
Epoch 9/10

1/1 [==============================] - 0s 91ms/step - loss: 17593.6719
Epoch 10/10

1/1 [==============================] - 0s 90ms/step - loss: 13048.2227

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.1427783e-01  4.4941158e+00  5.9625306e+00  3.9715893e+00
   6.9272289e+00  6.0859222e+00  4.2103405e+00  4.9534864e+00
   6.7908473e+00  5.3113976e+00  5.3870916e+00  6.1757245e+00
   5.9208064e+00  3.5855565e+00  5.3789892e+00  5.7991543e+00
   5.9171009e+00  4.8846779e+00  5.2751641e+00  5.7282410e+00
   4.6129251e+00  5.2847562e+00  4.7300730e+00  5.4384098e+00
   6.2715955e+00  5.6895800e+00  5.4201598e+00  6.0930405e+00
   4.5286813e+00  5.1359229e+00  6.0459533e+00  5.2682867e+00
   5.2803931e+00  5.6518583e+00  4.1797557e+00  4.9165463e+00
   5.2958622e+00  4.8879447e+00  5.1312804e+00  6.9027371e+00
   6.4369512e+00  6.1501641e+00  6.1375613e+00  5.3612618e+00
   4.8133607e+00  4.1976347e+00  5.3364716e+00  6.1192732e+00
   4.5678129e+00  4.5381680e+00  4.4008551e+00  5.8114939e+00
   5.9832854e+00  5.3644505e+00  3.6599107e+00  4.5650778e+00
   6.6852541e+00  5.0632048e+00  5.8008204e+00  5.1024013e+00
   1.0061443e+00  2.7729309e-01 -4.2481813e-01 -1.6832314e+00
   8.7652373e-01 -1.1732832e+00 -1.5299052e+00 -6.5797240e-02
  -2.9286462e-01  4.6642977e-01  2.1326768e-01 -7.9352891e-01
  -4.8177177e-01 -8.1005830e-01 -9.5977247e-01  1.2578536e+00
  -5.0005126e-01  3.7907723e-01  2.9954463e-02 -7.2297931e-01
  -8.3789289e-02 -5.6962490e-01  6.5264428e-01  1.3184229e-01
  -7.4406314e-01 -5.4267323e-01 -7.5610518e-01  1.0367754e+00
   7.6152718e-01  8.3654189e-01 -8.9417386e-01  6.2316287e-01
   7.9716170e-01 -3.6197820e-01  4.4774210e-01  1.2769353e-01
   1.0415859e+00 -1.2901924e+00 -1.2101021e+00 -1.5983857e-01
   1.8423927e+00 -1.2896149e+00  4.4253251e-01  2.6774496e-01
  -5.0591505e-01 -1.5196093e+00 -5.8895350e-04 -3.1384379e-01
   1.2390763e+00 -2.5100487e-01  2.0057371e-01  7.8416693e-01
   1.1318302e-01  1.0824296e+00  1.4198190e-01  5.1589298e-01
  -9.2742562e-01  1.6440985e+00 -4.5931157e-01 -1.5481923e+00
   1.5632202e-01 -3.6656737e-02  5.8506358e-01  1.6853881e+00
   1.2604157e+00 -1.4519984e-01  3.6340332e-01 -4.5949242e-01
  -9.1461933e-01 -2.5007743e-01 -4.8155415e-01 -8.4356141e-01
  -5.3712821e-01 -3.4682125e-01 -1.3098365e-01  2.3033586e-01
   7.7597797e-02 -8.4648359e-01  9.5704442e-01 -5.2881515e-01
  -4.2288673e-01  5.0490946e-02  7.1658510e-01  2.2204888e-01
   1.4470872e+00 -1.8775368e-01  1.1839678e+00  3.6635897e-01
  -1.9002527e-01  1.2740748e+00  9.8048466e-01  1.4894515e-01
  -1.4328420e+00 -8.3792132e-01  1.2121277e+00  1.2100005e+00
  -6.0482478e-01  1.5888978e+00  6.1994159e-01 -4.3566331e-01
   3.5066193e-01  1.4972746e+00 -8.0313683e-01 -6.6494501e-01
  -1.0750327e+00 -4.5702028e-01 -5.8284026e-01  1.3304814e+00
  -2.2059572e-01 -9.1819465e-04  1.2136855e+00 -2.1454862e-01
   2.1998970e-01  4.3187314e-01  9.7038126e-01  1.1680837e+00
  -9.9278665e-01 -1.3283585e+00 -3.4212485e-01  3.2000464e-01
   4.2961895e-02  5.5771594e+00  7.4776740e+00  5.4034357e+00
   5.4885387e+00  4.6652522e+00  6.5954638e+00  5.2069287e+00
   6.1430230e+00  6.7207232e+00  6.5222855e+00  5.6788230e+00
   6.2794623e+00  6.4223514e+00  4.8847570e+00  6.5329423e+00
   5.6419892e+00  7.1196976e+00  6.0802841e+00  4.8728156e+00
   5.6973109e+00  7.0874238e+00  6.8331137e+00  6.7624779e+00
   4.9904909e+00  6.5246572e+00  5.8207474e+00  6.6860518e+00
   5.9257374e+00  7.0710220e+00  7.6171145e+00  5.5164971e+00
   7.2369442e+00  5.8612757e+00  6.5661688e+00  4.8758178e+00
   4.6861057e+00  4.7660322e+00  7.1467009e+00  6.1214437e+00
   6.7066331e+00  5.0547490e+00  5.7273979e+00  6.0560431e+00
   5.8409276e+00  7.0391216e+00  5.8049822e+00  6.8577080e+00
   6.0919414e+00  4.9356737e+00  5.9848604e+00  6.4725599e+00
   7.6490507e+00  5.2384448e+00  7.3200655e+00  5.4370017e+00
   4.6302419e+00  6.1775193e+00  5.1002522e+00  4.2913122e+00
   1.4160684e+00  1.5749947e+00  8.6054248e-01  1.6041301e+00
   1.6116530e+00  3.9090753e-01  1.2851417e+00  1.5761027e+00
   5.5955416e-01  1.6349039e+00  1.3819878e+00  8.3170617e-01
   8.0529356e-01  5.3163338e-01  1.7214038e+00  1.7754204e+00
   1.6744252e+00  3.6489856e-01  8.1566226e-01  4.8714817e-01
   2.2862685e-01  1.0979254e+00  1.0534066e+00  2.0561824e+00
   1.3160698e+00  1.2151635e+00  5.8375728e-01  2.1495130e+00
   1.5117848e+00  2.0090232e+00  2.3153758e-01  3.9599127e-01
   9.8656321e-01  3.8337332e-01  2.0060887e+00  8.9989376e-01
   8.3910537e-01  5.3384638e-01  7.6135182e-01  2.6715884e+00
   1.5496991e+00  1.4277869e+00  2.1567255e-01  1.4520457e+00
   2.7658027e-01  1.5659330e+00  1.4797755e+00  9.8032075e-01
   1.8019300e+00  1.0392684e+00  2.9809582e-01  9.7109997e-01
   6.0245985e-01  5.9177375e-01  2.0705824e+00  1.2226303e+00
   6.1840630e-01  1.4973089e+00  2.9537159e-01  1.7864094e+00
   1.1555194e+00  1.7201145e+00  4.7649813e-01  9.0608603e-01
   8.5888201e-01  1.7986407e+00  1.7087773e+00  9.1842431e-01
   5.1827151e-01  7.5215209e-01  2.5609648e-01  2.0547515e-01
   5.4394490e-01  9.5065910e-01  1.4484987e+00  1.5324389e+00
   5.3715730e-01  1.8431752e+00  1.0151416e+00  4.5696747e-01
   1.4976807e+00  1.6666045e+00  1.9365249e+00  5.3371000e-01
   4.6151984e-01  2.5412507e+00  1.4596162e+00  4.3765831e-01
   4.8925436e-01  8.4602225e-01  2.2244735e+00  9.8369724e-01
   4.9786866e-01  1.3136666e+00  4.6701705e-01  6.1980891e-01
   2.2451591e+00  9.8318648e-01  6.2249821e-01  2.0818672e+00
   6.2280232e-01  1.0862132e+00  1.4285488e+00  1.3282814e+00
   1.4664001e+00  1.1878359e+00  1.4417522e+00  2.0624967e+00
   1.7201388e+00  8.7766880e-01  7.1805245e-01  8.2321250e-01
   4.5506263e-01  4.3014181e-01  8.7710899e-01  1.2777941e+00
   7.6098597e-01  6.8133205e-01  4.2366600e-01  5.2271104e-01
   4.6255584e+00 -8.0014429e+00 -6.4901552e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 21:13:03.823827
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    96.533
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 21:13:03.828773
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9337.32
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 21:13:03.832022
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.2909
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 21:13:03.835855
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -835.223
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139722061194744
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139719531319872
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139719531320376
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139719531320880
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139719531321384
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139719531321888

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f13b0001ef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.440849
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.421303
grad_step = 000002, loss = 0.404585
grad_step = 000003, loss = 0.383087
grad_step = 000004, loss = 0.360995
grad_step = 000005, loss = 0.343575
grad_step = 000006, loss = 0.323928
grad_step = 000007, loss = 0.306937
grad_step = 000008, loss = 0.299483
grad_step = 000009, loss = 0.291790
grad_step = 000010, loss = 0.278446
grad_step = 000011, loss = 0.266152
grad_step = 000012, loss = 0.256698
grad_step = 000013, loss = 0.247695
grad_step = 000014, loss = 0.238031
grad_step = 000015, loss = 0.228188
grad_step = 000016, loss = 0.218764
grad_step = 000017, loss = 0.210129
grad_step = 000018, loss = 0.202207
grad_step = 000019, loss = 0.194118
grad_step = 000020, loss = 0.185011
grad_step = 000021, loss = 0.175650
grad_step = 000022, loss = 0.167424
grad_step = 000023, loss = 0.160264
grad_step = 000024, loss = 0.153213
grad_step = 000025, loss = 0.145910
grad_step = 000026, loss = 0.138711
grad_step = 000027, loss = 0.132044
grad_step = 000028, loss = 0.125805
grad_step = 000029, loss = 0.119658
grad_step = 000030, loss = 0.113531
grad_step = 000031, loss = 0.107706
grad_step = 000032, loss = 0.102374
grad_step = 000033, loss = 0.097309
grad_step = 000034, loss = 0.092313
grad_step = 000035, loss = 0.087458
grad_step = 000036, loss = 0.082847
grad_step = 000037, loss = 0.078539
grad_step = 000038, loss = 0.074447
grad_step = 000039, loss = 0.070490
grad_step = 000040, loss = 0.066741
grad_step = 000041, loss = 0.063202
grad_step = 000042, loss = 0.059772
grad_step = 000043, loss = 0.056454
grad_step = 000044, loss = 0.053357
grad_step = 000045, loss = 0.050461
grad_step = 000046, loss = 0.047651
grad_step = 000047, loss = 0.044958
grad_step = 000048, loss = 0.042469
grad_step = 000049, loss = 0.040096
grad_step = 000050, loss = 0.037793
grad_step = 000051, loss = 0.035630
grad_step = 000052, loss = 0.033625
grad_step = 000053, loss = 0.031735
grad_step = 000054, loss = 0.029909
grad_step = 000055, loss = 0.028188
grad_step = 000056, loss = 0.026603
grad_step = 000057, loss = 0.025076
grad_step = 000058, loss = 0.023612
grad_step = 000059, loss = 0.022266
grad_step = 000060, loss = 0.021018
grad_step = 000061, loss = 0.019818
grad_step = 000062, loss = 0.018689
grad_step = 000063, loss = 0.017647
grad_step = 000064, loss = 0.016645
grad_step = 000065, loss = 0.015692
grad_step = 000066, loss = 0.014815
grad_step = 000067, loss = 0.013990
grad_step = 000068, loss = 0.013203
grad_step = 000069, loss = 0.012476
grad_step = 000070, loss = 0.011791
grad_step = 000071, loss = 0.011135
grad_step = 000072, loss = 0.010528
grad_step = 000073, loss = 0.009960
grad_step = 000074, loss = 0.009419
grad_step = 000075, loss = 0.008913
grad_step = 000076, loss = 0.008444
grad_step = 000077, loss = 0.007996
grad_step = 000078, loss = 0.007579
grad_step = 000079, loss = 0.007196
grad_step = 000080, loss = 0.006832
grad_step = 000081, loss = 0.006494
grad_step = 000082, loss = 0.006185
grad_step = 000083, loss = 0.005891
grad_step = 000084, loss = 0.005619
grad_step = 000085, loss = 0.005367
grad_step = 000086, loss = 0.005131
grad_step = 000087, loss = 0.004910
grad_step = 000088, loss = 0.004705
grad_step = 000089, loss = 0.004511
grad_step = 000090, loss = 0.004330
grad_step = 000091, loss = 0.004162
grad_step = 000092, loss = 0.004003
grad_step = 000093, loss = 0.003854
grad_step = 000094, loss = 0.003714
grad_step = 000095, loss = 0.003582
grad_step = 000096, loss = 0.003459
grad_step = 000097, loss = 0.003344
grad_step = 000098, loss = 0.003235
grad_step = 000099, loss = 0.003134
grad_step = 000100, loss = 0.003040
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002952
grad_step = 000102, loss = 0.002870
grad_step = 000103, loss = 0.002792
grad_step = 000104, loss = 0.002718
grad_step = 000105, loss = 0.002650
grad_step = 000106, loss = 0.002586
grad_step = 000107, loss = 0.002527
grad_step = 000108, loss = 0.002473
grad_step = 000109, loss = 0.002424
grad_step = 000110, loss = 0.002383
grad_step = 000111, loss = 0.002353
grad_step = 000112, loss = 0.002324
grad_step = 000113, loss = 0.002286
grad_step = 000114, loss = 0.002230
grad_step = 000115, loss = 0.002183
grad_step = 000116, loss = 0.002159
grad_step = 000117, loss = 0.002147
grad_step = 000118, loss = 0.002133
grad_step = 000119, loss = 0.002102
grad_step = 000120, loss = 0.002064
grad_step = 000121, loss = 0.002032
grad_step = 000122, loss = 0.002014
grad_step = 000123, loss = 0.002007
grad_step = 000124, loss = 0.002004
grad_step = 000125, loss = 0.002004
grad_step = 000126, loss = 0.001995
grad_step = 000127, loss = 0.001979
grad_step = 000128, loss = 0.001948
grad_step = 000129, loss = 0.001918
grad_step = 000130, loss = 0.001896
grad_step = 000131, loss = 0.001885
grad_step = 000132, loss = 0.001884
grad_step = 000133, loss = 0.001892
grad_step = 000134, loss = 0.001914
grad_step = 000135, loss = 0.001945
grad_step = 000136, loss = 0.001985
grad_step = 000137, loss = 0.001960
grad_step = 000138, loss = 0.001886
grad_step = 000139, loss = 0.001820
grad_step = 000140, loss = 0.001832
grad_step = 000141, loss = 0.001881
grad_step = 000142, loss = 0.001886
grad_step = 000143, loss = 0.001843
grad_step = 000144, loss = 0.001791
grad_step = 000145, loss = 0.001787
grad_step = 000146, loss = 0.001818
grad_step = 000147, loss = 0.001835
grad_step = 000148, loss = 0.001821
grad_step = 000149, loss = 0.001780
grad_step = 000150, loss = 0.001753
grad_step = 000151, loss = 0.001754
grad_step = 000152, loss = 0.001771
grad_step = 000153, loss = 0.001789
grad_step = 000154, loss = 0.001789
grad_step = 000155, loss = 0.001773
grad_step = 000156, loss = 0.001744
grad_step = 000157, loss = 0.001721
grad_step = 000158, loss = 0.001711
grad_step = 000159, loss = 0.001715
grad_step = 000160, loss = 0.001727
grad_step = 000161, loss = 0.001742
grad_step = 000162, loss = 0.001763
grad_step = 000163, loss = 0.001777
grad_step = 000164, loss = 0.001783
grad_step = 000165, loss = 0.001761
grad_step = 000166, loss = 0.001724
grad_step = 000167, loss = 0.001686
grad_step = 000168, loss = 0.001665
grad_step = 000169, loss = 0.001666
grad_step = 000170, loss = 0.001679
grad_step = 000171, loss = 0.001698
grad_step = 000172, loss = 0.001710
grad_step = 000173, loss = 0.001715
grad_step = 000174, loss = 0.001703
grad_step = 000175, loss = 0.001679
grad_step = 000176, loss = 0.001652
grad_step = 000177, loss = 0.001632
grad_step = 000178, loss = 0.001625
grad_step = 000179, loss = 0.001628
grad_step = 000180, loss = 0.001640
grad_step = 000181, loss = 0.001657
grad_step = 000182, loss = 0.001681
grad_step = 000183, loss = 0.001709
grad_step = 000184, loss = 0.001716
grad_step = 000185, loss = 0.001700
grad_step = 000186, loss = 0.001659
grad_step = 000187, loss = 0.001620
grad_step = 000188, loss = 0.001600
grad_step = 000189, loss = 0.001604
grad_step = 000190, loss = 0.001629
grad_step = 000191, loss = 0.001644
grad_step = 000192, loss = 0.001652
grad_step = 000193, loss = 0.001648
grad_step = 000194, loss = 0.001628
grad_step = 000195, loss = 0.001605
grad_step = 000196, loss = 0.001584
grad_step = 000197, loss = 0.001573
grad_step = 000198, loss = 0.001572
grad_step = 000199, loss = 0.001578
grad_step = 000200, loss = 0.001586
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001591
grad_step = 000202, loss = 0.001595
grad_step = 000203, loss = 0.001596
grad_step = 000204, loss = 0.001602
grad_step = 000205, loss = 0.001624
grad_step = 000206, loss = 0.001659
grad_step = 000207, loss = 0.001722
grad_step = 000208, loss = 0.001633
grad_step = 000209, loss = 0.001575
grad_step = 000210, loss = 0.001566
grad_step = 000211, loss = 0.001581
grad_step = 000212, loss = 0.001581
grad_step = 000213, loss = 0.001561
grad_step = 000214, loss = 0.001552
grad_step = 000215, loss = 0.001572
grad_step = 000216, loss = 0.001578
grad_step = 000217, loss = 0.001567
grad_step = 000218, loss = 0.001557
grad_step = 000219, loss = 0.001553
grad_step = 000220, loss = 0.001539
grad_step = 000221, loss = 0.001533
grad_step = 000222, loss = 0.001550
grad_step = 000223, loss = 0.001568
grad_step = 000224, loss = 0.001596
grad_step = 000225, loss = 0.001625
grad_step = 000226, loss = 0.001659
grad_step = 000227, loss = 0.001660
grad_step = 000228, loss = 0.001659
grad_step = 000229, loss = 0.001652
grad_step = 000230, loss = 0.001593
grad_step = 000231, loss = 0.001538
grad_step = 000232, loss = 0.001528
grad_step = 000233, loss = 0.001574
grad_step = 000234, loss = 0.001602
grad_step = 000235, loss = 0.001567
grad_step = 000236, loss = 0.001510
grad_step = 000237, loss = 0.001499
grad_step = 000238, loss = 0.001520
grad_step = 000239, loss = 0.001525
grad_step = 000240, loss = 0.001511
grad_step = 000241, loss = 0.001509
grad_step = 000242, loss = 0.001520
grad_step = 000243, loss = 0.001515
grad_step = 000244, loss = 0.001496
grad_step = 000245, loss = 0.001474
grad_step = 000246, loss = 0.001471
grad_step = 000247, loss = 0.001483
grad_step = 000248, loss = 0.001486
grad_step = 000249, loss = 0.001480
grad_step = 000250, loss = 0.001473
grad_step = 000251, loss = 0.001471
grad_step = 000252, loss = 0.001475
grad_step = 000253, loss = 0.001483
grad_step = 000254, loss = 0.001488
grad_step = 000255, loss = 0.001485
grad_step = 000256, loss = 0.001483
grad_step = 000257, loss = 0.001484
grad_step = 000258, loss = 0.001488
grad_step = 000259, loss = 0.001499
grad_step = 000260, loss = 0.001503
grad_step = 000261, loss = 0.001507
grad_step = 000262, loss = 0.001499
grad_step = 000263, loss = 0.001497
grad_step = 000264, loss = 0.001501
grad_step = 000265, loss = 0.001516
grad_step = 000266, loss = 0.001542
grad_step = 000267, loss = 0.001531
grad_step = 000268, loss = 0.001499
grad_step = 000269, loss = 0.001450
grad_step = 000270, loss = 0.001436
grad_step = 000271, loss = 0.001455
grad_step = 000272, loss = 0.001462
grad_step = 000273, loss = 0.001444
grad_step = 000274, loss = 0.001414
grad_step = 000275, loss = 0.001404
grad_step = 000276, loss = 0.001415
grad_step = 000277, loss = 0.001425
grad_step = 000278, loss = 0.001421
grad_step = 000279, loss = 0.001403
grad_step = 000280, loss = 0.001389
grad_step = 000281, loss = 0.001388
grad_step = 000282, loss = 0.001394
grad_step = 000283, loss = 0.001400
grad_step = 000284, loss = 0.001399
grad_step = 000285, loss = 0.001394
grad_step = 000286, loss = 0.001389
grad_step = 000287, loss = 0.001396
grad_step = 000288, loss = 0.001428
grad_step = 000289, loss = 0.001499
grad_step = 000290, loss = 0.001668
grad_step = 000291, loss = 0.001862
grad_step = 000292, loss = 0.002125
grad_step = 000293, loss = 0.002092
grad_step = 000294, loss = 0.001650
grad_step = 000295, loss = 0.001425
grad_step = 000296, loss = 0.001598
grad_step = 000297, loss = 0.001698
grad_step = 000298, loss = 0.001473
grad_step = 000299, loss = 0.001401
grad_step = 000300, loss = 0.001581
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001510
grad_step = 000302, loss = 0.001369
grad_step = 000303, loss = 0.001451
grad_step = 000304, loss = 0.001481
grad_step = 000305, loss = 0.001415
grad_step = 000306, loss = 0.001396
grad_step = 000307, loss = 0.001416
grad_step = 000308, loss = 0.001413
grad_step = 000309, loss = 0.001396
grad_step = 000310, loss = 0.001388
grad_step = 000311, loss = 0.001369
grad_step = 000312, loss = 0.001400
grad_step = 000313, loss = 0.001406
grad_step = 000314, loss = 0.001340
grad_step = 000315, loss = 0.001350
grad_step = 000316, loss = 0.001395
grad_step = 000317, loss = 0.001346
grad_step = 000318, loss = 0.001328
grad_step = 000319, loss = 0.001352
grad_step = 000320, loss = 0.001336
grad_step = 000321, loss = 0.001332
grad_step = 000322, loss = 0.001328
grad_step = 000323, loss = 0.001306
grad_step = 000324, loss = 0.001315
grad_step = 000325, loss = 0.001324
grad_step = 000326, loss = 0.001298
grad_step = 000327, loss = 0.001289
grad_step = 000328, loss = 0.001300
grad_step = 000329, loss = 0.001297
grad_step = 000330, loss = 0.001287
grad_step = 000331, loss = 0.001279
grad_step = 000332, loss = 0.001275
grad_step = 000333, loss = 0.001280
grad_step = 000334, loss = 0.001276
grad_step = 000335, loss = 0.001262
grad_step = 000336, loss = 0.001261
grad_step = 000337, loss = 0.001264
grad_step = 000338, loss = 0.001261
grad_step = 000339, loss = 0.001257
grad_step = 000340, loss = 0.001251
grad_step = 000341, loss = 0.001244
grad_step = 000342, loss = 0.001244
grad_step = 000343, loss = 0.001244
grad_step = 000344, loss = 0.001241
grad_step = 000345, loss = 0.001240
grad_step = 000346, loss = 0.001247
grad_step = 000347, loss = 0.001271
grad_step = 000348, loss = 0.001355
grad_step = 000349, loss = 0.001501
grad_step = 000350, loss = 0.001790
grad_step = 000351, loss = 0.001465
grad_step = 000352, loss = 0.001237
grad_step = 000353, loss = 0.001321
grad_step = 000354, loss = 0.001364
grad_step = 000355, loss = 0.001243
grad_step = 000356, loss = 0.001272
grad_step = 000357, loss = 0.001297
grad_step = 000358, loss = 0.001224
grad_step = 000359, loss = 0.001255
grad_step = 000360, loss = 0.001257
grad_step = 000361, loss = 0.001215
grad_step = 000362, loss = 0.001234
grad_step = 000363, loss = 0.001230
grad_step = 000364, loss = 0.001202
grad_step = 000365, loss = 0.001210
grad_step = 000366, loss = 0.001212
grad_step = 000367, loss = 0.001195
grad_step = 000368, loss = 0.001192
grad_step = 000369, loss = 0.001201
grad_step = 000370, loss = 0.001191
grad_step = 000371, loss = 0.001179
grad_step = 000372, loss = 0.001187
grad_step = 000373, loss = 0.001187
grad_step = 000374, loss = 0.001172
grad_step = 000375, loss = 0.001170
grad_step = 000376, loss = 0.001177
grad_step = 000377, loss = 0.001171
grad_step = 000378, loss = 0.001164
grad_step = 000379, loss = 0.001162
grad_step = 000380, loss = 0.001165
grad_step = 000381, loss = 0.001172
grad_step = 000382, loss = 0.001179
grad_step = 000383, loss = 0.001188
grad_step = 000384, loss = 0.001208
grad_step = 000385, loss = 0.001233
grad_step = 000386, loss = 0.001279
grad_step = 000387, loss = 0.001294
grad_step = 000388, loss = 0.001308
grad_step = 000389, loss = 0.001279
grad_step = 000390, loss = 0.001255
grad_step = 000391, loss = 0.001298
grad_step = 000392, loss = 0.001315
grad_step = 000393, loss = 0.001323
grad_step = 000394, loss = 0.001198
grad_step = 000395, loss = 0.001133
grad_step = 000396, loss = 0.001167
grad_step = 000397, loss = 0.001201
grad_step = 000398, loss = 0.001171
grad_step = 000399, loss = 0.001114
grad_step = 000400, loss = 0.001139
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001181
grad_step = 000402, loss = 0.001143
grad_step = 000403, loss = 0.001103
grad_step = 000404, loss = 0.001111
grad_step = 000405, loss = 0.001131
grad_step = 000406, loss = 0.001127
grad_step = 000407, loss = 0.001094
grad_step = 000408, loss = 0.001080
grad_step = 000409, loss = 0.001092
grad_step = 000410, loss = 0.001106
grad_step = 000411, loss = 0.001107
grad_step = 000412, loss = 0.001089
grad_step = 000413, loss = 0.001076
grad_step = 000414, loss = 0.001075
grad_step = 000415, loss = 0.001084
grad_step = 000416, loss = 0.001093
grad_step = 000417, loss = 0.001092
grad_step = 000418, loss = 0.001086
grad_step = 000419, loss = 0.001076
grad_step = 000420, loss = 0.001066
grad_step = 000421, loss = 0.001062
grad_step = 000422, loss = 0.001064
grad_step = 000423, loss = 0.001078
grad_step = 000424, loss = 0.001107
grad_step = 000425, loss = 0.001180
grad_step = 000426, loss = 0.001260
grad_step = 000427, loss = 0.001418
grad_step = 000428, loss = 0.001337
grad_step = 000429, loss = 0.001262
grad_step = 000430, loss = 0.001185
grad_step = 000431, loss = 0.001220
grad_step = 000432, loss = 0.001235
grad_step = 000433, loss = 0.001089
grad_step = 000434, loss = 0.001036
grad_step = 000435, loss = 0.001108
grad_step = 000436, loss = 0.001127
grad_step = 000437, loss = 0.001097
grad_step = 000438, loss = 0.001103
grad_step = 000439, loss = 0.001088
grad_step = 000440, loss = 0.001047
grad_step = 000441, loss = 0.001013
grad_step = 000442, loss = 0.001031
grad_step = 000443, loss = 0.001065
grad_step = 000444, loss = 0.001052
grad_step = 000445, loss = 0.001023
grad_step = 000446, loss = 0.001019
grad_step = 000447, loss = 0.001015
grad_step = 000448, loss = 0.001005
grad_step = 000449, loss = 0.001004
grad_step = 000450, loss = 0.001008
grad_step = 000451, loss = 0.001002
grad_step = 000452, loss = 0.001002
grad_step = 000453, loss = 0.001000
grad_step = 000454, loss = 0.000987
grad_step = 000455, loss = 0.000982
grad_step = 000456, loss = 0.000986
grad_step = 000457, loss = 0.000989
grad_step = 000458, loss = 0.000993
grad_step = 000459, loss = 0.001005
grad_step = 000460, loss = 0.001006
grad_step = 000461, loss = 0.001015
grad_step = 000462, loss = 0.001021
grad_step = 000463, loss = 0.001040
grad_step = 000464, loss = 0.001035
grad_step = 000465, loss = 0.001041
grad_step = 000466, loss = 0.001005
grad_step = 000467, loss = 0.000974
grad_step = 000468, loss = 0.000948
grad_step = 000469, loss = 0.000944
grad_step = 000470, loss = 0.000957
grad_step = 000471, loss = 0.000968
grad_step = 000472, loss = 0.000971
grad_step = 000473, loss = 0.000953
grad_step = 000474, loss = 0.000934
grad_step = 000475, loss = 0.000926
grad_step = 000476, loss = 0.000930
grad_step = 000477, loss = 0.000935
grad_step = 000478, loss = 0.000933
grad_step = 000479, loss = 0.000928
grad_step = 000480, loss = 0.000916
grad_step = 000481, loss = 0.000907
grad_step = 000482, loss = 0.000903
grad_step = 000483, loss = 0.000904
grad_step = 000484, loss = 0.000907
grad_step = 000485, loss = 0.000912
grad_step = 000486, loss = 0.000921
grad_step = 000487, loss = 0.000932
grad_step = 000488, loss = 0.000959
grad_step = 000489, loss = 0.000981
grad_step = 000490, loss = 0.001044
grad_step = 000491, loss = 0.001032
grad_step = 000492, loss = 0.001056
grad_step = 000493, loss = 0.001013
grad_step = 000494, loss = 0.001053
grad_step = 000495, loss = 0.001161
grad_step = 000496, loss = 0.001345
grad_step = 000497, loss = 0.001348
grad_step = 000498, loss = 0.001215
grad_step = 000499, loss = 0.000957
grad_step = 000500, loss = 0.000887
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001005
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

  date_run                              2020-05-15 21:13:22.487258
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.218861
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 21:13:22.493458
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.12238
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 21:13:22.500734
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.122064
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 21:13:22.506087
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.859601
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
0   2020-05-15 21:12:54.211314  ...    mean_absolute_error
1   2020-05-15 21:12:54.217818  ...     mean_squared_error
2   2020-05-15 21:12:54.222616  ...  median_absolute_error
3   2020-05-15 21:12:54.230316  ...               r2_score
4   2020-05-15 21:13:03.823827  ...    mean_absolute_error
5   2020-05-15 21:13:03.828773  ...     mean_squared_error
6   2020-05-15 21:13:03.832022  ...  median_absolute_error
7   2020-05-15 21:13:03.835855  ...               r2_score
8   2020-05-15 21:13:22.487258  ...    mean_absolute_error
9   2020-05-15 21:13:22.493458  ...     mean_squared_error
10  2020-05-15 21:13:22.500734  ...  median_absolute_error
11  2020-05-15 21:13:22.506087  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd877be2b38> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 33%|███▎      | 3235840/9912422 [00:00<00:00, 32245878.19it/s]9920512it [00:00, 34744597.18it/s]                             
0it [00:00, ?it/s]32768it [00:00, 711709.60it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 451962.07it/s]1654784it [00:00, 11584885.60it/s]                         
0it [00:00, ?it/s]8192it [00:00, 210072.93it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd82a59de10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd877bed908> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd82a59de10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd877bedfd0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd82735d470> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd877bed908> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd82a59de10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd877bedfd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd82735d470> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd8273e90b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f27f327c208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=e2a219d1989563717b374538a7f18ea35e55d803fa239daae11bc9149cb4bb6f
  Stored in directory: /tmp/pip-ephem-wheel-cache-5qwort9m/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f2792fa6d30> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 2s
 1294336/17464789 [=>............................] - ETA: 0s
 4636672/17464789 [======>.......................] - ETA: 0s
11591680/17464789 [==================>...........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 21:14:49.337837: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 21:14:49.342476: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-05-15 21:14:49.342616: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555eb8843200 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 21:14:49.342629: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4366 - accuracy: 0.5150
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5363 - accuracy: 0.5085 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5695 - accuracy: 0.5063
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6820 - accuracy: 0.4990
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6237 - accuracy: 0.5028
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6411 - accuracy: 0.5017
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6800 - accuracy: 0.4991
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6734 - accuracy: 0.4996
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6850 - accuracy: 0.4988
11000/25000 [============>.................] - ETA: 3s - loss: 7.6624 - accuracy: 0.5003
12000/25000 [=============>................] - ETA: 3s - loss: 7.6372 - accuracy: 0.5019
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6619 - accuracy: 0.5003
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6732 - accuracy: 0.4996
15000/25000 [=================>............] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6551 - accuracy: 0.5008
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6639 - accuracy: 0.5002
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6564 - accuracy: 0.5007
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6569 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6590 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6710 - accuracy: 0.4997
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6750 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6686 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6634 - accuracy: 0.5002
25000/25000 [==============================] - 7s 284us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 21:15:03.206809
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 21:15:03.206809  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<29:03:30, 8.24kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<20:34:31, 11.6kB/s].vector_cache/glove.6B.zip:   0%|          | 172k/862M [00:01<14:28:02, 16.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 713k/862M [00:01<10:08:18, 23.6kB/s].vector_cache/glove.6B.zip:   0%|          | 2.28M/862M [00:01<7:05:19, 33.7kB/s].vector_cache/glove.6B.zip:   1%|          | 5.58M/862M [00:01<4:56:43, 48.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.81M/862M [00:01<3:26:46, 68.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.1M/862M [00:01<2:24:07, 98.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.4M/862M [00:01<1:40:28, 140kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.5M/862M [00:01<1:10:05, 200kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.9M/862M [00:02<48:54, 285kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.2M/862M [00:02<34:10, 405kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.5M/862M [00:02<23:53, 577kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.7M/862M [00:02<16:44, 819kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:02<11:45, 1.16MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.2M/862M [00:02<08:17, 1.64MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:03<06:24, 2.11MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:03<04:35, 2.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<10:46, 1.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<09:59, 1.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:05<07:31, 1.78MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:05<05:23, 2.48MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:07<38:05, 351kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<28:13, 473kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<20:04, 664kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<16:49, 789kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<14:34, 912kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<10:53, 1.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:09<07:46, 1.70MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:11<1:39:24, 133kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:11<1:10:55, 186kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.9M/862M [00:11<49:53, 264kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:13<37:54, 347kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<27:53, 471kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:13<19:46, 663kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:15<16:52, 775kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:15<13:11, 991kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<09:33, 1.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:17<09:44, 1.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:17<08:09, 1.60MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.3M/862M [00:17<06:01, 2.15MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:19<07:16, 1.78MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:19<06:23, 2.02MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:19<04:48, 2.69MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<06:23, 2.01MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:21<05:48, 2.22MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:21<04:23, 2.93MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<06:05, 2.10MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:23<05:34, 2.30MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:23<04:13, 3.03MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<05:57, 2.14MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:25<05:31, 2.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.7M/862M [00:25<04:11, 3.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:54, 2.14MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:26, 2.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<04:05, 3.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:51, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:25, 2.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<04:18, 2.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<03:20, 3.76MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<02:45, 4.54MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<08:48, 1.42MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<09:09, 1.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<07:02, 1.78MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<05:06, 2.44MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:54, 1.58MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<06:56, 1.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<05:08, 2.41MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<03:46, 3.29MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<26:22, 470kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<18:56, 654kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<13:22, 922kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<15:17, 806kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<13:21, 922kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<09:54, 1.24MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<07:12, 1.70MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<08:36, 1.42MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<08:49, 1.39MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<06:47, 1.80MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 129M/862M [00:39<04:57, 2.47MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<07:22, 1.65MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<06:30, 1.87MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<04:51, 2.50MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<06:02, 2.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<05:27, 2.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<04:10, 2.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<03:07, 3.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<08:41, 1.39MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<11:51, 1.02MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<08:53, 1.35MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<06:24, 1.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<08:18, 1.44MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<07:10, 1.67MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<05:36, 2.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<04:25, 2.70MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<03:13, 3.69MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<18:32, 642kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<15:15, 779kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<11:13, 1.06MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<09:48, 1.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<08:15, 1.43MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<06:05, 1.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<06:44, 1.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<05:44, 2.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<04:41, 2.50MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<03:23, 3.45MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<16:34, 706kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<14:00, 834kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<10:23, 1.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:45<1:03:24, 183kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:45<53:58, 215kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:45<40:42, 286kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:46<29:02, 400kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:46<20:34, 563kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:46<14:26, 799kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:47<17:22, 663kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:47<12:23, 928kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:47<09:00, 1.28MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:47<06:35, 1.74MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:49<20:13, 567kB/s] .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:49<25:45, 445kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:49<20:41, 554kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:50<15:51, 722kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:50<11:33, 990kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:50<08:14, 1.38MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:51<10:39, 1.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:51<08:43, 1.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:51<06:23, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:51<04:40, 2.42MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [02:40<2:42:23, 69.7kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [02:40<2:04:49, 90.7kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [02:40<1:29:37, 126kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [02:41<1:03:21, 178kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [02:41<44:24, 254kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [04:55<5:51:35, 32.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [04:55<4:15:21, 44.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [04:56<3:01:29, 62.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [04:56<2:07:27, 88.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [04:56<1:29:10, 126kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [04:56<1:02:51, 178kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [04:58<45:29, 244kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [04:58<35:50, 310kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [04:58<25:58, 428kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [04:59<18:39, 595kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [04:59<13:13, 837kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [05:00<13:31, 817kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [05:00<11:16, 979kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [05:00<08:24, 1.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [05:01<06:04, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [05:02<07:32, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [05:02<06:46, 1.62MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [05:02<05:03, 2.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [05:02<03:40, 2.97MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [05:04<14:11, 768kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [05:04<11:14, 969kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [05:04<08:07, 1.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [05:06<07:55, 1.37MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [05:06<07:58, 1.36MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [05:06<06:07, 1.77MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [05:06<04:28, 2.42MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [05:08<06:20, 1.70MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [05:08<05:31, 1.95MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [05:08<04:06, 2.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [05:10<05:24, 1.98MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [05:10<04:53, 2.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [05:10<03:38, 2.93MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [05:12<05:04, 2.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [05:12<04:38, 2.29MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [05:12<03:30, 3.02MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [05:14<04:57, 2.13MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [05:14<04:32, 2.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [05:14<03:26, 3.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [05:16<04:53, 2.15MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [05:16<04:29, 2.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [05:16<03:21, 3.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [05:18<04:50, 2.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [05:18<04:25, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [05:18<03:21, 3.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [05:20<04:47, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [05:20<04:25, 2.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [05:20<03:20, 3.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [05:22<04:46, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [05:22<04:24, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [05:22<03:20, 3.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [05:24<05:42, 1.79MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [05:25<14:09, 722kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [05:25<12:07, 843kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [05:25<08:56, 1.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [05:25<06:29, 1.57MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [06:02<1:03:02, 161kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [06:02<52:27, 194kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [06:02<39:12, 259kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [06:02<27:55, 363kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [06:02<19:39, 514kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [06:18<46:28, 217kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [06:18<42:36, 237kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [06:18<32:13, 313kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [06:18<23:03, 437kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [06:18<16:20, 615kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [06:19<11:29, 870kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [06:19<10:58, 910kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [06:19<07:47, 1.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [06:21<10:04, 985kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [06:21<08:52, 1.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [06:21<06:38, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [06:23<06:17, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [06:23<05:21, 1.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [06:23<04:01, 2.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [06:23<02:57, 3.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [06:25<08:27, 1.16MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [06:25<08:00, 1.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [06:25<06:04, 1.61MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [06:25<04:22, 2.22MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [06:27<06:44, 1.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [06:27<05:43, 1.69MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [06:27<04:13, 2.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [06:29<05:21, 1.80MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [06:29<05:45, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [06:29<04:31, 2.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [06:29<03:16, 2.93MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [06:31<1:52:55, 84.8kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [06:31<1:19:58, 120kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [06:31<56:04, 170kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [06:33<41:21, 230kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [06:33<29:54, 318kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [06:33<21:07, 449kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [06:35<16:57, 556kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [06:35<16:38, 567kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [06:35<12:15, 769kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [06:35<08:43, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [06:37<08:35, 1.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [06:37<06:59, 1.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [06:37<05:07, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [06:39<05:45, 1.62MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [06:39<04:59, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [06:39<03:41, 2.52MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [06:41<04:44, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [06:41<05:12, 1.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [06:41<04:07, 2.24MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [06:43<04:21, 2.10MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [06:43<03:59, 2.30MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [06:43<03:01, 3.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [06:45<04:13, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [06:45<03:54, 2.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [06:45<02:56, 3.09MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [06:47<04:09, 2.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [06:47<03:50, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [06:47<02:52, 3.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [06:49<04:08, 2.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [06:49<03:49, 2.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [06:49<02:53, 3.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [06:51<04:07, 2.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [06:51<03:47, 2.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [06:51<02:52, 3.08MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [06:53<04:05, 2.15MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [06:53<03:46, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [06:53<02:49, 3.12MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [06:53<18:22, 478kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [06:53<12:57, 676kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [06:59<14:29, 601kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [06:59<17:19, 503kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [06:59<14:16, 610kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [06:59<10:25, 835kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [06:59<07:31, 1.15MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [07:01<07:00, 1.23MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [07:01<05:41, 1.52MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [07:01<04:24, 1.96MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [07:01<03:11, 2.69MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [07:03<06:34, 1.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [07:03<05:32, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [07:03<04:04, 2.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [07:05<04:44, 1.79MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [07:05<05:09, 1.65MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [07:05<04:00, 2.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [07:05<02:59, 2.83MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [07:07<04:19, 1.95MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [07:07<03:54, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [07:07<02:57, 2.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [07:07<02:11, 3.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [07:09<18:17, 457kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [07:09<14:06, 593kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [07:09<10:11, 820kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [07:09<07:13, 1.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [07:11<10:37, 781kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [07:11<08:39, 958kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [07:11<06:19, 1.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [07:11<04:32, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [07:13<08:41, 948kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [07:13<07:14, 1.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [07:13<05:19, 1.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [07:13<03:50, 2.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [07:15<08:14, 991kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [07:15<06:52, 1.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [07:15<05:02, 1.61MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [07:15<03:38, 2.23MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [07:17<08:36, 941kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [07:17<06:59, 1.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [07:17<05:06, 1.58MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [07:17<03:39, 2.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [07:19<17:34, 456kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [07:19<13:13, 606kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [07:19<09:27, 845kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [07:21<08:16, 960kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [07:21<06:42, 1.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [07:21<04:54, 1.61MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [07:23<05:06, 1.54MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [07:23<04:28, 1.76MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [07:23<03:20, 2.35MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [07:25<03:59, 1.96MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [07:25<03:40, 2.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [07:25<02:47, 2.79MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [07:25<02:03, 3.75MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [07:27<10:35, 731kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [07:27<08:28, 913kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [07:27<06:09, 1.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [07:27<04:24, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [07:29<09:58, 770kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [07:29<07:53, 972kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [07:29<05:42, 1.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [07:31<05:34, 1.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [07:31<04:45, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [07:31<03:32, 2.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [07:33<04:05, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [07:33<03:41, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [07:33<02:45, 2.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [07:33<02:01, 3.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [07:35<12:41, 589kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [07:35<09:38, 774kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [07:35<06:54, 1.08MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [07:37<06:34, 1.12MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [07:37<05:21, 1.38MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [07:37<03:54, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [07:39<04:27, 1.64MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [07:39<03:52, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [07:39<02:53, 2.52MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [07:40<03:44, 1.94MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [07:41<03:20, 2.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [07:41<02:29, 2.91MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [07:41<01:51, 3.89MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [07:42<08:57, 803kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [07:43<06:55, 1.04MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [07:43<05:06, 1.40MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [07:43<03:40, 1.95MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [07:44<06:02, 1.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [07:45<04:59, 1.43MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [07:45<03:38, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [07:46<04:07, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [07:47<03:37, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [07:47<02:42, 2.60MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [07:48<03:32, 1.97MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [07:49<03:11, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [07:49<02:24, 2.89MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [07:50<03:18, 2.09MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [07:50<03:01, 2.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [07:51<02:17, 3.00MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [07:52<03:12, 2.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [07:52<02:56, 2.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [07:53<02:12, 3.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [07:54<03:08, 2.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [07:54<02:53, 2.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [07:55<02:11, 3.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [07:56<03:06, 2.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [07:56<03:33, 1.88MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [07:56<02:46, 2.41MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [07:57<02:04, 3.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [07:58<03:24, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [07:58<03:04, 2.16MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [07:58<02:18, 2.86MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [08:00<03:09, 2.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [08:00<02:52, 2.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [08:00<02:10, 3.00MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [08:02<03:02, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [08:02<02:48, 2.31MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [08:02<02:07, 3.04MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [08:04<02:59, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [08:04<02:44, 2.34MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [08:04<02:04, 3.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [08:06<02:57, 2.16MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [08:06<02:43, 2.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [08:06<02:03, 3.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [08:08<02:55, 2.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [08:08<02:42, 2.33MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [08:08<02:00, 3.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [08:08<01:29, 4.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [08:10<1:14:58, 83.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [08:10<53:03, 117kB/s]   .vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [08:10<37:09, 167kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [08:12<27:18, 226kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [08:12<19:44, 312kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [08:12<13:53, 441kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [08:14<11:05, 549kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [08:14<08:23, 725kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [08:14<06:00, 1.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [08:16<05:52, 1.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [08:16<06:36, 911kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [08:16<05:14, 1.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [08:17<03:49, 1.57MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [08:17<02:44, 2.18MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [08:18<5:27:11, 18.2kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [08:18<3:49:25, 25.9kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [08:18<2:40:03, 37.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [08:20<1:52:42, 52.2kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [08:20<1:19:25, 74.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [08:20<55:29, 105kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [08:22<39:59, 145kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [08:22<28:33, 203kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [08:22<20:02, 289kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [08:24<15:17, 376kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [08:24<11:16, 509kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [08:24<07:58, 716kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [08:26<06:54, 822kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [08:26<05:18, 1.07MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [08:26<03:51, 1.47MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [08:26<02:45, 2.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [08:28<24:31, 229kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [08:28<17:42, 316kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [08:28<12:29, 447kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [08:30<09:59, 555kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [08:30<07:33, 732kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [08:30<05:23, 1.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [08:32<05:25, 1.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [08:32<06:03, 902kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [08:33<04:48, 1.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [08:33<03:27, 1.57MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [08:33<02:33, 2.12MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [08:34<04:09, 1.30MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [08:34<03:48, 1.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [08:35<02:50, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [08:35<02:06, 2.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [08:35<01:34, 3.40MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [08:36<17:40, 302kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [08:36<13:11, 404kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [08:37<09:25, 564kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [08:37<06:38, 796kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [08:38<07:24, 711kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [08:38<05:58, 881kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [08:39<04:20, 1.21MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [08:39<03:08, 1.66MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [08:40<03:56, 1.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [08:40<03:35, 1.45MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [08:41<02:42, 1.91MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [08:41<01:58, 2.62MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [08:42<04:05, 1.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [08:42<03:38, 1.41MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [08:42<02:43, 1.87MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [08:43<01:58, 2.58MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [08:44<06:26, 785kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [08:44<05:13, 967kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [08:44<03:48, 1.33MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [08:45<02:44, 1.83MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [08:46<04:04, 1.23MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [08:46<03:32, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [08:46<02:37, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [08:47<01:54, 2.59MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [08:48<04:08, 1.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [08:48<03:33, 1.38MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [08:48<02:37, 1.86MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [08:49<01:54, 2.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [08:50<04:21, 1.11MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [08:50<03:41, 1.31MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [08:50<02:43, 1.77MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [08:52<02:50, 1.69MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [08:52<02:29, 1.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [08:52<01:51, 2.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [08:54<02:20, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [08:54<02:09, 2.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [08:54<01:35, 2.93MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [08:54<01:13, 3.81MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [08:56<03:19, 1.40MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [08:56<03:04, 1.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [08:56<02:18, 2.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [08:56<01:41, 2.71MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [08:58<02:41, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [08:58<02:29, 1.84MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [08:58<01:53, 2.41MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [09:00<02:11, 2.06MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [09:00<02:06, 2.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [09:00<01:36, 2.80MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [09:02<01:59, 2.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [09:02<01:51, 2.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [09:02<01:23, 3.18MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [09:04<02:00, 2.17MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [09:04<01:51, 2.35MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [09:04<01:24, 3.09MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [09:06<01:59, 2.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [09:06<01:49, 2.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [09:06<01:23, 3.08MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [09:08<01:57, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [09:08<01:48, 2.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [09:08<01:21, 3.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [09:08<01:01, 4.10MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [09:10<05:37, 741kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [09:10<05:05, 817kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [09:10<03:49, 1.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [09:10<02:48, 1.47MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [09:10<02:03, 2.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 616M/862M [09:12<02:57, 1.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [09:12<02:42, 1.51MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [09:12<02:03, 1.98MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [09:12<01:31, 2.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [09:14<02:24, 1.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [09:14<02:18, 1.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [09:14<01:46, 2.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [09:14<01:20, 2.99MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [09:14<01:01, 3.89MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [09:16<06:43, 589kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [09:16<05:29, 720kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [09:16<04:01, 979kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [09:16<02:52, 1.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [09:18<03:10, 1.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [09:18<02:48, 1.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [09:18<02:06, 1.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [09:18<01:34, 2.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [09:18<01:12, 3.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [09:20<03:02, 1.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [09:20<03:00, 1.27MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [09:20<02:16, 1.67MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [09:20<01:42, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [09:20<01:15, 3.00MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [09:22<03:35, 1.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [09:22<03:02, 1.23MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [09:22<02:20, 1.60MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [09:22<01:42, 2.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [09:22<01:15, 2.92MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [09:24<04:33, 808kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [09:24<03:47, 969kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [09:24<02:47, 1.32MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [09:24<02:00, 1.81MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [09:26<02:39, 1.36MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [09:26<02:26, 1.48MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [09:26<01:50, 1.95MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [09:26<01:19, 2.68MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [09:28<03:15, 1.09MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [09:28<02:47, 1.27MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [09:28<02:03, 1.72MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [09:28<01:29, 2.35MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [09:30<02:40, 1.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [09:30<02:17, 1.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [09:30<01:42, 2.01MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [09:30<01:15, 2.73MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [09:32<02:22, 1.43MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [09:32<02:08, 1.59MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [09:32<01:36, 2.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [09:32<01:08, 2.91MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [09:34<06:59, 478kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [09:34<05:21, 622kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [09:34<03:48, 868kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [09:34<02:44, 1.20MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [09:36<02:58, 1.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [09:36<02:35, 1.26MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [09:36<01:54, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [09:36<01:24, 2.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [09:38<01:56, 1.65MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [09:38<01:48, 1.77MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [09:38<01:22, 2.31MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [09:38<00:59, 3.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [09:40<03:12, 976kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [09:40<02:41, 1.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [09:40<01:58, 1.57MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [09:40<01:25, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [09:42<03:01, 1.01MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [09:42<02:32, 1.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [09:42<01:52, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [09:42<01:22, 2.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [09:44<01:51, 1.61MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [09:44<01:41, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [09:44<01:16, 2.33MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [09:44<00:54, 3.20MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [09:46<06:11, 472kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [09:46<04:41, 621kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [09:46<03:21, 865kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [09:46<02:22, 1.21MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [09:48<02:58, 958kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [09:48<02:26, 1.16MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [09:48<01:48, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [09:48<01:17, 2.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [09:50<02:24, 1.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [09:50<01:59, 1.39MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [09:50<01:28, 1.87MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [09:50<01:04, 2.53MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [09:52<01:42, 1.58MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [09:52<01:33, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [09:52<01:16, 2.13MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [09:52<00:56, 2.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [09:53<01:20, 1.97MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [09:54<01:14, 2.13MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [09:54<00:55, 2.84MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [09:55<01:12, 2.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [09:56<01:22, 1.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [09:56<01:04, 2.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [09:56<00:48, 3.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [09:57<01:11, 2.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [09:58<01:05, 2.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [09:58<00:48, 3.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [09:58<00:35, 4.11MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [09:59<09:28, 257kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [10:00<06:52, 354kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [10:00<04:49, 499kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [10:01<03:53, 610kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [10:01<02:57, 800kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [10:02<02:06, 1.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [10:03<01:59, 1.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [10:03<01:39, 1.38MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [10:04<01:12, 1.87MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [10:05<01:18, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [10:05<01:08, 1.94MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [10:06<00:51, 2.58MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [10:07<01:05, 1.98MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [10:07<00:59, 2.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [10:08<00:43, 2.93MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [10:08<00:32, 3.93MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [10:09<05:45, 364kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [10:09<04:15, 490kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [10:09<02:59, 690kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [10:10<02:05, 971kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [10:11<04:22, 463kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [10:11<03:16, 618kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [10:11<02:18, 862kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [10:13<02:02, 958kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [10:13<01:35, 1.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [10:13<01:09, 1.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [10:15<01:14, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [10:15<01:03, 1.78MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [10:15<00:47, 2.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [10:15<00:34, 3.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [10:17<01:54, 958kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [10:17<01:31, 1.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [10:17<01:06, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [10:19<01:08, 1.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [10:19<00:59, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [10:19<00:43, 2.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [10:19<00:31, 3.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [10:21<01:31, 1.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [10:21<01:14, 1.34MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [10:21<00:53, 1.84MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [10:23<00:58, 1.66MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [10:23<00:50, 1.90MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [10:23<00:38, 2.48MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [10:23<00:27, 3.40MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [10:25<02:33, 604kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [10:25<01:56, 791kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [10:25<01:22, 1.10MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [10:27<01:17, 1.15MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [10:27<01:01, 1.44MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [10:27<00:44, 1.96MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [10:27<00:31, 2.70MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [10:29<06:01, 234kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [10:29<04:20, 323kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [10:29<03:00, 456kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [10:31<02:22, 565kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [10:31<01:47, 745kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [10:31<01:15, 1.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [10:33<01:09, 1.10MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [10:33<00:56, 1.35MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [10:33<00:40, 1.84MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [10:35<00:44, 1.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [10:35<00:38, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [10:35<00:27, 2.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [10:37<00:35, 1.94MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [10:37<00:30, 2.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [10:37<00:23, 2.84MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [10:37<00:17, 3.78MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [10:39<00:37, 1.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [10:39<00:34, 1.82MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [10:39<00:26, 2.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [10:41<00:28, 2.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [10:41<00:27, 2.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [10:41<00:22, 2.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [10:41<00:16, 3.52MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [10:43<00:27, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [10:43<00:27, 2.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [10:43<00:20, 2.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [10:45<00:23, 2.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [10:45<00:22, 2.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [10:45<00:16, 2.93MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [10:45<00:12, 3.96MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [10:47<01:28, 537kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [10:47<01:07, 694kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [10:47<00:47, 963kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [10:47<00:32, 1.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [10:49<00:53, 804kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [10:49<00:43, 996kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [10:49<00:30, 1.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [10:49<00:20, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [10:51<01:12, 539kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [10:51<00:54, 713kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [10:51<00:37, 992kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [10:53<00:32, 1.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [10:53<00:29, 1.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [10:53<00:21, 1.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [10:53<00:15, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [10:55<00:20, 1.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [10:55<00:17, 1.78MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [10:55<00:12, 2.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [10:57<00:14, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [10:57<00:12, 2.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [10:57<00:08, 2.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [10:58<00:10, 2.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [10:59<00:09, 2.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [10:59<00:06, 3.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [11:00<00:08, 2.12MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [11:01<00:07, 2.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [11:01<00:05, 3.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [11:02<00:06, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [11:03<00:05, 2.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [11:03<00:04, 3.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [11:04<00:04, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [11:04<00:04, 2.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [11:05<00:02, 3.06MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [11:06<00:02, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [11:06<00:02, 2.32MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [11:07<00:01, 3.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [11:08<00:00, 2.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [11:08<00:00, 2.42MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [11:09<00:00, 3.11MB/s].vector_cache/glove.6B.zip: 862MB [11:09, 1.29MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 820/400000 [00:00<00:48, 8195.55it/s]  0%|          | 1645/400000 [00:00<00:48, 8209.62it/s]  1%|          | 2460/400000 [00:00<00:48, 8189.38it/s]  1%|          | 3283/400000 [00:00<00:48, 8200.01it/s]  1%|          | 3982/400000 [00:00<00:50, 7793.63it/s]  1%|          | 4802/400000 [00:00<00:49, 7911.16it/s]  1%|▏         | 5623/400000 [00:00<00:49, 7997.00it/s]  2%|▏         | 6414/400000 [00:00<00:49, 7968.53it/s]  2%|▏         | 7239/400000 [00:00<00:48, 8050.52it/s]  2%|▏         | 8064/400000 [00:01<00:48, 8109.02it/s]  2%|▏         | 8877/400000 [00:01<00:48, 8113.49it/s]  2%|▏         | 9708/400000 [00:01<00:47, 8169.70it/s]  3%|▎         | 10518/400000 [00:01<00:47, 8146.75it/s]  3%|▎         | 11338/400000 [00:01<00:47, 8162.13it/s]  3%|▎         | 12169/400000 [00:01<00:47, 8204.43it/s]  3%|▎         | 13002/400000 [00:01<00:46, 8238.79it/s]  3%|▎         | 13824/400000 [00:01<00:47, 8216.09it/s]  4%|▎         | 14644/400000 [00:01<00:46, 8200.69it/s]  4%|▍         | 15473/400000 [00:01<00:46, 8224.68it/s]  4%|▍         | 16299/400000 [00:02<00:46, 8233.84it/s]  4%|▍         | 17128/400000 [00:02<00:46, 8248.00it/s]  4%|▍         | 17953/400000 [00:02<00:46, 8138.78it/s]  5%|▍         | 18773/400000 [00:02<00:46, 8154.40it/s]  5%|▍         | 19597/400000 [00:02<00:46, 8178.91it/s]  5%|▌         | 20415/400000 [00:02<00:47, 7925.76it/s]  5%|▌         | 21240/400000 [00:02<00:47, 8020.16it/s]  6%|▌         | 22073/400000 [00:02<00:46, 8109.62it/s]  6%|▌         | 22886/400000 [00:02<00:46, 8107.33it/s]  6%|▌         | 23715/400000 [00:02<00:46, 8146.41it/s]  6%|▌         | 24543/400000 [00:03<00:45, 8184.73it/s]  6%|▋         | 25373/400000 [00:03<00:45, 8217.93it/s]  7%|▋         | 26200/400000 [00:03<00:45, 8230.77it/s]  7%|▋         | 27024/400000 [00:03<00:45, 8224.28it/s]  7%|▋         | 27847/400000 [00:03<00:45, 8184.63it/s]  7%|▋         | 28666/400000 [00:03<00:45, 8159.41it/s]  7%|▋         | 29491/400000 [00:03<00:45, 8186.05it/s]  8%|▊         | 30323/400000 [00:03<00:44, 8223.22it/s]  8%|▊         | 31147/400000 [00:03<00:44, 8226.38it/s]  8%|▊         | 31973/400000 [00:03<00:44, 8235.56it/s]  8%|▊         | 32804/400000 [00:04<00:44, 8254.98it/s]  8%|▊         | 33630/400000 [00:04<00:45, 8087.61it/s]  9%|▊         | 34440/400000 [00:04<00:46, 7899.98it/s]  9%|▉         | 35232/400000 [00:04<00:46, 7777.66it/s]  9%|▉         | 36047/400000 [00:04<00:46, 7884.23it/s]  9%|▉         | 36839/400000 [00:04<00:46, 7894.04it/s]  9%|▉         | 37670/400000 [00:04<00:45, 8013.43it/s] 10%|▉         | 38494/400000 [00:04<00:44, 8079.13it/s] 10%|▉         | 39314/400000 [00:04<00:44, 8114.73it/s] 10%|█         | 40132/400000 [00:04<00:44, 8132.87it/s] 10%|█         | 40946/400000 [00:05<00:44, 8125.76it/s] 10%|█         | 41771/400000 [00:05<00:43, 8161.04it/s] 11%|█         | 42601/400000 [00:05<00:43, 8199.97it/s] 11%|█         | 43422/400000 [00:05<00:43, 8166.96it/s] 11%|█         | 44253/400000 [00:05<00:43, 8208.55it/s] 11%|█▏        | 45075/400000 [00:05<00:43, 8198.96it/s] 11%|█▏        | 45907/400000 [00:05<00:43, 8234.26it/s] 12%|█▏        | 46731/400000 [00:05<00:42, 8229.82it/s] 12%|█▏        | 47555/400000 [00:05<00:42, 8206.79it/s] 12%|█▏        | 48376/400000 [00:05<00:42, 8204.06it/s] 12%|█▏        | 49197/400000 [00:06<00:42, 8201.38it/s] 13%|█▎        | 50018/400000 [00:06<00:42, 8164.85it/s] 13%|█▎        | 50846/400000 [00:06<00:42, 8198.76it/s] 13%|█▎        | 51666/400000 [00:06<00:42, 8163.85it/s] 13%|█▎        | 52494/400000 [00:06<00:42, 8195.99it/s] 13%|█▎        | 53314/400000 [00:06<00:42, 8183.63it/s] 14%|█▎        | 54142/400000 [00:06<00:42, 8211.43it/s] 14%|█▎        | 54975/400000 [00:06<00:41, 8244.42it/s] 14%|█▍        | 55800/400000 [00:06<00:41, 8231.54it/s] 14%|█▍        | 56630/400000 [00:06<00:41, 8249.17it/s] 14%|█▍        | 57464/400000 [00:07<00:41, 8275.85it/s] 15%|█▍        | 58292/400000 [00:07<00:41, 8263.62it/s] 15%|█▍        | 59119/400000 [00:07<00:41, 8142.46it/s] 15%|█▍        | 59944/400000 [00:07<00:41, 8172.76it/s] 15%|█▌        | 60762/400000 [00:07<00:43, 7714.96it/s] 15%|█▌        | 61540/400000 [00:07<00:43, 7695.18it/s] 16%|█▌        | 62362/400000 [00:07<00:43, 7843.95it/s] 16%|█▌        | 63187/400000 [00:07<00:42, 7960.54it/s] 16%|█▌        | 64010/400000 [00:07<00:41, 8036.85it/s] 16%|█▌        | 64831/400000 [00:07<00:41, 8086.49it/s] 16%|█▋        | 65666/400000 [00:08<00:40, 8161.02it/s] 17%|█▋        | 66498/400000 [00:08<00:40, 8207.59it/s] 17%|█▋        | 67334/400000 [00:08<00:40, 8251.29it/s] 17%|█▋        | 68160/400000 [00:08<00:40, 8250.60it/s] 17%|█▋        | 68993/400000 [00:08<00:40, 8272.36it/s] 17%|█▋        | 69828/400000 [00:08<00:39, 8293.72it/s] 18%|█▊        | 70658/400000 [00:08<00:39, 8289.02it/s] 18%|█▊        | 71495/400000 [00:08<00:39, 8312.56it/s] 18%|█▊        | 72327/400000 [00:08<00:40, 8187.85it/s] 18%|█▊        | 73147/400000 [00:08<00:39, 8177.10it/s] 18%|█▊        | 73986/400000 [00:09<00:39, 8238.37it/s] 19%|█▊        | 74811/400000 [00:09<00:39, 8222.12it/s] 19%|█▉        | 75643/400000 [00:09<00:39, 8250.61it/s] 19%|█▉        | 76469/400000 [00:09<00:39, 8241.93it/s] 19%|█▉        | 77294/400000 [00:09<00:39, 8217.52it/s] 20%|█▉        | 78125/400000 [00:09<00:39, 8244.28it/s] 20%|█▉        | 78954/400000 [00:09<00:38, 8256.48it/s] 20%|█▉        | 79783/400000 [00:09<00:38, 8265.84it/s] 20%|██        | 80610/400000 [00:09<00:39, 8125.32it/s] 20%|██        | 81424/400000 [00:09<00:39, 8011.96it/s] 21%|██        | 82245/400000 [00:10<00:39, 8069.70it/s] 21%|██        | 83075/400000 [00:10<00:38, 8135.52it/s] 21%|██        | 83893/400000 [00:10<00:38, 8148.59it/s] 21%|██        | 84714/400000 [00:10<00:38, 8164.00it/s] 21%|██▏       | 85544/400000 [00:10<00:38, 8201.91it/s] 22%|██▏       | 86365/400000 [00:10<00:38, 8050.24it/s] 22%|██▏       | 87185/400000 [00:10<00:38, 8093.92it/s] 22%|██▏       | 88007/400000 [00:10<00:38, 8129.74it/s] 22%|██▏       | 88821/400000 [00:10<00:38, 8113.31it/s] 22%|██▏       | 89633/400000 [00:11<00:38, 8084.99it/s] 23%|██▎       | 90446/400000 [00:11<00:38, 8097.65it/s] 23%|██▎       | 91270/400000 [00:11<00:37, 8139.77it/s] 23%|██▎       | 92085/400000 [00:11<00:39, 7735.27it/s] 23%|██▎       | 92902/400000 [00:11<00:39, 7859.68it/s] 23%|██▎       | 93737/400000 [00:11<00:38, 7997.87it/s] 24%|██▎       | 94540/400000 [00:11<00:38, 7902.56it/s] 24%|██▍       | 95333/400000 [00:11<00:38, 7838.89it/s] 24%|██▍       | 96159/400000 [00:11<00:38, 7959.51it/s] 24%|██▍       | 96980/400000 [00:11<00:37, 8030.92it/s] 24%|██▍       | 97785/400000 [00:12<00:38, 7872.89it/s] 25%|██▍       | 98574/400000 [00:12<00:39, 7640.35it/s] 25%|██▍       | 99403/400000 [00:12<00:38, 7822.52it/s] 25%|██▌       | 100235/400000 [00:12<00:37, 7964.60it/s] 25%|██▌       | 101035/400000 [00:12<00:39, 7643.86it/s] 25%|██▌       | 101841/400000 [00:12<00:38, 7763.55it/s] 26%|██▌       | 102661/400000 [00:12<00:37, 7889.26it/s] 26%|██▌       | 103454/400000 [00:12<00:38, 7679.11it/s] 26%|██▌       | 104270/400000 [00:12<00:37, 7815.66it/s] 26%|██▋       | 105089/400000 [00:12<00:37, 7921.86it/s] 26%|██▋       | 105913/400000 [00:13<00:36, 8013.47it/s] 27%|██▋       | 106717/400000 [00:13<00:37, 7805.26it/s] 27%|██▋       | 107548/400000 [00:13<00:36, 7948.86it/s] 27%|██▋       | 108346/400000 [00:13<00:36, 7941.45it/s] 27%|██▋       | 109142/400000 [00:13<00:36, 7908.22it/s] 27%|██▋       | 109935/400000 [00:13<00:37, 7729.06it/s] 28%|██▊       | 110757/400000 [00:13<00:36, 7867.49it/s] 28%|██▊       | 111569/400000 [00:13<00:36, 7938.90it/s] 28%|██▊       | 112389/400000 [00:13<00:35, 8014.89it/s] 28%|██▊       | 113196/400000 [00:13<00:35, 8029.33it/s] 29%|██▊       | 114026/400000 [00:14<00:35, 8107.79it/s] 29%|██▊       | 114848/400000 [00:14<00:35, 8140.23it/s] 29%|██▉       | 115663/400000 [00:14<00:35, 8072.02it/s] 29%|██▉       | 116487/400000 [00:14<00:34, 8120.19it/s] 29%|██▉       | 117304/400000 [00:14<00:34, 8132.74it/s] 30%|██▉       | 118133/400000 [00:14<00:34, 8176.64it/s] 30%|██▉       | 118964/400000 [00:14<00:34, 8215.62it/s] 30%|██▉       | 119797/400000 [00:14<00:33, 8249.01it/s] 30%|███       | 120624/400000 [00:14<00:33, 8254.69it/s] 30%|███       | 121450/400000 [00:15<00:33, 8224.42it/s] 31%|███       | 122275/400000 [00:15<00:33, 8229.99it/s] 31%|███       | 123099/400000 [00:15<00:33, 8199.86it/s] 31%|███       | 123927/400000 [00:15<00:33, 8223.48it/s] 31%|███       | 124757/400000 [00:15<00:33, 8245.33it/s] 31%|███▏      | 125582/400000 [00:15<00:33, 8241.61it/s] 32%|███▏      | 126407/400000 [00:15<00:34, 7988.08it/s] 32%|███▏      | 127234/400000 [00:15<00:33, 8069.32it/s] 32%|███▏      | 128063/400000 [00:15<00:33, 8134.12it/s] 32%|███▏      | 128878/400000 [00:15<00:33, 8136.79it/s] 32%|███▏      | 129701/400000 [00:16<00:33, 8164.30it/s] 33%|███▎      | 130520/400000 [00:16<00:32, 8171.22it/s] 33%|███▎      | 131340/400000 [00:16<00:32, 8179.06it/s] 33%|███▎      | 132166/400000 [00:16<00:32, 8202.50it/s] 33%|███▎      | 132987/400000 [00:16<00:34, 7848.85it/s] 33%|███▎      | 133809/400000 [00:16<00:33, 7954.20it/s] 34%|███▎      | 134636/400000 [00:16<00:32, 8045.84it/s] 34%|███▍      | 135464/400000 [00:16<00:32, 8113.28it/s] 34%|███▍      | 136292/400000 [00:16<00:32, 8160.34it/s] 34%|███▍      | 137124/400000 [00:16<00:32, 8206.86it/s] 34%|███▍      | 137946/400000 [00:17<00:32, 8175.70it/s] 35%|███▍      | 138779/400000 [00:17<00:31, 8219.39it/s] 35%|███▍      | 139612/400000 [00:17<00:31, 8249.82it/s] 35%|███▌      | 140438/400000 [00:17<00:31, 8133.76it/s] 35%|███▌      | 141253/400000 [00:17<00:32, 8022.80it/s] 36%|███▌      | 142081/400000 [00:17<00:31, 8097.78it/s] 36%|███▌      | 142909/400000 [00:17<00:31, 8151.51it/s] 36%|███▌      | 143728/400000 [00:17<00:31, 8162.01it/s] 36%|███▌      | 144545/400000 [00:17<00:31, 7999.87it/s] 36%|███▋      | 145377/400000 [00:17<00:31, 8090.75it/s] 37%|███▋      | 146188/400000 [00:18<00:31, 8050.45it/s] 37%|███▋      | 147026/400000 [00:18<00:31, 8144.94it/s] 37%|███▋      | 147861/400000 [00:18<00:30, 8204.84it/s] 37%|███▋      | 148696/400000 [00:18<00:30, 8246.43it/s] 37%|███▋      | 149531/400000 [00:18<00:30, 8275.80it/s] 38%|███▊      | 150359/400000 [00:18<00:30, 8244.11it/s] 38%|███▊      | 151196/400000 [00:18<00:30, 8278.78it/s] 38%|███▊      | 152035/400000 [00:18<00:29, 8310.14it/s] 38%|███▊      | 152868/400000 [00:18<00:29, 8314.02it/s] 38%|███▊      | 153700/400000 [00:18<00:29, 8305.75it/s] 39%|███▊      | 154531/400000 [00:19<00:29, 8290.79it/s] 39%|███▉      | 155368/400000 [00:19<00:29, 8311.47it/s] 39%|███▉      | 156201/400000 [00:19<00:29, 8315.20it/s] 39%|███▉      | 157035/400000 [00:19<00:29, 8321.62it/s] 39%|███▉      | 157871/400000 [00:19<00:29, 8331.86it/s] 40%|███▉      | 158705/400000 [00:19<00:29, 8314.27it/s] 40%|███▉      | 159537/400000 [00:19<00:29, 8282.06it/s] 40%|████      | 160366/400000 [00:19<00:29, 8261.74it/s] 40%|████      | 161196/400000 [00:19<00:28, 8272.82it/s] 41%|████      | 162028/400000 [00:19<00:28, 8286.13it/s] 41%|████      | 162857/400000 [00:20<00:28, 8264.01it/s] 41%|████      | 163690/400000 [00:20<00:28, 8283.50it/s] 41%|████      | 164519/400000 [00:20<00:28, 8236.96it/s] 41%|████▏     | 165356/400000 [00:20<00:28, 8273.91it/s] 42%|████▏     | 166193/400000 [00:20<00:28, 8302.45it/s] 42%|████▏     | 167024/400000 [00:20<00:28, 8302.20it/s] 42%|████▏     | 167855/400000 [00:20<00:27, 8301.00it/s] 42%|████▏     | 168686/400000 [00:20<00:27, 8297.49it/s] 42%|████▏     | 169516/400000 [00:20<00:27, 8285.33it/s] 43%|████▎     | 170347/400000 [00:20<00:27, 8291.68it/s] 43%|████▎     | 171177/400000 [00:21<00:27, 8257.19it/s] 43%|████▎     | 172003/400000 [00:21<00:27, 8232.51it/s] 43%|████▎     | 172827/400000 [00:21<00:28, 8017.42it/s] 43%|████▎     | 173662/400000 [00:21<00:27, 8114.33it/s] 44%|████▎     | 174491/400000 [00:21<00:27, 8164.70it/s] 44%|████▍     | 175316/400000 [00:21<00:27, 8160.42it/s] 44%|████▍     | 176152/400000 [00:21<00:27, 8217.14it/s] 44%|████▍     | 176987/400000 [00:21<00:27, 8255.47it/s] 44%|████▍     | 177819/400000 [00:21<00:26, 8274.70it/s] 45%|████▍     | 178647/400000 [00:21<00:26, 8263.40it/s] 45%|████▍     | 179476/400000 [00:22<00:26, 8269.97it/s] 45%|████▌     | 180307/400000 [00:22<00:26, 8280.66it/s] 45%|████▌     | 181139/400000 [00:22<00:26, 8291.93it/s] 45%|████▌     | 181973/400000 [00:22<00:26, 8305.98it/s] 46%|████▌     | 182804/400000 [00:22<00:26, 8293.72it/s] 46%|████▌     | 183634/400000 [00:22<00:26, 8293.98it/s] 46%|████▌     | 184467/400000 [00:22<00:25, 8303.23it/s] 46%|████▋     | 185302/400000 [00:22<00:25, 8313.95it/s] 47%|████▋     | 186134/400000 [00:22<00:26, 8127.36it/s] 47%|████▋     | 186948/400000 [00:22<00:26, 7891.42it/s] 47%|████▋     | 187760/400000 [00:23<00:26, 7958.60it/s] 47%|████▋     | 188573/400000 [00:23<00:26, 8007.06it/s] 47%|████▋     | 189405/400000 [00:23<00:26, 8098.40it/s] 48%|████▊     | 190238/400000 [00:23<00:25, 8164.25it/s] 48%|████▊     | 191074/400000 [00:23<00:25, 8220.59it/s] 48%|████▊     | 191897/400000 [00:23<00:25, 8222.38it/s] 48%|████▊     | 192722/400000 [00:23<00:25, 8229.42it/s] 48%|████▊     | 193546/400000 [00:23<00:25, 8229.13it/s] 49%|████▊     | 194379/400000 [00:23<00:24, 8257.08it/s] 49%|████▉     | 195205/400000 [00:23<00:25, 8117.04it/s] 49%|████▉     | 196022/400000 [00:24<00:25, 8131.19it/s] 49%|████▉     | 196845/400000 [00:24<00:24, 8158.98it/s] 49%|████▉     | 197681/400000 [00:24<00:24, 8217.56it/s] 50%|████▉     | 198520/400000 [00:24<00:24, 8267.70it/s] 50%|████▉     | 199354/400000 [00:24<00:24, 8287.91it/s] 50%|█████     | 200184/400000 [00:24<00:24, 8285.73it/s] 50%|█████     | 201013/400000 [00:24<00:24, 8172.07it/s] 50%|█████     | 201844/400000 [00:24<00:24, 8210.77it/s] 51%|█████     | 202666/400000 [00:24<00:24, 8188.85it/s] 51%|█████     | 203486/400000 [00:25<00:24, 8124.82it/s] 51%|█████     | 204299/400000 [00:25<00:24, 8067.65it/s] 51%|█████▏    | 205121/400000 [00:25<00:24, 8111.70it/s] 51%|█████▏    | 205955/400000 [00:25<00:23, 8177.44it/s] 52%|█████▏    | 206792/400000 [00:25<00:23, 8233.09it/s] 52%|█████▏    | 207630/400000 [00:25<00:23, 8276.57it/s] 52%|█████▏    | 208458/400000 [00:25<00:23, 8276.20it/s] 52%|█████▏    | 209286/400000 [00:25<00:23, 8218.64it/s] 53%|█████▎    | 210117/400000 [00:25<00:23, 8244.74it/s] 53%|█████▎    | 210955/400000 [00:25<00:22, 8283.89it/s] 53%|█████▎    | 211785/400000 [00:26<00:22, 8286.95it/s] 53%|█████▎    | 212618/400000 [00:26<00:22, 8298.60it/s] 53%|█████▎    | 213448/400000 [00:26<00:22, 8269.20it/s] 54%|█████▎    | 214276/400000 [00:26<00:22, 8166.50it/s] 54%|█████▍    | 215093/400000 [00:26<00:23, 7905.54it/s] 54%|█████▍    | 215925/400000 [00:26<00:22, 8024.61it/s] 54%|█████▍    | 216761/400000 [00:26<00:22, 8121.54it/s] 54%|█████▍    | 217575/400000 [00:26<00:22, 8010.65it/s] 55%|█████▍    | 218403/400000 [00:26<00:22, 8087.70it/s] 55%|█████▍    | 219241/400000 [00:26<00:22, 8171.09it/s] 55%|█████▌    | 220078/400000 [00:27<00:21, 8228.28it/s] 55%|█████▌    | 220911/400000 [00:27<00:21, 8257.02it/s] 55%|█████▌    | 221738/400000 [00:27<00:21, 8238.94it/s] 56%|█████▌    | 222563/400000 [00:27<00:21, 8160.36it/s] 56%|█████▌    | 223380/400000 [00:27<00:22, 8015.37it/s] 56%|█████▌    | 224187/400000 [00:27<00:21, 8029.05it/s] 56%|█████▌    | 224991/400000 [00:27<00:21, 7987.64it/s] 56%|█████▋    | 225807/400000 [00:27<00:21, 8037.45it/s] 57%|█████▋    | 226638/400000 [00:27<00:21, 8115.25it/s] 57%|█████▋    | 227472/400000 [00:27<00:21, 8181.15it/s] 57%|█████▋    | 228301/400000 [00:28<00:20, 8212.32it/s] 57%|█████▋    | 229134/400000 [00:28<00:20, 8244.96it/s] 57%|█████▋    | 229959/400000 [00:28<00:20, 8236.80it/s] 58%|█████▊    | 230785/400000 [00:28<00:20, 8241.46it/s] 58%|█████▊    | 231614/400000 [00:28<00:20, 8255.03it/s] 58%|█████▊    | 232444/400000 [00:28<00:20, 8266.45it/s] 58%|█████▊    | 233271/400000 [00:28<00:20, 8177.90it/s] 59%|█████▊    | 234090/400000 [00:28<00:20, 8054.90it/s] 59%|█████▊    | 234914/400000 [00:28<00:20, 8108.35it/s] 59%|█████▉    | 235744/400000 [00:28<00:20, 8162.87it/s] 59%|█████▉    | 236570/400000 [00:29<00:19, 8189.05it/s] 59%|█████▉    | 237390/400000 [00:29<00:19, 8180.68it/s] 60%|█████▉    | 238209/400000 [00:29<00:19, 8178.60it/s] 60%|█████▉    | 239028/400000 [00:29<00:19, 8177.42it/s] 60%|█████▉    | 239862/400000 [00:29<00:19, 8223.55it/s] 60%|██████    | 240695/400000 [00:29<00:19, 8252.76it/s] 60%|██████    | 241521/400000 [00:29<00:19, 8235.08it/s] 61%|██████    | 242345/400000 [00:29<00:19, 7992.80it/s] 61%|██████    | 243180/400000 [00:29<00:19, 8095.17it/s] 61%|██████    | 244005/400000 [00:29<00:19, 8138.38it/s] 61%|██████    | 244837/400000 [00:30<00:18, 8189.89it/s] 61%|██████▏   | 245671/400000 [00:30<00:18, 8232.18it/s] 62%|██████▏   | 246495/400000 [00:30<00:18, 8200.34it/s] 62%|██████▏   | 247331/400000 [00:30<00:18, 8246.89it/s] 62%|██████▏   | 248164/400000 [00:30<00:18, 8269.21it/s] 62%|██████▏   | 248994/400000 [00:30<00:18, 8275.53it/s] 62%|██████▏   | 249824/400000 [00:30<00:18, 8280.24it/s] 63%|██████▎   | 250653/400000 [00:30<00:18, 8264.23it/s] 63%|██████▎   | 251480/400000 [00:30<00:18, 8165.06it/s] 63%|██████▎   | 252308/400000 [00:30<00:18, 8198.17it/s] 63%|██████▎   | 253143/400000 [00:31<00:17, 8243.04it/s] 63%|██████▎   | 253968/400000 [00:31<00:18, 8091.46it/s] 64%|██████▎   | 254779/400000 [00:31<00:17, 8094.41it/s] 64%|██████▍   | 255612/400000 [00:31<00:17, 8161.00it/s] 64%|██████▍   | 256437/400000 [00:31<00:17, 8185.54it/s] 64%|██████▍   | 257268/400000 [00:31<00:17, 8221.71it/s] 65%|██████▍   | 258102/400000 [00:31<00:17, 8256.83it/s] 65%|██████▍   | 258928/400000 [00:31<00:17, 8254.52it/s] 65%|██████▍   | 259754/400000 [00:31<00:17, 8199.86it/s] 65%|██████▌   | 260586/400000 [00:31<00:16, 8233.37it/s] 65%|██████▌   | 261423/400000 [00:32<00:16, 8272.57it/s] 66%|██████▌   | 262257/400000 [00:32<00:16, 8292.65it/s] 66%|██████▌   | 263087/400000 [00:32<00:16, 8257.51it/s] 66%|██████▌   | 263920/400000 [00:32<00:16, 8278.98it/s] 66%|██████▌   | 264748/400000 [00:32<00:16, 8254.85it/s] 66%|██████▋   | 265585/400000 [00:32<00:16, 8286.88it/s] 67%|██████▋   | 266414/400000 [00:32<00:16, 8271.04it/s] 67%|██████▋   | 267242/400000 [00:32<00:16, 8239.94it/s] 67%|██████▋   | 268078/400000 [00:32<00:15, 8273.55it/s] 67%|██████▋   | 268906/400000 [00:32<00:15, 8252.88it/s] 67%|██████▋   | 269736/400000 [00:33<00:15, 8265.86it/s] 68%|██████▊   | 270563/400000 [00:33<00:15, 8096.99it/s] 68%|██████▊   | 271379/400000 [00:33<00:15, 8115.43it/s] 68%|██████▊   | 272211/400000 [00:33<00:15, 8173.46it/s] 68%|██████▊   | 273046/400000 [00:33<00:15, 8224.22it/s] 68%|██████▊   | 273869/400000 [00:33<00:15, 8221.12it/s] 69%|██████▊   | 274692/400000 [00:33<00:15, 8223.20it/s] 69%|██████▉   | 275515/400000 [00:33<00:15, 8203.86it/s] 69%|██████▉   | 276349/400000 [00:33<00:15, 8242.83it/s] 69%|██████▉   | 277174/400000 [00:33<00:14, 8243.47it/s] 69%|██████▉   | 277999/400000 [00:34<00:14, 8239.83it/s] 70%|██████▉   | 278833/400000 [00:34<00:14, 8268.38it/s] 70%|██████▉   | 279660/400000 [00:34<00:14, 8227.35it/s] 70%|███████   | 280495/400000 [00:34<00:14, 8263.08it/s] 70%|███████   | 281329/400000 [00:34<00:14, 8284.58it/s] 71%|███████   | 282158/400000 [00:34<00:14, 8234.38it/s] 71%|███████   | 282982/400000 [00:34<00:14, 8212.92it/s] 71%|███████   | 283804/400000 [00:34<00:14, 8110.68it/s] 71%|███████   | 284633/400000 [00:34<00:14, 8160.89it/s] 71%|███████▏  | 285450/400000 [00:35<00:14, 8031.13it/s] 72%|███████▏  | 286271/400000 [00:35<00:14, 8081.96it/s] 72%|███████▏  | 287106/400000 [00:35<00:13, 8159.68it/s] 72%|███████▏  | 287923/400000 [00:35<00:13, 8023.47it/s] 72%|███████▏  | 288755/400000 [00:35<00:13, 8107.66it/s] 72%|███████▏  | 289582/400000 [00:35<00:13, 8153.02it/s] 73%|███████▎  | 290414/400000 [00:35<00:13, 8200.45it/s] 73%|███████▎  | 291239/400000 [00:35<00:13, 8215.10it/s] 73%|███████▎  | 292069/400000 [00:35<00:13, 8237.47it/s] 73%|███████▎  | 292894/400000 [00:35<00:13, 8215.71it/s] 73%|███████▎  | 293716/400000 [00:36<00:13, 8021.07it/s] 74%|███████▎  | 294551/400000 [00:36<00:12, 8114.32it/s] 74%|███████▍  | 295382/400000 [00:36<00:12, 8170.37it/s] 74%|███████▍  | 296201/400000 [00:36<00:12, 8173.51it/s] 74%|███████▍  | 297033/400000 [00:36<00:12, 8214.45it/s] 74%|███████▍  | 297855/400000 [00:36<00:12, 8033.42it/s] 75%|███████▍  | 298678/400000 [00:36<00:12, 8089.53it/s] 75%|███████▍  | 299495/400000 [00:36<00:12, 8110.72it/s] 75%|███████▌  | 300307/400000 [00:36<00:12, 8110.69it/s] 75%|███████▌  | 301133/400000 [00:36<00:12, 8152.52it/s] 75%|███████▌  | 301966/400000 [00:37<00:11, 8203.12it/s] 76%|███████▌  | 302798/400000 [00:37<00:11, 8235.96it/s] 76%|███████▌  | 303632/400000 [00:37<00:11, 8265.15it/s] 76%|███████▌  | 304464/400000 [00:37<00:11, 8280.64it/s] 76%|███████▋  | 305293/400000 [00:37<00:11, 8264.71it/s] 77%|███████▋  | 306120/400000 [00:37<00:11, 8230.14it/s] 77%|███████▋  | 306944/400000 [00:37<00:11, 8204.26it/s] 77%|███████▋  | 307765/400000 [00:37<00:11, 8122.43it/s] 77%|███████▋  | 308582/400000 [00:37<00:11, 8134.26it/s] 77%|███████▋  | 309396/400000 [00:37<00:11, 7994.39it/s] 78%|███████▊  | 310231/400000 [00:38<00:11, 8095.74it/s] 78%|███████▊  | 311042/400000 [00:38<00:11, 8084.05it/s] 78%|███████▊  | 311851/400000 [00:38<00:10, 8047.96it/s] 78%|███████▊  | 312681/400000 [00:38<00:10, 8121.26it/s] 78%|███████▊  | 313499/400000 [00:38<00:10, 8138.72it/s] 79%|███████▊  | 314336/400000 [00:38<00:10, 8205.89it/s] 79%|███████▉  | 315166/400000 [00:38<00:10, 8233.68it/s] 79%|███████▉  | 315998/400000 [00:38<00:10, 8257.47it/s] 79%|███████▉  | 316824/400000 [00:38<00:10, 8250.28it/s] 79%|███████▉  | 317650/400000 [00:38<00:10, 8225.69it/s] 80%|███████▉  | 318485/400000 [00:39<00:09, 8261.79it/s] 80%|███████▉  | 319316/400000 [00:39<00:09, 8275.28it/s] 80%|████████  | 320144/400000 [00:39<00:09, 8079.72it/s] 80%|████████  | 320976/400000 [00:39<00:09, 8148.35it/s] 80%|████████  | 321799/400000 [00:39<00:09, 8170.77it/s] 81%|████████  | 322634/400000 [00:39<00:09, 8223.05it/s] 81%|████████  | 323457/400000 [00:39<00:09, 8185.95it/s] 81%|████████  | 324288/400000 [00:39<00:09, 8219.96it/s] 81%|████████▏ | 325123/400000 [00:39<00:09, 8256.80it/s] 81%|████████▏ | 325949/400000 [00:39<00:09, 8187.05it/s] 82%|████████▏ | 326778/400000 [00:40<00:08, 8215.48it/s] 82%|████████▏ | 327600/400000 [00:40<00:08, 8153.31it/s] 82%|████████▏ | 328428/400000 [00:40<00:08, 8189.71it/s] 82%|████████▏ | 329265/400000 [00:40<00:08, 8242.66it/s] 83%|████████▎ | 330091/400000 [00:40<00:08, 8245.82it/s] 83%|████████▎ | 330920/400000 [00:40<00:08, 8257.14it/s] 83%|████████▎ | 331752/400000 [00:40<00:08, 8275.59it/s] 83%|████████▎ | 332585/400000 [00:40<00:08, 8290.96it/s] 83%|████████▎ | 333418/400000 [00:40<00:08, 8301.43it/s] 84%|████████▎ | 334249/400000 [00:40<00:07, 8242.45it/s] 84%|████████▍ | 335074/400000 [00:41<00:07, 8238.61it/s] 84%|████████▍ | 335898/400000 [00:41<00:07, 8191.82it/s] 84%|████████▍ | 336718/400000 [00:41<00:07, 8143.16it/s] 84%|████████▍ | 337533/400000 [00:41<00:07, 8038.29it/s] 85%|████████▍ | 338358/400000 [00:41<00:07, 8099.52it/s] 85%|████████▍ | 339169/400000 [00:41<00:07, 8070.76it/s] 85%|████████▍ | 339977/400000 [00:41<00:07, 8041.34it/s] 85%|████████▌ | 340794/400000 [00:41<00:07, 8078.78it/s] 85%|████████▌ | 341610/400000 [00:41<00:07, 8100.28it/s] 86%|████████▌ | 342421/400000 [00:41<00:07, 8077.35it/s] 86%|████████▌ | 343244/400000 [00:42<00:06, 8120.18it/s] 86%|████████▌ | 344067/400000 [00:42<00:06, 8149.93it/s] 86%|████████▌ | 344883/400000 [00:42<00:06, 8118.41it/s] 86%|████████▋ | 345711/400000 [00:42<00:06, 8166.10it/s] 87%|████████▋ | 346528/400000 [00:42<00:06, 8127.28it/s] 87%|████████▋ | 347357/400000 [00:42<00:06, 8173.19it/s] 87%|████████▋ | 348186/400000 [00:42<00:06, 8206.11it/s] 87%|████████▋ | 349011/400000 [00:42<00:06, 8217.03it/s] 87%|████████▋ | 349833/400000 [00:42<00:06, 8206.11it/s] 88%|████████▊ | 350654/400000 [00:42<00:06, 8143.91it/s] 88%|████████▊ | 351476/400000 [00:43<00:05, 8164.19it/s] 88%|████████▊ | 352306/400000 [00:43<00:05, 8204.36it/s] 88%|████████▊ | 353127/400000 [00:43<00:05, 7836.43it/s] 88%|████████▊ | 353952/400000 [00:43<00:05, 7953.54it/s] 89%|████████▊ | 354768/400000 [00:43<00:05, 8012.46it/s] 89%|████████▉ | 355603/400000 [00:43<00:05, 8110.25it/s] 89%|████████▉ | 356426/400000 [00:43<00:05, 8140.86it/s] 89%|████████▉ | 357250/400000 [00:43<00:05, 8168.10it/s] 90%|████████▉ | 358068/400000 [00:43<00:05, 8125.79it/s] 90%|████████▉ | 358882/400000 [00:44<00:05, 7818.39it/s] 90%|████████▉ | 359693/400000 [00:44<00:05, 7898.85it/s] 90%|█████████ | 360521/400000 [00:44<00:04, 8008.91it/s] 90%|█████████ | 361354/400000 [00:44<00:04, 8102.38it/s] 91%|█████████ | 362188/400000 [00:44<00:04, 8171.51it/s] 91%|█████████ | 363016/400000 [00:44<00:04, 8201.68it/s] 91%|█████████ | 363838/400000 [00:44<00:04, 8199.70it/s] 91%|█████████ | 364669/400000 [00:44<00:04, 8229.60it/s] 91%|█████████▏| 365493/400000 [00:44<00:04, 8225.99it/s] 92%|█████████▏| 366317/400000 [00:44<00:04, 8228.58it/s] 92%|█████████▏| 367141/400000 [00:45<00:04, 8055.66it/s] 92%|█████████▏| 367954/400000 [00:45<00:03, 8076.34it/s] 92%|█████████▏| 368771/400000 [00:45<00:03, 8103.72it/s] 92%|█████████▏| 369601/400000 [00:45<00:03, 8160.38it/s] 93%|█████████▎| 370420/400000 [00:45<00:03, 8168.40it/s] 93%|█████████▎| 371239/400000 [00:45<00:03, 8172.84it/s] 93%|█████████▎| 372057/400000 [00:45<00:03, 8062.99it/s] 93%|█████████▎| 372874/400000 [00:45<00:03, 8094.70it/s] 93%|█████████▎| 373697/400000 [00:45<00:03, 8132.49it/s] 94%|█████████▎| 374514/400000 [00:45<00:03, 8142.66it/s] 94%|█████████▍| 375340/400000 [00:46<00:03, 8176.30it/s] 94%|█████████▍| 376172/400000 [00:46<00:02, 8213.84it/s] 94%|█████████▍| 377002/400000 [00:46<00:02, 8237.17it/s] 94%|█████████▍| 377834/400000 [00:46<00:02, 8259.00it/s] 95%|█████████▍| 378663/400000 [00:46<00:02, 8267.49it/s] 95%|█████████▍| 379490/400000 [00:46<00:02, 8267.65it/s] 95%|█████████▌| 380323/400000 [00:46<00:02, 8285.97it/s] 95%|█████████▌| 381152/400000 [00:46<00:02, 8235.84it/s] 95%|█████████▌| 381981/400000 [00:46<00:02, 8251.25it/s] 96%|█████████▌| 382807/400000 [00:46<00:02, 8211.06it/s] 96%|█████████▌| 383630/400000 [00:47<00:01, 8214.94it/s] 96%|█████████▌| 384466/400000 [00:47<00:01, 8256.69it/s] 96%|█████████▋| 385305/400000 [00:47<00:01, 8295.64it/s] 97%|█████████▋| 386135/400000 [00:47<00:01, 8055.70it/s] 97%|█████████▋| 386957/400000 [00:47<00:01, 8103.46it/s] 97%|█████████▋| 387769/400000 [00:47<00:01, 7968.41it/s] 97%|█████████▋| 388601/400000 [00:47<00:01, 8068.43it/s] 97%|█████████▋| 389425/400000 [00:47<00:01, 8118.06it/s] 98%|█████████▊| 390260/400000 [00:47<00:01, 8185.24it/s] 98%|█████████▊| 391080/400000 [00:47<00:01, 8184.36it/s] 98%|█████████▊| 391914/400000 [00:48<00:00, 8229.71it/s] 98%|█████████▊| 392738/400000 [00:48<00:00, 8087.19it/s] 98%|█████████▊| 393548/400000 [00:48<00:00, 8046.25it/s] 99%|█████████▊| 394378/400000 [00:48<00:00, 8119.44it/s] 99%|█████████▉| 395204/400000 [00:48<00:00, 8159.98it/s] 99%|█████████▉| 396029/400000 [00:48<00:00, 8185.02it/s] 99%|█████████▉| 396848/400000 [00:48<00:00, 8059.04it/s] 99%|█████████▉| 397682/400000 [00:48<00:00, 8138.57it/s]100%|█████████▉| 398518/400000 [00:48<00:00, 8203.60it/s]100%|█████████▉| 399339/400000 [00:48<00:00, 8174.06it/s]100%|█████████▉| 399999/400000 [00:49<00:00, 8154.04it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9f18586940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010909743745610096 	 Accuracy: 55
Train Epoch: 1 	 Loss: 0.011152951813063096 	 Accuracy: 51

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
2020-05-15 21:28:51.146701: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-15 21:28:51.150637: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-05-15 21:28:51.150771: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560d6288bb30 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 21:28:51.150785: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f9f1b47b588> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6820 - accuracy: 0.4990
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5976 - accuracy: 0.5045 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6820 - accuracy: 0.4990
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7011 - accuracy: 0.4978
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6912 - accuracy: 0.4984
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6947 - accuracy: 0.4982
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7323 - accuracy: 0.4957
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6888 - accuracy: 0.4986
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6835 - accuracy: 0.4989
11000/25000 [============>.................] - ETA: 3s - loss: 7.6624 - accuracy: 0.5003
12000/25000 [=============>................] - ETA: 3s - loss: 7.6692 - accuracy: 0.4998
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6654 - accuracy: 0.5001
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6261 - accuracy: 0.5026
15000/25000 [=================>............] - ETA: 2s - loss: 7.6421 - accuracy: 0.5016
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6388 - accuracy: 0.5018
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6342 - accuracy: 0.5021
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6419 - accuracy: 0.5016
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6166 - accuracy: 0.5033
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6137 - accuracy: 0.5034
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6279 - accuracy: 0.5025
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6415 - accuracy: 0.5016
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6500 - accuracy: 0.5011
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6423 - accuracy: 0.5016
25000/25000 [==============================] - 7s 282us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f9e736c1668> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f9ecbad8be0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.6241 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.5724 - val_crf_viterbi_accuracy: 0.0133

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

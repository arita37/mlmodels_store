
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fea02e3ff60> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 22:13:11.538416
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 22:13:11.544169
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 22:13:11.548015
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 22:13:11.551755
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fea0ec09400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356331.2500
Epoch 2/10

1/1 [==============================] - 0s 116ms/step - loss: 257542.0781
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 155903.8438
Epoch 4/10

1/1 [==============================] - 0s 96ms/step - loss: 84915.1719
Epoch 5/10

1/1 [==============================] - 0s 98ms/step - loss: 43278.1641
Epoch 6/10

1/1 [==============================] - 0s 109ms/step - loss: 23192.2031
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 13195.6748
Epoch 8/10

1/1 [==============================] - 0s 93ms/step - loss: 8154.6631
Epoch 9/10

1/1 [==============================] - 0s 92ms/step - loss: 5493.3945
Epoch 10/10

1/1 [==============================] - 0s 95ms/step - loss: 4016.3503

  #### Inference Need return ypred, ytrue ######################### 
[[ 4.92410779e-01 -9.87880409e-01 -1.85369825e+00  1.11589980e+00
  -1.43244529e+00  7.42769480e-01 -9.86808658e-01 -1.15509236e+00
   2.04433113e-01  5.13048530e-01  1.25860524e+00 -2.89387763e-01
   1.45702243e+00  1.00461841e+00 -2.23141623e+00  7.24573731e-02
   1.53513789e+00 -6.55982018e-01 -1.97489524e+00  1.05343759e+00
   5.36093935e-02  1.03718054e+00  1.54428518e+00 -1.25441223e-01
   3.92771274e-01  3.00140285e+00  2.04194093e+00  1.45440626e+00
  -9.52687681e-01  1.54618382e+00  1.79848385e+00  1.17144728e+00
  -1.06837660e-01 -8.26904327e-02  4.31349814e-01 -5.83856106e-01
   1.36016119e+00  1.04043150e+00 -7.80156910e-01 -3.34428757e-01
  -8.83916855e-01  1.32150841e+00  1.79016864e+00 -1.09605813e+00
  -8.33900213e-01 -8.68554115e-01 -1.58438969e+00  5.08050442e-01
  -1.52554893e+00 -1.36378121e+00 -1.65165746e+00  2.37340569e+00
  -1.72591746e+00 -1.25714511e-01 -2.80998200e-01  8.28653932e-01
   4.65862244e-01 -1.44015759e-01  1.08731359e-01  3.83913487e-01
  -1.34915143e-01  1.58212161e+00  2.28090119e+00 -9.34505224e-01
  -4.66406047e-01  9.97962832e-01  1.64543891e+00 -2.48832178e+00
  -3.60513926e-02  2.95547664e-01 -9.05896127e-02  4.64383215e-01
   4.36218351e-01 -1.84937549e+00 -1.43953657e+00 -6.68383002e-01
   1.94137192e+00  6.62410498e-01  9.63272214e-01  3.22270393e-03
  -3.21991920e-01  1.46049380e+00 -1.76726282e-01  1.73042238e+00
   1.55687737e+00 -9.12460983e-01 -6.36712462e-02 -3.58000278e-01
  -1.27589166e-01  1.62499142e+00 -1.92838931e+00 -1.06273460e+00
  -9.46339130e-01  1.24373233e+00 -1.14682913e-01  4.38280910e-01
  -1.19193166e-01  2.13762611e-01 -3.02831483e+00  4.51633692e-01
  -7.87099361e-01 -6.49227679e-01 -2.24836871e-01 -2.38621640e+00
   3.65537494e-01 -7.65722513e-01  6.78396225e-01  6.80118859e-01
   1.11942434e+00 -1.72170568e+00  5.37548125e-01  1.77304161e+00
   1.08691895e+00  2.26169443e+00 -7.98881501e-02  7.23849982e-02
   2.96851575e-01 -3.75072092e-01 -8.56665313e-01 -2.08624554e+00
   2.10244447e-01  1.43513527e+01  1.15535240e+01  1.13095484e+01
   1.21928844e+01  1.22612743e+01  1.12290783e+01  1.19430084e+01
   1.10948505e+01  1.06166325e+01  1.12228622e+01  9.60221481e+00
   1.10636177e+01  1.00984383e+01  1.28581123e+01  9.43819714e+00
   1.10999393e+01  1.29955978e+01  9.50043392e+00  1.24428139e+01
   1.01086168e+01  1.14508896e+01  1.24041681e+01  1.08593712e+01
   1.20490704e+01  1.17218504e+01  1.26147289e+01  1.09791880e+01
   1.06793919e+01  1.05913315e+01  1.31125822e+01  1.21167965e+01
   1.18433743e+01  1.28742189e+01  1.27142448e+01  1.12143555e+01
   1.03873472e+01  1.14296284e+01  8.94019222e+00  1.11045313e+01
   1.06715994e+01  1.08319616e+01  1.03581743e+01  1.10939865e+01
   1.02686310e+01  1.16319504e+01  1.03128948e+01  9.34707355e+00
   9.43083286e+00  1.05541372e+01  1.25013466e+01  1.09778547e+01
   9.92846394e+00  1.03761444e+01  1.03538151e+01  1.29240885e+01
   1.35839148e+01  1.19819450e+01  1.08384361e+01  1.27143164e+01
   7.66489565e-01  1.20139539e+00  1.91776037e-01  1.84833050e-01
   2.23689222e+00  8.46668005e-01  6.15889132e-01  3.80705178e-01
   3.56779099e-01  2.90390491e-01  3.65444601e-01  2.00483656e+00
   2.78185844e-01  9.66445386e-01  1.07489789e+00  1.05154061e+00
   2.44414306e+00  7.65631199e-02  6.84520483e-01  1.31664896e+00
   2.60908413e+00  7.94871032e-01  3.11962509e+00  2.56889367e+00
   8.10343981e-01  9.44272637e-01  9.12551582e-01  1.47704124e-01
   1.80144191e+00  3.08433771e-01  1.61458910e-01  1.20582724e+00
   6.87501490e-01  1.84709191e-01  1.47230744e+00  1.19834304e-01
   2.66448617e-01  1.26402009e+00  1.00517285e+00  1.04266763e-01
   1.30127311e-01  2.28411341e+00  2.01887608e-01  2.29511619e+00
   1.42275858e+00  4.31668639e-01  2.59614289e-01  1.48292637e+00
   3.95573258e-01  2.08418560e+00  8.12630057e-02  8.37620795e-01
   2.81708050e+00  9.16259050e-01  1.84737682e-01  5.32790005e-01
   1.92496133e+00  2.60675645e+00  7.91600108e-01  4.09327507e+00
   2.17221403e+00  5.19618392e-01  1.95395303e+00  4.67661738e-01
   2.75050402e+00  3.95637989e-01  2.69971085e+00  2.95733690e+00
   4.08669567e+00  4.38118219e-01  1.01532006e+00  6.45879924e-01
   2.55612040e+00  8.39225471e-01  1.41020846e+00  3.81997395e+00
   2.29845524e+00  2.59998131e+00  1.33379161e-01  6.38609529e-01
   1.11658454e-01  4.34218824e-01  8.54666054e-01  3.00681973e+00
   1.87181234e+00  1.69213557e+00  4.69464481e-01  2.23235846e-01
   1.04078698e+00  6.36710882e-01  2.20608532e-01  1.39653528e+00
   1.60365093e+00  2.16231108e+00  1.52848959e+00  2.33787775e-01
   1.08248711e+00  2.21949518e-01  1.26413941e-01  2.13907671e+00
   3.12983274e-01  3.90321851e-01  8.24292898e-02  2.13619566e+00
   3.64376783e-01  2.06747484e+00  1.75434959e+00  3.66277337e-01
   2.66688490e+00  2.76141787e+00  1.98272741e+00  2.00465631e+00
   2.80195808e+00  1.94846284e+00  2.16345167e+00  9.94364321e-01
   1.72719812e+00  4.42059696e-01  4.69949186e-01  2.57866263e-01
   2.64663458e-01  1.04636688e+01  1.25212269e+01  1.32109547e+01
   1.35918894e+01  1.16333122e+01  1.22637863e+01  1.16600208e+01
   1.20616350e+01  1.43900537e+01  1.18356762e+01  9.62020397e+00
   1.06210728e+01  1.32613859e+01  1.02820749e+01  1.20029430e+01
   1.09595299e+01  1.16175003e+01  1.05075045e+01  1.15648909e+01
   1.17620821e+01  1.12752218e+01  9.90226364e+00  1.09765244e+01
   1.01162548e+01  1.03202848e+01  1.07662315e+01  1.17131481e+01
   1.14820967e+01  1.26111202e+01  1.10904016e+01  1.14261742e+01
   9.04467201e+00  1.13827581e+01  1.04151583e+01  1.10621843e+01
   1.22151623e+01  1.07718430e+01  1.08001547e+01  1.12301960e+01
   1.26478739e+01  1.38372164e+01  9.46713352e+00  1.30253944e+01
   1.16063051e+01  1.03832731e+01  1.01134787e+01  1.09031210e+01
   1.14793243e+01  9.64942169e+00  1.19579668e+01  1.13692102e+01
   9.92790699e+00  1.09174290e+01  9.65254974e+00  1.13908072e+01
   1.02503548e+01  1.18371067e+01  1.00121946e+01  1.02539282e+01
  -8.87144947e+00 -6.54280138e+00  1.50763416e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 22:13:21.595245
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   90.4811
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 22:13:21.599677
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8219.39
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 22:13:21.603861
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   90.8222
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 22:13:21.607900
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -735.105
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140642705241424
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140641612157336
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140641612157840
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140641612158344
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140641612158848
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140641612159352

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fe9ee82bda0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.575423
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.550347
grad_step = 000002, loss = 0.526695
grad_step = 000003, loss = 0.501396
grad_step = 000004, loss = 0.471937
grad_step = 000005, loss = 0.439898
grad_step = 000006, loss = 0.407973
grad_step = 000007, loss = 0.388335
grad_step = 000008, loss = 0.385184
grad_step = 000009, loss = 0.370289
grad_step = 000010, loss = 0.346254
grad_step = 000011, loss = 0.326725
grad_step = 000012, loss = 0.312228
grad_step = 000013, loss = 0.299633
grad_step = 000014, loss = 0.286912
grad_step = 000015, loss = 0.273945
grad_step = 000016, loss = 0.261987
grad_step = 000017, loss = 0.251165
grad_step = 000018, loss = 0.240004
grad_step = 000019, loss = 0.228143
grad_step = 000020, loss = 0.216254
grad_step = 000021, loss = 0.204444
grad_step = 000022, loss = 0.192810
grad_step = 000023, loss = 0.182126
grad_step = 000024, loss = 0.172792
grad_step = 000025, loss = 0.163732
grad_step = 000026, loss = 0.153892
grad_step = 000027, loss = 0.144010
grad_step = 000028, loss = 0.135266
grad_step = 000029, loss = 0.127288
grad_step = 000030, loss = 0.119114
grad_step = 000031, loss = 0.110956
grad_step = 000032, loss = 0.103459
grad_step = 000033, loss = 0.096558
grad_step = 000034, loss = 0.090043
grad_step = 000035, loss = 0.083953
grad_step = 000036, loss = 0.078088
grad_step = 000037, loss = 0.072318
grad_step = 000038, loss = 0.066946
grad_step = 000039, loss = 0.062162
grad_step = 000040, loss = 0.057604
grad_step = 000041, loss = 0.053125
grad_step = 000042, loss = 0.049076
grad_step = 000043, loss = 0.045443
grad_step = 000044, loss = 0.041977
grad_step = 000045, loss = 0.038723
grad_step = 000046, loss = 0.035761
grad_step = 000047, loss = 0.032973
grad_step = 000048, loss = 0.030338
grad_step = 000049, loss = 0.027903
grad_step = 000050, loss = 0.025635
grad_step = 000051, loss = 0.023609
grad_step = 000052, loss = 0.021806
grad_step = 000053, loss = 0.020056
grad_step = 000054, loss = 0.018409
grad_step = 000055, loss = 0.016971
grad_step = 000056, loss = 0.015629
grad_step = 000057, loss = 0.014345
grad_step = 000058, loss = 0.013192
grad_step = 000059, loss = 0.012146
grad_step = 000060, loss = 0.011195
grad_step = 000061, loss = 0.010333
grad_step = 000062, loss = 0.009531
grad_step = 000063, loss = 0.008816
grad_step = 000064, loss = 0.008173
grad_step = 000065, loss = 0.007568
grad_step = 000066, loss = 0.007033
grad_step = 000067, loss = 0.006554
grad_step = 000068, loss = 0.006094
grad_step = 000069, loss = 0.005682
grad_step = 000070, loss = 0.005324
grad_step = 000071, loss = 0.005000
grad_step = 000072, loss = 0.004705
grad_step = 000073, loss = 0.004434
grad_step = 000074, loss = 0.004194
grad_step = 000075, loss = 0.003976
grad_step = 000076, loss = 0.003770
grad_step = 000077, loss = 0.003591
grad_step = 000078, loss = 0.003430
grad_step = 000079, loss = 0.003282
grad_step = 000080, loss = 0.003113
grad_step = 000081, loss = 0.003026
grad_step = 000082, loss = 0.002896
grad_step = 000083, loss = 0.002807
grad_step = 000084, loss = 0.002714
grad_step = 000085, loss = 0.002642
grad_step = 000086, loss = 0.002573
grad_step = 000087, loss = 0.002512
grad_step = 000088, loss = 0.002447
grad_step = 000089, loss = 0.002405
grad_step = 000090, loss = 0.002358
grad_step = 000091, loss = 0.002311
grad_step = 000092, loss = 0.002288
grad_step = 000093, loss = 0.002249
grad_step = 000094, loss = 0.002225
grad_step = 000095, loss = 0.002202
grad_step = 000096, loss = 0.002177
grad_step = 000097, loss = 0.002160
grad_step = 000098, loss = 0.002140
grad_step = 000099, loss = 0.002124
grad_step = 000100, loss = 0.002109
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002098
grad_step = 000102, loss = 0.002083
grad_step = 000103, loss = 0.002075
grad_step = 000104, loss = 0.002064
grad_step = 000105, loss = 0.002055
grad_step = 000106, loss = 0.002047
grad_step = 000107, loss = 0.002039
grad_step = 000108, loss = 0.002032
grad_step = 000109, loss = 0.002025
grad_step = 000110, loss = 0.002020
grad_step = 000111, loss = 0.002013
grad_step = 000112, loss = 0.002008
grad_step = 000113, loss = 0.002003
grad_step = 000114, loss = 0.001999
grad_step = 000115, loss = 0.001993
grad_step = 000116, loss = 0.001989
grad_step = 000117, loss = 0.001985
grad_step = 000118, loss = 0.001981
grad_step = 000119, loss = 0.001976
grad_step = 000120, loss = 0.001972
grad_step = 000121, loss = 0.001968
grad_step = 000122, loss = 0.001965
grad_step = 000123, loss = 0.001960
grad_step = 000124, loss = 0.001957
grad_step = 000125, loss = 0.001953
grad_step = 000126, loss = 0.001950
grad_step = 000127, loss = 0.001946
grad_step = 000128, loss = 0.001942
grad_step = 000129, loss = 0.001939
grad_step = 000130, loss = 0.001935
grad_step = 000131, loss = 0.001932
grad_step = 000132, loss = 0.001928
grad_step = 000133, loss = 0.001925
grad_step = 000134, loss = 0.001921
grad_step = 000135, loss = 0.001918
grad_step = 000136, loss = 0.001915
grad_step = 000137, loss = 0.001911
grad_step = 000138, loss = 0.001908
grad_step = 000139, loss = 0.001904
grad_step = 000140, loss = 0.001901
grad_step = 000141, loss = 0.001897
grad_step = 000142, loss = 0.001894
grad_step = 000143, loss = 0.001891
grad_step = 000144, loss = 0.001887
grad_step = 000145, loss = 0.001884
grad_step = 000146, loss = 0.001881
grad_step = 000147, loss = 0.001877
grad_step = 000148, loss = 0.001874
grad_step = 000149, loss = 0.001871
grad_step = 000150, loss = 0.001867
grad_step = 000151, loss = 0.001864
grad_step = 000152, loss = 0.001861
grad_step = 000153, loss = 0.001858
grad_step = 000154, loss = 0.001855
grad_step = 000155, loss = 0.001852
grad_step = 000156, loss = 0.001850
grad_step = 000157, loss = 0.001850
grad_step = 000158, loss = 0.001855
grad_step = 000159, loss = 0.001866
grad_step = 000160, loss = 0.001890
grad_step = 000161, loss = 0.001935
grad_step = 000162, loss = 0.002012
grad_step = 000163, loss = 0.002089
grad_step = 000164, loss = 0.002130
grad_step = 000165, loss = 0.002042
grad_step = 000166, loss = 0.001893
grad_step = 000167, loss = 0.001817
grad_step = 000168, loss = 0.001867
grad_step = 000169, loss = 0.001960
grad_step = 000170, loss = 0.001968
grad_step = 000171, loss = 0.001883
grad_step = 000172, loss = 0.001809
grad_step = 000173, loss = 0.001826
grad_step = 000174, loss = 0.001887
grad_step = 000175, loss = 0.001896
grad_step = 000176, loss = 0.001842
grad_step = 000177, loss = 0.001795
grad_step = 000178, loss = 0.001808
grad_step = 000179, loss = 0.001845
grad_step = 000180, loss = 0.001849
grad_step = 000181, loss = 0.001815
grad_step = 000182, loss = 0.001785
grad_step = 000183, loss = 0.001789
grad_step = 000184, loss = 0.001812
grad_step = 000185, loss = 0.001818
grad_step = 000186, loss = 0.001799
grad_step = 000187, loss = 0.001777
grad_step = 000188, loss = 0.001773
grad_step = 000189, loss = 0.001786
grad_step = 000190, loss = 0.001796
grad_step = 000191, loss = 0.001791
grad_step = 000192, loss = 0.001775
grad_step = 000193, loss = 0.001763
grad_step = 000194, loss = 0.001761
grad_step = 000195, loss = 0.001768
grad_step = 000196, loss = 0.001773
grad_step = 000197, loss = 0.001772
grad_step = 000198, loss = 0.001764
grad_step = 000199, loss = 0.001755
grad_step = 000200, loss = 0.001749
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001748
grad_step = 000202, loss = 0.001749
grad_step = 000203, loss = 0.001752
grad_step = 000204, loss = 0.001753
grad_step = 000205, loss = 0.001751
grad_step = 000206, loss = 0.001748
grad_step = 000207, loss = 0.001743
grad_step = 000208, loss = 0.001738
grad_step = 000209, loss = 0.001734
grad_step = 000210, loss = 0.001731
grad_step = 000211, loss = 0.001729
grad_step = 000212, loss = 0.001727
grad_step = 000213, loss = 0.001726
grad_step = 000214, loss = 0.001725
grad_step = 000215, loss = 0.001725
grad_step = 000216, loss = 0.001725
grad_step = 000217, loss = 0.001727
grad_step = 000218, loss = 0.001731
grad_step = 000219, loss = 0.001738
grad_step = 000220, loss = 0.001752
grad_step = 000221, loss = 0.001777
grad_step = 000222, loss = 0.001820
grad_step = 000223, loss = 0.001885
grad_step = 000224, loss = 0.001972
grad_step = 000225, loss = 0.002038
grad_step = 000226, loss = 0.002024
grad_step = 000227, loss = 0.001901
grad_step = 000228, loss = 0.001756
grad_step = 000229, loss = 0.001698
grad_step = 000230, loss = 0.001750
grad_step = 000231, loss = 0.001835
grad_step = 000232, loss = 0.001855
grad_step = 000233, loss = 0.001790
grad_step = 000234, loss = 0.001709
grad_step = 000235, loss = 0.001686
grad_step = 000236, loss = 0.001725
grad_step = 000237, loss = 0.001777
grad_step = 000238, loss = 0.001788
grad_step = 000239, loss = 0.001750
grad_step = 000240, loss = 0.001697
grad_step = 000241, loss = 0.001673
grad_step = 000242, loss = 0.001686
grad_step = 000243, loss = 0.001715
grad_step = 000244, loss = 0.001731
grad_step = 000245, loss = 0.001720
grad_step = 000246, loss = 0.001693
grad_step = 000247, loss = 0.001668
grad_step = 000248, loss = 0.001662
grad_step = 000249, loss = 0.001672
grad_step = 000250, loss = 0.001687
grad_step = 000251, loss = 0.001696
grad_step = 000252, loss = 0.001690
grad_step = 000253, loss = 0.001676
grad_step = 000254, loss = 0.001661
grad_step = 000255, loss = 0.001653
grad_step = 000256, loss = 0.001653
grad_step = 000257, loss = 0.001658
grad_step = 000258, loss = 0.001666
grad_step = 000259, loss = 0.001671
grad_step = 000260, loss = 0.001674
grad_step = 000261, loss = 0.001673
grad_step = 000262, loss = 0.001666
grad_step = 000263, loss = 0.001657
grad_step = 000264, loss = 0.001649
grad_step = 000265, loss = 0.001645
grad_step = 000266, loss = 0.001643
grad_step = 000267, loss = 0.001641
grad_step = 000268, loss = 0.001642
grad_step = 000269, loss = 0.001645
grad_step = 000270, loss = 0.001646
grad_step = 000271, loss = 0.001648
grad_step = 000272, loss = 0.001649
grad_step = 000273, loss = 0.001651
grad_step = 000274, loss = 0.001653
grad_step = 000275, loss = 0.001655
grad_step = 000276, loss = 0.001656
grad_step = 000277, loss = 0.001660
grad_step = 000278, loss = 0.001665
grad_step = 000279, loss = 0.001671
grad_step = 000280, loss = 0.001679
grad_step = 000281, loss = 0.001689
grad_step = 000282, loss = 0.001704
grad_step = 000283, loss = 0.001721
grad_step = 000284, loss = 0.001739
grad_step = 000285, loss = 0.001750
grad_step = 000286, loss = 0.001748
grad_step = 000287, loss = 0.001729
grad_step = 000288, loss = 0.001697
grad_step = 000289, loss = 0.001661
grad_step = 000290, loss = 0.001633
grad_step = 000291, loss = 0.001619
grad_step = 000292, loss = 0.001618
grad_step = 000293, loss = 0.001627
grad_step = 000294, loss = 0.001643
grad_step = 000295, loss = 0.001662
grad_step = 000296, loss = 0.001682
grad_step = 000297, loss = 0.001702
grad_step = 000298, loss = 0.001718
grad_step = 000299, loss = 0.001729
grad_step = 000300, loss = 0.001731
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001722
grad_step = 000302, loss = 0.001703
grad_step = 000303, loss = 0.001676
grad_step = 000304, loss = 0.001648
grad_step = 000305, loss = 0.001624
grad_step = 000306, loss = 0.001609
grad_step = 000307, loss = 0.001603
grad_step = 000308, loss = 0.001607
grad_step = 000309, loss = 0.001615
grad_step = 000310, loss = 0.001626
grad_step = 000311, loss = 0.001636
grad_step = 000312, loss = 0.001644
grad_step = 000313, loss = 0.001650
grad_step = 000314, loss = 0.001652
grad_step = 000315, loss = 0.001651
grad_step = 000316, loss = 0.001647
grad_step = 000317, loss = 0.001641
grad_step = 000318, loss = 0.001633
grad_step = 000319, loss = 0.001626
grad_step = 000320, loss = 0.001618
grad_step = 000321, loss = 0.001610
grad_step = 000322, loss = 0.001603
grad_step = 000323, loss = 0.001597
grad_step = 000324, loss = 0.001593
grad_step = 000325, loss = 0.001589
grad_step = 000326, loss = 0.001587
grad_step = 000327, loss = 0.001586
grad_step = 000328, loss = 0.001585
grad_step = 000329, loss = 0.001584
grad_step = 000330, loss = 0.001584
grad_step = 000331, loss = 0.001584
grad_step = 000332, loss = 0.001585
grad_step = 000333, loss = 0.001587
grad_step = 000334, loss = 0.001592
grad_step = 000335, loss = 0.001601
grad_step = 000336, loss = 0.001619
grad_step = 000337, loss = 0.001655
grad_step = 000338, loss = 0.001723
grad_step = 000339, loss = 0.001844
grad_step = 000340, loss = 0.002040
grad_step = 000341, loss = 0.002282
grad_step = 000342, loss = 0.002437
grad_step = 000343, loss = 0.002293
grad_step = 000344, loss = 0.001877
grad_step = 000345, loss = 0.001594
grad_step = 000346, loss = 0.001695
grad_step = 000347, loss = 0.001943
grad_step = 000348, loss = 0.001978
grad_step = 000349, loss = 0.001740
grad_step = 000350, loss = 0.001580
grad_step = 000351, loss = 0.001689
grad_step = 000352, loss = 0.001835
grad_step = 000353, loss = 0.001763
grad_step = 000354, loss = 0.001592
grad_step = 000355, loss = 0.001603
grad_step = 000356, loss = 0.001713
grad_step = 000357, loss = 0.001721
grad_step = 000358, loss = 0.001636
grad_step = 000359, loss = 0.001578
grad_step = 000360, loss = 0.001615
grad_step = 000361, loss = 0.001671
grad_step = 000362, loss = 0.001654
grad_step = 000363, loss = 0.001577
grad_step = 000364, loss = 0.001569
grad_step = 000365, loss = 0.001622
grad_step = 000366, loss = 0.001634
grad_step = 000367, loss = 0.001595
grad_step = 000368, loss = 0.001560
grad_step = 000369, loss = 0.001576
grad_step = 000370, loss = 0.001600
grad_step = 000371, loss = 0.001600
grad_step = 000372, loss = 0.001569
grad_step = 000373, loss = 0.001554
grad_step = 000374, loss = 0.001568
grad_step = 000375, loss = 0.001582
grad_step = 000376, loss = 0.001577
grad_step = 000377, loss = 0.001555
grad_step = 000378, loss = 0.001548
grad_step = 000379, loss = 0.001559
grad_step = 000380, loss = 0.001568
grad_step = 000381, loss = 0.001559
grad_step = 000382, loss = 0.001547
grad_step = 000383, loss = 0.001545
grad_step = 000384, loss = 0.001548
grad_step = 000385, loss = 0.001552
grad_step = 000386, loss = 0.001550
grad_step = 000387, loss = 0.001545
grad_step = 000388, loss = 0.001538
grad_step = 000389, loss = 0.001538
grad_step = 000390, loss = 0.001541
grad_step = 000391, loss = 0.001542
grad_step = 000392, loss = 0.001540
grad_step = 000393, loss = 0.001535
grad_step = 000394, loss = 0.001532
grad_step = 000395, loss = 0.001532
grad_step = 000396, loss = 0.001533
grad_step = 000397, loss = 0.001533
grad_step = 000398, loss = 0.001532
grad_step = 000399, loss = 0.001530
grad_step = 000400, loss = 0.001527
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001525
grad_step = 000402, loss = 0.001525
grad_step = 000403, loss = 0.001526
grad_step = 000404, loss = 0.001525
grad_step = 000405, loss = 0.001524
grad_step = 000406, loss = 0.001522
grad_step = 000407, loss = 0.001520
grad_step = 000408, loss = 0.001519
grad_step = 000409, loss = 0.001518
grad_step = 000410, loss = 0.001517
grad_step = 000411, loss = 0.001517
grad_step = 000412, loss = 0.001516
grad_step = 000413, loss = 0.001515
grad_step = 000414, loss = 0.001514
grad_step = 000415, loss = 0.001513
grad_step = 000416, loss = 0.001512
grad_step = 000417, loss = 0.001511
grad_step = 000418, loss = 0.001510
grad_step = 000419, loss = 0.001509
grad_step = 000420, loss = 0.001507
grad_step = 000421, loss = 0.001506
grad_step = 000422, loss = 0.001505
grad_step = 000423, loss = 0.001504
grad_step = 000424, loss = 0.001503
grad_step = 000425, loss = 0.001502
grad_step = 000426, loss = 0.001501
grad_step = 000427, loss = 0.001500
grad_step = 000428, loss = 0.001499
grad_step = 000429, loss = 0.001498
grad_step = 000430, loss = 0.001497
grad_step = 000431, loss = 0.001496
grad_step = 000432, loss = 0.001495
grad_step = 000433, loss = 0.001495
grad_step = 000434, loss = 0.001494
grad_step = 000435, loss = 0.001495
grad_step = 000436, loss = 0.001498
grad_step = 000437, loss = 0.001505
grad_step = 000438, loss = 0.001521
grad_step = 000439, loss = 0.001554
grad_step = 000440, loss = 0.001621
grad_step = 000441, loss = 0.001737
grad_step = 000442, loss = 0.001916
grad_step = 000443, loss = 0.002112
grad_step = 000444, loss = 0.002188
grad_step = 000445, loss = 0.002031
grad_step = 000446, loss = 0.001739
grad_step = 000447, loss = 0.001540
grad_step = 000448, loss = 0.001552
grad_step = 000449, loss = 0.001687
grad_step = 000450, loss = 0.001782
grad_step = 000451, loss = 0.001722
grad_step = 000452, loss = 0.001562
grad_step = 000453, loss = 0.001484
grad_step = 000454, loss = 0.001560
grad_step = 000455, loss = 0.001656
grad_step = 000456, loss = 0.001636
grad_step = 000457, loss = 0.001541
grad_step = 000458, loss = 0.001494
grad_step = 000459, loss = 0.001511
grad_step = 000460, loss = 0.001538
grad_step = 000461, loss = 0.001551
grad_step = 000462, loss = 0.001548
grad_step = 000463, loss = 0.001516
grad_step = 000464, loss = 0.001477
grad_step = 000465, loss = 0.001472
grad_step = 000466, loss = 0.001503
grad_step = 000467, loss = 0.001524
grad_step = 000468, loss = 0.001506
grad_step = 000469, loss = 0.001476
grad_step = 000470, loss = 0.001463
grad_step = 000471, loss = 0.001470
grad_step = 000472, loss = 0.001479
grad_step = 000473, loss = 0.001482
grad_step = 000474, loss = 0.001480
grad_step = 000475, loss = 0.001472
grad_step = 000476, loss = 0.001459
grad_step = 000477, loss = 0.001451
grad_step = 000478, loss = 0.001454
grad_step = 000479, loss = 0.001462
grad_step = 000480, loss = 0.001465
grad_step = 000481, loss = 0.001459
grad_step = 000482, loss = 0.001451
grad_step = 000483, loss = 0.001446
grad_step = 000484, loss = 0.001444
grad_step = 000485, loss = 0.001442
grad_step = 000486, loss = 0.001441
grad_step = 000487, loss = 0.001442
grad_step = 000488, loss = 0.001444
grad_step = 000489, loss = 0.001444
grad_step = 000490, loss = 0.001441
grad_step = 000491, loss = 0.001437
grad_step = 000492, loss = 0.001433
grad_step = 000493, loss = 0.001431
grad_step = 000494, loss = 0.001429
grad_step = 000495, loss = 0.001427
grad_step = 000496, loss = 0.001425
grad_step = 000497, loss = 0.001424
grad_step = 000498, loss = 0.001424
grad_step = 000499, loss = 0.001424
grad_step = 000500, loss = 0.001423
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001422
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

  date_run                              2020-05-15 22:13:44.812565
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.28328
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 22:13:44.819926
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.202539
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 22:13:44.829299
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.159258
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 22:13:44.836741
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.07765
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
0   2020-05-15 22:13:11.538416  ...    mean_absolute_error
1   2020-05-15 22:13:11.544169  ...     mean_squared_error
2   2020-05-15 22:13:11.548015  ...  median_absolute_error
3   2020-05-15 22:13:11.551755  ...               r2_score
4   2020-05-15 22:13:21.595245  ...    mean_absolute_error
5   2020-05-15 22:13:21.599677  ...     mean_squared_error
6   2020-05-15 22:13:21.603861  ...  median_absolute_error
7   2020-05-15 22:13:21.607900  ...               r2_score
8   2020-05-15 22:13:44.812565  ...    mean_absolute_error
9   2020-05-15 22:13:44.819926  ...     mean_squared_error
10  2020-05-15 22:13:44.829299  ...  median_absolute_error
11  2020-05-15 22:13:44.836741  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f034639bfd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 314276.92it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 405724.42it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 561043.11it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 792270.81it/s] 75%|███████▌  | 7446528/9912422 [00:00<00:02, 1119436.77it/s]9920512it [00:00, 9923533.32it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 145468.37it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 306957.87it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 396575.01it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 548906.44it/s]1654784it [00:00, 2745356.34it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 52225.83it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02f8d9ce48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02f5be90b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02f5b5e4a8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02f8325080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02f8d9ce48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02f5b49ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02f5b5e4a8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02f82e36a0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02f8d9ce48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02f5be9048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb4b9b9a1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=7f4ccdda8909df78fcd3db5701c49bdfd951f8a9beb95b54456b11ca71cb13cb
  Stored in directory: /tmp/pip-ephem-wheel-cache-t3w3ev6a/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fb4598c3da0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 2:22
   40960/17464789 [..............................] - ETA: 56s 
   90112/17464789 [..............................] - ETA: 38s
  139264/17464789 [..............................] - ETA: 33s
  319488/17464789 [..............................] - ETA: 18s
  647168/17464789 [>.............................] - ETA: 10s
 1310720/17464789 [=>............................] - ETA: 5s 
 2613248/17464789 [===>..........................] - ETA: 3s
 5185536/17464789 [=======>......................] - ETA: 1s
 8216576/17464789 [=============>................] - ETA: 0s
11214848/17464789 [==================>...........] - ETA: 0s
14295040/17464789 [=======================>......] - ETA: 0s
17309696/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 22:15:18.005461: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 22:15:18.009891: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-05-15 22:15:18.010055: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f440da4ba0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 22:15:18.010074: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8353 - accuracy: 0.4890
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7816 - accuracy: 0.4925
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6768 - accuracy: 0.4993 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7011 - accuracy: 0.4978
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6114 - accuracy: 0.5036
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6334 - accuracy: 0.5022
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6338 - accuracy: 0.5021
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6417 - accuracy: 0.5016
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6394 - accuracy: 0.5018
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6636 - accuracy: 0.5002
11000/25000 [============>.................] - ETA: 4s - loss: 7.7084 - accuracy: 0.4973
12000/25000 [=============>................] - ETA: 4s - loss: 7.6781 - accuracy: 0.4992
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6312 - accuracy: 0.5023
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6349 - accuracy: 0.5021
15000/25000 [=================>............] - ETA: 3s - loss: 7.6441 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6302 - accuracy: 0.5024
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6432 - accuracy: 0.5015
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6453 - accuracy: 0.5014
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6610 - accuracy: 0.5004
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6659 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6464 - accuracy: 0.5013
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6560 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
25000/25000 [==============================] - 10s 384us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 22:15:35.303491
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 22:15:35.303491  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<11:50:22, 20.2kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:00<8:18:11, 28.8kB/s]  .vector_cache/glove.6B.zip:   1%|          | 6.45M/862M [00:00<5:46:22, 41.2kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:00<4:00:41, 58.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.7M/862M [00:00<2:46:59, 84.0kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.4M/862M [00:00<1:55:44, 120kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 40.2M/862M [00:01<1:20:00, 171kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.8M/862M [00:01<55:27, 244kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:01<39:18, 343kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<27:53, 482kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<9:11:20, 24.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:03<6:24:42, 34.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<4:34:40, 48.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:05<3:13:05, 69.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<2:16:34, 97.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:07<1:36:59, 137kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.5M/862M [00:07<1:08:02, 195kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.7M/862M [00:07<47:52, 277kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:07<33:53, 391kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<12:45:38, 17.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:09<8:56:44, 24.7kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:09<6:15:28, 35.2kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:09<4:22:45, 50.2kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:11<3:09:13, 69.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:11<2:13:27, 98.6kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:11<1:33:31, 140kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 75.1M/862M [00:11<1:05:44, 200kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:13<51:31, 254kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:13<37:58, 345kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:13<26:51, 487kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:13<19:03, 685kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:15<18:17, 713kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:15<13:52, 938kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:15<10:02, 1.29MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.5M/862M [00:15<07:18, 1.78MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:17<11:02, 1.17MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:17<08:47, 1.47MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:17<06:23, 2.02MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.9M/862M [00:17<04:45, 2.72MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:18<11:30, 1.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:19<09:10, 1.40MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:19<06:41, 1.92MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.9M/862M [00:19<04:56, 2.60MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:20<10:48, 1.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:21<08:56, 1.43MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:21<06:33, 1.95MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.8M/862M [00:21<04:50, 2.63MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:22<08:54, 1.43MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:23<07:20, 1.74MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.9M/862M [00:23<05:21, 2.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:23<03:59, 3.18MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<13:55, 911kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<10:50, 1.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:25<07:48, 1.62MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:25<07:26, 1.70MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<5:37:27, 37.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<3:56:36, 53.4kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<2:45:32, 76.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:26<1:55:56, 109kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<1:31:33, 137kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<1:05:11, 193kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<45:50, 274kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:28<32:17, 388kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<28:29, 439kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<21:37, 578kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<15:22, 812kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:30<10:59, 1.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<14:23, 864kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<11:12, 1.11MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<08:05, 1.53MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:32<05:55, 2.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<11:55, 1.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<09:23, 1.32MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<06:47, 1.82MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:34<04:59, 2.47MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:35<13:29, 912kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<10:48, 1.14MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<07:49, 1.57MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<05:42, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:37<10:12, 1.20MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<08:14, 1.48MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<05:59, 2.04MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:38<04:26, 2.74MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<12:39, 960kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<10:05, 1.20MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<07:17, 1.66MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:40<05:18, 2.28MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<13:49, 875kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<10:41, 1.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<07:44, 1.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<06:37, 1.82MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<5:30:58, 36.4kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:43<3:51:43, 51.9kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:43<2:42:04, 74.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<1:57:31, 102kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:45<1:23:13, 144kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<58:19, 205kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:45<40:57, 291kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<53:19, 223kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<38:18, 311kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<26:59, 440kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:47<19:01, 622kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<58:04, 204kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<42:02, 282kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<29:34, 399kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:49<20:51, 565kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<27:37, 426kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<20:16, 581kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<14:23, 816kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:51<10:14, 1.14MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<30:42, 381kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<22:28, 520kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<15:51, 735kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:53<11:18, 1.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<31:42, 367kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<23:04, 504kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:55<16:18, 711kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:55<11:36, 997kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:56<23:07, 500kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<17:14, 670kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:57<12:15, 940kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<11:36, 990kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<09:10, 1.25MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<06:35, 1.74MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [00:59<04:48, 2.38MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:00<1:06:19, 172kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<47:21, 241kB/s]  .vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<33:14, 343kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<26:26, 431kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<5:38:55, 33.6kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<3:57:19, 47.9kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:02<2:45:47, 68.4kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<2:00:59, 93.5kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<1:26:12, 131kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<1:00:24, 187kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:04<42:19, 266kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<51:46, 217kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<37:00, 304kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:06<26:03, 430kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<21:15, 526kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<15:47, 708kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:08<11:10, 997kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<11:29, 967kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<08:56, 1.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<06:24, 1.73MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<07:54, 1.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<06:18, 1.75MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<04:36, 2.39MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<06:07, 1.79MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<05:14, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<03:49, 2.85MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<05:42, 1.91MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<04:52, 2.23MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:16<03:33, 3.05MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<06:01, 1.80MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<05:09, 2.10MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:18<04:11, 2.57MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<4:46:20, 37.7kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<3:20:23, 53.8kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:19<2:20:00, 76.7kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<1:43:27, 104kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<1:13:14, 146kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<51:16, 208kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<39:08, 272kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<28:08, 378kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<19:50, 535kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<16:35, 638kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<12:26, 851kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<08:53, 1.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<08:44, 1.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<07:11, 1.46MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<05:12, 2.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<06:08, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<05:10, 2.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<03:44, 2.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<06:37, 1.57MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<05:25, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<03:58, 2.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<05:25, 1.90MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<04:38, 2.22MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:33<03:21, 3.05MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<06:39, 1.54MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<05:28, 1.87MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:35<03:56, 2.58MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<07:07, 1.43MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:45, 1.77MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<04:08, 2.44MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:38<06:30, 1.55MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<05:16, 1.91MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<03:48, 2.64MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<06:35, 1.52MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<05:26, 1.84MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:41<03:54, 2.55MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<07:21, 1.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<06:04, 1.64MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<04:21, 2.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<07:20, 1.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<05:50, 1.69MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<04:12, 2.34MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<06:25, 1.53MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<5:22:47, 30.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<3:45:30, 43.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<2:39:22, 61.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<1:53:01, 86.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<1:18:57, 123kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<58:04, 167kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<41:44, 232kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<29:13, 331kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<23:55, 403kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<18:07, 532kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:52<12:44, 753kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<12:41, 754kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<09:40, 988kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:54<06:49, 1.39MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<13:58, 680kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<11:08, 852kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:56<07:53, 1.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<08:54, 1.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<07:00, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:58<04:57, 1.89MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<21:36, 433kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<15:53, 589kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:00<11:08, 835kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<53:34, 174kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<38:10, 243kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:02<26:39, 346kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<26:43, 345kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<19:20, 477kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:04<13:32, 677kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<17:56, 510kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<13:17, 688kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<09:38, 945kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<4:58:44, 30.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<3:28:28, 43.5kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<2:27:49, 61.2kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<1:44:07, 86.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<1:13:44, 122kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<52:13, 172kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:11<36:26, 245kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<32:10, 277kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<23:11, 384kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:13<16:12, 545kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<29:46, 297kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<21:33, 410kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<16:18, 538kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<12:07, 723kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<09:43, 894kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<07:31, 1.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<06:31, 1.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<05:15, 1.64MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<04:56, 1.73MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<04:13, 2.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:23<03:01, 2.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<09:57, 853kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<07:40, 1.10MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:26<06:36, 1.28MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<05:14, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:27<03:43, 2.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<11:36, 720kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<09:01, 925kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:29<06:40, 1.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<4:21:08, 31.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:30<3:01:47, 45.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<2:11:00, 62.9kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<1:32:17, 89.2kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<1:05:21, 125kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<46:21, 176kB/s]  .vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:34<32:20, 251kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<26:57, 300kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<19:57, 405kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:36<13:56, 576kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<18:24, 436kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<13:27, 596kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<10:33, 755kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<07:53, 1.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:40<05:34, 1.42MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<10:26, 756kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<07:56, 993kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<06:41, 1.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<05:08, 1.52MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<04:46, 1.63MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<03:56, 1.96MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<03:53, 1.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<03:13, 2.38MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<03:25, 2.23MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<06:09, 1.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<04:28, 1.70MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<04:34, 1.65MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<03:43, 2.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:46, 1.98MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<02:46, 2.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:55<03:27, 2.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<03:00, 2.46MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<03:12, 2.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<02:42, 2.72MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<02:10, 3.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:59<4:04:37, 29.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<2:50:36, 42.6kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<2:00:35, 59.9kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<1:24:50, 85.1kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<59:58, 119kB/s]   .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<42:23, 169kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<30:30, 232kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<21:51, 324kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<16:14, 432kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<11:51, 591kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<09:17, 748kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<06:59, 992kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<05:54, 1.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<04:37, 1.49MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<04:14, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<03:21, 2.03MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:14<03:22, 1.99MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:48, 2.40MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<02:58, 2.24MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:31, 2.65MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:18<02:45, 2.39MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:21, 2.80MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<02:38, 2.48MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<02:16, 2.87MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:21<01:51, 3.50MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<3:32:48, 30.5kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:22<2:27:41, 43.6kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<1:48:45, 59.1kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<1:16:28, 83.9kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<54:00, 118kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<38:10, 166kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<27:26, 229kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<19:37, 320kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<14:33, 427kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<10:34, 587kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<08:17, 742kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<06:11, 990kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<05:14, 1.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<04:03, 1.49MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<03:44, 1.61MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<03:01, 1.99MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<03:00, 1.97MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:30, 2.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<02:38, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:14, 2.61MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<02:26, 2.37MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<02:05, 2.77MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:42<01:41, 3.39MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<3:08:36, 30.5kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<2:11:16, 43.3kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<1:32:13, 61.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<1:04:42, 86.9kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<45:27, 123kB/s]   .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<32:24, 171kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<22:53, 242kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<16:45, 327kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<12:03, 454kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<09:11, 589kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<06:46, 798kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<05:30, 969kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<04:10, 1.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<03:43, 1.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:55, 1.80MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:58<02:49, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<02:16, 2.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<02:22, 2.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<01:57, 2.63MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:02<02:08, 2.37MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:48, 2.81MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:03<01:27, 3.44MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<2:45:11, 30.4kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:04<1:54:30, 43.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:04<1:20:12, 61.7kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<5:51:56, 14.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<4:06:11, 20.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<2:51:01, 28.6kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<1:59:47, 40.7kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<1:23:35, 57.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<58:41, 81.9kB/s]  .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<41:19, 115kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<29:11, 162kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<20:52, 224kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<14:51, 314kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<10:59, 420kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<07:57, 578kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<06:12, 731kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<04:36, 982kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<03:53, 1.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<02:59, 1.49MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<02:45, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<02:11, 2.00MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<02:11, 1.98MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:47, 2.41MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<01:54, 2.24MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:36, 2.65MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<01:17, 3.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<2:16:57, 30.8kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<1:34:54, 43.7kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<1:06:38, 62.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<46:34, 87.6kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<32:40, 125kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<23:13, 173kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<16:27, 243kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<11:58, 329kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<08:36, 457kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<06:32, 593kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<04:48, 805kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<03:54, 975kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<02:57, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<02:37, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<02:03, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:42<01:59, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<01:36, 2.27MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:44<01:40, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:23, 2.59MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<01:30, 2.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:46<01:16, 2.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<01:01, 3.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<1:54:27, 30.4kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<1:19:02, 43.2kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<55:28, 61.4kB/s]  .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<38:38, 86.6kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:52<27:05, 123kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<19:11, 171kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<13:35, 240kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<09:50, 326kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<07:04, 452kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<05:21, 587kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<03:56, 796kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<03:10, 966kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<02:24, 1.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<02:07, 1.42MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:40, 1.79MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:03<01:36, 1.83MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:17, 2.25MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<01:20, 2.14MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:08, 2.50MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:07<01:12, 2.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:00, 2.74MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<01:07, 2.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<00:56, 2.87MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<00:45, 3.50MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<1:28:02, 30.4kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<1:00:25, 43.2kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<42:23, 61.4kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<29:21, 86.6kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<20:39, 123kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<14:29, 171kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<10:15, 240kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<07:22, 326kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<05:17, 453kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<03:58, 587kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<02:51, 811kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<02:19, 972kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<01:46, 1.28MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<01:32, 1.42MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<01:13, 1.78MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<01:09, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:56, 2.26MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<00:57, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:47, 2.57MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<00:51, 2.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:42, 2.77MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:47, 2.45MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:43, 2.66MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:33<00:34, 3.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<1:00:51, 30.8kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:34<41:05, 44.0kB/s]  .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<39:57, 45.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<28:00, 64.3kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<19:11, 90.6kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<13:27, 128kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<09:21, 178kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<06:39, 250kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<04:43, 339kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<03:25, 466kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<02:31, 606kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<01:51, 822kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:45<01:28, 992kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<01:07, 1.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<00:58, 1.44MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:45, 1.85MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:49<00:42, 1.87MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:34, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:51<00:34, 2.17MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:28, 2.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:52<00:22, 3.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<38:43, 31.3kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<25:44, 44.5kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<17:59, 63.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<12:04, 89.1kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<08:23, 126kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<05:44, 175kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<04:02, 247kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:01<02:48, 334kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<02:00, 463kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<01:27, 597kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<01:04, 797kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:49, 972kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<00:37, 1.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:30, 1.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:23, 1.81MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:21, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<00:17, 2.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:16, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:13, 2.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:13, 2.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:11, 2.76MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:11, 2.46MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:09, 2.70MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<00:09, 2.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:07, 2.95MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:07, 2.54MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:06, 2.93MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:20<00:05, 2.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<00:05, 2.61MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:03, 3.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<07:06, 28.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:22<03:45, 40.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<02:20, 55.7kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<01:33, 79.1kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:33, 111kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:19, 157kB/s].vector_cache/glove.6B.zip: 862MB [06:26, 2.23MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 756/400000 [00:00<00:52, 7535.66it/s]  0%|          | 1403/400000 [00:00<00:55, 7177.97it/s]  1%|          | 2194/400000 [00:00<00:53, 7381.39it/s]  1%|          | 2972/400000 [00:00<00:52, 7496.49it/s]  1%|          | 3649/400000 [00:00<00:54, 7259.40it/s]  1%|          | 4360/400000 [00:00<00:54, 7205.82it/s]  1%|▏         | 5060/400000 [00:00<00:55, 7142.60it/s]  1%|▏         | 5808/400000 [00:00<00:54, 7239.78it/s]  2%|▏         | 6502/400000 [00:00<00:55, 7145.22it/s]  2%|▏         | 7188/400000 [00:01<00:55, 7044.11it/s]  2%|▏         | 7968/400000 [00:01<00:54, 7254.51it/s]  2%|▏         | 8759/400000 [00:01<00:52, 7438.31it/s]  2%|▏         | 9496/400000 [00:01<00:53, 7252.42it/s]  3%|▎         | 10218/400000 [00:01<00:54, 7088.42it/s]  3%|▎         | 10925/400000 [00:01<00:56, 6902.28it/s]  3%|▎         | 11674/400000 [00:01<00:54, 7068.67it/s]  3%|▎         | 12440/400000 [00:01<00:53, 7236.08it/s]  3%|▎         | 13166/400000 [00:01<00:54, 7039.47it/s]  3%|▎         | 13920/400000 [00:01<00:53, 7181.24it/s]  4%|▎         | 14679/400000 [00:02<00:52, 7297.51it/s]  4%|▍         | 15463/400000 [00:02<00:51, 7450.06it/s]  4%|▍         | 16252/400000 [00:02<00:50, 7575.01it/s]  4%|▍         | 17029/400000 [00:02<00:50, 7631.02it/s]  4%|▍         | 17794/400000 [00:02<00:50, 7615.27it/s]  5%|▍         | 18557/400000 [00:02<00:50, 7573.78it/s]  5%|▍         | 19328/400000 [00:02<00:50, 7612.90it/s]  5%|▌         | 20101/400000 [00:02<00:49, 7647.38it/s]  5%|▌         | 20867/400000 [00:02<00:49, 7605.98it/s]  5%|▌         | 21633/400000 [00:02<00:49, 7619.75it/s]  6%|▌         | 22396/400000 [00:03<00:49, 7596.11it/s]  6%|▌         | 23167/400000 [00:03<00:49, 7629.02it/s]  6%|▌         | 23948/400000 [00:03<00:48, 7682.33it/s]  6%|▌         | 24717/400000 [00:03<00:49, 7606.85it/s]  6%|▋         | 25479/400000 [00:03<00:50, 7405.45it/s]  7%|▋         | 26221/400000 [00:03<00:50, 7385.80it/s]  7%|▋         | 26961/400000 [00:03<00:52, 7127.92it/s]  7%|▋         | 27677/400000 [00:03<00:54, 6894.78it/s]  7%|▋         | 28379/400000 [00:03<00:53, 6929.76it/s]  7%|▋         | 29155/400000 [00:03<00:51, 7158.71it/s]  7%|▋         | 29915/400000 [00:04<00:50, 7284.76it/s]  8%|▊         | 30647/400000 [00:04<00:50, 7264.89it/s]  8%|▊         | 31434/400000 [00:04<00:49, 7434.59it/s]  8%|▊         | 32187/400000 [00:04<00:49, 7460.01it/s]  8%|▊         | 32935/400000 [00:04<00:49, 7389.44it/s]  8%|▊         | 33676/400000 [00:04<00:50, 7216.37it/s]  9%|▊         | 34432/400000 [00:04<00:49, 7315.41it/s]  9%|▉         | 35208/400000 [00:04<00:49, 7441.20it/s]  9%|▉         | 35985/400000 [00:04<00:48, 7535.88it/s]  9%|▉         | 36741/400000 [00:04<00:48, 7428.07it/s]  9%|▉         | 37486/400000 [00:05<00:50, 7117.75it/s] 10%|▉         | 38202/400000 [00:05<00:51, 7010.04it/s] 10%|▉         | 38976/400000 [00:05<00:50, 7211.96it/s] 10%|▉         | 39762/400000 [00:05<00:48, 7394.24it/s] 10%|█         | 40506/400000 [00:05<00:49, 7245.25it/s] 10%|█         | 41234/400000 [00:05<00:51, 6974.60it/s] 10%|█         | 41946/400000 [00:05<00:51, 7015.30it/s] 11%|█         | 42717/400000 [00:05<00:49, 7207.85it/s] 11%|█         | 43469/400000 [00:05<00:48, 7297.93it/s] 11%|█         | 44241/400000 [00:06<00:47, 7418.60it/s] 11%|█         | 44986/400000 [00:06<00:48, 7346.17it/s] 11%|█▏        | 45745/400000 [00:06<00:47, 7417.38it/s] 12%|█▏        | 46506/400000 [00:06<00:47, 7472.17it/s] 12%|█▏        | 47276/400000 [00:06<00:46, 7529.42it/s] 12%|█▏        | 48046/400000 [00:06<00:46, 7578.90it/s] 12%|█▏        | 48805/400000 [00:06<00:46, 7475.67it/s] 12%|█▏        | 49554/400000 [00:06<00:48, 7285.97it/s] 13%|█▎        | 50348/400000 [00:06<00:46, 7469.63it/s] 13%|█▎        | 51098/400000 [00:06<00:48, 7225.43it/s] 13%|█▎        | 51855/400000 [00:07<00:47, 7325.33it/s] 13%|█▎        | 52591/400000 [00:07<00:48, 7217.04it/s] 13%|█▎        | 53315/400000 [00:07<00:48, 7099.23it/s] 14%|█▎        | 54030/400000 [00:07<00:48, 7113.71it/s] 14%|█▎        | 54809/400000 [00:07<00:47, 7301.56it/s] 14%|█▍        | 55542/400000 [00:07<00:47, 7270.06it/s] 14%|█▍        | 56308/400000 [00:07<00:46, 7380.23it/s] 14%|█▍        | 57098/400000 [00:07<00:45, 7528.50it/s] 14%|█▍        | 57853/400000 [00:07<00:45, 7460.30it/s] 15%|█▍        | 58601/400000 [00:07<00:45, 7432.25it/s] 15%|█▍        | 59387/400000 [00:08<00:45, 7551.77it/s] 15%|█▌        | 60149/400000 [00:08<00:44, 7570.29it/s] 15%|█▌        | 60912/400000 [00:08<00:44, 7585.70it/s] 15%|█▌        | 61672/400000 [00:08<00:46, 7350.88it/s] 16%|█▌        | 62410/400000 [00:08<00:47, 7169.51it/s] 16%|█▌        | 63130/400000 [00:08<00:47, 7065.35it/s] 16%|█▌        | 63839/400000 [00:08<00:47, 7063.04it/s] 16%|█▌        | 64640/400000 [00:08<00:45, 7322.04it/s] 16%|█▋        | 65376/400000 [00:08<00:46, 7262.90it/s] 17%|█▋        | 66117/400000 [00:09<00:45, 7306.29it/s] 17%|█▋        | 66850/400000 [00:09<00:45, 7266.95it/s] 17%|█▋        | 67579/400000 [00:09<00:47, 6981.36it/s] 17%|█▋        | 68345/400000 [00:09<00:46, 7171.63it/s] 17%|█▋        | 69134/400000 [00:09<00:44, 7371.34it/s] 17%|█▋        | 69922/400000 [00:09<00:43, 7515.86it/s] 18%|█▊        | 70678/400000 [00:09<00:44, 7357.71it/s] 18%|█▊        | 71417/400000 [00:09<00:45, 7181.75it/s] 18%|█▊        | 72139/400000 [00:09<00:45, 7155.44it/s] 18%|█▊        | 72893/400000 [00:09<00:45, 7266.27it/s] 18%|█▊        | 73668/400000 [00:10<00:44, 7403.36it/s] 19%|█▊        | 74440/400000 [00:10<00:43, 7493.66it/s] 19%|█▉        | 75221/400000 [00:10<00:42, 7585.23it/s] 19%|█▉        | 75991/400000 [00:10<00:42, 7618.23it/s] 19%|█▉        | 76758/400000 [00:10<00:42, 7632.69it/s] 19%|█▉        | 77554/400000 [00:10<00:41, 7727.49it/s] 20%|█▉        | 78328/400000 [00:10<00:41, 7660.51it/s] 20%|█▉        | 79119/400000 [00:10<00:41, 7731.13it/s] 20%|█▉        | 79893/400000 [00:10<00:42, 7568.35it/s] 20%|██        | 80659/400000 [00:10<00:42, 7594.45it/s] 20%|██        | 81447/400000 [00:11<00:41, 7676.17it/s] 21%|██        | 82216/400000 [00:11<00:43, 7357.14it/s] 21%|██        | 82962/400000 [00:11<00:42, 7385.97it/s] 21%|██        | 83715/400000 [00:11<00:42, 7424.90it/s] 21%|██        | 84460/400000 [00:11<00:43, 7210.75it/s] 21%|██▏       | 85184/400000 [00:11<00:45, 6908.03it/s] 21%|██▏       | 85880/400000 [00:11<00:46, 6784.26it/s] 22%|██▏       | 86609/400000 [00:11<00:45, 6925.83it/s] 22%|██▏       | 87348/400000 [00:11<00:44, 7057.37it/s] 22%|██▏       | 88098/400000 [00:12<00:43, 7182.88it/s] 22%|██▏       | 88884/400000 [00:12<00:42, 7371.10it/s] 22%|██▏       | 89634/400000 [00:12<00:41, 7407.62it/s] 23%|██▎       | 90377/400000 [00:12<00:41, 7372.59it/s] 23%|██▎       | 91151/400000 [00:12<00:41, 7473.69it/s] 23%|██▎       | 91900/400000 [00:12<00:41, 7336.66it/s] 23%|██▎       | 92636/400000 [00:12<00:42, 7314.90it/s] 23%|██▎       | 93369/400000 [00:12<00:43, 7096.86it/s] 24%|██▎       | 94081/400000 [00:12<00:43, 6974.32it/s] 24%|██▎       | 94781/400000 [00:12<00:44, 6918.78it/s] 24%|██▍       | 95475/400000 [00:13<00:44, 6890.69it/s] 24%|██▍       | 96189/400000 [00:13<00:43, 6961.20it/s] 24%|██▍       | 96887/400000 [00:13<00:44, 6810.49it/s] 24%|██▍       | 97570/400000 [00:13<00:44, 6775.65it/s] 25%|██▍       | 98261/400000 [00:13<00:44, 6813.89it/s] 25%|██▍       | 98954/400000 [00:13<00:43, 6845.95it/s] 25%|██▍       | 99640/400000 [00:13<00:43, 6834.61it/s] 25%|██▌       | 100377/400000 [00:13<00:42, 6986.01it/s] 25%|██▌       | 101123/400000 [00:13<00:41, 7121.37it/s] 25%|██▌       | 101837/400000 [00:13<00:43, 6869.61it/s] 26%|██▌       | 102579/400000 [00:14<00:42, 7025.78it/s] 26%|██▌       | 103339/400000 [00:14<00:41, 7187.27it/s] 26%|██▌       | 104076/400000 [00:14<00:40, 7239.86it/s] 26%|██▌       | 104865/400000 [00:14<00:39, 7422.52it/s] 26%|██▋       | 105610/400000 [00:14<00:39, 7419.98it/s] 27%|██▋       | 106357/400000 [00:14<00:39, 7434.90it/s] 27%|██▋       | 107102/400000 [00:14<00:39, 7414.43it/s] 27%|██▋       | 107845/400000 [00:14<00:39, 7385.32it/s] 27%|██▋       | 108585/400000 [00:14<00:40, 7137.97it/s] 27%|██▋       | 109302/400000 [00:14<00:41, 7083.84it/s] 28%|██▊       | 110013/400000 [00:15<00:41, 7030.15it/s] 28%|██▊       | 110718/400000 [00:15<00:41, 6993.78it/s] 28%|██▊       | 111435/400000 [00:15<00:40, 7043.53it/s] 28%|██▊       | 112141/400000 [00:15<00:40, 7027.82it/s] 28%|██▊       | 112903/400000 [00:15<00:39, 7195.44it/s] 28%|██▊       | 113674/400000 [00:15<00:39, 7340.57it/s] 29%|██▊       | 114454/400000 [00:15<00:38, 7472.44it/s] 29%|██▉       | 115203/400000 [00:15<00:38, 7463.33it/s] 29%|██▉       | 115951/400000 [00:15<00:38, 7418.63it/s] 29%|██▉       | 116720/400000 [00:15<00:37, 7495.83it/s] 29%|██▉       | 117471/400000 [00:16<00:39, 7159.92it/s] 30%|██▉       | 118249/400000 [00:16<00:38, 7333.12it/s] 30%|██▉       | 118986/400000 [00:16<00:38, 7339.27it/s] 30%|██▉       | 119757/400000 [00:16<00:37, 7446.20it/s] 30%|███       | 120504/400000 [00:16<00:37, 7375.51it/s] 30%|███       | 121244/400000 [00:16<00:38, 7179.59it/s] 31%|███       | 122015/400000 [00:16<00:37, 7329.70it/s] 31%|███       | 122751/400000 [00:16<00:39, 7055.41it/s] 31%|███       | 123461/400000 [00:16<00:39, 7012.22it/s] 31%|███       | 124165/400000 [00:17<00:40, 6833.85it/s] 31%|███       | 124852/400000 [00:17<00:40, 6811.68it/s] 31%|███▏      | 125618/400000 [00:17<00:38, 7044.80it/s] 32%|███▏      | 126368/400000 [00:17<00:38, 7173.75it/s] 32%|███▏      | 127110/400000 [00:17<00:37, 7244.93it/s] 32%|███▏      | 127881/400000 [00:17<00:36, 7377.30it/s] 32%|███▏      | 128653/400000 [00:17<00:36, 7472.81it/s] 32%|███▏      | 129403/400000 [00:17<00:36, 7449.11it/s] 33%|███▎      | 130167/400000 [00:17<00:35, 7499.22it/s] 33%|███▎      | 130918/400000 [00:17<00:36, 7446.05it/s] 33%|███▎      | 131672/400000 [00:18<00:35, 7459.52it/s] 33%|███▎      | 132419/400000 [00:18<00:37, 7215.64it/s] 33%|███▎      | 133143/400000 [00:18<00:37, 7071.05it/s] 33%|███▎      | 133890/400000 [00:18<00:37, 7185.04it/s] 34%|███▎      | 134678/400000 [00:18<00:35, 7378.94it/s] 34%|███▍      | 135446/400000 [00:18<00:35, 7464.48it/s] 34%|███▍      | 136197/400000 [00:18<00:35, 7475.91it/s] 34%|███▍      | 136969/400000 [00:18<00:34, 7546.25it/s] 34%|███▍      | 137743/400000 [00:18<00:34, 7600.74it/s] 35%|███▍      | 138510/400000 [00:18<00:34, 7619.56it/s] 35%|███▍      | 139297/400000 [00:19<00:33, 7692.25it/s] 35%|███▌      | 140067/400000 [00:19<00:34, 7547.46it/s] 35%|███▌      | 140823/400000 [00:19<00:34, 7510.13it/s] 35%|███▌      | 141575/400000 [00:19<00:34, 7484.21it/s] 36%|███▌      | 142343/400000 [00:19<00:34, 7541.27it/s] 36%|███▌      | 143117/400000 [00:19<00:33, 7598.76it/s] 36%|███▌      | 143878/400000 [00:19<00:35, 7182.97it/s] 36%|███▌      | 144638/400000 [00:19<00:34, 7302.55it/s] 36%|███▋      | 145395/400000 [00:19<00:34, 7380.48it/s] 37%|███▋      | 146183/400000 [00:20<00:33, 7521.21it/s] 37%|███▋      | 146938/400000 [00:20<00:33, 7450.60it/s] 37%|███▋      | 147727/400000 [00:20<00:33, 7575.81it/s] 37%|███▋      | 148487/400000 [00:20<00:34, 7350.51it/s] 37%|███▋      | 149225/400000 [00:20<00:34, 7249.32it/s] 37%|███▋      | 149958/400000 [00:20<00:34, 7270.63it/s] 38%|███▊      | 150750/400000 [00:20<00:33, 7453.60it/s] 38%|███▊      | 151498/400000 [00:20<00:33, 7399.08it/s] 38%|███▊      | 152265/400000 [00:20<00:33, 7477.18it/s] 38%|███▊      | 153015/400000 [00:20<00:33, 7297.86it/s] 38%|███▊      | 153775/400000 [00:21<00:33, 7384.62it/s] 39%|███▊      | 154520/400000 [00:21<00:33, 7401.89it/s] 39%|███▉      | 155262/400000 [00:21<00:34, 7141.01it/s] 39%|███▉      | 156023/400000 [00:21<00:33, 7274.02it/s] 39%|███▉      | 156770/400000 [00:21<00:33, 7331.70it/s] 39%|███▉      | 157554/400000 [00:21<00:32, 7476.57it/s] 40%|███▉      | 158340/400000 [00:21<00:31, 7587.21it/s] 40%|███▉      | 159101/400000 [00:21<00:31, 7535.66it/s] 40%|███▉      | 159879/400000 [00:21<00:31, 7605.04it/s] 40%|████      | 160641/400000 [00:21<00:31, 7600.02it/s] 40%|████      | 161402/400000 [00:22<00:32, 7333.31it/s] 41%|████      | 162138/400000 [00:22<00:32, 7334.65it/s] 41%|████      | 162897/400000 [00:22<00:32, 7409.01it/s] 41%|████      | 163640/400000 [00:22<00:31, 7406.56it/s] 41%|████      | 164382/400000 [00:22<00:31, 7388.26it/s] 41%|████▏     | 165122/400000 [00:22<00:32, 7193.74it/s] 41%|████▏     | 165843/400000 [00:22<00:33, 7050.94it/s] 42%|████▏     | 166554/400000 [00:22<00:33, 7068.14it/s] 42%|████▏     | 167320/400000 [00:22<00:32, 7235.60it/s] 42%|████▏     | 168064/400000 [00:22<00:31, 7294.69it/s] 42%|████▏     | 168811/400000 [00:23<00:31, 7346.34it/s] 42%|████▏     | 169547/400000 [00:23<00:31, 7201.68it/s] 43%|████▎     | 170269/400000 [00:23<00:32, 7000.50it/s] 43%|████▎     | 170972/400000 [00:23<00:32, 6943.50it/s] 43%|████▎     | 171668/400000 [00:23<00:32, 6943.69it/s] 43%|████▎     | 172364/400000 [00:23<00:32, 6939.56it/s] 43%|████▎     | 173059/400000 [00:23<00:32, 6888.18it/s] 43%|████▎     | 173813/400000 [00:23<00:31, 7070.58it/s] 44%|████▎     | 174522/400000 [00:23<00:32, 7001.57it/s] 44%|████▍     | 175245/400000 [00:24<00:31, 7066.10it/s] 44%|████▍     | 176011/400000 [00:24<00:30, 7233.43it/s] 44%|████▍     | 176737/400000 [00:24<00:30, 7240.95it/s] 44%|████▍     | 177463/400000 [00:24<00:31, 7163.10it/s] 45%|████▍     | 178181/400000 [00:24<00:30, 7158.27it/s] 45%|████▍     | 178898/400000 [00:24<00:31, 7024.30it/s] 45%|████▍     | 179663/400000 [00:24<00:30, 7199.02it/s] 45%|████▌     | 180407/400000 [00:24<00:30, 7260.05it/s] 45%|████▌     | 181135/400000 [00:24<00:30, 7249.18it/s] 45%|████▌     | 181924/400000 [00:24<00:29, 7428.42it/s] 46%|████▌     | 182684/400000 [00:25<00:29, 7475.87it/s] 46%|████▌     | 183463/400000 [00:25<00:28, 7564.51it/s] 46%|████▌     | 184221/400000 [00:25<00:29, 7388.71it/s] 46%|████▌     | 184962/400000 [00:25<00:29, 7247.81it/s] 46%|████▋     | 185689/400000 [00:25<00:29, 7150.20it/s] 47%|████▋     | 186427/400000 [00:25<00:29, 7216.76it/s] 47%|████▋     | 187212/400000 [00:25<00:28, 7394.51it/s] 47%|████▋     | 187994/400000 [00:25<00:28, 7515.14it/s] 47%|████▋     | 188792/400000 [00:25<00:27, 7646.51it/s] 47%|████▋     | 189565/400000 [00:25<00:27, 7669.79it/s] 48%|████▊     | 190336/400000 [00:26<00:27, 7681.64it/s] 48%|████▊     | 191111/400000 [00:26<00:27, 7701.31it/s] 48%|████▊     | 191882/400000 [00:26<00:28, 7328.90it/s] 48%|████▊     | 192620/400000 [00:26<00:29, 7129.63it/s] 48%|████▊     | 193338/400000 [00:26<00:29, 7026.66it/s] 49%|████▊     | 194044/400000 [00:26<00:29, 6973.83it/s] 49%|████▊     | 194760/400000 [00:26<00:29, 7028.34it/s] 49%|████▉     | 195539/400000 [00:26<00:28, 7238.81it/s] 49%|████▉     | 196266/400000 [00:26<00:28, 7069.78it/s] 49%|████▉     | 196992/400000 [00:26<00:28, 7123.97it/s] 49%|████▉     | 197759/400000 [00:27<00:27, 7277.93it/s] 50%|████▉     | 198553/400000 [00:27<00:26, 7463.76it/s] 50%|████▉     | 199355/400000 [00:27<00:26, 7620.41it/s] 50%|█████     | 200120/400000 [00:27<00:26, 7542.69it/s] 50%|█████     | 200877/400000 [00:27<00:26, 7390.53it/s] 50%|█████     | 201656/400000 [00:27<00:26, 7504.21it/s] 51%|█████     | 202409/400000 [00:27<00:27, 7249.98it/s] 51%|█████     | 203179/400000 [00:27<00:26, 7378.27it/s] 51%|█████     | 203922/400000 [00:27<00:26, 7393.49it/s] 51%|█████     | 204704/400000 [00:28<00:25, 7513.53it/s] 51%|█████▏    | 205465/400000 [00:28<00:25, 7540.40it/s] 52%|█████▏    | 206242/400000 [00:28<00:25, 7607.11it/s] 52%|█████▏    | 207004/400000 [00:28<00:25, 7434.64it/s] 52%|█████▏    | 207750/400000 [00:28<00:26, 7333.07it/s] 52%|█████▏    | 208485/400000 [00:28<00:26, 7201.11it/s] 52%|█████▏    | 209250/400000 [00:28<00:26, 7330.09it/s] 53%|█████▎    | 210009/400000 [00:28<00:25, 7406.17it/s] 53%|█████▎    | 210806/400000 [00:28<00:25, 7565.77it/s] 53%|█████▎    | 211591/400000 [00:28<00:24, 7647.70it/s] 53%|█████▎    | 212358/400000 [00:29<00:25, 7505.28it/s] 53%|█████▎    | 213134/400000 [00:29<00:24, 7579.48it/s] 53%|█████▎    | 213894/400000 [00:29<00:25, 7317.54it/s] 54%|█████▎    | 214629/400000 [00:29<00:26, 6974.76it/s] 54%|█████▍    | 215332/400000 [00:29<00:26, 6939.38it/s] 54%|█████▍    | 216030/400000 [00:29<00:27, 6671.73it/s] 54%|█████▍    | 216719/400000 [00:29<00:27, 6733.02it/s] 54%|█████▍    | 217396/400000 [00:29<00:27, 6695.66it/s] 55%|█████▍    | 218145/400000 [00:29<00:26, 6915.15it/s] 55%|█████▍    | 218904/400000 [00:29<00:25, 7103.38it/s] 55%|█████▍    | 219634/400000 [00:30<00:25, 7159.79it/s] 55%|█████▌    | 220383/400000 [00:30<00:24, 7255.02it/s] 55%|█████▌    | 221159/400000 [00:30<00:24, 7399.15it/s] 55%|█████▌    | 221917/400000 [00:30<00:23, 7451.91it/s] 56%|█████▌    | 222698/400000 [00:30<00:23, 7553.96it/s] 56%|█████▌    | 223458/400000 [00:30<00:23, 7567.38it/s] 56%|█████▌    | 224216/400000 [00:30<00:23, 7530.61it/s] 56%|█████▌    | 224989/400000 [00:30<00:23, 7587.40it/s] 56%|█████▋    | 225749/400000 [00:30<00:23, 7479.48it/s] 57%|█████▋    | 226507/400000 [00:30<00:23, 7508.79it/s] 57%|█████▋    | 227259/400000 [00:31<00:23, 7382.19it/s] 57%|█████▋    | 227999/400000 [00:31<00:23, 7224.19it/s] 57%|█████▋    | 228725/400000 [00:31<00:23, 7231.22it/s] 57%|█████▋    | 229466/400000 [00:31<00:23, 7282.87it/s] 58%|█████▊    | 230196/400000 [00:31<00:24, 7019.72it/s] 58%|█████▊    | 230901/400000 [00:31<00:24, 6799.71it/s] 58%|█████▊    | 231585/400000 [00:31<00:24, 6738.55it/s] 58%|█████▊    | 232262/400000 [00:31<00:24, 6736.87it/s] 58%|█████▊    | 232938/400000 [00:31<00:24, 6728.47it/s] 58%|█████▊    | 233613/400000 [00:32<00:24, 6687.42it/s] 59%|█████▊    | 234283/400000 [00:32<00:25, 6620.40it/s] 59%|█████▉    | 235062/400000 [00:32<00:23, 6932.40it/s] 59%|█████▉    | 235857/400000 [00:32<00:22, 7208.08it/s] 59%|█████▉    | 236622/400000 [00:32<00:22, 7334.61it/s] 59%|█████▉    | 237361/400000 [00:32<00:22, 7239.37it/s] 60%|█████▉    | 238089/400000 [00:32<00:23, 6858.43it/s] 60%|█████▉    | 238782/400000 [00:32<00:24, 6653.73it/s] 60%|█████▉    | 239454/400000 [00:32<00:24, 6532.43it/s] 60%|██████    | 240139/400000 [00:32<00:24, 6623.14it/s] 60%|██████    | 240902/400000 [00:33<00:23, 6894.02it/s] 60%|██████    | 241622/400000 [00:33<00:22, 6981.03it/s] 61%|██████    | 242325/400000 [00:33<00:23, 6842.16it/s] 61%|██████    | 243020/400000 [00:33<00:22, 6872.25it/s] 61%|██████    | 243792/400000 [00:33<00:21, 7104.16it/s] 61%|██████    | 244524/400000 [00:33<00:21, 7167.27it/s] 61%|██████▏   | 245284/400000 [00:33<00:21, 7290.24it/s] 62%|██████▏   | 246016/400000 [00:33<00:21, 7126.83it/s] 62%|██████▏   | 246776/400000 [00:33<00:21, 7261.28it/s] 62%|██████▏   | 247505/400000 [00:33<00:21, 7217.18it/s] 62%|██████▏   | 248229/400000 [00:34<00:21, 7183.25it/s] 62%|██████▏   | 248967/400000 [00:34<00:20, 7240.51it/s] 62%|██████▏   | 249767/400000 [00:34<00:20, 7452.44it/s] 63%|██████▎   | 250515/400000 [00:34<00:20, 7143.36it/s] 63%|██████▎   | 251278/400000 [00:34<00:20, 7282.26it/s] 63%|██████▎   | 252061/400000 [00:34<00:19, 7434.65it/s] 63%|██████▎   | 252876/400000 [00:34<00:19, 7635.60it/s] 63%|██████▎   | 253645/400000 [00:34<00:19, 7651.42it/s] 64%|██████▎   | 254413/400000 [00:34<00:19, 7300.55it/s] 64%|██████▍   | 255149/400000 [00:35<00:19, 7260.18it/s] 64%|██████▍   | 255893/400000 [00:35<00:19, 7313.00it/s] 64%|██████▍   | 256664/400000 [00:35<00:19, 7426.80it/s] 64%|██████▍   | 257409/400000 [00:35<00:19, 7295.79it/s] 65%|██████▍   | 258225/400000 [00:35<00:18, 7533.38it/s] 65%|██████▍   | 258995/400000 [00:35<00:18, 7581.24it/s] 65%|██████▍   | 259756/400000 [00:35<00:18, 7430.06it/s] 65%|██████▌   | 260523/400000 [00:35<00:18, 7499.77it/s] 65%|██████▌   | 261326/400000 [00:35<00:18, 7649.10it/s] 66%|██████▌   | 262093/400000 [00:35<00:18, 7441.77it/s] 66%|██████▌   | 262840/400000 [00:36<00:19, 7098.08it/s] 66%|██████▌   | 263556/400000 [00:36<00:19, 6965.15it/s] 66%|██████▌   | 264326/400000 [00:36<00:18, 7169.01it/s] 66%|██████▋   | 265077/400000 [00:36<00:18, 7265.15it/s] 66%|██████▋   | 265807/400000 [00:36<00:18, 7271.95it/s] 67%|██████▋   | 266578/400000 [00:36<00:18, 7395.61it/s] 67%|██████▋   | 267333/400000 [00:36<00:17, 7440.65it/s] 67%|██████▋   | 268079/400000 [00:36<00:18, 7213.70it/s] 67%|██████▋   | 268803/400000 [00:36<00:18, 7048.46it/s] 67%|██████▋   | 269595/400000 [00:37<00:17, 7287.47it/s] 68%|██████▊   | 270328/400000 [00:37<00:18, 7005.52it/s] 68%|██████▊   | 271034/400000 [00:37<00:18, 6818.05it/s] 68%|██████▊   | 271798/400000 [00:37<00:18, 7044.92it/s] 68%|██████▊   | 272532/400000 [00:37<00:17, 7128.67it/s] 68%|██████▊   | 273249/400000 [00:37<00:18, 6954.04it/s] 68%|██████▊   | 273949/400000 [00:37<00:18, 6698.21it/s] 69%|██████▊   | 274677/400000 [00:37<00:18, 6859.72it/s] 69%|██████▉   | 275437/400000 [00:37<00:17, 7065.42it/s] 69%|██████▉   | 276149/400000 [00:37<00:17, 7076.91it/s] 69%|██████▉   | 276860/400000 [00:38<00:17, 7037.94it/s] 69%|██████▉   | 277567/400000 [00:38<00:17, 6842.59it/s] 70%|██████▉   | 278327/400000 [00:38<00:17, 7051.90it/s] 70%|██████▉   | 279084/400000 [00:38<00:16, 7197.50it/s] 70%|██████▉   | 279807/400000 [00:38<00:16, 7087.31it/s] 70%|███████   | 280537/400000 [00:38<00:16, 7148.20it/s] 70%|███████   | 281316/400000 [00:38<00:16, 7327.07it/s] 71%|███████   | 282078/400000 [00:38<00:15, 7412.36it/s] 71%|███████   | 282836/400000 [00:38<00:15, 7460.91it/s] 71%|███████   | 283584/400000 [00:38<00:15, 7307.90it/s] 71%|███████   | 284323/400000 [00:39<00:15, 7331.55it/s] 71%|███████▏  | 285058/400000 [00:39<00:15, 7287.07it/s] 71%|███████▏  | 285788/400000 [00:39<00:16, 6879.34it/s] 72%|███████▏  | 286482/400000 [00:39<00:16, 6828.04it/s] 72%|███████▏  | 287231/400000 [00:39<00:16, 7013.49it/s] 72%|███████▏  | 288042/400000 [00:39<00:15, 7300.30it/s] 72%|███████▏  | 288806/400000 [00:39<00:15, 7397.47it/s] 72%|███████▏  | 289604/400000 [00:39<00:14, 7561.76it/s] 73%|███████▎  | 290365/400000 [00:39<00:14, 7343.62it/s] 73%|███████▎  | 291109/400000 [00:40<00:14, 7369.71it/s] 73%|███████▎  | 291849/400000 [00:40<00:15, 7168.82it/s] 73%|███████▎  | 292592/400000 [00:40<00:14, 7242.85it/s] 73%|███████▎  | 293393/400000 [00:40<00:14, 7455.18it/s] 74%|███████▎  | 294171/400000 [00:40<00:14, 7548.83it/s] 74%|███████▎  | 294933/400000 [00:40<00:13, 7569.82it/s] 74%|███████▍  | 295750/400000 [00:40<00:13, 7738.93it/s] 74%|███████▍  | 296527/400000 [00:40<00:13, 7706.44it/s] 74%|███████▍  | 297300/400000 [00:40<00:13, 7566.02it/s] 75%|███████▍  | 298059/400000 [00:40<00:13, 7494.12it/s] 75%|███████▍  | 298851/400000 [00:41<00:13, 7615.58it/s] 75%|███████▍  | 299614/400000 [00:41<00:13, 7405.75it/s] 75%|███████▌  | 300371/400000 [00:41<00:13, 7451.40it/s] 75%|███████▌  | 301118/400000 [00:41<00:13, 7413.00it/s] 75%|███████▌  | 301861/400000 [00:41<00:13, 7185.32it/s] 76%|███████▌  | 302630/400000 [00:41<00:13, 7325.84it/s] 76%|███████▌  | 303365/400000 [00:41<00:13, 7224.58it/s] 76%|███████▌  | 304090/400000 [00:41<00:13, 7103.64it/s] 76%|███████▌  | 304803/400000 [00:41<00:14, 6795.55it/s] 76%|███████▋  | 305560/400000 [00:41<00:13, 7010.60it/s] 77%|███████▋  | 306345/400000 [00:42<00:12, 7240.64it/s] 77%|███████▋  | 307129/400000 [00:42<00:12, 7409.86it/s] 77%|███████▋  | 307875/400000 [00:42<00:12, 7370.66it/s] 77%|███████▋  | 308651/400000 [00:42<00:12, 7482.62it/s] 77%|███████▋  | 309402/400000 [00:42<00:12, 7273.15it/s] 78%|███████▊  | 310171/400000 [00:42<00:12, 7392.18it/s] 78%|███████▊  | 310929/400000 [00:42<00:11, 7444.88it/s] 78%|███████▊  | 311676/400000 [00:42<00:11, 7375.35it/s] 78%|███████▊  | 312416/400000 [00:42<00:11, 7341.04it/s] 78%|███████▊  | 313212/400000 [00:43<00:11, 7516.21it/s] 78%|███████▊  | 313966/400000 [00:43<00:11, 7423.30it/s] 79%|███████▊  | 314710/400000 [00:43<00:11, 7330.35it/s] 79%|███████▉  | 315453/400000 [00:43<00:11, 7357.50it/s] 79%|███████▉  | 316200/400000 [00:43<00:11, 7390.64it/s] 79%|███████▉  | 316940/400000 [00:43<00:11, 7023.72it/s] 79%|███████▉  | 317695/400000 [00:43<00:11, 7172.39it/s] 80%|███████▉  | 318417/400000 [00:43<00:11, 7045.26it/s] 80%|███████▉  | 319125/400000 [00:43<00:11, 6972.78it/s] 80%|███████▉  | 319825/400000 [00:43<00:11, 6943.08it/s] 80%|████████  | 320522/400000 [00:44<00:11, 6791.25it/s] 80%|████████  | 321220/400000 [00:44<00:11, 6844.94it/s] 80%|████████  | 321940/400000 [00:44<00:11, 6946.57it/s] 81%|████████  | 322711/400000 [00:44<00:10, 7157.15it/s] 81%|████████  | 323457/400000 [00:44<00:10, 7243.17it/s] 81%|████████  | 324196/400000 [00:44<00:10, 7286.35it/s] 81%|████████  | 324960/400000 [00:44<00:10, 7388.29it/s] 81%|████████▏ | 325720/400000 [00:44<00:09, 7449.76it/s] 82%|████████▏ | 326472/400000 [00:44<00:09, 7469.95it/s] 82%|████████▏ | 327220/400000 [00:44<00:10, 7181.47it/s] 82%|████████▏ | 328016/400000 [00:45<00:09, 7396.95it/s] 82%|████████▏ | 328813/400000 [00:45<00:09, 7558.49it/s] 82%|████████▏ | 329573/400000 [00:45<00:09, 7294.64it/s] 83%|████████▎ | 330307/400000 [00:45<00:09, 7206.42it/s] 83%|████████▎ | 331031/400000 [00:45<00:09, 7086.04it/s] 83%|████████▎ | 331813/400000 [00:45<00:09, 7289.67it/s] 83%|████████▎ | 332546/400000 [00:45<00:09, 7236.71it/s] 83%|████████▎ | 333273/400000 [00:45<00:09, 6999.68it/s] 83%|████████▎ | 333977/400000 [00:45<00:09, 6863.61it/s] 84%|████████▎ | 334667/400000 [00:46<00:09, 6791.09it/s] 84%|████████▍ | 335361/400000 [00:46<00:09, 6834.64it/s] 84%|████████▍ | 336092/400000 [00:46<00:09, 6968.64it/s] 84%|████████▍ | 336868/400000 [00:46<00:08, 7187.56it/s] 84%|████████▍ | 337640/400000 [00:46<00:08, 7338.99it/s] 85%|████████▍ | 338407/400000 [00:46<00:08, 7434.20it/s] 85%|████████▍ | 339153/400000 [00:46<00:08, 7423.28it/s] 85%|████████▍ | 339948/400000 [00:46<00:07, 7568.65it/s] 85%|████████▌ | 340707/400000 [00:46<00:07, 7508.53it/s] 85%|████████▌ | 341460/400000 [00:46<00:07, 7444.05it/s] 86%|████████▌ | 342234/400000 [00:47<00:07, 7529.02it/s] 86%|████████▌ | 342997/400000 [00:47<00:07, 7558.41it/s] 86%|████████▌ | 343755/400000 [00:47<00:07, 7562.57it/s] 86%|████████▌ | 344512/400000 [00:47<00:07, 7413.30it/s] 86%|████████▋ | 345255/400000 [00:47<00:07, 7386.92it/s] 87%|████████▋ | 346058/400000 [00:47<00:07, 7566.85it/s] 87%|████████▋ | 346817/400000 [00:47<00:07, 7531.66it/s] 87%|████████▋ | 347629/400000 [00:47<00:06, 7698.71it/s] 87%|████████▋ | 348420/400000 [00:47<00:06, 7759.54it/s] 87%|████████▋ | 349198/400000 [00:47<00:06, 7472.13it/s] 87%|████████▋ | 349949/400000 [00:48<00:06, 7470.35it/s] 88%|████████▊ | 350733/400000 [00:48<00:06, 7577.00it/s] 88%|████████▊ | 351501/400000 [00:48<00:06, 7599.83it/s] 88%|████████▊ | 352263/400000 [00:48<00:06, 7352.91it/s] 88%|████████▊ | 353001/400000 [00:48<00:06, 7105.50it/s] 88%|████████▊ | 353716/400000 [00:48<00:06, 7113.47it/s] 89%|████████▊ | 354498/400000 [00:48<00:06, 7309.66it/s] 89%|████████▉ | 355267/400000 [00:48<00:06, 7419.39it/s] 89%|████████▉ | 356012/400000 [00:48<00:06, 7268.24it/s] 89%|████████▉ | 356742/400000 [00:48<00:06, 7085.64it/s] 89%|████████▉ | 357507/400000 [00:49<00:05, 7244.26it/s] 90%|████████▉ | 358235/400000 [00:49<00:05, 7200.22it/s] 90%|████████▉ | 359004/400000 [00:49<00:05, 7338.91it/s] 90%|████████▉ | 359740/400000 [00:49<00:05, 7107.11it/s] 90%|█████████ | 360533/400000 [00:49<00:05, 7333.57it/s] 90%|█████████ | 361338/400000 [00:49<00:05, 7534.01it/s] 91%|█████████ | 362096/400000 [00:49<00:05, 7327.61it/s] 91%|█████████ | 362885/400000 [00:49<00:04, 7487.50it/s] 91%|█████████ | 363638/400000 [00:49<00:05, 7196.59it/s] 91%|█████████ | 364363/400000 [00:50<00:05, 7043.90it/s] 91%|█████████▏| 365096/400000 [00:50<00:04, 7126.86it/s] 91%|█████████▏| 365866/400000 [00:50<00:04, 7287.01it/s] 92%|█████████▏| 366614/400000 [00:50<00:04, 7342.86it/s] 92%|█████████▏| 367351/400000 [00:50<00:04, 7129.73it/s] 92%|█████████▏| 368067/400000 [00:50<00:04, 6572.63it/s] 92%|█████████▏| 368746/400000 [00:50<00:04, 6634.87it/s] 92%|█████████▏| 369418/400000 [00:50<00:04, 6657.36it/s] 93%|█████████▎| 370192/400000 [00:50<00:04, 6947.43it/s] 93%|█████████▎| 370954/400000 [00:50<00:04, 7134.50it/s] 93%|█████████▎| 371724/400000 [00:51<00:03, 7295.16it/s] 93%|█████████▎| 372459/400000 [00:51<00:03, 7162.11it/s] 93%|█████████▎| 373180/400000 [00:51<00:03, 6926.07it/s] 93%|█████████▎| 373953/400000 [00:51<00:03, 7147.39it/s] 94%|█████████▎| 374689/400000 [00:51<00:03, 7208.59it/s] 94%|█████████▍| 375414/400000 [00:51<00:03, 7184.71it/s] 94%|█████████▍| 376213/400000 [00:51<00:03, 7407.55it/s] 94%|█████████▍| 377001/400000 [00:51<00:03, 7542.01it/s] 94%|█████████▍| 377785/400000 [00:51<00:02, 7628.07it/s] 95%|█████████▍| 378551/400000 [00:52<00:02, 7364.05it/s] 95%|█████████▍| 379292/400000 [00:52<00:02, 7080.49it/s] 95%|█████████▌| 380005/400000 [00:52<00:02, 6782.92it/s] 95%|█████████▌| 380690/400000 [00:52<00:02, 6746.06it/s] 95%|█████████▌| 381421/400000 [00:52<00:02, 6903.59it/s] 96%|█████████▌| 382116/400000 [00:52<00:02, 6854.19it/s] 96%|█████████▌| 382805/400000 [00:52<00:02, 6810.28it/s] 96%|█████████▌| 383489/400000 [00:52<00:02, 6445.82it/s] 96%|█████████▌| 384264/400000 [00:52<00:02, 6787.68it/s] 96%|█████████▌| 384991/400000 [00:52<00:02, 6923.35it/s] 96%|█████████▋| 385786/400000 [00:53<00:01, 7200.85it/s] 97%|█████████▋| 386577/400000 [00:53<00:01, 7397.91it/s] 97%|█████████▋| 387324/400000 [00:53<00:01, 7106.73it/s] 97%|█████████▋| 388042/400000 [00:53<00:01, 7032.49it/s] 97%|█████████▋| 388800/400000 [00:53<00:01, 7186.32it/s] 97%|█████████▋| 389604/400000 [00:53<00:01, 7420.75it/s] 98%|█████████▊| 390352/400000 [00:53<00:01, 7389.77it/s] 98%|█████████▊| 391095/400000 [00:53<00:01, 7044.81it/s] 98%|█████████▊| 391806/400000 [00:53<00:01, 7031.10it/s] 98%|█████████▊| 392565/400000 [00:54<00:01, 7181.98it/s] 98%|█████████▊| 393287/400000 [00:54<00:00, 6930.87it/s] 98%|█████████▊| 393985/400000 [00:54<00:00, 6819.52it/s] 99%|█████████▊| 394671/400000 [00:54<00:00, 6820.91it/s] 99%|█████████▉| 395435/400000 [00:54<00:00, 7046.18it/s] 99%|█████████▉| 396198/400000 [00:54<00:00, 7209.70it/s] 99%|█████████▉| 396969/400000 [00:54<00:00, 7351.46it/s] 99%|█████████▉| 397730/400000 [00:54<00:00, 7424.32it/s]100%|█████████▉| 398509/400000 [00:54<00:00, 7528.37it/s]100%|█████████▉| 399308/400000 [00:54<00:00, 7658.42it/s]100%|█████████▉| 399999/400000 [00:55<00:00, 7269.74it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f1173499c88> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011263853320065827 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011279869837107069 	 Accuracy: 50

  model saves at 50% accuracy 

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
2020-05-15 22:24:49.571901: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 22:24:49.575935: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-05-15 22:24:49.576089: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d0c7b3e3a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 22:24:49.576107: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f111ea74e10> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 8.0346 - accuracy: 0.4760
 2000/25000 [=>............................] - ETA: 10s - loss: 8.0270 - accuracy: 0.4765
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.9171 - accuracy: 0.4837 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8315 - accuracy: 0.4893
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7617 - accuracy: 0.4938
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7458 - accuracy: 0.4948
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7017 - accuracy: 0.4977
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6820 - accuracy: 0.4990
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6768 - accuracy: 0.4993
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6682 - accuracy: 0.4999
11000/25000 [============>.................] - ETA: 4s - loss: 7.6861 - accuracy: 0.4987
12000/25000 [=============>................] - ETA: 4s - loss: 7.6602 - accuracy: 0.5004
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6584 - accuracy: 0.5005
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6787 - accuracy: 0.4992
15000/25000 [=================>............] - ETA: 3s - loss: 7.6912 - accuracy: 0.4984
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6340 - accuracy: 0.5021
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6224 - accuracy: 0.5029
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6453 - accuracy: 0.5014
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6456 - accuracy: 0.5014
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6536 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6579 - accuracy: 0.5006
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6464 - accuracy: 0.5013
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6606 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6634 - accuracy: 0.5002
25000/25000 [==============================] - 9s 379us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f10dc3045c0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f10dff8a2b0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.1248 - crf_viterbi_accuracy: 0.6533 - val_loss: 1.1212 - val_crf_viterbi_accuracy: 0.6800

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

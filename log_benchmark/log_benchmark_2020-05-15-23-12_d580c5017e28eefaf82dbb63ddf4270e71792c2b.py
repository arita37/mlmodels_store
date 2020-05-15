
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f410728afd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 23:12:35.725743
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-15 23:12:35.729532
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-15 23:12:35.733025
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-15 23:12:35.736309
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f41132a2470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354084.4688
Epoch 2/10

1/1 [==============================] - 0s 110ms/step - loss: 322827.9062
Epoch 3/10

1/1 [==============================] - 0s 104ms/step - loss: 255494.2812
Epoch 4/10

1/1 [==============================] - 0s 110ms/step - loss: 189521.7812
Epoch 5/10

1/1 [==============================] - 0s 116ms/step - loss: 133959.4688
Epoch 6/10

1/1 [==============================] - 0s 110ms/step - loss: 89685.1875
Epoch 7/10

1/1 [==============================] - 0s 101ms/step - loss: 58827.8359
Epoch 8/10

1/1 [==============================] - 0s 105ms/step - loss: 39005.7266
Epoch 9/10

1/1 [==============================] - 0s 106ms/step - loss: 26535.2422
Epoch 10/10

1/1 [==============================] - 0s 102ms/step - loss: 18604.2422

  #### Inference Need return ypred, ytrue ######################### 
[[ 2.0402786e-01 -3.5608160e-01 -5.7063842e-01 -2.1485674e-01
  -5.7949769e-01  9.0114820e-01  3.2948646e-01 -4.5582891e-01
   4.5564234e-01  9.6789926e-02  6.1666292e-01  7.1464425e-01
  -6.7826355e-01 -7.5402522e-01  5.5699837e-01 -3.4677240e-01
   1.1024476e+00  7.0984948e-01  1.9193199e-01  3.7647927e-01
   9.8169369e-01 -1.0741230e+00 -1.5397945e-01 -5.7149667e-01
   5.3819829e-01 -3.3688322e-01  5.4597747e-01 -1.6578953e-01
  -1.9802852e-01 -3.0747434e-01  1.1905811e+00 -2.0749363e-01
  -6.0911530e-01  8.4913266e-01  5.9453762e-01 -1.2307017e+00
   3.9398104e-01 -8.4589326e-01  6.3612688e-01  2.1270861e-01
  -1.1312351e-02 -2.1320850e-01 -1.5886155e-01 -6.6003311e-01
  -6.8868256e-01  1.6877037e-01 -3.5604304e-01 -5.6615222e-01
  -2.1200180e-03 -4.5500287e-01  1.0294005e+00 -2.6657602e-01
   1.0606759e+00  3.2634699e-01  5.5273110e-01 -6.8523610e-01
   2.6159540e-01 -2.1717508e-01  3.3780947e-01 -5.1412261e-01
  -2.0138958e-01  4.2881336e+00  4.2600374e+00  4.3631258e+00
   4.0518546e+00  4.1357746e+00  3.9016433e+00  4.2714930e+00
   4.7929201e+00  4.1209903e+00  4.3858047e+00  4.2017784e+00
   5.7372026e+00  4.4754424e+00  3.9020293e+00  5.3140917e+00
   4.4968405e+00  4.0448017e+00  4.2649145e+00  4.3476071e+00
   4.2900300e+00  4.0537243e+00  2.8308330e+00  5.0070772e+00
   4.6342874e+00  5.2158909e+00  4.4113784e+00  4.6616240e+00
   4.0846009e+00  3.5220156e+00  4.0510035e+00  4.5404015e+00
   4.4921064e+00  4.3659005e+00  5.3552713e+00  4.8336544e+00
   4.4094625e+00  2.9964395e+00  3.6151357e+00  4.1704526e+00
   4.2992687e+00  3.3257127e+00  4.7974329e+00  4.0858927e+00
   3.5299592e+00  5.1110234e+00  4.7843270e+00  4.2606335e+00
   4.0015244e+00  3.5681481e+00  3.9849873e+00  4.3731890e+00
   3.9562297e+00  3.9997191e+00  3.9566543e+00  4.7180595e+00
   5.4217596e+00  3.5535102e+00  4.8802733e+00  4.2703209e+00
  -9.6240479e-01  1.2621096e+00 -1.5421271e-01  3.7012729e-01
  -1.0824579e+00 -4.4977418e-01 -2.1052244e-01 -1.7275119e-01
   1.4384571e-01 -4.4383588e-01 -5.7864189e-03 -4.5740247e-01
   1.1253319e+00 -5.4196429e-01  2.3382604e-02  4.9317122e-02
  -7.1576941e-01  7.5708401e-01  9.0559596e-01  5.6409323e-01
   5.0067067e-01  1.2291659e+00  3.8116416e-01  2.7135423e-01
   2.1005720e-01  6.2109685e-01  1.3546667e-01  5.3650081e-02
  -1.0059617e+00 -7.8533113e-02 -1.3367926e-01 -9.5139778e-01
  -1.0431552e+00 -5.4559380e-01  7.3933828e-01  1.6176701e-01
   1.7858124e-01 -9.7619045e-01 -3.3619830e-01 -1.0090640e+00
   6.2333047e-02  7.7597737e-02  4.5199472e-01 -4.4813016e-01
  -5.1547235e-01 -9.1304898e-01  2.3825756e-01  9.2630970e-01
   1.4635215e+00  3.0385858e-01  3.2232958e-01 -1.6635950e-01
  -2.1252191e-01 -1.3412265e-01 -7.3408115e-01 -5.6863338e-01
  -7.2225320e-01 -5.4264569e-01  9.8564482e-01 -1.3158140e-01
   9.9693769e-01  9.1049051e-01  4.8297822e-01  1.0669417e+00
   5.7001841e-01  2.3050237e+00  9.6362889e-01  1.4473388e+00
   1.1426635e+00  8.0257189e-01  5.7779729e-01  3.8809931e-01
   5.9302378e-01  7.2216719e-01  1.5479715e+00  5.1305598e-01
   8.3145410e-01  1.8168499e+00  1.3206098e+00  1.3057556e+00
   5.5645478e-01  3.6495173e-01  4.5394182e-01  1.5460479e+00
   1.0534453e+00  7.2272629e-01  1.0130131e+00  7.3839283e-01
   9.6732461e-01  1.4767778e+00  1.7094451e+00  7.3286712e-01
   1.7743403e+00  1.9412441e+00  6.8762022e-01  5.9811234e-01
   1.0961088e+00  1.8065107e+00  1.1015642e+00  4.3550956e-01
   1.1842463e+00  4.1237438e-01  1.2940034e+00  1.0438064e+00
   1.1427772e+00  1.7103517e+00  1.9491079e+00  4.9660146e-01
   1.2335637e+00  4.6445805e-01  1.7074254e+00  7.4182057e-01
   1.4041519e+00  3.2090354e-01  1.7581635e+00  4.9642098e-01
   1.3145628e+00  1.1106398e+00  2.2741103e-01  5.1424998e-01
   4.0033758e-02  5.2610111e+00  5.3519378e+00  5.0689750e+00
   5.0526547e+00  5.3138151e+00  5.0075603e+00  5.5172939e+00
   5.3960481e+00  5.2922325e+00  4.5662446e+00  5.1851425e+00
   4.8908601e+00  5.3519692e+00  5.4198389e+00  6.3445592e+00
   5.0694404e+00  5.2290840e+00  5.3042703e+00  4.1471567e+00
   5.2668614e+00  4.8534093e+00  4.9892497e+00  5.1947541e+00
   4.9031019e+00  6.2923760e+00  5.4885445e+00  5.6047707e+00
   5.8283067e+00  5.2910767e+00  5.2830529e+00  5.4204779e+00
   4.9146008e+00  5.5872078e+00  5.7264042e+00  4.7642593e+00
   4.8494296e+00  3.7657318e+00  5.2322249e+00  5.0966988e+00
   3.9060497e+00  6.1917253e+00  5.1084194e+00  5.4391980e+00
   4.4275618e+00  5.6561847e+00  4.1015253e+00  5.2711182e+00
   4.2061148e+00  4.6140656e+00  4.4820538e+00  5.1686883e+00
   6.0591440e+00  5.0632515e+00  5.7585888e+00  6.1258225e+00
   5.7099905e+00  4.8362002e+00  5.6064501e+00  5.2121100e+00
   1.0797095e+00  1.2558949e+00  1.1698037e+00  1.0286360e+00
   1.8673909e+00  1.3975637e+00  2.3142843e+00  6.4060807e-01
   1.0079709e+00  3.4444499e-01  1.9651133e+00  2.4418551e-01
   4.8256946e-01  7.9091895e-01  4.5264626e-01  1.6864330e+00
   1.4203305e+00  5.9985650e-01  5.8366728e-01  1.3108326e+00
   1.2887969e+00  8.1467080e-01  1.7191515e+00  1.0471312e+00
   1.3109529e+00  8.4390676e-01  7.2230494e-01  1.1383749e+00
   8.4965086e-01  1.0382138e+00  3.7716484e-01  6.3544762e-01
   9.2443353e-01  5.2899325e-01  5.3843743e-01  1.9039668e+00
   9.1471541e-01  5.7866693e-01  1.8642604e+00  9.6647090e-01
   5.6483245e-01  1.8649899e+00  5.6954962e-01  4.9259031e-01
   8.7415922e-01  1.9010054e+00  6.7945790e-01  1.2583767e+00
   1.7748438e+00  6.5828103e-01  1.6493511e+00  3.9871728e-01
   9.9795049e-01  1.5657512e+00  1.5631830e+00  2.2048068e+00
   1.6790788e+00  6.9061989e-01  1.0305307e+00  1.5154055e+00
  -6.7180386e+00 -6.7164624e-01 -2.5214603e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 23:12:44.963183
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   101.399
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-15 23:12:44.967757
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   10293.5
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-15 23:12:44.971372
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   101.489
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-15 23:12:44.974787
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -920.858
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139916929897024
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139915702616640
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139915702617144
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139915702617648
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139915702618152
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139915702618656

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f40f2ec4fd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.487757
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.452888
grad_step = 000002, loss = 0.426264
grad_step = 000003, loss = 0.400913
grad_step = 000004, loss = 0.376117
grad_step = 000005, loss = 0.356108
grad_step = 000006, loss = 0.350600
grad_step = 000007, loss = 0.345053
grad_step = 000008, loss = 0.330519
grad_step = 000009, loss = 0.315120
grad_step = 000010, loss = 0.303832
grad_step = 000011, loss = 0.296070
grad_step = 000012, loss = 0.288920
grad_step = 000013, loss = 0.281120
grad_step = 000014, loss = 0.272386
grad_step = 000015, loss = 0.262988
grad_step = 000016, loss = 0.254641
grad_step = 000017, loss = 0.247380
grad_step = 000018, loss = 0.239431
grad_step = 000019, loss = 0.231067
grad_step = 000020, loss = 0.223714
grad_step = 000021, loss = 0.217333
grad_step = 000022, loss = 0.210712
grad_step = 000023, loss = 0.203570
grad_step = 000024, loss = 0.196427
grad_step = 000025, loss = 0.189503
grad_step = 000026, loss = 0.182786
grad_step = 000027, loss = 0.176342
grad_step = 000028, loss = 0.170269
grad_step = 000029, loss = 0.164327
grad_step = 000030, loss = 0.158247
grad_step = 000031, loss = 0.152314
grad_step = 000032, loss = 0.146678
grad_step = 000033, loss = 0.141108
grad_step = 000034, loss = 0.135556
grad_step = 000035, loss = 0.130274
grad_step = 000036, loss = 0.125279
grad_step = 000037, loss = 0.120292
grad_step = 000038, loss = 0.115370
grad_step = 000039, loss = 0.110652
grad_step = 000040, loss = 0.106074
grad_step = 000041, loss = 0.101637
grad_step = 000042, loss = 0.097408
grad_step = 000043, loss = 0.093296
grad_step = 000044, loss = 0.089222
grad_step = 000045, loss = 0.085286
grad_step = 000046, loss = 0.081512
grad_step = 000047, loss = 0.077849
grad_step = 000048, loss = 0.074325
grad_step = 000049, loss = 0.070985
grad_step = 000050, loss = 0.067775
grad_step = 000051, loss = 0.064623
grad_step = 000052, loss = 0.061572
grad_step = 000053, loss = 0.058658
grad_step = 000054, loss = 0.055844
grad_step = 000055, loss = 0.053154
grad_step = 000056, loss = 0.050570
grad_step = 000057, loss = 0.048054
grad_step = 000058, loss = 0.045664
grad_step = 000059, loss = 0.043372
grad_step = 000060, loss = 0.041153
grad_step = 000061, loss = 0.039042
grad_step = 000062, loss = 0.037005
grad_step = 000063, loss = 0.035042
grad_step = 000064, loss = 0.033173
grad_step = 000065, loss = 0.031393
grad_step = 000066, loss = 0.029693
grad_step = 000067, loss = 0.028064
grad_step = 000068, loss = 0.026510
grad_step = 000069, loss = 0.025030
grad_step = 000070, loss = 0.023629
grad_step = 000071, loss = 0.022290
grad_step = 000072, loss = 0.021025
grad_step = 000073, loss = 0.019822
grad_step = 000074, loss = 0.018682
grad_step = 000075, loss = 0.017605
grad_step = 000076, loss = 0.016587
grad_step = 000077, loss = 0.015624
grad_step = 000078, loss = 0.014715
grad_step = 000079, loss = 0.013859
grad_step = 000080, loss = 0.013056
grad_step = 000081, loss = 0.012299
grad_step = 000082, loss = 0.011590
grad_step = 000083, loss = 0.010922
grad_step = 000084, loss = 0.010296
grad_step = 000085, loss = 0.009711
grad_step = 000086, loss = 0.009164
grad_step = 000087, loss = 0.008652
grad_step = 000088, loss = 0.008175
grad_step = 000089, loss = 0.007730
grad_step = 000090, loss = 0.007314
grad_step = 000091, loss = 0.006927
grad_step = 000092, loss = 0.006566
grad_step = 000093, loss = 0.006230
grad_step = 000094, loss = 0.005918
grad_step = 000095, loss = 0.005627
grad_step = 000096, loss = 0.005356
grad_step = 000097, loss = 0.005104
grad_step = 000098, loss = 0.004870
grad_step = 000099, loss = 0.004653
grad_step = 000100, loss = 0.004450
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.004262
grad_step = 000102, loss = 0.004088
grad_step = 000103, loss = 0.003926
grad_step = 000104, loss = 0.003776
grad_step = 000105, loss = 0.003637
grad_step = 000106, loss = 0.003508
grad_step = 000107, loss = 0.003389
grad_step = 000108, loss = 0.003279
grad_step = 000109, loss = 0.003177
grad_step = 000110, loss = 0.003084
grad_step = 000111, loss = 0.002997
grad_step = 000112, loss = 0.002918
grad_step = 000113, loss = 0.002844
grad_step = 000114, loss = 0.002777
grad_step = 000115, loss = 0.002716
grad_step = 000116, loss = 0.002660
grad_step = 000117, loss = 0.002615
grad_step = 000118, loss = 0.002578
grad_step = 000119, loss = 0.002533
grad_step = 000120, loss = 0.002487
grad_step = 000121, loss = 0.002452
grad_step = 000122, loss = 0.002430
grad_step = 000123, loss = 0.002397
grad_step = 000124, loss = 0.002365
grad_step = 000125, loss = 0.002348
grad_step = 000126, loss = 0.002329
grad_step = 000127, loss = 0.002308
grad_step = 000128, loss = 0.002287
grad_step = 000129, loss = 0.002273
grad_step = 000130, loss = 0.002263
grad_step = 000131, loss = 0.002248
grad_step = 000132, loss = 0.002234
grad_step = 000133, loss = 0.002223
grad_step = 000134, loss = 0.002215
grad_step = 000135, loss = 0.002208
grad_step = 000136, loss = 0.002198
grad_step = 000137, loss = 0.002188
grad_step = 000138, loss = 0.002180
grad_step = 000139, loss = 0.002172
grad_step = 000140, loss = 0.002167
grad_step = 000141, loss = 0.002163
grad_step = 000142, loss = 0.002159
grad_step = 000143, loss = 0.002158
grad_step = 000144, loss = 0.002160
grad_step = 000145, loss = 0.002161
grad_step = 000146, loss = 0.002156
grad_step = 000147, loss = 0.002142
grad_step = 000148, loss = 0.002126
grad_step = 000149, loss = 0.002117
grad_step = 000150, loss = 0.002115
grad_step = 000151, loss = 0.002118
grad_step = 000152, loss = 0.002123
grad_step = 000153, loss = 0.002125
grad_step = 000154, loss = 0.002123
grad_step = 000155, loss = 0.002114
grad_step = 000156, loss = 0.002100
grad_step = 000157, loss = 0.002087
grad_step = 000158, loss = 0.002078
grad_step = 000159, loss = 0.002074
grad_step = 000160, loss = 0.002074
grad_step = 000161, loss = 0.002077
grad_step = 000162, loss = 0.002085
grad_step = 000163, loss = 0.002101
grad_step = 000164, loss = 0.002126
grad_step = 000165, loss = 0.002149
grad_step = 000166, loss = 0.002143
grad_step = 000167, loss = 0.002105
grad_step = 000168, loss = 0.002057
grad_step = 000169, loss = 0.002037
grad_step = 000170, loss = 0.002052
grad_step = 000171, loss = 0.002080
grad_step = 000172, loss = 0.002090
grad_step = 000173, loss = 0.002074
grad_step = 000174, loss = 0.002042
grad_step = 000175, loss = 0.002019
grad_step = 000176, loss = 0.002019
grad_step = 000177, loss = 0.002035
grad_step = 000178, loss = 0.002048
grad_step = 000179, loss = 0.002048
grad_step = 000180, loss = 0.002034
grad_step = 000181, loss = 0.002014
grad_step = 000182, loss = 0.001999
grad_step = 000183, loss = 0.001995
grad_step = 000184, loss = 0.001999
grad_step = 000185, loss = 0.002006
grad_step = 000186, loss = 0.002014
grad_step = 000187, loss = 0.002021
grad_step = 000188, loss = 0.002024
grad_step = 000189, loss = 0.002021
grad_step = 000190, loss = 0.002013
grad_step = 000191, loss = 0.002003
grad_step = 000192, loss = 0.001990
grad_step = 000193, loss = 0.001978
grad_step = 000194, loss = 0.001969
grad_step = 000195, loss = 0.001963
grad_step = 000196, loss = 0.001959
grad_step = 000197, loss = 0.001956
grad_step = 000198, loss = 0.001955
grad_step = 000199, loss = 0.001957
grad_step = 000200, loss = 0.001961
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001973
grad_step = 000202, loss = 0.002002
grad_step = 000203, loss = 0.002070
grad_step = 000204, loss = 0.002191
grad_step = 000205, loss = 0.002341
grad_step = 000206, loss = 0.002327
grad_step = 000207, loss = 0.002110
grad_step = 000208, loss = 0.001938
grad_step = 000209, loss = 0.002013
grad_step = 000210, loss = 0.002139
grad_step = 000211, loss = 0.002073
grad_step = 000212, loss = 0.001942
grad_step = 000213, loss = 0.001978
grad_step = 000214, loss = 0.002063
grad_step = 000215, loss = 0.002002
grad_step = 000216, loss = 0.001921
grad_step = 000217, loss = 0.001970
grad_step = 000218, loss = 0.002014
grad_step = 000219, loss = 0.001944
grad_step = 000220, loss = 0.001912
grad_step = 000221, loss = 0.001959
grad_step = 000222, loss = 0.001962
grad_step = 000223, loss = 0.001920
grad_step = 000224, loss = 0.001909
grad_step = 000225, loss = 0.001929
grad_step = 000226, loss = 0.001933
grad_step = 000227, loss = 0.001910
grad_step = 000228, loss = 0.001893
grad_step = 000229, loss = 0.001904
grad_step = 000230, loss = 0.001914
grad_step = 000231, loss = 0.001900
grad_step = 000232, loss = 0.001881
grad_step = 000233, loss = 0.001885
grad_step = 000234, loss = 0.001895
grad_step = 000235, loss = 0.001889
grad_step = 000236, loss = 0.001875
grad_step = 000237, loss = 0.001870
grad_step = 000238, loss = 0.001873
grad_step = 000239, loss = 0.001875
grad_step = 000240, loss = 0.001874
grad_step = 000241, loss = 0.001869
grad_step = 000242, loss = 0.001860
grad_step = 000243, loss = 0.001853
grad_step = 000244, loss = 0.001854
grad_step = 000245, loss = 0.001857
grad_step = 000246, loss = 0.001855
grad_step = 000247, loss = 0.001848
grad_step = 000248, loss = 0.001842
grad_step = 000249, loss = 0.001838
grad_step = 000250, loss = 0.001834
grad_step = 000251, loss = 0.001831
grad_step = 000252, loss = 0.001830
grad_step = 000253, loss = 0.001831
grad_step = 000254, loss = 0.001833
grad_step = 000255, loss = 0.001837
grad_step = 000256, loss = 0.001845
grad_step = 000257, loss = 0.001866
grad_step = 000258, loss = 0.001877
grad_step = 000259, loss = 0.001890
grad_step = 000260, loss = 0.001868
grad_step = 000261, loss = 0.001859
grad_step = 000262, loss = 0.001855
grad_step = 000263, loss = 0.001841
grad_step = 000264, loss = 0.001826
grad_step = 000265, loss = 0.001811
grad_step = 000266, loss = 0.001803
grad_step = 000267, loss = 0.001801
grad_step = 000268, loss = 0.001808
grad_step = 000269, loss = 0.001820
grad_step = 000270, loss = 0.001822
grad_step = 000271, loss = 0.001814
grad_step = 000272, loss = 0.001800
grad_step = 000273, loss = 0.001792
grad_step = 000274, loss = 0.001791
grad_step = 000275, loss = 0.001800
grad_step = 000276, loss = 0.001833
grad_step = 000277, loss = 0.001867
grad_step = 000278, loss = 0.001874
grad_step = 000279, loss = 0.001797
grad_step = 000280, loss = 0.001770
grad_step = 000281, loss = 0.001800
grad_step = 000282, loss = 0.001799
grad_step = 000283, loss = 0.001779
grad_step = 000284, loss = 0.001777
grad_step = 000285, loss = 0.001774
grad_step = 000286, loss = 0.001768
grad_step = 000287, loss = 0.001772
grad_step = 000288, loss = 0.001767
grad_step = 000289, loss = 0.001752
grad_step = 000290, loss = 0.001755
grad_step = 000291, loss = 0.001765
grad_step = 000292, loss = 0.001755
grad_step = 000293, loss = 0.001744
grad_step = 000294, loss = 0.001750
grad_step = 000295, loss = 0.001758
grad_step = 000296, loss = 0.001757
grad_step = 000297, loss = 0.001762
grad_step = 000298, loss = 0.001784
grad_step = 000299, loss = 0.001808
grad_step = 000300, loss = 0.001840
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001889
grad_step = 000302, loss = 0.001937
grad_step = 000303, loss = 0.001930
grad_step = 000304, loss = 0.001866
grad_step = 000305, loss = 0.001780
grad_step = 000306, loss = 0.001729
grad_step = 000307, loss = 0.001738
grad_step = 000308, loss = 0.001782
grad_step = 000309, loss = 0.001798
grad_step = 000310, loss = 0.001772
grad_step = 000311, loss = 0.001735
grad_step = 000312, loss = 0.001724
grad_step = 000313, loss = 0.001733
grad_step = 000314, loss = 0.001736
grad_step = 000315, loss = 0.001732
grad_step = 000316, loss = 0.001728
grad_step = 000317, loss = 0.001722
grad_step = 000318, loss = 0.001718
grad_step = 000319, loss = 0.001704
grad_step = 000320, loss = 0.001696
grad_step = 000321, loss = 0.001704
grad_step = 000322, loss = 0.001714
grad_step = 000323, loss = 0.001711
grad_step = 000324, loss = 0.001695
grad_step = 000325, loss = 0.001686
grad_step = 000326, loss = 0.001687
grad_step = 000327, loss = 0.001687
grad_step = 000328, loss = 0.001683
grad_step = 000329, loss = 0.001680
grad_step = 000330, loss = 0.001681
grad_step = 000331, loss = 0.001683
grad_step = 000332, loss = 0.001682
grad_step = 000333, loss = 0.001677
grad_step = 000334, loss = 0.001670
grad_step = 000335, loss = 0.001667
grad_step = 000336, loss = 0.001669
grad_step = 000337, loss = 0.001673
grad_step = 000338, loss = 0.001681
grad_step = 000339, loss = 0.001689
grad_step = 000340, loss = 0.001697
grad_step = 000341, loss = 0.001704
grad_step = 000342, loss = 0.001724
grad_step = 000343, loss = 0.001747
grad_step = 000344, loss = 0.001769
grad_step = 000345, loss = 0.001775
grad_step = 000346, loss = 0.001761
grad_step = 000347, loss = 0.001727
grad_step = 000348, loss = 0.001687
grad_step = 000349, loss = 0.001651
grad_step = 000350, loss = 0.001638
grad_step = 000351, loss = 0.001647
grad_step = 000352, loss = 0.001665
grad_step = 000353, loss = 0.001682
grad_step = 000354, loss = 0.001680
grad_step = 000355, loss = 0.001666
grad_step = 000356, loss = 0.001643
grad_step = 000357, loss = 0.001627
grad_step = 000358, loss = 0.001622
grad_step = 000359, loss = 0.001625
grad_step = 000360, loss = 0.001633
grad_step = 000361, loss = 0.001640
grad_step = 000362, loss = 0.001645
grad_step = 000363, loss = 0.001645
grad_step = 000364, loss = 0.001642
grad_step = 000365, loss = 0.001634
grad_step = 000366, loss = 0.001625
grad_step = 000367, loss = 0.001614
grad_step = 000368, loss = 0.001606
grad_step = 000369, loss = 0.001600
grad_step = 000370, loss = 0.001597
grad_step = 000371, loss = 0.001595
grad_step = 000372, loss = 0.001595
grad_step = 000373, loss = 0.001595
grad_step = 000374, loss = 0.001594
grad_step = 000375, loss = 0.001592
grad_step = 000376, loss = 0.001591
grad_step = 000377, loss = 0.001590
grad_step = 000378, loss = 0.001590
grad_step = 000379, loss = 0.001594
grad_step = 000380, loss = 0.001608
grad_step = 000381, loss = 0.001644
grad_step = 000382, loss = 0.001696
grad_step = 000383, loss = 0.001775
grad_step = 000384, loss = 0.001768
grad_step = 000385, loss = 0.001770
grad_step = 000386, loss = 0.001782
grad_step = 000387, loss = 0.001707
grad_step = 000388, loss = 0.001604
grad_step = 000389, loss = 0.001581
grad_step = 000390, loss = 0.001607
grad_step = 000391, loss = 0.001636
grad_step = 000392, loss = 0.001664
grad_step = 000393, loss = 0.001619
grad_step = 000394, loss = 0.001569
grad_step = 000395, loss = 0.001558
grad_step = 000396, loss = 0.001568
grad_step = 000397, loss = 0.001596
grad_step = 000398, loss = 0.001619
grad_step = 000399, loss = 0.001609
grad_step = 000400, loss = 0.001590
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001569
grad_step = 000402, loss = 0.001542
grad_step = 000403, loss = 0.001534
grad_step = 000404, loss = 0.001538
grad_step = 000405, loss = 0.001542
grad_step = 000406, loss = 0.001552
grad_step = 000407, loss = 0.001562
grad_step = 000408, loss = 0.001560
grad_step = 000409, loss = 0.001554
grad_step = 000410, loss = 0.001550
grad_step = 000411, loss = 0.001537
grad_step = 000412, loss = 0.001526
grad_step = 000413, loss = 0.001518
grad_step = 000414, loss = 0.001511
grad_step = 000415, loss = 0.001505
grad_step = 000416, loss = 0.001503
grad_step = 000417, loss = 0.001502
grad_step = 000418, loss = 0.001501
grad_step = 000419, loss = 0.001502
grad_step = 000420, loss = 0.001504
grad_step = 000421, loss = 0.001508
grad_step = 000422, loss = 0.001514
grad_step = 000423, loss = 0.001527
grad_step = 000424, loss = 0.001547
grad_step = 000425, loss = 0.001580
grad_step = 000426, loss = 0.001620
grad_step = 000427, loss = 0.001673
grad_step = 000428, loss = 0.001691
grad_step = 000429, loss = 0.001675
grad_step = 000430, loss = 0.001602
grad_step = 000431, loss = 0.001515
grad_step = 000432, loss = 0.001471
grad_step = 000433, loss = 0.001491
grad_step = 000434, loss = 0.001536
grad_step = 000435, loss = 0.001555
grad_step = 000436, loss = 0.001530
grad_step = 000437, loss = 0.001484
grad_step = 000438, loss = 0.001457
grad_step = 000439, loss = 0.001464
grad_step = 000440, loss = 0.001488
grad_step = 000441, loss = 0.001508
grad_step = 000442, loss = 0.001505
grad_step = 000443, loss = 0.001487
grad_step = 000444, loss = 0.001461
grad_step = 000445, loss = 0.001443
grad_step = 000446, loss = 0.001438
grad_step = 000447, loss = 0.001443
grad_step = 000448, loss = 0.001454
grad_step = 000449, loss = 0.001462
grad_step = 000450, loss = 0.001464
grad_step = 000451, loss = 0.001457
grad_step = 000452, loss = 0.001446
grad_step = 000453, loss = 0.001432
grad_step = 000454, loss = 0.001422
grad_step = 000455, loss = 0.001417
grad_step = 000456, loss = 0.001417
grad_step = 000457, loss = 0.001420
grad_step = 000458, loss = 0.001424
grad_step = 000459, loss = 0.001427
grad_step = 000460, loss = 0.001427
grad_step = 000461, loss = 0.001426
grad_step = 000462, loss = 0.001421
grad_step = 000463, loss = 0.001417
grad_step = 000464, loss = 0.001411
grad_step = 000465, loss = 0.001405
grad_step = 000466, loss = 0.001399
grad_step = 000467, loss = 0.001395
grad_step = 000468, loss = 0.001390
grad_step = 000469, loss = 0.001387
grad_step = 000470, loss = 0.001384
grad_step = 000471, loss = 0.001382
grad_step = 000472, loss = 0.001379
grad_step = 000473, loss = 0.001377
grad_step = 000474, loss = 0.001376
grad_step = 000475, loss = 0.001374
grad_step = 000476, loss = 0.001373
grad_step = 000477, loss = 0.001373
grad_step = 000478, loss = 0.001374
grad_step = 000479, loss = 0.001378
grad_step = 000480, loss = 0.001388
grad_step = 000481, loss = 0.001405
grad_step = 000482, loss = 0.001440
grad_step = 000483, loss = 0.001482
grad_step = 000484, loss = 0.001551
grad_step = 000485, loss = 0.001573
grad_step = 000486, loss = 0.001571
grad_step = 000487, loss = 0.001498
grad_step = 000488, loss = 0.001414
grad_step = 000489, loss = 0.001370
grad_step = 000490, loss = 0.001367
grad_step = 000491, loss = 0.001398
grad_step = 000492, loss = 0.001431
grad_step = 000493, loss = 0.001412
grad_step = 000494, loss = 0.001363
grad_step = 000495, loss = 0.001332
grad_step = 000496, loss = 0.001351
grad_step = 000497, loss = 0.001378
grad_step = 000498, loss = 0.001372
grad_step = 000499, loss = 0.001354
grad_step = 000500, loss = 0.001351
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001349
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

  date_run                              2020-05-15 23:13:09.850940
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.248418
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-15 23:13:09.857078
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.159217
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-15 23:13:09.865068
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.14074
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-15 23:13:09.870888
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.41936
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
0   2020-05-15 23:12:35.725743  ...    mean_absolute_error
1   2020-05-15 23:12:35.729532  ...     mean_squared_error
2   2020-05-15 23:12:35.733025  ...  median_absolute_error
3   2020-05-15 23:12:35.736309  ...               r2_score
4   2020-05-15 23:12:44.963183  ...    mean_absolute_error
5   2020-05-15 23:12:44.967757  ...     mean_squared_error
6   2020-05-15 23:12:44.971372  ...  median_absolute_error
7   2020-05-15 23:12:44.974787  ...               r2_score
8   2020-05-15 23:13:09.850940  ...    mean_absolute_error
9   2020-05-15 23:13:09.857078  ...     mean_squared_error
10  2020-05-15 23:13:09.865068  ...  median_absolute_error
11  2020-05-15 23:13:09.870888  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd91fa19898> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 40%|████      | 3981312/9912422 [00:00<00:00, 39803573.15it/s]9920512it [00:00, 36541485.32it/s]                             
0it [00:00, ?it/s]32768it [00:00, 582924.20it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 457376.70it/s]1654784it [00:00, 11670419.93it/s]                         
0it [00:00, ?it/s]8192it [00:00, 192876.20it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd8d23c9e10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd8cf2140b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd8d23c9e10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd8d1950080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd8cf18a470> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd8cf1766d8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd8d23c9e10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd8d190e6a0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd8cf18a470> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd91f9d1e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f60fd9f7208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=2a38add034e1288a4c8173a0081c63cac902fc2b536ed0cc3964d5a633ba1ff3
  Stored in directory: /tmp/pip-ephem-wheel-cache-5bvefnj6/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f60966cff28> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1548288/17464789 [=>............................] - ETA: 0s
 4964352/17464789 [=======>......................] - ETA: 0s
 8355840/17464789 [=============>................] - ETA: 0s
10608640/17464789 [=================>............] - ETA: 0s
12500992/17464789 [====================>.........] - ETA: 0s
14450688/17464789 [=======================>......] - ETA: 0s
16367616/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-15 23:14:37.684831: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 23:14:37.688826: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 23:14:37.688954: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cd6de750a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 23:14:37.688969: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.9426 - accuracy: 0.4820
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8583 - accuracy: 0.4875 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8097 - accuracy: 0.4907
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7778 - accuracy: 0.4927
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7525 - accuracy: 0.4944
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6798 - accuracy: 0.4991
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6954 - accuracy: 0.4981
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7007 - accuracy: 0.4978
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7126 - accuracy: 0.4970
11000/25000 [============>.................] - ETA: 4s - loss: 7.6959 - accuracy: 0.4981
12000/25000 [=============>................] - ETA: 4s - loss: 7.6935 - accuracy: 0.4983
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6831 - accuracy: 0.4989
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6699 - accuracy: 0.4998
15000/25000 [=================>............] - ETA: 3s - loss: 7.6329 - accuracy: 0.5022
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6504 - accuracy: 0.5011
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6504 - accuracy: 0.5011
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6674 - accuracy: 0.4999
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6712 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6849 - accuracy: 0.4988
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6903 - accuracy: 0.4985
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6826 - accuracy: 0.4990
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6871 - accuracy: 0.4987
25000/25000 [==============================] - 10s 394us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-15 23:14:55.066236
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-15 23:14:55.066236  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<46:08:44, 5.19kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<32:32:04, 7.36kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<22:49:31, 10.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<15:58:45, 15.0kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<11:09:11, 21.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.53M/862M [00:02<7:45:19, 30.5kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 14.6M/862M [00:02<5:23:51, 43.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.4M/862M [00:02<3:45:47, 62.3kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.4M/862M [00:02<2:37:26, 88.9kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.8M/862M [00:02<1:49:42, 127kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 31.0M/862M [00:02<1:16:30, 181kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.3M/862M [00:02<53:22, 258kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 39.3M/862M [00:02<37:17, 368kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.6M/862M [00:03<26:03, 524kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.5M/862M [00:03<18:15, 744kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:03<12:59, 1.04MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:05<10:58, 1.23MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:05<11:22, 1.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<08:52, 1.51MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.8M/862M [00:05<06:23, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:07<10:04, 1.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:07<08:38, 1.55MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:07<06:23, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:09<07:13, 1.84MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:09<07:54, 1.68MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<06:07, 2.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.2M/862M [00:09<04:27, 2.97MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:11<09:21, 1.41MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:11<07:55, 1.67MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:11<05:52, 2.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:13<07:09, 1.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:13<07:41, 1.71MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:13<06:02, 2.18MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:15<06:20, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:15<05:45, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.4M/862M [00:15<04:21, 2.99MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:17<06:07, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:17<05:37, 2.32MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:17<04:15, 3.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:19<06:00, 2.16MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:19<05:32, 2.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.6M/862M [00:19<04:09, 3.11MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:21<05:56, 2.17MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:21<05:28, 2.35MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:21<04:08, 3.10MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:23<05:51, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:23<06:42, 1.91MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:23<05:19, 2.40MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:23<03:52, 3.30MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:25<13:20, 955kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:25<10:40, 1.19MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:25<07:47, 1.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<08:23, 1.51MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<07:09, 1.77MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<05:16, 2.40MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:29<06:41, 1.88MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<05:57, 2.11MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<04:26, 2.83MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:31<06:05, 2.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<05:32, 2.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<04:11, 2.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:53, 2.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<05:22, 2.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<04:04, 3.05MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:46, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<05:21, 2.32MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<04:03, 3.05MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:44, 2.15MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<05:16, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<03:59, 3.08MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:41, 2.15MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:28, 1.89MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:05, 2.41MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<03:43, 3.29MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<09:04, 1.34MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<07:37, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<05:43, 2.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<06:33, 1.85MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<08:33, 1.42MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<07:00, 1.73MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<05:07, 2.36MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<07:01, 1.72MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<08:50, 1.36MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<07:12, 1.67MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<05:17, 2.27MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:46<07:07, 1.68MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:53, 1.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<07:10, 1.67MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<05:14, 2.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:19, 1.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:48, 1.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:53, 1.73MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<05:13, 2.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<03:49, 3.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<1:36:32, 123kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<1:10:58, 167kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<50:21, 235kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<35:37, 332kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<27:03, 436kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<22:21, 527kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<16:28, 715kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<11:43, 1.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<12:09, 964kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<11:41, 1.00MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:59, 1.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<06:29, 1.80MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<08:35, 1.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<09:09, 1.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:05, 1.64MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<05:10, 2.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<06:46, 1.71MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<08:06, 1.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:21, 1.82MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:44, 2.44MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:46, 1.99MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:59, 1.64MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:33, 2.07MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:04, 2.82MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<06:25, 1.78MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:27, 1.53MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<05:51, 1.95MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<04:15, 2.67MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<06:52, 1.65MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<07:35, 1.50MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<05:54, 1.92MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<04:22, 2.59MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<05:51, 1.93MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<07:10, 1.57MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<05:41, 1.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:10, 2.69MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<05:54, 1.90MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<07:00, 1.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<05:31, 2.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:00, 2.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<07:23, 1.51MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<08:04, 1.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:23, 1.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<04:38, 2.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<07:15, 1.53MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<07:46, 1.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<06:03, 1.83MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<04:25, 2.49MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<07:41, 1.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<08:05, 1.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:20, 1.74MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<04:36, 2.38MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<08:39, 1.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<08:44, 1.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<06:41, 1.63MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<04:51, 2.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<06:53, 1.58MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<07:29, 1.45MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<05:49, 1.87MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<04:13, 2.57MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<07:01, 1.54MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<07:34, 1.43MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:51, 1.84MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<04:16, 2.52MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<06:27, 1.66MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<07:09, 1.50MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<05:33, 1.93MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<04:03, 2.64MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<06:10, 1.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<06:56, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<05:30, 1.93MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<04:02, 2.63MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<07:11, 1.48MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<08:15, 1.28MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:26<06:28, 1.64MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<04:41, 2.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<06:15, 1.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<06:56, 1.52MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:24, 1.95MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<03:58, 2.64MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<05:48, 1.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<09:09, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<07:30, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<05:32, 1.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<04:04, 2.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<07:03, 1.47MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<07:29, 1.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<05:46, 1.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:12, 2.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<06:11, 1.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<07:07, 1.45MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<05:35, 1.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:06, 2.51MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<06:31, 1.57MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<07:20, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<05:42, 1.79MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<04:15, 2.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<05:10, 1.97MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<06:23, 1.59MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<05:09, 1.97MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<03:44, 2.72MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<06:42, 1.51MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<07:02, 1.44MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<05:28, 1.85MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<04:00, 2.51MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<05:34, 1.80MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<06:47, 1.48MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:28, 1.83MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:59, 2.51MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<06:09, 1.62MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<06:44, 1.48MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<05:19, 1.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<03:52, 2.56MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<08:01, 1.23MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<07:44, 1.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<06:01, 1.64MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:21, 2.26MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:48<06:11, 1.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<06:40, 1.48MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<05:15, 1.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<03:48, 2.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<07:29, 1.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<07:32, 1.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<05:51, 1.67MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:14, 2.29MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<07:45, 1.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<07:50, 1.24MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<06:05, 1.59MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:23, 2.20MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<06:41, 1.44MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<06:59, 1.38MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<05:23, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<03:54, 2.45MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<06:00, 1.59MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:56<06:25, 1.49MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<05:02, 1.89MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<03:40, 2.59MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<07:05, 1.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<07:10, 1.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<05:29, 1.73MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<03:57, 2.39MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<06:38, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<07:13, 1.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<05:40, 1.66MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:08, 2.27MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<06:31, 1.43MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<06:59, 1.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<05:27, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:57, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<07:04, 1.31MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<07:06, 1.30MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<05:29, 1.69MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:58, 2.32MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<07:49, 1.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<07:58, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<06:11, 1.49MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:27, 2.06MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<06:52, 1.33MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:08<06:46, 1.35MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<05:09, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:44, 2.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:10<05:56, 1.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<06:05, 1.49MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<04:41, 1.94MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:24, 2.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:12<08:43, 1.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<07:57, 1.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<06:01, 1.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<04:20, 2.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<09:47, 913kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<09:09, 977kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:14<06:54, 1.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:56, 1.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<07:03, 1.26MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<06:54, 1.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<05:15, 1.69MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:46, 2.34MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:18<06:52, 1.28MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:18<07:04, 1.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<05:24, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:55, 2.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<05:26, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:20<05:45, 1.52MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<04:26, 1.96MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:15, 2.67MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:22<05:00, 1.73MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<05:08, 1.68MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<04:00, 2.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:53, 2.97MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<19:46, 435kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<16:02, 536kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<11:44, 731kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<08:17, 1.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<11:07, 766kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<09:40, 881kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<07:11, 1.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<05:06, 1.66MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<08:33, 989kB/s] .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<07:56, 1.06MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:28<05:58, 1.41MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<04:16, 1.97MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<06:55, 1.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<06:13, 1.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<04:42, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<04:43, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:32<05:08, 1.62MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<04:03, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<02:56, 2.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:34<12:03, 684kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:34<10:11, 809kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<07:29, 1.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<05:20, 1.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<07:53, 1.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<06:46, 1.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<05:03, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<04:57, 1.64MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<06:45, 1.20MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<05:33, 1.46MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:04, 1.98MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:40<04:40, 1.72MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:40<04:55, 1.63MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<03:50, 2.09MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<02:45, 2.88MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<2:35:16, 51.3kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:42<1:50:17, 72.2kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<1:17:25, 103kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<53:57, 147kB/s]  .vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<53:16, 148kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:44<38:20, 206kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<27:03, 291kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<20:18, 386kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:46<15:37, 501kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<11:13, 696kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<07:53, 984kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<19:28, 399kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<16:32, 469kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:48<12:16, 632kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<08:44, 884kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<08:00, 960kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<07:02, 1.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<05:16, 1.46MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:57, 1.54MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:54, 1.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<03:47, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:53, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:08, 1.82MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<03:14, 2.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<03:30, 2.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<05:15, 1.42MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:56<04:22, 1.71MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:13, 2.31MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<04:05, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<04:12, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<03:16, 2.26MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<03:31, 2.09MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<03:50, 1.91MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:01, 2.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<03:19, 2.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<03:41, 1.97MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<02:54, 2.49MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<02:06, 3.43MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<16:00, 450kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<12:33, 574kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<09:06, 790kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<07:31, 950kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<06:36, 1.08MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<04:54, 1.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<03:30, 2.02MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<06:24, 1.10MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<05:46, 1.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:21, 1.62MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:07<03:05, 2.26MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:09<1:10:32, 99.4kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<50:39, 138kB/s]   .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<35:42, 196kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<25:58, 267kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<19:25, 357kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<13:51, 500kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<09:42, 708kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<17:13, 399kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<13:36, 504kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<09:59, 686kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<07:09, 956kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<05:12, 1.31MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<06:38, 1.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<05:17, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:53, 1.74MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<04:25, 1.52MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<05:31, 1.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<04:26, 1.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:15, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<03:59, 1.67MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<03:24, 1.95MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:33, 2.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<03:33, 1.86MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<03:31, 1.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:42, 2.42MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<03:03, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<03:35, 1.81MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<02:49, 2.30MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:09, 3.00MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<02:57, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<02:58, 2.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<02:15, 2.84MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<02:47, 2.28MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:27<03:09, 2.02MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<02:29, 2.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<02:47, 2.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<03:08, 2.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<02:28, 2.54MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:31<02:46, 2.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:31<03:07, 2.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<02:26, 2.56MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<01:46, 3.49MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<05:49, 1.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:33<06:23, 966kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<05:03, 1.22MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:39, 1.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:35<04:04, 1.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:35<03:58, 1.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:03, 2.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<03:08, 1.93MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<03:17, 1.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:32, 2.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<01:49, 3.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<30:39, 195kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:39<22:31, 265kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<15:57, 373kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<11:11, 529kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<10:56, 540kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:41<08:43, 677kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<06:20, 928kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<05:23, 1.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:43<04:49, 1.21MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<03:37, 1.60MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<03:29, 1.65MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<04:36, 1.25MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<03:45, 1.53MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:45, 2.08MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:20, 1.71MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:47<03:24, 1.67MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<02:37, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<01:52, 2.99MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<10:33, 533kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<08:27, 665kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<06:07, 917kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:21, 1.28MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:59, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<04:34, 1.22MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<03:26, 1.61MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:26, 2.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<41:44, 132kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:53<30:14, 181kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<21:20, 256kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<14:53, 365kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<15:30, 350kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:55<11:50, 458kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<08:31, 635kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<05:58, 897kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<09:58, 537kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<07:57, 673kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<05:46, 923kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<04:04, 1.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<09:04, 583kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<07:17, 724kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<05:18, 992kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<03:46, 1.39MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<04:54, 1.06MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<04:23, 1.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<03:18, 1.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<03:09, 1.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:09, 1.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<02:24, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<01:44, 2.92MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<04:04, 1.25MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:46, 1.34MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:51, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:02, 2.47MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<08:17, 604kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<06:42, 747kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<04:53, 1.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<03:28, 1.43MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<04:30, 1.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<04:03, 1.22MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:04, 1.60MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:15, 2.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:49, 1.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:51, 1.71MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:13, 2.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:20, 2.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:32, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<01:59, 2.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:10, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<02:08, 2.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:38, 2.86MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<02:03, 2.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<02:14, 2.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<01:44, 2.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:16, 3.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<02:59, 1.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<03:49, 1.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:05, 1.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:14, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<02:44, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:42, 1.67MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:03, 2.18MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:29, 2.98MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<03:22, 1.32MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<03:10, 1.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:22, 1.87MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:44, 2.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<02:40, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<02:54, 1.51MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:13, 1.97MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:37, 2.69MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:36, 1.66MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:34, 1.67MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<01:57, 2.20MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:25, 2.99MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<02:57, 1.43MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<02:48, 1.51MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:07, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:32, 2.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<03:34, 1.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<03:13, 1.29MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<02:26, 1.71MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<02:23, 1.72MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<02:25, 1.70MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<01:52, 2.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<01:58, 2.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<02:05, 1.93MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<01:36, 2.49MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:11, 3.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:36<02:27, 1.62MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:25, 1.63MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<01:52, 2.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<01:57, 1.99MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<02:03, 1.90MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:35, 2.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:08, 3.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<03:55, 981kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<03:24, 1.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:32, 1.50MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:42<02:24, 1.57MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<02:20, 1.61MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:46, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<01:18, 2.85MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<02:07, 1.74MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<02:07, 1.73MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<01:37, 2.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:11, 3.06MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<02:12, 1.64MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<02:12, 1.65MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<01:40, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:12, 2.97MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<08:56, 399kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:48<06:52, 518kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<04:56, 717kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<04:00, 872kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<03:26, 1.01MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<02:33, 1.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<02:20, 1.46MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<02:14, 1.53MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<01:41, 2.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:12, 2.79MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<04:24, 763kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<03:39, 916kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<02:40, 1.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:53, 1.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<05:04, 649kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<04:07, 798kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<02:59, 1.09MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:56<02:06, 1.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<05:04, 635kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<04:20, 742kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<03:11, 1.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<02:17, 1.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:21, 1.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:27, 1.28MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:54, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:21, 2.28MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<02:23, 1.29MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<02:23, 1.29MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:49, 1.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:19, 2.30MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:50, 1.63MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:59, 1.51MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<01:33, 1.92MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:07, 2.63MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:05<02:50, 1.04MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:40, 1.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<02:00, 1.46MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:25, 2.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<02:03, 1.40MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<02:06, 1.36MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:37, 1.76MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:10, 2.40MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:34, 1.78MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<01:45, 1.60MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:23, 2.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<00:59, 2.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:36, 1.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:29, 1.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:52, 1.45MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:20, 2.00MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:47, 1.49MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:49, 1.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:24, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:00, 2.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<03:43, 696kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<03:09, 821kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:20, 1.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:38, 1.55MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<02:45, 916kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:28, 1.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:17<01:50, 1.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:18, 1.90MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:59, 1.24MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:53, 1.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:25, 1.71MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:01, 2.36MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:37, 1.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:41, 1.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:18, 1.83MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<00:55, 2.52MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:36, 1.44MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:39, 1.39MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:17, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:55, 2.45MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<02:13, 1.01MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<02:04, 1.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:33, 1.44MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:07, 1.97MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:21, 1.60MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:22, 1.58MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:03, 2.03MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:45, 2.80MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<03:54, 541kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<03:09, 668kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<02:18, 911kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<01:35, 1.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<10:18, 198kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<07:35, 269kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<05:22, 377kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<03:42, 535kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<06:37, 298kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<05:01, 393kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<03:35, 547kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<02:28, 773kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<04:52, 392kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<03:52, 492kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<02:48, 675kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:57, 949kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<02:29, 737kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<02:05, 879kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:32, 1.18MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<01:04, 1.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<04:22, 406kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<03:29, 508kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<02:31, 694kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:45, 978kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<02:16, 750kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<01:59, 856kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:28, 1.14MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:01, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<01:50, 890kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<01:39, 981kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<01:14, 1.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:52, 1.81MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<01:40, 932kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<01:28, 1.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<01:06, 1.40MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:46, 1.95MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<03:02, 490kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<02:29, 598kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<01:48, 818kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:15, 1.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<01:30, 947kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<01:23, 1.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:01, 1.37MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:43, 1.90MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<01:08, 1.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<01:03, 1.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:47, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:33, 2.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<01:43, 747kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<01:31, 846kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:07, 1.14MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:47, 1.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<00:50, 1.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<00:51, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:39, 1.82MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:27, 2.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<01:09, 987kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<01:02, 1.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:45, 1.48MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:32, 2.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<00:46, 1.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:45, 1.43MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:34, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:23, 2.56MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<02:01, 499kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<01:36, 626kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<01:09, 857kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:47, 1.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:03<02:10, 432kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:03<01:42, 552kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:12, 764kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:50, 1.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:52, 1.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:48, 1.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:35, 1.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:25, 1.99MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:31, 1.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<00:33, 1.42MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:25, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:17, 2.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:30, 1.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:09<00:31, 1.39MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:09<00:24, 1.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:16, 2.46MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:40, 974kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:11<00:37, 1.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:27, 1.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:19, 1.92MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:23, 1.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:24, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:18, 1.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:12, 2.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:32, 978kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:15<00:30, 1.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:15<00:22, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:15, 1.92MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:19, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:20, 1.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:15, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:09, 2.40MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:24, 965kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:21, 1.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:15, 1.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:09, 2.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<01:37, 196kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:21<01:12, 262kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:49, 366kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:30, 519kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:30, 496kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:23<00:23, 626kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:16, 859kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:09, 1.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<01:02, 173kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:45, 233kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:30, 327kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:16, 464kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:12, 517kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:10, 625kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:06, 846kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.19MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:03, 815kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:02, 907kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:01, 1.20MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 738/400000 [00:00<00:54, 7370.76it/s]  0%|          | 1561/400000 [00:00<00:52, 7608.21it/s]  1%|          | 2357/400000 [00:00<00:51, 7709.07it/s]  1%|          | 3185/400000 [00:00<00:50, 7871.19it/s]  1%|          | 4005/400000 [00:00<00:49, 7964.55it/s]  1%|          | 4793/400000 [00:00<00:49, 7938.79it/s]  1%|▏         | 5602/400000 [00:00<00:49, 7983.04it/s]  2%|▏         | 6399/400000 [00:00<00:49, 7977.13it/s]  2%|▏         | 7215/400000 [00:00<00:48, 8030.71it/s]  2%|▏         | 8035/400000 [00:01<00:48, 8079.46it/s]  2%|▏         | 8847/400000 [00:01<00:48, 8090.54it/s]  2%|▏         | 9642/400000 [00:01<00:49, 7931.47it/s]  3%|▎         | 10431/400000 [00:01<00:49, 7917.73it/s]  3%|▎         | 11258/400000 [00:01<00:48, 8018.10it/s]  3%|▎         | 12079/400000 [00:01<00:48, 8074.16it/s]  3%|▎         | 12886/400000 [00:01<00:47, 8069.40it/s]  3%|▎         | 13691/400000 [00:01<00:48, 7895.01it/s]  4%|▎         | 14480/400000 [00:01<00:49, 7773.64it/s]  4%|▍         | 15265/400000 [00:01<00:49, 7794.38it/s]  4%|▍         | 16057/400000 [00:02<00:49, 7831.07it/s]  4%|▍         | 16847/400000 [00:02<00:48, 7851.48it/s]  4%|▍         | 17633/400000 [00:02<00:48, 7831.54it/s]  5%|▍         | 18417/400000 [00:02<00:48, 7794.53it/s]  5%|▍         | 19197/400000 [00:02<00:48, 7778.34it/s]  5%|▍         | 19983/400000 [00:02<00:48, 7801.73it/s]  5%|▌         | 20764/400000 [00:02<00:49, 7729.09it/s]  5%|▌         | 21538/400000 [00:02<00:49, 7720.67it/s]  6%|▌         | 22327/400000 [00:02<00:48, 7770.40it/s]  6%|▌         | 23144/400000 [00:02<00:47, 7882.22it/s]  6%|▌         | 23941/400000 [00:03<00:47, 7907.43it/s]  6%|▌         | 24750/400000 [00:03<00:47, 7959.97it/s]  6%|▋         | 25574/400000 [00:03<00:46, 8039.69it/s]  7%|▋         | 26386/400000 [00:03<00:46, 8063.03it/s]  7%|▋         | 27210/400000 [00:03<00:45, 8111.74it/s]  7%|▋         | 28058/400000 [00:03<00:45, 8217.92it/s]  7%|▋         | 28889/400000 [00:03<00:45, 8242.67it/s]  7%|▋         | 29714/400000 [00:03<00:45, 8163.49it/s]  8%|▊         | 30531/400000 [00:03<00:46, 7938.04it/s]  8%|▊         | 31328/400000 [00:03<00:46, 7946.43it/s]  8%|▊         | 32124/400000 [00:04<00:46, 7891.10it/s]  8%|▊         | 32915/400000 [00:04<00:46, 7877.24it/s]  8%|▊         | 33704/400000 [00:04<00:46, 7840.74it/s]  9%|▊         | 34489/400000 [00:04<00:46, 7789.29it/s]  9%|▉         | 35275/400000 [00:04<00:46, 7807.69it/s]  9%|▉         | 36057/400000 [00:04<00:46, 7771.38it/s]  9%|▉         | 36868/400000 [00:04<00:46, 7869.29it/s]  9%|▉         | 37656/400000 [00:04<00:46, 7779.67it/s] 10%|▉         | 38435/400000 [00:04<00:47, 7618.20it/s] 10%|▉         | 39198/400000 [00:04<00:47, 7562.70it/s] 10%|▉         | 39956/400000 [00:05<00:47, 7525.31it/s] 10%|█         | 40740/400000 [00:05<00:47, 7614.30it/s] 10%|█         | 41561/400000 [00:05<00:46, 7783.38it/s] 11%|█         | 42348/400000 [00:05<00:45, 7806.84it/s] 11%|█         | 43173/400000 [00:05<00:44, 7934.19it/s] 11%|█         | 43985/400000 [00:05<00:44, 7988.50it/s] 11%|█         | 44785/400000 [00:05<00:44, 7974.08it/s] 11%|█▏        | 45584/400000 [00:05<00:45, 7856.99it/s] 12%|█▏        | 46371/400000 [00:05<00:45, 7689.37it/s] 12%|█▏        | 47197/400000 [00:05<00:44, 7848.70it/s] 12%|█▏        | 47985/400000 [00:06<00:44, 7856.51it/s] 12%|█▏        | 48818/400000 [00:06<00:43, 7991.82it/s] 12%|█▏        | 49661/400000 [00:06<00:43, 8115.58it/s] 13%|█▎        | 50475/400000 [00:06<00:43, 8086.56it/s] 13%|█▎        | 51285/400000 [00:06<00:43, 8051.32it/s] 13%|█▎        | 52091/400000 [00:06<00:43, 8038.24it/s] 13%|█▎        | 52896/400000 [00:06<00:43, 7893.39it/s] 13%|█▎        | 53687/400000 [00:06<00:44, 7796.84it/s] 14%|█▎        | 54468/400000 [00:06<00:45, 7659.84it/s] 14%|█▍        | 55287/400000 [00:06<00:44, 7810.89it/s] 14%|█▍        | 56108/400000 [00:07<00:43, 7925.05it/s] 14%|█▍        | 56922/400000 [00:07<00:42, 7985.84it/s] 14%|█▍        | 57724/400000 [00:07<00:42, 7995.89it/s] 15%|█▍        | 58525/400000 [00:07<00:42, 7964.25it/s] 15%|█▍        | 59323/400000 [00:07<00:42, 7937.80it/s] 15%|█▌        | 60137/400000 [00:07<00:42, 7993.21it/s] 15%|█▌        | 60967/400000 [00:07<00:41, 8080.70it/s] 15%|█▌        | 61776/400000 [00:07<00:42, 7935.87it/s] 16%|█▌        | 62571/400000 [00:07<00:42, 7880.64it/s] 16%|█▌        | 63370/400000 [00:08<00:42, 7912.76it/s] 16%|█▌        | 64172/400000 [00:08<00:42, 7944.44it/s] 16%|█▌        | 64994/400000 [00:08<00:41, 8022.86it/s] 16%|█▋        | 65814/400000 [00:08<00:41, 8073.12it/s] 17%|█▋        | 66622/400000 [00:08<00:42, 7933.47it/s] 17%|█▋        | 67433/400000 [00:08<00:41, 7982.65it/s] 17%|█▋        | 68253/400000 [00:08<00:41, 8045.71it/s] 17%|█▋        | 69059/400000 [00:08<00:41, 8021.56it/s] 17%|█▋        | 69881/400000 [00:08<00:40, 8079.62it/s] 18%|█▊        | 70707/400000 [00:08<00:40, 8129.28it/s] 18%|█▊        | 71521/400000 [00:09<00:40, 8097.96it/s] 18%|█▊        | 72332/400000 [00:09<00:40, 8048.79it/s] 18%|█▊        | 73183/400000 [00:09<00:39, 8180.57it/s] 19%|█▊        | 74036/400000 [00:09<00:39, 8281.49it/s] 19%|█▊        | 74865/400000 [00:09<00:39, 8139.44it/s] 19%|█▉        | 75681/400000 [00:09<00:40, 7992.72it/s] 19%|█▉        | 76482/400000 [00:09<00:42, 7672.93it/s] 19%|█▉        | 77253/400000 [00:09<00:42, 7570.01it/s] 20%|█▉        | 78051/400000 [00:09<00:41, 7687.44it/s] 20%|█▉        | 78853/400000 [00:09<00:41, 7782.23it/s] 20%|█▉        | 79647/400000 [00:10<00:40, 7828.16it/s] 20%|██        | 80432/400000 [00:10<00:40, 7819.67it/s] 20%|██        | 81216/400000 [00:10<00:40, 7825.29it/s] 21%|██        | 82020/400000 [00:10<00:40, 7887.20it/s] 21%|██        | 82811/400000 [00:10<00:40, 7890.96it/s] 21%|██        | 83601/400000 [00:10<00:40, 7847.13it/s] 21%|██        | 84387/400000 [00:10<00:40, 7763.15it/s] 21%|██▏       | 85180/400000 [00:10<00:40, 7811.96it/s] 21%|██▏       | 85962/400000 [00:10<00:40, 7754.93it/s] 22%|██▏       | 86766/400000 [00:10<00:39, 7836.75it/s] 22%|██▏       | 87551/400000 [00:11<00:39, 7836.57it/s] 22%|██▏       | 88370/400000 [00:11<00:39, 7936.51it/s] 22%|██▏       | 89175/400000 [00:11<00:39, 7967.90it/s] 22%|██▏       | 89986/400000 [00:11<00:38, 8009.25it/s] 23%|██▎       | 90788/400000 [00:11<00:38, 7932.98it/s] 23%|██▎       | 91610/400000 [00:11<00:38, 8015.44it/s] 23%|██▎       | 92413/400000 [00:11<00:38, 7958.14it/s] 23%|██▎       | 93210/400000 [00:11<00:39, 7723.92it/s] 23%|██▎       | 93985/400000 [00:11<00:39, 7702.93it/s] 24%|██▎       | 94757/400000 [00:11<00:39, 7658.28it/s] 24%|██▍       | 95580/400000 [00:12<00:38, 7820.32it/s] 24%|██▍       | 96382/400000 [00:12<00:38, 7878.33it/s] 24%|██▍       | 97175/400000 [00:12<00:38, 7891.82it/s] 24%|██▍       | 97978/400000 [00:12<00:38, 7931.82it/s] 25%|██▍       | 98773/400000 [00:12<00:37, 7934.21it/s] 25%|██▍       | 99567/400000 [00:12<00:38, 7895.26it/s] 25%|██▌       | 100379/400000 [00:12<00:37, 7960.42it/s] 25%|██▌       | 101188/400000 [00:12<00:37, 7996.41it/s] 26%|██▌       | 102011/400000 [00:12<00:36, 8064.50it/s] 26%|██▌       | 102818/400000 [00:12<00:37, 7938.52it/s] 26%|██▌       | 103613/400000 [00:13<00:37, 7940.19it/s] 26%|██▌       | 104431/400000 [00:13<00:36, 8009.44it/s] 26%|██▋       | 105247/400000 [00:13<00:36, 8052.50it/s] 27%|██▋       | 106053/400000 [00:13<00:36, 7985.35it/s] 27%|██▋       | 106852/400000 [00:13<00:37, 7916.96it/s] 27%|██▋       | 107645/400000 [00:13<00:37, 7882.68it/s] 27%|██▋       | 108444/400000 [00:13<00:36, 7912.24it/s] 27%|██▋       | 109250/400000 [00:13<00:36, 7955.81it/s] 28%|██▊       | 110059/400000 [00:13<00:36, 7992.10it/s] 28%|██▊       | 110859/400000 [00:14<00:36, 7869.27it/s] 28%|██▊       | 111647/400000 [00:14<00:36, 7814.46it/s] 28%|██▊       | 112447/400000 [00:14<00:36, 7867.96it/s] 28%|██▊       | 113235/400000 [00:14<00:36, 7768.86it/s] 29%|██▊       | 114049/400000 [00:14<00:36, 7876.41it/s] 29%|██▊       | 114838/400000 [00:14<00:36, 7867.74it/s] 29%|██▉       | 115626/400000 [00:14<00:36, 7828.58it/s] 29%|██▉       | 116410/400000 [00:14<00:36, 7754.18it/s] 29%|██▉       | 117219/400000 [00:14<00:36, 7849.71it/s] 30%|██▉       | 118005/400000 [00:14<00:36, 7824.97it/s] 30%|██▉       | 118788/400000 [00:15<00:36, 7663.85it/s] 30%|██▉       | 119568/400000 [00:15<00:36, 7704.12it/s] 30%|███       | 120364/400000 [00:15<00:35, 7778.13it/s] 30%|███       | 121181/400000 [00:15<00:35, 7891.21it/s] 31%|███       | 122009/400000 [00:15<00:34, 8001.86it/s] 31%|███       | 122811/400000 [00:15<00:34, 7984.37it/s] 31%|███       | 123624/400000 [00:15<00:34, 8025.18it/s] 31%|███       | 124439/400000 [00:15<00:34, 8061.03it/s] 31%|███▏      | 125247/400000 [00:15<00:34, 8064.57it/s] 32%|███▏      | 126054/400000 [00:15<00:34, 8029.05it/s] 32%|███▏      | 126858/400000 [00:16<00:34, 7988.77it/s] 32%|███▏      | 127658/400000 [00:16<00:34, 7987.97it/s] 32%|███▏      | 128476/400000 [00:16<00:33, 8042.60it/s] 32%|███▏      | 129283/400000 [00:16<00:33, 8050.04it/s] 33%|███▎      | 130089/400000 [00:16<00:33, 8035.37it/s] 33%|███▎      | 130893/400000 [00:16<00:33, 7992.24it/s] 33%|███▎      | 131693/400000 [00:16<00:33, 7981.20it/s] 33%|███▎      | 132501/400000 [00:16<00:33, 8007.97it/s] 33%|███▎      | 133302/400000 [00:16<00:33, 7970.07it/s] 34%|███▎      | 134111/400000 [00:16<00:33, 8003.08it/s] 34%|███▎      | 134912/400000 [00:17<00:33, 7974.83it/s] 34%|███▍      | 135710/400000 [00:17<00:33, 7949.82it/s] 34%|███▍      | 136519/400000 [00:17<00:32, 7988.54it/s] 34%|███▍      | 137320/400000 [00:17<00:32, 7992.34it/s] 35%|███▍      | 138124/400000 [00:17<00:32, 8006.37it/s] 35%|███▍      | 138925/400000 [00:17<00:32, 7966.31it/s] 35%|███▍      | 139722/400000 [00:17<00:32, 7931.19it/s] 35%|███▌      | 140516/400000 [00:17<00:32, 7933.31it/s] 35%|███▌      | 141310/400000 [00:17<00:32, 7919.86it/s] 36%|███▌      | 142115/400000 [00:17<00:32, 7956.75it/s] 36%|███▌      | 142928/400000 [00:18<00:32, 8007.05it/s] 36%|███▌      | 143735/400000 [00:18<00:31, 8023.63it/s] 36%|███▌      | 144575/400000 [00:18<00:31, 8130.12it/s] 36%|███▋      | 145423/400000 [00:18<00:30, 8231.97it/s] 37%|███▋      | 146261/400000 [00:18<00:30, 8275.34it/s] 37%|███▋      | 147090/400000 [00:18<00:30, 8200.06it/s] 37%|███▋      | 147916/400000 [00:18<00:30, 8216.40it/s] 37%|███▋      | 148770/400000 [00:18<00:30, 8310.17it/s] 37%|███▋      | 149627/400000 [00:18<00:29, 8386.28it/s] 38%|███▊      | 150483/400000 [00:18<00:29, 8435.23it/s] 38%|███▊      | 151327/400000 [00:19<00:29, 8378.51it/s] 38%|███▊      | 152166/400000 [00:19<00:29, 8349.54it/s] 38%|███▊      | 153002/400000 [00:19<00:29, 8332.47it/s] 38%|███▊      | 153836/400000 [00:19<00:29, 8311.34it/s] 39%|███▊      | 154668/400000 [00:19<00:29, 8201.74it/s] 39%|███▉      | 155489/400000 [00:19<00:29, 8164.12it/s] 39%|███▉      | 156306/400000 [00:19<00:30, 8120.34it/s] 39%|███▉      | 157131/400000 [00:19<00:29, 8156.19it/s] 39%|███▉      | 157947/400000 [00:19<00:29, 8123.76it/s] 40%|███▉      | 158764/400000 [00:19<00:29, 8137.43it/s] 40%|███▉      | 159578/400000 [00:20<00:30, 7904.70it/s] 40%|████      | 160371/400000 [00:20<00:30, 7893.03it/s] 40%|████      | 161204/400000 [00:20<00:29, 8017.61it/s] 41%|████      | 162022/400000 [00:20<00:29, 8065.46it/s] 41%|████      | 162844/400000 [00:20<00:29, 8108.83it/s] 41%|████      | 163656/400000 [00:20<00:29, 8059.10it/s] 41%|████      | 164486/400000 [00:20<00:28, 8129.34it/s] 41%|████▏     | 165357/400000 [00:20<00:28, 8293.42it/s] 42%|████▏     | 166230/400000 [00:20<00:27, 8417.24it/s] 42%|████▏     | 167090/400000 [00:20<00:27, 8469.46it/s] 42%|████▏     | 167938/400000 [00:21<00:27, 8340.42it/s] 42%|████▏     | 168774/400000 [00:21<00:28, 8222.45it/s] 42%|████▏     | 169598/400000 [00:21<00:28, 8108.87it/s] 43%|████▎     | 170411/400000 [00:21<00:28, 8041.22it/s] 43%|████▎     | 171217/400000 [00:21<00:28, 7952.12it/s] 43%|████▎     | 172014/400000 [00:21<00:28, 7886.31it/s] 43%|████▎     | 172804/400000 [00:21<00:28, 7861.39it/s] 43%|████▎     | 173591/400000 [00:21<00:28, 7809.30it/s] 44%|████▎     | 174373/400000 [00:21<00:28, 7783.00it/s] 44%|████▍     | 175186/400000 [00:22<00:28, 7883.93it/s] 44%|████▍     | 175977/400000 [00:22<00:28, 7890.50it/s] 44%|████▍     | 176767/400000 [00:22<00:28, 7715.01it/s] 44%|████▍     | 177586/400000 [00:22<00:28, 7849.07it/s] 45%|████▍     | 178373/400000 [00:22<00:28, 7821.15it/s] 45%|████▍     | 179157/400000 [00:22<00:28, 7807.64it/s] 45%|████▍     | 179939/400000 [00:22<00:29, 7547.77it/s] 45%|████▌     | 180758/400000 [00:22<00:28, 7728.27it/s] 45%|████▌     | 181574/400000 [00:22<00:27, 7852.75it/s] 46%|████▌     | 182403/400000 [00:22<00:27, 7978.25it/s] 46%|████▌     | 183206/400000 [00:23<00:27, 7992.78it/s] 46%|████▌     | 184007/400000 [00:23<00:27, 7846.91it/s] 46%|████▌     | 184794/400000 [00:23<00:27, 7692.42it/s] 46%|████▋     | 185577/400000 [00:23<00:27, 7731.10it/s] 47%|████▋     | 186352/400000 [00:23<00:27, 7706.02it/s] 47%|████▋     | 187144/400000 [00:23<00:27, 7768.79it/s] 47%|████▋     | 187922/400000 [00:23<00:27, 7747.05it/s] 47%|████▋     | 188720/400000 [00:23<00:27, 7813.42it/s] 47%|████▋     | 189502/400000 [00:23<00:26, 7810.28it/s] 48%|████▊     | 190310/400000 [00:23<00:26, 7888.20it/s] 48%|████▊     | 191149/400000 [00:24<00:26, 8029.93it/s] 48%|████▊     | 191957/400000 [00:24<00:25, 8042.97it/s] 48%|████▊     | 192801/400000 [00:24<00:25, 8156.08it/s] 48%|████▊     | 193635/400000 [00:24<00:25, 8208.77it/s] 49%|████▊     | 194457/400000 [00:24<00:25, 8193.18it/s] 49%|████▉     | 195291/400000 [00:24<00:24, 8236.50it/s] 49%|████▉     | 196116/400000 [00:24<00:24, 8217.20it/s] 49%|████▉     | 196939/400000 [00:24<00:24, 8183.51it/s] 49%|████▉     | 197775/400000 [00:24<00:24, 8233.53it/s] 50%|████▉     | 198599/400000 [00:24<00:25, 8032.96it/s] 50%|████▉     | 199404/400000 [00:25<00:25, 7826.22it/s] 50%|█████     | 200189/400000 [00:25<00:26, 7683.57it/s] 50%|█████     | 200960/400000 [00:25<00:26, 7651.48it/s] 50%|█████     | 201727/400000 [00:25<00:26, 7571.35it/s] 51%|█████     | 202506/400000 [00:25<00:25, 7634.21it/s] 51%|█████     | 203271/400000 [00:25<00:25, 7596.84it/s] 51%|█████     | 204032/400000 [00:25<00:25, 7550.46it/s] 51%|█████     | 204788/400000 [00:25<00:26, 7447.83it/s] 51%|█████▏    | 205579/400000 [00:25<00:25, 7578.80it/s] 52%|█████▏    | 206377/400000 [00:25<00:25, 7694.52it/s] 52%|█████▏    | 207169/400000 [00:26<00:24, 7758.36it/s] 52%|█████▏    | 207957/400000 [00:26<00:24, 7792.37it/s] 52%|█████▏    | 208755/400000 [00:26<00:24, 7846.14it/s] 52%|█████▏    | 209579/400000 [00:26<00:23, 7960.24it/s] 53%|█████▎    | 210376/400000 [00:26<00:24, 7837.34it/s] 53%|█████▎    | 211181/400000 [00:26<00:23, 7897.54it/s] 53%|█████▎    | 211972/400000 [00:26<00:23, 7896.69it/s] 53%|█████▎    | 212779/400000 [00:26<00:23, 7947.77it/s] 53%|█████▎    | 213575/400000 [00:26<00:23, 7939.55it/s] 54%|█████▎    | 214374/400000 [00:26<00:23, 7952.97it/s] 54%|█████▍    | 215189/400000 [00:27<00:23, 8010.77it/s] 54%|█████▍    | 215991/400000 [00:27<00:23, 7962.79it/s] 54%|█████▍    | 216788/400000 [00:27<00:23, 7935.85it/s] 54%|█████▍    | 217593/400000 [00:27<00:22, 7969.22it/s] 55%|█████▍    | 218409/400000 [00:27<00:22, 8022.82it/s] 55%|█████▍    | 219212/400000 [00:27<00:22, 7907.68it/s] 55%|█████▌    | 220004/400000 [00:27<00:22, 7888.71it/s] 55%|█████▌    | 220809/400000 [00:27<00:22, 7935.48it/s] 55%|█████▌    | 221609/400000 [00:27<00:22, 7952.54it/s] 56%|█████▌    | 222455/400000 [00:28<00:21, 8097.93it/s] 56%|█████▌    | 223267/400000 [00:28<00:21, 8103.22it/s] 56%|█████▌    | 224078/400000 [00:28<00:21, 8024.34it/s] 56%|█████▌    | 224882/400000 [00:28<00:21, 7992.12it/s] 56%|█████▋    | 225688/400000 [00:28<00:21, 8011.19it/s] 57%|█████▋    | 226490/400000 [00:28<00:21, 7969.97it/s] 57%|█████▋    | 227288/400000 [00:28<00:21, 7903.06it/s] 57%|█████▋    | 228079/400000 [00:28<00:21, 7883.27it/s] 57%|█████▋    | 228871/400000 [00:28<00:21, 7894.18it/s] 57%|█████▋    | 229673/400000 [00:28<00:21, 7930.28it/s] 58%|█████▊    | 230475/400000 [00:29<00:21, 7955.31it/s] 58%|█████▊    | 231271/400000 [00:29<00:21, 7861.17it/s] 58%|█████▊    | 232067/400000 [00:29<00:21, 7887.73it/s] 58%|█████▊    | 232857/400000 [00:29<00:21, 7862.21it/s] 58%|█████▊    | 233644/400000 [00:29<00:21, 7752.11it/s] 59%|█████▊    | 234420/400000 [00:29<00:21, 7654.79it/s] 59%|█████▉    | 235187/400000 [00:29<00:21, 7580.63it/s] 59%|█████▉    | 235952/400000 [00:29<00:21, 7599.37it/s] 59%|█████▉    | 236736/400000 [00:29<00:21, 7667.70it/s] 59%|█████▉    | 237525/400000 [00:29<00:21, 7731.05it/s] 60%|█████▉    | 238299/400000 [00:30<00:20, 7700.96it/s] 60%|█████▉    | 239070/400000 [00:30<00:21, 7584.24it/s] 60%|█████▉    | 239855/400000 [00:30<00:20, 7659.69it/s] 60%|██████    | 240650/400000 [00:30<00:20, 7741.41it/s] 60%|██████    | 241425/400000 [00:30<00:20, 7627.20it/s] 61%|██████    | 242189/400000 [00:30<00:20, 7535.80it/s] 61%|██████    | 242944/400000 [00:30<00:21, 7418.04it/s] 61%|██████    | 243687/400000 [00:30<00:21, 7340.23it/s] 61%|██████    | 244438/400000 [00:30<00:21, 7389.20it/s] 61%|██████▏   | 245219/400000 [00:30<00:20, 7508.66it/s] 61%|██████▏   | 245998/400000 [00:31<00:20, 7589.00it/s] 62%|██████▏   | 246776/400000 [00:31<00:20, 7643.57it/s] 62%|██████▏   | 247544/400000 [00:31<00:19, 7651.58it/s] 62%|██████▏   | 248310/400000 [00:31<00:20, 7559.46it/s] 62%|██████▏   | 249088/400000 [00:31<00:19, 7624.13it/s] 62%|██████▏   | 249874/400000 [00:31<00:19, 7692.34it/s] 63%|██████▎   | 250658/400000 [00:31<00:19, 7733.79it/s] 63%|██████▎   | 251432/400000 [00:31<00:19, 7730.45it/s] 63%|██████▎   | 252206/400000 [00:31<00:19, 7683.97it/s] 63%|██████▎   | 252977/400000 [00:31<00:19, 7689.44it/s] 63%|██████▎   | 253759/400000 [00:32<00:18, 7726.70it/s] 64%|██████▎   | 254544/400000 [00:32<00:18, 7761.92it/s] 64%|██████▍   | 255342/400000 [00:32<00:18, 7824.19it/s] 64%|██████▍   | 256125/400000 [00:32<00:18, 7784.34it/s] 64%|██████▍   | 256918/400000 [00:32<00:18, 7825.61it/s] 64%|██████▍   | 257716/400000 [00:32<00:18, 7870.90it/s] 65%|██████▍   | 258504/400000 [00:32<00:17, 7872.95it/s] 65%|██████▍   | 259310/400000 [00:32<00:17, 7927.04it/s] 65%|██████▌   | 260103/400000 [00:32<00:17, 7796.70it/s] 65%|██████▌   | 260926/400000 [00:32<00:17, 7919.92it/s] 65%|██████▌   | 261744/400000 [00:33<00:17, 7996.05it/s] 66%|██████▌   | 262545/400000 [00:33<00:17, 7987.07it/s] 66%|██████▌   | 263356/400000 [00:33<00:17, 8023.34it/s] 66%|██████▌   | 264159/400000 [00:33<00:17, 7934.21it/s] 66%|██████▌   | 264953/400000 [00:33<00:17, 7881.99it/s] 66%|██████▋   | 265742/400000 [00:33<00:17, 7589.14it/s] 67%|██████▋   | 266533/400000 [00:33<00:17, 7681.88it/s] 67%|██████▋   | 267357/400000 [00:33<00:16, 7840.15it/s] 67%|██████▋   | 268144/400000 [00:33<00:16, 7830.81it/s] 67%|██████▋   | 268937/400000 [00:33<00:16, 7859.93it/s] 67%|██████▋   | 269727/400000 [00:34<00:16, 7869.99it/s] 68%|██████▊   | 270515/400000 [00:34<00:16, 7865.27it/s] 68%|██████▊   | 271324/400000 [00:34<00:16, 7929.94it/s] 68%|██████▊   | 272118/400000 [00:34<00:16, 7890.35it/s] 68%|██████▊   | 272908/400000 [00:34<00:16, 7844.70it/s] 68%|██████▊   | 273714/400000 [00:34<00:15, 7905.99it/s] 69%|██████▊   | 274505/400000 [00:34<00:15, 7904.16it/s] 69%|██████▉   | 275296/400000 [00:34<00:16, 7624.33it/s] 69%|██████▉   | 276061/400000 [00:34<00:16, 7612.09it/s] 69%|██████▉   | 276852/400000 [00:35<00:16, 7696.42it/s] 69%|██████▉   | 277624/400000 [00:35<00:16, 7552.27it/s] 70%|██████▉   | 278381/400000 [00:35<00:16, 7519.97it/s] 70%|██████▉   | 279135/400000 [00:35<00:16, 7451.96it/s] 70%|██████▉   | 279921/400000 [00:35<00:15, 7556.84it/s] 70%|███████   | 280725/400000 [00:35<00:15, 7693.35it/s] 70%|███████   | 281496/400000 [00:35<00:15, 7656.61it/s] 71%|███████   | 282287/400000 [00:35<00:15, 7730.35it/s] 71%|███████   | 283088/400000 [00:35<00:14, 7811.53it/s] 71%|███████   | 283899/400000 [00:35<00:14, 7896.67it/s] 71%|███████   | 284692/400000 [00:36<00:14, 7905.64it/s] 71%|███████▏  | 285484/400000 [00:36<00:14, 7889.33it/s] 72%|███████▏  | 286298/400000 [00:36<00:14, 7962.64it/s] 72%|███████▏  | 287126/400000 [00:36<00:14, 8052.24it/s] 72%|███████▏  | 287932/400000 [00:36<00:14, 7936.54it/s] 72%|███████▏  | 288727/400000 [00:36<00:14, 7889.37it/s] 72%|███████▏  | 289517/400000 [00:36<00:14, 7635.11it/s] 73%|███████▎  | 290301/400000 [00:36<00:14, 7692.86it/s] 73%|███████▎  | 291120/400000 [00:36<00:13, 7833.91it/s] 73%|███████▎  | 291913/400000 [00:36<00:13, 7861.37it/s] 73%|███████▎  | 292743/400000 [00:37<00:13, 7986.30it/s] 73%|███████▎  | 293593/400000 [00:37<00:13, 8132.68it/s] 74%|███████▎  | 294421/400000 [00:37<00:12, 8175.19it/s] 74%|███████▍  | 295265/400000 [00:37<00:12, 8252.15it/s] 74%|███████▍  | 296092/400000 [00:37<00:12, 8043.59it/s] 74%|███████▍  | 296908/400000 [00:37<00:12, 8075.62it/s] 74%|███████▍  | 297717/400000 [00:37<00:12, 7998.51it/s] 75%|███████▍  | 298539/400000 [00:37<00:12, 8061.61it/s] 75%|███████▍  | 299356/400000 [00:37<00:12, 8091.78it/s] 75%|███████▌  | 300166/400000 [00:37<00:12, 8054.24it/s] 75%|███████▌  | 300972/400000 [00:38<00:12, 8046.54it/s] 75%|███████▌  | 301796/400000 [00:38<00:12, 8102.89it/s] 76%|███████▌  | 302620/400000 [00:38<00:11, 8143.26it/s] 76%|███████▌  | 303452/400000 [00:38<00:11, 8193.64it/s] 76%|███████▌  | 304272/400000 [00:38<00:11, 8114.13it/s] 76%|███████▋  | 305084/400000 [00:38<00:11, 8059.61it/s] 76%|███████▋  | 305891/400000 [00:38<00:11, 7899.06it/s] 77%|███████▋  | 306682/400000 [00:38<00:11, 7887.02it/s] 77%|███████▋  | 307475/400000 [00:38<00:11, 7898.63it/s] 77%|███████▋  | 308266/400000 [00:38<00:11, 7662.15it/s] 77%|███████▋  | 309063/400000 [00:39<00:11, 7750.45it/s] 77%|███████▋  | 309882/400000 [00:39<00:11, 7876.10it/s] 78%|███████▊  | 310697/400000 [00:39<00:11, 7954.39it/s] 78%|███████▊  | 311507/400000 [00:39<00:11, 7997.37it/s] 78%|███████▊  | 312308/400000 [00:39<00:11, 7917.12it/s] 78%|███████▊  | 313101/400000 [00:39<00:11, 7828.38it/s] 78%|███████▊  | 313898/400000 [00:39<00:10, 7869.99it/s] 79%|███████▊  | 314687/400000 [00:39<00:10, 7874.15it/s] 79%|███████▉  | 315481/400000 [00:39<00:10, 7890.64it/s] 79%|███████▉  | 316271/400000 [00:39<00:10, 7866.44it/s] 79%|███████▉  | 317058/400000 [00:40<00:10, 7800.01it/s] 79%|███████▉  | 317839/400000 [00:40<00:10, 7799.82it/s] 80%|███████▉  | 318620/400000 [00:40<00:10, 7791.97it/s] 80%|███████▉  | 319417/400000 [00:40<00:10, 7844.10it/s] 80%|████████  | 320209/400000 [00:40<00:10, 7864.59it/s] 80%|████████  | 320996/400000 [00:40<00:10, 7849.16it/s] 80%|████████  | 321793/400000 [00:40<00:09, 7882.47it/s] 81%|████████  | 322582/400000 [00:40<00:09, 7853.00it/s] 81%|████████  | 323370/400000 [00:40<00:09, 7858.41it/s] 81%|████████  | 324156/400000 [00:40<00:09, 7833.48it/s] 81%|████████  | 324940/400000 [00:41<00:09, 7802.70it/s] 81%|████████▏ | 325721/400000 [00:41<00:09, 7800.08it/s] 82%|████████▏ | 326502/400000 [00:41<00:09, 7797.82it/s] 82%|████████▏ | 327304/400000 [00:41<00:09, 7861.89it/s] 82%|████████▏ | 328102/400000 [00:41<00:09, 7896.29it/s] 82%|████████▏ | 328892/400000 [00:41<00:09, 7858.56it/s] 82%|████████▏ | 329691/400000 [00:41<00:08, 7897.31it/s] 83%|████████▎ | 330481/400000 [00:41<00:08, 7817.21it/s] 83%|████████▎ | 331264/400000 [00:41<00:08, 7648.27it/s] 83%|████████▎ | 332030/400000 [00:42<00:08, 7576.01it/s] 83%|████████▎ | 332837/400000 [00:42<00:08, 7716.35it/s] 83%|████████▎ | 333647/400000 [00:42<00:08, 7817.54it/s] 84%|████████▎ | 334451/400000 [00:42<00:08, 7882.90it/s] 84%|████████▍ | 335248/400000 [00:42<00:08, 7907.64it/s] 84%|████████▍ | 336040/400000 [00:42<00:08, 7907.34it/s] 84%|████████▍ | 336832/400000 [00:42<00:08, 7804.52it/s] 84%|████████▍ | 337622/400000 [00:42<00:07, 7830.79it/s] 85%|████████▍ | 338437/400000 [00:42<00:07, 7922.22it/s] 85%|████████▍ | 339244/400000 [00:42<00:07, 7964.43it/s] 85%|████████▌ | 340041/400000 [00:43<00:07, 7966.01it/s] 85%|████████▌ | 340848/400000 [00:43<00:07, 7996.60it/s] 85%|████████▌ | 341650/400000 [00:43<00:07, 8002.74it/s] 86%|████████▌ | 342454/400000 [00:43<00:07, 8011.53it/s] 86%|████████▌ | 343262/400000 [00:43<00:07, 8031.57it/s] 86%|████████▌ | 344066/400000 [00:43<00:07, 7838.38it/s] 86%|████████▌ | 344856/400000 [00:43<00:07, 7854.83it/s] 86%|████████▋ | 345659/400000 [00:43<00:06, 7904.79it/s] 87%|████████▋ | 346466/400000 [00:43<00:06, 7951.33it/s] 87%|████████▋ | 347291/400000 [00:43<00:06, 8036.17it/s] 87%|████████▋ | 348096/400000 [00:44<00:06, 8020.34it/s] 87%|████████▋ | 348899/400000 [00:44<00:06, 8000.47it/s] 87%|████████▋ | 349700/400000 [00:44<00:06, 7993.72it/s] 88%|████████▊ | 350516/400000 [00:44<00:06, 8040.36it/s] 88%|████████▊ | 351328/400000 [00:44<00:06, 8063.90it/s] 88%|████████▊ | 352135/400000 [00:44<00:06, 7784.22it/s] 88%|████████▊ | 352916/400000 [00:44<00:06, 7160.35it/s] 88%|████████▊ | 353680/400000 [00:44<00:06, 7296.81it/s] 89%|████████▊ | 354419/400000 [00:44<00:06, 7273.94it/s] 89%|████████▉ | 355163/400000 [00:44<00:06, 7320.39it/s] 89%|████████▉ | 355900/400000 [00:45<00:06, 7307.36it/s] 89%|████████▉ | 356690/400000 [00:45<00:05, 7474.87it/s] 89%|████████▉ | 357441/400000 [00:45<00:05, 7418.22it/s] 90%|████████▉ | 358205/400000 [00:45<00:05, 7482.69it/s] 90%|████████▉ | 358997/400000 [00:45<00:05, 7608.50it/s] 90%|████████▉ | 359760/400000 [00:45<00:05, 7597.15it/s] 90%|█████████ | 360558/400000 [00:45<00:05, 7708.07it/s] 90%|█████████ | 361368/400000 [00:45<00:04, 7820.32it/s] 91%|█████████ | 362155/400000 [00:45<00:04, 7834.31it/s] 91%|█████████ | 362963/400000 [00:45<00:04, 7902.77it/s] 91%|█████████ | 363755/400000 [00:46<00:04, 7800.50it/s] 91%|█████████ | 364543/400000 [00:46<00:04, 7822.20it/s] 91%|█████████▏| 365350/400000 [00:46<00:04, 7891.89it/s] 92%|█████████▏| 366161/400000 [00:46<00:04, 7954.31it/s] 92%|█████████▏| 366963/400000 [00:46<00:04, 7972.45it/s] 92%|█████████▏| 367761/400000 [00:46<00:04, 7843.20it/s] 92%|█████████▏| 368547/400000 [00:46<00:04, 7846.23it/s] 92%|█████████▏| 369364/400000 [00:46<00:03, 7939.11it/s] 93%|█████████▎| 370159/400000 [00:46<00:03, 7900.12it/s] 93%|█████████▎| 370950/400000 [00:46<00:03, 7857.44it/s] 93%|█████████▎| 371737/400000 [00:47<00:03, 7711.71it/s] 93%|█████████▎| 372522/400000 [00:47<00:03, 7750.18it/s] 93%|█████████▎| 373305/400000 [00:47<00:03, 7771.35it/s] 94%|█████████▎| 374092/400000 [00:47<00:03, 7799.78it/s] 94%|█████████▎| 374885/400000 [00:47<00:03, 7837.25it/s] 94%|█████████▍| 375670/400000 [00:47<00:03, 7620.91it/s] 94%|█████████▍| 376450/400000 [00:47<00:03, 7673.20it/s] 94%|█████████▍| 377253/400000 [00:47<00:02, 7773.05it/s] 95%|█████████▍| 378053/400000 [00:47<00:02, 7839.53it/s] 95%|█████████▍| 378843/400000 [00:48<00:02, 7770.80it/s] 95%|█████████▍| 379654/400000 [00:48<00:02, 7866.95it/s] 95%|█████████▌| 380509/400000 [00:48<00:02, 8059.06it/s] 95%|█████████▌| 381379/400000 [00:48<00:02, 8239.11it/s] 96%|█████████▌| 382264/400000 [00:48<00:02, 8411.14it/s] 96%|█████████▌| 383135/400000 [00:48<00:01, 8496.92it/s] 96%|█████████▌| 383987/400000 [00:48<00:01, 8252.90it/s] 96%|█████████▌| 384817/400000 [00:48<00:01, 8264.98it/s] 96%|█████████▋| 385690/400000 [00:48<00:01, 8397.01it/s] 97%|█████████▋| 386532/400000 [00:48<00:01, 8372.91it/s] 97%|█████████▋| 387371/400000 [00:49<00:01, 8273.88it/s] 97%|█████████▋| 388200/400000 [00:49<00:01, 8252.71it/s] 97%|█████████▋| 389085/400000 [00:49<00:01, 8420.97it/s] 97%|█████████▋| 389966/400000 [00:49<00:01, 8533.38it/s] 98%|█████████▊| 390850/400000 [00:49<00:01, 8621.67it/s] 98%|█████████▊| 391714/400000 [00:49<00:00, 8573.17it/s] 98%|█████████▊| 392573/400000 [00:49<00:00, 8362.72it/s] 98%|█████████▊| 393456/400000 [00:49<00:00, 8495.85it/s] 99%|█████████▊| 394308/400000 [00:49<00:00, 8375.51it/s] 99%|█████████▉| 395148/400000 [00:49<00:00, 8339.70it/s] 99%|█████████▉| 396025/400000 [00:50<00:00, 8463.14it/s] 99%|█████████▉| 396873/400000 [00:50<00:00, 7890.10it/s] 99%|█████████▉| 397680/400000 [00:50<00:00, 7940.65it/s]100%|█████████▉| 398500/400000 [00:50<00:00, 8015.48it/s]100%|█████████▉| 399354/400000 [00:50<00:00, 8163.48it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7912.47it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f83cbb0b940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01145323117574056 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.01099444791226084 	 Accuracy: 62

  model saves at 62% accuracy 

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
2020-05-15 23:24:10.808702: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 23:24:10.813121: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-15 23:24:10.813306: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564b236cc9d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-15 23:24:10.813322: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f8377b2e9b0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.0993 - accuracy: 0.5370
 2000/25000 [=>............................] - ETA: 10s - loss: 7.2756 - accuracy: 0.5255
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.3293 - accuracy: 0.5220 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.4060 - accuracy: 0.5170
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.4060 - accuracy: 0.5170
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.4673 - accuracy: 0.5130
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.4936 - accuracy: 0.5113
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.5861 - accuracy: 0.5052
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6002 - accuracy: 0.5043
10000/25000 [===========>..................] - ETA: 5s - loss: 7.5915 - accuracy: 0.5049
11000/25000 [============>.................] - ETA: 4s - loss: 7.6178 - accuracy: 0.5032
12000/25000 [=============>................] - ETA: 4s - loss: 7.6398 - accuracy: 0.5017
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6501 - accuracy: 0.5011
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6699 - accuracy: 0.4998
15000/25000 [=================>............] - ETA: 3s - loss: 7.6533 - accuracy: 0.5009
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6407 - accuracy: 0.5017
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6314 - accuracy: 0.5023
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6428 - accuracy: 0.5016
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6440 - accuracy: 0.5015
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6444 - accuracy: 0.5015
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6548 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6520 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6564 - accuracy: 0.5007
25000/25000 [==============================] - 10s 411us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f8330899668> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f83482a2c88> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4091 - crf_viterbi_accuracy: 0.1600 - val_loss: 1.3603 - val_crf_viterbi_accuracy: 0.1200

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

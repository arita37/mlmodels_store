
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f13a60a8f28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-16 00:18:21.557804
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-16 00:18:21.561849
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-16 00:18:21.564993
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-16 00:18:21.568179
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f13b1e723c8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 352190.9688
Epoch 2/10

1/1 [==============================] - 0s 100ms/step - loss: 226991.2188
Epoch 3/10

1/1 [==============================] - 0s 94ms/step - loss: 113038.3359
Epoch 4/10

1/1 [==============================] - 0s 92ms/step - loss: 48617.3008
Epoch 5/10

1/1 [==============================] - 0s 90ms/step - loss: 22887.3027
Epoch 6/10

1/1 [==============================] - 0s 90ms/step - loss: 12584.1855
Epoch 7/10

1/1 [==============================] - 0s 92ms/step - loss: 7890.7041
Epoch 8/10

1/1 [==============================] - 0s 95ms/step - loss: 5462.8057
Epoch 9/10

1/1 [==============================] - 0s 97ms/step - loss: 4089.4688
Epoch 10/10

1/1 [==============================] - 0s 96ms/step - loss: 3263.7856

  #### Inference Need return ypred, ytrue ######################### 
[[ 7.29945183e-01 -7.46812284e-01 -3.10212040e+00 -5.57167172e-01
   9.38929677e-01 -1.70004261e+00 -2.73215771e-02 -1.80583620e+00
   1.82485533e+00  2.23470360e-01 -3.91599894e-01  7.63660848e-01
  -6.69861659e-02  1.77836418e-01  1.24755725e-01  1.86888790e+00
   1.15699458e+00  1.43841267e-01  2.79541612e-02 -1.59190845e+00
   5.33888340e-01 -2.31429830e-01  2.18249261e-01  1.39063513e+00
  -1.72020686e+00  6.33406162e-01  6.07082248e-01  1.05167317e+00
   5.74677110e-01 -4.00492609e-01 -1.06042564e+00  1.29910171e+00
  -3.41297179e-01 -1.41908050e+00 -1.44763839e+00  1.04676425e+00
  -1.47186840e+00 -1.61403489e+00 -1.22246945e+00 -8.48092675e-01
  -4.62330133e-02 -7.27710068e-01  8.72213185e-01  8.21519434e-01
  -6.37050986e-01  2.90442705e-01 -8.75599027e-01 -3.18561888e+00
  -1.28873014e+00 -6.17070675e-01 -1.10791671e+00  1.31344867e+00
   3.96081805e-01  6.26401067e-01 -7.07014680e-01 -2.62224913e+00
   2.30535537e-01 -2.57414389e+00 -1.04231083e+00 -1.25903681e-01
   1.46999609e+00 -7.39790559e-01 -7.14856505e-01  1.48602557e+00
  -8.37016821e-01  8.51845860e-01  1.36606634e+00  1.37162244e+00
  -1.53823495e-01 -3.32879275e-02 -1.18135512e-02 -3.80340219e-01
   4.47400481e-01 -1.47223830e+00 -1.09537995e+00 -7.30004668e-01
  -1.43607664e+00 -6.22030497e-02  3.69244367e-01 -1.00232732e+00
   2.31051707e+00 -5.28172553e-01 -1.72020411e+00 -2.94226670e+00
  -5.03775060e-01  5.96238911e-01  6.38092756e-02 -2.77406549e+00
   6.65162504e-01  1.61092311e-01  9.46897328e-01 -2.70875394e-01
   8.47092271e-01  1.45018148e+00 -4.64234501e-01 -1.19732904e+00
   2.11349249e-01  3.47206414e-01 -7.02607274e-01  2.56209284e-01
  -5.82000375e-01 -2.01238155e+00  8.97152543e-01 -5.33402741e-01
  -4.16957945e-01 -2.01407105e-01 -7.86941588e-01  2.08095178e-01
   1.24682397e-01  1.18820870e+00 -5.29471993e-01 -1.27201438e+00
   5.32508373e-01 -3.95899177e-01  1.24192089e-01  6.20599449e-01
   1.17226970e+00  8.86114836e-01 -6.36743069e-01  1.41353977e+00
   3.32718760e-01  1.25248013e+01  1.23042526e+01  1.39143581e+01
   1.35355587e+01  1.25766735e+01  1.06763268e+01  1.45965729e+01
   1.41813850e+01  1.19984064e+01  1.37732382e+01  1.27168665e+01
   1.27170525e+01  1.33515482e+01  1.28305645e+01  1.34418325e+01
   1.18553782e+01  1.45137787e+01  1.26595764e+01  1.31351004e+01
   1.19191151e+01  1.11415377e+01  1.14963732e+01  1.43942051e+01
   1.21499128e+01  1.18872452e+01  1.15290956e+01  1.26566238e+01
   1.29310465e+01  1.32429924e+01  1.35205564e+01  9.42546463e+00
   1.13597193e+01  1.34062567e+01  1.43585367e+01  1.37294941e+01
   1.17535400e+01  1.23847542e+01  1.27134018e+01  1.16055746e+01
   1.20989685e+01  1.22533131e+01  1.13134584e+01  1.15379839e+01
   1.33678865e+01  1.19595795e+01  1.23357162e+01  1.34460764e+01
   1.21409693e+01  1.22974682e+01  1.31571112e+01  1.22351637e+01
   1.05369778e+01  1.09993095e+01  1.18528957e+01  1.36896334e+01
   1.45456295e+01  1.19306965e+01  1.26247921e+01  1.19047213e+01
   1.40108097e+00  1.38799691e+00  7.66534150e-01  7.81698525e-01
   4.35777009e-01  2.85140395e+00  9.76067424e-01  5.68523943e-01
   2.94764817e-01  4.55842495e-01  8.32198262e-02  5.58505416e-01
   2.73141360e+00  7.56486297e-01  1.87596941e+00  4.14972925e+00
   2.30895996e-01  9.94768143e-01  9.17351246e-02  6.06424272e-01
   2.80452132e-01  3.21924210e+00  1.15680683e+00  3.38940501e-01
   8.78558040e-01  1.89490330e+00  4.52313542e-01  1.98292971e-01
   1.87559843e-01  2.04167366e+00  1.88659668e+00  1.98554611e+00
   3.24491501e+00  3.16115761e+00  1.75423527e+00  1.23317587e+00
   1.38963389e+00  2.53994536e+00  2.50279331e+00  1.73620248e+00
   9.82776880e-02  1.27904010e+00  1.67079163e+00  1.74924672e+00
   2.66451120e+00  2.13670683e+00  3.98667216e-01  4.53953385e-01
   4.86598313e-01  1.72289944e+00  1.33722997e+00  2.90400219e+00
   1.39189589e+00  1.67709994e+00  5.72127461e-01  5.30412436e-01
   1.66189659e+00  7.35181689e-01  3.32561445e+00  2.57916927e-01
   6.73449993e-01  4.36711311e-01  8.54762733e-01  1.05311263e+00
   2.33993435e+00  3.10801935e+00  1.30104625e+00  2.66539478e+00
   4.21403468e-01  2.13357544e+00  1.70241117e-01  2.39238501e+00
   1.93570495e-01  3.14546537e+00  1.20393515e+00  2.07897425e+00
   1.57391810e+00  3.09639096e-01  3.41434574e+00  3.63617897e-01
   7.75468111e-01  1.43187451e+00  1.97801924e+00  6.17516935e-01
   1.70150518e+00  2.65260363e+00  9.55376625e-02  9.49861348e-01
   7.76054323e-01  5.12526274e-01  2.92812967e+00  1.56567335e-01
   6.95361495e-02  1.84874070e+00  1.46424603e+00  1.83498323e-01
   3.32915831e+00  2.44448185e+00  4.35842931e-01  1.27059460e+00
   1.66694641e-01  8.97650123e-01  1.29387105e+00  2.44067240e+00
   7.44371831e-01  2.35520840e-01  1.41648769e-01  7.35186219e-01
   2.88483143e-01  2.05483723e+00  8.41437340e-01  3.97852612e+00
   1.21093988e+00  1.16051769e+00  3.14666271e-01  3.33659291e-01
   1.78764546e+00  2.61478090e+00  1.99865472e+00  1.59356022e+00
   9.32998419e-01  1.46994991e+01  1.13127127e+01  1.07975197e+01
   1.39833813e+01  1.25561209e+01  1.38680143e+01  1.14196920e+01
   1.03838072e+01  1.45387735e+01  1.13398609e+01  1.13988466e+01
   1.21599464e+01  1.10540142e+01  1.21953316e+01  1.17841387e+01
   1.24784307e+01  1.36243601e+01  1.20332537e+01  1.28154087e+01
   1.27606115e+01  1.12385540e+01  1.27596111e+01  1.08088093e+01
   1.28618908e+01  1.42170353e+01  1.27288857e+01  1.40499640e+01
   1.25196896e+01  1.30297165e+01  1.41892118e+01  1.17637796e+01
   1.34586535e+01  1.08455744e+01  1.21736603e+01  1.11638908e+01
   1.20579958e+01  1.32223940e+01  1.22487469e+01  1.13218260e+01
   1.19158764e+01  1.36304550e+01  1.11408300e+01  1.22714739e+01
   1.07993603e+01  1.43194809e+01  1.08017750e+01  1.27298346e+01
   1.22156076e+01  1.36258059e+01  1.02263918e+01  1.32299395e+01
   1.34978065e+01  1.38146687e+01  9.87993908e+00  1.26221895e+01
   1.38844652e+01  1.09917688e+01  1.23428154e+01  1.14917631e+01
  -6.96549082e+00 -9.16634178e+00  1.14342470e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-16 00:18:31.327919
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   89.4312
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-16 00:18:31.331565
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8022.64
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-16 00:18:31.334564
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   89.4553
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-16 00:18:31.337414
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -717.484
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139722024486336
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139721066344968
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139721066345472
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139721066345976
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139721066346480
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139721066346984

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f1391a15e10> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.489421
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.453905
grad_step = 000002, loss = 0.430984
grad_step = 000003, loss = 0.408119
grad_step = 000004, loss = 0.385492
grad_step = 000005, loss = 0.363390
grad_step = 000006, loss = 0.343011
grad_step = 000007, loss = 0.326537
grad_step = 000008, loss = 0.319945
grad_step = 000009, loss = 0.311055
grad_step = 000010, loss = 0.297500
grad_step = 000011, loss = 0.285593
grad_step = 000012, loss = 0.277304
grad_step = 000013, loss = 0.270555
grad_step = 000014, loss = 0.263158
grad_step = 000015, loss = 0.254327
grad_step = 000016, loss = 0.244407
grad_step = 000017, loss = 0.234598
grad_step = 000018, loss = 0.226140
grad_step = 000019, loss = 0.219174
grad_step = 000020, loss = 0.212292
grad_step = 000021, loss = 0.204240
grad_step = 000022, loss = 0.195488
grad_step = 000023, loss = 0.187329
grad_step = 000024, loss = 0.180213
grad_step = 000025, loss = 0.173656
grad_step = 000026, loss = 0.167019
grad_step = 000027, loss = 0.158981
grad_step = 000028, loss = 0.150524
grad_step = 000029, loss = 0.142452
grad_step = 000030, loss = 0.135229
grad_step = 000031, loss = 0.128545
grad_step = 000032, loss = 0.122119
grad_step = 000033, loss = 0.116198
grad_step = 000034, loss = 0.110815
grad_step = 000035, loss = 0.105700
grad_step = 000036, loss = 0.100755
grad_step = 000037, loss = 0.095786
grad_step = 000038, loss = 0.090754
grad_step = 000039, loss = 0.085915
grad_step = 000040, loss = 0.081330
grad_step = 000041, loss = 0.076885
grad_step = 000042, loss = 0.072698
grad_step = 000043, loss = 0.068728
grad_step = 000044, loss = 0.064887
grad_step = 000045, loss = 0.061302
grad_step = 000046, loss = 0.057943
grad_step = 000047, loss = 0.054673
grad_step = 000048, loss = 0.051520
grad_step = 000049, loss = 0.048455
grad_step = 000050, loss = 0.045540
grad_step = 000051, loss = 0.042853
grad_step = 000052, loss = 0.040280
grad_step = 000053, loss = 0.037816
grad_step = 000054, loss = 0.035496
grad_step = 000055, loss = 0.033304
grad_step = 000056, loss = 0.031244
grad_step = 000057, loss = 0.029277
grad_step = 000058, loss = 0.027397
grad_step = 000059, loss = 0.025635
grad_step = 000060, loss = 0.023992
grad_step = 000061, loss = 0.022446
grad_step = 000062, loss = 0.020996
grad_step = 000063, loss = 0.019616
grad_step = 000064, loss = 0.018324
grad_step = 000065, loss = 0.017123
grad_step = 000066, loss = 0.016009
grad_step = 000067, loss = 0.014970
grad_step = 000068, loss = 0.013999
grad_step = 000069, loss = 0.013109
grad_step = 000070, loss = 0.012287
grad_step = 000071, loss = 0.011515
grad_step = 000072, loss = 0.010795
grad_step = 000073, loss = 0.010124
grad_step = 000074, loss = 0.009500
grad_step = 000075, loss = 0.008919
grad_step = 000076, loss = 0.008376
grad_step = 000077, loss = 0.007877
grad_step = 000078, loss = 0.007417
grad_step = 000079, loss = 0.006990
grad_step = 000080, loss = 0.006592
grad_step = 000081, loss = 0.006221
grad_step = 000082, loss = 0.005880
grad_step = 000083, loss = 0.005563
grad_step = 000084, loss = 0.005269
grad_step = 000085, loss = 0.004998
grad_step = 000086, loss = 0.004751
grad_step = 000087, loss = 0.004525
grad_step = 000088, loss = 0.004315
grad_step = 000089, loss = 0.004121
grad_step = 000090, loss = 0.003941
grad_step = 000091, loss = 0.003776
grad_step = 000092, loss = 0.003623
grad_step = 000093, loss = 0.003484
grad_step = 000094, loss = 0.003358
grad_step = 000095, loss = 0.003241
grad_step = 000096, loss = 0.003134
grad_step = 000097, loss = 0.003035
grad_step = 000098, loss = 0.002945
grad_step = 000099, loss = 0.002862
grad_step = 000100, loss = 0.002787
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002719
grad_step = 000102, loss = 0.002657
grad_step = 000103, loss = 0.002601
grad_step = 000104, loss = 0.002549
grad_step = 000105, loss = 0.002503
grad_step = 000106, loss = 0.002461
grad_step = 000107, loss = 0.002422
grad_step = 000108, loss = 0.002387
grad_step = 000109, loss = 0.002356
grad_step = 000110, loss = 0.002327
grad_step = 000111, loss = 0.002301
grad_step = 000112, loss = 0.002278
grad_step = 000113, loss = 0.002256
grad_step = 000114, loss = 0.002236
grad_step = 000115, loss = 0.002219
grad_step = 000116, loss = 0.002202
grad_step = 000117, loss = 0.002187
grad_step = 000118, loss = 0.002173
grad_step = 000119, loss = 0.002160
grad_step = 000120, loss = 0.002148
grad_step = 000121, loss = 0.002137
grad_step = 000122, loss = 0.002126
grad_step = 000123, loss = 0.002117
grad_step = 000124, loss = 0.002107
grad_step = 000125, loss = 0.002098
grad_step = 000126, loss = 0.002090
grad_step = 000127, loss = 0.002082
grad_step = 000128, loss = 0.002074
grad_step = 000129, loss = 0.002066
grad_step = 000130, loss = 0.002060
grad_step = 000131, loss = 0.002054
grad_step = 000132, loss = 0.002053
grad_step = 000133, loss = 0.002060
grad_step = 000134, loss = 0.002082
grad_step = 000135, loss = 0.002090
grad_step = 000136, loss = 0.002064
grad_step = 000137, loss = 0.002014
grad_step = 000138, loss = 0.002014
grad_step = 000139, loss = 0.002039
grad_step = 000140, loss = 0.002018
grad_step = 000141, loss = 0.001984
grad_step = 000142, loss = 0.001985
grad_step = 000143, loss = 0.001995
grad_step = 000144, loss = 0.001977
grad_step = 000145, loss = 0.001953
grad_step = 000146, loss = 0.001956
grad_step = 000147, loss = 0.001959
grad_step = 000148, loss = 0.001937
grad_step = 000149, loss = 0.001922
grad_step = 000150, loss = 0.001923
grad_step = 000151, loss = 0.001917
grad_step = 000152, loss = 0.001900
grad_step = 000153, loss = 0.001887
grad_step = 000154, loss = 0.001885
grad_step = 000155, loss = 0.001878
grad_step = 000156, loss = 0.001861
grad_step = 000157, loss = 0.001849
grad_step = 000158, loss = 0.001844
grad_step = 000159, loss = 0.001836
grad_step = 000160, loss = 0.001823
grad_step = 000161, loss = 0.001812
grad_step = 000162, loss = 0.001809
grad_step = 000163, loss = 0.001811
grad_step = 000164, loss = 0.001813
grad_step = 000165, loss = 0.001796
grad_step = 000166, loss = 0.001784
grad_step = 000167, loss = 0.001776
grad_step = 000168, loss = 0.001769
grad_step = 000169, loss = 0.001752
grad_step = 000170, loss = 0.001742
grad_step = 000171, loss = 0.001746
grad_step = 000172, loss = 0.001750
grad_step = 000173, loss = 0.001768
grad_step = 000174, loss = 0.001813
grad_step = 000175, loss = 0.001900
grad_step = 000176, loss = 0.001782
grad_step = 000177, loss = 0.001766
grad_step = 000178, loss = 0.001751
grad_step = 000179, loss = 0.001752
grad_step = 000180, loss = 0.001789
grad_step = 000181, loss = 0.001710
grad_step = 000182, loss = 0.001715
grad_step = 000183, loss = 0.001736
grad_step = 000184, loss = 0.001709
grad_step = 000185, loss = 0.001712
grad_step = 000186, loss = 0.001686
grad_step = 000187, loss = 0.001693
grad_step = 000188, loss = 0.001705
grad_step = 000189, loss = 0.001677
grad_step = 000190, loss = 0.001676
grad_step = 000191, loss = 0.001670
grad_step = 000192, loss = 0.001674
grad_step = 000193, loss = 0.001662
grad_step = 000194, loss = 0.001646
grad_step = 000195, loss = 0.001653
grad_step = 000196, loss = 0.001645
grad_step = 000197, loss = 0.001645
grad_step = 000198, loss = 0.001645
grad_step = 000199, loss = 0.001625
grad_step = 000200, loss = 0.001632
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001631
grad_step = 000202, loss = 0.001625
grad_step = 000203, loss = 0.001630
grad_step = 000204, loss = 0.001612
grad_step = 000205, loss = 0.001609
grad_step = 000206, loss = 0.001612
grad_step = 000207, loss = 0.001601
grad_step = 000208, loss = 0.001607
grad_step = 000209, loss = 0.001606
grad_step = 000210, loss = 0.001600
grad_step = 000211, loss = 0.001605
grad_step = 000212, loss = 0.001604
grad_step = 000213, loss = 0.001600
grad_step = 000214, loss = 0.001607
grad_step = 000215, loss = 0.001602
grad_step = 000216, loss = 0.001600
grad_step = 000217, loss = 0.001596
grad_step = 000218, loss = 0.001588
grad_step = 000219, loss = 0.001580
grad_step = 000220, loss = 0.001574
grad_step = 000221, loss = 0.001569
grad_step = 000222, loss = 0.001568
grad_step = 000223, loss = 0.001568
grad_step = 000224, loss = 0.001568
grad_step = 000225, loss = 0.001571
grad_step = 000226, loss = 0.001570
grad_step = 000227, loss = 0.001570
grad_step = 000228, loss = 0.001570
grad_step = 000229, loss = 0.001568
grad_step = 000230, loss = 0.001564
grad_step = 000231, loss = 0.001562
grad_step = 000232, loss = 0.001556
grad_step = 000233, loss = 0.001552
grad_step = 000234, loss = 0.001547
grad_step = 000235, loss = 0.001543
grad_step = 000236, loss = 0.001540
grad_step = 000237, loss = 0.001537
grad_step = 000238, loss = 0.001534
grad_step = 000239, loss = 0.001533
grad_step = 000240, loss = 0.001531
grad_step = 000241, loss = 0.001529
grad_step = 000242, loss = 0.001527
grad_step = 000243, loss = 0.001527
grad_step = 000244, loss = 0.001528
grad_step = 000245, loss = 0.001532
grad_step = 000246, loss = 0.001546
grad_step = 000247, loss = 0.001577
grad_step = 000248, loss = 0.001654
grad_step = 000249, loss = 0.001720
grad_step = 000250, loss = 0.001798
grad_step = 000251, loss = 0.001654
grad_step = 000252, loss = 0.001539
grad_step = 000253, loss = 0.001551
grad_step = 000254, loss = 0.001609
grad_step = 000255, loss = 0.001609
grad_step = 000256, loss = 0.001533
grad_step = 000257, loss = 0.001533
grad_step = 000258, loss = 0.001580
grad_step = 000259, loss = 0.001551
grad_step = 000260, loss = 0.001512
grad_step = 000261, loss = 0.001527
grad_step = 000262, loss = 0.001543
grad_step = 000263, loss = 0.001514
grad_step = 000264, loss = 0.001499
grad_step = 000265, loss = 0.001524
grad_step = 000266, loss = 0.001516
grad_step = 000267, loss = 0.001487
grad_step = 000268, loss = 0.001502
grad_step = 000269, loss = 0.001511
grad_step = 000270, loss = 0.001485
grad_step = 000271, loss = 0.001484
grad_step = 000272, loss = 0.001496
grad_step = 000273, loss = 0.001484
grad_step = 000274, loss = 0.001478
grad_step = 000275, loss = 0.001479
grad_step = 000276, loss = 0.001473
grad_step = 000277, loss = 0.001475
grad_step = 000278, loss = 0.001474
grad_step = 000279, loss = 0.001463
grad_step = 000280, loss = 0.001462
grad_step = 000281, loss = 0.001466
grad_step = 000282, loss = 0.001461
grad_step = 000283, loss = 0.001456
grad_step = 000284, loss = 0.001455
grad_step = 000285, loss = 0.001451
grad_step = 000286, loss = 0.001449
grad_step = 000287, loss = 0.001451
grad_step = 000288, loss = 0.001448
grad_step = 000289, loss = 0.001444
grad_step = 000290, loss = 0.001441
grad_step = 000291, loss = 0.001439
grad_step = 000292, loss = 0.001436
grad_step = 000293, loss = 0.001434
grad_step = 000294, loss = 0.001434
grad_step = 000295, loss = 0.001432
grad_step = 000296, loss = 0.001430
grad_step = 000297, loss = 0.001430
grad_step = 000298, loss = 0.001428
grad_step = 000299, loss = 0.001426
grad_step = 000300, loss = 0.001424
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001424
grad_step = 000302, loss = 0.001423
grad_step = 000303, loss = 0.001422
grad_step = 000304, loss = 0.001424
grad_step = 000305, loss = 0.001427
grad_step = 000306, loss = 0.001433
grad_step = 000307, loss = 0.001443
grad_step = 000308, loss = 0.001463
grad_step = 000309, loss = 0.001480
grad_step = 000310, loss = 0.001508
grad_step = 000311, loss = 0.001502
grad_step = 000312, loss = 0.001492
grad_step = 000313, loss = 0.001440
grad_step = 000314, loss = 0.001403
grad_step = 000315, loss = 0.001401
grad_step = 000316, loss = 0.001423
grad_step = 000317, loss = 0.001443
grad_step = 000318, loss = 0.001427
grad_step = 000319, loss = 0.001402
grad_step = 000320, loss = 0.001388
grad_step = 000321, loss = 0.001394
grad_step = 000322, loss = 0.001409
grad_step = 000323, loss = 0.001411
grad_step = 000324, loss = 0.001403
grad_step = 000325, loss = 0.001386
grad_step = 000326, loss = 0.001377
grad_step = 000327, loss = 0.001378
grad_step = 000328, loss = 0.001384
grad_step = 000329, loss = 0.001391
grad_step = 000330, loss = 0.001392
grad_step = 000331, loss = 0.001390
grad_step = 000332, loss = 0.001383
grad_step = 000333, loss = 0.001375
grad_step = 000334, loss = 0.001368
grad_step = 000335, loss = 0.001362
grad_step = 000336, loss = 0.001360
grad_step = 000337, loss = 0.001360
grad_step = 000338, loss = 0.001361
grad_step = 000339, loss = 0.001362
grad_step = 000340, loss = 0.001366
grad_step = 000341, loss = 0.001369
grad_step = 000342, loss = 0.001375
grad_step = 000343, loss = 0.001379
grad_step = 000344, loss = 0.001388
grad_step = 000345, loss = 0.001392
grad_step = 000346, loss = 0.001400
grad_step = 000347, loss = 0.001393
grad_step = 000348, loss = 0.001386
grad_step = 000349, loss = 0.001366
grad_step = 000350, loss = 0.001349
grad_step = 000351, loss = 0.001337
grad_step = 000352, loss = 0.001334
grad_step = 000353, loss = 0.001338
grad_step = 000354, loss = 0.001345
grad_step = 000355, loss = 0.001350
grad_step = 000356, loss = 0.001350
grad_step = 000357, loss = 0.001347
grad_step = 000358, loss = 0.001338
grad_step = 000359, loss = 0.001330
grad_step = 000360, loss = 0.001323
grad_step = 000361, loss = 0.001318
grad_step = 000362, loss = 0.001316
grad_step = 000363, loss = 0.001316
grad_step = 000364, loss = 0.001318
grad_step = 000365, loss = 0.001321
grad_step = 000366, loss = 0.001328
grad_step = 000367, loss = 0.001338
grad_step = 000368, loss = 0.001358
grad_step = 000369, loss = 0.001378
grad_step = 000370, loss = 0.001419
grad_step = 000371, loss = 0.001421
grad_step = 000372, loss = 0.001424
grad_step = 000373, loss = 0.001366
grad_step = 000374, loss = 0.001316
grad_step = 000375, loss = 0.001295
grad_step = 000376, loss = 0.001312
grad_step = 000377, loss = 0.001344
grad_step = 000378, loss = 0.001347
grad_step = 000379, loss = 0.001332
grad_step = 000380, loss = 0.001298
grad_step = 000381, loss = 0.001284
grad_step = 000382, loss = 0.001293
grad_step = 000383, loss = 0.001308
grad_step = 000384, loss = 0.001316
grad_step = 000385, loss = 0.001303
grad_step = 000386, loss = 0.001286
grad_step = 000387, loss = 0.001273
grad_step = 000388, loss = 0.001272
grad_step = 000389, loss = 0.001279
grad_step = 000390, loss = 0.001286
grad_step = 000391, loss = 0.001291
grad_step = 000392, loss = 0.001286
grad_step = 000393, loss = 0.001279
grad_step = 000394, loss = 0.001268
grad_step = 000395, loss = 0.001260
grad_step = 000396, loss = 0.001255
grad_step = 000397, loss = 0.001254
grad_step = 000398, loss = 0.001255
grad_step = 000399, loss = 0.001258
grad_step = 000400, loss = 0.001260
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001261
grad_step = 000402, loss = 0.001261
grad_step = 000403, loss = 0.001259
grad_step = 000404, loss = 0.001257
grad_step = 000405, loss = 0.001253
grad_step = 000406, loss = 0.001249
grad_step = 000407, loss = 0.001244
grad_step = 000408, loss = 0.001239
grad_step = 000409, loss = 0.001235
grad_step = 000410, loss = 0.001231
grad_step = 000411, loss = 0.001227
grad_step = 000412, loss = 0.001224
grad_step = 000413, loss = 0.001222
grad_step = 000414, loss = 0.001219
grad_step = 000415, loss = 0.001217
grad_step = 000416, loss = 0.001215
grad_step = 000417, loss = 0.001213
grad_step = 000418, loss = 0.001211
grad_step = 000419, loss = 0.001210
grad_step = 000420, loss = 0.001209
grad_step = 000421, loss = 0.001211
grad_step = 000422, loss = 0.001217
grad_step = 000423, loss = 0.001238
grad_step = 000424, loss = 0.001282
grad_step = 000425, loss = 0.001406
grad_step = 000426, loss = 0.001493
grad_step = 000427, loss = 0.001653
grad_step = 000428, loss = 0.001410
grad_step = 000429, loss = 0.001229
grad_step = 000430, loss = 0.001239
grad_step = 000431, loss = 0.001343
grad_step = 000432, loss = 0.001370
grad_step = 000433, loss = 0.001232
grad_step = 000434, loss = 0.001233
grad_step = 000435, loss = 0.001338
grad_step = 000436, loss = 0.001275
grad_step = 000437, loss = 0.001194
grad_step = 000438, loss = 0.001217
grad_step = 000439, loss = 0.001254
grad_step = 000440, loss = 0.001222
grad_step = 000441, loss = 0.001174
grad_step = 000442, loss = 0.001197
grad_step = 000443, loss = 0.001234
grad_step = 000444, loss = 0.001207
grad_step = 000445, loss = 0.001174
grad_step = 000446, loss = 0.001167
grad_step = 000447, loss = 0.001180
grad_step = 000448, loss = 0.001196
grad_step = 000449, loss = 0.001186
grad_step = 000450, loss = 0.001160
grad_step = 000451, loss = 0.001148
grad_step = 000452, loss = 0.001158
grad_step = 000453, loss = 0.001166
grad_step = 000454, loss = 0.001161
grad_step = 000455, loss = 0.001153
grad_step = 000456, loss = 0.001140
grad_step = 000457, loss = 0.001137
grad_step = 000458, loss = 0.001141
grad_step = 000459, loss = 0.001144
grad_step = 000460, loss = 0.001139
grad_step = 000461, loss = 0.001128
grad_step = 000462, loss = 0.001124
grad_step = 000463, loss = 0.001126
grad_step = 000464, loss = 0.001126
grad_step = 000465, loss = 0.001125
grad_step = 000466, loss = 0.001120
grad_step = 000467, loss = 0.001114
grad_step = 000468, loss = 0.001110
grad_step = 000469, loss = 0.001109
grad_step = 000470, loss = 0.001110
grad_step = 000471, loss = 0.001109
grad_step = 000472, loss = 0.001107
grad_step = 000473, loss = 0.001104
grad_step = 000474, loss = 0.001100
grad_step = 000475, loss = 0.001096
grad_step = 000476, loss = 0.001092
grad_step = 000477, loss = 0.001090
grad_step = 000478, loss = 0.001088
grad_step = 000479, loss = 0.001086
grad_step = 000480, loss = 0.001084
grad_step = 000481, loss = 0.001083
grad_step = 000482, loss = 0.001081
grad_step = 000483, loss = 0.001080
grad_step = 000484, loss = 0.001079
grad_step = 000485, loss = 0.001078
grad_step = 000486, loss = 0.001077
grad_step = 000487, loss = 0.001077
grad_step = 000488, loss = 0.001078
grad_step = 000489, loss = 0.001082
grad_step = 000490, loss = 0.001086
grad_step = 000491, loss = 0.001097
grad_step = 000492, loss = 0.001103
grad_step = 000493, loss = 0.001116
grad_step = 000494, loss = 0.001113
grad_step = 000495, loss = 0.001113
grad_step = 000496, loss = 0.001090
grad_step = 000497, loss = 0.001070
grad_step = 000498, loss = 0.001051
grad_step = 000499, loss = 0.001041
grad_step = 000500, loss = 0.001043
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001050
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

  date_run                              2020-05-16 00:18:53.254670
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.264996
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-16 00:18:53.260034
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.192495
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-16 00:18:53.266675
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.130561
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-16 00:18:53.271251
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.92503
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
0   2020-05-16 00:18:21.557804  ...    mean_absolute_error
1   2020-05-16 00:18:21.561849  ...     mean_squared_error
2   2020-05-16 00:18:21.564993  ...  median_absolute_error
3   2020-05-16 00:18:21.568179  ...               r2_score
4   2020-05-16 00:18:31.327919  ...    mean_absolute_error
5   2020-05-16 00:18:31.331565  ...     mean_squared_error
6   2020-05-16 00:18:31.334564  ...  median_absolute_error
7   2020-05-16 00:18:31.337414  ...               r2_score
8   2020-05-16 00:18:53.254670  ...    mean_absolute_error
9   2020-05-16 00:18:53.260034  ...     mean_squared_error
10  2020-05-16 00:18:53.266675  ...  median_absolute_error
11  2020-05-16 00:18:53.271251  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff3d896bfd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███       | 3080192/9912422 [00:00<00:00, 30593629.97it/s]9920512it [00:00, 30980832.20it/s]                             
0it [00:00, ?it/s]32768it [00:00, 716081.49it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 479279.94it/s]1654784it [00:00, 11879130.60it/s]                         
0it [00:00, ?it/s]8192it [00:00, 220291.45it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff38b36de80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff38a99c0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff38b36de80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff38a8f30f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff38812e4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff388119748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff38b36de80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff38a8b1710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff38812e4e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff3d8976ef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f1a7456c1d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=b76b83cd5a2ede04297803474d9520c39ab358cd357ff411fb0a2a37334d550a
  Stored in directory: /tmp/pip-ephem-wheel-cache-6139ynhn/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1a0c367748> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2301952/17464789 [==>...........................] - ETA: 0s
10584064/17464789 [=================>............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-16 00:20:17.852454: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 00:20:17.856778: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-16 00:20:17.856946: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5570266f5900 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 00:20:17.856961: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4060 - accuracy: 0.5170
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5516 - accuracy: 0.5075 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5695 - accuracy: 0.5063
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5248 - accuracy: 0.5092
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6084 - accuracy: 0.5038
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5440 - accuracy: 0.5080
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5133 - accuracy: 0.5100
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5325 - accuracy: 0.5088
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5900 - accuracy: 0.5050
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5915 - accuracy: 0.5049
11000/25000 [============>.................] - ETA: 4s - loss: 7.6248 - accuracy: 0.5027
12000/25000 [=============>................] - ETA: 4s - loss: 7.6206 - accuracy: 0.5030
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6100 - accuracy: 0.5037
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6097 - accuracy: 0.5037
15000/25000 [=================>............] - ETA: 3s - loss: 7.6104 - accuracy: 0.5037
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6331 - accuracy: 0.5022
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6071 - accuracy: 0.5039
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6308 - accuracy: 0.5023
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6126 - accuracy: 0.5035
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6222 - accuracy: 0.5029
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6316 - accuracy: 0.5023
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6569 - accuracy: 0.5006
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6606 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6609 - accuracy: 0.5004
25000/25000 [==============================] - 9s 368us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-16 00:20:33.652489
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-16 00:20:33.652489  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:30:27, 11.7kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:35:29, 16.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:16:07, 23.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:11:47, 33.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:01:30, 47.5kB/s].vector_cache/glove.6B.zip:   1%|          | 8.85M/862M [00:01<3:29:51, 67.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:01<2:26:26, 96.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.1M/862M [00:01<1:42:00, 138kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:01<1:11:13, 197kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.4M/862M [00:01<49:40, 281kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.3M/862M [00:01<34:43, 400kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.0M/862M [00:01<24:15, 569kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.8M/862M [00:02<17:00, 808kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.6M/862M [00:02<11:55, 1.15MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.3M/862M [00:02<08:25, 1.61MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.0M/862M [00:02<05:57, 2.27MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:02<05:42, 2.37MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.8M/862M [00:02<05:20, 2.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<05:45, 2.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.1M/862M [00:05<07:55, 1.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:05<06:20, 2.11MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.2M/862M [00:05<04:41, 2.85MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:07<06:43, 1.99MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:07<06:28, 2.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.6M/862M [00:07<04:54, 2.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:07<03:35, 3.70MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<38:35, 344kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:09<28:40, 463kB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.8M/862M [00:09<20:23, 650kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:11<16:54, 782kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:11<13:02, 1.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.1M/862M [00:11<09:24, 1.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:11<06:46, 1.94MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:13<19:24, 677kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 73.8M/862M [00:13<16:16, 807kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:13<11:56, 1.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:13<08:33, 1.53MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<10:54, 1.20MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:15<08:59, 1.45MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.6M/862M [00:15<06:37, 1.97MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:16<07:39, 1.70MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:17<08:01, 1.62MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.8M/862M [00:17<06:12, 2.09MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:17<04:30, 2.87MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<09:43, 1.33MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:19<08:08, 1.59MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.9M/862M [00:19<06:01, 2.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<07:16, 1.77MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.2M/862M [00:21<07:46, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:21<06:06, 2.11MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<06:19, 2.02MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:22<05:45, 2.22MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.1M/862M [00:23<04:17, 2.98MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<05:59, 2.13MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:24<05:30, 2.31MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<04:10, 3.04MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<05:54, 2.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:50, 1.85MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<05:21, 2.36MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<04:02, 3.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:03, 2.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:35, 2.25MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:12, 2.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:46, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:36, 1.89MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<05:14, 2.38MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:31<03:46, 3.30MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<1:33:42, 133kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<1:06:52, 186kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<47:02, 264kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<35:44, 347kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<27:34, 449kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<19:55, 621kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<15:52, 776kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<12:22, 994kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<08:55, 1.38MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<09:06, 1.35MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:53, 1.38MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:50, 1.79MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<05:20, 2.28MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<8:16:53, 24.5kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<5:48:15, 35.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<4:03:05, 49.9kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<2:57:00, 68.5kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<2:06:43, 95.6kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<1:29:13, 136kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<1:02:36, 193kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<46:42, 258kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<33:57, 355kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<24:00, 501kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<19:28, 615kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<14:52, 805kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<10:39, 1.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<11:22, 1.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<19:47, 602kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<16:26, 724kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<12:13, 973kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<08:51, 1.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:37, 1.37MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<07:15, 1.63MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<05:20, 2.21MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:28, 1.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<06:56, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<05:25, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<03:54, 2.99MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<13:28, 869kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<10:39, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<07:44, 1.51MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<08:07, 1.43MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:52, 1.69MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<05:06, 2.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:18, 1.84MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:36, 2.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<04:09, 2.77MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:38, 2.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:13, 2.20MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<04:31, 2.54MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<03:45, 3.05MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<03:01, 3.79MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<02:33, 4.48MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<02:16, 5.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<21:40, 527kB/s] .vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<16:49, 679kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<12:11, 936kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<08:39, 1.31MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<16:37, 683kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<13:11, 861kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<09:32, 1.19MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<06:47, 1.66MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<52:51, 214kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<38:20, 294kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<27:07, 415kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<21:13, 529kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<17:21, 647kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<12:42, 882kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<08:59, 1.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<13:24, 832kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<10:32, 1.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<07:39, 1.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<07:55, 1.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<07:49, 1.42MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<05:56, 1.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<04:21, 2.54MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<06:34, 1.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:43, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<04:15, 2.58MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<05:31, 1.98MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<06:09, 1.78MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<04:52, 2.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<05:09, 2.11MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<04:44, 2.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<03:35, 3.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:02, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<04:38, 2.33MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<03:30, 3.07MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<05:00, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<04:34, 2.35MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<03:25, 3.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<04:56, 2.16MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<05:38, 1.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<04:26, 2.40MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<03:12, 3.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<36:52, 288kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<26:42, 397kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<18:57, 558kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<13:20, 790kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<51:45, 204kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<38:22, 275kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<27:17, 385kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<19:09, 547kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<20:30, 511kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<15:24, 679kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<11:01, 946kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<10:07, 1.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<09:13, 1.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<06:54, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<04:56, 2.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<09:52, 1.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<07:59, 1.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<05:48, 1.77MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<06:27, 1.59MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:36<05:34, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<04:09, 2.46MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<05:19, 1.92MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<04:44, 2.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<03:34, 2.84MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<04:49, 2.10MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<05:27, 1.86MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<04:19, 2.34MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<04:39, 2.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<04:19, 2.32MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<03:15, 3.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<04:34, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:44<04:14, 2.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<03:12, 3.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<04:35, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:46<04:13, 2.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<03:09, 3.12MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<04:33, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<05:12, 1.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<04:05, 2.41MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<02:57, 3.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<10:48, 905kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:50<08:34, 1.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<06:14, 1.56MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<06:37, 1.46MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:52<05:38, 1.72MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<04:09, 2.33MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<05:12, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:54<04:37, 2.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<03:26, 2.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:40, 2.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:14, 1.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:56<04:09, 2.30MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:25, 2.14MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:05, 2.33MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<03:04, 3.08MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:21, 2.17MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:01, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<03:00, 3.12MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:18, 2.17MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<03:58, 2.36MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<03:00, 3.10MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<04:17, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<03:57, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<03:00, 3.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<04:16, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<03:56, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<02:57, 3.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<04:14, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<03:55, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<02:58, 3.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<04:13, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<03:44, 2.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<02:50, 3.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:07, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<04:45, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<03:43, 2.42MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<02:43, 3.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<05:49, 1.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<05:00, 1.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<03:41, 2.42MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<04:39, 1.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<04:10, 2.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<03:08, 2.81MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<04:30, 1.96MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<04:58, 1.77MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<03:55, 2.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<04:09, 2.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<03:49, 2.28MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<02:52, 3.03MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<04:00, 2.17MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<04:37, 1.88MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<03:37, 2.40MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<02:40, 3.23MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<04:46, 1.81MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<04:05, 2.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<03:02, 2.82MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<02:14, 3.80MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:25<17:40, 484kB/s] .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<13:15, 644kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<09:26, 902kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<08:33, 990kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<06:51, 1.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<05:00, 1.69MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:29<05:28, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<04:43, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:30, 2.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<04:25, 1.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<03:56, 2.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<02:57, 2.81MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<04:01, 2.06MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<03:39, 2.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<02:45, 2.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<03:52, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<03:32, 2.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<02:41, 3.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<03:47, 2.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<03:29, 2.33MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<02:38, 3.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<03:44, 2.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:39<03:20, 2.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<02:32, 3.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<01:52, 4.28MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<33:43, 237kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<24:25, 327kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<17:12, 463kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<13:51, 572kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<10:22, 763kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:43<07:31, 1.05MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<05:19, 1.48MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<33:19, 236kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<24:07, 326kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<17:01, 460kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<13:42, 568kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<10:23, 749kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<07:26, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<07:00, 1.10MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<05:42, 1.35MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<04:10, 1.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:43, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<04:05, 1.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:02, 2.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<03:55, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<03:31, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<02:39, 2.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:34, 2.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<04:04, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<03:11, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:18, 3.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<13:25, 555kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:57<10:09, 732kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<07:17, 1.02MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<06:47, 1.09MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<06:15, 1.18MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:59<04:45, 1.55MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<04:29, 1.63MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:53, 1.88MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<02:54, 2.50MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:42, 1.95MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:21, 2.15MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<02:29, 2.89MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:25, 2.09MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:00, 2.38MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<02:21, 3.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<01:43, 4.13MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<28:02, 253kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<20:21, 349kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<14:22, 492kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<11:40, 602kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<08:53, 791kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<06:20, 1.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<06:04, 1.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<05:40, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<04:16, 1.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:03, 2.26MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<07:42, 894kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<06:05, 1.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:26, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<04:41, 1.46MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 453M/862M [03:14<04:40, 1.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<03:34, 1.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<02:35, 2.61MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<04:36, 1.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:54, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:54, 2.32MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<03:35, 1.87MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<03:53, 1.72MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:03, 2.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<03:11, 2.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<02:49, 2.34MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<02:07, 3.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<01:34, 4.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<17:52, 367kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<13:10, 497kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<09:21, 697kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<08:01, 809kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<06:16, 1.03MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:32, 1.42MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<04:40, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<03:55, 1.63MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<02:54, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<03:31, 1.80MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<03:06, 2.04MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<02:20, 2.71MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:30<03:06, 2.02MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<02:50, 2.21MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<02:07, 2.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<02:57, 2.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<02:42, 2.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<02:02, 3.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<02:52, 2.14MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<02:38, 2.32MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<02:00, 3.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<02:50, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<02:35, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<01:57, 3.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<02:47, 2.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:38<02:33, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<01:56, 3.08MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<02:45, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<02:32, 2.33MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<01:55, 3.07MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<02:43, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<02:30, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<01:52, 3.12MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:44<02:41, 2.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<02:28, 2.33MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<01:52, 3.07MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:39, 2.15MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:46<03:02, 1.88MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<02:22, 2.41MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<01:43, 3.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:41, 1.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<03:10, 1.78MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<02:21, 2.39MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<02:56, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<02:37, 2.12MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<01:58, 2.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:40, 2.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:26, 2.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<01:49, 3.02MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:33, 2.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:52, 1.90MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<02:14, 2.42MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<01:39, 3.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<03:07, 1.72MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:45, 1.95MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:03, 2.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:40, 1.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:25, 2.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<01:49, 2.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:30, 2.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<02:18, 2.27MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<01:43, 3.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:24, 2.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:46, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:10, 2.37MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<01:34, 3.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<05:06, 1.00MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<04:06, 1.24MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:59, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<03:15, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:48, 1.80MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:03, 2.43MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:37, 1.90MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:19, 2.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<01:43, 2.86MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:22, 2.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<02:05, 2.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<01:34, 3.09MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:14, 2.15MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<02:34, 1.88MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:00, 2.41MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:28, 3.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:53, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:30, 1.90MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<01:52, 2.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<02:24, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<02:05, 2.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<01:32, 3.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:08, 4.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<18:15, 254kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<13:14, 350kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<09:19, 493kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<07:34, 603kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<05:45, 792kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<04:07, 1.10MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<03:55, 1.14MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:21<03:12, 1.40MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:21, 1.90MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<02:40, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:23<02:20, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<01:44, 2.53MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:14, 1.95MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:25<02:01, 2.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<01:30, 2.86MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:03, 2.08MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<02:19, 1.84MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<01:48, 2.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:18, 3.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<03:06, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:29<02:31, 1.67MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<01:51, 2.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:20, 3.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<17:39, 235kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:31<12:46, 325kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<08:58, 459kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<07:11, 568kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:33<05:26, 750kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:53, 1.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:35<03:38, 1.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<02:57, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:09, 1.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:26, 1.62MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<02:02, 1.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:29, 2.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:05, 3.54MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<15:24, 252kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:39<11:09, 347kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<07:51, 490kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<06:21, 600kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:41<04:50, 787kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<03:26, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<03:16, 1.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:37, 1.43MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:54, 1.94MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:11, 1.67MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:54, 1.92MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<01:25, 2.56MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:50, 1.96MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:39, 2.17MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<01:14, 2.87MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:41, 2.08MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:32, 2.28MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<01:09, 3.00MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:37, 2.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:29, 2.32MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:07, 3.05MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<01:35, 2.14MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:26, 2.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:05, 3.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<01:32, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:25, 2.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:04, 3.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<01:30, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:43, 1.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:20, 2.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<00:59, 3.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:44, 1.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:32, 2.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:09, 2.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:31, 2.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:42, 1.83MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:20, 2.32MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<00:57, 3.21MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:02<09:12, 332kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<06:45, 451kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<04:45, 636kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<03:59, 748kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<03:24, 875kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:31, 1.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:46, 1.65MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<04:39, 627kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<03:33, 819kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:32, 1.14MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<02:25, 1.18MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:58, 1.44MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:26, 1.95MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<01:39, 1.68MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<01:44, 1.60MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:19, 2.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<00:57, 2.86MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<02:35, 1.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<02:05, 1.29MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:31, 1.76MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<01:40, 1.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:26, 1.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:04, 2.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<01:20, 1.91MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<01:12, 2.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<00:53, 2.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:12, 2.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:22, 1.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:04, 2.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<01:08, 2.15MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<01:02, 2.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<00:47, 3.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:22<01:05, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<01:00, 2.34MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<00:45, 3.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<01:04, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<00:58, 2.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<00:44, 3.07MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<01:02, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 729M/862M [05:26<00:57, 2.34MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<00:42, 3.11MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<00:59, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:28<01:08, 1.89MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<00:54, 2.38MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<00:57, 2.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:30<00:53, 2.35MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:39, 3.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:55, 2.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<00:51, 2.35MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<00:38, 3.09MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:54, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<00:50, 2.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:37, 3.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:52, 2.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<00:46, 2.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<00:35, 3.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:25, 4.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<07:11, 253kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<05:23, 337kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<03:50, 470kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<02:53, 605kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<02:12, 792kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:33, 1.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:27, 1.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:22, 1.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:01, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:44, 2.23MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<01:01, 1.58MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:52, 1.83MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:38, 2.44MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:48, 1.92MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:52, 1.76MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:41, 2.22MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:42, 2.09MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:38, 2.29MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:28, 3.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:39, 2.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:36, 2.31MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:26, 3.08MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:19, 4.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<03:25, 393kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<02:31, 530kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:45, 743kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:30, 849kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:10, 1.08MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:50, 1.48MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:51, 1.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:50, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:38, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:27, 2.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:40, 1.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:35, 1.91MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:25, 2.55MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:32, 1.97MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:29, 2.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:21, 2.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:28, 2.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:26, 2.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:19, 3.04MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:03<00:26, 2.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:23, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:17, 3.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:24, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:21, 2.34MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:16, 3.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:07<00:22, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:20, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:14, 3.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:09<00:20, 2.16MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:09<00:18, 2.34MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:13, 3.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:18, 2.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:11<00:16, 2.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:12, 3.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:16, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:14, 2.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:10, 3.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:15<00:14, 2.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:15<00:13, 2.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:09, 3.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:17<00:12, 2.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<00:11, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:08, 3.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<00:10, 2.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:09, 2.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:06, 3.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:07, 2.34MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:05, 3.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:06, 2.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:06, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:04, 3.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:04, 2.33MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:02, 3.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:02, 2.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:02, 2.72MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 3.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:05, 408kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:03, 550kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 770kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 861/400000 [00:00<00:46, 8608.39it/s]  0%|          | 1678/400000 [00:00<00:47, 8471.09it/s]  1%|          | 2556/400000 [00:00<00:46, 8559.12it/s]  1%|          | 3440/400000 [00:00<00:45, 8640.70it/s]  1%|          | 4329/400000 [00:00<00:45, 8713.46it/s]  1%|▏         | 5194/400000 [00:00<00:45, 8693.95it/s]  2%|▏         | 6012/400000 [00:00<00:46, 8531.46it/s]  2%|▏         | 6842/400000 [00:00<00:46, 8459.31it/s]  2%|▏         | 7643/400000 [00:00<00:47, 8318.49it/s]  2%|▏         | 8442/400000 [00:01<00:47, 8165.73it/s]  2%|▏         | 9236/400000 [00:01<00:48, 8088.33it/s]  3%|▎         | 10029/400000 [00:01<00:48, 8016.16it/s]  3%|▎         | 10866/400000 [00:01<00:47, 8117.79it/s]  3%|▎         | 11680/400000 [00:01<00:47, 8122.16it/s]  3%|▎         | 12489/400000 [00:01<00:47, 8112.47it/s]  3%|▎         | 13316/400000 [00:01<00:47, 8157.69it/s]  4%|▎         | 14130/400000 [00:01<00:47, 8083.55it/s]  4%|▎         | 14963/400000 [00:01<00:47, 8155.29it/s]  4%|▍         | 15823/400000 [00:01<00:46, 8281.47it/s]  4%|▍         | 16665/400000 [00:02<00:46, 8320.41it/s]  4%|▍         | 17499/400000 [00:02<00:45, 8323.10it/s]  5%|▍         | 18332/400000 [00:02<00:45, 8319.40it/s]  5%|▍         | 19172/400000 [00:02<00:45, 8341.41it/s]  5%|▌         | 20078/400000 [00:02<00:44, 8543.71it/s]  5%|▌         | 20934/400000 [00:02<00:44, 8532.92it/s]  5%|▌         | 21789/400000 [00:02<00:44, 8507.33it/s]  6%|▌         | 22641/400000 [00:02<00:44, 8406.85it/s]  6%|▌         | 23483/400000 [00:02<00:45, 8331.19it/s]  6%|▌         | 24355/400000 [00:02<00:44, 8442.00it/s]  6%|▋         | 25260/400000 [00:03<00:43, 8613.07it/s]  7%|▋         | 26124/400000 [00:03<00:43, 8619.66it/s]  7%|▋         | 26987/400000 [00:03<00:43, 8595.04it/s]  7%|▋         | 27884/400000 [00:03<00:42, 8701.44it/s]  7%|▋         | 28775/400000 [00:03<00:42, 8761.15it/s]  7%|▋         | 29652/400000 [00:03<00:42, 8754.93it/s]  8%|▊         | 30548/400000 [00:03<00:41, 8814.58it/s]  8%|▊         | 31430/400000 [00:03<00:42, 8638.76it/s]  8%|▊         | 32334/400000 [00:03<00:42, 8753.11it/s]  8%|▊         | 33216/400000 [00:03<00:41, 8769.18it/s]  9%|▊         | 34104/400000 [00:04<00:41, 8801.56it/s]  9%|▊         | 34985/400000 [00:04<00:41, 8750.06it/s]  9%|▉         | 35861/400000 [00:04<00:42, 8667.96it/s]  9%|▉         | 36752/400000 [00:04<00:41, 8737.30it/s]  9%|▉         | 37631/400000 [00:04<00:41, 8752.48it/s] 10%|▉         | 38507/400000 [00:04<00:41, 8739.98it/s] 10%|▉         | 39396/400000 [00:04<00:41, 8782.74it/s] 10%|█         | 40275/400000 [00:04<00:42, 8544.13it/s] 10%|█         | 41131/400000 [00:04<00:42, 8459.40it/s] 10%|█         | 41992/400000 [00:04<00:42, 8501.78it/s] 11%|█         | 42872/400000 [00:05<00:41, 8586.23it/s] 11%|█         | 43732/400000 [00:05<00:41, 8581.06it/s] 11%|█         | 44591/400000 [00:05<00:42, 8404.33it/s] 11%|█▏        | 45435/400000 [00:05<00:42, 8413.73it/s] 12%|█▏        | 46282/400000 [00:05<00:41, 8430.38it/s] 12%|█▏        | 47126/400000 [00:05<00:42, 8386.44it/s] 12%|█▏        | 47966/400000 [00:05<00:42, 8343.28it/s] 12%|█▏        | 48801/400000 [00:05<00:42, 8184.34it/s] 12%|█▏        | 49622/400000 [00:05<00:42, 8189.70it/s] 13%|█▎        | 50486/400000 [00:05<00:42, 8317.50it/s] 13%|█▎        | 51323/400000 [00:06<00:41, 8332.56it/s] 13%|█▎        | 52157/400000 [00:06<00:42, 8200.79it/s] 13%|█▎        | 52979/400000 [00:06<00:42, 8100.42it/s] 13%|█▎        | 53828/400000 [00:06<00:42, 8212.90it/s] 14%|█▎        | 54718/400000 [00:06<00:41, 8406.56it/s] 14%|█▍        | 55592/400000 [00:06<00:40, 8500.09it/s] 14%|█▍        | 56444/400000 [00:06<00:40, 8503.59it/s] 14%|█▍        | 57296/400000 [00:06<00:40, 8396.72it/s] 15%|█▍        | 58148/400000 [00:06<00:40, 8430.59it/s] 15%|█▍        | 59004/400000 [00:06<00:40, 8468.96it/s] 15%|█▍        | 59868/400000 [00:07<00:39, 8517.21it/s] 15%|█▌        | 60753/400000 [00:07<00:39, 8613.25it/s] 15%|█▌        | 61615/400000 [00:07<00:39, 8545.35it/s] 16%|█▌        | 62471/400000 [00:07<00:40, 8373.78it/s] 16%|█▌        | 63310/400000 [00:07<00:40, 8369.30it/s] 16%|█▌        | 64173/400000 [00:07<00:39, 8444.00it/s] 16%|█▋        | 65036/400000 [00:07<00:39, 8496.49it/s] 16%|█▋        | 65887/400000 [00:07<00:40, 8244.89it/s] 17%|█▋        | 66714/400000 [00:07<00:40, 8214.43it/s] 17%|█▋        | 67564/400000 [00:07<00:40, 8297.13it/s] 17%|█▋        | 68408/400000 [00:08<00:39, 8338.99it/s] 17%|█▋        | 69243/400000 [00:08<00:39, 8338.35it/s] 18%|█▊        | 70078/400000 [00:08<00:39, 8258.53it/s] 18%|█▊        | 70947/400000 [00:08<00:39, 8381.28it/s] 18%|█▊        | 71806/400000 [00:08<00:38, 8442.61it/s] 18%|█▊        | 72651/400000 [00:08<00:39, 8376.17it/s] 18%|█▊        | 73506/400000 [00:08<00:38, 8426.15it/s] 19%|█▊        | 74350/400000 [00:08<00:39, 8345.93it/s] 19%|█▉        | 75186/400000 [00:08<00:38, 8334.08it/s] 19%|█▉        | 76057/400000 [00:09<00:38, 8442.96it/s] 19%|█▉        | 76933/400000 [00:09<00:37, 8533.81it/s] 19%|█▉        | 77843/400000 [00:09<00:37, 8693.60it/s] 20%|█▉        | 78726/400000 [00:09<00:36, 8732.37it/s] 20%|█▉        | 79614/400000 [00:09<00:36, 8774.93it/s] 20%|██        | 80520/400000 [00:09<00:36, 8856.05it/s] 20%|██        | 81464/400000 [00:09<00:35, 9023.22it/s] 21%|██        | 82376/400000 [00:09<00:35, 9049.47it/s] 21%|██        | 83282/400000 [00:09<00:35, 8897.61it/s] 21%|██        | 84180/400000 [00:09<00:35, 8921.80it/s] 21%|██▏       | 85074/400000 [00:10<00:35, 8832.66it/s] 22%|██▏       | 86029/400000 [00:10<00:34, 9032.93it/s] 22%|██▏       | 86934/400000 [00:10<00:34, 8950.94it/s] 22%|██▏       | 87831/400000 [00:10<00:35, 8754.11it/s] 22%|██▏       | 88709/400000 [00:10<00:35, 8706.28it/s] 22%|██▏       | 89582/400000 [00:10<00:35, 8625.01it/s] 23%|██▎       | 90456/400000 [00:10<00:35, 8656.93it/s] 23%|██▎       | 91328/400000 [00:10<00:35, 8674.30it/s] 23%|██▎       | 92197/400000 [00:10<00:35, 8552.36it/s] 23%|██▎       | 93054/400000 [00:10<00:35, 8529.31it/s] 23%|██▎       | 93957/400000 [00:11<00:35, 8672.19it/s] 24%|██▎       | 94909/400000 [00:11<00:34, 8908.53it/s] 24%|██▍       | 95803/400000 [00:11<00:34, 8909.24it/s] 24%|██▍       | 96696/400000 [00:11<00:35, 8617.84it/s] 24%|██▍       | 97615/400000 [00:11<00:34, 8781.81it/s] 25%|██▍       | 98529/400000 [00:11<00:33, 8885.07it/s] 25%|██▍       | 99438/400000 [00:11<00:33, 8943.13it/s] 25%|██▌       | 100370/400000 [00:11<00:33, 9050.40it/s] 25%|██▌       | 101277/400000 [00:11<00:33, 8963.71it/s] 26%|██▌       | 102175/400000 [00:11<00:33, 8874.83it/s] 26%|██▌       | 103064/400000 [00:12<00:33, 8805.68it/s] 26%|██▌       | 103946/400000 [00:12<00:33, 8726.35it/s] 26%|██▌       | 104820/400000 [00:12<00:34, 8658.50it/s] 26%|██▋       | 105687/400000 [00:12<00:34, 8626.59it/s] 27%|██▋       | 106584/400000 [00:12<00:33, 8725.02it/s] 27%|██▋       | 107471/400000 [00:12<00:33, 8767.16it/s] 27%|██▋       | 108387/400000 [00:12<00:32, 8878.82it/s] 27%|██▋       | 109284/400000 [00:12<00:32, 8903.70it/s] 28%|██▊       | 110175/400000 [00:12<00:33, 8779.72it/s] 28%|██▊       | 111054/400000 [00:12<00:33, 8730.70it/s] 28%|██▊       | 111946/400000 [00:13<00:32, 8785.15it/s] 28%|██▊       | 112839/400000 [00:13<00:32, 8826.67it/s] 28%|██▊       | 113752/400000 [00:13<00:32, 8913.35it/s] 29%|██▊       | 114644/400000 [00:13<00:32, 8677.13it/s] 29%|██▉       | 115626/400000 [00:13<00:31, 8988.12it/s] 29%|██▉       | 116529/400000 [00:13<00:31, 8932.12it/s] 29%|██▉       | 117426/400000 [00:13<00:32, 8704.99it/s] 30%|██▉       | 118300/400000 [00:13<00:32, 8677.45it/s] 30%|██▉       | 119171/400000 [00:13<00:32, 8577.36it/s] 30%|███       | 120031/400000 [00:14<00:32, 8568.31it/s] 30%|███       | 120985/400000 [00:14<00:31, 8835.90it/s] 30%|███       | 121918/400000 [00:14<00:30, 8977.60it/s] 31%|███       | 122878/400000 [00:14<00:30, 9153.89it/s] 31%|███       | 123797/400000 [00:14<00:30, 9086.08it/s] 31%|███       | 124709/400000 [00:14<00:30, 9093.42it/s] 31%|███▏      | 125664/400000 [00:14<00:29, 9224.63it/s] 32%|███▏      | 126588/400000 [00:14<00:30, 9090.21it/s] 32%|███▏      | 127499/400000 [00:14<00:30, 9011.39it/s] 32%|███▏      | 128402/400000 [00:14<00:30, 8879.86it/s] 32%|███▏      | 129292/400000 [00:15<00:30, 8741.60it/s] 33%|███▎      | 130172/400000 [00:15<00:30, 8756.49it/s] 33%|███▎      | 131049/400000 [00:15<00:30, 8720.44it/s] 33%|███▎      | 131922/400000 [00:15<00:31, 8593.17it/s] 33%|███▎      | 132783/400000 [00:15<00:31, 8491.62it/s] 33%|███▎      | 133650/400000 [00:15<00:31, 8542.86it/s] 34%|███▎      | 134527/400000 [00:15<00:30, 8608.48it/s] 34%|███▍      | 135389/400000 [00:15<00:31, 8526.39it/s] 34%|███▍      | 136243/400000 [00:15<00:31, 8379.84it/s] 34%|███▍      | 137082/400000 [00:15<00:31, 8334.75it/s] 34%|███▍      | 137971/400000 [00:16<00:30, 8493.01it/s] 35%|███▍      | 138892/400000 [00:16<00:30, 8695.56it/s] 35%|███▍      | 139784/400000 [00:16<00:29, 8761.66it/s] 35%|███▌      | 140662/400000 [00:16<00:30, 8641.47it/s] 35%|███▌      | 141528/400000 [00:16<00:30, 8360.94it/s] 36%|███▌      | 142377/400000 [00:16<00:30, 8398.93it/s] 36%|███▌      | 143267/400000 [00:16<00:30, 8541.91it/s] 36%|███▌      | 144148/400000 [00:16<00:29, 8619.79it/s] 36%|███▋      | 145019/400000 [00:16<00:29, 8645.91it/s] 36%|███▋      | 145885/400000 [00:16<00:29, 8582.37it/s] 37%|███▋      | 146745/400000 [00:17<00:29, 8544.39it/s] 37%|███▋      | 147647/400000 [00:17<00:29, 8679.50it/s] 37%|███▋      | 148531/400000 [00:17<00:28, 8726.83it/s] 37%|███▋      | 149436/400000 [00:17<00:28, 8820.80it/s] 38%|███▊      | 150319/400000 [00:17<00:28, 8775.49it/s] 38%|███▊      | 151227/400000 [00:17<00:28, 8863.75it/s] 38%|███▊      | 152146/400000 [00:17<00:27, 8957.42it/s] 38%|███▊      | 153043/400000 [00:17<00:27, 8945.56it/s] 38%|███▊      | 153939/400000 [00:17<00:27, 8918.56it/s] 39%|███▊      | 154832/400000 [00:17<00:28, 8705.50it/s] 39%|███▉      | 155704/400000 [00:18<00:28, 8656.35it/s] 39%|███▉      | 156571/400000 [00:18<00:28, 8574.48it/s] 39%|███▉      | 157430/400000 [00:18<00:28, 8466.20it/s] 40%|███▉      | 158278/400000 [00:18<00:29, 8198.06it/s] 40%|███▉      | 159101/400000 [00:18<00:29, 8065.34it/s] 40%|███▉      | 159947/400000 [00:18<00:29, 8179.38it/s] 40%|████      | 160778/400000 [00:18<00:29, 8215.94it/s] 40%|████      | 161601/400000 [00:18<00:29, 8196.40it/s] 41%|████      | 162424/400000 [00:18<00:28, 8203.94it/s] 41%|████      | 163246/400000 [00:19<00:28, 8183.64it/s] 41%|████      | 164092/400000 [00:19<00:28, 8262.24it/s] 41%|████      | 164942/400000 [00:19<00:28, 8330.74it/s] 41%|████▏     | 165807/400000 [00:19<00:27, 8422.20it/s] 42%|████▏     | 166673/400000 [00:19<00:27, 8491.79it/s] 42%|████▏     | 167529/400000 [00:19<00:27, 8508.28it/s] 42%|████▏     | 168429/400000 [00:19<00:26, 8649.53it/s] 42%|████▏     | 169333/400000 [00:19<00:26, 8760.71it/s] 43%|████▎     | 170211/400000 [00:19<00:26, 8740.95it/s] 43%|████▎     | 171086/400000 [00:19<00:26, 8659.66it/s] 43%|████▎     | 171953/400000 [00:20<00:27, 8352.71it/s] 43%|████▎     | 172792/400000 [00:20<00:27, 8285.09it/s] 43%|████▎     | 173672/400000 [00:20<00:26, 8432.97it/s] 44%|████▎     | 174572/400000 [00:20<00:26, 8594.55it/s] 44%|████▍     | 175475/400000 [00:20<00:25, 8718.87it/s] 44%|████▍     | 176349/400000 [00:20<00:25, 8644.52it/s] 44%|████▍     | 177256/400000 [00:20<00:25, 8766.08it/s] 45%|████▍     | 178135/400000 [00:20<00:25, 8707.86it/s] 45%|████▍     | 179007/400000 [00:20<00:25, 8558.18it/s] 45%|████▍     | 179873/400000 [00:20<00:25, 8576.28it/s] 45%|████▌     | 180732/400000 [00:21<00:25, 8472.76it/s] 45%|████▌     | 181587/400000 [00:21<00:25, 8493.89it/s] 46%|████▌     | 182438/400000 [00:21<00:25, 8497.45it/s] 46%|████▌     | 183293/400000 [00:21<00:25, 8509.72it/s] 46%|████▌     | 184145/400000 [00:21<00:25, 8510.59it/s] 46%|████▌     | 184997/400000 [00:21<00:25, 8379.16it/s] 46%|████▋     | 185836/400000 [00:21<00:26, 8152.96it/s] 47%|████▋     | 186654/400000 [00:21<00:26, 8117.06it/s] 47%|████▋     | 187493/400000 [00:21<00:25, 8194.72it/s] 47%|████▋     | 188352/400000 [00:21<00:25, 8307.42it/s] 47%|████▋     | 189242/400000 [00:22<00:24, 8475.90it/s] 48%|████▊     | 190126/400000 [00:22<00:24, 8580.02it/s] 48%|████▊     | 190986/400000 [00:22<00:25, 8262.33it/s] 48%|████▊     | 191816/400000 [00:22<00:25, 8264.20it/s] 48%|████▊     | 192657/400000 [00:22<00:24, 8305.88it/s] 48%|████▊     | 193514/400000 [00:22<00:24, 8381.44it/s] 49%|████▊     | 194407/400000 [00:22<00:24, 8537.15it/s] 49%|████▉     | 195268/400000 [00:22<00:23, 8558.29it/s] 49%|████▉     | 196126/400000 [00:22<00:24, 8256.86it/s] 49%|████▉     | 196955/400000 [00:22<00:24, 8130.13it/s] 49%|████▉     | 197798/400000 [00:23<00:24, 8216.41it/s] 50%|████▉     | 198622/400000 [00:23<00:24, 8193.05it/s] 50%|████▉     | 199489/400000 [00:23<00:24, 8329.41it/s] 50%|█████     | 200324/400000 [00:23<00:24, 8257.43it/s] 50%|█████     | 201151/400000 [00:23<00:24, 8061.09it/s] 50%|█████     | 201980/400000 [00:23<00:24, 8128.05it/s] 51%|█████     | 202842/400000 [00:23<00:23, 8267.82it/s] 51%|█████     | 203690/400000 [00:23<00:23, 8329.41it/s] 51%|█████     | 204552/400000 [00:23<00:23, 8412.41it/s] 51%|█████▏    | 205395/400000 [00:24<00:23, 8250.49it/s] 52%|█████▏    | 206222/400000 [00:24<00:23, 8109.80it/s] 52%|█████▏    | 207078/400000 [00:24<00:23, 8239.34it/s] 52%|█████▏    | 207914/400000 [00:24<00:23, 8274.14it/s] 52%|█████▏    | 208743/400000 [00:24<00:23, 8262.10it/s] 52%|█████▏    | 209605/400000 [00:24<00:22, 8366.11it/s] 53%|█████▎    | 210443/400000 [00:24<00:23, 8139.93it/s] 53%|█████▎    | 211270/400000 [00:24<00:23, 8177.11it/s] 53%|█████▎    | 212133/400000 [00:24<00:22, 8307.25it/s] 53%|█████▎    | 213015/400000 [00:24<00:22, 8454.05it/s] 53%|█████▎    | 213863/400000 [00:25<00:22, 8444.02it/s] 54%|█████▎    | 214767/400000 [00:25<00:21, 8613.06it/s] 54%|█████▍    | 215630/400000 [00:25<00:22, 8277.25it/s] 54%|█████▍    | 216480/400000 [00:25<00:22, 8341.25it/s] 54%|█████▍    | 217335/400000 [00:25<00:21, 8400.38it/s] 55%|█████▍    | 218178/400000 [00:25<00:21, 8349.18it/s] 55%|█████▍    | 219083/400000 [00:25<00:21, 8544.57it/s] 55%|█████▍    | 219955/400000 [00:25<00:20, 8595.85it/s] 55%|█████▌    | 220817/400000 [00:25<00:21, 8175.15it/s] 55%|█████▌    | 221660/400000 [00:25<00:21, 8247.90it/s] 56%|█████▌    | 222518/400000 [00:26<00:21, 8342.58it/s] 56%|█████▌    | 223434/400000 [00:26<00:20, 8570.79it/s] 56%|█████▌    | 224367/400000 [00:26<00:19, 8783.69it/s] 56%|█████▋    | 225251/400000 [00:26<00:19, 8798.04it/s] 57%|█████▋    | 226134/400000 [00:26<00:20, 8405.84it/s] 57%|█████▋    | 227024/400000 [00:26<00:20, 8547.37it/s] 57%|█████▋    | 227999/400000 [00:26<00:19, 8871.19it/s] 57%|█████▋    | 228893/400000 [00:26<00:19, 8778.46it/s] 57%|█████▋    | 229776/400000 [00:26<00:19, 8745.72it/s] 58%|█████▊    | 230654/400000 [00:27<00:19, 8480.39it/s] 58%|█████▊    | 231507/400000 [00:27<00:20, 8219.02it/s] 58%|█████▊    | 232362/400000 [00:27<00:20, 8315.20it/s] 58%|█████▊    | 233228/400000 [00:27<00:19, 8414.60it/s] 59%|█████▊    | 234125/400000 [00:27<00:19, 8571.36it/s] 59%|█████▉    | 235021/400000 [00:27<00:19, 8682.42it/s] 59%|█████▉    | 235892/400000 [00:27<00:19, 8346.98it/s] 59%|█████▉    | 236762/400000 [00:27<00:19, 8449.01it/s] 59%|█████▉    | 237634/400000 [00:27<00:19, 8527.05it/s] 60%|█████▉    | 238533/400000 [00:27<00:18, 8660.17it/s] 60%|█████▉    | 239402/400000 [00:28<00:18, 8532.08it/s] 60%|██████    | 240314/400000 [00:28<00:18, 8698.27it/s] 60%|██████    | 241197/400000 [00:28<00:18, 8736.50it/s] 61%|██████    | 242135/400000 [00:28<00:17, 8917.93it/s] 61%|██████    | 243111/400000 [00:28<00:17, 9153.79it/s] 61%|██████    | 244030/400000 [00:28<00:17, 9030.10it/s] 61%|██████    | 244936/400000 [00:28<00:17, 9022.23it/s] 61%|██████▏   | 245840/400000 [00:28<00:17, 8984.18it/s] 62%|██████▏   | 246740/400000 [00:28<00:17, 8808.19it/s] 62%|██████▏   | 247623/400000 [00:28<00:17, 8757.01it/s] 62%|██████▏   | 248500/400000 [00:29<00:17, 8579.40it/s] 62%|██████▏   | 249360/400000 [00:29<00:18, 8159.32it/s] 63%|██████▎   | 250188/400000 [00:29<00:18, 8194.34it/s] 63%|██████▎   | 251012/400000 [00:29<00:18, 8182.97it/s] 63%|██████▎   | 251913/400000 [00:29<00:17, 8410.90it/s] 63%|██████▎   | 252758/400000 [00:29<00:17, 8413.96it/s] 63%|██████▎   | 253675/400000 [00:29<00:16, 8625.23it/s] 64%|██████▎   | 254566/400000 [00:29<00:16, 8708.48it/s] 64%|██████▍   | 255440/400000 [00:29<00:16, 8700.43it/s] 64%|██████▍   | 256312/400000 [00:29<00:16, 8645.91it/s] 64%|██████▍   | 257178/400000 [00:30<00:16, 8447.64it/s] 65%|██████▍   | 258060/400000 [00:30<00:16, 8553.18it/s] 65%|██████▍   | 258938/400000 [00:30<00:16, 8617.68it/s] 65%|██████▍   | 259802/400000 [00:30<00:16, 8556.89it/s] 65%|██████▌   | 260667/400000 [00:30<00:16, 8582.02it/s] 65%|██████▌   | 261526/400000 [00:30<00:16, 8449.52it/s] 66%|██████▌   | 262372/400000 [00:30<00:16, 8411.90it/s] 66%|██████▌   | 263237/400000 [00:30<00:16, 8480.70it/s] 66%|██████▌   | 264106/400000 [00:30<00:15, 8542.41it/s] 66%|██████▌   | 264986/400000 [00:30<00:15, 8616.50it/s] 66%|██████▋   | 265849/400000 [00:31<00:15, 8538.67it/s] 67%|██████▋   | 266704/400000 [00:31<00:15, 8393.64it/s] 67%|██████▋   | 267592/400000 [00:31<00:15, 8532.81it/s] 67%|██████▋   | 268447/400000 [00:31<00:15, 8487.67it/s] 67%|██████▋   | 269323/400000 [00:31<00:15, 8565.52it/s] 68%|██████▊   | 270181/400000 [00:31<00:15, 8429.41it/s] 68%|██████▊   | 271081/400000 [00:31<00:15, 8590.56it/s] 68%|██████▊   | 271960/400000 [00:31<00:14, 8648.29it/s] 68%|██████▊   | 272839/400000 [00:31<00:14, 8687.70it/s] 68%|██████▊   | 273709/400000 [00:32<00:14, 8586.93it/s] 69%|██████▊   | 274569/400000 [00:32<00:14, 8372.53it/s] 69%|██████▉   | 275409/400000 [00:32<00:14, 8378.45it/s] 69%|██████▉   | 276275/400000 [00:32<00:14, 8458.03it/s] 69%|██████▉   | 277137/400000 [00:32<00:14, 8505.28it/s] 70%|██████▉   | 278014/400000 [00:32<00:14, 8580.77it/s] 70%|██████▉   | 278873/400000 [00:32<00:14, 8516.11it/s] 70%|██████▉   | 279757/400000 [00:32<00:13, 8610.49it/s] 70%|███████   | 280619/400000 [00:32<00:13, 8607.47it/s] 70%|███████   | 281487/400000 [00:32<00:13, 8628.30it/s] 71%|███████   | 282351/400000 [00:33<00:13, 8572.59it/s] 71%|███████   | 283209/400000 [00:33<00:13, 8470.25it/s] 71%|███████   | 284070/400000 [00:33<00:13, 8509.30it/s] 71%|███████   | 284932/400000 [00:33<00:13, 8539.55it/s] 71%|███████▏  | 285787/400000 [00:33<00:13, 8481.45it/s] 72%|███████▏  | 286648/400000 [00:33<00:13, 8517.99it/s] 72%|███████▏  | 287501/400000 [00:33<00:13, 8503.86it/s] 72%|███████▏  | 288375/400000 [00:33<00:13, 8573.22it/s] 72%|███████▏  | 289249/400000 [00:33<00:12, 8620.79it/s] 73%|███████▎  | 290186/400000 [00:33<00:12, 8831.03it/s] 73%|███████▎  | 291194/400000 [00:34<00:11, 9170.55it/s] 73%|███████▎  | 292116/400000 [00:34<00:12, 8938.18it/s] 73%|███████▎  | 293015/400000 [00:34<00:12, 8914.71it/s] 73%|███████▎  | 293910/400000 [00:34<00:12, 8828.55it/s] 74%|███████▎  | 294796/400000 [00:34<00:11, 8825.88it/s] 74%|███████▍  | 295699/400000 [00:34<00:11, 8885.77it/s] 74%|███████▍  | 296620/400000 [00:34<00:11, 8976.39it/s] 74%|███████▍  | 297519/400000 [00:34<00:11, 8880.84it/s] 75%|███████▍  | 298445/400000 [00:34<00:11, 8989.36it/s] 75%|███████▍  | 299382/400000 [00:34<00:11, 9098.08it/s] 75%|███████▌  | 300293/400000 [00:35<00:11, 9041.85it/s] 75%|███████▌  | 301198/400000 [00:35<00:10, 9010.15it/s] 76%|███████▌  | 302100/400000 [00:35<00:11, 8851.10it/s] 76%|███████▌  | 302987/400000 [00:35<00:10, 8850.87it/s] 76%|███████▌  | 303873/400000 [00:35<00:11, 8704.11it/s] 76%|███████▌  | 304769/400000 [00:35<00:10, 8776.52it/s] 76%|███████▋  | 305648/400000 [00:35<00:10, 8717.98it/s] 77%|███████▋  | 306521/400000 [00:35<00:10, 8574.82it/s] 77%|███████▋  | 307380/400000 [00:35<00:10, 8473.73it/s] 77%|███████▋  | 308229/400000 [00:35<00:10, 8346.91it/s] 77%|███████▋  | 309110/400000 [00:36<00:10, 8480.59it/s] 77%|███████▋  | 309960/400000 [00:36<00:10, 8484.80it/s] 78%|███████▊  | 310839/400000 [00:36<00:10, 8572.88it/s] 78%|███████▊  | 311698/400000 [00:36<00:10, 8485.52it/s] 78%|███████▊  | 312582/400000 [00:36<00:10, 8587.92it/s] 78%|███████▊  | 313493/400000 [00:36<00:09, 8735.49it/s] 79%|███████▊  | 314368/400000 [00:36<00:09, 8584.08it/s] 79%|███████▉  | 315228/400000 [00:36<00:10, 8460.17it/s] 79%|███████▉  | 316076/400000 [00:36<00:09, 8441.65it/s] 79%|███████▉  | 316928/400000 [00:37<00:09, 8463.76it/s] 79%|███████▉  | 317827/400000 [00:37<00:09, 8614.43it/s] 80%|███████▉  | 318690/400000 [00:37<00:09, 8503.08it/s] 80%|███████▉  | 319545/400000 [00:37<00:09, 8516.19it/s] 80%|████████  | 320427/400000 [00:37<00:09, 8604.61it/s] 80%|████████  | 321307/400000 [00:37<00:09, 8660.11it/s] 81%|████████  | 322174/400000 [00:37<00:09, 8602.82it/s] 81%|████████  | 323113/400000 [00:37<00:08, 8823.51it/s] 81%|████████  | 323998/400000 [00:37<00:08, 8742.25it/s] 81%|████████  | 324874/400000 [00:37<00:08, 8724.81it/s] 81%|████████▏ | 325757/400000 [00:38<00:08, 8754.62it/s] 82%|████████▏ | 326636/400000 [00:38<00:08, 8762.50it/s] 82%|████████▏ | 327513/400000 [00:38<00:08, 8720.18it/s] 82%|████████▏ | 328386/400000 [00:38<00:08, 8690.96it/s] 82%|████████▏ | 329257/400000 [00:38<00:08, 8696.27it/s] 83%|████████▎ | 330127/400000 [00:38<00:08, 8597.77it/s] 83%|████████▎ | 330988/400000 [00:38<00:08, 8597.49it/s] 83%|████████▎ | 331882/400000 [00:38<00:07, 8696.26it/s] 83%|████████▎ | 332773/400000 [00:38<00:07, 8757.86it/s] 83%|████████▎ | 333755/400000 [00:38<00:07, 9050.00it/s] 84%|████████▎ | 334701/400000 [00:39<00:07, 9166.27it/s] 84%|████████▍ | 335620/400000 [00:39<00:07, 8946.28it/s] 84%|████████▍ | 336518/400000 [00:39<00:07, 8826.97it/s] 84%|████████▍ | 337444/400000 [00:39<00:06, 8951.12it/s] 85%|████████▍ | 338342/400000 [00:39<00:06, 8908.83it/s] 85%|████████▍ | 339239/400000 [00:39<00:06, 8923.40it/s] 85%|████████▌ | 340138/400000 [00:39<00:06, 8940.62it/s] 85%|████████▌ | 341033/400000 [00:39<00:06, 8818.60it/s] 85%|████████▌ | 341916/400000 [00:39<00:06, 8821.54it/s] 86%|████████▌ | 342891/400000 [00:39<00:06, 9080.79it/s] 86%|████████▌ | 343846/400000 [00:40<00:06, 9214.46it/s] 86%|████████▌ | 344788/400000 [00:40<00:05, 9274.74it/s] 86%|████████▋ | 345718/400000 [00:40<00:05, 9176.05it/s] 87%|████████▋ | 346663/400000 [00:40<00:05, 9255.92it/s] 87%|████████▋ | 347630/400000 [00:40<00:05, 9375.67it/s] 87%|████████▋ | 348613/400000 [00:40<00:05, 9506.68it/s] 87%|████████▋ | 349587/400000 [00:40<00:05, 9574.33it/s] 88%|████████▊ | 350546/400000 [00:40<00:05, 9307.00it/s] 88%|████████▊ | 351480/400000 [00:40<00:05, 9186.43it/s] 88%|████████▊ | 352427/400000 [00:40<00:05, 9264.97it/s] 88%|████████▊ | 353398/400000 [00:41<00:04, 9389.57it/s] 89%|████████▊ | 354369/400000 [00:41<00:04, 9482.29it/s] 89%|████████▉ | 355319/400000 [00:41<00:04, 9463.44it/s] 89%|████████▉ | 356267/400000 [00:41<00:04, 9432.26it/s] 89%|████████▉ | 357211/400000 [00:41<00:04, 9357.65it/s] 90%|████████▉ | 358165/400000 [00:41<00:04, 9409.48it/s] 90%|████████▉ | 359121/400000 [00:41<00:04, 9452.91it/s] 90%|█████████ | 360107/400000 [00:41<00:04, 9569.74it/s] 90%|█████████ | 361065/400000 [00:41<00:04, 9534.05it/s] 91%|█████████ | 362030/400000 [00:41<00:03, 9566.83it/s] 91%|█████████ | 362988/400000 [00:42<00:03, 9567.81it/s] 91%|█████████ | 363947/400000 [00:42<00:03, 9572.41it/s] 91%|█████████ | 364905/400000 [00:42<00:03, 9440.27it/s] 91%|█████████▏| 365850/400000 [00:42<00:03, 8919.01it/s] 92%|█████████▏| 366749/400000 [00:42<00:03, 8422.05it/s] 92%|█████████▏| 367643/400000 [00:42<00:03, 8569.46it/s] 92%|█████████▏| 368508/400000 [00:42<00:03, 8543.98it/s] 92%|█████████▏| 369368/400000 [00:42<00:03, 8546.50it/s] 93%|█████████▎| 370343/400000 [00:42<00:03, 8874.86it/s] 93%|█████████▎| 371261/400000 [00:43<00:03, 8963.90it/s] 93%|█████████▎| 372255/400000 [00:43<00:03, 9233.95it/s] 93%|█████████▎| 373222/400000 [00:43<00:02, 9359.29it/s] 94%|█████████▎| 374162/400000 [00:43<00:02, 9142.27it/s] 94%|█████████▍| 375098/400000 [00:43<00:02, 9203.80it/s] 94%|█████████▍| 376056/400000 [00:43<00:02, 9310.64it/s] 94%|█████████▍| 376996/400000 [00:43<00:02, 9335.00it/s] 94%|█████████▍| 377966/400000 [00:43<00:02, 9439.30it/s] 95%|█████████▍| 378912/400000 [00:43<00:02, 9280.74it/s] 95%|█████████▍| 379842/400000 [00:43<00:02, 9209.30it/s] 95%|█████████▌| 380793/400000 [00:44<00:02, 9295.12it/s] 95%|█████████▌| 381732/400000 [00:44<00:01, 9319.93it/s] 96%|█████████▌| 382665/400000 [00:44<00:01, 9318.35it/s] 96%|█████████▌| 383598/400000 [00:44<00:01, 9215.45it/s] 96%|█████████▌| 384524/400000 [00:44<00:01, 9228.22it/s] 96%|█████████▋| 385494/400000 [00:44<00:01, 9364.86it/s] 97%|█████████▋| 386432/400000 [00:44<00:01, 9244.60it/s] 97%|█████████▋| 387378/400000 [00:44<00:01, 9306.91it/s] 97%|█████████▋| 388310/400000 [00:44<00:01, 9253.72it/s] 97%|█████████▋| 389280/400000 [00:44<00:01, 9380.83it/s] 98%|█████████▊| 390256/400000 [00:45<00:01, 9490.51it/s] 98%|█████████▊| 391206/400000 [00:45<00:00, 9468.12it/s] 98%|█████████▊| 392168/400000 [00:45<00:00, 9510.22it/s] 98%|█████████▊| 393120/400000 [00:45<00:00, 9392.29it/s] 99%|█████████▊| 394060/400000 [00:45<00:00, 9323.56it/s] 99%|█████████▊| 394993/400000 [00:45<00:00, 9183.73it/s] 99%|█████████▉| 395954/400000 [00:45<00:00, 9305.41it/s] 99%|█████████▉| 396886/400000 [00:45<00:00, 9211.80it/s] 99%|█████████▉| 397809/400000 [00:45<00:00, 9089.62it/s]100%|█████████▉| 398766/400000 [00:45<00:00, 9228.38it/s]100%|█████████▉| 399693/400000 [00:46<00:00, 9239.85it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8676.10it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f5b6b691940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011170549027738993 	 Accuracy: 50
Train Epoch: 1 	 Loss: 0.010997582438797456 	 Accuracy: 64

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
2020-05-16 00:29:23.846832: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 00:29:23.850029: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-16 00:29:23.850141: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ee511652d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 00:29:23.850154: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f5b6f087eb8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.2066 - accuracy: 0.5300
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4750 - accuracy: 0.5125 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5797 - accuracy: 0.5057
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.4558 - accuracy: 0.5138
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5194 - accuracy: 0.5096
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5823 - accuracy: 0.5055
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6031 - accuracy: 0.5041
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5804 - accuracy: 0.5056
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6036 - accuracy: 0.5041
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5762 - accuracy: 0.5059
11000/25000 [============>.................] - ETA: 3s - loss: 7.6137 - accuracy: 0.5035
12000/25000 [=============>................] - ETA: 3s - loss: 7.6091 - accuracy: 0.5038
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6430 - accuracy: 0.5015
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6622 - accuracy: 0.5003
15000/25000 [=================>............] - ETA: 2s - loss: 7.6830 - accuracy: 0.4989
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6992 - accuracy: 0.4979
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6738 - accuracy: 0.4995
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6658 - accuracy: 0.5001
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6651 - accuracy: 0.5001
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6593 - accuracy: 0.5005
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6457 - accuracy: 0.5014
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6353 - accuracy: 0.5020
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6551 - accuracy: 0.5008
25000/25000 [==============================] - 8s 335us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f5ad03dc6a0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f5b14a6d588> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 954ms/step - loss: 1.6781 - crf_viterbi_accuracy: 0.5067 - val_loss: 1.6152 - val_crf_viterbi_accuracy: 0.5067

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

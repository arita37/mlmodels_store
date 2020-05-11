
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/a690875d5616548867153f383f824cbfc6a9260e', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'a690875d5616548867153f383f824cbfc6a9260e', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/a690875d5616548867153f383f824cbfc6a9260e

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/a690875d5616548867153f383f824cbfc6a9260e

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f5e703f1fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 04:13:11.248219
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 04:13:11.251913
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 04:13:11.255053
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 04:13:11.258262
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f5e7c1b5470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354530.5000
Epoch 2/10

1/1 [==============================] - 0s 115ms/step - loss: 284049.0312
Epoch 3/10

1/1 [==============================] - 0s 110ms/step - loss: 211131.1250
Epoch 4/10

1/1 [==============================] - 0s 92ms/step - loss: 140046.2969
Epoch 5/10

1/1 [==============================] - 0s 105ms/step - loss: 85213.6250
Epoch 6/10

1/1 [==============================] - 0s 91ms/step - loss: 50250.1680
Epoch 7/10

1/1 [==============================] - 0s 88ms/step - loss: 31014.8301
Epoch 8/10

1/1 [==============================] - 0s 89ms/step - loss: 20335.8398
Epoch 9/10

1/1 [==============================] - 0s 91ms/step - loss: 14143.7891
Epoch 10/10

1/1 [==============================] - 0s 94ms/step - loss: 10375.8984

  #### Inference Need return ypred, ytrue ######################### 
[[-7.68186450e-02  1.43676639e+00  3.22584540e-01 -6.79194450e-01
   1.65238595e+00 -6.46555305e-01 -6.33006334e-01 -2.60132551e-02
   3.51451814e-01  8.38179708e-01  3.84644091e-01  1.39549565e+00
  -5.26602387e-01 -1.17480375e-01  2.83747911e-03 -1.97560027e-01
   8.71810913e-01  7.21367240e-01  5.53066581e-02  5.69816053e-01
  -1.23090529e+00 -1.56962490e+00  8.05576563e-01  1.65227246e+00
   7.96536624e-01 -6.13202214e-01  3.87923121e-01 -7.83562660e-01
  -1.00463140e+00 -2.99518406e-02  1.16788697e+00  1.21456504e-01
   3.32538575e-01 -4.14869636e-01 -1.37888980e+00  1.00088871e+00
  -4.81185347e-01  2.21683383e-02  1.42231178e+00  6.58396244e-01
  -1.01749861e+00 -4.58360016e-01 -6.65373504e-01 -2.14259028e-02
   1.47640300e+00 -1.09960771e+00  7.08649218e-01 -6.28006101e-01
  -1.30885363e-01  6.38703346e-01  2.95300007e-01  5.05314708e-01
   9.70415115e-01 -1.09560549e+00  1.21164314e-01 -2.14172244e+00
  -7.37217247e-01 -8.88788342e-01  8.77507031e-01  2.79308259e-01
   1.60090595e-01  5.80140877e+00  5.43609858e+00  7.27139521e+00
   4.07916355e+00  7.15820408e+00  6.02007294e+00  5.72878933e+00
   5.60956287e+00  5.22333193e+00  4.50624418e+00  7.10337162e+00
   6.38035107e+00  5.43541718e+00  6.17133713e+00  6.16155958e+00
   6.75965452e+00  6.71163654e+00  5.96028519e+00  6.01408291e+00
   4.81773996e+00  5.84815741e+00  4.13011169e+00  6.07956839e+00
   6.85269117e+00  6.01308823e+00  7.31826591e+00  6.81155825e+00
   4.63450766e+00  6.04369068e+00  5.92407084e+00  6.10595798e+00
   5.63913298e+00  4.45252657e+00  5.59457874e+00  7.95584297e+00
   6.67972183e+00  5.56029844e+00  7.25152016e+00  6.23539925e+00
   5.74216366e+00  7.43404961e+00  6.22816515e+00  4.44593191e+00
   4.82426262e+00  6.43218756e+00  6.87160301e+00  6.61254406e+00
   5.85401678e+00  4.75278234e+00  5.82857513e+00  4.87389660e+00
   5.83776760e+00  4.65245199e+00  6.01717520e+00  5.29251719e+00
   6.35411263e+00  5.99921274e+00  5.20506907e+00  5.17370558e+00
  -4.33863580e-01  2.77460486e-01  9.19597864e-01 -1.31707609e-01
  -6.45549655e-01 -1.67132401e+00 -7.72516489e-01 -5.36056459e-01
  -3.73286039e-01 -6.86441839e-01 -1.01108313e-01  7.14173675e-01
   1.18695867e+00  8.51784170e-01 -1.03081465e-01  7.82221973e-01
  -1.06084943e-02  1.65210068e-01  9.46165740e-01 -1.03784645e+00
  -1.05988562e-01 -3.14888358e-01 -1.77463740e-01 -2.11817789e+00
   2.84904152e-01 -1.01048565e+00  6.42939210e-01 -1.55129623e+00
   3.30698609e-01  2.91981220e-01 -4.42365140e-01  1.37139511e+00
   1.15119553e+00 -9.30765271e-02  1.19523180e+00  1.08883739e-01
   6.23302877e-01  1.17179751e+00  1.62847948e+00  7.59559989e-01
   7.38971233e-01  8.67708683e-01  1.01173413e+00  1.63144588e+00
  -7.31225014e-01 -4.42286074e-01 -2.45863259e-01  7.19313860e-01
   9.49248910e-01 -7.67586768e-01 -9.10777271e-01  1.12393653e+00
   4.14927453e-01  4.68949825e-01  3.56323004e-01 -7.02412903e-01
   4.29971993e-01  8.24966609e-01 -1.11557889e+00  1.07412314e+00
   1.05162406e+00  2.43069673e+00  1.74241805e+00  1.94761372e+00
   1.38515937e+00  2.72497535e-01  5.20007968e-01  1.48917031e+00
   1.68797123e+00  4.00417984e-01  1.15737259e+00  1.12503839e+00
   6.92577481e-01  1.99234653e+00  1.17796636e+00  1.47446752e+00
   2.38904285e+00  1.93600380e+00  6.94990337e-01  1.45881605e+00
   9.85398889e-01  2.22224176e-01  6.32130504e-01  4.13003802e-01
   1.92813897e+00  2.21394563e+00  2.34545946e-01  2.40481043e+00
   1.61184263e+00  2.62671828e-01  6.09129429e-01  2.11141062e+00
   1.49358392e+00  2.33948326e+00  5.44065595e-01  7.17817307e-01
   5.57782590e-01  7.01215029e-01  1.02860975e+00  2.24981403e+00
   2.10174203e-01  4.12873387e-01  6.13670349e-01  2.21059442e-01
   6.11453533e-01  2.61570156e-01  8.77194166e-01  1.24051511e+00
   1.72053325e+00  1.17445016e+00  1.25470686e+00  4.10508454e-01
   2.01560163e+00  1.46844494e+00  6.95053816e-01  1.45695102e+00
   1.34463668e+00  1.52593303e+00  8.01309347e-01  1.53490973e+00
   4.84459996e-02  6.16251945e+00  6.05491161e+00  4.72699022e+00
   6.50956726e+00  6.47433615e+00  7.84218597e+00  5.93644285e+00
   8.22062016e+00  5.86653709e+00  7.17612219e+00  6.36410666e+00
   7.53360891e+00  6.64706039e+00  7.14637613e+00  6.75902653e+00
   7.43107319e+00  6.84813881e+00  7.28483295e+00  7.56176233e+00
   6.87353039e+00  6.77652264e+00  6.31159592e+00  6.71193695e+00
   6.20730925e+00  7.30317593e+00  6.52729797e+00  7.00476074e+00
   7.13214016e+00  6.65094757e+00  6.72758961e+00  5.39202499e+00
   5.95804119e+00  7.10594893e+00  6.84224749e+00  7.38022423e+00
   6.14795303e+00  7.69575357e+00  7.79246092e+00  7.66252041e+00
   5.82099390e+00  7.10820389e+00  6.47378588e+00  7.89620018e+00
   5.99701691e+00  7.18277216e+00  7.10389328e+00  6.72315025e+00
   7.34414196e+00  7.02538824e+00  7.57838869e+00  6.31853151e+00
   7.71432066e+00  7.23443127e+00  6.66838741e+00  5.77357435e+00
   7.44090557e+00  7.56217480e+00  7.41124487e+00  6.14406395e+00
   3.12467480e+00  4.31750357e-01  6.48263335e-01  1.06359053e+00
   1.54807031e+00  1.12334645e+00  1.89537644e-01  2.38804555e+00
   1.60959685e+00  1.15691543e+00  3.68243098e-01  1.31778646e+00
   1.14434397e+00  4.39867437e-01  4.07912850e-01  6.60793662e-01
   8.75749111e-01  1.31557178e+00  1.39500034e+00  7.66907394e-01
   1.72432303e+00  2.87720680e-01  4.20079231e-01  7.68439710e-01
   1.31069136e+00  5.41308880e-01  2.20864582e+00  6.96346223e-01
   9.73298550e-01  8.72701943e-01  1.35718942e-01  1.05124402e+00
   9.95015085e-01  1.46241271e+00  6.93884611e-01  1.62696457e+00
   4.71996665e-01  1.38958132e+00  7.28366733e-01  1.99585855e-01
   2.17451477e+00  2.12436342e+00  1.10188627e+00  2.51103425e+00
   1.64973116e+00  1.29206681e+00  2.14518404e+00  1.17625201e+00
   1.09474480e+00  1.39400148e+00  7.90544450e-01  2.32287264e+00
   2.11826897e+00  1.49940050e+00  1.37816489e+00  1.12263310e+00
   1.86812913e+00  1.84924054e+00  2.78295279e-01  8.59706044e-01
  -1.00338535e+01  3.99818087e+00 -2.80066299e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 04:13:21.164094
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.9632
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 04:13:21.168110
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9420.75
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 04:13:21.172121
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.9805
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 04:13:21.176288
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -842.694
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140043244841896
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140042017656168
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140042017656672
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140042017243544
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140042017244048
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140042017244552

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f5e78038ef0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.684387
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.638384
grad_step = 000002, loss = 0.598095
grad_step = 000003, loss = 0.553824
grad_step = 000004, loss = 0.506939
grad_step = 000005, loss = 0.463334
grad_step = 000006, loss = 0.424950
grad_step = 000007, loss = 0.399881
grad_step = 000008, loss = 0.390654
grad_step = 000009, loss = 0.371683
grad_step = 000010, loss = 0.345097
grad_step = 000011, loss = 0.326217
grad_step = 000012, loss = 0.315660
grad_step = 000013, loss = 0.308056
grad_step = 000014, loss = 0.299146
grad_step = 000015, loss = 0.288058
grad_step = 000016, loss = 0.275111
grad_step = 000017, loss = 0.261311
grad_step = 000018, loss = 0.247579
grad_step = 000019, loss = 0.234449
grad_step = 000020, loss = 0.223107
grad_step = 000021, loss = 0.213884
grad_step = 000022, loss = 0.204702
grad_step = 000023, loss = 0.194016
grad_step = 000024, loss = 0.183140
grad_step = 000025, loss = 0.173827
grad_step = 000026, loss = 0.165764
grad_step = 000027, loss = 0.157758
grad_step = 000028, loss = 0.149590
grad_step = 000029, loss = 0.141481
grad_step = 000030, loss = 0.133415
grad_step = 000031, loss = 0.125437
grad_step = 000032, loss = 0.117900
grad_step = 000033, loss = 0.111240
grad_step = 000034, loss = 0.105233
grad_step = 000035, loss = 0.099081
grad_step = 000036, loss = 0.092779
grad_step = 000037, loss = 0.087001
grad_step = 000038, loss = 0.081783
grad_step = 000039, loss = 0.076733
grad_step = 000040, loss = 0.071758
grad_step = 000041, loss = 0.067011
grad_step = 000042, loss = 0.062466
grad_step = 000043, loss = 0.058031
grad_step = 000044, loss = 0.053923
grad_step = 000045, loss = 0.050235
grad_step = 000046, loss = 0.046651
grad_step = 000047, loss = 0.043075
grad_step = 000048, loss = 0.039778
grad_step = 000049, loss = 0.036786
grad_step = 000050, loss = 0.033903
grad_step = 000051, loss = 0.031112
grad_step = 000052, loss = 0.028472
grad_step = 000053, loss = 0.026019
grad_step = 000054, loss = 0.023794
grad_step = 000055, loss = 0.021744
grad_step = 000056, loss = 0.019776
grad_step = 000057, loss = 0.017930
grad_step = 000058, loss = 0.016293
grad_step = 000059, loss = 0.014814
grad_step = 000060, loss = 0.013409
grad_step = 000061, loss = 0.012103
grad_step = 000062, loss = 0.010931
grad_step = 000063, loss = 0.009875
grad_step = 000064, loss = 0.008932
grad_step = 000065, loss = 0.008090
grad_step = 000066, loss = 0.007313
grad_step = 000067, loss = 0.006615
grad_step = 000068, loss = 0.006037
grad_step = 000069, loss = 0.005525
grad_step = 000070, loss = 0.005040
grad_step = 000071, loss = 0.004631
grad_step = 000072, loss = 0.004293
grad_step = 000073, loss = 0.003984
grad_step = 000074, loss = 0.003718
grad_step = 000075, loss = 0.003494
grad_step = 000076, loss = 0.003304
grad_step = 000077, loss = 0.003140
grad_step = 000078, loss = 0.002998
grad_step = 000079, loss = 0.002883
grad_step = 000080, loss = 0.002784
grad_step = 000081, loss = 0.002696
grad_step = 000082, loss = 0.002622
grad_step = 000083, loss = 0.002560
grad_step = 000084, loss = 0.002510
grad_step = 000085, loss = 0.002463
grad_step = 000086, loss = 0.002421
grad_step = 000087, loss = 0.002389
grad_step = 000088, loss = 0.002360
grad_step = 000089, loss = 0.002332
grad_step = 000090, loss = 0.002308
grad_step = 000091, loss = 0.002290
grad_step = 000092, loss = 0.002273
grad_step = 000093, loss = 0.002257
grad_step = 000094, loss = 0.002244
grad_step = 000095, loss = 0.002231
grad_step = 000096, loss = 0.002219
grad_step = 000097, loss = 0.002208
grad_step = 000098, loss = 0.002199
grad_step = 000099, loss = 0.002189
grad_step = 000100, loss = 0.002179
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002170
grad_step = 000102, loss = 0.002161
grad_step = 000103, loss = 0.002151
grad_step = 000104, loss = 0.002141
grad_step = 000105, loss = 0.002131
grad_step = 000106, loss = 0.002121
grad_step = 000107, loss = 0.002110
grad_step = 000108, loss = 0.002099
grad_step = 000109, loss = 0.002088
grad_step = 000110, loss = 0.002077
grad_step = 000111, loss = 0.002066
grad_step = 000112, loss = 0.002054
grad_step = 000113, loss = 0.002042
grad_step = 000114, loss = 0.002030
grad_step = 000115, loss = 0.002019
grad_step = 000116, loss = 0.002007
grad_step = 000117, loss = 0.001995
grad_step = 000118, loss = 0.001984
grad_step = 000119, loss = 0.001972
grad_step = 000120, loss = 0.001969
grad_step = 000121, loss = 0.002024
grad_step = 000122, loss = 0.002156
grad_step = 000123, loss = 0.001996
grad_step = 000124, loss = 0.001952
grad_step = 000125, loss = 0.002059
grad_step = 000126, loss = 0.001924
grad_step = 000127, loss = 0.002002
grad_step = 000128, loss = 0.001976
grad_step = 000129, loss = 0.001939
grad_step = 000130, loss = 0.001981
grad_step = 000131, loss = 0.001896
grad_step = 000132, loss = 0.001946
grad_step = 000133, loss = 0.001887
grad_step = 000134, loss = 0.001906
grad_step = 000135, loss = 0.001923
grad_step = 000136, loss = 0.001858
grad_step = 000137, loss = 0.001906
grad_step = 000138, loss = 0.001876
grad_step = 000139, loss = 0.001848
grad_step = 000140, loss = 0.001882
grad_step = 000141, loss = 0.001845
grad_step = 000142, loss = 0.001838
grad_step = 000143, loss = 0.001858
grad_step = 000144, loss = 0.001824
grad_step = 000145, loss = 0.001825
grad_step = 000146, loss = 0.001837
grad_step = 000147, loss = 0.001808
grad_step = 000148, loss = 0.001809
grad_step = 000149, loss = 0.001818
grad_step = 000150, loss = 0.001796
grad_step = 000151, loss = 0.001790
grad_step = 000152, loss = 0.001799
grad_step = 000153, loss = 0.001787
grad_step = 000154, loss = 0.001774
grad_step = 000155, loss = 0.001778
grad_step = 000156, loss = 0.001777
grad_step = 000157, loss = 0.001765
grad_step = 000158, loss = 0.001757
grad_step = 000159, loss = 0.001760
grad_step = 000160, loss = 0.001759
grad_step = 000161, loss = 0.001749
grad_step = 000162, loss = 0.001741
grad_step = 000163, loss = 0.001739
grad_step = 000164, loss = 0.001740
grad_step = 000165, loss = 0.001737
grad_step = 000166, loss = 0.001730
grad_step = 000167, loss = 0.001723
grad_step = 000168, loss = 0.001718
grad_step = 000169, loss = 0.001717
grad_step = 000170, loss = 0.001716
grad_step = 000171, loss = 0.001715
grad_step = 000172, loss = 0.001713
grad_step = 000173, loss = 0.001709
grad_step = 000174, loss = 0.001705
grad_step = 000175, loss = 0.001701
grad_step = 000176, loss = 0.001697
grad_step = 000177, loss = 0.001693
grad_step = 000178, loss = 0.001690
grad_step = 000179, loss = 0.001686
grad_step = 000180, loss = 0.001684
grad_step = 000181, loss = 0.001682
grad_step = 000182, loss = 0.001682
grad_step = 000183, loss = 0.001684
grad_step = 000184, loss = 0.001691
grad_step = 000185, loss = 0.001703
grad_step = 000186, loss = 0.001725
grad_step = 000187, loss = 0.001740
grad_step = 000188, loss = 0.001729
grad_step = 000189, loss = 0.001688
grad_step = 000190, loss = 0.001659
grad_step = 000191, loss = 0.001649
grad_step = 000192, loss = 0.001656
grad_step = 000193, loss = 0.001672
grad_step = 000194, loss = 0.001665
grad_step = 000195, loss = 0.001641
grad_step = 000196, loss = 0.001630
grad_step = 000197, loss = 0.001628
grad_step = 000198, loss = 0.001634
grad_step = 000199, loss = 0.001643
grad_step = 000200, loss = 0.001638
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001627
grad_step = 000202, loss = 0.001615
grad_step = 000203, loss = 0.001603
grad_step = 000204, loss = 0.001600
grad_step = 000205, loss = 0.001601
grad_step = 000206, loss = 0.001604
grad_step = 000207, loss = 0.001611
grad_step = 000208, loss = 0.001619
grad_step = 000209, loss = 0.001634
grad_step = 000210, loss = 0.001649
grad_step = 000211, loss = 0.001667
grad_step = 000212, loss = 0.001662
grad_step = 000213, loss = 0.001634
grad_step = 000214, loss = 0.001590
grad_step = 000215, loss = 0.001567
grad_step = 000216, loss = 0.001573
grad_step = 000217, loss = 0.001593
grad_step = 000218, loss = 0.001606
grad_step = 000219, loss = 0.001593
grad_step = 000220, loss = 0.001570
grad_step = 000221, loss = 0.001552
grad_step = 000222, loss = 0.001551
grad_step = 000223, loss = 0.001562
grad_step = 000224, loss = 0.001571
grad_step = 000225, loss = 0.001572
grad_step = 000226, loss = 0.001560
grad_step = 000227, loss = 0.001545
grad_step = 000228, loss = 0.001534
grad_step = 000229, loss = 0.001532
grad_step = 000230, loss = 0.001534
grad_step = 000231, loss = 0.001540
grad_step = 000232, loss = 0.001547
grad_step = 000233, loss = 0.001551
grad_step = 000234, loss = 0.001554
grad_step = 000235, loss = 0.001554
grad_step = 000236, loss = 0.001551
grad_step = 000237, loss = 0.001545
grad_step = 000238, loss = 0.001537
grad_step = 000239, loss = 0.001528
grad_step = 000240, loss = 0.001518
grad_step = 000241, loss = 0.001511
grad_step = 000242, loss = 0.001505
grad_step = 000243, loss = 0.001502
grad_step = 000244, loss = 0.001501
grad_step = 000245, loss = 0.001502
grad_step = 000246, loss = 0.001506
grad_step = 000247, loss = 0.001515
grad_step = 000248, loss = 0.001538
grad_step = 000249, loss = 0.001590
grad_step = 000250, loss = 0.001688
grad_step = 000251, loss = 0.001777
grad_step = 000252, loss = 0.001767
grad_step = 000253, loss = 0.001578
grad_step = 000254, loss = 0.001494
grad_step = 000255, loss = 0.001590
grad_step = 000256, loss = 0.001609
grad_step = 000257, loss = 0.001512
grad_step = 000258, loss = 0.001515
grad_step = 000259, loss = 0.001574
grad_step = 000260, loss = 0.001532
grad_step = 000261, loss = 0.001486
grad_step = 000262, loss = 0.001523
grad_step = 000263, loss = 0.001548
grad_step = 000264, loss = 0.001503
grad_step = 000265, loss = 0.001476
grad_step = 000266, loss = 0.001505
grad_step = 000267, loss = 0.001521
grad_step = 000268, loss = 0.001495
grad_step = 000269, loss = 0.001472
grad_step = 000270, loss = 0.001483
grad_step = 000271, loss = 0.001501
grad_step = 000272, loss = 0.001497
grad_step = 000273, loss = 0.001475
grad_step = 000274, loss = 0.001464
grad_step = 000275, loss = 0.001473
grad_step = 000276, loss = 0.001485
grad_step = 000277, loss = 0.001485
grad_step = 000278, loss = 0.001474
grad_step = 000279, loss = 0.001461
grad_step = 000280, loss = 0.001457
grad_step = 000281, loss = 0.001462
grad_step = 000282, loss = 0.001469
grad_step = 000283, loss = 0.001471
grad_step = 000284, loss = 0.001467
grad_step = 000285, loss = 0.001460
grad_step = 000286, loss = 0.001453
grad_step = 000287, loss = 0.001449
grad_step = 000288, loss = 0.001448
grad_step = 000289, loss = 0.001450
grad_step = 000290, loss = 0.001453
grad_step = 000291, loss = 0.001455
grad_step = 000292, loss = 0.001458
grad_step = 000293, loss = 0.001460
grad_step = 000294, loss = 0.001462
grad_step = 000295, loss = 0.001464
grad_step = 000296, loss = 0.001468
grad_step = 000297, loss = 0.001471
grad_step = 000298, loss = 0.001475
grad_step = 000299, loss = 0.001479
grad_step = 000300, loss = 0.001482
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001481
grad_step = 000302, loss = 0.001475
grad_step = 000303, loss = 0.001464
grad_step = 000304, loss = 0.001450
grad_step = 000305, loss = 0.001438
grad_step = 000306, loss = 0.001429
grad_step = 000307, loss = 0.001426
grad_step = 000308, loss = 0.001428
grad_step = 000309, loss = 0.001432
grad_step = 000310, loss = 0.001436
grad_step = 000311, loss = 0.001439
grad_step = 000312, loss = 0.001440
grad_step = 000313, loss = 0.001438
grad_step = 000314, loss = 0.001435
grad_step = 000315, loss = 0.001431
grad_step = 000316, loss = 0.001426
grad_step = 000317, loss = 0.001421
grad_step = 000318, loss = 0.001417
grad_step = 000319, loss = 0.001413
grad_step = 000320, loss = 0.001411
grad_step = 000321, loss = 0.001408
grad_step = 000322, loss = 0.001406
grad_step = 000323, loss = 0.001405
grad_step = 000324, loss = 0.001403
grad_step = 000325, loss = 0.001402
grad_step = 000326, loss = 0.001401
grad_step = 000327, loss = 0.001400
grad_step = 000328, loss = 0.001399
grad_step = 000329, loss = 0.001401
grad_step = 000330, loss = 0.001406
grad_step = 000331, loss = 0.001422
grad_step = 000332, loss = 0.001464
grad_step = 000333, loss = 0.001560
grad_step = 000334, loss = 0.001754
grad_step = 000335, loss = 0.001972
grad_step = 000336, loss = 0.001983
grad_step = 000337, loss = 0.001632
grad_step = 000338, loss = 0.001403
grad_step = 000339, loss = 0.001593
grad_step = 000340, loss = 0.001666
grad_step = 000341, loss = 0.001444
grad_step = 000342, loss = 0.001457
grad_step = 000343, loss = 0.001588
grad_step = 000344, loss = 0.001456
grad_step = 000345, loss = 0.001415
grad_step = 000346, loss = 0.001524
grad_step = 000347, loss = 0.001453
grad_step = 000348, loss = 0.001395
grad_step = 000349, loss = 0.001467
grad_step = 000350, loss = 0.001452
grad_step = 000351, loss = 0.001387
grad_step = 000352, loss = 0.001408
grad_step = 000353, loss = 0.001440
grad_step = 000354, loss = 0.001394
grad_step = 000355, loss = 0.001376
grad_step = 000356, loss = 0.001413
grad_step = 000357, loss = 0.001401
grad_step = 000358, loss = 0.001369
grad_step = 000359, loss = 0.001377
grad_step = 000360, loss = 0.001394
grad_step = 000361, loss = 0.001376
grad_step = 000362, loss = 0.001359
grad_step = 000363, loss = 0.001372
grad_step = 000364, loss = 0.001377
grad_step = 000365, loss = 0.001360
grad_step = 000366, loss = 0.001352
grad_step = 000367, loss = 0.001360
grad_step = 000368, loss = 0.001364
grad_step = 000369, loss = 0.001354
grad_step = 000370, loss = 0.001345
grad_step = 000371, loss = 0.001347
grad_step = 000372, loss = 0.001352
grad_step = 000373, loss = 0.001350
grad_step = 000374, loss = 0.001342
grad_step = 000375, loss = 0.001337
grad_step = 000376, loss = 0.001339
grad_step = 000377, loss = 0.001342
grad_step = 000378, loss = 0.001339
grad_step = 000379, loss = 0.001335
grad_step = 000380, loss = 0.001331
grad_step = 000381, loss = 0.001331
grad_step = 000382, loss = 0.001332
grad_step = 000383, loss = 0.001332
grad_step = 000384, loss = 0.001329
grad_step = 000385, loss = 0.001325
grad_step = 000386, loss = 0.001323
grad_step = 000387, loss = 0.001323
grad_step = 000388, loss = 0.001323
grad_step = 000389, loss = 0.001322
grad_step = 000390, loss = 0.001321
grad_step = 000391, loss = 0.001319
grad_step = 000392, loss = 0.001317
grad_step = 000393, loss = 0.001316
grad_step = 000394, loss = 0.001315
grad_step = 000395, loss = 0.001315
grad_step = 000396, loss = 0.001317
grad_step = 000397, loss = 0.001318
grad_step = 000398, loss = 0.001321
grad_step = 000399, loss = 0.001321
grad_step = 000400, loss = 0.001322
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001320
grad_step = 000402, loss = 0.001317
grad_step = 000403, loss = 0.001312
grad_step = 000404, loss = 0.001309
grad_step = 000405, loss = 0.001308
grad_step = 000406, loss = 0.001310
grad_step = 000407, loss = 0.001315
grad_step = 000408, loss = 0.001320
grad_step = 000409, loss = 0.001327
grad_step = 000410, loss = 0.001333
grad_step = 000411, loss = 0.001339
grad_step = 000412, loss = 0.001342
grad_step = 000413, loss = 0.001345
grad_step = 000414, loss = 0.001345
grad_step = 000415, loss = 0.001346
grad_step = 000416, loss = 0.001344
grad_step = 000417, loss = 0.001344
grad_step = 000418, loss = 0.001342
grad_step = 000419, loss = 0.001336
grad_step = 000420, loss = 0.001326
grad_step = 000421, loss = 0.001307
grad_step = 000422, loss = 0.001291
grad_step = 000423, loss = 0.001284
grad_step = 000424, loss = 0.001288
grad_step = 000425, loss = 0.001299
grad_step = 000426, loss = 0.001309
grad_step = 000427, loss = 0.001311
grad_step = 000428, loss = 0.001306
grad_step = 000429, loss = 0.001299
grad_step = 000430, loss = 0.001297
grad_step = 000431, loss = 0.001301
grad_step = 000432, loss = 0.001311
grad_step = 000433, loss = 0.001318
grad_step = 000434, loss = 0.001322
grad_step = 000435, loss = 0.001316
grad_step = 000436, loss = 0.001308
grad_step = 000437, loss = 0.001298
grad_step = 000438, loss = 0.001291
grad_step = 000439, loss = 0.001287
grad_step = 000440, loss = 0.001286
grad_step = 000441, loss = 0.001283
grad_step = 000442, loss = 0.001279
grad_step = 000443, loss = 0.001273
grad_step = 000444, loss = 0.001264
grad_step = 000445, loss = 0.001258
grad_step = 000446, loss = 0.001255
grad_step = 000447, loss = 0.001256
grad_step = 000448, loss = 0.001260
grad_step = 000449, loss = 0.001265
grad_step = 000450, loss = 0.001269
grad_step = 000451, loss = 0.001273
grad_step = 000452, loss = 0.001279
grad_step = 000453, loss = 0.001297
grad_step = 000454, loss = 0.001329
grad_step = 000455, loss = 0.001385
grad_step = 000456, loss = 0.001451
grad_step = 000457, loss = 0.001520
grad_step = 000458, loss = 0.001501
grad_step = 000459, loss = 0.001410
grad_step = 000460, loss = 0.001286
grad_step = 000461, loss = 0.001252
grad_step = 000462, loss = 0.001316
grad_step = 000463, loss = 0.001362
grad_step = 000464, loss = 0.001320
grad_step = 000465, loss = 0.001248
grad_step = 000466, loss = 0.001243
grad_step = 000467, loss = 0.001290
grad_step = 000468, loss = 0.001312
grad_step = 000469, loss = 0.001291
grad_step = 000470, loss = 0.001252
grad_step = 000471, loss = 0.001233
grad_step = 000472, loss = 0.001242
grad_step = 000473, loss = 0.001264
grad_step = 000474, loss = 0.001275
grad_step = 000475, loss = 0.001258
grad_step = 000476, loss = 0.001236
grad_step = 000477, loss = 0.001223
grad_step = 000478, loss = 0.001225
grad_step = 000479, loss = 0.001239
grad_step = 000480, loss = 0.001253
grad_step = 000481, loss = 0.001258
grad_step = 000482, loss = 0.001250
grad_step = 000483, loss = 0.001241
grad_step = 000484, loss = 0.001236
grad_step = 000485, loss = 0.001238
grad_step = 000486, loss = 0.001248
grad_step = 000487, loss = 0.001256
grad_step = 000488, loss = 0.001257
grad_step = 000489, loss = 0.001249
grad_step = 000490, loss = 0.001236
grad_step = 000491, loss = 0.001224
grad_step = 000492, loss = 0.001215
grad_step = 000493, loss = 0.001212
grad_step = 000494, loss = 0.001212
grad_step = 000495, loss = 0.001211
grad_step = 000496, loss = 0.001205
grad_step = 000497, loss = 0.001198
grad_step = 000498, loss = 0.001192
grad_step = 000499, loss = 0.001188
grad_step = 000500, loss = 0.001188
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001190
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

  date_run                              2020-05-11 04:13:42.761725
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.209469
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 04:13:42.768045
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0932199
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 04:13:42.774304
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.137071
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 04:13:42.780262
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.416509
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
0   2020-05-11 04:13:11.248219  ...    mean_absolute_error
1   2020-05-11 04:13:11.251913  ...     mean_squared_error
2   2020-05-11 04:13:11.255053  ...  median_absolute_error
3   2020-05-11 04:13:11.258262  ...               r2_score
4   2020-05-11 04:13:21.164094  ...    mean_absolute_error
5   2020-05-11 04:13:21.168110  ...     mean_squared_error
6   2020-05-11 04:13:21.172121  ...  median_absolute_error
7   2020-05-11 04:13:21.176288  ...               r2_score
8   2020-05-11 04:13:42.761725  ...    mean_absolute_error
9   2020-05-11 04:13:42.768045  ...     mean_squared_error
10  2020-05-11 04:13:42.774304  ...  median_absolute_error
11  2020-05-11 04:13:42.780262  ...               r2_score

[12 rows x 6 columns] 
  File "pydantic/main.py", line 778, in pydantic.main.create_model
TypeError: create_model() takes exactly 1 positional argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_gluon notfound, create_model() takes exactly 1 positional argument (0 given), tuple index out of range
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do timeseries 





 ************************************************************************************************************************

  vision_mnist 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_cnn/mnist 

  Model List [{'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}}, {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}}] 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 314718.79it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 404504.88it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 561092.20it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 792462.40it/s] 77%|███████▋  | 7634944/9912422 [00:00<00:02, 1120608.47it/s]9920512it [00:00, 10508207.42it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 151002.78it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 312136.14it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 403934.19it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 559238.48it/s]1654784it [00:00, 2809092.79it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 54316.74it/s]            >>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9f5760fba8> <class 'mlmodels.model_tch.torchhub.Model'>
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

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet18/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9ef4d62d30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9f575d2e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9ef4d62a90> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9f5760fba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9f09fccda0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9f5760fba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9efe07c588> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9f5761bfd0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9f09fccda0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9f5761b7b8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f05577c61d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=bbfb50bfecb0888026f1462d5400650687f4af0f46493cefe03aae4a1956a542
  Stored in directory: /tmp/pip-ephem-wheel-cache-rklrr7hl/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.0.2; however, version 20.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f04ef3ad080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 44s
   57344/17464789 [..............................] - ETA: 38s
  106496/17464789 [..............................] - ETA: 31s
  229376/17464789 [..............................] - ETA: 19s
  475136/17464789 [..............................] - ETA: 11s
  958464/17464789 [>.............................] - ETA: 6s 
 1900544/17464789 [==>...........................] - ETA: 3s
 3776512/17464789 [=====>........................] - ETA: 1s
 5742592/17464789 [========>.....................] - ETA: 1s
 7790592/17464789 [============>.................] - ETA: 0s
 9953280/17464789 [================>.............] - ETA: 0s
12230656/17464789 [====================>.........] - ETA: 0s
14606336/17464789 [========================>.....] - ETA: 0s
17096704/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 04:15:14.361605: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 04:15:14.367387: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-11 04:15:14.367550: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ad1adbcaa0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 04:15:14.367569: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6206 - accuracy: 0.5030
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6053 - accuracy: 0.5040 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7228 - accuracy: 0.4963
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8315 - accuracy: 0.4893
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7954 - accuracy: 0.4916
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7714 - accuracy: 0.4932
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7411 - accuracy: 0.4951
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7318 - accuracy: 0.4958
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7211 - accuracy: 0.4964
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7356 - accuracy: 0.4955
11000/25000 [============>.................] - ETA: 4s - loss: 7.7168 - accuracy: 0.4967
12000/25000 [=============>................] - ETA: 3s - loss: 7.7177 - accuracy: 0.4967
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7280 - accuracy: 0.4960
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7017 - accuracy: 0.4977
15000/25000 [=================>............] - ETA: 2s - loss: 7.7014 - accuracy: 0.4977
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6992 - accuracy: 0.4979
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7162 - accuracy: 0.4968
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7084 - accuracy: 0.4973
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6908 - accuracy: 0.4984
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6873 - accuracy: 0.4987
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6893 - accuracy: 0.4985
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6764 - accuracy: 0.4994
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6726 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
25000/25000 [==============================] - 9s 352us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 04:15:30.122894
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 04:15:30.122894  model_keras.textcnn.py  ...    0.5  accuracy_score

[1 rows x 6 columns] 
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do text_classification 





 ************************************************************************************************************************

  nlp_reuters 

  json_path /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/benchmark_text/ 

  Model List [{'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': 'dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': 'ztest/ml_keras/textvae/'}}, {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': True, 'mode': 'test_repo', 'path': 'dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': 'ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}}, {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': 'dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}, {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}}, {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_classifier.py', 'task_name': 'binary', 'model_type': 'xlnet', 'model_name': 'xlnet-base-cased', 'learning_rate': 0.001, 'sequence_length': 56, 'num_classes': 2, 'drop_out': 0.5, 'l2_reg_lambda': 0.0, 'optimization': 'adam', 'embedding_size': 300, 'filter_sizes': [3, 4, 5], 'num_filters': 128, 'do_train': True, 'do_eval': True, 'fp16': False, 'fp16_opt_level': 'O1', 'max_seq_length': 128, 'output_mode': 'classification', 'cache_dir': 'mlmodels/ztest/'}, 'data_pars': {'data_dir': './mlmodels/dataset/text/yelp_reviews/', 'negative_data_file': './dataset/rt-polaritydata/rt-polarity.neg', 'DEV_SAMPLE_PERCENTAGE': 0.1, 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'train': 'True', 'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'cache_dir': 'mlmodels/ztest/'}, 'compute_pars': {'epochs': 10, 'batch_size': 128, 'return_pred': 'True', 'train_batch_size': 8, 'eval_batch_size': 8, 'gradient_accumulation_steps': 1, 'num_train_epochs': 1, 'weight_decay': 0, 'learning_rate': 4e-05, 'adam_epsilon': 1e-08, 'warmup_ratio': 0.06, 'warmup_steps': 0, 'max_grad_norm': 1.0, 'logging_steps': 50, 'evaluate_during_training': False, 'num_samples': 500, 'save_steps': 100, 'eval_all_checkpoints': True, 'overwrite_output_dir': True, 'reprocess_input_data': False}, 'out_pars': {'output_dir': './mlmodels/dataset/text/yelp_reviews/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6], 'modelpath': './output/model/model.h5'}}, {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': 'dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': 1}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}}, {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}}, {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': 'dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}}] 

  


### Running {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': 'dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': 'ztest/ml_keras/textvae/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/textvae/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_keras.textvae.py', 'MAX_NB_WORDS': 12000, 'EMBEDDING_DIM': 50, 'latent_dim': 32, 'intermediate_dim': 96, 'epsilon_std': 0.1, 'num_sampled': 500, 'optimizer': 'adam'}, 'data_pars': {'train': True, 'MAX_SEQUENCE_LENGTH': 15, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv', 'glove_embedding': 'dataset/text/glove/glove.6B.50d.txt'}, 'compute_pars': {'epochs': 1, 'batch_size': 100, 'VALIDATION_SPLIT': 0.2}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/textvae/'}} [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv' 

  


### Running {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': True, 'mode': 'test_repo', 'path': 'dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': 'ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'mode': 'test_repo', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'} 

  #### Setup Model   ############################################## 
Using TensorFlow backend.
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
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_ops.py:2509: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 75)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 75, 40)            1720      
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
2020-05-11 04:15:36.614342: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 04:15:36.619111: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-11 04:15:36.619252: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555d68674fd0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 04:15:36.619267: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fe44d4cad68> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4330 - crf_viterbi_accuracy: 0.0000e+00 - val_loss: 1.4232 - val_crf_viterbi_accuracy: 0.0133

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  {'model_pars': {'model_uri': 'model_keras.namentity_crm_bilstm.py', 'embedding': 40, 'optimizer': 'rmsprop'}, 'data_pars': {'train': False, 'mode': 'test_repo', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ner_dataset.csv', 'location_type': 'repo', 'data_type': 'text', 'data_loader': 'mlmodels.data:import_data_fromfile', 'data_loader_pars': {'size': 50}, 'data_processor': 'mlmodels.model_keras.prepocess:process', 'data_processor_pars': {'split': 0.5, 'max_len': 75}, 'max_len': 75, 'size': [0, 1, 2], 'output_size': [0, 6]}, 'compute_pars': {'epochs': 1, 'batch_size': 64}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/ml_keras/namentity_crm_bilstm/', 'data_type': 'pandas'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


### Running {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': 'dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'IMDB', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_keras.Autokeras.py', 'max_trials': 1}, 'data_pars': {'dataset': 'IMDB', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'num_words': 1000, 'validation_split': 0.15, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 1, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'}} Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range 

  


### Running {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': 'dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': 1, 'maxlen': 40, 'max_features': 5} {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'} 

  #### Setup Model   ############################################## 
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            (None, 40)           0                                            
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 40, 50)       250         input_2[0][0]                    
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 38, 128)      19328       embedding_2[0][0]                
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 37, 128)      25728       embedding_2[0][0]                
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 36, 128)      32128       embedding_2[0][0]                
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
dense_2 (Dense)                 (None, 1)            385         concatenate_1[0][0]              
==================================================================================================
Total params: 77,819
Trainable params: 77,819
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fe46995c048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.3600 - accuracy: 0.5200
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4520 - accuracy: 0.5140 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6155 - accuracy: 0.5033
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6743 - accuracy: 0.4995
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6482 - accuracy: 0.5012
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6411 - accuracy: 0.5017
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6601 - accuracy: 0.5004
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6551 - accuracy: 0.5008
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6544 - accuracy: 0.5008
11000/25000 [============>.................] - ETA: 4s - loss: 7.6889 - accuracy: 0.4985
12000/25000 [=============>................] - ETA: 3s - loss: 7.7165 - accuracy: 0.4967
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7280 - accuracy: 0.4960
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7269 - accuracy: 0.4961
15000/25000 [=================>............] - ETA: 2s - loss: 7.7136 - accuracy: 0.4969
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7145 - accuracy: 0.4969
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7289 - accuracy: 0.4959
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6998 - accuracy: 0.4978
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6916 - accuracy: 0.4984
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7057 - accuracy: 0.4974
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7002 - accuracy: 0.4978
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6889 - accuracy: 0.4985
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6806 - accuracy: 0.4991
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6768 - accuracy: 0.4993
25000/25000 [==============================] - 9s 352us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  {'model_pars': {'model_uri': 'model_keras.textcnn.py', 'maxlen': 40, 'max_features': 5, 'embedding_dims': 50}, 'data_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/imdb.csv', 'train': False, 'maxlen': 40, 'max_features': 5}, 'compute_pars': {'engine': 'adam', 'loss': 'binary_crossentropy', 'metrics': ['accuracy'], 'batch_size': 1000, 'epochs': 1}, 'out_pars': {'path': './output/textcnn_keras//model.h5', 'model_path': './output/textcnn_keras/model.h5'}} module 'sklearn.metrics' has no attribute 'accuracy, f1_score' 

  


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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fe3fe7479b0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

  {'notes': 'Using Yelp Reviews dataset', 'model_pars': {'model_uri': 'model_tch.transformer_sentence.py', 'embedding_model': 'BERT', 'embedding_model_name': 'bert-base-uncased'}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/', 'train_path': 'AllNLI', 'train_type': 'NLI', 'test_path': 'stsbenchmark', 'test_type': 'sts', 'train': True}, 'compute_pars': {'loss': 'SoftmaxLoss', 'batch_size': 32, 'num_epochs': 1, 'evaluation_steps': 10, 'warmup_steps': 100}, 'out_pars': {'path': './output/transformer_sentence/', 'modelpath': './output/transformer_sentence/model.h5'}} 'model_path' 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'data_path': 'dataset/recommender/IMDB_sample.txt', 'train_path': 'dataset/recommender/IMDB_train.csv', 'valid_path': 'dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64} {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'} 

  #### Setup Model   ############################################## 
{'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}

  #### Fit  ####################################################### 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 140, in benchmark_run
    metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 60, in metric_eval
    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
AttributeError: module 'sklearn.metrics' has no attribute 'accuracy, f1_score'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_keras.Autokeras notfound, No module named 'autokeras', tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 140, in benchmark_run
    metric_val = metric_eval(actual=ytrue, pred=ypred,  metric_name=metric)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 60, in metric_eval
    metric = getattr(importlib.import_module("sklearn.metrics"), metric_name)
AttributeError: module 'sklearn.metrics' has no attribute 'accuracy, f1_score'
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 70, in module_load
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 82, in module_load
    model_name = str(Path(model_uri).parts[-2]) + "." + str(model_name)
IndexError: tuple index out of range

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 119, in benchmark_run
    module    = module_load(model_uri)   # "model_tch.torchhub.py"
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 87, in module_load
    raise NameError(f"Module {model_name} notfound, {e1}, {e2}")
NameError: Module model_tch.transformer_classifier notfound, No module named 'util_transformer', tuple index out of range
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/transformer_sentence.py", line 164, in fit
    output_path      = out_pars["model_path"]
KeyError: 'model_path'
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<13:52:48, 17.3kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:00<9:43:51, 24.6kB/s]  .vector_cache/glove.6B.zip:   1%|          | 5.78M/862M [00:00<6:46:15, 35.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.1M/862M [00:00<4:41:39, 50.2kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.8M/862M [00:00<3:15:10, 71.7kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.7M/862M [00:00<2:15:13, 102kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 40.8M/862M [00:01<1:33:40, 146kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.1M/862M [00:01<1:04:52, 209kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:01<45:39, 296kB/s]  .vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:01<32:06, 419kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<13:24:39, 16.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:03<9:23:21, 23.8kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<6:34:51, 33.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.0M/862M [00:05<4:37:12, 48.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.4M/862M [00:05<3:13:19, 68.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<2:43:59, 81.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:07<1:55:38, 115kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<1:22:35, 160kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:09<58:25, 226kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:11<42:48, 308kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:11<30:44, 428kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:13<23:27, 559kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:13<17:08, 764kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:15<13:58, 933kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:15<10:34, 1.23MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:17<09:21, 1.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:17<07:19, 1.77MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:18<07:05, 1.82MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:19<05:42, 2.26MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:20<05:58, 2.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:21<05:13, 2.45MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:22<05:32, 2.30MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:23<05:04, 2.51MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<05:24, 2.35MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<04:59, 2.54MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<05:19, 2.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<04:59, 2.53MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<05:19, 2.36MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<04:18, 2.91MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<05:00, 2.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<04:11, 2.98MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<04:51, 2.56MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<04:12, 2.95MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<04:49, 2.56MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<04:13, 2.92MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<04:48, 2.55MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<04:18, 2.85MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<03:27, 3.54MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<7:50:03, 26.0kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:38<5:27:54, 37.1kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<3:54:13, 51.9kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<2:44:45, 73.7kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:40<1:54:51, 105kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<2:25:33, 83.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<1:42:26, 118kB/s] .vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<1:13:17, 164kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<51:58, 231kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<38:02, 314kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<27:16, 438kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<20:53, 568kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<15:22, 772kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<12:30, 944kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<09:20, 1.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<08:22, 1.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<06:30, 1.80MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<06:24, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<05:14, 2.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:55<05:25, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<04:23, 2.64MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:57<04:52, 2.36MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<04:05, 2.81MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [00:59<04:37, 2.48MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<03:51, 2.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:01<04:28, 2.55MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<04:12, 2.71MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:03<04:35, 2.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:03<04:10, 2.71MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<04:34, 2.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<03:54, 2.88MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:07<04:26, 2.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:07<03:42, 3.01MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<04:19, 2.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<03:35, 3.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:09<02:36, 4.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<39:32, 279kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<28:30, 387kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<20:17, 542kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<7:13:12, 25.4kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:13<5:02:09, 36.3kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<3:35:08, 50.8kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<2:31:23, 72.1kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<1:47:00, 101kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<1:15:25, 144kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<54:12, 199kB/s]  .vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<38:24, 281kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<28:27, 377kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<20:30, 522kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<15:58, 667kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<11:40, 912kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<09:49, 1.08MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<07:35, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<06:52, 1.53MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<05:17, 1.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<05:22, 1.94MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<04:18, 2.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<04:40, 2.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<03:44, 2.77MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:32<04:15, 2.42MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<03:54, 2.64MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:34<04:13, 2.42MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:34<03:27, 2.95MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:36<04:01, 2.52MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:36<03:45, 2.71MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:38<04:06, 2.46MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:38<03:51, 2.62MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<04:09, 2.42MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<03:54, 2.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<04:10, 2.39MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:42<03:29, 2.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<03:57, 2.50MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<03:13, 3.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<03:49, 2.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<03:26, 2.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:47<02:45, 3.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<6:19:51, 25.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:48<4:24:43, 36.8kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<3:08:54, 51.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<2:12:51, 73.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:50<1:32:34, 104kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<1:15:53, 127kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<53:43, 179kB/s]  .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<38:49, 246kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<27:36, 346kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<20:43, 458kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<14:54, 636kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<11:54, 792kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<08:54, 1.06MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<07:38, 1.23MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<05:51, 1.60MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<05:33, 1.67MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<04:20, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:03<04:29, 2.06MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:53, 2.37MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<04:04, 2.25MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<03:16, 2.79MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<03:44, 2.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:21, 2.71MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<03:40, 2.45MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<03:04, 2.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<03:33, 2.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<03:04, 2.91MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<03:29, 2.54MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:13<03:05, 2.87MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:15<03:28, 2.53MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:15<02:47, 3.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<03:23, 2.58MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<03:12, 2.72MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<03:30, 2.47MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<03:16, 2.65MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:19<02:21, 3.66MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<15:56, 540kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:21<11:34, 743kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<08:24, 1.02MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<5:30:47, 25.9kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:23<3:50:27, 36.9kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<2:44:00, 51.7kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<1:56:00, 73.1kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:25<1:20:41, 104kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<1:13:59, 114kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<52:05, 161kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<37:34, 222kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<26:46, 311kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<19:53, 416kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<14:34, 567kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<11:21, 723kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<08:43, 941kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<07:15, 1.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<05:23, 1.51MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<05:05, 1.59MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:01, 2.01MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<04:02, 1.98MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:24, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<03:33, 2.23MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:59, 2.65MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<03:17, 2.39MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:43, 2.88MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<03:08, 2.48MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:37, 2.97MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:46<03:03, 2.53MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<02:33, 3.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<02:58, 2.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<02:29, 3.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:56, 2.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:27, 3.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<02:53, 2.60MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<02:22, 3.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<02:50, 2.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<02:41, 2.76MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:57, 2.50MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<02:24, 3.07MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:57<01:58, 3.72MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<4:44:03, 25.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [02:58<3:17:38, 36.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:58<2:18:19, 52.5kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [02:58<1:38:24, 73.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:00<7:40:29, 15.8kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<5:21:43, 22.5kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<3:44:57, 32.0kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<2:37:38, 45.6kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<1:50:33, 64.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:04<1:17:41, 91.5kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<55:01, 128kB/s]   .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<39:00, 181kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<28:06, 248kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<20:17, 344kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:08<14:08, 489kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<21:39, 319kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<15:32, 444kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<11:50, 578kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<08:56, 765kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<07:11, 942kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<05:39, 1.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<04:54, 1.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<03:55, 1.71MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:16<02:47, 2.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<08:32, 776kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<06:24, 1.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<05:26, 1.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<04:25, 1.48MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<04:01, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<03:20, 1.94MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<03:16, 1.96MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<02:37, 2.45MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:49, 2.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:18, 2.75MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:27<02:35, 2.43MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<02:18, 2.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<02:31, 2.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:14, 2.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:31<02:28, 2.49MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:05, 2.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<02:23, 2.54MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<01:56, 3.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<02:19, 2.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<02:10, 2.75MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<02:23, 2.49MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<02:04, 2.86MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<02:19, 2.52MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<02:09, 2.73MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<02:21, 2.47MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<02:13, 2.61MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<02:22, 2.41MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<02:13, 2.58MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<02:22, 2.39MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<02:04, 2.72MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:46<01:39, 3.39MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<3:32:22, 26.4kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:47<2:27:24, 37.8kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<1:45:28, 52.6kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<1:14:11, 74.7kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<52:10, 105kB/s]   .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<36:42, 149kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<26:17, 206kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<18:37, 290kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<13:45, 388kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<10:07, 527kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<07:46, 678kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<05:39, 929kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<04:45, 1.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<03:36, 1.44MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<03:19, 1.55MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<02:36, 1.97MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:36, 1.94MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:07, 2.38MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:04<02:14, 2.23MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:53, 2.63MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<02:03, 2.39MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:44, 2.82MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<01:57, 2.49MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<01:38, 2.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<01:52, 2.55MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<01:33, 3.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:12<01:49, 2.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:12<01:30, 3.12MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:14<01:46, 2.61MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:14<01:30, 3.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:16<01:46, 2.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<01:39, 2.76MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:18<01:48, 2.49MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<01:33, 2.90MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<01:45, 2.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<01:39, 2.67MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:47, 2.44MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:27, 2.98MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<01:10, 3.66MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<2:48:02, 25.7kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:24<1:56:26, 36.7kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<1:22:43, 51.4kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<58:28, 72.7kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:26<40:33, 104kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<29:54, 140kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<21:18, 196kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:28<14:45, 279kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<15:15, 270kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<11:04, 371kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:30<07:43, 527kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<07:07, 567kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<05:22, 752kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<04:17, 928kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<03:15, 1.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<02:50, 1.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<02:11, 1.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<02:06, 1.81MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:49, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:38<01:17, 2.92MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<11:32, 327kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<08:20, 451kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<06:17, 588kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<04:45, 778kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:42<03:18, 1.10MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<14:04, 258kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<10:12, 356kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<07:32, 473kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<05:36, 634kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:46<03:53, 898kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<06:50, 510kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:47<05:08, 678kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:48<03:34, 961kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<07:25, 462kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<05:30, 622kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:50<03:49, 881kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<05:44, 585kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<04:21, 769kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<03:04, 1.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<03:05, 1.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<02:32, 1.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:48, 1.80MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<02:19, 1.39MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<02:03, 1.56MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:55<01:30, 2.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<01:40, 1.88MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<01:31, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:57<01:06, 2.81MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<01:38, 1.88MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [04:59<01:32, 2.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [04:59<01:05, 2.77MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<01:13, 2.47MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<1:53:33, 26.7kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:01<1:18:31, 38.1kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<55:18, 53.5kB/s]  .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<38:56, 75.9kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<27:05, 107kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<19:14, 150kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:05<13:18, 214kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<10:24, 271kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<07:36, 371kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:07<05:14, 527kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<06:12, 444kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<04:35, 598kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:09<03:12, 845kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<03:08, 855kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<02:27, 1.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:11<01:42, 1.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<22:21, 117kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<15:52, 165kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:13<11:00, 234kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<08:20, 305kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<06:06, 416kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<04:13, 591kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<04:17, 577kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<03:15, 761kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:17<02:15, 1.07MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<02:49, 856kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<02:10, 1.11MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:19<01:30, 1.56MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:20<09:01, 259kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<06:31, 358kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:21<04:28, 509kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<06:19, 359kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:22<04:35, 494kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:23<03:09, 701kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<04:39, 473kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<03:34, 615kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:25<02:28, 870kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<02:32, 841kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<02:04, 1.03MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:27<01:27, 1.44MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<01:39, 1.25MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<01:30, 1.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:28<01:03, 1.91MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<01:16, 1.56MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<01:11, 1.67MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:30<00:52, 2.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<00:59, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:58, 1.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:32<00:42, 2.65MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:33<00:37, 3.04MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<1:06:25, 28.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:34<46:04, 40.3kB/s]  .vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<31:43, 57.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<22:16, 80.9kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:36<15:05, 115kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<17:57, 96.9kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<12:38, 137kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:38<08:39, 195kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<06:42, 249kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<04:52, 342kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:40<03:20, 486kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<03:00, 533kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<02:13, 720kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:42<01:31, 1.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:51, 826kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<01:25, 1.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:44<00:57, 1.52MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<32:22, 45.2kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<23:05, 63.3kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:46<15:53, 90.3kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:47<11:07, 125kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<07:57, 175kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:48<05:24, 249kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:49<04:17, 309kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<03:11, 415kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:50<02:09, 589kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:51<02:07, 591kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:51<01:40, 750kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:52<01:08, 1.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<01:19, 897kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:53<01:06, 1.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:54<00:45, 1.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:55<01:00, 1.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:55<00:51, 1.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:56<00:35, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<00:53, 1.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:57<00:46, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:57<00:31, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<00:47, 1.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [05:59<00:37, 1.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [05:59<00:25, 2.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:46, 1.18MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:01<00:39, 1.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:01<00:26, 1.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<04:15, 203kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<34:44, 24.8kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<23:48, 35.5kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<15:50, 50.2kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:05<11:02, 71.3kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:05<07:10, 102kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<06:51, 106kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<04:49, 149kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:07<03:09, 213kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<02:31, 260kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:09<01:50, 355kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:09<01:12, 505kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<01:04, 549kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:49, 703kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:34, 982kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:28, 1.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:22, 1.39MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:13<00:14, 1.94MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:21, 1.28MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:16, 1.59MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:15<00:10, 2.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:16<00:25, 896kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:19, 1.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:17<00:12, 1.65MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:16, 1.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:14, 1.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:19<00:08, 1.80MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<00:10, 1.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<00:09, 1.57MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:05, 2.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:07, 1.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:05, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:23<00:03, 2.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:24<00:04, 1.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:24<00:03, 1.61MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:25<00:01, 2.23MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:01, 1.55MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:00, 1.86MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 723/400000 [00:00<00:55, 7222.22it/s]  0%|          | 1510/400000 [00:00<00:53, 7404.90it/s]  1%|          | 2321/400000 [00:00<00:52, 7600.67it/s]  1%|          | 3112/400000 [00:00<00:51, 7680.77it/s]  1%|          | 3858/400000 [00:00<00:52, 7610.59it/s]  1%|          | 4661/400000 [00:00<00:51, 7730.25it/s]  1%|▏         | 5459/400000 [00:00<00:50, 7801.15it/s]  2%|▏         | 6264/400000 [00:00<00:50, 7871.38it/s]  2%|▏         | 7017/400000 [00:00<00:50, 7763.66it/s]  2%|▏         | 7776/400000 [00:01<00:50, 7708.44it/s]  2%|▏         | 8557/400000 [00:01<00:50, 7737.71it/s]  2%|▏         | 9318/400000 [00:01<00:52, 7463.69it/s]  3%|▎         | 10057/400000 [00:01<00:54, 7134.25it/s]  3%|▎         | 10809/400000 [00:01<00:53, 7226.29it/s]  3%|▎         | 11565/400000 [00:01<00:53, 7322.68it/s]  3%|▎         | 12357/400000 [00:01<00:51, 7491.03it/s]  3%|▎         | 13129/400000 [00:01<00:51, 7556.69it/s]  3%|▎         | 13886/400000 [00:01<00:52, 7310.61it/s]  4%|▎         | 14620/400000 [00:01<00:54, 7102.99it/s]  4%|▍         | 15334/400000 [00:02<00:55, 6900.09it/s]  4%|▍         | 16071/400000 [00:02<00:54, 7032.72it/s]  4%|▍         | 16867/400000 [00:02<00:52, 7285.09it/s]  4%|▍         | 17636/400000 [00:02<00:51, 7400.75it/s]  5%|▍         | 18444/400000 [00:02<00:50, 7590.65it/s]  5%|▍         | 19224/400000 [00:02<00:49, 7651.08it/s]  5%|▍         | 19992/400000 [00:02<00:49, 7612.43it/s]  5%|▌         | 20756/400000 [00:02<00:51, 7361.82it/s]  5%|▌         | 21545/400000 [00:02<00:50, 7512.02it/s]  6%|▌         | 22314/400000 [00:02<00:49, 7557.75it/s]  6%|▌         | 23072/400000 [00:03<00:50, 7477.83it/s]  6%|▌         | 23834/400000 [00:03<00:50, 7516.79it/s]  6%|▌         | 24587/400000 [00:03<00:50, 7407.12it/s]  6%|▋         | 25329/400000 [00:03<00:52, 7170.86it/s]  7%|▋         | 26049/400000 [00:03<00:53, 7012.68it/s]  7%|▋         | 26761/400000 [00:03<00:52, 7042.94it/s]  7%|▋         | 27491/400000 [00:03<00:52, 7117.94it/s]  7%|▋         | 28324/400000 [00:03<00:49, 7442.33it/s]  7%|▋         | 29153/400000 [00:03<00:48, 7677.06it/s]  7%|▋         | 29962/400000 [00:04<00:47, 7795.75it/s]  8%|▊         | 30746/400000 [00:04<00:47, 7788.28it/s]  8%|▊         | 31549/400000 [00:04<00:46, 7857.87it/s]  8%|▊         | 32337/400000 [00:04<00:46, 7845.68it/s]  8%|▊         | 33124/400000 [00:04<00:48, 7556.02it/s]  8%|▊         | 33883/400000 [00:04<00:50, 7261.35it/s]  9%|▊         | 34614/400000 [00:04<00:51, 7057.07it/s]  9%|▉         | 35325/400000 [00:04<00:52, 6890.30it/s]  9%|▉         | 36052/400000 [00:04<00:52, 6998.82it/s]  9%|▉         | 36828/400000 [00:04<00:50, 7209.95it/s]  9%|▉         | 37626/400000 [00:05<00:48, 7422.73it/s] 10%|▉         | 38439/400000 [00:05<00:47, 7618.69it/s] 10%|▉         | 39245/400000 [00:05<00:46, 7744.06it/s] 10%|█         | 40049/400000 [00:05<00:45, 7829.16it/s] 10%|█         | 40835/400000 [00:05<00:45, 7835.76it/s] 10%|█         | 41636/400000 [00:05<00:45, 7882.29it/s] 11%|█         | 42426/400000 [00:05<00:45, 7859.22it/s] 11%|█         | 43234/400000 [00:05<00:45, 7922.99it/s] 11%|█         | 44032/400000 [00:05<00:44, 7939.87it/s] 11%|█         | 44845/400000 [00:05<00:44, 7995.88it/s] 11%|█▏        | 45646/400000 [00:06<00:44, 7999.07it/s] 12%|█▏        | 46447/400000 [00:06<00:45, 7833.09it/s] 12%|█▏        | 47233/400000 [00:06<00:44, 7840.58it/s] 12%|█▏        | 48018/400000 [00:06<00:45, 7743.08it/s] 12%|█▏        | 48794/400000 [00:06<00:45, 7693.86it/s] 12%|█▏        | 49564/400000 [00:06<00:46, 7598.38it/s] 13%|█▎        | 50387/400000 [00:06<00:44, 7776.11it/s] 13%|█▎        | 51212/400000 [00:06<00:44, 7908.04it/s] 13%|█▎        | 52006/400000 [00:06<00:43, 7916.99it/s] 13%|█▎        | 52843/400000 [00:06<00:43, 8045.84it/s] 13%|█▎        | 53649/400000 [00:07<00:44, 7832.85it/s] 14%|█▎        | 54440/400000 [00:07<00:43, 7854.84it/s] 14%|█▍        | 55249/400000 [00:07<00:43, 7922.38it/s] 14%|█▍        | 56044/400000 [00:07<00:43, 7928.70it/s] 14%|█▍        | 56838/400000 [00:07<00:44, 7792.01it/s] 14%|█▍        | 57630/400000 [00:07<00:43, 7828.85it/s] 15%|█▍        | 58469/400000 [00:07<00:42, 7988.25it/s] 15%|█▍        | 59296/400000 [00:07<00:42, 8068.94it/s] 15%|█▌        | 60140/400000 [00:07<00:41, 8174.57it/s] 15%|█▌        | 60959/400000 [00:07<00:41, 8173.85it/s] 15%|█▌        | 61778/400000 [00:08<00:42, 7874.22it/s] 16%|█▌        | 62569/400000 [00:08<00:43, 7819.04it/s] 16%|█▌        | 63354/400000 [00:08<00:43, 7723.11it/s] 16%|█▌        | 64129/400000 [00:08<00:45, 7436.19it/s] 16%|█▌        | 64877/400000 [00:08<00:46, 7243.72it/s] 16%|█▋        | 65605/400000 [00:08<00:47, 7098.07it/s] 17%|█▋        | 66360/400000 [00:08<00:46, 7226.68it/s] 17%|█▋        | 67171/400000 [00:08<00:44, 7469.14it/s] 17%|█▋        | 67987/400000 [00:08<00:43, 7663.07it/s] 17%|█▋        | 68774/400000 [00:09<00:42, 7721.04it/s] 17%|█▋        | 69551/400000 [00:09<00:42, 7732.35it/s] 18%|█▊        | 70327/400000 [00:09<00:44, 7424.94it/s] 18%|█▊        | 71074/400000 [00:09<00:45, 7215.93it/s] 18%|█▊        | 71800/400000 [00:09<00:46, 7064.96it/s] 18%|█▊        | 72512/400000 [00:09<00:46, 7081.00it/s] 18%|█▊        | 73283/400000 [00:09<00:45, 7256.40it/s] 19%|█▊        | 74030/400000 [00:09<00:44, 7318.76it/s] 19%|█▊        | 74764/400000 [00:09<00:45, 7114.53it/s] 19%|█▉        | 75479/400000 [00:09<00:46, 6976.35it/s] 19%|█▉        | 76180/400000 [00:10<00:47, 6885.95it/s] 19%|█▉        | 76965/400000 [00:10<00:45, 7147.45it/s] 19%|█▉        | 77726/400000 [00:10<00:44, 7278.32it/s] 20%|█▉        | 78483/400000 [00:10<00:43, 7361.19it/s] 20%|█▉        | 79248/400000 [00:10<00:43, 7443.31it/s] 20%|█▉        | 79995/400000 [00:10<00:44, 7240.27it/s] 20%|██        | 80753/400000 [00:10<00:43, 7337.52it/s] 20%|██        | 81493/400000 [00:10<00:43, 7309.12it/s] 21%|██        | 82265/400000 [00:10<00:42, 7425.95it/s] 21%|██        | 83035/400000 [00:11<00:42, 7503.75it/s] 21%|██        | 83793/400000 [00:11<00:42, 7524.84it/s] 21%|██        | 84547/400000 [00:11<00:44, 7121.03it/s] 21%|██▏       | 85265/400000 [00:11<00:44, 7117.55it/s] 21%|██▏       | 85984/400000 [00:11<00:43, 7136.86it/s] 22%|██▏       | 86701/400000 [00:11<00:44, 7007.90it/s] 22%|██▏       | 87405/400000 [00:11<00:45, 6827.15it/s] 22%|██▏       | 88132/400000 [00:11<00:44, 6952.87it/s] 22%|██▏       | 88830/400000 [00:11<00:45, 6787.57it/s] 22%|██▏       | 89518/400000 [00:11<00:45, 6812.85it/s] 23%|██▎       | 90230/400000 [00:12<00:44, 6901.78it/s] 23%|██▎       | 90922/400000 [00:12<00:44, 6886.36it/s] 23%|██▎       | 91654/400000 [00:12<00:43, 7010.45it/s] 23%|██▎       | 92365/400000 [00:12<00:43, 7037.37it/s] 23%|██▎       | 93070/400000 [00:12<00:44, 6909.16it/s] 23%|██▎       | 93763/400000 [00:12<00:45, 6798.59it/s] 24%|██▎       | 94445/400000 [00:12<00:45, 6761.89it/s] 24%|██▍       | 95123/400000 [00:12<00:45, 6684.61it/s] 24%|██▍       | 95808/400000 [00:12<00:45, 6733.34it/s] 24%|██▍       | 96538/400000 [00:12<00:44, 6893.51it/s] 24%|██▍       | 97244/400000 [00:13<00:43, 6942.02it/s] 24%|██▍       | 97940/400000 [00:13<00:44, 6863.29it/s] 25%|██▍       | 98628/400000 [00:13<00:46, 6519.02it/s] 25%|██▍       | 99285/400000 [00:13<00:47, 6331.72it/s] 25%|██▍       | 99966/400000 [00:13<00:46, 6467.56it/s] 25%|██▌       | 100617/400000 [00:13<00:46, 6438.41it/s] 25%|██▌       | 101332/400000 [00:13<00:45, 6634.96it/s] 26%|██▌       | 102046/400000 [00:13<00:43, 6777.24it/s] 26%|██▌       | 102735/400000 [00:13<00:43, 6808.53it/s] 26%|██▌       | 103461/400000 [00:14<00:42, 6906.17it/s] 26%|██▌       | 104154/400000 [00:14<00:43, 6835.39it/s] 26%|██▌       | 104840/400000 [00:14<00:43, 6793.40it/s] 26%|██▋       | 105521/400000 [00:14<00:44, 6654.97it/s] 27%|██▋       | 106272/400000 [00:14<00:42, 6889.38it/s] 27%|██▋       | 107037/400000 [00:14<00:41, 7098.80it/s] 27%|██▋       | 107787/400000 [00:14<00:40, 7213.37it/s] 27%|██▋       | 108532/400000 [00:14<00:40, 7281.56it/s] 27%|██▋       | 109263/400000 [00:14<00:42, 6886.56it/s] 28%|██▊       | 110015/400000 [00:14<00:41, 7064.71it/s] 28%|██▊       | 110811/400000 [00:15<00:39, 7310.52it/s] 28%|██▊       | 111550/400000 [00:15<00:39, 7332.00it/s] 28%|██▊       | 112303/400000 [00:15<00:38, 7388.59it/s] 28%|██▊       | 113045/400000 [00:15<00:39, 7344.83it/s] 28%|██▊       | 113797/400000 [00:15<00:38, 7393.95it/s] 29%|██▊       | 114538/400000 [00:15<00:39, 7309.27it/s] 29%|██▉       | 115271/400000 [00:15<00:39, 7172.60it/s] 29%|██▉       | 116003/400000 [00:15<00:39, 7216.18it/s] 29%|██▉       | 116726/400000 [00:15<00:39, 7161.70it/s] 29%|██▉       | 117450/400000 [00:15<00:39, 7183.81it/s] 30%|██▉       | 118213/400000 [00:16<00:38, 7311.32it/s] 30%|██▉       | 119003/400000 [00:16<00:37, 7476.37it/s] 30%|██▉       | 119767/400000 [00:16<00:37, 7472.19it/s] 30%|███       | 120516/400000 [00:16<00:37, 7397.06it/s] 30%|███       | 121257/400000 [00:16<00:38, 7170.89it/s] 30%|███       | 121983/400000 [00:16<00:38, 7195.25it/s] 31%|███       | 122705/400000 [00:16<00:38, 7157.23it/s] 31%|███       | 123422/400000 [00:16<00:40, 6821.49it/s] 31%|███       | 124150/400000 [00:16<00:39, 6949.45it/s] 31%|███       | 124915/400000 [00:16<00:38, 7144.45it/s] 31%|███▏      | 125667/400000 [00:17<00:37, 7251.82it/s] 32%|███▏      | 126396/400000 [00:17<00:37, 7222.01it/s] 32%|███▏      | 127174/400000 [00:17<00:36, 7379.97it/s] 32%|███▏      | 127960/400000 [00:17<00:36, 7516.48it/s] 32%|███▏      | 128745/400000 [00:17<00:35, 7611.69it/s] 32%|███▏      | 129512/400000 [00:17<00:35, 7627.60it/s] 33%|███▎      | 130290/400000 [00:17<00:35, 7672.11it/s] 33%|███▎      | 131064/400000 [00:17<00:34, 7689.22it/s] 33%|███▎      | 131834/400000 [00:17<00:36, 7372.90it/s] 33%|███▎      | 132575/400000 [00:18<00:37, 7227.16it/s] 33%|███▎      | 133301/400000 [00:18<00:37, 7201.47it/s] 34%|███▎      | 134080/400000 [00:18<00:36, 7368.04it/s] 34%|███▎      | 134853/400000 [00:18<00:35, 7468.56it/s] 34%|███▍      | 135631/400000 [00:18<00:34, 7558.83it/s] 34%|███▍      | 136389/400000 [00:18<00:35, 7427.26it/s] 34%|███▍      | 137183/400000 [00:18<00:34, 7572.17it/s] 34%|███▍      | 137966/400000 [00:18<00:34, 7646.27it/s] 35%|███▍      | 138733/400000 [00:18<00:34, 7604.92it/s] 35%|███▍      | 139495/400000 [00:18<00:34, 7495.97it/s] 35%|███▌      | 140246/400000 [00:19<00:35, 7330.22it/s] 35%|███▌      | 140981/400000 [00:19<00:36, 7177.87it/s] 35%|███▌      | 141701/400000 [00:19<00:36, 7028.63it/s] 36%|███▌      | 142406/400000 [00:19<00:37, 6940.24it/s] 36%|███▌      | 143134/400000 [00:19<00:36, 7035.88it/s] 36%|███▌      | 143917/400000 [00:19<00:35, 7255.11it/s] 36%|███▌      | 144646/400000 [00:19<00:35, 7163.56it/s] 36%|███▋      | 145377/400000 [00:19<00:35, 7205.94it/s] 37%|███▋      | 146126/400000 [00:19<00:34, 7288.32it/s] 37%|███▋      | 146914/400000 [00:19<00:33, 7454.54it/s] 37%|███▋      | 147689/400000 [00:20<00:33, 7540.59it/s] 37%|███▋      | 148484/400000 [00:20<00:32, 7656.34it/s] 37%|███▋      | 149252/400000 [00:20<00:33, 7593.20it/s] 38%|███▊      | 150013/400000 [00:20<00:33, 7571.67it/s] 38%|███▊      | 150819/400000 [00:20<00:32, 7708.81it/s] 38%|███▊      | 151592/400000 [00:20<00:32, 7528.48it/s] 38%|███▊      | 152367/400000 [00:20<00:32, 7582.14it/s] 38%|███▊      | 153165/400000 [00:20<00:32, 7695.17it/s] 38%|███▊      | 153936/400000 [00:20<00:32, 7558.78it/s] 39%|███▊      | 154694/400000 [00:20<00:33, 7420.46it/s] 39%|███▉      | 155438/400000 [00:21<00:34, 7123.65it/s] 39%|███▉      | 156155/400000 [00:21<00:34, 6990.83it/s] 39%|███▉      | 156858/400000 [00:21<00:36, 6740.28it/s] 39%|███▉      | 157620/400000 [00:21<00:34, 6981.84it/s] 40%|███▉      | 158430/400000 [00:21<00:33, 7282.81it/s] 40%|███▉      | 159231/400000 [00:21<00:32, 7485.86it/s] 40%|████      | 160027/400000 [00:21<00:31, 7614.83it/s] 40%|████      | 160794/400000 [00:21<00:31, 7574.19it/s] 40%|████      | 161555/400000 [00:21<00:31, 7473.77it/s] 41%|████      | 162306/400000 [00:22<00:31, 7470.88it/s] 41%|████      | 163085/400000 [00:22<00:31, 7562.69it/s] 41%|████      | 163854/400000 [00:22<00:31, 7598.43it/s] 41%|████      | 164616/400000 [00:22<00:30, 7598.11it/s] 41%|████▏     | 165377/400000 [00:22<00:31, 7472.68it/s] 42%|████▏     | 166163/400000 [00:22<00:30, 7584.75it/s] 42%|████▏     | 166958/400000 [00:22<00:30, 7690.04it/s] 42%|████▏     | 167739/400000 [00:22<00:30, 7723.33it/s] 42%|████▏     | 168513/400000 [00:22<00:30, 7594.38it/s] 42%|████▏     | 169274/400000 [00:22<00:30, 7513.59it/s] 43%|████▎     | 170037/400000 [00:23<00:30, 7546.19it/s] 43%|████▎     | 170802/400000 [00:23<00:30, 7576.58it/s] 43%|████▎     | 171590/400000 [00:23<00:29, 7662.48it/s] 43%|████▎     | 172359/400000 [00:23<00:29, 7669.29it/s] 43%|████▎     | 173127/400000 [00:23<00:29, 7594.30it/s] 43%|████▎     | 173910/400000 [00:23<00:29, 7660.85it/s] 44%|████▎     | 174691/400000 [00:23<00:29, 7703.40it/s] 44%|████▍     | 175475/400000 [00:23<00:28, 7742.61it/s] 44%|████▍     | 176258/400000 [00:23<00:28, 7768.03it/s] 44%|████▍     | 177036/400000 [00:23<00:29, 7455.03it/s] 44%|████▍     | 177785/400000 [00:24<00:30, 7359.23it/s] 45%|████▍     | 178564/400000 [00:24<00:29, 7481.18it/s] 45%|████▍     | 179315/400000 [00:24<00:29, 7374.40it/s] 45%|████▌     | 180055/400000 [00:24<00:30, 7168.09it/s] 45%|████▌     | 180775/400000 [00:24<00:31, 6978.83it/s] 45%|████▌     | 181476/400000 [00:24<00:31, 6927.66it/s] 46%|████▌     | 182275/400000 [00:24<00:30, 7215.05it/s] 46%|████▌     | 183066/400000 [00:24<00:29, 7410.11it/s] 46%|████▌     | 183812/400000 [00:24<00:29, 7278.08it/s] 46%|████▌     | 184571/400000 [00:25<00:29, 7366.48it/s] 46%|████▋     | 185335/400000 [00:25<00:28, 7444.01it/s] 47%|████▋     | 186090/400000 [00:25<00:28, 7473.66it/s] 47%|████▋     | 186889/400000 [00:25<00:27, 7619.48it/s] 47%|████▋     | 187682/400000 [00:25<00:27, 7709.33it/s] 47%|████▋     | 188455/400000 [00:25<00:27, 7640.08it/s] 47%|████▋     | 189221/400000 [00:25<00:28, 7395.22it/s] 47%|████▋     | 189964/400000 [00:25<00:29, 7210.15it/s] 48%|████▊     | 190688/400000 [00:25<00:29, 7078.41it/s] 48%|████▊     | 191399/400000 [00:25<00:30, 6918.61it/s] 48%|████▊     | 192094/400000 [00:26<00:30, 6859.74it/s] 48%|████▊     | 192830/400000 [00:26<00:29, 7000.16it/s] 48%|████▊     | 193597/400000 [00:26<00:28, 7187.44it/s] 49%|████▊     | 194360/400000 [00:26<00:28, 7311.90it/s] 49%|████▉     | 195137/400000 [00:26<00:27, 7442.18it/s] 49%|████▉     | 195945/400000 [00:26<00:26, 7621.61it/s] 49%|████▉     | 196710/400000 [00:26<00:26, 7617.65it/s] 49%|████▉     | 197528/400000 [00:26<00:26, 7777.81it/s] 50%|████▉     | 198325/400000 [00:26<00:25, 7833.28it/s] 50%|████▉     | 199110/400000 [00:26<00:26, 7463.55it/s] 50%|████▉     | 199862/400000 [00:27<00:27, 7193.76it/s] 50%|█████     | 200587/400000 [00:27<00:28, 6947.55it/s] 50%|█████     | 201355/400000 [00:27<00:27, 7149.79it/s] 51%|█████     | 202136/400000 [00:27<00:26, 7334.53it/s] 51%|█████     | 202905/400000 [00:27<00:26, 7435.61it/s] 51%|█████     | 203653/400000 [00:27<00:26, 7329.87it/s] 51%|█████     | 204389/400000 [00:27<00:27, 7127.24it/s] 51%|█████▏    | 205106/400000 [00:27<00:28, 6919.63it/s] 51%|█████▏    | 205828/400000 [00:27<00:27, 7004.40it/s] 52%|█████▏    | 206601/400000 [00:28<00:26, 7206.63it/s] 52%|█████▏    | 207385/400000 [00:28<00:26, 7385.04it/s] 52%|█████▏    | 208196/400000 [00:28<00:25, 7587.50it/s] 52%|█████▏    | 209014/400000 [00:28<00:24, 7754.84it/s] 52%|█████▏    | 209794/400000 [00:28<00:25, 7590.87it/s] 53%|█████▎    | 210582/400000 [00:28<00:24, 7673.29it/s] 53%|█████▎    | 211409/400000 [00:28<00:24, 7842.91it/s] 53%|█████▎    | 212196/400000 [00:28<00:24, 7552.30it/s] 53%|█████▎    | 212956/400000 [00:28<00:24, 7553.02it/s] 53%|█████▎    | 213715/400000 [00:28<00:25, 7353.87it/s] 54%|█████▎    | 214454/400000 [00:29<00:26, 7135.40it/s] 54%|█████▍    | 215172/400000 [00:29<00:26, 7002.61it/s] 54%|█████▍    | 215876/400000 [00:29<00:26, 6852.14it/s] 54%|█████▍    | 216565/400000 [00:29<00:26, 6833.76it/s] 54%|█████▍    | 217280/400000 [00:29<00:26, 6924.39it/s] 55%|█████▍    | 218059/400000 [00:29<00:25, 7161.95it/s] 55%|█████▍    | 218825/400000 [00:29<00:24, 7297.67it/s] 55%|█████▍    | 219624/400000 [00:29<00:24, 7491.96it/s] 55%|█████▌    | 220444/400000 [00:29<00:23, 7689.02it/s] 55%|█████▌    | 221217/400000 [00:29<00:23, 7465.30it/s] 55%|█████▌    | 221968/400000 [00:30<00:24, 7281.39it/s] 56%|█████▌    | 222732/400000 [00:30<00:24, 7383.63it/s] 56%|█████▌    | 223536/400000 [00:30<00:23, 7567.52it/s] 56%|█████▌    | 224296/400000 [00:30<00:24, 7295.22it/s] 56%|█████▋    | 225030/400000 [00:30<00:24, 7071.34it/s] 56%|█████▋    | 225835/400000 [00:30<00:23, 7338.71it/s] 57%|█████▋    | 226645/400000 [00:30<00:22, 7551.45it/s] 57%|█████▋    | 227406/400000 [00:30<00:23, 7485.71it/s] 57%|█████▋    | 228159/400000 [00:30<00:23, 7397.73it/s] 57%|█████▋    | 228918/400000 [00:31<00:22, 7452.83it/s] 57%|█████▋    | 229666/400000 [00:31<00:23, 7269.51it/s] 58%|█████▊    | 230396/400000 [00:31<00:23, 7140.16it/s] 58%|█████▊    | 231113/400000 [00:31<00:24, 7031.24it/s] 58%|█████▊    | 231819/400000 [00:31<00:24, 6900.08it/s] 58%|█████▊    | 232511/400000 [00:31<00:24, 6868.50it/s] 58%|█████▊    | 233324/400000 [00:31<00:23, 7202.93it/s] 59%|█████▊    | 234068/400000 [00:31<00:22, 7269.15it/s] 59%|█████▊    | 234834/400000 [00:31<00:22, 7379.47it/s] 59%|█████▉    | 235625/400000 [00:31<00:21, 7529.19it/s] 59%|█████▉    | 236408/400000 [00:32<00:21, 7616.84it/s] 59%|█████▉    | 237191/400000 [00:32<00:21, 7676.33it/s] 59%|█████▉    | 237961/400000 [00:32<00:21, 7574.72it/s] 60%|█████▉    | 238720/400000 [00:32<00:22, 7232.61it/s] 60%|█████▉    | 239451/400000 [00:32<00:22, 7253.75it/s] 60%|██████    | 240262/400000 [00:32<00:21, 7489.33it/s] 60%|██████    | 241015/400000 [00:32<00:21, 7405.27it/s] 60%|██████    | 241759/400000 [00:32<00:22, 7155.45it/s] 61%|██████    | 242479/400000 [00:32<00:22, 6989.48it/s] 61%|██████    | 243202/400000 [00:33<00:22, 7058.34it/s] 61%|██████    | 244010/400000 [00:33<00:21, 7336.50it/s] 61%|██████    | 244834/400000 [00:33<00:20, 7584.49it/s] 61%|██████▏   | 245648/400000 [00:33<00:19, 7741.26it/s] 62%|██████▏   | 246451/400000 [00:33<00:19, 7823.87it/s] 62%|██████▏   | 247237/400000 [00:33<00:19, 7788.82it/s] 62%|██████▏   | 248028/400000 [00:33<00:19, 7823.14it/s] 62%|██████▏   | 248836/400000 [00:33<00:19, 7896.61it/s] 62%|██████▏   | 249666/400000 [00:33<00:18, 8010.54it/s] 63%|██████▎   | 250493/400000 [00:33<00:18, 8086.05it/s] 63%|██████▎   | 251303/400000 [00:34<00:18, 7860.90it/s] 63%|██████▎   | 252120/400000 [00:34<00:18, 7950.63it/s] 63%|██████▎   | 252925/400000 [00:34<00:18, 7979.22it/s] 63%|██████▎   | 253725/400000 [00:34<00:18, 7820.96it/s] 64%|██████▎   | 254509/400000 [00:34<00:18, 7820.24it/s] 64%|██████▍   | 255293/400000 [00:34<00:18, 7733.30it/s] 64%|██████▍   | 256129/400000 [00:34<00:18, 7885.64it/s] 64%|██████▍   | 256964/400000 [00:34<00:17, 8017.90it/s] 64%|██████▍   | 257772/400000 [00:34<00:17, 8035.22it/s] 65%|██████▍   | 258577/400000 [00:34<00:17, 7981.54it/s] 65%|██████▍   | 259376/400000 [00:35<00:18, 7802.37it/s] 65%|██████▌   | 260158/400000 [00:35<00:18, 7695.19it/s] 65%|██████▌   | 260929/400000 [00:35<00:18, 7688.64it/s] 65%|██████▌   | 261699/400000 [00:35<00:17, 7685.29it/s] 66%|██████▌   | 262518/400000 [00:35<00:17, 7827.86it/s] 66%|██████▌   | 263302/400000 [00:35<00:17, 7805.76it/s] 66%|██████▌   | 264106/400000 [00:35<00:17, 7871.92it/s] 66%|██████▌   | 264923/400000 [00:35<00:16, 7957.73it/s] 66%|██████▋   | 265745/400000 [00:35<00:16, 8032.54it/s] 67%|██████▋   | 266549/400000 [00:35<00:16, 7954.94it/s] 67%|██████▋   | 267346/400000 [00:36<00:17, 7762.70it/s] 67%|██████▋   | 268124/400000 [00:36<00:17, 7469.33it/s] 67%|██████▋   | 268875/400000 [00:36<00:18, 7204.45it/s] 67%|██████▋   | 269600/400000 [00:36<00:18, 7136.49it/s] 68%|██████▊   | 270433/400000 [00:36<00:17, 7454.93it/s] 68%|██████▊   | 271194/400000 [00:36<00:17, 7499.16it/s] 68%|██████▊   | 272016/400000 [00:36<00:16, 7701.60it/s] 68%|██████▊   | 272848/400000 [00:36<00:16, 7876.72it/s] 68%|██████▊   | 273640/400000 [00:36<00:16, 7721.76it/s] 69%|██████▊   | 274432/400000 [00:36<00:16, 7780.13it/s] 69%|██████▉   | 275213/400000 [00:37<00:16, 7747.70it/s] 69%|██████▉   | 275997/400000 [00:37<00:15, 7772.64it/s] 69%|██████▉   | 276776/400000 [00:37<00:15, 7753.09it/s] 69%|██████▉   | 277553/400000 [00:37<00:15, 7757.14it/s] 70%|██████▉   | 278330/400000 [00:37<00:16, 7502.49it/s] 70%|██████▉   | 279083/400000 [00:37<00:16, 7368.44it/s] 70%|██████▉   | 279877/400000 [00:37<00:15, 7529.22it/s] 70%|███████   | 280661/400000 [00:37<00:15, 7618.49it/s] 70%|███████   | 281456/400000 [00:37<00:15, 7714.03it/s] 71%|███████   | 282266/400000 [00:38<00:15, 7823.04it/s] 71%|███████   | 283050/400000 [00:38<00:15, 7737.98it/s] 71%|███████   | 283826/400000 [00:38<00:15, 7706.62it/s] 71%|███████   | 284598/400000 [00:38<00:14, 7708.59it/s] 71%|███████▏  | 285429/400000 [00:38<00:14, 7878.65it/s] 72%|███████▏  | 286219/400000 [00:38<00:14, 7788.99it/s] 72%|███████▏  | 287000/400000 [00:38<00:14, 7687.67it/s] 72%|███████▏  | 287811/400000 [00:38<00:14, 7807.58it/s] 72%|███████▏  | 288661/400000 [00:38<00:13, 8003.04it/s] 72%|███████▏  | 289511/400000 [00:38<00:13, 8144.38it/s] 73%|███████▎  | 290336/400000 [00:39<00:13, 8173.33it/s] 73%|███████▎  | 291155/400000 [00:39<00:13, 8177.63it/s] 73%|███████▎  | 291974/400000 [00:39<00:13, 8092.45it/s] 73%|███████▎  | 292790/400000 [00:39<00:13, 8111.18it/s] 73%|███████▎  | 293602/400000 [00:39<00:13, 7686.62it/s] 74%|███████▎  | 294376/400000 [00:39<00:14, 7307.39it/s] 74%|███████▍  | 295115/400000 [00:39<00:14, 7115.09it/s] 74%|███████▍  | 295912/400000 [00:39<00:14, 7349.35it/s] 74%|███████▍  | 296773/400000 [00:39<00:13, 7684.65it/s] 74%|███████▍  | 297550/400000 [00:39<00:13, 7554.94it/s] 75%|███████▍  | 298312/400000 [00:40<00:13, 7424.31it/s] 75%|███████▍  | 299060/400000 [00:40<00:13, 7409.47it/s] 75%|███████▍  | 299805/400000 [00:40<00:13, 7274.06it/s] 75%|███████▌  | 300621/400000 [00:40<00:13, 7518.73it/s] 75%|███████▌  | 301413/400000 [00:40<00:12, 7632.27it/s] 76%|███████▌  | 302200/400000 [00:40<00:12, 7702.00it/s] 76%|███████▌  | 303047/400000 [00:40<00:12, 7915.25it/s] 76%|███████▌  | 303876/400000 [00:40<00:11, 8022.86it/s] 76%|███████▌  | 304681/400000 [00:40<00:12, 7814.52it/s] 76%|███████▋  | 305532/400000 [00:41<00:11, 8008.35it/s] 77%|███████▋  | 306337/400000 [00:41<00:12, 7746.74it/s] 77%|███████▋  | 307154/400000 [00:41<00:11, 7867.05it/s] 77%|███████▋  | 307994/400000 [00:41<00:11, 8018.71it/s] 77%|███████▋  | 308856/400000 [00:41<00:11, 8189.63it/s] 77%|███████▋  | 309679/400000 [00:41<00:11, 8040.82it/s] 78%|███████▊  | 310501/400000 [00:41<00:11, 8092.04it/s] 78%|███████▊  | 311349/400000 [00:41<00:10, 8203.09it/s] 78%|███████▊  | 312186/400000 [00:41<00:10, 8251.49it/s] 78%|███████▊  | 313042/400000 [00:41<00:10, 8341.64it/s] 78%|███████▊  | 313878/400000 [00:42<00:11, 7808.30it/s] 79%|███████▊  | 314667/400000 [00:42<00:11, 7464.32it/s] 79%|███████▉  | 315423/400000 [00:42<00:11, 7195.27it/s] 79%|███████▉  | 316167/400000 [00:42<00:11, 7264.85it/s] 79%|███████▉  | 316907/400000 [00:42<00:11, 7303.48it/s] 79%|███████▉  | 317668/400000 [00:42<00:11, 7390.70it/s] 80%|███████▉  | 318411/400000 [00:42<00:11, 7229.48it/s] 80%|███████▉  | 319137/400000 [00:42<00:11, 6968.36it/s] 80%|███████▉  | 319838/400000 [00:42<00:11, 6829.03it/s] 80%|████████  | 320627/400000 [00:43<00:11, 7114.11it/s] 80%|████████  | 321440/400000 [00:43<00:10, 7390.83it/s] 81%|████████  | 322298/400000 [00:43<00:10, 7709.66it/s] 81%|████████  | 323137/400000 [00:43<00:09, 7879.44it/s] 81%|████████  | 323997/400000 [00:43<00:09, 8082.28it/s] 81%|████████  | 324845/400000 [00:43<00:09, 8196.61it/s] 81%|████████▏ | 325670/400000 [00:43<00:09, 8069.09it/s] 82%|████████▏ | 326524/400000 [00:43<00:08, 8198.66it/s] 82%|████████▏ | 327348/400000 [00:43<00:08, 8208.92it/s] 82%|████████▏ | 328197/400000 [00:43<00:08, 8290.99it/s] 82%|████████▏ | 329028/400000 [00:44<00:09, 7626.95it/s] 82%|████████▏ | 329803/400000 [00:44<00:09, 7384.02it/s] 83%|████████▎ | 330660/400000 [00:44<00:09, 7702.59it/s] 83%|████████▎ | 331491/400000 [00:44<00:08, 7874.72it/s] 83%|████████▎ | 332344/400000 [00:44<00:08, 8059.05it/s] 83%|████████▎ | 333167/400000 [00:44<00:08, 8107.49it/s] 84%|████████▎ | 334003/400000 [00:44<00:08, 8180.74it/s] 84%|████████▎ | 334860/400000 [00:44<00:07, 8292.12it/s] 84%|████████▍ | 335706/400000 [00:44<00:07, 8339.96it/s] 84%|████████▍ | 336566/400000 [00:44<00:07, 8414.72it/s] 84%|████████▍ | 337410/400000 [00:45<00:07, 8224.97it/s] 85%|████████▍ | 338235/400000 [00:45<00:08, 7699.95it/s] 85%|████████▍ | 339014/400000 [00:45<00:08, 7442.01it/s] 85%|████████▍ | 339826/400000 [00:45<00:07, 7632.69it/s] 85%|████████▌ | 340664/400000 [00:45<00:07, 7840.42it/s] 85%|████████▌ | 341455/400000 [00:45<00:07, 7672.51it/s] 86%|████████▌ | 342228/400000 [00:45<00:07, 7381.02it/s] 86%|████████▌ | 342972/400000 [00:45<00:07, 7177.11it/s] 86%|████████▌ | 343696/400000 [00:45<00:07, 7057.89it/s] 86%|████████▌ | 344406/400000 [00:46<00:08, 6939.85it/s] 86%|████████▋ | 345173/400000 [00:46<00:07, 7141.68it/s] 86%|████████▋ | 345892/400000 [00:46<00:07, 7033.26it/s] 87%|████████▋ | 346700/400000 [00:46<00:07, 7317.59it/s] 87%|████████▋ | 347529/400000 [00:46<00:06, 7583.22it/s] 87%|████████▋ | 348340/400000 [00:46<00:06, 7732.89it/s] 87%|████████▋ | 349159/400000 [00:46<00:06, 7862.31it/s] 88%|████████▊ | 350006/400000 [00:46<00:06, 8033.68it/s] 88%|████████▊ | 350838/400000 [00:46<00:06, 8116.52it/s] 88%|████████▊ | 351665/400000 [00:46<00:05, 8160.64it/s] 88%|████████▊ | 352484/400000 [00:47<00:05, 8145.07it/s] 88%|████████▊ | 353300/400000 [00:47<00:06, 7722.39it/s] 89%|████████▊ | 354078/400000 [00:47<00:06, 7409.89it/s] 89%|████████▊ | 354826/400000 [00:47<00:06, 7182.47it/s] 89%|████████▉ | 355551/400000 [00:47<00:06, 7038.87it/s] 89%|████████▉ | 356349/400000 [00:47<00:05, 7295.72it/s] 89%|████████▉ | 357146/400000 [00:47<00:05, 7483.15it/s] 89%|████████▉ | 357901/400000 [00:47<00:05, 7500.86it/s] 90%|████████▉ | 358681/400000 [00:47<00:05, 7587.07it/s] 90%|████████▉ | 359443/400000 [00:48<00:05, 7587.32it/s] 90%|█████████ | 360204/400000 [00:48<00:05, 7504.95it/s] 90%|█████████ | 361049/400000 [00:48<00:05, 7765.21it/s] 90%|█████████ | 361865/400000 [00:48<00:04, 7878.57it/s] 91%|█████████ | 362708/400000 [00:48<00:04, 8035.13it/s] 91%|█████████ | 363515/400000 [00:48<00:04, 8027.17it/s] 91%|█████████ | 364320/400000 [00:48<00:04, 7547.56it/s] 91%|█████████▏| 365082/400000 [00:48<00:05, 6782.74it/s] 91%|█████████▏| 365780/400000 [00:48<00:05, 6576.13it/s] 92%|█████████▏| 366453/400000 [00:49<00:05, 6466.58it/s] 92%|█████████▏| 367149/400000 [00:49<00:04, 6606.60it/s] 92%|█████████▏| 367852/400000 [00:49<00:04, 6727.88it/s] 92%|█████████▏| 368663/400000 [00:49<00:04, 7090.38it/s] 92%|█████████▏| 369409/400000 [00:49<00:04, 7196.82it/s] 93%|█████████▎| 370264/400000 [00:49<00:03, 7553.44it/s] 93%|█████████▎| 371083/400000 [00:49<00:03, 7731.98it/s] 93%|█████████▎| 371936/400000 [00:49<00:03, 7954.11it/s] 93%|█████████▎| 372787/400000 [00:49<00:03, 8109.77it/s] 93%|█████████▎| 373628/400000 [00:49<00:03, 8194.97it/s] 94%|█████████▎| 374452/400000 [00:50<00:03, 8024.92it/s] 94%|█████████▍| 375259/400000 [00:50<00:03, 8036.11it/s] 94%|█████████▍| 376119/400000 [00:50<00:02, 8196.07it/s] 94%|█████████▍| 376978/400000 [00:50<00:02, 8308.22it/s] 94%|█████████▍| 377812/400000 [00:50<00:02, 7820.34it/s] 95%|█████████▍| 378602/400000 [00:50<00:02, 7383.24it/s] 95%|█████████▍| 379351/400000 [00:50<00:02, 7096.52it/s] 95%|█████████▌| 380071/400000 [00:50<00:02, 6962.92it/s] 95%|█████████▌| 380881/400000 [00:50<00:02, 7266.81it/s] 95%|█████████▌| 381721/400000 [00:50<00:02, 7571.91it/s] 96%|█████████▌| 382557/400000 [00:51<00:02, 7791.21it/s] 96%|█████████▌| 383345/400000 [00:51<00:02, 7802.98it/s] 96%|█████████▌| 384138/400000 [00:51<00:02, 7840.45it/s] 96%|█████████▌| 384927/400000 [00:51<00:01, 7779.77it/s] 96%|█████████▋| 385759/400000 [00:51<00:01, 7933.34it/s] 97%|█████████▋| 386605/400000 [00:51<00:01, 8082.22it/s] 97%|█████████▋| 387416/400000 [00:51<00:01, 7932.73it/s] 97%|█████████▋| 388281/400000 [00:51<00:01, 8133.52it/s] 97%|█████████▋| 389151/400000 [00:51<00:01, 8294.88it/s] 98%|█████████▊| 390019/400000 [00:51<00:01, 8406.57it/s] 98%|█████████▊| 390878/400000 [00:52<00:01, 8458.92it/s] 98%|█████████▊| 391726/400000 [00:52<00:00, 8320.21it/s] 98%|█████████▊| 392560/400000 [00:52<00:00, 8310.11it/s] 98%|█████████▊| 393433/400000 [00:52<00:00, 8430.01it/s] 99%|█████████▊| 394298/400000 [00:52<00:00, 8492.12it/s] 99%|█████████▉| 395169/400000 [00:52<00:00, 8556.19it/s] 99%|█████████▉| 396026/400000 [00:52<00:00, 8221.30it/s] 99%|█████████▉| 396852/400000 [00:52<00:00, 8136.89it/s] 99%|█████████▉| 397669/400000 [00:52<00:00, 7959.41it/s]100%|█████████▉| 398468/400000 [00:53<00:00, 7832.14it/s]100%|█████████▉| 399264/400000 [00:53<00:00, 7868.99it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7516.40it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fe44cadc128> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011093370563141436 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.012775814054801712 	 Accuracy: 45

  model saves at 45% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15747 out of table with 15667 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


### Running {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': 'dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': 'ztest/model_tch/MATCHZOO/BERT/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_tch.matchzoo_models.py', 'model': 'BERT', 'pretrained': 0, 'embedding_output_dim': 100, 'mode': 'bert-base-uncased', 'dropout_rate': 0.2}, 'data_pars': {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'train_batch_size': 4, 'test_batch_size': 1}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 5, 'epochs': 10, 'learning_rate': 5e-05, 'beta1': 0.9, 'beta2': 0.98, 'eps': 1e-08, 'warmup_steps': 6, 't_total': -1}, 'out_pars': {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'}} 'model_pars' 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text/ 

  Empty DataFrame
Columns: [date_run, model_uri, json, dataset_uri, metric, metric_name]
Index: [] 

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
RuntimeError: index out of range: Tried to access index 15747 out of table with 15667 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

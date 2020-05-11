
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fa24f382f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 05:12:15.972242
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 05:12:15.975575
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 05:12:15.978469
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 05:12:15.981468
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fa25b146438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 353799.2812
Epoch 2/10

1/1 [==============================] - 0s 105ms/step - loss: 254406.0469
Epoch 3/10

1/1 [==============================] - 0s 107ms/step - loss: 150673.6250
Epoch 4/10

1/1 [==============================] - 0s 112ms/step - loss: 81003.6172
Epoch 5/10

1/1 [==============================] - 0s 102ms/step - loss: 43713.3438
Epoch 6/10

1/1 [==============================] - 0s 96ms/step - loss: 25454.9473
Epoch 7/10

1/1 [==============================] - 0s 94ms/step - loss: 16233.6406
Epoch 8/10

1/1 [==============================] - 0s 97ms/step - loss: 11215.6904
Epoch 9/10

1/1 [==============================] - 0s 99ms/step - loss: 8258.5459
Epoch 10/10

1/1 [==============================] - 0s 99ms/step - loss: 6413.1333

  #### Inference Need return ypred, ytrue ######################### 
[[ 4.19271678e-01  1.42497087e+00 -2.91918963e-01  7.04447687e-01
   5.12358844e-01  8.88476968e-01 -2.45383084e-01  4.51086640e-01
   4.55058008e-01 -1.77995205e+00  7.42826402e-01  2.00250596e-01
  -9.49579656e-01  3.83398324e-01  5.31985879e-01 -1.81097090e-02
  -2.82774329e-01 -5.13227046e-01  7.37532735e-01  1.38801634e-02
  -9.88058090e-01 -7.32996523e-01  8.27228308e-01 -1.06670642e+00
  -3.08065504e-01 -1.38344979e+00  1.73724830e+00 -1.67030668e+00
  -4.87722963e-01  2.64353007e-01  1.09984887e+00 -1.18009061e-01
  -1.09317899e-03 -1.35927558e-01 -1.48310757e+00  2.25710332e-01
  -6.08911037e-01 -2.62950331e-01  1.41226029e+00 -1.26437873e-01
  -4.88663167e-01  3.02041560e-01  7.62976229e-01  1.47362396e-01
   1.39208806e+00 -1.46109784e+00  4.08850700e-01 -1.21237218e+00
   2.53369391e-01 -1.16075166e-02 -1.55236185e-01  5.56023359e-01
   7.07740784e-02  3.65854740e-01  9.19051528e-01 -7.98242211e-01
   6.95970058e-01  1.33955628e-02 -7.82076418e-02 -9.87480760e-01
  -3.97084244e-02  7.86590481e+00  8.58259487e+00  6.20417881e+00
   8.59401131e+00  7.87955952e+00  7.36895752e+00  1.03997116e+01
   7.95293713e+00  9.31208134e+00  8.68855286e+00  8.58191586e+00
   7.47570133e+00  8.43977737e+00  8.03917599e+00  7.83919954e+00
   7.76287270e+00  9.20439625e+00  8.13094711e+00  9.60405922e+00
   8.73370647e+00  8.21769142e+00  8.76449299e+00  9.44580173e+00
   9.19212532e+00  9.53784180e+00  9.32404423e+00  1.01584597e+01
   8.01135540e+00  7.70218039e+00  6.74138975e+00  9.35058117e+00
   1.02707033e+01  7.74239111e+00  9.28902245e+00  7.71285582e+00
   8.57964897e+00  9.32268047e+00  7.45108557e+00  6.43672705e+00
   6.74568415e+00  7.76759481e+00  7.99159718e+00  1.02320795e+01
   8.82378864e+00  7.41141939e+00  8.53484249e+00  8.44608974e+00
   8.29568672e+00  9.23449707e+00  8.34934425e+00  8.80101299e+00
   1.08489599e+01  8.69148159e+00  7.98350334e+00  6.56564522e+00
   8.17274952e+00  6.52333546e+00  7.00288582e+00  8.31488037e+00
  -7.91938782e-01 -7.65833437e-01  2.29746759e-01  8.96042824e-01
   1.22186732e+00  1.30577540e+00  8.98270965e-01  2.70097047e-01
   7.03926206e-01  1.46755695e+00  1.34248984e+00  8.15921664e-01
   5.34861267e-01 -1.60218611e-01 -7.39092767e-01 -1.02830780e+00
  -2.01563895e-01 -9.39447403e-01  8.66260588e-01 -1.21819031e+00
  -1.69971168e-01  1.31559992e+00  6.19694710e-01  6.54312432e-01
   1.44641709e+00 -7.44132876e-01  2.58459258e+00  1.35080472e-01
   3.52768242e-01 -6.72533989e-01 -5.73782325e-01  9.89688694e-01
  -1.78468257e-01 -1.50379300e-01 -7.42717683e-02 -9.44489062e-01
   7.75530398e-01  1.23013413e+00 -3.60767365e-01  3.09453845e-01
  -6.85568929e-01 -4.53284681e-01 -2.38485575e+00  2.54996538e-01
   1.13059759e-01  2.51095951e-01 -1.29780924e+00  1.55863810e+00
   1.53886485e+00  2.13953090e+00  6.76198006e-02 -9.95202780e-01
  -2.15866476e-01  1.39865065e+00 -1.01029062e+00  4.92631048e-01
  -2.94798315e-01  8.21029663e-01  3.62575054e-04 -4.76298928e-01
   1.84219933e+00  1.32896328e+00  3.08107018e-01  4.14573193e+00
   1.63729644e+00  2.13480234e+00  3.33221495e-01  6.63849711e-01
   1.58910275e-01  2.35300708e+00  2.65630424e-01  5.67077458e-01
   1.52428579e+00  6.43982351e-01  2.72704458e+00  1.48861372e+00
   3.25323761e-01  4.14566875e-01  5.67029536e-01  2.77434921e+00
   5.57667613e-01  2.09713840e+00  3.04832995e-01  1.03032041e+00
   7.94427633e-01  2.82327843e+00  6.42954588e-01  6.83090508e-01
   5.29665411e-01  9.49796319e-01  1.29309857e+00  6.19378805e-01
   1.16849458e+00  8.91445637e-01  4.38808918e-01  2.78132439e-01
   5.26738644e-01  1.58047104e+00  9.98641431e-01  4.67617691e-01
   8.25492501e-01  1.24216843e+00  3.48822296e-01  3.15515637e-01
   2.82182097e-01  1.34302950e+00  2.01375055e+00  3.37013364e-01
   6.75025702e-01  1.58443093e-01  2.51489782e+00  1.79586959e+00
   6.13804281e-01  1.55337548e+00  1.52601218e+00  3.24303508e-01
   1.99365437e+00  4.57190514e-01  1.02830434e+00  2.31447756e-01
   3.19492459e-01  7.85920143e+00  7.36406517e+00  8.79694271e+00
   7.87320852e+00  8.37512684e+00  8.36577034e+00  8.68760014e+00
   8.79360104e+00  8.69639111e+00  8.19107628e+00  8.72398186e+00
   7.32455397e+00  9.93743610e+00  7.30281734e+00  8.82838726e+00
   8.39801216e+00  9.49016857e+00  7.41614914e+00  9.04227924e+00
   8.89512348e+00  9.16152573e+00  8.25876522e+00  8.28876877e+00
   7.75795460e+00  8.17300129e+00  1.03424273e+01  7.88928366e+00
   7.95776653e+00  7.41517162e+00  7.69774818e+00  8.19991398e+00
   8.51737118e+00  7.19509220e+00  8.72609329e+00  8.66132450e+00
   9.42168713e+00  9.31041241e+00  9.09323883e+00  9.02075386e+00
   9.52489758e+00  6.74828339e+00  6.80056381e+00  7.63879299e+00
   9.93033600e+00  9.84367466e+00  7.16357374e+00  7.76004696e+00
   9.15394783e+00  8.17841244e+00  8.61355877e+00  8.49885654e+00
   8.49842644e+00  9.53967285e+00  8.84609604e+00  9.09696293e+00
   8.31313515e+00  8.78142929e+00  8.63911819e+00  8.35377216e+00
   8.98195028e-01  2.52894449e+00  3.66337061e-01  7.87281394e-02
   1.59435582e+00  2.08738208e-01  2.10348487e+00  8.81522775e-01
   2.64700651e-01  1.83800459e-01  1.27760863e+00  1.13780940e+00
   3.08760822e-01  1.59883213e+00  2.40703535e+00  6.85211062e-01
   9.66930449e-01  3.75733852e-01  5.50544679e-01  1.98628247e+00
   1.81265998e+00  1.18918538e+00  6.98678732e-01  8.10904086e-01
   1.38793242e+00  5.18136978e-01  1.48674726e+00  6.28490329e-01
   4.77260113e-01  9.78814065e-01  6.05905294e-01  3.72629106e-01
   6.11016393e-01  2.20158243e+00  2.99593627e-01  1.18827295e+00
   1.67858887e+00  1.54511929e+00  1.14746284e+00  1.25219584e+00
   4.20925617e-01  6.15675449e-01  7.16049612e-01  3.59514952e-01
   1.46722960e+00  3.68935585e-01  5.08862972e-01  4.21894670e-01
   1.53153276e+00  4.65163708e-01  2.12132740e+00  1.90110540e+00
   4.27159250e-01  1.88690424e-01  5.45970082e-01  1.40960670e+00
   7.77843177e-01  2.73807096e+00  1.74407387e+00  8.03105891e-01
  -7.03979588e+00  4.97949600e+00 -7.43447971e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 05:12:25.437832
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.5876
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 05:12:25.441495
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8787.05
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 05:12:25.444716
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.4874
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 05:12:25.448179
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -785.942
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140334748515744
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140333655731616
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140333655732120
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140333655318992
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140333655319496
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140333655320000

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fa23b7932b0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.687587
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.637561
grad_step = 000002, loss = 0.594025
grad_step = 000003, loss = 0.546115
grad_step = 000004, loss = 0.491614
grad_step = 000005, loss = 0.435298
grad_step = 000006, loss = 0.395005
grad_step = 000007, loss = 0.386670
grad_step = 000008, loss = 0.373178
grad_step = 000009, loss = 0.340881
grad_step = 000010, loss = 0.315683
grad_step = 000011, loss = 0.303076
grad_step = 000012, loss = 0.296475
grad_step = 000013, loss = 0.286735
grad_step = 000014, loss = 0.273397
grad_step = 000015, loss = 0.259387
grad_step = 000016, loss = 0.246253
grad_step = 000017, loss = 0.235009
grad_step = 000018, loss = 0.224779
grad_step = 000019, loss = 0.213455
grad_step = 000020, loss = 0.200912
grad_step = 000021, loss = 0.188945
grad_step = 000022, loss = 0.178818
grad_step = 000023, loss = 0.170263
grad_step = 000024, loss = 0.162348
grad_step = 000025, loss = 0.154323
grad_step = 000026, loss = 0.145994
grad_step = 000027, loss = 0.137654
grad_step = 000028, loss = 0.129759
grad_step = 000029, loss = 0.122452
grad_step = 000030, loss = 0.115513
grad_step = 000031, loss = 0.108675
grad_step = 000032, loss = 0.101964
grad_step = 000033, loss = 0.095650
grad_step = 000034, loss = 0.089987
grad_step = 000035, loss = 0.084948
grad_step = 000036, loss = 0.080210
grad_step = 000037, loss = 0.075583
grad_step = 000038, loss = 0.071118
grad_step = 000039, loss = 0.066808
grad_step = 000040, loss = 0.062704
grad_step = 000041, loss = 0.058930
grad_step = 000042, loss = 0.055410
grad_step = 000043, loss = 0.051997
grad_step = 000044, loss = 0.048739
grad_step = 000045, loss = 0.045731
grad_step = 000046, loss = 0.042980
grad_step = 000047, loss = 0.040444
grad_step = 000048, loss = 0.038024
grad_step = 000049, loss = 0.035675
grad_step = 000050, loss = 0.033477
grad_step = 000051, loss = 0.031448
grad_step = 000052, loss = 0.029521
grad_step = 000053, loss = 0.027659
grad_step = 000054, loss = 0.025870
grad_step = 000055, loss = 0.024187
grad_step = 000056, loss = 0.022640
grad_step = 000057, loss = 0.021193
grad_step = 000058, loss = 0.019813
grad_step = 000059, loss = 0.018520
grad_step = 000060, loss = 0.017317
grad_step = 000061, loss = 0.016186
grad_step = 000062, loss = 0.015095
grad_step = 000063, loss = 0.014037
grad_step = 000064, loss = 0.013057
grad_step = 000065, loss = 0.012159
grad_step = 000066, loss = 0.011312
grad_step = 000067, loss = 0.010511
grad_step = 000068, loss = 0.009766
grad_step = 000069, loss = 0.009090
grad_step = 000070, loss = 0.008463
grad_step = 000071, loss = 0.007863
grad_step = 000072, loss = 0.007304
grad_step = 000073, loss = 0.006792
grad_step = 000074, loss = 0.006322
grad_step = 000075, loss = 0.005881
grad_step = 000076, loss = 0.005471
grad_step = 000077, loss = 0.005102
grad_step = 000078, loss = 0.004768
grad_step = 000079, loss = 0.004461
grad_step = 000080, loss = 0.004180
grad_step = 000081, loss = 0.003926
grad_step = 000082, loss = 0.003694
grad_step = 000083, loss = 0.003482
grad_step = 000084, loss = 0.003290
grad_step = 000085, loss = 0.003116
grad_step = 000086, loss = 0.002962
grad_step = 000087, loss = 0.002824
grad_step = 000088, loss = 0.002701
grad_step = 000089, loss = 0.002593
grad_step = 000090, loss = 0.002498
grad_step = 000091, loss = 0.002414
grad_step = 000092, loss = 0.002340
grad_step = 000093, loss = 0.002275
grad_step = 000094, loss = 0.002219
grad_step = 000095, loss = 0.002171
grad_step = 000096, loss = 0.002129
grad_step = 000097, loss = 0.002093
grad_step = 000098, loss = 0.002064
grad_step = 000099, loss = 0.002040
grad_step = 000100, loss = 0.002019
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002002
grad_step = 000102, loss = 0.001988
grad_step = 000103, loss = 0.001976
grad_step = 000104, loss = 0.001965
grad_step = 000105, loss = 0.001957
grad_step = 000106, loss = 0.001951
grad_step = 000107, loss = 0.001945
grad_step = 000108, loss = 0.001940
grad_step = 000109, loss = 0.001936
grad_step = 000110, loss = 0.001933
grad_step = 000111, loss = 0.001929
grad_step = 000112, loss = 0.001926
grad_step = 000113, loss = 0.001923
grad_step = 000114, loss = 0.001920
grad_step = 000115, loss = 0.001917
grad_step = 000116, loss = 0.001913
grad_step = 000117, loss = 0.001910
grad_step = 000118, loss = 0.001907
grad_step = 000119, loss = 0.001903
grad_step = 000120, loss = 0.001899
grad_step = 000121, loss = 0.001895
grad_step = 000122, loss = 0.001891
grad_step = 000123, loss = 0.001887
grad_step = 000124, loss = 0.001883
grad_step = 000125, loss = 0.001878
grad_step = 000126, loss = 0.001874
grad_step = 000127, loss = 0.001869
grad_step = 000128, loss = 0.001865
grad_step = 000129, loss = 0.001860
grad_step = 000130, loss = 0.001856
grad_step = 000131, loss = 0.001851
grad_step = 000132, loss = 0.001846
grad_step = 000133, loss = 0.001841
grad_step = 000134, loss = 0.001837
grad_step = 000135, loss = 0.001832
grad_step = 000136, loss = 0.001827
grad_step = 000137, loss = 0.001823
grad_step = 000138, loss = 0.001818
grad_step = 000139, loss = 0.001813
grad_step = 000140, loss = 0.001809
grad_step = 000141, loss = 0.001804
grad_step = 000142, loss = 0.001799
grad_step = 000143, loss = 0.001795
grad_step = 000144, loss = 0.001790
grad_step = 000145, loss = 0.001786
grad_step = 000146, loss = 0.001782
grad_step = 000147, loss = 0.001777
grad_step = 000148, loss = 0.001773
grad_step = 000149, loss = 0.001769
grad_step = 000150, loss = 0.001764
grad_step = 000151, loss = 0.001760
grad_step = 000152, loss = 0.001756
grad_step = 000153, loss = 0.001751
grad_step = 000154, loss = 0.001747
grad_step = 000155, loss = 0.001743
grad_step = 000156, loss = 0.001738
grad_step = 000157, loss = 0.001734
grad_step = 000158, loss = 0.001729
grad_step = 000159, loss = 0.001725
grad_step = 000160, loss = 0.001720
grad_step = 000161, loss = 0.001716
grad_step = 000162, loss = 0.001710
grad_step = 000163, loss = 0.001706
grad_step = 000164, loss = 0.001701
grad_step = 000165, loss = 0.001696
grad_step = 000166, loss = 0.001693
grad_step = 000167, loss = 0.001691
grad_step = 000168, loss = 0.001690
grad_step = 000169, loss = 0.001688
grad_step = 000170, loss = 0.001684
grad_step = 000171, loss = 0.001675
grad_step = 000172, loss = 0.001665
grad_step = 000173, loss = 0.001656
grad_step = 000174, loss = 0.001650
grad_step = 000175, loss = 0.001647
grad_step = 000176, loss = 0.001644
grad_step = 000177, loss = 0.001643
grad_step = 000178, loss = 0.001641
grad_step = 000179, loss = 0.001639
grad_step = 000180, loss = 0.001634
grad_step = 000181, loss = 0.001628
grad_step = 000182, loss = 0.001617
grad_step = 000183, loss = 0.001604
grad_step = 000184, loss = 0.001591
grad_step = 000185, loss = 0.001583
grad_step = 000186, loss = 0.001578
grad_step = 000187, loss = 0.001575
grad_step = 000188, loss = 0.001575
grad_step = 000189, loss = 0.001576
grad_step = 000190, loss = 0.001579
grad_step = 000191, loss = 0.001577
grad_step = 000192, loss = 0.001571
grad_step = 000193, loss = 0.001556
grad_step = 000194, loss = 0.001538
grad_step = 000195, loss = 0.001522
grad_step = 000196, loss = 0.001514
grad_step = 000197, loss = 0.001518
grad_step = 000198, loss = 0.001521
grad_step = 000199, loss = 0.001513
grad_step = 000200, loss = 0.001498
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001485
grad_step = 000202, loss = 0.001485
grad_step = 000203, loss = 0.001489
grad_step = 000204, loss = 0.001489
grad_step = 000205, loss = 0.001481
grad_step = 000206, loss = 0.001478
grad_step = 000207, loss = 0.001497
grad_step = 000208, loss = 0.001535
grad_step = 000209, loss = 0.001519
grad_step = 000210, loss = 0.001453
grad_step = 000211, loss = 0.001441
grad_step = 000212, loss = 0.001478
grad_step = 000213, loss = 0.001465
grad_step = 000214, loss = 0.001424
grad_step = 000215, loss = 0.001430
grad_step = 000216, loss = 0.001451
grad_step = 000217, loss = 0.001543
grad_step = 000218, loss = 0.001513
grad_step = 000219, loss = 0.001723
grad_step = 000220, loss = 0.001855
grad_step = 000221, loss = 0.001497
grad_step = 000222, loss = 0.001462
grad_step = 000223, loss = 0.001668
grad_step = 000224, loss = 0.001471
grad_step = 000225, loss = 0.001465
grad_step = 000226, loss = 0.001548
grad_step = 000227, loss = 0.001407
grad_step = 000228, loss = 0.001507
grad_step = 000229, loss = 0.001468
grad_step = 000230, loss = 0.001386
grad_step = 000231, loss = 0.001495
grad_step = 000232, loss = 0.001406
grad_step = 000233, loss = 0.001393
grad_step = 000234, loss = 0.001454
grad_step = 000235, loss = 0.001370
grad_step = 000236, loss = 0.001387
grad_step = 000237, loss = 0.001412
grad_step = 000238, loss = 0.001350
grad_step = 000239, loss = 0.001373
grad_step = 000240, loss = 0.001377
grad_step = 000241, loss = 0.001341
grad_step = 000242, loss = 0.001365
grad_step = 000243, loss = 0.001353
grad_step = 000244, loss = 0.001330
grad_step = 000245, loss = 0.001348
grad_step = 000246, loss = 0.001333
grad_step = 000247, loss = 0.001321
grad_step = 000248, loss = 0.001335
grad_step = 000249, loss = 0.001322
grad_step = 000250, loss = 0.001317
grad_step = 000251, loss = 0.001335
grad_step = 000252, loss = 0.001355
grad_step = 000253, loss = 0.001430
grad_step = 000254, loss = 0.001554
grad_step = 000255, loss = 0.001474
grad_step = 000256, loss = 0.001319
grad_step = 000257, loss = 0.001381
grad_step = 000258, loss = 0.001433
grad_step = 000259, loss = 0.001325
grad_step = 000260, loss = 0.001368
grad_step = 000261, loss = 0.001379
grad_step = 000262, loss = 0.001323
grad_step = 000263, loss = 0.001342
grad_step = 000264, loss = 0.001346
grad_step = 000265, loss = 0.001315
grad_step = 000266, loss = 0.001319
grad_step = 000267, loss = 0.001333
grad_step = 000268, loss = 0.001302
grad_step = 000269, loss = 0.001299
grad_step = 000270, loss = 0.001324
grad_step = 000271, loss = 0.001298
grad_step = 000272, loss = 0.001284
grad_step = 000273, loss = 0.001305
grad_step = 000274, loss = 0.001297
grad_step = 000275, loss = 0.001284
grad_step = 000276, loss = 0.001287
grad_step = 000277, loss = 0.001282
grad_step = 000278, loss = 0.001286
grad_step = 000279, loss = 0.001313
grad_step = 000280, loss = 0.001284
grad_step = 000281, loss = 0.001273
grad_step = 000282, loss = 0.001282
grad_step = 000283, loss = 0.001278
grad_step = 000284, loss = 0.001264
grad_step = 000285, loss = 0.001274
grad_step = 000286, loss = 0.001269
grad_step = 000287, loss = 0.001263
grad_step = 000288, loss = 0.001264
grad_step = 000289, loss = 0.001260
grad_step = 000290, loss = 0.001255
grad_step = 000291, loss = 0.001258
grad_step = 000292, loss = 0.001255
grad_step = 000293, loss = 0.001249
grad_step = 000294, loss = 0.001250
grad_step = 000295, loss = 0.001248
grad_step = 000296, loss = 0.001244
grad_step = 000297, loss = 0.001244
grad_step = 000298, loss = 0.001243
grad_step = 000299, loss = 0.001239
grad_step = 000300, loss = 0.001239
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001238
grad_step = 000302, loss = 0.001235
grad_step = 000303, loss = 0.001234
grad_step = 000304, loss = 0.001236
grad_step = 000305, loss = 0.001235
grad_step = 000306, loss = 0.001235
grad_step = 000307, loss = 0.001238
grad_step = 000308, loss = 0.001241
grad_step = 000309, loss = 0.001244
grad_step = 000310, loss = 0.001247
grad_step = 000311, loss = 0.001248
grad_step = 000312, loss = 0.001240
grad_step = 000313, loss = 0.001229
grad_step = 000314, loss = 0.001217
grad_step = 000315, loss = 0.001207
grad_step = 000316, loss = 0.001204
grad_step = 000317, loss = 0.001207
grad_step = 000318, loss = 0.001211
grad_step = 000319, loss = 0.001213
grad_step = 000320, loss = 0.001215
grad_step = 000321, loss = 0.001214
grad_step = 000322, loss = 0.001210
grad_step = 000323, loss = 0.001205
grad_step = 000324, loss = 0.001199
grad_step = 000325, loss = 0.001193
grad_step = 000326, loss = 0.001187
grad_step = 000327, loss = 0.001181
grad_step = 000328, loss = 0.001177
grad_step = 000329, loss = 0.001173
grad_step = 000330, loss = 0.001170
grad_step = 000331, loss = 0.001167
grad_step = 000332, loss = 0.001165
grad_step = 000333, loss = 0.001163
grad_step = 000334, loss = 0.001165
grad_step = 000335, loss = 0.001176
grad_step = 000336, loss = 0.001213
grad_step = 000337, loss = 0.001326
grad_step = 000338, loss = 0.001513
grad_step = 000339, loss = 0.001722
grad_step = 000340, loss = 0.001502
grad_step = 000341, loss = 0.001225
grad_step = 000342, loss = 0.001259
grad_step = 000343, loss = 0.001376
grad_step = 000344, loss = 0.001329
grad_step = 000345, loss = 0.001200
grad_step = 000346, loss = 0.001264
grad_step = 000347, loss = 0.001328
grad_step = 000348, loss = 0.001172
grad_step = 000349, loss = 0.001206
grad_step = 000350, loss = 0.001283
grad_step = 000351, loss = 0.001163
grad_step = 000352, loss = 0.001171
grad_step = 000353, loss = 0.001238
grad_step = 000354, loss = 0.001157
grad_step = 000355, loss = 0.001154
grad_step = 000356, loss = 0.001196
grad_step = 000357, loss = 0.001139
grad_step = 000358, loss = 0.001143
grad_step = 000359, loss = 0.001174
grad_step = 000360, loss = 0.001129
grad_step = 000361, loss = 0.001120
grad_step = 000362, loss = 0.001152
grad_step = 000363, loss = 0.001130
grad_step = 000364, loss = 0.001113
grad_step = 000365, loss = 0.001133
grad_step = 000366, loss = 0.001121
grad_step = 000367, loss = 0.001102
grad_step = 000368, loss = 0.001115
grad_step = 000369, loss = 0.001117
grad_step = 000370, loss = 0.001101
grad_step = 000371, loss = 0.001105
grad_step = 000372, loss = 0.001112
grad_step = 000373, loss = 0.001101
grad_step = 000374, loss = 0.001098
grad_step = 000375, loss = 0.001105
grad_step = 000376, loss = 0.001101
grad_step = 000377, loss = 0.001095
grad_step = 000378, loss = 0.001100
grad_step = 000379, loss = 0.001101
grad_step = 000380, loss = 0.001097
grad_step = 000381, loss = 0.001101
grad_step = 000382, loss = 0.001108
grad_step = 000383, loss = 0.001108
grad_step = 000384, loss = 0.001110
grad_step = 000385, loss = 0.001112
grad_step = 000386, loss = 0.001106
grad_step = 000387, loss = 0.001095
grad_step = 000388, loss = 0.001087
grad_step = 000389, loss = 0.001078
grad_step = 000390, loss = 0.001067
grad_step = 000391, loss = 0.001060
grad_step = 000392, loss = 0.001059
grad_step = 000393, loss = 0.001058
grad_step = 000394, loss = 0.001058
grad_step = 000395, loss = 0.001061
grad_step = 000396, loss = 0.001066
grad_step = 000397, loss = 0.001069
grad_step = 000398, loss = 0.001073
grad_step = 000399, loss = 0.001080
grad_step = 000400, loss = 0.001087
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001091
grad_step = 000402, loss = 0.001093
grad_step = 000403, loss = 0.001086
grad_step = 000404, loss = 0.001072
grad_step = 000405, loss = 0.001053
grad_step = 000406, loss = 0.001037
grad_step = 000407, loss = 0.001027
grad_step = 000408, loss = 0.001024
grad_step = 000409, loss = 0.001027
grad_step = 000410, loss = 0.001034
grad_step = 000411, loss = 0.001040
grad_step = 000412, loss = 0.001044
grad_step = 000413, loss = 0.001045
grad_step = 000414, loss = 0.001045
grad_step = 000415, loss = 0.001043
grad_step = 000416, loss = 0.001040
grad_step = 000417, loss = 0.001037
grad_step = 000418, loss = 0.001034
grad_step = 000419, loss = 0.001031
grad_step = 000420, loss = 0.001027
grad_step = 000421, loss = 0.001023
grad_step = 000422, loss = 0.001019
grad_step = 000423, loss = 0.001016
grad_step = 000424, loss = 0.001013
grad_step = 000425, loss = 0.001011
grad_step = 000426, loss = 0.001009
grad_step = 000427, loss = 0.001007
grad_step = 000428, loss = 0.001007
grad_step = 000429, loss = 0.001007
grad_step = 000430, loss = 0.001008
grad_step = 000431, loss = 0.001013
grad_step = 000432, loss = 0.001022
grad_step = 000433, loss = 0.001039
grad_step = 000434, loss = 0.001068
grad_step = 000435, loss = 0.001115
grad_step = 000436, loss = 0.001157
grad_step = 000437, loss = 0.001200
grad_step = 000438, loss = 0.001152
grad_step = 000439, loss = 0.001048
grad_step = 000440, loss = 0.000995
grad_step = 000441, loss = 0.001037
grad_step = 000442, loss = 0.001088
grad_step = 000443, loss = 0.001095
grad_step = 000444, loss = 0.001103
grad_step = 000445, loss = 0.001072
grad_step = 000446, loss = 0.001016
grad_step = 000447, loss = 0.000997
grad_step = 000448, loss = 0.000995
grad_step = 000449, loss = 0.001005
grad_step = 000450, loss = 0.001033
grad_step = 000451, loss = 0.001038
grad_step = 000452, loss = 0.001025
grad_step = 000453, loss = 0.001011
grad_step = 000454, loss = 0.000989
grad_step = 000455, loss = 0.000977
grad_step = 000456, loss = 0.000984
grad_step = 000457, loss = 0.000990
grad_step = 000458, loss = 0.000997
grad_step = 000459, loss = 0.001007
grad_step = 000460, loss = 0.001006
grad_step = 000461, loss = 0.001002
grad_step = 000462, loss = 0.000996
grad_step = 000463, loss = 0.000984
grad_step = 000464, loss = 0.000976
grad_step = 000465, loss = 0.000973
grad_step = 000466, loss = 0.000969
grad_step = 000467, loss = 0.000970
grad_step = 000468, loss = 0.000974
grad_step = 000469, loss = 0.000975
grad_step = 000470, loss = 0.000977
grad_step = 000471, loss = 0.000979
grad_step = 000472, loss = 0.000979
grad_step = 000473, loss = 0.000978
grad_step = 000474, loss = 0.000978
grad_step = 000475, loss = 0.000976
grad_step = 000476, loss = 0.000976
grad_step = 000477, loss = 0.000976
grad_step = 000478, loss = 0.000976
grad_step = 000479, loss = 0.000976
grad_step = 000480, loss = 0.000978
grad_step = 000481, loss = 0.000981
grad_step = 000482, loss = 0.000986
grad_step = 000483, loss = 0.000994
grad_step = 000484, loss = 0.001006
grad_step = 000485, loss = 0.001024
grad_step = 000486, loss = 0.001048
grad_step = 000487, loss = 0.001079
grad_step = 000488, loss = 0.001109
grad_step = 000489, loss = 0.001133
grad_step = 000490, loss = 0.001131
grad_step = 000491, loss = 0.001099
grad_step = 000492, loss = 0.001040
grad_step = 000493, loss = 0.000982
grad_step = 000494, loss = 0.000952
grad_step = 000495, loss = 0.000960
grad_step = 000496, loss = 0.000991
grad_step = 000497, loss = 0.001016
grad_step = 000498, loss = 0.001018
grad_step = 000499, loss = 0.000996
grad_step = 000500, loss = 0.000967
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000948
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

  date_run                              2020-05-11 05:12:47.563429
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.237441
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 05:12:47.569358
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.137765
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 05:12:47.576443
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.141156
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 05:12:47.582469
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.09338
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
0   2020-05-11 05:12:15.972242  ...    mean_absolute_error
1   2020-05-11 05:12:15.975575  ...     mean_squared_error
2   2020-05-11 05:12:15.978469  ...  median_absolute_error
3   2020-05-11 05:12:15.981468  ...               r2_score
4   2020-05-11 05:12:25.437832  ...    mean_absolute_error
5   2020-05-11 05:12:25.441495  ...     mean_squared_error
6   2020-05-11 05:12:25.444716  ...  median_absolute_error
7   2020-05-11 05:12:25.448179  ...               r2_score
8   2020-05-11 05:12:47.563429  ...    mean_absolute_error
9   2020-05-11 05:12:47.569358  ...     mean_squared_error
10  2020-05-11 05:12:47.576443  ...  median_absolute_error
11  2020-05-11 05:12:47.582469  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 41%|████      | 4038656/9912422 [00:00<00:00, 40343987.39it/s]9920512it [00:00, 37272369.36it/s]                             
0it [00:00, ?it/s]32768it [00:00, 675541.67it/s]
0it [00:00, ?it/s]  5%|▌         | 90112/1648877 [00:00<00:01, 894033.48it/s]1654784it [00:00, 12228354.28it/s]                         
0it [00:00, ?it/s]8192it [00:00, 193611.99it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f945a8a5ba8> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f93f7ff9c18> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f945a868e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f93f7ff9e10> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f945a8a5ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f940d262dd8> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f945a868e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f94013125c0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f945a8a5ba8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f94013125c0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f945a868e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f977c0bb208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=59088696e355687326cebe7960588f86ef4c73c771a8a09eec04d004fd319dc9
  Stored in directory: /tmp/pip-ephem-wheel-cache-gptyk643/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f977222a048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2285568/17464789 [==>...........................] - ETA: 0s
 9060352/17464789 [==============>...............] - ETA: 0s
12132352/17464789 [===================>..........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 05:14:12.038976: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 05:14:12.044216: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-11 05:14:12.044675: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56116856b450 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 05:14:12.044926: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.9580 - accuracy: 0.4810
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7663 - accuracy: 0.4935 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7535 - accuracy: 0.4943
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7855 - accuracy: 0.4922
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7096 - accuracy: 0.4972
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6601 - accuracy: 0.5004
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6436 - accuracy: 0.5015
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6768 - accuracy: 0.4993
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6697 - accuracy: 0.4998
11000/25000 [============>.................] - ETA: 4s - loss: 7.6527 - accuracy: 0.5009
12000/25000 [=============>................] - ETA: 3s - loss: 7.6730 - accuracy: 0.4996
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6749 - accuracy: 0.4995
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6327 - accuracy: 0.5022
15000/25000 [=================>............] - ETA: 2s - loss: 7.6216 - accuracy: 0.5029
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6110 - accuracy: 0.5036
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6251 - accuracy: 0.5027
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6291 - accuracy: 0.5024
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6577 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6590 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6688 - accuracy: 0.4999
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 9s 357us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 05:14:27.460190
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 05:14:27.460190  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 05:14:33.175073: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 05:14:33.179630: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-11 05:14:33.179750: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560c5cbe2a90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 05:14:33.179764: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fe0bdaf4d30> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.0227 - crf_viterbi_accuracy: 0.3333 - val_loss: 0.9939 - val_crf_viterbi_accuracy: 0.2800

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fe0b2e9bf60> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.4366 - accuracy: 0.5150
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5900 - accuracy: 0.5050 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5746 - accuracy: 0.5060
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5555 - accuracy: 0.5073
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5378 - accuracy: 0.5084
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5695 - accuracy: 0.5063
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5812 - accuracy: 0.5056
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5957 - accuracy: 0.5046
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5900 - accuracy: 0.5050
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5884 - accuracy: 0.5051
11000/25000 [============>.................] - ETA: 4s - loss: 7.6276 - accuracy: 0.5025
12000/25000 [=============>................] - ETA: 4s - loss: 7.6002 - accuracy: 0.5043
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6135 - accuracy: 0.5035
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6141 - accuracy: 0.5034
15000/25000 [=================>............] - ETA: 3s - loss: 7.6349 - accuracy: 0.5021
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6446 - accuracy: 0.5014
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6576 - accuracy: 0.5006
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6573 - accuracy: 0.5006
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6844 - accuracy: 0.4988
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6804 - accuracy: 0.4991
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6674 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6541 - accuracy: 0.5008
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 9s 362us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fe06ec47320> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:05<173:58:20, 1.38kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:06<122:00:34, 1.96kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:06<85:26:44, 2.80kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:06<59:46:44, 4.00kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:06<41:42:55, 5.72kB/s].vector_cache/glove.6B.zip:   1%|          | 9.68M/862M [00:06<28:59:47, 8.17kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:06<20:13:20, 11.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.6M/862M [00:06<14:03:46, 16.7kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.3M/862M [00:06<9:46:44, 23.8kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 30.3M/862M [00:07<6:47:54, 34.0kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.1M/862M [00:07<4:43:38, 48.5kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.2M/862M [00:07<3:17:23, 69.3kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.5M/862M [00:07<2:17:44, 98.9kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.1M/862M [00:07<1:35:50, 141kB/s] .vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:07<1:07:34, 200kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:09<48:58, 274kB/s]  .vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:09<36:09, 372kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:09<25:40, 523kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:11<20:28, 653kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:11<15:53, 841kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:11<11:29, 1.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:13<10:50, 1.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:13<10:18, 1.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:13<07:49, 1.70MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.4M/862M [00:14<05:37, 2.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:15<12:14, 1.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:15<09:56, 1.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:15<07:14, 1.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:17<08:08, 1.62MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:17<08:23, 1.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:17<06:25, 2.05MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.1M/862M [00:17<04:42, 2.79MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:19<08:15, 1.59MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:19<07:07, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.3M/862M [00:19<05:19, 2.46MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:21<06:46, 1.92MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:21<07:24, 1.76MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:21<05:51, 2.22MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:21<04:15, 3.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:23<1:48:35, 119kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:23<1:17:17, 168kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:23<54:17, 238kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:25<40:55, 315kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:25<31:16, 412kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:25<22:31, 571kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:27<17:47, 721kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:27<13:47, 930kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:27<09:54, 1.29MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:29<09:54, 1.29MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:29<09:32, 1.34MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:29<07:20, 1.73MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:31<07:10, 1.77MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:31<06:27, 1.96MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:31<04:50, 2.61MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:33<06:17, 2.01MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:33<07:00, 1.80MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:33<05:32, 2.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:35<05:53, 2.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:35<05:24, 2.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:35<04:03, 3.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:37<05:45, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:37<05:18, 2.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:37<04:01, 3.09MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:39<05:45, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:39<05:17, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:39<03:57, 3.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:41<05:42, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:41<05:15, 2.35MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:41<03:55, 3.13MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:43<05:40, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:43<05:13, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:43<03:58, 3.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:45<05:39, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:45<06:27, 1.89MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:45<05:08, 2.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:47<05:33, 2.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:47<05:07, 2.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:47<03:53, 3.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:49<05:32, 2.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:49<05:07, 2.36MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:49<03:53, 3.09MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:50<05:32, 2.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:51<05:12, 2.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:51<03:58, 3.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:52<05:22, 2.22MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:53<06:25, 1.86MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:53<05:03, 2.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:53<03:40, 3.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:54<09:29, 1.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:55<07:45, 1.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:55<05:49, 2.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:56<06:35, 1.79MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:57<07:16, 1.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:57<05:44, 2.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:57<04:09, 2.82MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:58<14:01, 836kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:59<11:07, 1.05MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:59<08:05, 1.44MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:00<08:12, 1.42MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:01<06:49, 1.71MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:01<05:04, 2.29MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:01<03:43, 3.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:02<11:42, 989kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:02<09:17, 1.25MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:03<06:45, 1.71MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:03<04:53, 2.35MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:04<13:32, 851kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:05<12:02, 955kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:05<08:58, 1.28MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:05<06:25, 1.78MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:06<08:57, 1.28MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:06<07:33, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:07<05:35, 2.04MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:08<06:22, 1.79MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:08<07:00, 1.62MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:09<05:32, 2.05MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:09<03:59, 2.83MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:10<14:52, 760kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:10<11:41, 967kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:11<08:26, 1.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:12<08:18, 1.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:12<08:19, 1.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:13<06:27, 1.74MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<04:38, 2.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:14<15:11, 735kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:14<11:52, 940kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:15<08:33, 1.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:16<08:23, 1.32MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:16<06:55, 1.60MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:17<05:12, 2.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:18<05:59, 1.84MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:18<06:39, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:19<05:16, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:19<03:48, 2.88MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:20<14:22, 762kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:20<11:16, 971kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:21<08:11, 1.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:22<08:04, 1.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:22<08:05, 1.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:23<06:10, 1.76MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:23<04:27, 2.43MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:24<08:14, 1.31MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:24<06:58, 1.55MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:25<05:10, 2.09MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:26<05:56, 1.81MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:26<06:33, 1.64MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:27<05:11, 2.07MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:27<03:44, 2.86MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:28<14:03, 759kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:28<11:03, 965kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:28<08:01, 1.33MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:30<07:54, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:30<07:54, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:30<06:07, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:31<04:24, 2.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:32<15:50, 666kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:32<12:15, 859kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:32<08:51, 1.19MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:34<08:27, 1.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:34<08:16, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:34<06:22, 1.64MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<04:34, 2.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:36<14:17, 728kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:36<11:09, 932kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:36<08:04, 1.28MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:38<07:53, 1.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:38<07:51, 1.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:38<05:59, 1.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:39<04:30, 2.28MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:40<05:53, 1.74MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:40<06:47, 1.51MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:40<05:17, 1.93MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:41<03:51, 2.65MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:42<06:09, 1.65MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:42<06:48, 1.50MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:42<05:22, 1.89MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:43<03:53, 2.61MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:44<09:18, 1.09MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:44<08:55, 1.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:44<06:44, 1.50MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:44<04:51, 2.07MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:46<06:54, 1.46MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:46<07:04, 1.42MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:46<05:31, 1.82MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:47<04:00, 2.50MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:48<10:13, 976kB/s] .vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:48<09:23, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:48<07:07, 1.40MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<05:05, 1.95MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:50<11:20, 875kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:50<10:08, 977kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:50<07:34, 1.31MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<05:23, 1.83MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<05:31, 1.78MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:52<7:09:43, 22.9kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:52<5:01:40, 32.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:52<3:30:53, 46.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:54<2:29:17, 65.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:54<1:46:33, 91.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:54<1:15:01, 130kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:54<52:23, 185kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:56<46:51, 207kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:56<35:04, 277kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:56<25:00, 388kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:56<17:41, 547kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:58<14:38, 658kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:58<12:13, 789kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:58<09:02, 1.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:58<06:24, 1.50MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:00<13:04, 732kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:00<11:06, 861kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:00<08:14, 1.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<05:51, 1.62MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:02<12:45, 745kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:02<10:52, 874kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:02<08:01, 1.18MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<05:45, 1.64MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:04<07:33, 1.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:04<07:13, 1.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:04<05:31, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:04<03:58, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:06<17:38, 531kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:06<14:10, 660kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:06<10:22, 901kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:06<07:20, 1.27MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:08<18:32, 501kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:08<15:16, 609kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:08<11:13, 826kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:08<07:58, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:10<11:39, 792kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:10<10:18, 894kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:10<07:45, 1.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<05:31, 1.66MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:12<10:07, 905kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:12<09:15, 989kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:12<06:57, 1.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:12<04:58, 1.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:14<07:35, 1.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:14<07:27, 1.22MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:14<05:40, 1.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:14<04:05, 2.21MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:16<06:14, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:16<06:30, 1.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:16<05:00, 1.80MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<03:35, 2.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:18<08:43, 1.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:18<08:11, 1.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:18<06:15, 1.43MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:18<04:29, 1.98MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:20<09:20, 950kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:20<08:13, 1.08MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:20<06:10, 1.44MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<04:24, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:22<19:29, 452kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:22<15:22, 573kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:22<11:10, 787kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<07:53, 1.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:24<31:49, 275kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:24<24:24, 358kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:24<17:30, 498kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:24<12:18, 706kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:26<13:39, 635kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:26<11:40, 742kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:26<08:41, 997kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:26<06:10, 1.39MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:28<10:19, 833kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:28<09:20, 921kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:28<06:58, 1.23MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:28<05:00, 1.71MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:30<06:19, 1.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:30<06:20, 1.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:30<04:54, 1.73MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<03:33, 2.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:32<08:42, 972kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:32<08:10, 1.03MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:32<06:08, 1.38MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:32<04:24, 1.91MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:34<05:58, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:34<06:08, 1.37MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:34<04:42, 1.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<03:23, 2.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:36<06:16, 1.33MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:36<06:01, 1.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:36<04:36, 1.80MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:36<03:19, 2.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:38<19:10, 431kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:38<15:26, 535kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:38<11:18, 729kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<07:59, 1.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:40<10:38, 770kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:40<09:21, 874kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:40<06:58, 1.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<04:58, 1.63MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:42<06:27, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:42<06:25, 1.26MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:42<04:53, 1.66MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:42<03:31, 2.29MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:43<05:51, 1.37MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:44<05:58, 1.35MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:44<04:34, 1.76MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:44<03:21, 2.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:45<04:25, 1.80MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:46<04:57, 1.61MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:46<03:51, 2.06MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:46<02:48, 2.82MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:47<04:51, 1.63MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:48<05:17, 1.50MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:48<04:05, 1.93MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:48<02:59, 2.62MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:49<04:21, 1.80MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:50<04:31, 1.73MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:50<03:32, 2.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<02:34, 3.02MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:51<17:06, 454kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:52<13:52, 560kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:52<10:05, 769kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:52<07:10, 1.08MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:53<07:23, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:54<06:37, 1.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:54<04:55, 1.56MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:54<03:33, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:55<05:31, 1.38MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:56<05:18, 1.44MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:56<04:01, 1.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:56<02:54, 2.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:57<06:05, 1.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:57<05:41, 1.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:58<04:19, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:58<03:07, 2.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:59<21:38, 346kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:59<16:58, 441kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:00<12:16, 610kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:00<08:39, 860kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:01<08:47, 844kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:01<07:54, 939kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:02<05:57, 1.24MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:02<04:15, 1.73MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:03<08:07, 906kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:03<07:24, 994kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:04<05:36, 1.31MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<04:01, 1.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:05<07:54, 922kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:05<06:51, 1.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:06<05:04, 1.43MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<03:36, 2.00MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:07<09:53, 729kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:07<08:14, 876kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:07<06:01, 1.19MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:08<04:17, 1.67MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:09<07:30, 953kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:09<06:55, 1.03MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:09<05:13, 1.37MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:10<03:47, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:11<04:30, 1.57MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:11<04:49, 1.46MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:11<03:47, 1.86MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<02:44, 2.56MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:13<07:28, 937kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:13<06:57, 1.01MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:13<05:16, 1.33MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:14<03:46, 1.85MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:15<08:03, 861kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:15<07:11, 964kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:15<05:25, 1.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:16<03:51, 1.78MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:17<08:42, 789kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:17<07:13, 951kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:17<05:17, 1.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<03:51, 1.77MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:19<04:41, 1.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:19<04:49, 1.41MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:19<03:44, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:20<02:41, 2.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:21<10:07, 665kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:21<08:31, 789kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:21<06:16, 1.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:21<04:29, 1.49MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:23<05:18, 1.26MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:23<05:08, 1.30MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:23<03:56, 1.69MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<02:49, 2.33MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:25<15:57, 413kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:25<12:34, 524kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:25<09:05, 724kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<06:25, 1.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:27<07:33, 863kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:27<06:42, 974kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:27<05:01, 1.29MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:27<03:35, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:29<15:11, 425kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:29<11:37, 556kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:29<08:21, 770kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:31<06:57, 918kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:31<06:10, 1.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:31<04:36, 1.38MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<03:17, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:33<05:39, 1.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:33<05:15, 1.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:33<03:57, 1.59MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<02:51, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:35<04:17, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:35<04:17, 1.46MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:35<03:16, 1.91MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:35<02:21, 2.64MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:37<05:47, 1.07MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:37<05:19, 1.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:37<04:01, 1.53MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:39<03:47, 1.61MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:39<03:54, 1.57MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:39<03:00, 2.03MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:39<02:10, 2.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:41<05:14, 1.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:41<04:53, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:41<03:41, 1.63MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:41<02:41, 2.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:43<03:45, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:43<03:51, 1.55MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:43<02:59, 1.99MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:45<03:02, 1.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:45<03:24, 1.73MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:45<02:38, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<01:54, 3.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:47<04:21, 1.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:47<04:14, 1.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:47<03:13, 1.81MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<02:18, 2.50MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:49<04:58, 1.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:49<04:42, 1.22MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:49<03:32, 1.62MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:49<02:32, 2.24MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:51<04:24, 1.29MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:51<03:52, 1.47MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:51<02:52, 1.98MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:53<03:04, 1.83MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:53<03:10, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:53<02:26, 2.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:53<01:45, 3.17MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:54<10:54, 510kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:55<08:38, 643kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:55<06:16, 885kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:55<04:24, 1.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:56<23:22, 235kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:57<17:21, 317kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:57<12:22, 443kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:58<09:29, 572kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:59<07:38, 710kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:59<05:34, 972kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:00<04:46, 1.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:01<04:18, 1.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:01<03:15, 1.64MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:02<03:09, 1.68MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:02<03:10, 1.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:03<02:27, 2.14MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:04<02:35, 2.02MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:04<02:45, 1.89MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:05<02:07, 2.44MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:05<01:32, 3.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:06<06:04, 848kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:06<05:13, 985kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:07<03:51, 1.33MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:07<02:45, 1.85MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:08<04:14, 1.20MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:08<03:55, 1.29MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:08<02:56, 1.72MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:09<02:06, 2.38MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:10<04:21, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:10<04:00, 1.25MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:10<03:01, 1.65MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:12<02:55, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:12<02:59, 1.66MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:12<02:18, 2.14MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:14<02:25, 2.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:14<02:34, 1.89MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:14<02:01, 2.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:16<02:12, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:16<02:26, 1.97MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:16<01:53, 2.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<01:24, 3.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:18<02:40, 1.78MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:18<02:43, 1.74MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:18<02:06, 2.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<01:30, 3.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:20<06:34, 710kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:20<05:27, 856kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:20<04:01, 1.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:22<03:33, 1.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:22<03:19, 1.38MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:22<02:30, 1.83MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<01:47, 2.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:24<04:39, 975kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:24<04:04, 1.11MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:24<03:01, 1.49MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:24<02:11, 2.05MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:26<03:05, 1.44MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:26<02:58, 1.50MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:26<02:16, 1.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:28<02:19, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:28<02:26, 1.80MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:28<01:54, 2.30MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:30<02:02, 2.12MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:30<02:13, 1.95MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:30<01:44, 2.47MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:32<01:55, 2.22MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:32<02:09, 1.97MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:32<01:42, 2.49MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:34<01:53, 2.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:34<02:06, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:34<01:40, 2.50MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:36<01:50, 2.23MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:36<02:04, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:36<01:38, 2.51MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:38<01:48, 2.24MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:38<02:01, 1.99MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:38<01:34, 2.56MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<01:08, 3.49MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:40<03:51, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:40<03:18, 1.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:40<02:33, 1.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:40<01:51, 2.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:42<02:18, 1.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:42<02:19, 1.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:42<01:46, 2.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:42<01:18, 2.97MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:44<02:19, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:44<02:19, 1.65MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:44<01:56, 1.98MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:44<01:42, 2.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:44<01:32, 2.49MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:44<01:24, 2.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:44<01:19, 2.88MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:45<01:12, 3.15MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:45<01:09, 3.26MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:45<01:08, 3.33MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:46<05:05, 742kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:46<04:24, 856kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:46<03:25, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:46<02:42, 1.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:46<02:08, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:46<01:47, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:46<01:33, 2.40MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:46<01:22, 2.71MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:47<01:16, 2.93MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:47<01:10, 3.19MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:47<01:07, 3.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:48<04:52, 761kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:48<04:10, 890kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:48<03:13, 1.15MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:48<02:32, 1.45MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:48<02:04, 1.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:48<01:43, 2.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:48<01:29, 2.45MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:49<01:19, 2.76MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:49<01:10, 3.12MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:50<03:41, 986kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:50<03:19, 1.09MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:50<02:38, 1.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:50<02:07, 1.70MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:50<01:46, 2.04MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:50<01:30, 2.38MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:50<01:18, 2.75MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<01:11, 3.01MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:51<01:06, 3.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:52<03:51, 925kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:52<03:26, 1.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:52<02:42, 1.31MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:52<02:10, 1.64MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:52<01:47, 1.99MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:52<01:30, 2.34MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:52<01:19, 2.66MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:52<01:10, 3.00MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:53<01:05, 3.24MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:53<04:14, 826kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:54<04:24, 795kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:54<03:28, 1.01MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:54<02:41, 1.30MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:54<02:08, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:54<01:42, 2.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:54<01:29, 2.32MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:54<01:17, 2.67MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:55<01:07, 3.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:55<03:17, 1.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:56<02:58, 1.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:56<02:22, 1.45MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:56<01:51, 1.83MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:56<01:32, 2.21MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:56<01:19, 2.58MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<01:09, 2.93MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<01:02, 3.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:57<02:31, 1.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:58<03:02, 1.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:58<02:26, 1.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:58<01:59, 1.68MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:58<01:34, 2.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:58<01:23, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:58<01:09, 2.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:58<01:04, 3.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:58<00:59, 3.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:58<00:51, 3.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:59<09:05, 363kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:00<07:31, 439kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:00<05:35, 588kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:00<04:05, 803kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:00<03:04, 1.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:00<02:20, 1.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:00<01:49, 1.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:00<01:27, 2.21MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [05:01<04:29, 718kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [05:01<04:04, 791kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:02<03:07, 1.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:02<02:21, 1.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:02<01:49, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:02<01:26, 2.22MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<01:09, 2.73MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:03<03:02, 1.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:03<02:59, 1.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:04<02:20, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:04<01:47, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:04<01:22, 2.26MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:04<01:05, 2.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:04<00:54, 3.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:05<03:03, 1.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:05<03:40, 841kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:06<02:56, 1.05MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:06<02:11, 1.40MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:06<01:37, 1.88MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<01:15, 2.43MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:07<02:21, 1.28MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:07<02:53, 1.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:08<02:20, 1.29MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:08<01:44, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:08<01:19, 2.25MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:08<01:00, 2.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:09<04:01, 735kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:09<03:47, 780kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:10<02:54, 1.01MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:10<02:07, 1.38MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:10<01:35, 1.84MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:10<01:10, 2.47MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:11<02:59, 967kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:11<02:58, 970kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:11<02:17, 1.25MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:12<01:41, 1.69MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:12<01:14, 2.29MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:13<02:30, 1.12MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:13<03:25, 823kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:13<02:46, 1.01MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:14<02:01, 1.38MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:14<01:28, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:14<01:06, 2.49MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:15<03:18, 833kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:15<02:58, 921kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:15<02:13, 1.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:16<01:36, 1.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:16<01:11, 2.26MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:17<02:09, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:17<02:42, 992kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:17<02:11, 1.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:18<01:36, 1.65MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:18<01:09, 2.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:19<03:03, 855kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:19<03:12, 813kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:19<02:30, 1.04MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:20<01:48, 1.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<01:17, 1.96MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:21<27:40, 91.8kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:21<20:15, 125kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:21<14:22, 176kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<10:01, 250kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:23<07:29, 330kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:23<06:03, 408kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:23<04:26, 555kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:23<03:07, 778kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:25<02:45, 873kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:25<02:39, 902kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:25<02:02, 1.17MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<01:27, 1.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:27<01:43, 1.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:27<01:50, 1.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:27<01:25, 1.63MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<01:01, 2.25MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<00:45, 3.00MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:29<25:46, 87.9kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:29<18:40, 121kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:29<13:11, 171kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<09:08, 243kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:31<07:00, 313kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:31<05:26, 403kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:31<03:54, 559kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:31<02:43, 789kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:33<02:36, 816kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:33<02:24, 882kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:33<01:49, 1.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:33<01:17, 1.60MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:35<01:36, 1.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:35<01:41, 1.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:35<01:17, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:35<00:56, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:37<01:09, 1.72MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:37<01:16, 1.56MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:37<01:00, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<00:43, 2.66MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:39<01:27, 1.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:39<01:31, 1.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:39<01:10, 1.62MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<00:50, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:41<01:19, 1.40MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:41<01:16, 1.45MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:41<00:58, 1.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<00:41, 2.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:43<08:58, 198kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:43<06:34, 270kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:43<04:38, 380kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:45<03:27, 495kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:45<03:11, 536kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:45<02:25, 702kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:45<01:43, 974kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:47<01:30, 1.09MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:47<01:25, 1.15MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:47<01:03, 1.53MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<00:44, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:49<01:16, 1.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:49<01:13, 1.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:49<00:56, 1.67MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:49<00:39, 2.31MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:51<03:05, 488kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:51<02:28, 607kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:51<01:47, 832kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:51<01:14, 1.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:53<01:31, 945kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:53<01:21, 1.05MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:53<01:00, 1.40MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:53<00:42, 1.95MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:55<01:08, 1.20MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:55<01:05, 1.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:55<00:49, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:55<00:35, 2.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:57<00:48, 1.60MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:57<00:50, 1.53MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:57<00:38, 1.99MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:57<00:27, 2.73MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:59<00:48, 1.52MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:59<00:49, 1.48MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:59<00:38, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<00:26, 2.60MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:01<01:46, 654kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:01<01:29, 778kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:01<01:05, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<00:44, 1.47MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:03<02:24, 454kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:03<01:49, 592kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:03<01:17, 820kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:05<01:03, 963kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:05<00:56, 1.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:05<00:42, 1.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:05<00:30, 1.92MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:05<00:22, 2.53MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:07<01:43, 552kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:07<01:52, 507kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:07<01:25, 662kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:07<01:01, 910kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:07<00:43, 1.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:09<00:45, 1.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:09<01:01, 866kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:09<00:50, 1.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:09<00:36, 1.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:09<00:26, 1.93MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:11<00:34, 1.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:11<00:50, 962kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:11<00:42, 1.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:11<00:30, 1.55MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:11<00:21, 2.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:13<00:31, 1.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:13<00:46, 967kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:13<00:38, 1.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:13<00:27, 1.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:13<00:19, 2.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:14<00:28, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:15<00:30, 1.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:15<00:23, 1.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:15<00:16, 2.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<00:12, 3.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:16<02:09, 281kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:17<01:48, 336kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:17<01:19, 453kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:17<00:54, 634kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:17<00:37, 891kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:17<00:26, 1.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:18<27:34, 19.5kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:19<19:30, 27.5kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:19<13:35, 39.1kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:19<09:11, 55.7kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:19<06:12, 79.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<04:09, 113kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:20<04:53, 95.9kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:21<03:38, 128kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:21<02:34, 179kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:21<01:44, 254kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:21<01:09, 360kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:22<00:58, 410kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:23<00:51, 463kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:23<00:38, 614kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:23<00:26, 853kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:17, 1.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:24<00:24, 804kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:25<00:26, 733kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:25<00:20, 932kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:25<00:14, 1.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:25<00:09, 1.75MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:26<00:14, 1.05MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:26<00:17, 897kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:27<00:13, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:27<00:09, 1.54MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:27<00:05, 2.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:28<00:10, 1.12MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:28<00:12, 894kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:29<00:10, 1.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:29<00:06, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:29<00:04, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:29<00:03, 2.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:29<00:02, 3.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<00:02, 3.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:30<00:12, 586kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:30<00:12, 609kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:31<00:08, 788kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:31<00:05, 1.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:31<00:03, 1.44MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:31<00:02, 1.91MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:31<00:01, 2.48MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:32<00:30, 109kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:32<00:21, 146kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:33<00:13, 203kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:33<00:06, 287kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:33<00:02, 403kB/s].vector_cache/glove.6B.zip: 862MB [06:33, 2.19MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 827/400000 [00:00<00:48, 8265.32it/s]  0%|          | 1653/400000 [00:00<00:48, 8262.49it/s]  1%|          | 2467/400000 [00:00<00:48, 8223.12it/s]  1%|          | 3353/400000 [00:00<00:47, 8403.57it/s]  1%|          | 4136/400000 [00:00<00:48, 8221.78it/s]  1%|          | 4943/400000 [00:00<00:48, 8174.45it/s]  1%|▏         | 5782/400000 [00:00<00:47, 8234.25it/s]  2%|▏         | 6684/400000 [00:00<00:46, 8454.53it/s]  2%|▏         | 7544/400000 [00:00<00:46, 8493.90it/s]  2%|▏         | 8362/400000 [00:01<00:47, 8285.91it/s]  2%|▏         | 9221/400000 [00:01<00:46, 8374.19it/s]  3%|▎         | 10165/400000 [00:01<00:44, 8667.42it/s]  3%|▎         | 11111/400000 [00:01<00:43, 8890.79it/s]  3%|▎         | 12042/400000 [00:01<00:43, 9010.44it/s]  3%|▎         | 12941/400000 [00:01<00:43, 8945.71it/s]  3%|▎         | 13840/400000 [00:01<00:43, 8956.55it/s]  4%|▎         | 14805/400000 [00:01<00:42, 9152.19it/s]  4%|▍         | 15773/400000 [00:01<00:41, 9301.98it/s]  4%|▍         | 16750/400000 [00:01<00:40, 9437.38it/s]  4%|▍         | 17695/400000 [00:02<00:40, 9421.60it/s]  5%|▍         | 18653/400000 [00:02<00:40, 9466.87it/s]  5%|▍         | 19601/400000 [00:02<00:41, 9275.31it/s]  5%|▌         | 20530/400000 [00:02<00:41, 9194.57it/s]  5%|▌         | 21522/400000 [00:02<00:40, 9399.86it/s]  6%|▌         | 22464/400000 [00:02<00:40, 9358.40it/s]  6%|▌         | 23402/400000 [00:02<00:40, 9282.26it/s]  6%|▌         | 24332/400000 [00:02<00:40, 9252.05it/s]  6%|▋         | 25258/400000 [00:02<00:41, 9133.26it/s]  7%|▋         | 26242/400000 [00:02<00:40, 9332.81it/s]  7%|▋         | 27181/400000 [00:03<00:39, 9349.22it/s]  7%|▋         | 28118/400000 [00:03<00:40, 9195.47it/s]  7%|▋         | 29039/400000 [00:03<00:40, 9107.36it/s]  7%|▋         | 29966/400000 [00:03<00:40, 9153.27it/s]  8%|▊         | 30903/400000 [00:03<00:40, 9215.31it/s]  8%|▊         | 31826/400000 [00:03<00:40, 9000.31it/s]  8%|▊         | 32755/400000 [00:03<00:40, 9083.58it/s]  8%|▊         | 33700/400000 [00:03<00:39, 9188.81it/s]  9%|▊         | 34621/400000 [00:03<00:39, 9170.42it/s]  9%|▉         | 35539/400000 [00:03<00:40, 9024.40it/s]  9%|▉         | 36443/400000 [00:04<00:42, 8582.25it/s]  9%|▉         | 37381/400000 [00:04<00:41, 8805.27it/s] 10%|▉         | 38267/400000 [00:04<00:41, 8773.02it/s] 10%|▉         | 39193/400000 [00:04<00:40, 8912.01it/s] 10%|█         | 40155/400000 [00:04<00:39, 9112.10it/s] 10%|█         | 41070/400000 [00:04<00:40, 8811.55it/s] 11%|█         | 42033/400000 [00:04<00:39, 9040.35it/s] 11%|█         | 42995/400000 [00:04<00:38, 9206.07it/s] 11%|█         | 43923/400000 [00:04<00:38, 9226.01it/s] 11%|█         | 44873/400000 [00:04<00:38, 9303.26it/s] 11%|█▏        | 45806/400000 [00:05<00:38, 9283.58it/s] 12%|█▏        | 46739/400000 [00:05<00:37, 9296.74it/s] 12%|█▏        | 47670/400000 [00:05<00:38, 9270.56it/s] 12%|█▏        | 48598/400000 [00:05<00:38, 9133.39it/s] 12%|█▏        | 49513/400000 [00:05<00:39, 8941.21it/s] 13%|█▎        | 50409/400000 [00:05<00:40, 8628.77it/s] 13%|█▎        | 51333/400000 [00:05<00:39, 8800.02it/s] 13%|█▎        | 52290/400000 [00:05<00:38, 9015.83it/s] 13%|█▎        | 53200/400000 [00:05<00:38, 9039.80it/s] 14%|█▎        | 54152/400000 [00:06<00:37, 9178.47it/s] 14%|█▍        | 55073/400000 [00:06<00:38, 9028.39it/s] 14%|█▍        | 55978/400000 [00:06<00:38, 8991.63it/s] 14%|█▍        | 56879/400000 [00:06<00:38, 8877.39it/s] 14%|█▍        | 57773/400000 [00:06<00:38, 8893.70it/s] 15%|█▍        | 58720/400000 [00:06<00:37, 9057.60it/s] 15%|█▍        | 59628/400000 [00:06<00:37, 8979.94it/s] 15%|█▌        | 60528/400000 [00:06<00:38, 8862.07it/s] 15%|█▌        | 61513/400000 [00:06<00:37, 9135.46it/s] 16%|█▌        | 62486/400000 [00:06<00:36, 9305.79it/s] 16%|█▌        | 63454/400000 [00:07<00:35, 9414.69it/s] 16%|█▌        | 64398/400000 [00:07<00:36, 9270.91it/s] 16%|█▋        | 65328/400000 [00:07<00:37, 8948.20it/s] 17%|█▋        | 66247/400000 [00:07<00:37, 9017.77it/s] 17%|█▋        | 67189/400000 [00:07<00:36, 9134.69it/s] 17%|█▋        | 68139/400000 [00:07<00:35, 9238.71it/s] 17%|█▋        | 69065/400000 [00:07<00:36, 9014.76it/s] 17%|█▋        | 69997/400000 [00:07<00:36, 9103.19it/s] 18%|█▊        | 70968/400000 [00:07<00:35, 9276.86it/s] 18%|█▊        | 71946/400000 [00:07<00:34, 9420.11it/s] 18%|█▊        | 72902/400000 [00:08<00:34, 9461.10it/s] 18%|█▊        | 73850/400000 [00:08<00:35, 9314.02it/s] 19%|█▊        | 74809/400000 [00:08<00:34, 9394.16it/s] 19%|█▉        | 75778/400000 [00:08<00:34, 9480.93it/s] 19%|█▉        | 76751/400000 [00:08<00:33, 9553.92it/s] 19%|█▉        | 77708/400000 [00:08<00:35, 9086.26it/s] 20%|█▉        | 78623/400000 [00:08<00:35, 9043.20it/s] 20%|█▉        | 79532/400000 [00:08<00:35, 8979.34it/s] 20%|██        | 80433/400000 [00:08<00:36, 8716.54it/s] 20%|██        | 81309/400000 [00:09<00:37, 8476.74it/s] 21%|██        | 82229/400000 [00:09<00:36, 8677.18it/s] 21%|██        | 83101/400000 [00:09<00:37, 8418.38it/s] 21%|██        | 84019/400000 [00:09<00:36, 8631.88it/s] 21%|██        | 84916/400000 [00:09<00:36, 8728.76it/s] 21%|██▏       | 85844/400000 [00:09<00:35, 8886.09it/s] 22%|██▏       | 86736/400000 [00:09<00:35, 8781.13it/s] 22%|██▏       | 87617/400000 [00:09<00:37, 8297.26it/s] 22%|██▏       | 88454/400000 [00:09<00:38, 8079.48it/s] 22%|██▏       | 89269/400000 [00:09<00:39, 7895.71it/s] 23%|██▎       | 90064/400000 [00:10<00:39, 7814.75it/s] 23%|██▎       | 90853/400000 [00:10<00:39, 7836.47it/s] 23%|██▎       | 91640/400000 [00:10<00:39, 7735.83it/s] 23%|██▎       | 92416/400000 [00:10<00:40, 7652.97it/s] 23%|██▎       | 93184/400000 [00:10<00:40, 7632.09it/s] 23%|██▎       | 93949/400000 [00:10<00:41, 7437.09it/s] 24%|██▎       | 94695/400000 [00:10<00:41, 7327.24it/s] 24%|██▍       | 95535/400000 [00:10<00:39, 7617.85it/s] 24%|██▍       | 96428/400000 [00:10<00:38, 7967.25it/s] 24%|██▍       | 97391/400000 [00:10<00:36, 8401.59it/s] 25%|██▍       | 98333/400000 [00:11<00:34, 8681.53it/s] 25%|██▍       | 99291/400000 [00:11<00:33, 8932.40it/s] 25%|██▌       | 100194/400000 [00:11<00:33, 8870.73it/s] 25%|██▌       | 101100/400000 [00:11<00:33, 8926.12it/s] 26%|██▌       | 102079/400000 [00:11<00:32, 9165.54it/s] 26%|██▌       | 103031/400000 [00:11<00:32, 9266.89it/s] 26%|██▌       | 103999/400000 [00:11<00:31, 9385.15it/s] 26%|██▌       | 104941/400000 [00:11<00:31, 9258.73it/s] 26%|██▋       | 105904/400000 [00:11<00:31, 9366.02it/s] 27%|██▋       | 106843/400000 [00:12<00:32, 9130.74it/s] 27%|██▋       | 107759/400000 [00:12<00:32, 9129.95it/s] 27%|██▋       | 108681/400000 [00:12<00:31, 9149.70it/s] 27%|██▋       | 109598/400000 [00:12<00:33, 8761.54it/s] 28%|██▊       | 110530/400000 [00:12<00:32, 8920.88it/s] 28%|██▊       | 111426/400000 [00:12<00:32, 8832.22it/s] 28%|██▊       | 112374/400000 [00:12<00:31, 9015.76it/s] 28%|██▊       | 113325/400000 [00:12<00:31, 9157.28it/s] 29%|██▊       | 114244/400000 [00:12<00:31, 9118.12it/s] 29%|██▉       | 115234/400000 [00:12<00:30, 9336.97it/s] 29%|██▉       | 116178/400000 [00:13<00:30, 9366.44it/s] 29%|██▉       | 117132/400000 [00:13<00:30, 9416.40it/s] 30%|██▉       | 118075/400000 [00:13<00:30, 9193.24it/s] 30%|██▉       | 118997/400000 [00:13<00:31, 9053.70it/s] 30%|██▉       | 119924/400000 [00:13<00:30, 9114.57it/s] 30%|███       | 120837/400000 [00:13<00:30, 9032.14it/s] 30%|███       | 121803/400000 [00:13<00:30, 9211.60it/s] 31%|███       | 122739/400000 [00:13<00:29, 9255.27it/s] 31%|███       | 123666/400000 [00:13<00:30, 9012.83it/s] 31%|███       | 124570/400000 [00:13<00:31, 8880.74it/s] 31%|███▏      | 125461/400000 [00:14<00:31, 8714.05it/s] 32%|███▏      | 126335/400000 [00:14<00:31, 8632.68it/s] 32%|███▏      | 127244/400000 [00:14<00:31, 8762.34it/s] 32%|███▏      | 128122/400000 [00:14<00:31, 8576.94it/s] 32%|███▏      | 129084/400000 [00:14<00:30, 8863.28it/s] 32%|███▏      | 129975/400000 [00:14<00:31, 8592.77it/s] 33%|███▎      | 130839/400000 [00:14<00:31, 8439.38it/s] 33%|███▎      | 131687/400000 [00:14<00:31, 8438.71it/s] 33%|███▎      | 132534/400000 [00:14<00:32, 8170.60it/s] 33%|███▎      | 133355/400000 [00:15<00:33, 8051.55it/s] 34%|███▎      | 134239/400000 [00:15<00:32, 8271.13it/s] 34%|███▍      | 135176/400000 [00:15<00:30, 8570.99it/s] 34%|███▍      | 136073/400000 [00:15<00:30, 8684.51it/s] 34%|███▍      | 136959/400000 [00:15<00:30, 8735.20it/s] 34%|███▍      | 137836/400000 [00:15<00:30, 8681.35it/s] 35%|███▍      | 138707/400000 [00:15<00:31, 8276.98it/s] 35%|███▍      | 139541/400000 [00:15<00:32, 7902.80it/s] 35%|███▌      | 140339/400000 [00:15<00:33, 7719.66it/s] 35%|███▌      | 141118/400000 [00:15<00:33, 7707.47it/s] 35%|███▌      | 141948/400000 [00:16<00:32, 7876.02it/s] 36%|███▌      | 142825/400000 [00:16<00:31, 8122.90it/s] 36%|███▌      | 143783/400000 [00:16<00:30, 8511.17it/s] 36%|███▌      | 144673/400000 [00:16<00:29, 8623.79it/s] 36%|███▋      | 145594/400000 [00:16<00:28, 8789.34it/s] 37%|███▋      | 146514/400000 [00:16<00:28, 8908.59it/s] 37%|███▋      | 147409/400000 [00:16<00:28, 8907.89it/s] 37%|███▋      | 148331/400000 [00:16<00:27, 8997.20it/s] 37%|███▋      | 149233/400000 [00:16<00:28, 8698.60it/s] 38%|███▊      | 150107/400000 [00:16<00:29, 8580.17it/s] 38%|███▊      | 151036/400000 [00:17<00:28, 8779.86it/s] 38%|███▊      | 151970/400000 [00:17<00:27, 8940.16it/s] 38%|███▊      | 152867/400000 [00:17<00:27, 8909.12it/s] 38%|███▊      | 153760/400000 [00:17<00:28, 8684.66it/s] 39%|███▊      | 154711/400000 [00:17<00:27, 8915.28it/s] 39%|███▉      | 155639/400000 [00:17<00:27, 9019.72it/s] 39%|███▉      | 156580/400000 [00:17<00:26, 9132.10it/s] 39%|███▉      | 157523/400000 [00:17<00:26, 9219.09it/s] 40%|███▉      | 158447/400000 [00:17<00:26, 9008.62it/s] 40%|███▉      | 159396/400000 [00:17<00:26, 9147.65it/s] 40%|████      | 160313/400000 [00:18<00:26, 9118.00it/s] 40%|████      | 161265/400000 [00:18<00:25, 9234.13it/s] 41%|████      | 162207/400000 [00:18<00:25, 9287.11it/s] 41%|████      | 163137/400000 [00:18<00:26, 9059.79it/s] 41%|████      | 164099/400000 [00:18<00:25, 9220.02it/s] 41%|████▏     | 165024/400000 [00:18<00:26, 8944.90it/s] 41%|████▏     | 165923/400000 [00:18<00:26, 8956.62it/s] 42%|████▏     | 166821/400000 [00:18<00:26, 8778.24it/s] 42%|████▏     | 167702/400000 [00:18<00:27, 8360.00it/s] 42%|████▏     | 168544/400000 [00:19<00:27, 8322.19it/s] 42%|████▏     | 169470/400000 [00:19<00:26, 8580.81it/s] 43%|████▎     | 170420/400000 [00:19<00:25, 8836.59it/s] 43%|████▎     | 171373/400000 [00:19<00:25, 9031.22it/s] 43%|████▎     | 172281/400000 [00:19<00:26, 8714.94it/s] 43%|████▎     | 173159/400000 [00:19<00:26, 8548.04it/s] 44%|████▎     | 174116/400000 [00:19<00:25, 8828.73it/s] 44%|████▍     | 175042/400000 [00:19<00:25, 8950.69it/s] 44%|████▍     | 176021/400000 [00:19<00:24, 9185.43it/s] 44%|████▍     | 176945/400000 [00:19<00:24, 9067.37it/s] 44%|████▍     | 177929/400000 [00:20<00:23, 9283.96it/s] 45%|████▍     | 178898/400000 [00:20<00:23, 9401.92it/s] 45%|████▍     | 179867/400000 [00:20<00:23, 9484.73it/s] 45%|████▌     | 180818/400000 [00:20<00:23, 9415.30it/s] 45%|████▌     | 181762/400000 [00:20<00:24, 9059.48it/s] 46%|████▌     | 182673/400000 [00:20<00:24, 8952.79it/s] 46%|████▌     | 183572/400000 [00:20<00:25, 8543.27it/s] 46%|████▌     | 184488/400000 [00:20<00:24, 8717.11it/s] 46%|████▋     | 185365/400000 [00:20<00:25, 8505.68it/s] 47%|████▋     | 186221/400000 [00:21<00:25, 8303.90it/s] 47%|████▋     | 187149/400000 [00:21<00:24, 8573.37it/s] 47%|████▋     | 188030/400000 [00:21<00:24, 8641.34it/s] 47%|████▋     | 188986/400000 [00:21<00:23, 8896.48it/s] 47%|████▋     | 189881/400000 [00:21<00:23, 8901.58it/s] 48%|████▊     | 190775/400000 [00:21<00:23, 8796.56it/s] 48%|████▊     | 191658/400000 [00:21<00:23, 8738.67it/s] 48%|████▊     | 192534/400000 [00:21<00:23, 8735.12it/s] 48%|████▊     | 193436/400000 [00:21<00:23, 8816.15it/s] 49%|████▊     | 194335/400000 [00:21<00:23, 8866.68it/s] 49%|████▉     | 195264/400000 [00:22<00:22, 8987.53it/s] 49%|████▉     | 196228/400000 [00:22<00:22, 9172.56it/s] 49%|████▉     | 197147/400000 [00:22<00:22, 9153.96it/s] 50%|████▉     | 198068/400000 [00:22<00:22, 9170.63it/s] 50%|████▉     | 198986/400000 [00:22<00:21, 9156.89it/s] 50%|████▉     | 199926/400000 [00:22<00:21, 9225.31it/s] 50%|█████     | 200875/400000 [00:22<00:21, 9299.01it/s] 50%|█████     | 201836/400000 [00:22<00:21, 9388.93it/s] 51%|█████     | 202811/400000 [00:22<00:20, 9492.06it/s] 51%|█████     | 203789/400000 [00:22<00:20, 9571.71it/s] 51%|█████     | 204747/400000 [00:23<00:20, 9462.23it/s] 51%|█████▏    | 205694/400000 [00:23<00:20, 9397.87it/s] 52%|█████▏    | 206665/400000 [00:23<00:20, 9487.22it/s] 52%|█████▏    | 207615/400000 [00:23<00:20, 9387.40it/s] 52%|█████▏    | 208571/400000 [00:23<00:20, 9435.94it/s] 52%|█████▏    | 209516/400000 [00:23<00:20, 9193.31it/s] 53%|█████▎    | 210438/400000 [00:23<00:20, 9134.00it/s] 53%|█████▎    | 211378/400000 [00:23<00:20, 9210.18it/s] 53%|█████▎    | 212314/400000 [00:23<00:20, 9252.69it/s] 53%|█████▎    | 213241/400000 [00:23<00:20, 9092.72it/s] 54%|█████▎    | 214152/400000 [00:24<00:21, 8763.39it/s] 54%|█████▍    | 215032/400000 [00:24<00:21, 8686.31it/s] 54%|█████▍    | 215904/400000 [00:24<00:21, 8459.51it/s] 54%|█████▍    | 216754/400000 [00:24<00:21, 8381.03it/s] 54%|█████▍    | 217670/400000 [00:24<00:21, 8599.33it/s] 55%|█████▍    | 218550/400000 [00:24<00:20, 8658.45it/s] 55%|█████▍    | 219419/400000 [00:24<00:21, 8425.12it/s] 55%|█████▌    | 220290/400000 [00:24<00:21, 8507.85it/s] 55%|█████▌    | 221182/400000 [00:24<00:20, 8626.68it/s] 56%|█████▌    | 222047/400000 [00:25<00:20, 8552.49it/s] 56%|█████▌    | 222977/400000 [00:25<00:20, 8763.65it/s] 56%|█████▌    | 223921/400000 [00:25<00:19, 8955.87it/s] 56%|█████▌    | 224823/400000 [00:25<00:19, 8974.15it/s] 56%|█████▋    | 225723/400000 [00:25<00:19, 8793.60it/s] 57%|█████▋    | 226605/400000 [00:25<00:20, 8648.72it/s] 57%|█████▋    | 227472/400000 [00:25<00:21, 8184.06it/s] 57%|█████▋    | 228298/400000 [00:25<00:20, 8190.47it/s] 57%|█████▋    | 229237/400000 [00:25<00:20, 8512.81it/s] 58%|█████▊    | 230198/400000 [00:25<00:19, 8814.44it/s] 58%|█████▊    | 231087/400000 [00:26<00:19, 8775.57it/s] 58%|█████▊    | 231999/400000 [00:26<00:18, 8875.19it/s] 58%|█████▊    | 232902/400000 [00:26<00:18, 8920.71it/s] 58%|█████▊    | 233854/400000 [00:26<00:18, 9090.91it/s] 59%|█████▊    | 234766/400000 [00:26<00:18, 8939.28it/s] 59%|█████▉    | 235663/400000 [00:26<00:18, 8792.88it/s] 59%|█████▉    | 236618/400000 [00:26<00:18, 9004.78it/s] 59%|█████▉    | 237564/400000 [00:26<00:17, 9134.02it/s] 60%|█████▉    | 238505/400000 [00:26<00:17, 9212.98it/s] 60%|█████▉    | 239459/400000 [00:26<00:17, 9306.32it/s] 60%|██████    | 240392/400000 [00:27<00:17, 9093.49it/s] 60%|██████    | 241330/400000 [00:27<00:17, 9177.47it/s] 61%|██████    | 242250/400000 [00:27<00:17, 8899.23it/s] 61%|██████    | 243143/400000 [00:27<00:17, 8891.15it/s] 61%|██████    | 244063/400000 [00:27<00:17, 8980.07it/s] 61%|██████    | 244963/400000 [00:27<00:17, 8833.05it/s] 61%|██████▏   | 245929/400000 [00:27<00:16, 9065.54it/s] 62%|██████▏   | 246839/400000 [00:27<00:16, 9027.14it/s] 62%|██████▏   | 247813/400000 [00:27<00:16, 9227.64it/s] 62%|██████▏   | 248739/400000 [00:27<00:16, 9080.49it/s] 62%|██████▏   | 249650/400000 [00:28<00:16, 8901.49it/s] 63%|██████▎   | 250599/400000 [00:28<00:16, 9068.63it/s] 63%|██████▎   | 251509/400000 [00:28<00:16, 9045.08it/s] 63%|██████▎   | 252416/400000 [00:28<00:16, 8756.71it/s] 63%|██████▎   | 253295/400000 [00:28<00:17, 8606.71it/s] 64%|██████▎   | 254159/400000 [00:28<00:17, 8386.43it/s] 64%|██████▍   | 255033/400000 [00:28<00:17, 8488.88it/s] 64%|██████▍   | 256011/400000 [00:28<00:16, 8836.96it/s] 64%|██████▍   | 256938/400000 [00:28<00:15, 8961.90it/s] 64%|██████▍   | 257839/400000 [00:29<00:15, 8888.87it/s] 65%|██████▍   | 258731/400000 [00:29<00:15, 8838.42it/s] 65%|██████▍   | 259702/400000 [00:29<00:15, 9080.42it/s] 65%|██████▌   | 260650/400000 [00:29<00:15, 9194.57it/s] 65%|██████▌   | 261575/400000 [00:29<00:15, 9208.93it/s] 66%|██████▌   | 262498/400000 [00:29<00:15, 8932.81it/s] 66%|██████▌   | 263395/400000 [00:29<00:15, 8890.25it/s] 66%|██████▌   | 264287/400000 [00:29<00:15, 8882.57it/s] 66%|██████▋   | 265177/400000 [00:29<00:15, 8632.21it/s] 67%|██████▋   | 266085/400000 [00:29<00:15, 8761.23it/s] 67%|██████▋   | 266972/400000 [00:30<00:15, 8791.85it/s] 67%|██████▋   | 267853/400000 [00:30<00:15, 8792.52it/s] 67%|██████▋   | 268761/400000 [00:30<00:14, 8875.00it/s] 67%|██████▋   | 269650/400000 [00:30<00:14, 8861.21it/s] 68%|██████▊   | 270557/400000 [00:30<00:14, 8920.83it/s] 68%|██████▊   | 271450/400000 [00:30<00:14, 8831.90it/s] 68%|██████▊   | 272334/400000 [00:30<00:14, 8744.39it/s] 68%|██████▊   | 273210/400000 [00:30<00:14, 8714.05it/s] 69%|██████▊   | 274087/400000 [00:30<00:14, 8729.23it/s] 69%|██████▉   | 275005/400000 [00:30<00:14, 8858.27it/s] 69%|██████▉   | 275892/400000 [00:31<00:14, 8858.48it/s] 69%|██████▉   | 276779/400000 [00:31<00:13, 8810.66it/s] 69%|██████▉   | 277732/400000 [00:31<00:13, 9014.31it/s] 70%|██████▉   | 278687/400000 [00:31<00:13, 9166.86it/s] 70%|██████▉   | 279631/400000 [00:31<00:13, 9245.76it/s] 70%|███████   | 280557/400000 [00:31<00:13, 9068.52it/s] 70%|███████   | 281466/400000 [00:31<00:13, 9054.13it/s] 71%|███████   | 282402/400000 [00:31<00:12, 9143.49it/s] 71%|███████   | 283332/400000 [00:31<00:12, 9186.20it/s] 71%|███████   | 284291/400000 [00:31<00:12, 9301.36it/s] 71%|███████▏  | 285255/400000 [00:32<00:12, 9396.45it/s] 72%|███████▏  | 286196/400000 [00:32<00:12, 9210.68it/s] 72%|███████▏  | 287119/400000 [00:32<00:12, 9131.50it/s] 72%|███████▏  | 288034/400000 [00:32<00:12, 9122.89it/s] 72%|███████▏  | 288998/400000 [00:32<00:11, 9269.71it/s] 72%|███████▏  | 289927/400000 [00:32<00:11, 9270.09it/s] 73%|███████▎  | 290855/400000 [00:32<00:11, 9222.46it/s] 73%|███████▎  | 291823/400000 [00:32<00:11, 9353.45it/s] 73%|███████▎  | 292780/400000 [00:32<00:11, 9416.41it/s] 73%|███████▎  | 293762/400000 [00:32<00:11, 9531.97it/s] 74%|███████▎  | 294722/400000 [00:33<00:11, 9548.75it/s] 74%|███████▍  | 295678/400000 [00:33<00:11, 9322.12it/s] 74%|███████▍  | 296612/400000 [00:33<00:11, 9251.99it/s] 74%|███████▍  | 297539/400000 [00:33<00:11, 9159.75it/s] 75%|███████▍  | 298457/400000 [00:33<00:11, 8879.14it/s] 75%|███████▍  | 299348/400000 [00:33<00:11, 8562.84it/s] 75%|███████▌  | 300269/400000 [00:33<00:11, 8744.97it/s] 75%|███████▌  | 301215/400000 [00:33<00:11, 8946.01it/s] 76%|███████▌  | 302114/400000 [00:33<00:11, 8642.87it/s] 76%|███████▌  | 303061/400000 [00:34<00:10, 8875.04it/s] 76%|███████▌  | 303954/400000 [00:34<00:10, 8776.76it/s] 76%|███████▌  | 304836/400000 [00:34<00:10, 8728.73it/s] 76%|███████▋  | 305712/400000 [00:34<00:10, 8642.51it/s] 77%|███████▋  | 306650/400000 [00:34<00:10, 8849.99it/s] 77%|███████▋  | 307612/400000 [00:34<00:10, 9067.37it/s] 77%|███████▋  | 308522/400000 [00:34<00:10, 9015.49it/s] 77%|███████▋  | 309482/400000 [00:34<00:09, 9180.86it/s] 78%|███████▊  | 310445/400000 [00:34<00:09, 9309.56it/s] 78%|███████▊  | 311379/400000 [00:34<00:09, 9313.73it/s] 78%|███████▊  | 312312/400000 [00:35<00:09, 9245.09it/s] 78%|███████▊  | 313238/400000 [00:35<00:09, 9121.15it/s] 79%|███████▊  | 314202/400000 [00:35<00:09, 9269.86it/s] 79%|███████▉  | 315147/400000 [00:35<00:09, 9320.56it/s] 79%|███████▉  | 316081/400000 [00:35<00:09, 9294.77it/s] 79%|███████▉  | 317012/400000 [00:35<00:09, 8683.07it/s] 79%|███████▉  | 317889/400000 [00:35<00:09, 8622.56it/s] 80%|███████▉  | 318788/400000 [00:35<00:09, 8726.22it/s] 80%|███████▉  | 319729/400000 [00:35<00:08, 8919.91it/s] 80%|████████  | 320674/400000 [00:35<00:08, 9070.94it/s] 80%|████████  | 321635/400000 [00:36<00:08, 9226.14it/s] 81%|████████  | 322561/400000 [00:36<00:08, 8974.66it/s] 81%|████████  | 323503/400000 [00:36<00:08, 9102.30it/s] 81%|████████  | 324429/400000 [00:36<00:08, 9147.11it/s] 81%|████████▏ | 325346/400000 [00:36<00:08, 8837.89it/s] 82%|████████▏ | 326280/400000 [00:36<00:08, 8941.65it/s] 82%|████████▏ | 327178/400000 [00:36<00:08, 8889.12it/s] 82%|████████▏ | 328132/400000 [00:36<00:07, 9073.25it/s] 82%|████████▏ | 329083/400000 [00:36<00:07, 9199.46it/s] 83%|████████▎ | 330026/400000 [00:37<00:07, 9265.61it/s] 83%|████████▎ | 330992/400000 [00:37<00:07, 9380.02it/s] 83%|████████▎ | 331932/400000 [00:37<00:07, 8909.42it/s] 83%|████████▎ | 332829/400000 [00:37<00:07, 8717.26it/s] 83%|████████▎ | 333781/400000 [00:37<00:07, 8941.20it/s] 84%|████████▎ | 334723/400000 [00:37<00:07, 9079.35it/s] 84%|████████▍ | 335658/400000 [00:37<00:07, 9155.99it/s] 84%|████████▍ | 336577/400000 [00:37<00:07, 8993.34it/s] 84%|████████▍ | 337540/400000 [00:37<00:06, 9173.87it/s] 85%|████████▍ | 338491/400000 [00:37<00:06, 9270.64it/s] 85%|████████▍ | 339448/400000 [00:38<00:06, 9357.26it/s] 85%|████████▌ | 340386/400000 [00:38<00:06, 9285.83it/s] 85%|████████▌ | 341316/400000 [00:38<00:06, 9106.10it/s] 86%|████████▌ | 342253/400000 [00:38<00:06, 9182.25it/s] 86%|████████▌ | 343200/400000 [00:38<00:06, 9264.22it/s] 86%|████████▌ | 344128/400000 [00:38<00:06, 9243.44it/s] 86%|████████▋ | 345054/400000 [00:38<00:06, 9063.87it/s] 86%|████████▋ | 345962/400000 [00:38<00:06, 8761.57it/s] 87%|████████▋ | 346870/400000 [00:38<00:06, 8852.55it/s] 87%|████████▋ | 347758/400000 [00:38<00:06, 8546.49it/s] 87%|████████▋ | 348617/400000 [00:39<00:06, 8469.64it/s] 87%|████████▋ | 349480/400000 [00:39<00:05, 8516.37it/s] 88%|████████▊ | 350370/400000 [00:39<00:05, 8625.71it/s] 88%|████████▊ | 351301/400000 [00:39<00:05, 8818.81it/s] 88%|████████▊ | 352261/400000 [00:39<00:05, 9036.43it/s] 88%|████████▊ | 353172/400000 [00:39<00:05, 9055.79it/s] 89%|████████▊ | 354091/400000 [00:39<00:05, 9094.04it/s] 89%|████████▉ | 355014/400000 [00:39<00:04, 9133.70it/s] 89%|████████▉ | 355937/400000 [00:39<00:04, 9161.07it/s] 89%|████████▉ | 356854/400000 [00:39<00:04, 8917.50it/s] 89%|████████▉ | 357748/400000 [00:40<00:04, 8780.05it/s] 90%|████████▉ | 358628/400000 [00:40<00:04, 8526.06it/s] 90%|████████▉ | 359489/400000 [00:40<00:04, 8548.51it/s] 90%|█████████ | 360351/400000 [00:40<00:04, 8568.04it/s] 90%|█████████ | 361210/400000 [00:40<00:04, 8378.85it/s] 91%|█████████ | 362050/400000 [00:40<00:04, 8084.93it/s] 91%|█████████ | 362879/400000 [00:40<00:04, 8143.89it/s] 91%|█████████ | 363738/400000 [00:40<00:04, 8271.14it/s] 91%|█████████ | 364681/400000 [00:40<00:04, 8586.09it/s] 91%|█████████▏| 365594/400000 [00:41<00:03, 8740.96it/s] 92%|█████████▏| 366475/400000 [00:41<00:03, 8760.39it/s] 92%|█████████▏| 367354/400000 [00:41<00:03, 8682.97it/s] 92%|█████████▏| 368241/400000 [00:41<00:03, 8737.60it/s] 92%|█████████▏| 369117/400000 [00:41<00:03, 8738.56it/s] 93%|█████████▎| 370036/400000 [00:41<00:03, 8867.25it/s] 93%|█████████▎| 370943/400000 [00:41<00:03, 8924.19it/s] 93%|█████████▎| 371837/400000 [00:41<00:03, 8855.35it/s] 93%|█████████▎| 372800/400000 [00:41<00:02, 9071.57it/s] 93%|█████████▎| 373761/400000 [00:41<00:02, 9224.29it/s] 94%|█████████▎| 374710/400000 [00:42<00:02, 9301.13it/s] 94%|█████████▍| 375656/400000 [00:42<00:02, 9346.69it/s] 94%|█████████▍| 376592/400000 [00:42<00:02, 8943.84it/s] 94%|█████████▍| 377516/400000 [00:42<00:02, 9027.94it/s] 95%|█████████▍| 378438/400000 [00:42<00:02, 9084.46it/s] 95%|█████████▍| 379357/400000 [00:42<00:02, 9114.73it/s] 95%|█████████▌| 380271/400000 [00:42<00:02, 9060.57it/s] 95%|█████████▌| 381179/400000 [00:42<00:02, 8937.28it/s] 96%|█████████▌| 382141/400000 [00:42<00:01, 9129.73it/s] 96%|█████████▌| 383076/400000 [00:42<00:01, 9192.54it/s] 96%|█████████▌| 383997/400000 [00:43<00:01, 9123.60it/s] 96%|█████████▌| 384927/400000 [00:43<00:01, 9173.50it/s] 96%|█████████▋| 385846/400000 [00:43<00:01, 9030.94it/s] 97%|█████████▋| 386785/400000 [00:43<00:01, 9132.30it/s] 97%|█████████▋| 387703/400000 [00:43<00:01, 9146.41it/s] 97%|█████████▋| 388625/400000 [00:43<00:01, 9166.54it/s] 97%|█████████▋| 389543/400000 [00:43<00:01, 9157.69it/s] 98%|█████████▊| 390460/400000 [00:43<00:01, 9067.58it/s] 98%|█████████▊| 391368/400000 [00:43<00:00, 9002.92it/s] 98%|█████████▊| 392269/400000 [00:43<00:00, 8949.00it/s] 98%|█████████▊| 393191/400000 [00:44<00:00, 9027.74it/s] 99%|█████████▊| 394095/400000 [00:44<00:00, 8939.04it/s] 99%|█████████▊| 394990/400000 [00:44<00:00, 8655.85it/s] 99%|█████████▉| 395858/400000 [00:44<00:00, 8603.74it/s] 99%|█████████▉| 396723/400000 [00:44<00:00, 8615.85it/s] 99%|█████████▉| 397618/400000 [00:44<00:00, 8711.31it/s]100%|█████████▉| 398532/400000 [00:44<00:00, 8833.48it/s]100%|█████████▉| 399417/400000 [00:44<00:00, 8721.33it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8916.59it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fe0789e0ba8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010849195488532725 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.012352116929248822 	 Accuracy: 47

  model saves at 47% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15821 out of table with 15588 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15821 out of table with 15588 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

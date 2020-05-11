
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/22f2b7c7253266907172fe15dac6b61745a76480', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '22f2b7c7253266907172fe15dac6b61745a76480', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/22f2b7c7253266907172fe15dac6b61745a76480

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/22f2b7c7253266907172fe15dac6b61745a76480

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fb4a8956f98> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 06:14:45.881038
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-11 06:14:45.884314
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-11 06:14:45.887487
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-11 06:14:45.890626
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fb4b471a438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 360384.0312
Epoch 2/10

1/1 [==============================] - 0s 90ms/step - loss: 266083.3438
Epoch 3/10

1/1 [==============================] - 0s 94ms/step - loss: 152735.1875
Epoch 4/10

1/1 [==============================] - 0s 92ms/step - loss: 76328.3047
Epoch 5/10

1/1 [==============================] - 0s 91ms/step - loss: 36711.4922
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 18495.0078
Epoch 7/10

1/1 [==============================] - 0s 89ms/step - loss: 10476.1289
Epoch 8/10

1/1 [==============================] - 0s 108ms/step - loss: 6585.7646
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 4726.4868
Epoch 10/10

1/1 [==============================] - 0s 91ms/step - loss: 3572.2026

  #### Inference Need return ypred, ytrue ######################### 
[[-1.82808161e-01 -2.76140749e-01 -7.63267279e-03 -1.49295235e+00
   3.73793423e-01 -3.19480491e+00  1.60327363e+00  6.01962447e-01
  -1.31557798e+00 -2.03643441e-02 -1.56757450e+00  1.82701051e-01
  -1.40789354e+00 -1.58138573e-01 -1.43562829e+00 -2.05550385e+00
  -9.82626438e-01  2.18634397e-01  1.52883756e+00 -1.50245023e+00
  -4.08008456e-01  2.25089073e-01  1.99610412e-01  8.99492621e-01
   1.68600404e+00  5.60015082e-01 -7.73774445e-01  5.54582000e-01
  -1.03882909e+00 -1.31864572e+00 -6.66943550e-01 -2.40600395e+00
  -7.00701296e-01  1.11143351e+00  2.23118544e+00  1.81477797e+00
   2.65473127e-02 -7.38809168e-01 -2.69863665e-01 -1.02863252e+00
  -1.22818208e+00  1.44984698e+00  6.92445934e-02 -4.00600046e-01
   8.22748899e-01 -1.26964402e+00 -3.84146690e-01 -7.45727003e-01
  -1.77440810e+00  1.91818023e+00 -5.39924562e-01 -3.42224836e-01
   2.28309751e-01  1.14975274e-01 -2.37556353e-01 -4.13490415e-01
   2.45197153e+00 -2.46345162e+00 -1.17671871e+00 -1.62834454e+00
   2.05949008e-01  1.37096605e+01  1.48348875e+01  1.28578548e+01
   1.37713699e+01  1.13041916e+01  1.21075621e+01  1.32479048e+01
   1.15257492e+01  1.09169464e+01  1.17838478e+01  1.18991957e+01
   1.29460831e+01  1.14256172e+01  1.17151308e+01  1.18495903e+01
   1.20731182e+01  1.16809683e+01  1.24095984e+01  1.16512222e+01
   1.30153399e+01  1.09462461e+01  1.07891836e+01  1.21421652e+01
   1.24189310e+01  1.06154537e+01  1.04678345e+01  1.11968861e+01
   1.25745564e+01  1.30318518e+01  1.12437897e+01  1.29794188e+01
   1.07764893e+01  1.37966423e+01  1.24737406e+01  1.31702280e+01
   1.12980518e+01  1.21679783e+01  1.18981056e+01  1.18609829e+01
   1.30574493e+01  1.01493483e+01  1.36293516e+01  1.23147001e+01
   1.37143650e+01  1.27155418e+01  1.37432070e+01  1.17660532e+01
   1.11888208e+01  1.22889633e+01  1.51885204e+01  1.16068077e+01
   1.13107128e+01  1.20597105e+01  1.19835424e+01  1.25405607e+01
   1.20185204e+01  1.24137106e+01  1.33956013e+01  1.07821255e+01
   1.50249243e+00  7.26256311e-01  2.19996428e+00  1.31026506e+00
  -1.25151682e+00 -3.36521417e-01  6.30197406e-01  9.33986545e-01
   1.31983972e+00 -6.80338383e-01 -2.08872765e-01  1.63010764e+00
  -8.84014428e-01 -5.13076425e-01 -5.02913058e-01 -1.64266324e+00
  -1.26452446e+00  9.85468745e-01 -1.01392782e+00  2.07289338e+00
  -2.42966247e+00 -2.25288701e+00 -4.03165638e-01 -4.60720062e-02
  -1.87008023e-01  1.09752727e+00 -1.78253531e+00 -4.77728218e-01
  -2.01918960e-01  1.74444914e-03  7.19062090e-02  1.30699432e+00
   1.21091890e+00 -9.12641287e-01 -1.02505410e+00 -2.98451114e+00
   1.52902710e+00  2.24881977e-01 -9.12084103e-01  4.32129741e-01
  -2.52473783e+00 -8.12169969e-01 -1.26650643e+00 -2.64373481e-01
  -9.29331779e-02  1.03369617e+00 -1.51460207e+00 -6.73780739e-01
   8.50699782e-01 -1.03898239e+00 -5.11502147e-01  7.81786442e-01
   1.39816642e+00 -5.45033097e-01  1.31302905e+00  7.68357575e-01
   1.13844883e+00  8.69221807e-01  1.73525608e+00 -3.17191958e-01
   7.83676326e-01  2.88292432e+00  1.43599153e-01  8.27836335e-01
   1.13821447e+00  1.59641314e+00  1.04410803e+00  1.15502524e+00
   1.47355652e+00  6.34665370e-01  1.39462447e+00  2.03781009e-01
   2.02713776e+00  1.96087289e+00  3.02423179e-01  1.01124024e+00
   8.50610435e-01  1.99696302e-01  3.64507258e-01  6.14139438e-01
   1.31602752e+00  7.88719654e-01  3.73127341e-01  1.57260466e+00
   3.13304782e-01  2.05896497e+00  2.68818378e-01  6.41082466e-01
   2.60107708e+00  2.07386112e+00  1.76491964e+00  1.12852442e+00
   1.21924281e-01  2.60728359e+00  3.00412774e-01  1.64929342e+00
   1.17572641e+00  1.99268520e-01  1.84465230e-01  1.73962021e+00
   9.12339985e-01  3.36916447e-01  2.17474222e+00  2.60311961e-01
   1.99998760e+00  1.40084505e-01  1.99112868e+00  1.62924016e+00
   7.16816545e-01  1.22017741e-01  1.87359142e+00  1.82392812e+00
   1.15539551e-01  1.75122368e+00  2.48354936e+00  8.51502776e-01
   2.08062577e+00  2.55385637e-01  2.20869780e-01  2.56744003e+00
   1.01634860e-01  1.30023432e+01  1.06924305e+01  1.13451662e+01
   1.16479559e+01  1.38547735e+01  1.33652639e+01  1.13137817e+01
   1.06041079e+01  1.27630816e+01  1.15292263e+01  1.11409359e+01
   1.24813795e+01  1.24429398e+01  1.12864265e+01  1.33480921e+01
   1.19588928e+01  1.24582930e+01  1.16328545e+01  1.15875235e+01
   1.25272913e+01  1.07670650e+01  1.17371473e+01  1.16426668e+01
   1.21234474e+01  1.03331137e+01  1.04975100e+01  1.10487289e+01
   1.23907299e+01  1.16840744e+01  1.12633724e+01  1.16351452e+01
   1.15686865e+01  1.15220299e+01  1.35952387e+01  1.34928026e+01
   1.18041039e+01  1.18330698e+01  1.11008549e+01  1.21216555e+01
   1.24400635e+01  1.22081318e+01  1.15254612e+01  1.14587383e+01
   1.27987413e+01  1.19882536e+01  1.09965448e+01  1.17852106e+01
   1.20773115e+01  1.06505709e+01  1.11637211e+01  1.35553055e+01
   1.10130196e+01  1.19548750e+01  1.28619804e+01  1.07395658e+01
   1.16770010e+01  1.22951422e+01  1.22755680e+01  1.12743492e+01
   2.88861465e+00  5.33746719e-01  1.84695947e+00  2.42390394e-01
   5.60971677e-01  4.34005976e-01  3.38830709e+00  1.56796384e+00
   1.24077642e+00  3.51224804e+00  1.00932550e+00  2.44953334e-01
   4.15720463e-01  1.98695982e+00  2.11213231e-01  1.13118315e+00
   9.68672574e-01  2.13541889e+00  6.49659097e-01  1.31208539e+00
   1.06232405e+00  1.24147403e+00  1.02442551e+00  8.12072158e-01
   1.97940946e+00  1.95334744e+00  9.49244857e-01  5.65383434e-01
   1.09241700e+00  1.75595164e-01  2.87122345e+00  8.82334709e-02
   1.98347664e+00  3.48633528e-01  6.54788435e-01  1.25390100e+00
   8.38706017e-01  2.30607247e+00  6.29969299e-01  1.07949591e+00
   2.18833637e+00  7.61699915e-01  3.51838827e-01  1.07214153e+00
   1.98522747e-01  1.10902154e+00  2.37958956e+00  1.32859933e+00
   1.98777246e+00  1.56165433e+00  4.63612497e-01  1.42333066e+00
   1.81350625e+00  1.59929025e+00  2.35522652e+00  4.53261733e-01
   5.57272732e-01  8.46229255e-01  2.12199569e-01  3.36744976e+00
  -9.80771637e+00  6.59382772e+00 -1.17661829e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 06:14:56.726830
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   89.9501
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-11 06:14:56.731901
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                      8119
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-11 06:14:56.736180
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                     90.07
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-11 06:14:56.739941
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -726.114
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140413557210528
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140411027434912
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140411027435416
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140411027030480
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140411027030984
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140411027031488

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fb4b059def0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.538854
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.505404
grad_step = 000002, loss = 0.482264
grad_step = 000003, loss = 0.458794
grad_step = 000004, loss = 0.434704
grad_step = 000005, loss = 0.414198
grad_step = 000006, loss = 0.399830
grad_step = 000007, loss = 0.385794
grad_step = 000008, loss = 0.369919
grad_step = 000009, loss = 0.350918
grad_step = 000010, loss = 0.332666
grad_step = 000011, loss = 0.314191
grad_step = 000012, loss = 0.296025
grad_step = 000013, loss = 0.277772
grad_step = 000014, loss = 0.261268
grad_step = 000015, loss = 0.246675
grad_step = 000016, loss = 0.232619
grad_step = 000017, loss = 0.220775
grad_step = 000018, loss = 0.210046
grad_step = 000019, loss = 0.197916
grad_step = 000020, loss = 0.186519
grad_step = 000021, loss = 0.177255
grad_step = 000022, loss = 0.168766
grad_step = 000023, loss = 0.159520
grad_step = 000024, loss = 0.149440
grad_step = 000025, loss = 0.139292
grad_step = 000026, loss = 0.129653
grad_step = 000027, loss = 0.120628
grad_step = 000028, loss = 0.111934
grad_step = 000029, loss = 0.103439
grad_step = 000030, loss = 0.095621
grad_step = 000031, loss = 0.088322
grad_step = 000032, loss = 0.081251
grad_step = 000033, loss = 0.074439
grad_step = 000034, loss = 0.067848
grad_step = 000035, loss = 0.061827
grad_step = 000036, loss = 0.056094
grad_step = 000037, loss = 0.050332
grad_step = 000038, loss = 0.044971
grad_step = 000039, loss = 0.040243
grad_step = 000040, loss = 0.036103
grad_step = 000041, loss = 0.032210
grad_step = 000042, loss = 0.028524
grad_step = 000043, loss = 0.025109
grad_step = 000044, loss = 0.022145
grad_step = 000045, loss = 0.019543
grad_step = 000046, loss = 0.017144
grad_step = 000047, loss = 0.015074
grad_step = 000048, loss = 0.013431
grad_step = 000049, loss = 0.012053
grad_step = 000050, loss = 0.010721
grad_step = 000051, loss = 0.009519
grad_step = 000052, loss = 0.008570
grad_step = 000053, loss = 0.007759
grad_step = 000054, loss = 0.006994
grad_step = 000055, loss = 0.006373
grad_step = 000056, loss = 0.005923
grad_step = 000057, loss = 0.005539
grad_step = 000058, loss = 0.005166
grad_step = 000059, loss = 0.004832
grad_step = 000060, loss = 0.004563
grad_step = 000061, loss = 0.004305
grad_step = 000062, loss = 0.004063
grad_step = 000063, loss = 0.003880
grad_step = 000064, loss = 0.003744
grad_step = 000065, loss = 0.003600
grad_step = 000066, loss = 0.003457
grad_step = 000067, loss = 0.003351
grad_step = 000068, loss = 0.003250
grad_step = 000069, loss = 0.003140
grad_step = 000070, loss = 0.003047
grad_step = 000071, loss = 0.002970
grad_step = 000072, loss = 0.002893
grad_step = 000073, loss = 0.002809
grad_step = 000074, loss = 0.002736
grad_step = 000075, loss = 0.002673
grad_step = 000076, loss = 0.002614
grad_step = 000077, loss = 0.002563
grad_step = 000078, loss = 0.002521
grad_step = 000079, loss = 0.002480
grad_step = 000080, loss = 0.002441
grad_step = 000081, loss = 0.002408
grad_step = 000082, loss = 0.002380
grad_step = 000083, loss = 0.002357
grad_step = 000084, loss = 0.002339
grad_step = 000085, loss = 0.002323
grad_step = 000086, loss = 0.002308
grad_step = 000087, loss = 0.002295
grad_step = 000088, loss = 0.002283
grad_step = 000089, loss = 0.002270
grad_step = 000090, loss = 0.002259
grad_step = 000091, loss = 0.002249
grad_step = 000092, loss = 0.002239
grad_step = 000093, loss = 0.002229
grad_step = 000094, loss = 0.002219
grad_step = 000095, loss = 0.002208
grad_step = 000096, loss = 0.002198
grad_step = 000097, loss = 0.002190
grad_step = 000098, loss = 0.002180
grad_step = 000099, loss = 0.002172
grad_step = 000100, loss = 0.002163
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002154
grad_step = 000102, loss = 0.002146
grad_step = 000103, loss = 0.002138
grad_step = 000104, loss = 0.002130
grad_step = 000105, loss = 0.002122
grad_step = 000106, loss = 0.002115
grad_step = 000107, loss = 0.002107
grad_step = 000108, loss = 0.002100
grad_step = 000109, loss = 0.002093
grad_step = 000110, loss = 0.002086
grad_step = 000111, loss = 0.002079
grad_step = 000112, loss = 0.002072
grad_step = 000113, loss = 0.002065
grad_step = 000114, loss = 0.002059
grad_step = 000115, loss = 0.002052
grad_step = 000116, loss = 0.002046
grad_step = 000117, loss = 0.002039
grad_step = 000118, loss = 0.002033
grad_step = 000119, loss = 0.002027
grad_step = 000120, loss = 0.002020
grad_step = 000121, loss = 0.002014
grad_step = 000122, loss = 0.002007
grad_step = 000123, loss = 0.002000
grad_step = 000124, loss = 0.001994
grad_step = 000125, loss = 0.001987
grad_step = 000126, loss = 0.001980
grad_step = 000127, loss = 0.001973
grad_step = 000128, loss = 0.001966
grad_step = 000129, loss = 0.001959
grad_step = 000130, loss = 0.001952
grad_step = 000131, loss = 0.001946
grad_step = 000132, loss = 0.001942
grad_step = 000133, loss = 0.001943
grad_step = 000134, loss = 0.001955
grad_step = 000135, loss = 0.001984
grad_step = 000136, loss = 0.002024
grad_step = 000137, loss = 0.002059
grad_step = 000138, loss = 0.002033
grad_step = 000139, loss = 0.001963
grad_step = 000140, loss = 0.001896
grad_step = 000141, loss = 0.001891
grad_step = 000142, loss = 0.001931
grad_step = 000143, loss = 0.001960
grad_step = 000144, loss = 0.001945
grad_step = 000145, loss = 0.001894
grad_step = 000146, loss = 0.001857
grad_step = 000147, loss = 0.001858
grad_step = 000148, loss = 0.001883
grad_step = 000149, loss = 0.001902
grad_step = 000150, loss = 0.001895
grad_step = 000151, loss = 0.001867
grad_step = 000152, loss = 0.001836
grad_step = 000153, loss = 0.001820
grad_step = 000154, loss = 0.001823
grad_step = 000155, loss = 0.001836
grad_step = 000156, loss = 0.001850
grad_step = 000157, loss = 0.001858
grad_step = 000158, loss = 0.001857
grad_step = 000159, loss = 0.001846
grad_step = 000160, loss = 0.001829
grad_step = 000161, loss = 0.001809
grad_step = 000162, loss = 0.001792
grad_step = 000163, loss = 0.001780
grad_step = 000164, loss = 0.001774
grad_step = 000165, loss = 0.001772
grad_step = 000166, loss = 0.001774
grad_step = 000167, loss = 0.001780
grad_step = 000168, loss = 0.001797
grad_step = 000169, loss = 0.001834
grad_step = 000170, loss = 0.001910
grad_step = 000171, loss = 0.002019
grad_step = 000172, loss = 0.002101
grad_step = 000173, loss = 0.002058
grad_step = 000174, loss = 0.001873
grad_step = 000175, loss = 0.001750
grad_step = 000176, loss = 0.001807
grad_step = 000177, loss = 0.001919
grad_step = 000178, loss = 0.001905
grad_step = 000179, loss = 0.001781
grad_step = 000180, loss = 0.001739
grad_step = 000181, loss = 0.001809
grad_step = 000182, loss = 0.001851
grad_step = 000183, loss = 0.001803
grad_step = 000184, loss = 0.001733
grad_step = 000185, loss = 0.001743
grad_step = 000186, loss = 0.001796
grad_step = 000187, loss = 0.001794
grad_step = 000188, loss = 0.001744
grad_step = 000189, loss = 0.001719
grad_step = 000190, loss = 0.001743
grad_step = 000191, loss = 0.001771
grad_step = 000192, loss = 0.001757
grad_step = 000193, loss = 0.001724
grad_step = 000194, loss = 0.001710
grad_step = 000195, loss = 0.001727
grad_step = 000196, loss = 0.001747
grad_step = 000197, loss = 0.001740
grad_step = 000198, loss = 0.001717
grad_step = 000199, loss = 0.001703
grad_step = 000200, loss = 0.001707
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001720
grad_step = 000202, loss = 0.001724
grad_step = 000203, loss = 0.001715
grad_step = 000204, loss = 0.001701
grad_step = 000205, loss = 0.001694
grad_step = 000206, loss = 0.001697
grad_step = 000207, loss = 0.001704
grad_step = 000208, loss = 0.001706
grad_step = 000209, loss = 0.001700
grad_step = 000210, loss = 0.001693
grad_step = 000211, loss = 0.001686
grad_step = 000212, loss = 0.001685
grad_step = 000213, loss = 0.001687
grad_step = 000214, loss = 0.001689
grad_step = 000215, loss = 0.001690
grad_step = 000216, loss = 0.001688
grad_step = 000217, loss = 0.001685
grad_step = 000218, loss = 0.001680
grad_step = 000219, loss = 0.001677
grad_step = 000220, loss = 0.001674
grad_step = 000221, loss = 0.001673
grad_step = 000222, loss = 0.001673
grad_step = 000223, loss = 0.001673
grad_step = 000224, loss = 0.001674
grad_step = 000225, loss = 0.001675
grad_step = 000226, loss = 0.001676
grad_step = 000227, loss = 0.001678
grad_step = 000228, loss = 0.001681
grad_step = 000229, loss = 0.001687
grad_step = 000230, loss = 0.001696
grad_step = 000231, loss = 0.001708
grad_step = 000232, loss = 0.001725
grad_step = 000233, loss = 0.001746
grad_step = 000234, loss = 0.001777
grad_step = 000235, loss = 0.001799
grad_step = 000236, loss = 0.001810
grad_step = 000237, loss = 0.001789
grad_step = 000238, loss = 0.001746
grad_step = 000239, loss = 0.001692
grad_step = 000240, loss = 0.001656
grad_step = 000241, loss = 0.001654
grad_step = 000242, loss = 0.001676
grad_step = 000243, loss = 0.001709
grad_step = 000244, loss = 0.001736
grad_step = 000245, loss = 0.001753
grad_step = 000246, loss = 0.001746
grad_step = 000247, loss = 0.001722
grad_step = 000248, loss = 0.001686
grad_step = 000249, loss = 0.001656
grad_step = 000250, loss = 0.001639
grad_step = 000251, loss = 0.001640
grad_step = 000252, loss = 0.001652
grad_step = 000253, loss = 0.001665
grad_step = 000254, loss = 0.001674
grad_step = 000255, loss = 0.001671
grad_step = 000256, loss = 0.001663
grad_step = 000257, loss = 0.001649
grad_step = 000258, loss = 0.001635
grad_step = 000259, loss = 0.001627
grad_step = 000260, loss = 0.001624
grad_step = 000261, loss = 0.001626
grad_step = 000262, loss = 0.001631
grad_step = 000263, loss = 0.001635
grad_step = 000264, loss = 0.001641
grad_step = 000265, loss = 0.001646
grad_step = 000266, loss = 0.001651
grad_step = 000267, loss = 0.001656
grad_step = 000268, loss = 0.001661
grad_step = 000269, loss = 0.001666
grad_step = 000270, loss = 0.001670
grad_step = 000271, loss = 0.001670
grad_step = 000272, loss = 0.001665
grad_step = 000273, loss = 0.001656
grad_step = 000274, loss = 0.001642
grad_step = 000275, loss = 0.001626
grad_step = 000276, loss = 0.001611
grad_step = 000277, loss = 0.001599
grad_step = 000278, loss = 0.001591
grad_step = 000279, loss = 0.001587
grad_step = 000280, loss = 0.001586
grad_step = 000281, loss = 0.001587
grad_step = 000282, loss = 0.001590
grad_step = 000283, loss = 0.001597
grad_step = 000284, loss = 0.001611
grad_step = 000285, loss = 0.001635
grad_step = 000286, loss = 0.001677
grad_step = 000287, loss = 0.001735
grad_step = 000288, loss = 0.001800
grad_step = 000289, loss = 0.001827
grad_step = 000290, loss = 0.001789
grad_step = 000291, loss = 0.001682
grad_step = 000292, loss = 0.001588
grad_step = 000293, loss = 0.001567
grad_step = 000294, loss = 0.001613
grad_step = 000295, loss = 0.001670
grad_step = 000296, loss = 0.001696
grad_step = 000297, loss = 0.001663
grad_step = 000298, loss = 0.001599
grad_step = 000299, loss = 0.001550
grad_step = 000300, loss = 0.001544
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001573
grad_step = 000302, loss = 0.001611
grad_step = 000303, loss = 0.001623
grad_step = 000304, loss = 0.001609
grad_step = 000305, loss = 0.001578
grad_step = 000306, loss = 0.001552
grad_step = 000307, loss = 0.001542
grad_step = 000308, loss = 0.001545
grad_step = 000309, loss = 0.001546
grad_step = 000310, loss = 0.001541
grad_step = 000311, loss = 0.001535
grad_step = 000312, loss = 0.001536
grad_step = 000313, loss = 0.001546
grad_step = 000314, loss = 0.001559
grad_step = 000315, loss = 0.001561
grad_step = 000316, loss = 0.001547
grad_step = 000317, loss = 0.001525
grad_step = 000318, loss = 0.001512
grad_step = 000319, loss = 0.001512
grad_step = 000320, loss = 0.001515
grad_step = 000321, loss = 0.001513
grad_step = 000322, loss = 0.001506
grad_step = 000323, loss = 0.001500
grad_step = 000324, loss = 0.001500
grad_step = 000325, loss = 0.001507
grad_step = 000326, loss = 0.001518
grad_step = 000327, loss = 0.001532
grad_step = 000328, loss = 0.001553
grad_step = 000329, loss = 0.001585
grad_step = 000330, loss = 0.001642
grad_step = 000331, loss = 0.001722
grad_step = 000332, loss = 0.001771
grad_step = 000333, loss = 0.001737
grad_step = 000334, loss = 0.001588
grad_step = 000335, loss = 0.001493
grad_step = 000336, loss = 0.001531
grad_step = 000337, loss = 0.001601
grad_step = 000338, loss = 0.001608
grad_step = 000339, loss = 0.001544
grad_step = 000340, loss = 0.001539
grad_step = 000341, loss = 0.001562
grad_step = 000342, loss = 0.001541
grad_step = 000343, loss = 0.001484
grad_step = 000344, loss = 0.001478
grad_step = 000345, loss = 0.001528
grad_step = 000346, loss = 0.001546
grad_step = 000347, loss = 0.001523
grad_step = 000348, loss = 0.001495
grad_step = 000349, loss = 0.001516
grad_step = 000350, loss = 0.001541
grad_step = 000351, loss = 0.001534
grad_step = 000352, loss = 0.001499
grad_step = 000353, loss = 0.001492
grad_step = 000354, loss = 0.001505
grad_step = 000355, loss = 0.001501
grad_step = 000356, loss = 0.001473
grad_step = 000357, loss = 0.001455
grad_step = 000358, loss = 0.001462
grad_step = 000359, loss = 0.001469
grad_step = 000360, loss = 0.001463
grad_step = 000361, loss = 0.001446
grad_step = 000362, loss = 0.001441
grad_step = 000363, loss = 0.001447
grad_step = 000364, loss = 0.001454
grad_step = 000365, loss = 0.001450
grad_step = 000366, loss = 0.001440
grad_step = 000367, loss = 0.001435
grad_step = 000368, loss = 0.001439
grad_step = 000369, loss = 0.001447
grad_step = 000370, loss = 0.001455
grad_step = 000371, loss = 0.001466
grad_step = 000372, loss = 0.001496
grad_step = 000373, loss = 0.001564
grad_step = 000374, loss = 0.001707
grad_step = 000375, loss = 0.001890
grad_step = 000376, loss = 0.002121
grad_step = 000377, loss = 0.002100
grad_step = 000378, loss = 0.001831
grad_step = 000379, loss = 0.001551
grad_step = 000380, loss = 0.001529
grad_step = 000381, loss = 0.001651
grad_step = 000382, loss = 0.001716
grad_step = 000383, loss = 0.001698
grad_step = 000384, loss = 0.001535
grad_step = 000385, loss = 0.001488
grad_step = 000386, loss = 0.001577
grad_step = 000387, loss = 0.001633
grad_step = 000388, loss = 0.001570
grad_step = 000389, loss = 0.001455
grad_step = 000390, loss = 0.001494
grad_step = 000391, loss = 0.001578
grad_step = 000392, loss = 0.001529
grad_step = 000393, loss = 0.001421
grad_step = 000394, loss = 0.001464
grad_step = 000395, loss = 0.001514
grad_step = 000396, loss = 0.001496
grad_step = 000397, loss = 0.001435
grad_step = 000398, loss = 0.001427
grad_step = 000399, loss = 0.001477
grad_step = 000400, loss = 0.001451
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001425
grad_step = 000402, loss = 0.001432
grad_step = 000403, loss = 0.001436
grad_step = 000404, loss = 0.001416
grad_step = 000405, loss = 0.001408
grad_step = 000406, loss = 0.001424
grad_step = 000407, loss = 0.001417
grad_step = 000408, loss = 0.001400
grad_step = 000409, loss = 0.001388
grad_step = 000410, loss = 0.001401
grad_step = 000411, loss = 0.001411
grad_step = 000412, loss = 0.001391
grad_step = 000413, loss = 0.001384
grad_step = 000414, loss = 0.001387
grad_step = 000415, loss = 0.001390
grad_step = 000416, loss = 0.001389
grad_step = 000417, loss = 0.001381
grad_step = 000418, loss = 0.001379
grad_step = 000419, loss = 0.001380
grad_step = 000420, loss = 0.001377
grad_step = 000421, loss = 0.001370
grad_step = 000422, loss = 0.001370
grad_step = 000423, loss = 0.001373
grad_step = 000424, loss = 0.001372
grad_step = 000425, loss = 0.001369
grad_step = 000426, loss = 0.001366
grad_step = 000427, loss = 0.001363
grad_step = 000428, loss = 0.001363
grad_step = 000429, loss = 0.001362
grad_step = 000430, loss = 0.001358
grad_step = 000431, loss = 0.001355
grad_step = 000432, loss = 0.001353
grad_step = 000433, loss = 0.001353
grad_step = 000434, loss = 0.001352
grad_step = 000435, loss = 0.001350
grad_step = 000436, loss = 0.001349
grad_step = 000437, loss = 0.001347
grad_step = 000438, loss = 0.001346
grad_step = 000439, loss = 0.001346
grad_step = 000440, loss = 0.001346
grad_step = 000441, loss = 0.001347
grad_step = 000442, loss = 0.001349
grad_step = 000443, loss = 0.001356
grad_step = 000444, loss = 0.001370
grad_step = 000445, loss = 0.001395
grad_step = 000446, loss = 0.001446
grad_step = 000447, loss = 0.001505
grad_step = 000448, loss = 0.001604
grad_step = 000449, loss = 0.001656
grad_step = 000450, loss = 0.001692
grad_step = 000451, loss = 0.001603
grad_step = 000452, loss = 0.001479
grad_step = 000453, loss = 0.001378
grad_step = 000454, loss = 0.001354
grad_step = 000455, loss = 0.001390
grad_step = 000456, loss = 0.001452
grad_step = 000457, loss = 0.001518
grad_step = 000458, loss = 0.001483
grad_step = 000459, loss = 0.001408
grad_step = 000460, loss = 0.001335
grad_step = 000461, loss = 0.001324
grad_step = 000462, loss = 0.001367
grad_step = 000463, loss = 0.001410
grad_step = 000464, loss = 0.001417
grad_step = 000465, loss = 0.001375
grad_step = 000466, loss = 0.001332
grad_step = 000467, loss = 0.001314
grad_step = 000468, loss = 0.001326
grad_step = 000469, loss = 0.001351
grad_step = 000470, loss = 0.001362
grad_step = 000471, loss = 0.001361
grad_step = 000472, loss = 0.001347
grad_step = 000473, loss = 0.001335
grad_step = 000474, loss = 0.001324
grad_step = 000475, loss = 0.001313
grad_step = 000476, loss = 0.001305
grad_step = 000477, loss = 0.001302
grad_step = 000478, loss = 0.001305
grad_step = 000479, loss = 0.001315
grad_step = 000480, loss = 0.001324
grad_step = 000481, loss = 0.001320
grad_step = 000482, loss = 0.001309
grad_step = 000483, loss = 0.001296
grad_step = 000484, loss = 0.001287
grad_step = 000485, loss = 0.001286
grad_step = 000486, loss = 0.001286
grad_step = 000487, loss = 0.001284
grad_step = 000488, loss = 0.001282
grad_step = 000489, loss = 0.001280
grad_step = 000490, loss = 0.001279
grad_step = 000491, loss = 0.001281
grad_step = 000492, loss = 0.001285
grad_step = 000493, loss = 0.001289
grad_step = 000494, loss = 0.001294
grad_step = 000495, loss = 0.001299
grad_step = 000496, loss = 0.001302
grad_step = 000497, loss = 0.001308
grad_step = 000498, loss = 0.001312
grad_step = 000499, loss = 0.001316
grad_step = 000500, loss = 0.001317
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001317
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

  date_run                              2020-05-11 06:15:15.244697
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.215288
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-11 06:15:15.251427
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.111527
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-11 06:15:15.258768
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.131622
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-11 06:15:15.270655
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.694696
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
0   2020-05-11 06:14:45.881038  ...    mean_absolute_error
1   2020-05-11 06:14:45.884314  ...     mean_squared_error
2   2020-05-11 06:14:45.887487  ...  median_absolute_error
3   2020-05-11 06:14:45.890626  ...               r2_score
4   2020-05-11 06:14:56.726830  ...    mean_absolute_error
5   2020-05-11 06:14:56.731901  ...     mean_squared_error
6   2020-05-11 06:14:56.736180  ...  median_absolute_error
7   2020-05-11 06:14:56.739941  ...               r2_score
8   2020-05-11 06:15:15.244697  ...    mean_absolute_error
9   2020-05-11 06:15:15.251427  ...     mean_squared_error
10  2020-05-11 06:15:15.258768  ...  median_absolute_error
11  2020-05-11 06:15:15.270655  ...               r2_score

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 40%|███▉      | 3940352/9912422 [00:00<00:00, 38504450.25it/s]9920512it [00:00, 31023727.03it/s]                             
0it [00:00, ?it/s]32768it [00:00, 770553.21it/s]
0it [00:00, ?it/s]  4%|▍         | 73728/1648877 [00:00<00:02, 734763.35it/s]1654784it [00:00, 10818473.11it/s]                         
0it [00:00, ?it/s]8192it [00:00, 202345.83it/s]>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02a5614780> <class 'mlmodels.model_tch.torchhub.Model'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0242d5bb00> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02a55cbe48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0242d5bda0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02a55cbe48> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02a5614e80> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02a5614780> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0257582f98> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02a55cbe48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f0242d5bda0> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f02a55cbe48> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fe2fd1061d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=9535c4ad0616a67ebd3142a604b16b2f5729b994ffe369e49fc09e2cbd004e64
  Stored in directory: /tmp/pip-ephem-wheel-cache-hv1tlok2/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fe2f3490080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1482752/17464789 [=>............................] - ETA: 0s
 3571712/17464789 [=====>........................] - ETA: 0s
 6168576/17464789 [=========>....................] - ETA: 0s
11231232/17464789 [==================>...........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-11 06:16:40.921016: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 06:16:40.925177: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095239999 Hz
2020-05-11 06:16:40.925752: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5626db96c980 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 06:16:40.925766: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.7126 - accuracy: 0.4970
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6053 - accuracy: 0.5040 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6743 - accuracy: 0.4995
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7218 - accuracy: 0.4964
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7126 - accuracy: 0.4970
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7433 - accuracy: 0.4950
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7318 - accuracy: 0.4958
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7331 - accuracy: 0.4957
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7510 - accuracy: 0.4945
11000/25000 [============>.................] - ETA: 3s - loss: 7.7001 - accuracy: 0.4978
12000/25000 [=============>................] - ETA: 3s - loss: 7.6909 - accuracy: 0.4984
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7350 - accuracy: 0.4955
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7400 - accuracy: 0.4952
15000/25000 [=================>............] - ETA: 2s - loss: 7.7249 - accuracy: 0.4962
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7088 - accuracy: 0.4972
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6856 - accuracy: 0.4988
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6998 - accuracy: 0.4978
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7118 - accuracy: 0.4971
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7149 - accuracy: 0.4969
21000/25000 [========================>.....] - ETA: 0s - loss: 7.7053 - accuracy: 0.4975
22000/25000 [=========================>....] - ETA: 0s - loss: 7.7022 - accuracy: 0.4977
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6846 - accuracy: 0.4988
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6807 - accuracy: 0.4991
25000/25000 [==============================] - 7s 270us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-11 06:16:54.116179
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-11 06:16:54.116179  model_keras.textcnn.py  ...    0.5  accuracy_score

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
2020-05-11 06:16:59.888098: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-11 06:16:59.893824: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095239999 Hz
2020-05-11 06:16:59.893970: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55819ae44a20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-11 06:16:59.893982: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f8b3d4dfda0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.2436 - crf_viterbi_accuracy: 0.3333 - val_loss: 1.2422 - val_crf_viterbi_accuracy: 0.8800

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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f8b33780898> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.2986 - accuracy: 0.5240
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4520 - accuracy: 0.5140 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6717 - accuracy: 0.4997
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7395 - accuracy: 0.4952
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6636 - accuracy: 0.5002
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6462 - accuracy: 0.5013
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6250 - accuracy: 0.5027
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.7109 - accuracy: 0.4971
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6728 - accuracy: 0.4996
11000/25000 [============>.................] - ETA: 3s - loss: 7.6834 - accuracy: 0.4989
12000/25000 [=============>................] - ETA: 3s - loss: 7.6858 - accuracy: 0.4988
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6796 - accuracy: 0.4992
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6831 - accuracy: 0.4989
15000/25000 [=================>............] - ETA: 2s - loss: 7.7055 - accuracy: 0.4975
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6992 - accuracy: 0.4979
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6946 - accuracy: 0.4982
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6947 - accuracy: 0.4982
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7142 - accuracy: 0.4969
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7065 - accuracy: 0.4974
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6885 - accuracy: 0.4986
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6694 - accuracy: 0.4998
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
25000/25000 [==============================] - 7s 273us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f8afbe473c8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:22:16, 11.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:11:39, 15.8kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:41:26, 22.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:29:31, 31.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.61M/862M [00:01<5:13:53, 45.6kB/s].vector_cache/glove.6B.zip:   1%|          | 9.27M/862M [00:01<3:38:20, 65.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.8M/862M [00:01<2:32:20, 92.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.6M/862M [00:01<1:46:07, 133kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.0M/862M [00:01<1:13:59, 189kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:01<51:37, 270kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.6M/862M [00:01<36:04, 384kB/s].vector_cache/glove.6B.zip:   4%|▍         | 36.1M/862M [00:02<25:09, 547kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.1M/862M [00:02<17:41, 776kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.8M/862M [00:02<12:23, 1.10MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.4M/862M [00:02<08:42, 1.55MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:02<07:01, 1.92MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<06:48, 1.97MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<06:40, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<05:07, 2.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<06:07, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<06:00, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.8M/862M [00:06<04:37, 2.88MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<05:52, 2.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<07:23, 1.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:08<05:51, 2.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.0M/862M [00:08<04:18, 3.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<07:19, 1.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<06:34, 2.01MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.3M/862M [00:10<04:57, 2.66MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<06:22, 2.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<05:48, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.6M/862M [00:12<04:23, 2.99MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<06:09, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<05:38, 2.32MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.8M/862M [00:14<04:16, 3.05MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<06:04, 2.14MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<06:48, 1.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:16<05:21, 2.43MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:16<03:53, 3.33MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<11:07, 1.16MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<09:17, 1.39MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.0M/862M [00:18<06:49, 1.89MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<07:45, 1.66MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<06:33, 1.96MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:20<05:00, 2.57MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:20<03:38, 3.52MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<21:53, 586kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<16:38, 770kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<11:54, 1.07MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<11:20, 1.12MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<09:15, 1.38MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:24<06:44, 1.89MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<07:43, 1.64MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:42, 1.89MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<05:00, 2.52MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:30, 1.94MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:49, 2.17MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<04:23, 2.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:01, 2.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:47, 1.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<05:17, 2.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:30<03:53, 3.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<07:51, 1.59MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:44, 1.85MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<05:02, 2.47MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:25, 1.93MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:47, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:18, 2.87MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:54, 2.09MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:39, 1.85MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:17, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:40, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:15, 2.33MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<03:59, 3.06MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<05:37, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:27, 1.89MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:08, 2.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<05:33, 2.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:09, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<03:53, 3.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<02:52, 4.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<46:48, 258kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<35:14, 342kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<25:15, 477kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<19:33, 613kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<14:54, 804kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<10:40, 1.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<10:16, 1.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:23, 1.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<06:08, 1.94MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<07:05, 1.67MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:11, 1.91MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:37, 2.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<06:01, 1.96MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:25, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:05, 2.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:37, 2.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:08, 2.28MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:53, 3.00MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<05:28, 2.13MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:02, 2.31MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<03:48, 3.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<05:24, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<04:58, 2.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<03:46, 3.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:21, 2.15MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<04:55, 2.34MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<03:44, 3.07MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:18, 2.15MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<04:54, 2.33MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<03:40, 3.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:14, 2.17MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:51, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<03:38, 3.11MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:11, 2.18MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:56, 1.90MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<04:39, 2.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<03:23, 3.31MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:49, 1.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<07:20, 1.53MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<05:22, 2.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:22, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<05:36, 1.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<04:12, 2.65MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<05:33, 2.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<06:09, 1.80MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<04:48, 2.31MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:11<03:30, 3.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:58, 1.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:43, 1.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<04:59, 2.21MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<06:03, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<06:29, 1.69MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:06, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:19, 2.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<04:51, 2.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<03:40, 2.96MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:05, 2.12MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:47, 1.87MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:31, 2.39MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<03:18, 3.26MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<07:36, 1.41MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:26, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<04:45, 2.26MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:48, 1.84MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:14, 2.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:56, 2.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:03, 2.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:55, 1.79MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:43, 2.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<03:26, 3.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<18:57, 557kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<14:20, 735kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<10:17, 1.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<09:38, 1.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<08:55, 1.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<06:44, 1.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:29<04:48, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<23:43, 439kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<17:41, 588kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<12:36, 823kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<11:11, 925kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<08:54, 1.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<06:28, 1.59MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<06:56, 1.48MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<07:00, 1.46MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:20, 1.92MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<03:55, 2.61MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:56, 1.72MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:13, 1.95MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:54, 2.60MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:06, 1.99MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:38, 1.79MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:27, 2.27MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:44, 2.12MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:22, 2.30MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:18, 3.03MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:39, 2.15MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:20, 1.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<04:14, 2.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:33, 2.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<04:12, 2.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:11, 3.10MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:32, 2.17MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<05:12, 1.90MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<04:03, 2.43MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<03:01, 3.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:01, 1.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:31, 2.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<03:24, 2.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:39, 2.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:15, 2.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:13, 3.00MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:31, 2.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:10, 2.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:09, 3.05MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:28, 2.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:05, 2.34MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<03:04, 3.11MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:24, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:03, 2.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:02, 3.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:22, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:02, 2.34MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:01, 3.12MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:19, 2.17MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:59, 2.35MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:01, 3.09MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:18, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:58, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<02:58, 3.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:16, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:56, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<02:59, 3.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:15, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:55, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<02:58, 3.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:13, 2.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:49, 1.88MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:46, 2.41MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<02:46, 3.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<05:41, 1.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:55, 1.83MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<03:38, 2.47MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:38, 1.93MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:10, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:08, 2.85MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:17, 2.07MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:49, 1.84MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:47, 2.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:14<02:43, 3.24MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<22:42, 389kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<16:48, 525kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<11:57, 736kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<10:23, 843kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<08:10, 1.07MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<05:53, 1.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<06:09, 1.41MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:20<06:06, 1.42MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:38, 1.87MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<03:22, 2.57MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<05:59, 1.44MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<05:04, 1.70MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:44, 2.30MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:34, 1.87MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:58, 1.72MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:52, 2.21MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<02:47, 3.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<10:34, 803kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<08:17, 1.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<06:00, 1.41MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<06:10, 1.37MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:11, 1.62MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<03:48, 2.21MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:38, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:06, 2.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:04, 2.70MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:06, 2.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<03:43, 2.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<02:49, 2.93MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:54, 2.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:34, 2.29MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<02:42, 3.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:49, 2.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:30, 2.32MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<02:39, 3.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<03:45, 2.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:27, 2.33MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<02:37, 3.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<03:43, 2.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:26, 2.33MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<02:36, 3.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:41, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:23, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<02:34, 3.07MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:39, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:22, 2.33MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<02:33, 3.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:37, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:20, 2.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:45<02:29, 3.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:33, 2.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:16, 2.35MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<02:29, 3.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:32, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:16, 2.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:28, 3.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:31, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:14, 2.35MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<02:25, 3.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:29, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:12, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:25, 3.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:27, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:57, 1.89MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:05, 2.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:55<02:15, 3.29MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<05:16, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:26, 1.66MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<03:17, 2.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:00, 1.83MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:18, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:19, 2.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<02:25, 3.00MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<04:55, 1.48MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<04:12, 1.72MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<03:07, 2.31MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:51, 1.86MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<04:10, 1.72MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:17, 2.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:03<02:22, 3.01MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<27:23, 260kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<19:46, 360kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<13:58, 508kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<09:49, 719kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<35:14, 200kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<26:06, 270kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<18:32, 379kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:07<12:59, 538kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<13:30, 517kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<10:11, 685kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<07:17, 954kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<06:41, 1.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<06:06, 1.13MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:37, 1.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<03:17, 2.08MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<6:35:56, 17.3kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<4:37:35, 24.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<3:13:45, 35.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<2:16:30, 49.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<1:36:54, 69.9kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<1:08:03, 99.3kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<47:22, 142kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<57:18, 117kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<40:46, 164kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<28:36, 233kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<21:28, 309kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<15:42, 422kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<11:07, 594kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<09:17, 707kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<07:10, 915kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<05:10, 1.26MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<05:08, 1.27MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<04:16, 1.52MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<03:08, 2.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:42, 1.74MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:55, 1.64MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:01, 2.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<02:11, 2.91MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<05:10, 1.23MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:18, 1.48MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:09, 2.00MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:40, 1.71MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:13, 1.95MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:22, 2.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:08, 1.98MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:50, 2.19MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:08, 2.89MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:56, 2.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:41, 2.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:02, 3.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:51, 2.13MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:37, 2.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<01:57, 3.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:48, 2.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:34, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<01:56, 3.08MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:45, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<02:32, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<01:55, 3.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:43, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:36, 2.25MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<01:57, 3.00MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:41, 2.16MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<03:02, 1.92MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:24, 2.40MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<02:36, 2.20MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:26, 2.35MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<01:51, 3.09MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<02:35, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:58, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:19, 2.44MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<01:42, 3.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:17, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:53, 1.94MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:48<02:09, 2.58MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:48, 1.98MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:06, 1.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:26, 2.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:50<01:45, 3.11MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<5:17:12, 17.3kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<3:42:20, 24.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<2:35:02, 35.1kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<1:49:04, 49.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<1:16:49, 70.3kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<53:38, 100kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<38:33, 138kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<27:31, 194kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<19:18, 275kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<14:40, 359kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<10:47, 487kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<07:39, 684kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<06:33, 793kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<05:36, 926kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<04:08, 1.25MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<02:56, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<05:03, 1.01MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:04, 1.26MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<02:58, 1.72MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:15, 1.56MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:48, 1.80MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:04<02:04, 2.42MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:37, 1.91MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:20, 2.13MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:06<01:45, 2.82MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:23, 2.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:10, 2.27MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:38, 2.99MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:17, 2.13MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<02:06, 2.31MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:35, 3.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:13, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:03, 2.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<01:32, 3.07MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:11, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:01, 2.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:31, 3.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:09, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<01:59, 2.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:30, 3.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:08, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<01:57, 2.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:29, 3.07MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:05, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:24, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<01:52, 2.40MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:20<01:21, 3.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<03:06, 1.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:37, 1.69MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:56, 2.27MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:22, 1.84MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:06, 2.07MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:34, 2.75MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:08, 2.01MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:22, 1.81MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:52, 2.28MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:59, 2.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:50, 2.30MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:23, 3.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:56, 2.15MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:46, 2.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:19, 3.10MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:53, 2.17MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:09, 1.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:43, 2.38MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:50, 2.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:41, 2.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:17, 3.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:49, 2.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<02:07, 1.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<01:40, 2.35MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<01:12, 3.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<28:38, 136kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<20:25, 190kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<14:17, 270kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<10:48, 354kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<07:56, 481kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<05:37, 675kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<04:47, 785kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<03:43, 1.01MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<02:41, 1.39MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:44, 1.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:40, 1.38MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:01, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<01:27, 2.50MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:41, 1.35MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:15, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:39, 2.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:59, 1.79MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:45, 2.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<01:18, 2.69MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:43, 2.02MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:55, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:30, 2.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<01:04, 3.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:56, 1.16MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<02:24, 1.42MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:44, 1.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:59, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:04, 1.61MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:35, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<01:08, 2.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<03:09, 1.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:33, 1.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:50, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:01, 1.58MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:04, 1.54MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:35, 2.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<01:08, 2.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:22, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:58, 1.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:27, 2.14MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:43, 1.78MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:50, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:25, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:01, 2.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:02, 1.47MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:43, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:16, 2.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:34, 1.87MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:23, 2.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:02, 2.78MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:23, 2.05MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:16, 2.25MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<00:57, 2.97MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:19, 2.12MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:12, 2.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<00:54, 3.03MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:16, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:10, 2.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<00:52, 3.06MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<01:14, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:24, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:05, 2.41MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<00:48, 3.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:34, 1.65MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:22, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:00, 2.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:17, 1.95MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:09, 2.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:51, 2.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:10, 2.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:01, 2.38MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<00:47, 3.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:34, 4.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<09:24, 254kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<06:48, 350kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<04:45, 494kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<03:50, 604kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<02:54, 793kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<02:04, 1.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:57, 1.15MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:36, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<01:09, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:19, 1.65MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:08, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<00:50, 2.56MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:04, 1.96MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:58, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<00:43, 2.87MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:58, 2.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:06, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:51, 2.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<00:37, 3.23MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:28, 1.34MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:13, 1.60MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<00:53, 2.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:03, 1.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:56, 2.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:41, 2.70MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:54, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:49, 2.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:36, 2.94MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:47, 2.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:13, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:59, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:43, 2.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:54, 1.85MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:48, 2.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:36, 2.74MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:47, 2.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:43, 2.23MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:32, 2.95MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:43, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:49, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:38, 2.38MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:27, 3.23MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:53, 1.67MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:46, 1.92MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:33, 2.59MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:42, 1.98MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:47, 1.79MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:37, 2.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:38, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:34, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:25, 3.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:35, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:40, 1.89MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:52<00:31, 2.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<00:22, 3.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<1:09:47, 17.3kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<48:42, 24.6kB/s]  .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<33:23, 35.1kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<22:55, 49.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<16:04, 70.3kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<11:00, 100kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<07:42, 139kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<05:28, 194kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<03:45, 275kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<02:46, 360kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<02:02, 488kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<01:24, 685kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<01:10, 794kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<01:00, 927kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:43, 1.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:29, 1.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:06, 781kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:51, 999kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:36, 1.38MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:35, 1.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:29, 1.61MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:21, 2.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:24, 1.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:21, 2.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:15, 2.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:19, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:17, 2.22MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:12, 2.94MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:16, 2.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:15, 2.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:10, 3.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:14, 2.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:13, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:09, 3.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:12, 2.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:08, 3.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:10, 2.15MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:12, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:09, 2.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:07, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:05, 3.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:06, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:05, 2.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:03, 3.27MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:22<00:02, 4.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:41, 254kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:29, 350kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:17, 494kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:10, 604kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:07, 792kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:04, 1.11MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:02, 1.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.44MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.91MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 829/400000 [00:00<00:48, 8287.16it/s]  0%|          | 1794/400000 [00:00<00:46, 8653.09it/s]  1%|          | 2723/400000 [00:00<00:44, 8833.99it/s]  1%|          | 3602/400000 [00:00<00:44, 8820.73it/s]  1%|          | 4514/400000 [00:00<00:44, 8907.32it/s]  1%|▏         | 5417/400000 [00:00<00:44, 8942.88it/s]  2%|▏         | 6314/400000 [00:00<00:43, 8948.39it/s]  2%|▏         | 7255/400000 [00:00<00:43, 9080.04it/s]  2%|▏         | 8165/400000 [00:00<00:43, 9084.00it/s]  2%|▏         | 9084/400000 [00:01<00:42, 9113.30it/s]  2%|▏         | 9973/400000 [00:01<00:43, 8976.17it/s]  3%|▎         | 10856/400000 [00:01<00:43, 8930.75it/s]  3%|▎         | 11777/400000 [00:01<00:43, 9012.14it/s]  3%|▎         | 12710/400000 [00:01<00:42, 9103.68it/s]  3%|▎         | 13616/400000 [00:01<00:42, 9075.47it/s]  4%|▎         | 14536/400000 [00:01<00:42, 9110.39it/s]  4%|▍         | 15474/400000 [00:01<00:41, 9188.96it/s]  4%|▍         | 16417/400000 [00:01<00:41, 9259.61it/s]  4%|▍         | 17383/400000 [00:01<00:40, 9375.80it/s]  5%|▍         | 18321/400000 [00:02<00:40, 9323.88it/s]  5%|▍         | 19254/400000 [00:02<00:42, 9046.20it/s]  5%|▌         | 20196/400000 [00:02<00:41, 9154.75it/s]  5%|▌         | 21148/400000 [00:02<00:40, 9259.79it/s]  6%|▌         | 22097/400000 [00:02<00:40, 9327.38it/s]  6%|▌         | 23071/400000 [00:02<00:39, 9445.65it/s]  6%|▌         | 24017/400000 [00:02<00:40, 9389.76it/s]  6%|▌         | 24988/400000 [00:02<00:39, 9483.07it/s]  6%|▋         | 25938/400000 [00:02<00:39, 9369.54it/s]  7%|▋         | 26876/400000 [00:02<00:40, 9308.25it/s]  7%|▋         | 27808/400000 [00:03<00:41, 8950.74it/s]  7%|▋         | 28707/400000 [00:03<00:41, 8843.66it/s]  7%|▋         | 29621/400000 [00:03<00:41, 8929.23it/s]  8%|▊         | 30517/400000 [00:03<00:41, 8845.22it/s]  8%|▊         | 31404/400000 [00:03<00:41, 8807.56it/s]  8%|▊         | 32307/400000 [00:03<00:41, 8871.45it/s]  8%|▊         | 33196/400000 [00:03<00:41, 8870.47it/s]  9%|▊         | 34084/400000 [00:03<00:41, 8712.97it/s]  9%|▊         | 34959/400000 [00:03<00:41, 8722.82it/s]  9%|▉         | 35858/400000 [00:03<00:41, 8798.26it/s]  9%|▉         | 36739/400000 [00:04<00:43, 8412.73it/s]  9%|▉         | 37585/400000 [00:04<00:43, 8319.94it/s] 10%|▉         | 38421/400000 [00:04<00:45, 8030.05it/s] 10%|▉         | 39272/400000 [00:04<00:44, 8166.60it/s] 10%|█         | 40154/400000 [00:04<00:43, 8351.03it/s] 10%|█         | 41022/400000 [00:04<00:42, 8444.65it/s] 10%|█         | 41870/400000 [00:04<00:42, 8391.79it/s] 11%|█         | 42728/400000 [00:04<00:42, 8439.08it/s] 11%|█         | 43607/400000 [00:04<00:41, 8540.34it/s] 11%|█         | 44463/400000 [00:04<00:41, 8500.83it/s] 11%|█▏        | 45341/400000 [00:05<00:41, 8580.63it/s] 12%|█▏        | 46200/400000 [00:05<00:41, 8571.87it/s] 12%|█▏        | 47098/400000 [00:05<00:40, 8690.33it/s] 12%|█▏        | 47990/400000 [00:05<00:40, 8756.57it/s] 12%|█▏        | 48867/400000 [00:05<00:40, 8713.93it/s] 12%|█▏        | 49761/400000 [00:05<00:39, 8780.26it/s] 13%|█▎        | 50640/400000 [00:05<00:40, 8702.91it/s] 13%|█▎        | 51514/400000 [00:05<00:40, 8711.32it/s] 13%|█▎        | 52386/400000 [00:05<00:41, 8442.08it/s] 13%|█▎        | 53233/400000 [00:06<00:41, 8432.14it/s] 14%|█▎        | 54152/400000 [00:06<00:40, 8644.91it/s] 14%|█▍        | 55026/400000 [00:06<00:39, 8671.26it/s] 14%|█▍        | 55911/400000 [00:06<00:39, 8710.68it/s] 14%|█▍        | 56784/400000 [00:06<00:39, 8588.04it/s] 14%|█▍        | 57657/400000 [00:06<00:39, 8629.90it/s] 15%|█▍        | 58532/400000 [00:06<00:39, 8661.75it/s] 15%|█▍        | 59433/400000 [00:06<00:38, 8762.27it/s] 15%|█▌        | 60318/400000 [00:06<00:38, 8786.40it/s] 15%|█▌        | 61260/400000 [00:06<00:37, 8966.94it/s] 16%|█▌        | 62187/400000 [00:07<00:37, 9054.93it/s] 16%|█▌        | 63094/400000 [00:07<00:40, 8355.72it/s] 16%|█▌        | 63948/400000 [00:07<00:39, 8409.30it/s] 16%|█▌        | 64865/400000 [00:07<00:38, 8622.17it/s] 16%|█▋        | 65735/400000 [00:07<00:39, 8538.79it/s] 17%|█▋        | 66664/400000 [00:07<00:38, 8750.93it/s] 17%|█▋        | 67569/400000 [00:07<00:37, 8836.63it/s] 17%|█▋        | 68457/400000 [00:07<00:37, 8844.92it/s] 17%|█▋        | 69362/400000 [00:07<00:37, 8903.80it/s] 18%|█▊        | 70255/400000 [00:07<00:37, 8876.67it/s] 18%|█▊        | 71144/400000 [00:08<00:37, 8787.19it/s] 18%|█▊        | 72032/400000 [00:08<00:37, 8812.75it/s] 18%|█▊        | 72915/400000 [00:08<00:37, 8780.72it/s] 18%|█▊        | 73794/400000 [00:08<00:38, 8565.48it/s] 19%|█▊        | 74680/400000 [00:08<00:37, 8649.80it/s] 19%|█▉        | 75617/400000 [00:08<00:36, 8851.87it/s] 19%|█▉        | 76505/400000 [00:08<00:36, 8803.86it/s] 19%|█▉        | 77418/400000 [00:08<00:36, 8898.09it/s] 20%|█▉        | 78318/400000 [00:08<00:36, 8926.68it/s] 20%|█▉        | 79212/400000 [00:08<00:36, 8896.62it/s] 20%|██        | 80103/400000 [00:09<00:36, 8772.03it/s] 20%|██        | 80982/400000 [00:09<00:37, 8605.43it/s] 20%|██        | 81844/400000 [00:09<00:37, 8543.13it/s] 21%|██        | 82709/400000 [00:09<00:37, 8574.17it/s] 21%|██        | 83575/400000 [00:09<00:36, 8598.75it/s] 21%|██        | 84436/400000 [00:09<00:36, 8601.38it/s] 21%|██▏       | 85349/400000 [00:09<00:35, 8753.39it/s] 22%|██▏       | 86276/400000 [00:09<00:35, 8901.12it/s] 22%|██▏       | 87209/400000 [00:09<00:34, 9025.14it/s] 22%|██▏       | 88113/400000 [00:09<00:34, 9016.70it/s] 22%|██▏       | 89016/400000 [00:10<00:34, 8902.87it/s] 22%|██▏       | 89908/400000 [00:10<00:35, 8729.38it/s] 23%|██▎       | 90783/400000 [00:10<00:35, 8712.30it/s] 23%|██▎       | 91668/400000 [00:10<00:35, 8751.85it/s] 23%|██▎       | 92544/400000 [00:10<00:35, 8703.34it/s] 23%|██▎       | 93441/400000 [00:10<00:34, 8779.12it/s] 24%|██▎       | 94320/400000 [00:10<00:35, 8722.97it/s] 24%|██▍       | 95193/400000 [00:10<00:34, 8718.90it/s] 24%|██▍       | 96066/400000 [00:10<00:35, 8682.18it/s] 24%|██▍       | 96965/400000 [00:10<00:34, 8771.49it/s] 24%|██▍       | 97843/400000 [00:11<00:34, 8720.25it/s] 25%|██▍       | 98716/400000 [00:11<00:35, 8600.69it/s] 25%|██▍       | 99588/400000 [00:11<00:34, 8636.02it/s] 25%|██▌       | 100453/400000 [00:11<00:35, 8398.33it/s] 25%|██▌       | 101295/400000 [00:11<00:36, 8211.48it/s] 26%|██▌       | 102178/400000 [00:11<00:35, 8385.47it/s] 26%|██▌       | 103027/400000 [00:11<00:35, 8415.69it/s] 26%|██▌       | 103911/400000 [00:11<00:34, 8536.03it/s] 26%|██▌       | 104787/400000 [00:11<00:34, 8601.91it/s] 26%|██▋       | 105659/400000 [00:12<00:34, 8636.18it/s] 27%|██▋       | 106524/400000 [00:12<00:36, 8132.10it/s] 27%|██▋       | 107411/400000 [00:12<00:35, 8338.11it/s] 27%|██▋       | 108288/400000 [00:12<00:34, 8462.22it/s] 27%|██▋       | 109139/400000 [00:12<00:34, 8466.02it/s] 28%|██▊       | 110032/400000 [00:12<00:33, 8598.46it/s] 28%|██▊       | 110922/400000 [00:12<00:33, 8686.59it/s] 28%|██▊       | 111793/400000 [00:12<00:33, 8676.42it/s] 28%|██▊       | 112663/400000 [00:12<00:34, 8402.00it/s] 28%|██▊       | 113507/400000 [00:12<00:34, 8351.77it/s] 29%|██▊       | 114345/400000 [00:13<00:34, 8322.48it/s] 29%|██▉       | 115179/400000 [00:13<00:34, 8229.53it/s] 29%|██▉       | 116004/400000 [00:13<00:36, 7844.82it/s] 29%|██▉       | 116819/400000 [00:13<00:35, 7932.98it/s] 29%|██▉       | 117633/400000 [00:13<00:35, 7993.49it/s] 30%|██▉       | 118458/400000 [00:13<00:34, 8066.21it/s] 30%|██▉       | 119296/400000 [00:13<00:34, 8155.44it/s] 30%|███       | 120148/400000 [00:13<00:33, 8259.33it/s] 30%|███       | 120980/400000 [00:13<00:33, 8276.36it/s] 30%|███       | 121815/400000 [00:13<00:33, 8295.11it/s] 31%|███       | 122646/400000 [00:14<00:34, 8121.38it/s] 31%|███       | 123545/400000 [00:14<00:33, 8363.07it/s] 31%|███       | 124428/400000 [00:14<00:32, 8494.52it/s] 31%|███▏      | 125280/400000 [00:14<00:32, 8362.28it/s] 32%|███▏      | 126155/400000 [00:14<00:32, 8472.68it/s] 32%|███▏      | 127005/400000 [00:14<00:32, 8364.63it/s] 32%|███▏      | 127859/400000 [00:14<00:32, 8415.86it/s] 32%|███▏      | 128718/400000 [00:14<00:32, 8467.13it/s] 32%|███▏      | 129566/400000 [00:14<00:32, 8356.14it/s] 33%|███▎      | 130423/400000 [00:15<00:32, 8417.41it/s] 33%|███▎      | 131266/400000 [00:15<00:32, 8397.88it/s] 33%|███▎      | 132171/400000 [00:15<00:31, 8583.18it/s] 33%|███▎      | 133052/400000 [00:15<00:30, 8647.37it/s] 33%|███▎      | 133918/400000 [00:15<00:31, 8563.41it/s] 34%|███▎      | 134776/400000 [00:15<00:31, 8544.12it/s] 34%|███▍      | 135632/400000 [00:15<00:30, 8540.18it/s] 34%|███▍      | 136496/400000 [00:15<00:30, 8567.63it/s] 34%|███▍      | 137387/400000 [00:15<00:30, 8666.57it/s] 35%|███▍      | 138262/400000 [00:15<00:30, 8690.70it/s] 35%|███▍      | 139167/400000 [00:16<00:29, 8794.11it/s] 35%|███▌      | 140047/400000 [00:16<00:29, 8709.88it/s] 35%|███▌      | 140919/400000 [00:16<00:29, 8670.47it/s] 35%|███▌      | 141792/400000 [00:16<00:29, 8686.12it/s] 36%|███▌      | 142665/400000 [00:16<00:29, 8697.50it/s] 36%|███▌      | 143540/400000 [00:16<00:29, 8710.72it/s] 36%|███▌      | 144418/400000 [00:16<00:29, 8728.89it/s] 36%|███▋      | 145292/400000 [00:16<00:29, 8541.18it/s] 37%|███▋      | 146181/400000 [00:16<00:29, 8641.94it/s] 37%|███▋      | 147047/400000 [00:16<00:29, 8609.22it/s] 37%|███▋      | 147932/400000 [00:17<00:29, 8677.47it/s] 37%|███▋      | 148810/400000 [00:17<00:28, 8707.81it/s] 37%|███▋      | 149682/400000 [00:17<00:29, 8422.19it/s] 38%|███▊      | 150531/400000 [00:17<00:29, 8441.70it/s] 38%|███▊      | 151400/400000 [00:17<00:29, 8513.74it/s] 38%|███▊      | 152319/400000 [00:17<00:28, 8705.49it/s] 38%|███▊      | 153192/400000 [00:17<00:28, 8666.81it/s] 39%|███▊      | 154096/400000 [00:17<00:28, 8773.19it/s] 39%|███▊      | 154984/400000 [00:17<00:27, 8802.49it/s] 39%|███▉      | 155866/400000 [00:17<00:27, 8769.13it/s] 39%|███▉      | 156744/400000 [00:18<00:28, 8663.96it/s] 39%|███▉      | 157612/400000 [00:18<00:29, 8315.34it/s] 40%|███▉      | 158495/400000 [00:18<00:28, 8461.72it/s] 40%|███▉      | 159387/400000 [00:18<00:28, 8592.32it/s] 40%|████      | 160271/400000 [00:18<00:27, 8664.68it/s] 40%|████      | 161140/400000 [00:18<00:27, 8652.73it/s] 41%|████      | 162007/400000 [00:18<00:27, 8638.24it/s] 41%|████      | 162881/400000 [00:18<00:27, 8666.79it/s] 41%|████      | 163756/400000 [00:18<00:27, 8689.83it/s] 41%|████      | 164626/400000 [00:18<00:27, 8613.59it/s] 41%|████▏     | 165496/400000 [00:19<00:27, 8637.89it/s] 42%|████▏     | 166361/400000 [00:19<00:27, 8573.39it/s] 42%|████▏     | 167233/400000 [00:19<00:27, 8615.91it/s] 42%|████▏     | 168095/400000 [00:19<00:27, 8448.35it/s] 42%|████▏     | 168962/400000 [00:19<00:27, 8512.35it/s] 42%|████▏     | 169841/400000 [00:19<00:26, 8592.02it/s] 43%|████▎     | 170701/400000 [00:19<00:26, 8543.38it/s] 43%|████▎     | 171556/400000 [00:19<00:26, 8540.05it/s] 43%|████▎     | 172439/400000 [00:19<00:26, 8622.66it/s] 43%|████▎     | 173322/400000 [00:19<00:26, 8683.12it/s] 44%|████▎     | 174221/400000 [00:20<00:25, 8771.06it/s] 44%|████▍     | 175099/400000 [00:20<00:25, 8737.91it/s] 44%|████▍     | 175984/400000 [00:20<00:25, 8769.91it/s] 44%|████▍     | 176862/400000 [00:20<00:25, 8636.62it/s] 44%|████▍     | 177727/400000 [00:20<00:25, 8589.49it/s] 45%|████▍     | 178587/400000 [00:20<00:26, 8492.21it/s] 45%|████▍     | 179437/400000 [00:20<00:26, 8287.50it/s] 45%|████▌     | 180317/400000 [00:20<00:26, 8432.33it/s] 45%|████▌     | 181176/400000 [00:20<00:25, 8478.11it/s] 46%|████▌     | 182052/400000 [00:20<00:25, 8560.02it/s] 46%|████▌     | 182910/400000 [00:21<00:25, 8547.21it/s] 46%|████▌     | 183766/400000 [00:21<00:25, 8538.10it/s] 46%|████▌     | 184621/400000 [00:21<00:25, 8517.07it/s] 46%|████▋     | 185502/400000 [00:21<00:24, 8602.28it/s] 47%|████▋     | 186380/400000 [00:21<00:24, 8651.53it/s] 47%|████▋     | 187246/400000 [00:21<00:24, 8607.25it/s] 47%|████▋     | 188131/400000 [00:21<00:24, 8678.48it/s] 47%|████▋     | 189000/400000 [00:21<00:24, 8671.57it/s] 47%|████▋     | 189868/400000 [00:21<00:24, 8541.98it/s] 48%|████▊     | 190738/400000 [00:22<00:24, 8586.51it/s] 48%|████▊     | 191598/400000 [00:22<00:24, 8538.75it/s] 48%|████▊     | 192453/400000 [00:22<00:24, 8438.34it/s] 48%|████▊     | 193345/400000 [00:22<00:24, 8575.55it/s] 49%|████▊     | 194204/400000 [00:22<00:24, 8396.66it/s] 49%|████▉     | 195101/400000 [00:22<00:23, 8560.69it/s] 49%|████▉     | 195959/400000 [00:22<00:23, 8557.19it/s] 49%|████▉     | 196816/400000 [00:22<00:23, 8490.08it/s] 49%|████▉     | 197750/400000 [00:22<00:23, 8727.73it/s] 50%|████▉     | 198656/400000 [00:22<00:22, 8822.98it/s] 50%|████▉     | 199553/400000 [00:23<00:22, 8864.72it/s] 50%|█████     | 200441/400000 [00:23<00:22, 8788.66it/s] 50%|█████     | 201373/400000 [00:23<00:22, 8799.06it/s] 51%|█████     | 202261/400000 [00:23<00:22, 8822.41it/s] 51%|█████     | 203170/400000 [00:23<00:22, 8898.38it/s] 51%|█████     | 204063/400000 [00:23<00:21, 8907.06it/s] 51%|█████     | 204971/400000 [00:23<00:21, 8955.59it/s] 51%|█████▏    | 205873/400000 [00:23<00:21, 8974.16it/s] 52%|█████▏    | 206771/400000 [00:23<00:21, 8968.56it/s] 52%|█████▏    | 207700/400000 [00:23<00:21, 9060.62it/s] 52%|█████▏    | 208627/400000 [00:24<00:20, 9122.33it/s] 52%|█████▏    | 209540/400000 [00:24<00:21, 8923.98it/s] 53%|█████▎    | 210434/400000 [00:24<00:21, 8897.41it/s] 53%|█████▎    | 211356/400000 [00:24<00:20, 8989.85it/s] 53%|█████▎    | 212264/400000 [00:24<00:20, 9015.84it/s] 53%|█████▎    | 213167/400000 [00:24<00:20, 9016.92it/s] 54%|█████▎    | 214070/400000 [00:24<00:20, 8977.24it/s] 54%|█████▎    | 214969/400000 [00:24<00:20, 8945.35it/s] 54%|█████▍    | 215864/400000 [00:24<00:20, 8877.50it/s] 54%|█████▍    | 216766/400000 [00:24<00:20, 8918.29it/s] 54%|█████▍    | 217659/400000 [00:25<00:20, 8865.04it/s] 55%|█████▍    | 218546/400000 [00:25<00:20, 8849.81it/s] 55%|█████▍    | 219457/400000 [00:25<00:20, 8925.99it/s] 55%|█████▌    | 220428/400000 [00:25<00:19, 9145.18it/s] 55%|█████▌    | 221345/400000 [00:25<00:19, 9122.97it/s] 56%|█████▌    | 222267/400000 [00:25<00:19, 9150.20it/s] 56%|█████▌    | 223189/400000 [00:25<00:19, 9170.04it/s] 56%|█████▌    | 224107/400000 [00:25<00:19, 9151.82it/s] 56%|█████▋    | 225026/400000 [00:25<00:19, 9161.60it/s] 56%|█████▋    | 225952/400000 [00:25<00:18, 9189.64it/s] 57%|█████▋    | 226872/400000 [00:26<00:18, 9132.81it/s] 57%|█████▋    | 227791/400000 [00:26<00:18, 9148.84it/s] 57%|█████▋    | 228725/400000 [00:26<00:18, 9203.78it/s] 57%|█████▋    | 229646/400000 [00:26<00:18, 9183.68it/s] 58%|█████▊    | 230565/400000 [00:26<00:19, 8902.23it/s] 58%|█████▊    | 231532/400000 [00:26<00:18, 9117.33it/s] 58%|█████▊    | 232457/400000 [00:26<00:18, 9156.57it/s] 58%|█████▊    | 233375/400000 [00:26<00:18, 9105.51it/s] 59%|█████▊    | 234287/400000 [00:26<00:18, 8920.01it/s] 59%|█████▉    | 235181/400000 [00:26<00:18, 8676.03it/s] 59%|█████▉    | 236061/400000 [00:27<00:18, 8712.02it/s] 59%|█████▉    | 236935/400000 [00:27<00:18, 8708.94it/s] 59%|█████▉    | 237808/400000 [00:27<00:18, 8655.96it/s] 60%|█████▉    | 238800/400000 [00:27<00:17, 8999.94it/s] 60%|█████▉    | 239736/400000 [00:27<00:17, 9103.79it/s] 60%|██████    | 240680/400000 [00:27<00:17, 9199.97it/s] 60%|██████    | 241610/400000 [00:27<00:17, 9227.35it/s] 61%|██████    | 242545/400000 [00:27<00:16, 9262.63it/s] 61%|██████    | 243473/400000 [00:27<00:17, 9188.86it/s] 61%|██████    | 244393/400000 [00:27<00:16, 9191.21it/s] 61%|██████▏   | 245331/400000 [00:28<00:16, 9244.96it/s] 62%|██████▏   | 246257/400000 [00:28<00:16, 9159.62it/s] 62%|██████▏   | 247202/400000 [00:28<00:16, 9241.84it/s] 62%|██████▏   | 248190/400000 [00:28<00:16, 9423.64it/s] 62%|██████▏   | 249134/400000 [00:28<00:16, 9177.14it/s] 63%|██████▎   | 250070/400000 [00:28<00:16, 9230.54it/s] 63%|██████▎   | 251024/400000 [00:28<00:15, 9319.87it/s] 63%|██████▎   | 251983/400000 [00:28<00:15, 9398.70it/s] 63%|██████▎   | 252944/400000 [00:28<00:15, 9458.19it/s] 63%|██████▎   | 253891/400000 [00:29<00:15, 9382.83it/s] 64%|██████▎   | 254831/400000 [00:29<00:15, 9147.37it/s] 64%|██████▍   | 255748/400000 [00:29<00:16, 9000.29it/s] 64%|██████▍   | 256688/400000 [00:29<00:15, 9115.93it/s] 64%|██████▍   | 257602/400000 [00:29<00:15, 9037.84it/s] 65%|██████▍   | 258508/400000 [00:29<00:15, 8978.37it/s] 65%|██████▍   | 259407/400000 [00:29<00:15, 8974.91it/s] 65%|██████▌   | 260339/400000 [00:29<00:15, 9075.08it/s] 65%|██████▌   | 261280/400000 [00:29<00:15, 9170.53it/s] 66%|██████▌   | 262198/400000 [00:29<00:15, 9108.54it/s] 66%|██████▌   | 263110/400000 [00:30<00:15, 8889.81it/s] 66%|██████▌   | 264001/400000 [00:30<00:15, 8830.59it/s] 66%|██████▌   | 264910/400000 [00:30<00:15, 8905.68it/s] 66%|██████▋   | 265850/400000 [00:30<00:14, 9046.67it/s] 67%|██████▋   | 266768/400000 [00:30<00:14, 9083.93it/s] 67%|██████▋   | 267707/400000 [00:30<00:14, 9171.77it/s] 67%|██████▋   | 268626/400000 [00:30<00:14, 9131.24it/s] 67%|██████▋   | 269540/400000 [00:30<00:14, 9055.33it/s] 68%|██████▊   | 270464/400000 [00:30<00:14, 9108.78it/s] 68%|██████▊   | 271376/400000 [00:30<00:14, 9008.35it/s] 68%|██████▊   | 272278/400000 [00:31<00:14, 8818.21it/s] 68%|██████▊   | 273224/400000 [00:31<00:14, 9000.68it/s] 69%|██████▊   | 274187/400000 [00:31<00:13, 9179.24it/s] 69%|██████▉   | 275172/400000 [00:31<00:13, 9367.79it/s] 69%|██████▉   | 276144/400000 [00:31<00:13, 9467.55it/s] 69%|██████▉   | 277093/400000 [00:31<00:13, 9376.09it/s] 70%|██████▉   | 278056/400000 [00:31<00:12, 9450.29it/s] 70%|██████▉   | 279003/400000 [00:31<00:13, 9202.38it/s] 70%|███████   | 280024/400000 [00:31<00:12, 9482.70it/s] 70%|███████   | 280977/400000 [00:31<00:12, 9477.46it/s] 70%|███████   | 281928/400000 [00:32<00:12, 9437.78it/s] 71%|███████   | 282920/400000 [00:32<00:12, 9575.13it/s] 71%|███████   | 283880/400000 [00:32<00:12, 9280.48it/s] 71%|███████   | 284812/400000 [00:32<00:12, 9176.54it/s] 71%|███████▏  | 285733/400000 [00:32<00:12, 9070.80it/s] 72%|███████▏  | 286686/400000 [00:32<00:12, 9202.64it/s] 72%|███████▏  | 287675/400000 [00:32<00:11, 9397.45it/s] 72%|███████▏  | 288627/400000 [00:32<00:11, 9432.89it/s] 72%|███████▏  | 289620/400000 [00:32<00:11, 9575.82it/s] 73%|███████▎  | 290580/400000 [00:32<00:11, 9539.05it/s] 73%|███████▎  | 291536/400000 [00:33<00:11, 9176.58it/s] 73%|███████▎  | 292498/400000 [00:33<00:11, 9305.23it/s] 73%|███████▎  | 293432/400000 [00:33<00:11, 9245.13it/s] 74%|███████▎  | 294359/400000 [00:33<00:11, 9220.52it/s] 74%|███████▍  | 295283/400000 [00:33<00:11, 9146.04it/s] 74%|███████▍  | 296224/400000 [00:33<00:11, 9222.14it/s] 74%|███████▍  | 297148/400000 [00:33<00:11, 9153.18it/s] 75%|███████▍  | 298072/400000 [00:33<00:11, 9176.54it/s] 75%|███████▍  | 299007/400000 [00:33<00:10, 9227.60it/s] 75%|███████▍  | 299931/400000 [00:34<00:11, 8954.86it/s] 75%|███████▌  | 300829/400000 [00:34<00:11, 8824.33it/s] 75%|███████▌  | 301784/400000 [00:34<00:10, 9029.43it/s] 76%|███████▌  | 302706/400000 [00:34<00:10, 9085.23it/s] 76%|███████▌  | 303706/400000 [00:34<00:10, 9340.27it/s] 76%|███████▌  | 304644/400000 [00:34<00:10, 9335.90it/s] 76%|███████▋  | 305596/400000 [00:34<00:10, 9389.27it/s] 77%|███████▋  | 306569/400000 [00:34<00:09, 9488.19it/s] 77%|███████▋  | 307520/400000 [00:34<00:09, 9449.98it/s] 77%|███████▋  | 308466/400000 [00:34<00:09, 9404.17it/s] 77%|███████▋  | 309414/400000 [00:35<00:09, 9424.89it/s] 78%|███████▊  | 310363/400000 [00:35<00:09, 9441.48it/s] 78%|███████▊  | 311336/400000 [00:35<00:09, 9523.82it/s] 78%|███████▊  | 312321/400000 [00:35<00:09, 9616.09it/s] 78%|███████▊  | 313302/400000 [00:35<00:08, 9671.76it/s] 79%|███████▊  | 314270/400000 [00:35<00:09, 9520.07it/s] 79%|███████▉  | 315239/400000 [00:35<00:08, 9570.08it/s] 79%|███████▉  | 316197/400000 [00:35<00:08, 9427.98it/s] 79%|███████▉  | 317141/400000 [00:35<00:08, 9342.44it/s] 80%|███████▉  | 318077/400000 [00:35<00:08, 9222.23it/s] 80%|███████▉  | 319001/400000 [00:36<00:08, 9071.82it/s] 80%|███████▉  | 319910/400000 [00:36<00:09, 8781.35it/s] 80%|████████  | 320902/400000 [00:36<00:08, 9094.51it/s] 80%|████████  | 321852/400000 [00:36<00:08, 9211.00it/s] 81%|████████  | 322796/400000 [00:36<00:08, 9277.86it/s] 81%|████████  | 323727/400000 [00:36<00:08, 9195.58it/s] 81%|████████  | 324649/400000 [00:36<00:08, 9142.91it/s] 81%|████████▏ | 325582/400000 [00:36<00:08, 9196.51it/s] 82%|████████▏ | 326552/400000 [00:36<00:07, 9341.86it/s] 82%|████████▏ | 327534/400000 [00:36<00:07, 9477.97it/s] 82%|████████▏ | 328484/400000 [00:37<00:07, 9314.04it/s] 82%|████████▏ | 329418/400000 [00:37<00:07, 9198.33it/s] 83%|████████▎ | 330346/400000 [00:37<00:07, 9221.89it/s] 83%|████████▎ | 331271/400000 [00:37<00:07, 9229.02it/s] 83%|████████▎ | 332228/400000 [00:37<00:07, 9325.95it/s] 83%|████████▎ | 333162/400000 [00:37<00:07, 9290.57it/s] 84%|████████▎ | 334092/400000 [00:37<00:07, 9189.10it/s] 84%|████████▍ | 335020/400000 [00:37<00:07, 9213.81it/s] 84%|████████▍ | 335945/400000 [00:37<00:06, 9223.74it/s] 84%|████████▍ | 336888/400000 [00:37<00:06, 9282.81it/s] 84%|████████▍ | 337817/400000 [00:38<00:06, 8918.00it/s] 85%|████████▍ | 338730/400000 [00:38<00:06, 8979.41it/s] 85%|████████▍ | 339655/400000 [00:38<00:06, 9058.69it/s] 85%|████████▌ | 340642/400000 [00:38<00:06, 9286.79it/s] 85%|████████▌ | 341607/400000 [00:38<00:06, 9390.72it/s] 86%|████████▌ | 342570/400000 [00:38<00:06, 9458.30it/s] 86%|████████▌ | 343518/400000 [00:38<00:05, 9421.21it/s] 86%|████████▌ | 344473/400000 [00:38<00:05, 9457.94it/s] 86%|████████▋ | 345439/400000 [00:38<00:05, 9515.77it/s] 87%|████████▋ | 346392/400000 [00:39<00:05, 9499.69it/s] 87%|████████▋ | 347343/400000 [00:39<00:05, 9273.12it/s] 87%|████████▋ | 348272/400000 [00:39<00:05, 9100.21it/s] 87%|████████▋ | 349220/400000 [00:39<00:05, 9210.32it/s] 88%|████████▊ | 350179/400000 [00:39<00:05, 9320.94it/s] 88%|████████▊ | 351170/400000 [00:39<00:05, 9487.36it/s] 88%|████████▊ | 352121/400000 [00:39<00:05, 9405.87it/s] 88%|████████▊ | 353063/400000 [00:39<00:05, 9311.61it/s] 89%|████████▊ | 354029/400000 [00:39<00:04, 9411.08it/s] 89%|████████▊ | 354972/400000 [00:39<00:04, 9362.46it/s] 89%|████████▉ | 355922/400000 [00:40<00:04, 9401.59it/s] 89%|████████▉ | 356863/400000 [00:40<00:04, 9300.28it/s] 89%|████████▉ | 357794/400000 [00:40<00:04, 8985.55it/s] 90%|████████▉ | 358696/400000 [00:40<00:04, 8953.73it/s] 90%|████████▉ | 359610/400000 [00:40<00:04, 9007.98it/s] 90%|█████████ | 360588/400000 [00:40<00:04, 9224.01it/s] 90%|█████████ | 361560/400000 [00:40<00:04, 9364.59it/s] 91%|█████████ | 362499/400000 [00:40<00:04, 9351.97it/s] 91%|█████████ | 363436/400000 [00:40<00:03, 9275.16it/s] 91%|█████████ | 364382/400000 [00:40<00:03, 9329.32it/s] 91%|█████████▏| 365316/400000 [00:41<00:03, 9216.25it/s] 92%|█████████▏| 366266/400000 [00:41<00:03, 9298.69it/s] 92%|█████████▏| 367197/400000 [00:41<00:03, 9289.84it/s] 92%|█████████▏| 368127/400000 [00:41<00:03, 9215.98it/s] 92%|█████████▏| 369128/400000 [00:41<00:03, 9437.84it/s] 93%|█████████▎| 370098/400000 [00:41<00:03, 9514.22it/s] 93%|█████████▎| 371051/400000 [00:41<00:03, 9479.67it/s] 93%|█████████▎| 372000/400000 [00:41<00:02, 9425.60it/s] 93%|█████████▎| 372944/400000 [00:41<00:02, 9287.75it/s] 93%|█████████▎| 373880/400000 [00:41<00:02, 9308.48it/s] 94%|█████████▎| 374812/400000 [00:42<00:02, 9272.12it/s] 94%|█████████▍| 375784/400000 [00:42<00:02, 9401.98it/s] 94%|█████████▍| 376726/400000 [00:42<00:02, 9019.49it/s] 94%|█████████▍| 377668/400000 [00:42<00:02, 9135.81it/s] 95%|█████████▍| 378660/400000 [00:42<00:02, 9357.29it/s] 95%|█████████▍| 379600/400000 [00:42<00:02, 9326.20it/s] 95%|█████████▌| 380544/400000 [00:42<00:02, 9349.96it/s] 95%|█████████▌| 381481/400000 [00:42<00:02, 9101.16it/s] 96%|█████████▌| 382394/400000 [00:42<00:01, 8951.92it/s] 96%|█████████▌| 383300/400000 [00:43<00:01, 8983.17it/s] 96%|█████████▌| 384237/400000 [00:43<00:01, 9095.15it/s] 96%|█████████▋| 385149/400000 [00:43<00:01, 8935.74it/s] 97%|█████████▋| 386115/400000 [00:43<00:01, 9140.63it/s] 97%|█████████▋| 387064/400000 [00:43<00:01, 9240.95it/s] 97%|█████████▋| 388005/400000 [00:43<00:01, 9289.70it/s] 97%|█████████▋| 388936/400000 [00:43<00:01, 9273.20it/s] 97%|█████████▋| 389885/400000 [00:43<00:01, 9336.59it/s] 98%|█████████▊| 390861/400000 [00:43<00:00, 9458.52it/s] 98%|█████████▊| 391869/400000 [00:43<00:00, 9633.29it/s] 98%|█████████▊| 392834/400000 [00:44<00:00, 9607.45it/s] 98%|█████████▊| 393796/400000 [00:44<00:00, 9537.24it/s] 99%|█████████▊| 394751/400000 [00:44<00:00, 9357.99it/s] 99%|█████████▉| 395722/400000 [00:44<00:00, 9458.64it/s] 99%|█████████▉| 396690/400000 [00:44<00:00, 9522.19it/s] 99%|█████████▉| 397644/400000 [00:44<00:00, 9373.69it/s]100%|█████████▉| 398583/400000 [00:44<00:00, 9194.25it/s]100%|█████████▉| 399505/400000 [00:44<00:00, 9148.18it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8929.90it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f8afb86ec50> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011130676675422679 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.010915767588344305 	 Accuracy: 67

  model saves at 67% accuracy 

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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/matchzoo_models.py", line 241, in __init__
    mpars =json_norm(model_pars['model_pars'])
KeyError: 'model_pars'
python /home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py --do nlp_reuters 

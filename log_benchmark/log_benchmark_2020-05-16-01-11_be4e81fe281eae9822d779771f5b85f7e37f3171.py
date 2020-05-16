
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/be4e81fe281eae9822d779771f5b85f7e37f3171', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'be4e81fe281eae9822d779771f5b85f7e37f3171', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/be4e81fe281eae9822d779771f5b85f7e37f3171

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/be4e81fe281eae9822d779771f5b85f7e37f3171

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f7825cbbfd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-16 01:12:14.356864
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-16 01:12:14.360923
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-16 01:12:14.364932
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-16 01:12:14.368955
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f7831a85400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 356678.8125
Epoch 2/10

1/1 [==============================] - 0s 103ms/step - loss: 316699.2500
Epoch 3/10

1/1 [==============================] - 0s 98ms/step - loss: 234743.6250
Epoch 4/10

1/1 [==============================] - 0s 97ms/step - loss: 172480.0312
Epoch 5/10

1/1 [==============================] - 0s 98ms/step - loss: 123390.6719
Epoch 6/10

1/1 [==============================] - 0s 101ms/step - loss: 86512.9609
Epoch 7/10

1/1 [==============================] - 0s 98ms/step - loss: 60445.3867
Epoch 8/10

1/1 [==============================] - 0s 99ms/step - loss: 42969.6055
Epoch 9/10

1/1 [==============================] - 0s 96ms/step - loss: 31409.0938
Epoch 10/10

1/1 [==============================] - 0s 102ms/step - loss: 23670.9590

  #### Inference Need return ypred, ytrue ######################### 
[[ 2.08681226e-01 -5.98177791e-01  5.12634158e-01  1.79536119e-01
  -5.20299137e-01  3.67062390e-01 -1.35090858e-01 -9.26131487e-01
  -8.09109688e-01 -1.20193076e+00 -4.54632014e-01 -2.65487254e-01
  -5.21275401e-03 -5.37927270e-01  7.20920935e-02 -1.28430712e+00
   1.04946166e-01 -8.93544257e-02 -1.46517485e-01 -7.69315004e-01
  -8.65060836e-02  1.05983210e+00 -9.63601112e-01 -4.82584536e-01
   1.06832898e+00  3.02151442e-01  7.80714899e-02  6.16577387e-01
  -8.50259602e-01  5.65169394e-01  1.51636958e-01 -3.55331957e-01
   3.65348637e-01  1.58366188e-01 -9.78945494e-01 -3.31728756e-01
  -7.01138735e-01 -1.08264945e-01  5.53630471e-01 -2.88459361e-01
   1.48499385e-01  3.28083038e-01 -1.46995187e-02  1.87781066e-01
   4.17794585e-01  1.02932417e+00  2.89552212e-01  4.95321751e-02
  -4.40554023e-01  8.48780155e-01 -5.40006697e-01 -4.56553519e-01
   1.97170377e-01  1.96864933e-01 -6.48122728e-01  5.72989106e-01
  -7.62297869e-01  6.28277063e-01 -7.44101286e-01  1.10199165e+00
   3.77937764e-01  1.54518470e-01  8.18827271e-01  6.97511077e-01
   4.42208201e-02  6.86525702e-01 -5.71850538e-01 -8.15083802e-01
  -4.17791963e-01 -1.25164241e-01 -4.79385883e-01  9.10825849e-01
   1.03279434e-01 -6.63187504e-01  9.34212863e-01 -1.86968982e-01
  -3.63138020e-02 -1.08221996e+00  8.31425726e-01 -3.14529240e-03
   1.52887374e-01  5.87535560e-01  2.78646380e-01 -3.74588966e-01
  -3.89968455e-01  2.11348712e-01 -4.15672511e-01  2.45031655e-01
  -4.72534060e-01  3.46414894e-01 -2.49473210e-02 -3.32447618e-01
   5.68027198e-02  7.91216969e-01  1.58806235e-01 -2.38524362e-01
  -5.24474382e-02  9.16147470e-01 -2.03184009e-01  6.40659630e-02
   2.00015828e-01  1.26023889e-01  2.08476216e-01  2.29278088e-01
   1.64829940e-01 -7.76975751e-01 -1.05723631e+00  1.99433282e-01
  -9.33597267e-01 -1.98993236e-01 -8.99196982e-01  7.43951917e-01
   7.37717211e-01  4.57790554e-01 -6.84141338e-01  8.53067756e-01
   3.67933363e-01  4.34213489e-01 -9.43530381e-01  1.14700884e-01
  -1.17830709e-01  3.55984235e+00  3.94632578e+00  3.72081447e+00
   3.35119867e+00  3.64898014e+00  4.11587143e+00  3.41255808e+00
   4.25560665e+00  3.12437987e+00  3.41942477e+00  2.97576904e+00
   3.77522349e+00  3.04334569e+00  3.98763275e+00  4.05824709e+00
   4.37547064e+00  3.23942542e+00  2.49831438e+00  3.57757998e+00
   3.85850501e+00  4.43241549e+00  4.35303116e+00  2.55842900e+00
   3.37754869e+00  3.77703452e+00  3.21039486e+00  3.74106359e+00
   4.16312313e+00  4.03710604e+00  3.93172431e+00  3.17944646e+00
   3.68784928e+00  4.46079731e+00  4.64663935e+00  4.04384899e+00
   2.77419281e+00  3.57179022e+00  3.47472954e+00  3.70897579e+00
   3.59520960e+00  2.58489633e+00  3.25576496e+00  3.77303910e+00
   4.22103643e+00  4.17023659e+00  4.72242260e+00  3.92314458e+00
   4.10015440e+00  3.53307819e+00  3.84556293e+00  3.94484353e+00
   4.28669691e+00  4.01146603e+00  3.67119431e+00  3.05998540e+00
   3.14621520e+00  3.69135761e+00  3.46745491e+00  3.11535597e+00
   7.47734427e-01  1.85100043e+00  1.33829427e+00  5.69555223e-01
   3.38325262e-01  1.13520133e+00  1.04200077e+00  1.38962090e+00
   1.76551604e+00  1.25805974e+00  4.92587686e-01  8.39904487e-01
   1.17003942e+00  7.69302785e-01  6.39350533e-01  1.93519688e+00
   1.24741554e+00  4.11728263e-01  1.32090950e+00  7.14634180e-01
   5.53717613e-01  9.73484814e-01  9.82315242e-01  1.16574585e+00
   4.61757779e-01  1.92072284e+00  4.37059700e-01  1.45387864e+00
   7.00390577e-01  6.61862731e-01  1.82898533e+00  5.49928129e-01
   1.79572129e+00  4.13735271e-01  1.68703723e+00  9.33536172e-01
   1.29693949e+00  6.82807803e-01  7.19582558e-01  5.87486446e-01
   9.85271156e-01  5.21007299e-01  5.82005918e-01  1.60364723e+00
   1.48893571e+00  1.74394739e+00  5.52569747e-01  1.11182928e+00
   6.53129160e-01  1.02981031e+00  1.34943056e+00  1.32065034e+00
   1.28518987e+00  1.02352560e+00  6.94671094e-01  8.52413774e-01
   4.33040738e-01  1.35855567e+00  1.65878689e+00  8.20840001e-01
   7.01451123e-01  2.21193171e+00  8.00665736e-01  1.51452136e+00
   6.02332294e-01  8.05258572e-01  1.25200975e+00  3.37286532e-01
   6.37911916e-01  1.57800019e+00  7.52294004e-01  6.53934419e-01
   7.83097029e-01  9.75740790e-01  2.86861658e-01  1.28962171e+00
   1.90555847e+00  4.28002656e-01  7.01311350e-01  9.70748663e-01
   3.44859004e-01  1.56643546e+00  2.99625874e-01  1.35029161e+00
   1.13022602e+00  8.16989779e-01  5.59703112e-01  5.32579243e-01
   1.27515531e+00  1.59225225e+00  7.20487118e-01  1.67689204e+00
   1.61699355e+00  1.03196120e+00  4.81096566e-01  1.54023027e+00
   8.93673003e-01  3.80989254e-01  6.66991472e-01  1.14558399e+00
   1.37900627e+00  4.19488549e-01  1.82324076e+00  3.88359904e-01
   1.75989962e+00  1.77510595e+00  1.32158673e+00  5.95844924e-01
   8.12834084e-01  8.04653883e-01  6.94384277e-01  9.64029729e-01
   2.97752202e-01  2.07171440e+00  3.88426900e-01  1.21965587e+00
   1.58369255e+00  4.02303278e-01  1.58826804e+00  5.76088727e-01
   2.02932358e-02  3.92885733e+00  4.25951290e+00  4.12271690e+00
   3.70334673e+00  4.00524569e+00  5.07848597e+00  3.85463858e+00
   3.78718948e+00  3.55931997e+00  5.21553230e+00  4.58874798e+00
   3.97845602e+00  4.62794971e+00  5.32962704e+00  4.68602705e+00
   4.10323668e+00  4.95977354e+00  3.89785099e+00  4.69832897e+00
   4.69677019e+00  4.60003805e+00  5.00358295e+00  5.11584949e+00
   5.46551943e+00  4.62797022e+00  3.99563980e+00  5.39302492e+00
   4.29622459e+00  4.50851154e+00  3.82208014e+00  4.40920258e+00
   3.85042286e+00  4.38025379e+00  4.49742699e+00  4.64132833e+00
   5.39801979e+00  4.22420835e+00  5.04345179e+00  3.72570753e+00
   4.46771908e+00  4.37914658e+00  4.22652960e+00  4.70537519e+00
   5.07367802e+00  3.69176865e+00  4.33846426e+00  5.41315889e+00
   3.41620636e+00  3.64507246e+00  4.40089798e+00  4.86257839e+00
   4.67991018e+00  4.84823608e+00  4.44844961e+00  4.36485958e+00
   4.64725876e+00  4.28835869e+00  5.46754694e+00  3.73880625e+00
  -7.38927650e+00 -4.66831923e+00  3.36570930e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-16 01:12:24.908202
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.2266
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-16 01:12:24.914077
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9662.99
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-16 01:12:24.917216
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.8084
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-16 01:12:24.920663
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -864.389
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140153664579120
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140152706233008
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140152706233512
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140152706234016
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140152706234520
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140152706235024

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f78259864e0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.618873
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.583924
grad_step = 000002, loss = 0.550819
grad_step = 000003, loss = 0.516246
grad_step = 000004, loss = 0.473795
grad_step = 000005, loss = 0.429180
grad_step = 000006, loss = 0.395989
grad_step = 000007, loss = 0.392990
grad_step = 000008, loss = 0.385173
grad_step = 000009, loss = 0.359828
grad_step = 000010, loss = 0.335959
grad_step = 000011, loss = 0.321314
grad_step = 000012, loss = 0.311903
grad_step = 000013, loss = 0.302781
grad_step = 000014, loss = 0.291629
grad_step = 000015, loss = 0.279086
grad_step = 000016, loss = 0.266358
grad_step = 000017, loss = 0.253729
grad_step = 000018, loss = 0.242297
grad_step = 000019, loss = 0.232448
grad_step = 000020, loss = 0.222564
grad_step = 000021, loss = 0.212270
grad_step = 000022, loss = 0.202858
grad_step = 000023, loss = 0.193972
grad_step = 000024, loss = 0.184740
grad_step = 000025, loss = 0.175561
grad_step = 000026, loss = 0.166980
grad_step = 000027, loss = 0.159010
grad_step = 000028, loss = 0.151400
grad_step = 000029, loss = 0.143904
grad_step = 000030, loss = 0.136436
grad_step = 000031, loss = 0.129120
grad_step = 000032, loss = 0.122181
grad_step = 000033, loss = 0.115748
grad_step = 000034, loss = 0.109707
grad_step = 000035, loss = 0.103802
grad_step = 000036, loss = 0.097978
grad_step = 000037, loss = 0.092367
grad_step = 000038, loss = 0.087009
grad_step = 000039, loss = 0.081831
grad_step = 000040, loss = 0.076824
grad_step = 000041, loss = 0.072074
grad_step = 000042, loss = 0.067619
grad_step = 000043, loss = 0.063375
grad_step = 000044, loss = 0.059223
grad_step = 000045, loss = 0.055168
grad_step = 000046, loss = 0.051352
grad_step = 000047, loss = 0.047849
grad_step = 000048, loss = 0.044547
grad_step = 000049, loss = 0.041310
grad_step = 000050, loss = 0.038174
grad_step = 000051, loss = 0.035230
grad_step = 000052, loss = 0.032497
grad_step = 000053, loss = 0.029954
grad_step = 000054, loss = 0.027582
grad_step = 000055, loss = 0.025349
grad_step = 000056, loss = 0.023217
grad_step = 000057, loss = 0.021203
grad_step = 000058, loss = 0.019368
grad_step = 000059, loss = 0.017717
grad_step = 000060, loss = 0.016192
grad_step = 000061, loss = 0.014771
grad_step = 000062, loss = 0.013468
grad_step = 000063, loss = 0.012273
grad_step = 000064, loss = 0.011179
grad_step = 000065, loss = 0.010201
grad_step = 000066, loss = 0.009332
grad_step = 000067, loss = 0.008545
grad_step = 000068, loss = 0.007828
grad_step = 000069, loss = 0.007199
grad_step = 000070, loss = 0.006642
grad_step = 000071, loss = 0.006135
grad_step = 000072, loss = 0.005682
grad_step = 000073, loss = 0.005288
grad_step = 000074, loss = 0.004939
grad_step = 000075, loss = 0.004631
grad_step = 000076, loss = 0.004360
grad_step = 000077, loss = 0.004112
grad_step = 000078, loss = 0.003889
grad_step = 000079, loss = 0.003695
grad_step = 000080, loss = 0.003524
grad_step = 000081, loss = 0.003371
grad_step = 000082, loss = 0.003235
grad_step = 000083, loss = 0.003109
grad_step = 000084, loss = 0.002999
grad_step = 000085, loss = 0.002905
grad_step = 000086, loss = 0.002819
grad_step = 000087, loss = 0.002741
grad_step = 000088, loss = 0.002674
grad_step = 000089, loss = 0.002615
grad_step = 000090, loss = 0.002564
grad_step = 000091, loss = 0.002520
grad_step = 000092, loss = 0.002480
grad_step = 000093, loss = 0.002445
grad_step = 000094, loss = 0.002415
grad_step = 000095, loss = 0.002390
grad_step = 000096, loss = 0.002368
grad_step = 000097, loss = 0.002349
grad_step = 000098, loss = 0.002331
grad_step = 000099, loss = 0.002317
grad_step = 000100, loss = 0.002304
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002291
grad_step = 000102, loss = 0.002281
grad_step = 000103, loss = 0.002272
grad_step = 000104, loss = 0.002263
grad_step = 000105, loss = 0.002255
grad_step = 000106, loss = 0.002247
grad_step = 000107, loss = 0.002240
grad_step = 000108, loss = 0.002233
grad_step = 000109, loss = 0.002226
grad_step = 000110, loss = 0.002220
grad_step = 000111, loss = 0.002214
grad_step = 000112, loss = 0.002207
grad_step = 000113, loss = 0.002201
grad_step = 000114, loss = 0.002195
grad_step = 000115, loss = 0.002190
grad_step = 000116, loss = 0.002184
grad_step = 000117, loss = 0.002178
grad_step = 000118, loss = 0.002173
grad_step = 000119, loss = 0.002167
grad_step = 000120, loss = 0.002162
grad_step = 000121, loss = 0.002156
grad_step = 000122, loss = 0.002151
grad_step = 000123, loss = 0.002146
grad_step = 000124, loss = 0.002141
grad_step = 000125, loss = 0.002136
grad_step = 000126, loss = 0.002131
grad_step = 000127, loss = 0.002126
grad_step = 000128, loss = 0.002122
grad_step = 000129, loss = 0.002117
grad_step = 000130, loss = 0.002112
grad_step = 000131, loss = 0.002108
grad_step = 000132, loss = 0.002103
grad_step = 000133, loss = 0.002098
grad_step = 000134, loss = 0.002094
grad_step = 000135, loss = 0.002089
grad_step = 000136, loss = 0.002085
grad_step = 000137, loss = 0.002080
grad_step = 000138, loss = 0.002075
grad_step = 000139, loss = 0.002071
grad_step = 000140, loss = 0.002066
grad_step = 000141, loss = 0.002062
grad_step = 000142, loss = 0.002057
grad_step = 000143, loss = 0.002052
grad_step = 000144, loss = 0.002048
grad_step = 000145, loss = 0.002043
grad_step = 000146, loss = 0.002038
grad_step = 000147, loss = 0.002033
grad_step = 000148, loss = 0.002028
grad_step = 000149, loss = 0.002023
grad_step = 000150, loss = 0.002018
grad_step = 000151, loss = 0.002014
grad_step = 000152, loss = 0.002009
grad_step = 000153, loss = 0.002003
grad_step = 000154, loss = 0.001998
grad_step = 000155, loss = 0.001993
grad_step = 000156, loss = 0.001988
grad_step = 000157, loss = 0.001984
grad_step = 000158, loss = 0.001979
grad_step = 000159, loss = 0.001975
grad_step = 000160, loss = 0.001970
grad_step = 000161, loss = 0.001965
grad_step = 000162, loss = 0.001958
grad_step = 000163, loss = 0.001952
grad_step = 000164, loss = 0.001948
grad_step = 000165, loss = 0.001943
grad_step = 000166, loss = 0.001939
grad_step = 000167, loss = 0.001936
grad_step = 000168, loss = 0.001935
grad_step = 000169, loss = 0.001932
grad_step = 000170, loss = 0.001928
grad_step = 000171, loss = 0.001922
grad_step = 000172, loss = 0.001910
grad_step = 000173, loss = 0.001903
grad_step = 000174, loss = 0.001902
grad_step = 000175, loss = 0.001899
grad_step = 000176, loss = 0.001897
grad_step = 000177, loss = 0.001895
grad_step = 000178, loss = 0.001888
grad_step = 000179, loss = 0.001880
grad_step = 000180, loss = 0.001874
grad_step = 000181, loss = 0.001866
grad_step = 000182, loss = 0.001862
grad_step = 000183, loss = 0.001860
grad_step = 000184, loss = 0.001857
grad_step = 000185, loss = 0.001856
grad_step = 000186, loss = 0.001860
grad_step = 000187, loss = 0.001870
grad_step = 000188, loss = 0.001876
grad_step = 000189, loss = 0.001879
grad_step = 000190, loss = 0.001860
grad_step = 000191, loss = 0.001835
grad_step = 000192, loss = 0.001821
grad_step = 000193, loss = 0.001828
grad_step = 000194, loss = 0.001840
grad_step = 000195, loss = 0.001842
grad_step = 000196, loss = 0.001833
grad_step = 000197, loss = 0.001814
grad_step = 000198, loss = 0.001801
grad_step = 000199, loss = 0.001800
grad_step = 000200, loss = 0.001806
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001812
grad_step = 000202, loss = 0.001809
grad_step = 000203, loss = 0.001801
grad_step = 000204, loss = 0.001789
grad_step = 000205, loss = 0.001779
grad_step = 000206, loss = 0.001775
grad_step = 000207, loss = 0.001776
grad_step = 000208, loss = 0.001779
grad_step = 000209, loss = 0.001781
grad_step = 000210, loss = 0.001783
grad_step = 000211, loss = 0.001781
grad_step = 000212, loss = 0.001778
grad_step = 000213, loss = 0.001770
grad_step = 000214, loss = 0.001761
grad_step = 000215, loss = 0.001753
grad_step = 000216, loss = 0.001747
grad_step = 000217, loss = 0.001743
grad_step = 000218, loss = 0.001740
grad_step = 000219, loss = 0.001739
grad_step = 000220, loss = 0.001739
grad_step = 000221, loss = 0.001741
grad_step = 000222, loss = 0.001748
grad_step = 000223, loss = 0.001764
grad_step = 000224, loss = 0.001786
grad_step = 000225, loss = 0.001820
grad_step = 000226, loss = 0.001809
grad_step = 000227, loss = 0.001775
grad_step = 000228, loss = 0.001726
grad_step = 000229, loss = 0.001723
grad_step = 000230, loss = 0.001750
grad_step = 000231, loss = 0.001751
grad_step = 000232, loss = 0.001729
grad_step = 000233, loss = 0.001712
grad_step = 000234, loss = 0.001714
grad_step = 000235, loss = 0.001725
grad_step = 000236, loss = 0.001726
grad_step = 000237, loss = 0.001713
grad_step = 000238, loss = 0.001695
grad_step = 000239, loss = 0.001690
grad_step = 000240, loss = 0.001699
grad_step = 000241, loss = 0.001705
grad_step = 000242, loss = 0.001702
grad_step = 000243, loss = 0.001695
grad_step = 000244, loss = 0.001687
grad_step = 000245, loss = 0.001678
grad_step = 000246, loss = 0.001674
grad_step = 000247, loss = 0.001678
grad_step = 000248, loss = 0.001681
grad_step = 000249, loss = 0.001681
grad_step = 000250, loss = 0.001675
grad_step = 000251, loss = 0.001670
grad_step = 000252, loss = 0.001665
grad_step = 000253, loss = 0.001662
grad_step = 000254, loss = 0.001660
grad_step = 000255, loss = 0.001658
grad_step = 000256, loss = 0.001658
grad_step = 000257, loss = 0.001659
grad_step = 000258, loss = 0.001659
grad_step = 000259, loss = 0.001658
grad_step = 000260, loss = 0.001656
grad_step = 000261, loss = 0.001653
grad_step = 000262, loss = 0.001651
grad_step = 000263, loss = 0.001649
grad_step = 000264, loss = 0.001647
grad_step = 000265, loss = 0.001645
grad_step = 000266, loss = 0.001642
grad_step = 000267, loss = 0.001638
grad_step = 000268, loss = 0.001635
grad_step = 000269, loss = 0.001632
grad_step = 000270, loss = 0.001630
grad_step = 000271, loss = 0.001628
grad_step = 000272, loss = 0.001626
grad_step = 000273, loss = 0.001625
grad_step = 000274, loss = 0.001623
grad_step = 000275, loss = 0.001622
grad_step = 000276, loss = 0.001621
grad_step = 000277, loss = 0.001621
grad_step = 000278, loss = 0.001624
grad_step = 000279, loss = 0.001634
grad_step = 000280, loss = 0.001654
grad_step = 000281, loss = 0.001703
grad_step = 000282, loss = 0.001760
grad_step = 000283, loss = 0.001836
grad_step = 000284, loss = 0.001779
grad_step = 000285, loss = 0.001679
grad_step = 000286, loss = 0.001613
grad_step = 000287, loss = 0.001656
grad_step = 000288, loss = 0.001715
grad_step = 000289, loss = 0.001665
grad_step = 000290, loss = 0.001606
grad_step = 000291, loss = 0.001621
grad_step = 000292, loss = 0.001658
grad_step = 000293, loss = 0.001649
grad_step = 000294, loss = 0.001603
grad_step = 000295, loss = 0.001607
grad_step = 000296, loss = 0.001634
grad_step = 000297, loss = 0.001619
grad_step = 000298, loss = 0.001597
grad_step = 000299, loss = 0.001601
grad_step = 000300, loss = 0.001611
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001603
grad_step = 000302, loss = 0.001590
grad_step = 000303, loss = 0.001592
grad_step = 000304, loss = 0.001599
grad_step = 000305, loss = 0.001593
grad_step = 000306, loss = 0.001584
grad_step = 000307, loss = 0.001583
grad_step = 000308, loss = 0.001585
grad_step = 000309, loss = 0.001584
grad_step = 000310, loss = 0.001576
grad_step = 000311, loss = 0.001574
grad_step = 000312, loss = 0.001578
grad_step = 000313, loss = 0.001576
grad_step = 000314, loss = 0.001569
grad_step = 000315, loss = 0.001565
grad_step = 000316, loss = 0.001568
grad_step = 000317, loss = 0.001570
grad_step = 000318, loss = 0.001566
grad_step = 000319, loss = 0.001561
grad_step = 000320, loss = 0.001559
grad_step = 000321, loss = 0.001559
grad_step = 000322, loss = 0.001559
grad_step = 000323, loss = 0.001557
grad_step = 000324, loss = 0.001554
grad_step = 000325, loss = 0.001551
grad_step = 000326, loss = 0.001550
grad_step = 000327, loss = 0.001550
grad_step = 000328, loss = 0.001549
grad_step = 000329, loss = 0.001548
grad_step = 000330, loss = 0.001545
grad_step = 000331, loss = 0.001542
grad_step = 000332, loss = 0.001540
grad_step = 000333, loss = 0.001538
grad_step = 000334, loss = 0.001537
grad_step = 000335, loss = 0.001536
grad_step = 000336, loss = 0.001535
grad_step = 000337, loss = 0.001534
grad_step = 000338, loss = 0.001532
grad_step = 000339, loss = 0.001530
grad_step = 000340, loss = 0.001528
grad_step = 000341, loss = 0.001527
grad_step = 000342, loss = 0.001525
grad_step = 000343, loss = 0.001525
grad_step = 000344, loss = 0.001524
grad_step = 000345, loss = 0.001525
grad_step = 000346, loss = 0.001529
grad_step = 000347, loss = 0.001536
grad_step = 000348, loss = 0.001552
grad_step = 000349, loss = 0.001574
grad_step = 000350, loss = 0.001618
grad_step = 000351, loss = 0.001651
grad_step = 000352, loss = 0.001688
grad_step = 000353, loss = 0.001654
grad_step = 000354, loss = 0.001593
grad_step = 000355, loss = 0.001533
grad_step = 000356, loss = 0.001518
grad_step = 000357, loss = 0.001547
grad_step = 000358, loss = 0.001577
grad_step = 000359, loss = 0.001584
grad_step = 000360, loss = 0.001553
grad_step = 000361, loss = 0.001521
grad_step = 000362, loss = 0.001508
grad_step = 000363, loss = 0.001517
grad_step = 000364, loss = 0.001533
grad_step = 000365, loss = 0.001532
grad_step = 000366, loss = 0.001521
grad_step = 000367, loss = 0.001503
grad_step = 000368, loss = 0.001491
grad_step = 000369, loss = 0.001491
grad_step = 000370, loss = 0.001497
grad_step = 000371, loss = 0.001505
grad_step = 000372, loss = 0.001509
grad_step = 000373, loss = 0.001508
grad_step = 000374, loss = 0.001498
grad_step = 000375, loss = 0.001486
grad_step = 000376, loss = 0.001474
grad_step = 000377, loss = 0.001469
grad_step = 000378, loss = 0.001471
grad_step = 000379, loss = 0.001475
grad_step = 000380, loss = 0.001478
grad_step = 000381, loss = 0.001473
grad_step = 000382, loss = 0.001471
grad_step = 000383, loss = 0.001472
grad_step = 000384, loss = 0.001479
grad_step = 000385, loss = 0.001491
grad_step = 000386, loss = 0.001513
grad_step = 000387, loss = 0.001532
grad_step = 000388, loss = 0.001560
grad_step = 000389, loss = 0.001561
grad_step = 000390, loss = 0.001540
grad_step = 000391, loss = 0.001492
grad_step = 000392, loss = 0.001451
grad_step = 000393, loss = 0.001442
grad_step = 000394, loss = 0.001461
grad_step = 000395, loss = 0.001485
grad_step = 000396, loss = 0.001494
grad_step = 000397, loss = 0.001496
grad_step = 000398, loss = 0.001474
grad_step = 000399, loss = 0.001452
grad_step = 000400, loss = 0.001433
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001432
grad_step = 000402, loss = 0.001443
grad_step = 000403, loss = 0.001451
grad_step = 000404, loss = 0.001454
grad_step = 000405, loss = 0.001452
grad_step = 000406, loss = 0.001443
grad_step = 000407, loss = 0.001434
grad_step = 000408, loss = 0.001425
grad_step = 000409, loss = 0.001425
grad_step = 000410, loss = 0.001427
grad_step = 000411, loss = 0.001429
grad_step = 000412, loss = 0.001433
grad_step = 000413, loss = 0.001443
grad_step = 000414, loss = 0.001448
grad_step = 000415, loss = 0.001450
grad_step = 000416, loss = 0.001434
grad_step = 000417, loss = 0.001419
grad_step = 000418, loss = 0.001407
grad_step = 000419, loss = 0.001397
grad_step = 000420, loss = 0.001390
grad_step = 000421, loss = 0.001390
grad_step = 000422, loss = 0.001396
grad_step = 000423, loss = 0.001403
grad_step = 000424, loss = 0.001407
grad_step = 000425, loss = 0.001411
grad_step = 000426, loss = 0.001427
grad_step = 000427, loss = 0.001444
grad_step = 000428, loss = 0.001488
grad_step = 000429, loss = 0.001492
grad_step = 000430, loss = 0.001493
grad_step = 000431, loss = 0.001455
grad_step = 000432, loss = 0.001420
grad_step = 000433, loss = 0.001393
grad_step = 000434, loss = 0.001384
grad_step = 000435, loss = 0.001408
grad_step = 000436, loss = 0.001440
grad_step = 000437, loss = 0.001434
grad_step = 000438, loss = 0.001407
grad_step = 000439, loss = 0.001396
grad_step = 000440, loss = 0.001402
grad_step = 000441, loss = 0.001397
grad_step = 000442, loss = 0.001372
grad_step = 000443, loss = 0.001356
grad_step = 000444, loss = 0.001359
grad_step = 000445, loss = 0.001354
grad_step = 000446, loss = 0.001339
grad_step = 000447, loss = 0.001328
grad_step = 000448, loss = 0.001329
grad_step = 000449, loss = 0.001332
grad_step = 000450, loss = 0.001325
grad_step = 000451, loss = 0.001319
grad_step = 000452, loss = 0.001321
grad_step = 000453, loss = 0.001327
grad_step = 000454, loss = 0.001331
grad_step = 000455, loss = 0.001341
grad_step = 000456, loss = 0.001373
grad_step = 000457, loss = 0.001438
grad_step = 000458, loss = 0.001553
grad_step = 000459, loss = 0.001685
grad_step = 000460, loss = 0.001800
grad_step = 000461, loss = 0.001650
grad_step = 000462, loss = 0.001413
grad_step = 000463, loss = 0.001313
grad_step = 000464, loss = 0.001413
grad_step = 000465, loss = 0.001508
grad_step = 000466, loss = 0.001441
grad_step = 000467, loss = 0.001338
grad_step = 000468, loss = 0.001330
grad_step = 000469, loss = 0.001387
grad_step = 000470, loss = 0.001418
grad_step = 000471, loss = 0.001370
grad_step = 000472, loss = 0.001300
grad_step = 000473, loss = 0.001310
grad_step = 000474, loss = 0.001353
grad_step = 000475, loss = 0.001352
grad_step = 000476, loss = 0.001323
grad_step = 000477, loss = 0.001300
grad_step = 000478, loss = 0.001288
grad_step = 000479, loss = 0.001290
grad_step = 000480, loss = 0.001308
grad_step = 000481, loss = 0.001314
grad_step = 000482, loss = 0.001290
grad_step = 000483, loss = 0.001263
grad_step = 000484, loss = 0.001262
grad_step = 000485, loss = 0.001278
grad_step = 000486, loss = 0.001282
grad_step = 000487, loss = 0.001275
grad_step = 000488, loss = 0.001265
grad_step = 000489, loss = 0.001255
grad_step = 000490, loss = 0.001248
grad_step = 000491, loss = 0.001247
grad_step = 000492, loss = 0.001255
grad_step = 000493, loss = 0.001259
grad_step = 000494, loss = 0.001252
grad_step = 000495, loss = 0.001242
grad_step = 000496, loss = 0.001236
grad_step = 000497, loss = 0.001234
grad_step = 000498, loss = 0.001232
grad_step = 000499, loss = 0.001231
grad_step = 000500, loss = 0.001232
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001232
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

  date_run                              2020-05-16 01:12:48.101067
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.296335
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-16 01:12:48.106450
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.245804
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-16 01:12:48.112365
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.144092
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-16 01:12:48.117223
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.73508
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
0   2020-05-16 01:12:14.356864  ...    mean_absolute_error
1   2020-05-16 01:12:14.360923  ...     mean_squared_error
2   2020-05-16 01:12:14.364932  ...  median_absolute_error
3   2020-05-16 01:12:14.368955  ...               r2_score
4   2020-05-16 01:12:24.908202  ...    mean_absolute_error
5   2020-05-16 01:12:24.914077  ...     mean_squared_error
6   2020-05-16 01:12:24.917216  ...  median_absolute_error
7   2020-05-16 01:12:24.920663  ...               r2_score
8   2020-05-16 01:12:48.101067  ...    mean_absolute_error
9   2020-05-16 01:12:48.106450  ...     mean_squared_error
10  2020-05-16 01:12:48.112365  ...  median_absolute_error
11  2020-05-16 01:12:48.117223  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b76bfe898> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 22%|██▏       | 2220032/9912422 [00:00<00:00, 21902171.28it/s] 91%|█████████ | 9035776/9912422 [00:00<00:00, 27492622.93it/s]9920512it [00:00, 30398251.31it/s]                             
0it [00:00, ?it/s]32768it [00:00, 697553.44it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 486386.75it/s]1654784it [00:00, 11504618.23it/s]                         
0it [00:00, ?it/s]8192it [00:00, 257917.27it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b295ace10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b263fa0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b295ace10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b28b35080> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

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

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b26373518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b26357be0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b295ace10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b28af26a0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b26373518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
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

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f7b76bb6e80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f8602c211d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=17b8b02773be31d1138db2167bdc206ab3a5cf217339fc7e4eafd06176f56ca7
  Stored in directory: /tmp/pip-ephem-wheel-cache-tzin89zd/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f859aa1c6a0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1638400/17464789 [=>............................] - ETA: 0s
 6430720/17464789 [==========>...................] - ETA: 0s
10846208/17464789 [=================>............] - ETA: 0s
14737408/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-16 01:14:13.264590: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 01:14:13.268690: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-16 01:14:13.268820: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55eb2856b070 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 01:14:13.268834: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9273 - accuracy: 0.4830
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7356 - accuracy: 0.4955 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7944 - accuracy: 0.4917
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7280 - accuracy: 0.4960
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7402 - accuracy: 0.4952
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7765 - accuracy: 0.4928
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.8068 - accuracy: 0.4909
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7605 - accuracy: 0.4939
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7467 - accuracy: 0.4948
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7786 - accuracy: 0.4927
11000/25000 [============>.................] - ETA: 4s - loss: 7.7335 - accuracy: 0.4956
12000/25000 [=============>................] - ETA: 4s - loss: 7.7216 - accuracy: 0.4964
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7185 - accuracy: 0.4966
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7367 - accuracy: 0.4954
15000/25000 [=================>............] - ETA: 3s - loss: 7.7494 - accuracy: 0.4946
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7471 - accuracy: 0.4947
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7487 - accuracy: 0.4946
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7356 - accuracy: 0.4955
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7255 - accuracy: 0.4962
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7226 - accuracy: 0.4963
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7090 - accuracy: 0.4972
22000/25000 [=========================>....] - ETA: 0s - loss: 7.7057 - accuracy: 0.4975
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6920 - accuracy: 0.4983
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6749 - accuracy: 0.4995
25000/25000 [==============================] - 9s 371us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-16 01:14:29.701615
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-16 01:14:29.701615  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<41:23:11, 5.79kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<29:12:13, 8.20kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<20:29:41, 11.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<14:20:57, 16.7kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<10:00:57, 23.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.49M/862M [00:02<6:57:54, 34.0kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 15.1M/862M [00:02<4:50:41, 48.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.7M/862M [00:02<3:22:13, 69.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.1M/862M [00:02<2:21:07, 99.0kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.5M/862M [00:02<1:38:21, 141kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:02<1:08:37, 201kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.0M/862M [00:02<47:52, 287kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 41.4M/862M [00:02<33:26, 409kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.6M/862M [00:02<23:23, 582kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.0M/862M [00:03<16:22, 826kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:03<11:46, 1.15MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<08:37, 1.56MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:05<12:07:05, 18.5kB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:05<8:29:32, 26.4kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 58.3M/862M [00:05<5:55:52, 37.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:07<4:14:30, 52.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:07<3:01:24, 73.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:07<2:07:37, 105kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 62.8M/862M [00:07<1:29:12, 149kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:09<1:10:03, 190kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:09<50:26, 264kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:09<35:33, 373kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:11<27:49, 476kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:11<22:10, 597kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:11<16:10, 817kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:12<13:25, 981kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:13<10:44, 1.23MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:13<07:47, 1.68MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:14<08:31, 1.54MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:15<08:38, 1.52MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:15<06:38, 1.97MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.6M/862M [00:15<04:47, 2.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:16<13:44, 949kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:17<10:58, 1.19MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:17<07:56, 1.64MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:18<08:36, 1.51MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<07:21, 1.76MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.3M/862M [00:19<05:25, 2.39MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:20<06:53, 1.87MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<05:53, 2.19MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:21<04:25, 2.91MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.4M/862M [00:21<03:15, 3.94MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:22<50:44, 253kB/s] .vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<36:49, 348kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:23<26:00, 492kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:24<21:11, 602kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<16:08, 790kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:24<11:32, 1.10MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<11:02, 1.15MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<09:01, 1.40MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:37, 1.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<07:37, 1.66MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:38, 1.90MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<04:57, 2.54MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:26, 1.95MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<05:47, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:18, 2.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<05:57, 2.10MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:33<41:52, 298kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<30:21, 409kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<21:48, 569kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<15:24, 803kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<15:47, 782kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<13:36, 907kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<10:09, 1.21MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<09:04, 1.35MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<07:36, 1.61MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:38, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<06:47, 1.80MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<07:09, 1.71MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:32, 2.20MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:04, 2.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<07:08, 1.70MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<06:16, 1.94MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<04:41, 2.58MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<06:06, 1.98MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<06:45, 1.79MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:20, 2.26MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:46<05:40, 2.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:46<05:00, 2.40MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<03:44, 3.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<02:47, 4.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<49:55, 239kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<36:08, 330kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<25:30, 467kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:50<20:37, 575kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:50<16:52, 703kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<12:19, 962kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<08:45, 1.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<08:03, 1.47MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:52<8:03:12, 24.4kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<5:38:38, 34.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<3:56:20, 49.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:54<2:52:54, 67.9kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:54<2:03:24, 95.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<1:26:52, 135kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<1:00:40, 192kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<1:18:15, 149kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<55:59, 208kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<39:24, 295kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<30:10, 384kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:58<23:37, 491kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<17:08, 676kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<12:06, 953kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<42:58, 268kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<31:17, 368kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<22:09, 519kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<18:06, 633kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<15:06, 758kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<11:06, 1.03MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<07:54, 1.44MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<11:25, 998kB/s] .vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<09:11, 1.24MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<06:40, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<07:16, 1.56MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<06:16, 1.81MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<04:40, 2.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:51, 1.92MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<05:16, 2.13MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<03:59, 2.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<05:21, 2.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<04:56, 2.26MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<03:42, 3.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<05:07, 2.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<04:46, 2.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<03:37, 3.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<05:06, 2.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:13<05:58, 1.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<04:40, 2.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<03:26, 3.19MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:15<06:24, 1.71MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:37, 1.95MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<04:13, 2.59MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:17<05:27, 2.00MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<04:57, 2.20MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<03:45, 2.90MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:06, 2.12MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<04:31, 2.39MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<03:24, 3.17MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<02:31, 4.26MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<22:50, 472kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<18:18, 589kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<13:16, 810kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<09:26, 1.14MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:23<10:30, 1.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<08:29, 1.26MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:12, 1.72MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<06:46, 1.57MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:25<07:01, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:24, 1.96MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<03:55, 2.70MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<07:36, 1.39MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:27<06:27, 1.64MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:44, 2.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<05:43, 1.83MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:29<05:07, 2.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<03:51, 2.72MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:31<05:05, 2.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<04:38, 2.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<03:31, 2.95MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:33<04:50, 2.14MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:18, 2.40MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<03:13, 3.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<02:53, 3.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<7:37:28, 22.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<5:20:26, 32.1kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<3:43:22, 45.8kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<2:47:45, 61.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:37<1:59:35, 85.5kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<1:24:05, 121kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<58:50, 173kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:39<44:52, 226kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<32:27, 313kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<22:55, 442kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:41<18:19, 551kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:41<13:52, 727kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<09:56, 1.01MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<09:14, 1.08MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:43<08:38, 1.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<06:35, 1.52MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<04:42, 2.12MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<27:10, 366kB/s] .vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:45<19:53, 500kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<14:08, 703kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<09:59, 989kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<25:35, 386kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:47<18:58, 521kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<13:30, 730kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<11:39, 842kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:49<10:18, 953kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<07:43, 1.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<05:30, 1.77MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<27:12, 358kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<20:03, 486kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<14:13, 683kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<12:08, 797kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<09:30, 1.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<06:53, 1.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<07:01, 1.37MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<06:59, 1.37MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:55<05:19, 1.80MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<03:51, 2.48MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<06:48, 1.40MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<05:46, 1.65MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:14, 2.24MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<05:07, 1.85MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:37, 2.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<03:26, 2.74MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:30, 2.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:11, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:03, 2.31MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<02:57, 3.17MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<07:01, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<05:54, 1.58MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:22, 2.13MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:04<05:10, 1.79MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<05:37, 1.65MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:25, 2.09MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<03:12, 2.87MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<29:48, 309kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<21:51, 421kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<15:29, 592kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<12:51, 710kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:08<10:57, 833kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<08:06, 1.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<05:45, 1.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:10<26:39, 340kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<19:27, 466kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<13:49, 653kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<09:45, 922kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<23:46, 378kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<17:35, 511kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<12:30, 716kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<10:46, 828kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<08:29, 1.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<06:08, 1.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<06:19, 1.40MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<05:22, 1.65MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:56, 2.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<04:48, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<04:18, 2.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:14, 2.71MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<04:15, 2.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:20<04:54, 1.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<03:53, 2.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<02:49, 3.07MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<30:59, 279kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:22<22:36, 382kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<15:58, 540kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<11:38, 737kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:24<6:23:37, 22.4kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<4:28:38, 31.9kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<3:07:21, 45.6kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<2:14:33, 63.3kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<1:35:59, 88.7kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<1:07:29, 126kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<47:10, 180kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:28<36:30, 232kB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:28<26:27, 319kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<18:39, 451kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<14:52, 563kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<12:12, 686kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<08:56, 936kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<06:25, 1.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<06:40, 1.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<05:23, 1.54MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<04:08, 2.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:01, 2.73MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:16, 1.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<04:32, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:21, 2.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:15, 1.92MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<03:41, 2.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<02:52, 2.84MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<02:07, 3.82MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<05:53, 1.38MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:38<04:59, 1.62MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:39, 2.20MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:22, 1.84MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:40<04:47, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<03:46, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<02:43, 2.93MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<07:10, 1.11MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<05:51, 1.36MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<04:17, 1.85MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:49, 1.64MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:12, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<03:08, 2.51MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<04:01, 1.95MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:37, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<02:44, 2.85MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<03:43, 2.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:25, 2.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<02:34, 3.01MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<03:33, 2.16MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<03:18, 2.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<02:30, 3.06MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:32, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:16, 2.32MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<02:27, 3.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:27, 2.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:12, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<02:24, 3.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<03:24, 2.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<04:00, 1.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:08, 2.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:19, 3.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<04:01, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<03:37, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<02:43, 2.72MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<03:35, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<04:05, 1.80MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:11, 2.30MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:20, 3.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<04:24, 1.65MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:51, 1.89MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<02:51, 2.54MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:03<03:39, 1.97MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<03:18, 2.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<02:29, 2.88MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<03:24, 2.10MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:07, 2.28MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<02:20, 3.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<03:17, 2.15MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<03:02, 2.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<02:18, 3.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:09<03:15, 2.16MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:09<02:59, 2.34MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<02:16, 3.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:03, 3.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:11<4:53:11, 23.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<3:25:17, 33.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<2:22:49, 48.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<1:45:20, 65.3kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:13<1:15:07, 91.5kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<52:50, 130kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<37:48, 180kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:15<27:02, 251kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<19:02, 356kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<13:20, 505kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<21:38, 311kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:17<15:51, 424kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<11:13, 598kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<09:21, 713kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:19<07:15, 917kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<05:14, 1.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:09, 1.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<05:01, 1.31MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:21<03:48, 1.73MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:47, 2.36MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:52, 1.69MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:23<03:24, 1.92MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<02:31, 2.58MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<03:14, 2.00MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:25<03:38, 1.77MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:25<02:53, 2.23MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:05, 3.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<17:24, 368kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<12:50, 498kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<09:07, 698kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<07:48, 810kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<06:49, 927kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<05:06, 1.24MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:37, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<21:18, 294kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<15:33, 402kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<10:59, 567kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<09:02, 685kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<07:35, 815kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<05:37, 1.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:53, 1.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:04, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:59, 2.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:28, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:43, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:54, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<02:06, 2.85MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<44:08, 136kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<31:30, 190kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<22:05, 269kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<16:43, 354kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<12:58, 456kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<09:22, 629kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<06:34, 889kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<18:37, 314kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<13:38, 428kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<09:39, 602kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<08:03, 716kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<06:14, 924kB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<04:30, 1.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<04:26, 1.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<03:42, 1.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<02:43, 2.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<03:11, 1.76MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<02:49, 1.99MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:06, 2.67MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<02:44, 2.03MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<02:29, 2.23MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<01:52, 2.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<02:35, 2.12MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<02:22, 2.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<01:47, 3.04MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<02:31, 2.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<02:20, 2.32MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<01:45, 3.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<02:26, 2.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<02:16, 2.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<01:43, 3.08MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:58<02:25, 2.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<02:15, 2.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<01:42, 3.08MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<01:28, 3.54MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:00<3:54:16, 22.3kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<2:43:53, 31.9kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<1:53:47, 45.5kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<1:23:33, 61.8kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<59:31, 86.7kB/s]  .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<41:47, 123kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<29:09, 176kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:04<22:16, 229kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<16:06, 316kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<11:21, 446kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<09:03, 555kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<06:52, 731kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<04:54, 1.02MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<04:32, 1.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<04:15, 1.17MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<03:13, 1.53MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<02:17, 2.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<36:27, 134kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<26:00, 188kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<18:14, 266kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<13:47, 350kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:12<10:09, 474kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<07:11, 666kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<06:05, 781kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:14<04:45, 999kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:25, 1.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<03:28, 1.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:55, 1.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:09, 2.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:34, 1.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:47, 1.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<02:11, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:34, 2.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<04:46, 951kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:49, 1.19MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<02:46, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:21<02:56, 1.52MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:01, 1.48MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:20, 1.90MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:40, 2.62MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<11:17, 390kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<08:22, 526kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<05:56, 737kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<05:07, 848kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<04:30, 962kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:21, 1.29MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<02:23, 1.79MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<03:23, 1.26MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:49, 1.51MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:04, 2.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:24, 1.75MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:29<02:07, 1.98MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:33, 2.67MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<02:03, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:31<01:52, 2.21MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:24, 2.92MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:33<01:55, 2.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<01:46, 2.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:20, 3.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<01:51, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<01:43, 2.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:18, 3.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<01:48, 2.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:04, 1.90MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<01:37, 2.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:12, 3.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<01:56, 1.99MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<01:45, 2.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:18, 2.91MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<01:46, 2.13MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<01:38, 2.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:13, 3.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<01:43, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:43<01:35, 2.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:11, 3.10MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<01:41, 2.17MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:45<01:30, 2.43MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:07, 3.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<00:59, 3.61MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:47<2:38:40, 22.6kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<1:50:53, 32.3kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<1:16:36, 46.1kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:49<57:31, 61.2kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:49<40:58, 85.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<28:46, 122kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<19:53, 174kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<24:52, 139kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:51<17:41, 195kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<12:24, 276kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<08:37, 393kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<12:36, 268kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:53<09:09, 369kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<06:26, 521kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<05:14, 633kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<04:00, 825kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<02:51, 1.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:43, 1.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:57<02:15, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:38, 1.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:51, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<01:37, 1.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:12, 2.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:34, 1.98MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<01:45, 1.76MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:01<01:23, 2.22MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:00, 3.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:11, 1.38MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<01:51, 1.63MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:22, 2.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:37, 1.82MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:27, 2.04MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:04, 2.71MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:25, 2.03MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:18, 2.22MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<00:58, 2.93MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:19, 2.13MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<01:32, 1.84MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:12, 2.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<00:52, 3.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<01:31, 1.82MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:21, 2.02MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:00, 2.72MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:12<01:19, 2.05MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:12, 2.24MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<00:54, 2.95MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:14<01:14, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:08, 2.29MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<00:51, 3.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:10, 2.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:05, 2.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<00:48, 3.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:08, 2.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:04, 2.32MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<00:47, 3.09MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:06, 2.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:01, 2.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<00:45, 3.13MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:03, 2.21MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<01:15, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<00:58, 2.38MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<00:42, 3.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:24, 1.63MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:13, 1.86MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:53, 2.51MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:07, 1.97MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:26<01:01, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:45, 2.86MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<01:01, 2.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:28<01:09, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<00:53, 2.37MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<00:38, 3.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:30<01:28, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:15, 1.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<00:55, 2.23MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<01:05, 1.84MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<01:11, 1.68MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:32<00:55, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:40, 2.93MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:40, 2.85MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:34<1:17:34, 25.1kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<53:39, 35.8kB/s]  .vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<37:04, 50.6kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<26:18, 71.2kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:36<18:23, 101kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<12:33, 144kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<14:12, 127kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:38<10:05, 178kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<07:00, 254kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<05:12, 335kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:40<04:00, 434kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<02:51, 602kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<02:00, 845kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:51, 898kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:42<01:28, 1.13MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:03, 1.55MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<01:05, 1.46MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<00:56, 1.71MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:40, 2.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:49, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:46<00:43, 2.09MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:32, 2.77MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:42, 2.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<00:48, 1.81MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<00:38, 2.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:27, 3.11MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<03:39, 382kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:50<02:41, 517kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:52, 726kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<01:35, 837kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<01:14, 1.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:53, 1.46MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<00:53, 1.41MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:45, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:32, 2.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:39, 1.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:33, 2.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:24, 2.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:17, 3.84MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<04:26, 253kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<03:12, 348kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<02:13, 491kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<01:44, 604kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<01:26, 732kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:02, 999kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:43, 1.40MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:57, 1.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:45, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:32, 1.76MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:35, 1.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:36, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:27, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:19, 2.69MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:36, 1.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:30, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:21, 2.24MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:25, 1.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:22, 2.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:16, 2.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:09<00:20, 2.05MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:19, 2.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:13, 2.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:18, 2.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:16, 2.30MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:12, 3.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:15, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:14, 2.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:10, 3.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:13, 2.18MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:15<00:16, 1.87MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:12, 2.34MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:08, 3.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<01:29, 295kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<01:04, 404kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:42, 570kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:32, 685kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:24, 887kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:16, 1.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:14, 1.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<00:13, 1.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<00:10, 1.70MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:23<00:12, 1.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:23<00:09, 1.41MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.91MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:25<00:05, 1.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:06, 1.57MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:04, 2.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:02, 2.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<00:04, 1.32MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:03, 1.57MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 2.11MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 1.78MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:29<00:00, 2.01MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 856/400000 [00:00<00:46, 8554.36it/s]  0%|          | 1734/400000 [00:00<00:46, 8618.44it/s]  1%|          | 2671/400000 [00:00<00:44, 8830.44it/s]  1%|          | 3575/400000 [00:00<00:44, 8890.18it/s]  1%|          | 4410/400000 [00:00<00:45, 8717.93it/s]  1%|▏         | 5261/400000 [00:00<00:45, 8653.11it/s]  2%|▏         | 6101/400000 [00:00<00:45, 8575.58it/s]  2%|▏         | 6935/400000 [00:00<00:46, 8502.72it/s]  2%|▏         | 7868/400000 [00:00<00:44, 8734.45it/s]  2%|▏         | 8715/400000 [00:01<00:45, 8651.08it/s]  2%|▏         | 9613/400000 [00:01<00:44, 8744.91it/s]  3%|▎         | 10549/400000 [00:01<00:43, 8920.48it/s]  3%|▎         | 11453/400000 [00:01<00:43, 8954.91it/s]  3%|▎         | 12376/400000 [00:01<00:42, 9035.57it/s]  3%|▎         | 13275/400000 [00:01<00:44, 8762.73it/s]  4%|▎         | 14200/400000 [00:01<00:43, 8902.82it/s]  4%|▍         | 15154/400000 [00:01<00:42, 9082.71it/s]  4%|▍         | 16083/400000 [00:01<00:41, 9142.20it/s]  4%|▍         | 16998/400000 [00:01<00:42, 8989.03it/s]  4%|▍         | 17898/400000 [00:02<00:43, 8730.08it/s]  5%|▍         | 18774/400000 [00:02<00:44, 8608.59it/s]  5%|▍         | 19637/400000 [00:02<00:44, 8597.30it/s]  5%|▌         | 20499/400000 [00:02<00:44, 8494.40it/s]  5%|▌         | 21381/400000 [00:02<00:44, 8587.59it/s]  6%|▌         | 22241/400000 [00:02<00:44, 8550.47it/s]  6%|▌         | 23134/400000 [00:02<00:43, 8659.40it/s]  6%|▌         | 24001/400000 [00:02<00:43, 8620.14it/s]  6%|▌         | 24913/400000 [00:02<00:42, 8764.00it/s]  6%|▋         | 25791/400000 [00:02<00:43, 8556.08it/s]  7%|▋         | 26649/400000 [00:03<00:43, 8509.24it/s]  7%|▋         | 27525/400000 [00:03<00:43, 8580.47it/s]  7%|▋         | 28436/400000 [00:03<00:42, 8731.92it/s]  7%|▋         | 29379/400000 [00:03<00:41, 8927.88it/s]  8%|▊         | 30288/400000 [00:03<00:41, 8971.74it/s]  8%|▊         | 31187/400000 [00:03<00:41, 8907.84it/s]  8%|▊         | 32079/400000 [00:03<00:41, 8769.92it/s]  8%|▊         | 32958/400000 [00:03<00:42, 8631.76it/s]  8%|▊         | 33823/400000 [00:03<00:43, 8506.05it/s]  9%|▊         | 34675/400000 [00:03<00:43, 8446.55it/s]  9%|▉         | 35532/400000 [00:04<00:42, 8483.00it/s]  9%|▉         | 36441/400000 [00:04<00:42, 8656.16it/s]  9%|▉         | 37309/400000 [00:04<00:42, 8570.11it/s] 10%|▉         | 38168/400000 [00:04<00:42, 8554.57it/s] 10%|▉         | 39084/400000 [00:04<00:41, 8725.57it/s] 10%|▉         | 39958/400000 [00:04<00:41, 8699.03it/s] 10%|█         | 40848/400000 [00:04<00:41, 8756.53it/s] 10%|█         | 41725/400000 [00:04<00:41, 8695.15it/s] 11%|█         | 42621/400000 [00:04<00:40, 8772.04it/s] 11%|█         | 43499/400000 [00:04<00:41, 8687.92it/s] 11%|█         | 44369/400000 [00:05<00:41, 8583.28it/s] 11%|█▏        | 45262/400000 [00:05<00:40, 8681.81it/s] 12%|█▏        | 46156/400000 [00:05<00:40, 8756.82it/s] 12%|█▏        | 47033/400000 [00:05<00:40, 8734.31it/s] 12%|█▏        | 47907/400000 [00:05<00:40, 8714.73it/s] 12%|█▏        | 48779/400000 [00:05<00:41, 8455.95it/s] 12%|█▏        | 49627/400000 [00:05<00:41, 8385.18it/s] 13%|█▎        | 50551/400000 [00:05<00:40, 8624.29it/s] 13%|█▎        | 51417/400000 [00:05<00:40, 8602.21it/s] 13%|█▎        | 52280/400000 [00:06<00:40, 8542.61it/s] 13%|█▎        | 53153/400000 [00:06<00:40, 8596.94it/s] 14%|█▎        | 54049/400000 [00:06<00:39, 8700.73it/s] 14%|█▎        | 54921/400000 [00:06<00:39, 8683.58it/s] 14%|█▍        | 55833/400000 [00:06<00:39, 8809.92it/s] 14%|█▍        | 56715/400000 [00:06<00:39, 8795.78it/s] 14%|█▍        | 57598/400000 [00:06<00:38, 8805.88it/s] 15%|█▍        | 58540/400000 [00:06<00:38, 8979.45it/s] 15%|█▍        | 59508/400000 [00:06<00:37, 9177.78it/s] 15%|█▌        | 60428/400000 [00:06<00:36, 9179.32it/s] 15%|█▌        | 61348/400000 [00:07<00:38, 8903.38it/s] 16%|█▌        | 62242/400000 [00:07<00:38, 8840.76it/s] 16%|█▌        | 63129/400000 [00:07<00:38, 8799.02it/s] 16%|█▌        | 64011/400000 [00:07<00:38, 8641.52it/s] 16%|█▌        | 64887/400000 [00:07<00:38, 8675.31it/s] 16%|█▋        | 65818/400000 [00:07<00:37, 8853.47it/s] 17%|█▋        | 66706/400000 [00:07<00:38, 8710.64it/s] 17%|█▋        | 67579/400000 [00:07<00:38, 8551.57it/s] 17%|█▋        | 68506/400000 [00:07<00:37, 8754.32it/s] 17%|█▋        | 69392/400000 [00:07<00:37, 8783.59it/s] 18%|█▊        | 70305/400000 [00:08<00:37, 8884.57it/s] 18%|█▊        | 71195/400000 [00:08<00:37, 8833.90it/s] 18%|█▊        | 72138/400000 [00:08<00:36, 9003.42it/s] 18%|█▊        | 73106/400000 [00:08<00:35, 9193.41it/s] 19%|█▊        | 74044/400000 [00:08<00:35, 9247.19it/s] 19%|█▊        | 74971/400000 [00:08<00:36, 9008.79it/s] 19%|█▉        | 75875/400000 [00:08<00:36, 8934.72it/s] 19%|█▉        | 76771/400000 [00:08<00:36, 8921.01it/s] 19%|█▉        | 77665/400000 [00:08<00:36, 8882.35it/s] 20%|█▉        | 78555/400000 [00:08<00:36, 8770.77it/s] 20%|█▉        | 79434/400000 [00:09<00:37, 8521.06it/s] 20%|██        | 80289/400000 [00:09<00:37, 8446.58it/s] 20%|██        | 81212/400000 [00:09<00:36, 8666.51it/s] 21%|██        | 82131/400000 [00:09<00:36, 8814.95it/s] 21%|██        | 83045/400000 [00:09<00:35, 8908.39it/s] 21%|██        | 83938/400000 [00:09<00:35, 8908.25it/s] 21%|██        | 84831/400000 [00:09<00:35, 8807.86it/s] 21%|██▏       | 85714/400000 [00:09<00:35, 8763.43it/s] 22%|██▏       | 86655/400000 [00:09<00:35, 8946.68it/s] 22%|██▏       | 87552/400000 [00:09<00:34, 8944.27it/s] 22%|██▏       | 88448/400000 [00:10<00:35, 8791.16it/s] 22%|██▏       | 89329/400000 [00:10<00:35, 8767.62it/s] 23%|██▎       | 90269/400000 [00:10<00:34, 8947.67it/s] 23%|██▎       | 91210/400000 [00:10<00:34, 9079.66it/s] 23%|██▎       | 92143/400000 [00:10<00:33, 9151.27it/s] 23%|██▎       | 93060/400000 [00:10<00:34, 8892.88it/s] 23%|██▎       | 93952/400000 [00:10<00:34, 8770.26it/s] 24%|██▎       | 94851/400000 [00:10<00:34, 8833.56it/s] 24%|██▍       | 95777/400000 [00:10<00:33, 8955.57it/s] 24%|██▍       | 96675/400000 [00:11<00:34, 8905.97it/s] 24%|██▍       | 97567/400000 [00:11<00:33, 8898.19it/s] 25%|██▍       | 98467/400000 [00:11<00:33, 8924.22it/s] 25%|██▍       | 99412/400000 [00:11<00:33, 9074.92it/s] 25%|██▌       | 100321/400000 [00:11<00:33, 9055.00it/s] 25%|██▌       | 101228/400000 [00:11<00:33, 8861.91it/s] 26%|██▌       | 102149/400000 [00:11<00:33, 8961.75it/s] 26%|██▌       | 103047/400000 [00:11<00:33, 8764.31it/s] 26%|██▌       | 103926/400000 [00:11<00:34, 8605.62it/s] 26%|██▌       | 104833/400000 [00:11<00:33, 8739.34it/s] 26%|██▋       | 105709/400000 [00:12<00:36, 8106.82it/s] 27%|██▋       | 106531/400000 [00:12<00:36, 8132.69it/s] 27%|██▋       | 107352/400000 [00:12<00:36, 8079.07it/s] 27%|██▋       | 108191/400000 [00:12<00:35, 8169.47it/s] 27%|██▋       | 109096/400000 [00:12<00:34, 8413.52it/s] 28%|██▊       | 110035/400000 [00:12<00:33, 8683.75it/s] 28%|██▊       | 110909/400000 [00:12<00:33, 8601.99it/s] 28%|██▊       | 111826/400000 [00:12<00:32, 8763.54it/s] 28%|██▊       | 112801/400000 [00:12<00:31, 9035.63it/s] 28%|██▊       | 113710/400000 [00:12<00:32, 8917.94it/s] 29%|██▊       | 114606/400000 [00:13<00:32, 8915.31it/s] 29%|██▉       | 115501/400000 [00:13<00:32, 8739.30it/s] 29%|██▉       | 116418/400000 [00:13<00:31, 8863.42it/s] 29%|██▉       | 117316/400000 [00:13<00:31, 8895.71it/s] 30%|██▉       | 118217/400000 [00:13<00:31, 8926.96it/s] 30%|██▉       | 119157/400000 [00:13<00:30, 9061.45it/s] 30%|███       | 120065/400000 [00:13<00:31, 9001.01it/s] 30%|███       | 120971/400000 [00:13<00:30, 9015.78it/s] 30%|███       | 121874/400000 [00:13<00:32, 8602.60it/s] 31%|███       | 122739/400000 [00:14<00:32, 8416.18it/s] 31%|███       | 123667/400000 [00:14<00:31, 8657.09it/s] 31%|███       | 124538/400000 [00:14<00:32, 8507.04it/s] 31%|███▏      | 125393/400000 [00:14<00:33, 8120.45it/s] 32%|███▏      | 126321/400000 [00:14<00:32, 8435.48it/s] 32%|███▏      | 127257/400000 [00:14<00:31, 8692.97it/s] 32%|███▏      | 128134/400000 [00:14<00:31, 8708.52it/s] 32%|███▏      | 129031/400000 [00:14<00:30, 8785.28it/s] 32%|███▏      | 129928/400000 [00:14<00:30, 8838.51it/s] 33%|███▎      | 130815/400000 [00:14<00:30, 8770.86it/s] 33%|███▎      | 131695/400000 [00:15<00:31, 8646.84it/s] 33%|███▎      | 132587/400000 [00:15<00:30, 8723.74it/s] 33%|███▎      | 133461/400000 [00:15<00:30, 8619.35it/s] 34%|███▎      | 134388/400000 [00:15<00:30, 8804.19it/s] 34%|███▍      | 135271/400000 [00:15<00:30, 8720.78it/s] 34%|███▍      | 136145/400000 [00:15<00:30, 8673.30it/s] 34%|███▍      | 137014/400000 [00:15<00:30, 8594.52it/s] 34%|███▍      | 137875/400000 [00:15<00:30, 8580.78it/s] 35%|███▍      | 138841/400000 [00:15<00:29, 8875.16it/s] 35%|███▍      | 139772/400000 [00:15<00:28, 9000.72it/s] 35%|███▌      | 140678/400000 [00:16<00:28, 9016.64it/s] 35%|███▌      | 141630/400000 [00:16<00:28, 9161.69it/s] 36%|███▌      | 142548/400000 [00:16<00:28, 9044.27it/s] 36%|███▌      | 143455/400000 [00:16<00:28, 9031.15it/s] 36%|███▌      | 144360/400000 [00:16<00:29, 8770.42it/s] 36%|███▋      | 145272/400000 [00:16<00:28, 8871.80it/s] 37%|███▋      | 146176/400000 [00:16<00:28, 8921.21it/s] 37%|███▋      | 147070/400000 [00:16<00:28, 8843.65it/s] 37%|███▋      | 147994/400000 [00:16<00:28, 8958.21it/s] 37%|███▋      | 148911/400000 [00:16<00:27, 9018.46it/s] 37%|███▋      | 149814/400000 [00:17<00:28, 8924.64it/s] 38%|███▊      | 150708/400000 [00:17<00:27, 8903.41it/s] 38%|███▊      | 151599/400000 [00:17<00:28, 8599.52it/s] 38%|███▊      | 152465/400000 [00:17<00:28, 8616.16it/s] 38%|███▊      | 153405/400000 [00:17<00:27, 8834.30it/s] 39%|███▊      | 154310/400000 [00:17<00:27, 8896.96it/s] 39%|███▉      | 155202/400000 [00:17<00:27, 8850.17it/s] 39%|███▉      | 156089/400000 [00:17<00:28, 8596.66it/s] 39%|███▉      | 157025/400000 [00:17<00:27, 8809.56it/s] 39%|███▉      | 157934/400000 [00:17<00:27, 8891.62it/s] 40%|███▉      | 158912/400000 [00:18<00:26, 9139.22it/s] 40%|███▉      | 159830/400000 [00:18<00:26, 9123.94it/s] 40%|████      | 160745/400000 [00:18<00:26, 8951.01it/s] 40%|████      | 161690/400000 [00:18<00:26, 9094.28it/s] 41%|████      | 162602/400000 [00:18<00:26, 9075.93it/s] 41%|████      | 163535/400000 [00:18<00:25, 9150.12it/s] 41%|████      | 164452/400000 [00:18<00:26, 9034.65it/s] 41%|████▏     | 165357/400000 [00:18<00:26, 8865.51it/s] 42%|████▏     | 166246/400000 [00:18<00:26, 8841.37it/s] 42%|████▏     | 167132/400000 [00:19<00:26, 8842.57it/s] 42%|████▏     | 168089/400000 [00:19<00:25, 9048.09it/s] 42%|████▏     | 168996/400000 [00:19<00:25, 8944.83it/s] 42%|████▏     | 169892/400000 [00:19<00:25, 8894.09it/s] 43%|████▎     | 170783/400000 [00:19<00:26, 8717.04it/s] 43%|████▎     | 171722/400000 [00:19<00:25, 8904.83it/s] 43%|████▎     | 172671/400000 [00:19<00:25, 9072.37it/s] 43%|████▎     | 173581/400000 [00:19<00:25, 9040.99it/s] 44%|████▎     | 174487/400000 [00:19<00:25, 8971.12it/s] 44%|████▍     | 175427/400000 [00:19<00:24, 9093.04it/s] 44%|████▍     | 176338/400000 [00:20<00:25, 8923.90it/s] 44%|████▍     | 177232/400000 [00:20<00:25, 8903.98it/s] 45%|████▍     | 178124/400000 [00:20<00:25, 8771.43it/s] 45%|████▍     | 179003/400000 [00:20<00:25, 8772.12it/s] 45%|████▍     | 179935/400000 [00:20<00:24, 8926.74it/s] 45%|████▌     | 180829/400000 [00:20<00:25, 8765.74it/s] 45%|████▌     | 181708/400000 [00:20<00:25, 8587.94it/s] 46%|████▌     | 182569/400000 [00:20<00:25, 8592.15it/s] 46%|████▌     | 183449/400000 [00:20<00:25, 8651.82it/s] 46%|████▌     | 184335/400000 [00:20<00:24, 8710.19it/s] 46%|████▋     | 185207/400000 [00:21<00:24, 8636.97it/s] 47%|████▋     | 186072/400000 [00:21<00:25, 8342.30it/s] 47%|████▋     | 186914/400000 [00:21<00:25, 8363.55it/s] 47%|████▋     | 187753/400000 [00:21<00:25, 8338.04it/s] 47%|████▋     | 188608/400000 [00:21<00:25, 8394.31it/s] 47%|████▋     | 189460/400000 [00:21<00:24, 8430.94it/s] 48%|████▊     | 190336/400000 [00:21<00:24, 8526.38it/s] 48%|████▊     | 191237/400000 [00:21<00:24, 8663.15it/s] 48%|████▊     | 192108/400000 [00:21<00:23, 8674.35it/s] 48%|████▊     | 192977/400000 [00:21<00:24, 8462.95it/s] 48%|████▊     | 193825/400000 [00:22<00:24, 8464.88it/s] 49%|████▊     | 194673/400000 [00:22<00:24, 8351.97it/s] 49%|████▉     | 195510/400000 [00:22<00:25, 8085.51it/s] 49%|████▉     | 196340/400000 [00:22<00:24, 8147.26it/s] 49%|████▉     | 197237/400000 [00:22<00:24, 8377.60it/s] 50%|████▉     | 198107/400000 [00:22<00:23, 8469.62it/s] 50%|████▉     | 198962/400000 [00:22<00:23, 8492.50it/s] 50%|████▉     | 199813/400000 [00:22<00:23, 8481.85it/s] 50%|█████     | 200736/400000 [00:22<00:22, 8692.93it/s] 50%|█████     | 201652/400000 [00:23<00:22, 8825.50it/s] 51%|█████     | 202595/400000 [00:23<00:21, 8998.51it/s] 51%|█████     | 203523/400000 [00:23<00:21, 9080.65it/s] 51%|█████     | 204433/400000 [00:23<00:21, 9015.14it/s] 51%|█████▏    | 205336/400000 [00:23<00:21, 8920.80it/s] 52%|█████▏    | 206236/400000 [00:23<00:21, 8943.90it/s] 52%|█████▏    | 207169/400000 [00:23<00:21, 9055.80it/s] 52%|█████▏    | 208085/400000 [00:23<00:21, 9085.66it/s] 52%|█████▏    | 208995/400000 [00:23<00:21, 8984.29it/s] 52%|█████▏    | 209895/400000 [00:23<00:21, 8798.98it/s] 53%|█████▎    | 210777/400000 [00:24<00:22, 8553.25it/s] 53%|█████▎    | 211635/400000 [00:24<00:22, 8463.96it/s] 53%|█████▎    | 212484/400000 [00:24<00:22, 8412.73it/s] 53%|█████▎    | 213327/400000 [00:24<00:22, 8256.14it/s] 54%|█████▎    | 214178/400000 [00:24<00:22, 8328.84it/s] 54%|█████▍    | 215013/400000 [00:24<00:23, 7943.56it/s] 54%|█████▍    | 215868/400000 [00:24<00:22, 8114.40it/s] 54%|█████▍    | 216776/400000 [00:24<00:21, 8380.84it/s] 54%|█████▍    | 217672/400000 [00:24<00:21, 8545.82it/s] 55%|█████▍    | 218531/400000 [00:24<00:21, 8494.66it/s] 55%|█████▍    | 219384/400000 [00:25<00:21, 8504.33it/s] 55%|█████▌    | 220302/400000 [00:25<00:20, 8694.84it/s] 55%|█████▌    | 221175/400000 [00:25<00:20, 8699.70it/s] 56%|█████▌    | 222047/400000 [00:25<00:20, 8680.07it/s] 56%|█████▌    | 222975/400000 [00:25<00:20, 8850.29it/s] 56%|█████▌    | 223862/400000 [00:25<00:20, 8723.51it/s] 56%|█████▌    | 224736/400000 [00:25<00:20, 8581.31it/s] 56%|█████▋    | 225626/400000 [00:25<00:20, 8673.93it/s] 57%|█████▋    | 226499/400000 [00:25<00:19, 8688.82it/s] 57%|█████▋    | 227419/400000 [00:25<00:19, 8834.64it/s] 57%|█████▋    | 228319/400000 [00:26<00:19, 8881.61it/s] 57%|█████▋    | 229290/400000 [00:26<00:18, 9114.09it/s] 58%|█████▊    | 230247/400000 [00:26<00:18, 9245.37it/s] 58%|█████▊    | 231174/400000 [00:26<00:18, 9041.18it/s] 58%|█████▊    | 232081/400000 [00:26<00:19, 8800.59it/s] 58%|█████▊    | 232993/400000 [00:26<00:18, 8892.37it/s] 58%|█████▊    | 233914/400000 [00:26<00:18, 8983.17it/s] 59%|█████▊    | 234815/400000 [00:26<00:18, 8826.41it/s] 59%|█████▉    | 235700/400000 [00:26<00:18, 8675.57it/s] 59%|█████▉    | 236621/400000 [00:27<00:18, 8828.01it/s] 59%|█████▉    | 237506/400000 [00:27<00:18, 8737.99it/s] 60%|█████▉    | 238382/400000 [00:27<00:18, 8522.55it/s] 60%|█████▉    | 239237/400000 [00:27<00:18, 8465.18it/s] 60%|██████    | 240086/400000 [00:27<00:19, 8354.38it/s] 60%|██████    | 240975/400000 [00:27<00:18, 8506.65it/s] 60%|██████    | 241870/400000 [00:27<00:18, 8634.57it/s] 61%|██████    | 242815/400000 [00:27<00:17, 8863.66it/s] 61%|██████    | 243710/400000 [00:27<00:17, 8887.72it/s] 61%|██████    | 244601/400000 [00:27<00:17, 8781.16it/s] 61%|██████▏   | 245551/400000 [00:28<00:17, 8984.16it/s] 62%|██████▏   | 246475/400000 [00:28<00:16, 9056.46it/s] 62%|██████▏   | 247385/400000 [00:28<00:16, 9067.45it/s] 62%|██████▏   | 248330/400000 [00:28<00:16, 9177.47it/s] 62%|██████▏   | 249249/400000 [00:28<00:16, 8956.26it/s] 63%|██████▎   | 250147/400000 [00:28<00:17, 8727.04it/s] 63%|██████▎   | 251023/400000 [00:28<00:17, 8555.23it/s] 63%|██████▎   | 251882/400000 [00:28<00:17, 8329.33it/s] 63%|██████▎   | 252719/400000 [00:28<00:18, 7918.86it/s] 63%|██████▎   | 253518/400000 [00:28<00:18, 7877.92it/s] 64%|██████▎   | 254327/400000 [00:29<00:18, 7939.09it/s] 64%|██████▍   | 255159/400000 [00:29<00:17, 8048.43it/s] 64%|██████▍   | 255982/400000 [00:29<00:17, 8101.36it/s] 64%|██████▍   | 256795/400000 [00:29<00:17, 8037.43it/s] 64%|██████▍   | 257620/400000 [00:29<00:17, 8099.08it/s] 65%|██████▍   | 258438/400000 [00:29<00:17, 8119.64it/s] 65%|██████▍   | 259251/400000 [00:29<00:17, 8031.46it/s] 65%|██████▌   | 260055/400000 [00:29<00:17, 8000.72it/s] 65%|██████▌   | 260927/400000 [00:29<00:16, 8202.80it/s] 65%|██████▌   | 261816/400000 [00:29<00:16, 8395.25it/s] 66%|██████▌   | 262727/400000 [00:30<00:15, 8597.21it/s] 66%|██████▌   | 263624/400000 [00:30<00:15, 8705.26it/s] 66%|██████▌   | 264497/400000 [00:30<00:15, 8692.81it/s] 66%|██████▋   | 265368/400000 [00:30<00:15, 8567.78it/s] 67%|██████▋   | 266227/400000 [00:30<00:15, 8566.61it/s] 67%|██████▋   | 267085/400000 [00:30<00:15, 8526.68it/s] 67%|██████▋   | 267939/400000 [00:30<00:15, 8422.17it/s] 67%|██████▋   | 268818/400000 [00:30<00:15, 8529.09it/s] 67%|██████▋   | 269695/400000 [00:30<00:15, 8599.42it/s] 68%|██████▊   | 270571/400000 [00:31<00:14, 8645.48it/s] 68%|██████▊   | 271501/400000 [00:31<00:14, 8829.77it/s] 68%|██████▊   | 272386/400000 [00:31<00:14, 8823.38it/s] 68%|██████▊   | 273270/400000 [00:31<00:14, 8740.98it/s] 69%|██████▊   | 274145/400000 [00:31<00:14, 8593.79it/s] 69%|██████▉   | 275006/400000 [00:31<00:14, 8483.55it/s] 69%|██████▉   | 275913/400000 [00:31<00:14, 8649.90it/s] 69%|██████▉   | 276846/400000 [00:31<00:13, 8841.65it/s] 69%|██████▉   | 277774/400000 [00:31<00:13, 8966.48it/s] 70%|██████▉   | 278673/400000 [00:31<00:13, 8703.54it/s] 70%|██████▉   | 279581/400000 [00:32<00:13, 8810.95it/s] 70%|███████   | 280478/400000 [00:32<00:13, 8857.87it/s] 70%|███████   | 281366/400000 [00:32<00:14, 8434.49it/s] 71%|███████   | 282215/400000 [00:32<00:14, 8328.67it/s] 71%|███████   | 283053/400000 [00:32<00:14, 8190.03it/s] 71%|███████   | 283938/400000 [00:32<00:13, 8376.82it/s] 71%|███████   | 284801/400000 [00:32<00:13, 8449.39it/s] 71%|███████▏  | 285693/400000 [00:32<00:13, 8584.51it/s] 72%|███████▏  | 286554/400000 [00:32<00:13, 8471.88it/s] 72%|███████▏  | 287404/400000 [00:32<00:13, 8380.71it/s] 72%|███████▏  | 288253/400000 [00:33<00:13, 8411.41it/s] 72%|███████▏  | 289118/400000 [00:33<00:13, 8481.56it/s] 72%|███████▏  | 289968/400000 [00:33<00:12, 8475.69it/s] 73%|███████▎  | 290881/400000 [00:33<00:12, 8661.08it/s] 73%|███████▎  | 291749/400000 [00:33<00:12, 8630.64it/s] 73%|███████▎  | 292645/400000 [00:33<00:12, 8726.57it/s] 73%|███████▎  | 293526/400000 [00:33<00:12, 8749.39it/s] 74%|███████▎  | 294430/400000 [00:33<00:11, 8832.37it/s] 74%|███████▍  | 295314/400000 [00:33<00:12, 8560.74it/s] 74%|███████▍  | 296173/400000 [00:33<00:12, 8259.50it/s] 74%|███████▍  | 297041/400000 [00:34<00:12, 8379.97it/s] 74%|███████▍  | 297932/400000 [00:34<00:11, 8531.44it/s] 75%|███████▍  | 298789/400000 [00:34<00:12, 8272.87it/s] 75%|███████▍  | 299621/400000 [00:34<00:12, 8286.71it/s] 75%|███████▌  | 300473/400000 [00:34<00:11, 8352.75it/s] 75%|███████▌  | 301378/400000 [00:34<00:11, 8547.92it/s] 76%|███████▌  | 302236/400000 [00:34<00:11, 8522.46it/s] 76%|███████▌  | 303130/400000 [00:34<00:11, 8641.59it/s] 76%|███████▌  | 303996/400000 [00:34<00:11, 8530.35it/s] 76%|███████▌  | 304851/400000 [00:35<00:11, 8295.97it/s] 76%|███████▋  | 305689/400000 [00:35<00:11, 8320.45it/s] 77%|███████▋  | 306618/400000 [00:35<00:10, 8587.15it/s] 77%|███████▋  | 307491/400000 [00:35<00:10, 8626.81it/s] 77%|███████▋  | 308357/400000 [00:35<00:10, 8565.78it/s] 77%|███████▋  | 309216/400000 [00:35<00:10, 8540.46it/s] 78%|███████▊  | 310072/400000 [00:35<00:10, 8410.87it/s] 78%|███████▊  | 310953/400000 [00:35<00:10, 8526.68it/s] 78%|███████▊  | 311835/400000 [00:35<00:10, 8612.47it/s] 78%|███████▊  | 312698/400000 [00:35<00:10, 8590.89it/s] 78%|███████▊  | 313558/400000 [00:36<00:10, 8562.28it/s] 79%|███████▊  | 314478/400000 [00:36<00:09, 8742.42it/s] 79%|███████▉  | 315396/400000 [00:36<00:09, 8865.39it/s] 79%|███████▉  | 316284/400000 [00:36<00:09, 8804.00it/s] 79%|███████▉  | 317222/400000 [00:36<00:09, 8967.46it/s] 80%|███████▉  | 318121/400000 [00:36<00:09, 8663.34it/s] 80%|███████▉  | 319044/400000 [00:36<00:09, 8824.04it/s] 80%|███████▉  | 319949/400000 [00:36<00:09, 8889.40it/s] 80%|████████  | 320873/400000 [00:36<00:08, 8991.25it/s] 80%|████████  | 321789/400000 [00:36<00:08, 9040.94it/s] 81%|████████  | 322695/400000 [00:37<00:08, 8848.94it/s] 81%|████████  | 323582/400000 [00:37<00:08, 8851.09it/s] 81%|████████  | 324469/400000 [00:37<00:08, 8674.58it/s] 81%|████████▏ | 325345/400000 [00:37<00:08, 8698.38it/s] 82%|████████▏ | 326274/400000 [00:37<00:08, 8866.07it/s] 82%|████████▏ | 327163/400000 [00:37<00:08, 8825.58it/s] 82%|████████▏ | 328047/400000 [00:37<00:08, 8697.53it/s] 82%|████████▏ | 328964/400000 [00:37<00:08, 8833.77it/s] 82%|████████▏ | 329859/400000 [00:37<00:07, 8865.88it/s] 83%|████████▎ | 330747/400000 [00:37<00:07, 8668.36it/s] 83%|████████▎ | 331616/400000 [00:38<00:07, 8602.13it/s] 83%|████████▎ | 332484/400000 [00:38<00:07, 8624.52it/s] 83%|████████▎ | 333407/400000 [00:38<00:07, 8793.92it/s] 84%|████████▎ | 334309/400000 [00:38<00:07, 8858.45it/s] 84%|████████▍ | 335256/400000 [00:38<00:07, 9030.91it/s] 84%|████████▍ | 336161/400000 [00:38<00:07, 8858.61it/s] 84%|████████▍ | 337104/400000 [00:38<00:06, 9022.00it/s] 85%|████████▍ | 338036/400000 [00:38<00:06, 9108.67it/s] 85%|████████▍ | 338949/400000 [00:38<00:06, 8879.71it/s] 85%|████████▍ | 339840/400000 [00:39<00:06, 8773.69it/s] 85%|████████▌ | 340720/400000 [00:39<00:06, 8664.55it/s] 85%|████████▌ | 341657/400000 [00:39<00:06, 8862.43it/s] 86%|████████▌ | 342584/400000 [00:39<00:06, 8979.63it/s] 86%|████████▌ | 343484/400000 [00:39<00:06, 8934.90it/s] 86%|████████▌ | 344415/400000 [00:39<00:06, 9040.97it/s] 86%|████████▋ | 345321/400000 [00:39<00:06, 8717.49it/s] 87%|████████▋ | 346209/400000 [00:39<00:06, 8763.28it/s] 87%|████████▋ | 347088/400000 [00:39<00:06, 8725.04it/s] 87%|████████▋ | 347966/400000 [00:39<00:05, 8739.70it/s] 87%|████████▋ | 348842/400000 [00:40<00:05, 8691.20it/s] 87%|████████▋ | 349713/400000 [00:40<00:05, 8383.95it/s] 88%|████████▊ | 350590/400000 [00:40<00:05, 8494.64it/s] 88%|████████▊ | 351457/400000 [00:40<00:05, 8544.85it/s] 88%|████████▊ | 352351/400000 [00:40<00:05, 8659.65it/s] 88%|████████▊ | 353219/400000 [00:40<00:05, 8558.33it/s] 89%|████████▊ | 354077/400000 [00:40<00:05, 8425.13it/s] 89%|████████▊ | 354955/400000 [00:40<00:05, 8527.62it/s] 89%|████████▉ | 355855/400000 [00:40<00:05, 8663.13it/s] 89%|████████▉ | 356747/400000 [00:40<00:04, 8736.36it/s] 89%|████████▉ | 357640/400000 [00:41<00:04, 8789.92it/s] 90%|████████▉ | 358520/400000 [00:41<00:04, 8604.74it/s] 90%|████████▉ | 359461/400000 [00:41<00:04, 8829.51it/s] 90%|█████████ | 360361/400000 [00:41<00:04, 8878.39it/s] 90%|█████████ | 361285/400000 [00:41<00:04, 8983.15it/s] 91%|█████████ | 362185/400000 [00:41<00:04, 8986.17it/s] 91%|█████████ | 363085/400000 [00:41<00:04, 8818.16it/s] 91%|█████████ | 363980/400000 [00:41<00:04, 8857.24it/s] 91%|█████████ | 364890/400000 [00:41<00:03, 8927.16it/s] 91%|█████████▏| 365784/400000 [00:41<00:03, 8929.97it/s] 92%|█████████▏| 366678/400000 [00:42<00:04, 8012.88it/s] 92%|█████████▏| 367498/400000 [00:42<00:04, 7961.19it/s] 92%|█████████▏| 368307/400000 [00:42<00:03, 7983.36it/s] 92%|█████████▏| 369115/400000 [00:42<00:03, 8011.04it/s] 93%|█████████▎| 370017/400000 [00:42<00:03, 8288.41it/s] 93%|█████████▎| 370885/400000 [00:42<00:03, 8401.44it/s] 93%|█████████▎| 371731/400000 [00:42<00:03, 8302.06it/s] 93%|█████████▎| 372670/400000 [00:42<00:03, 8599.80it/s] 93%|█████████▎| 373539/400000 [00:42<00:03, 8625.20it/s] 94%|█████████▎| 374441/400000 [00:43<00:02, 8739.54it/s] 94%|█████████▍| 375385/400000 [00:43<00:02, 8936.13it/s] 94%|█████████▍| 376282/400000 [00:43<00:02, 8706.53it/s] 94%|█████████▍| 377167/400000 [00:43<00:02, 8748.00it/s] 95%|█████████▍| 378045/400000 [00:43<00:02, 8737.77it/s] 95%|█████████▍| 378944/400000 [00:43<00:02, 8811.94it/s] 95%|█████████▍| 379843/400000 [00:43<00:02, 8862.46it/s] 95%|█████████▌| 380731/400000 [00:43<00:02, 8735.88it/s] 95%|█████████▌| 381606/400000 [00:43<00:02, 8583.28it/s] 96%|█████████▌| 382466/400000 [00:43<00:02, 8495.57it/s] 96%|█████████▌| 383352/400000 [00:44<00:01, 8600.68it/s] 96%|█████████▌| 384308/400000 [00:44<00:01, 8864.69it/s] 96%|█████████▋| 385198/400000 [00:44<00:01, 8677.45it/s] 97%|█████████▋| 386069/400000 [00:44<00:01, 8538.60it/s] 97%|█████████▋| 386926/400000 [00:44<00:01, 8511.96it/s] 97%|█████████▋| 387779/400000 [00:44<00:01, 8345.44it/s] 97%|█████████▋| 388644/400000 [00:44<00:01, 8432.54it/s] 97%|█████████▋| 389502/400000 [00:44<00:01, 8473.97it/s] 98%|█████████▊| 390411/400000 [00:44<00:01, 8648.79it/s] 98%|█████████▊| 391330/400000 [00:44<00:00, 8801.56it/s] 98%|█████████▊| 392213/400000 [00:45<00:00, 8670.93it/s] 98%|█████████▊| 393082/400000 [00:45<00:00, 8671.40it/s] 98%|█████████▊| 393951/400000 [00:45<00:00, 8315.31it/s] 99%|█████████▊| 394810/400000 [00:45<00:00, 8394.85it/s] 99%|█████████▉| 395653/400000 [00:45<00:00, 8362.08it/s] 99%|█████████▉| 396492/400000 [00:45<00:00, 8302.33it/s] 99%|█████████▉| 397373/400000 [00:45<00:00, 8445.77it/s]100%|█████████▉| 398220/400000 [00:45<00:00, 8358.61it/s]100%|█████████▉| 399129/400000 [00:45<00:00, 8565.25it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8698.36it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f3ce2daed30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01107453703027597 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.010934141567319531 	 Accuracy: 66

  model saves at 66% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15866 out of table with 15791 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15866 out of table with 15791 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-16 01:23:24.186612: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 01:23:24.190339: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-16 01:23:24.190487: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b2599e5200 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 01:23:24.190504: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f3c8c2da2e8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7126 - accuracy: 0.4970
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6896 - accuracy: 0.4985 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6513 - accuracy: 0.5010
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6091 - accuracy: 0.5038
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5685 - accuracy: 0.5064
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5440 - accuracy: 0.5080
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5461 - accuracy: 0.5079
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5574 - accuracy: 0.5071
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5883 - accuracy: 0.5051
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
11000/25000 [============>.................] - ETA: 4s - loss: 7.6387 - accuracy: 0.5018
12000/25000 [=============>................] - ETA: 4s - loss: 7.6526 - accuracy: 0.5009
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6961 - accuracy: 0.4981
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6743 - accuracy: 0.4995
15000/25000 [=================>............] - ETA: 3s - loss: 7.6952 - accuracy: 0.4981
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6676 - accuracy: 0.4999
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6702 - accuracy: 0.4998
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6626 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6567 - accuracy: 0.5006
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6834 - accuracy: 0.4989
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6720 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6698 - accuracy: 0.4998
25000/25000 [==============================] - 9s 376us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f3c39edc390> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f3c8f78fd68> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.5306 - crf_viterbi_accuracy: 0.2533 - val_loss: 1.3980 - val_crf_viterbi_accuracy: 0.2133

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

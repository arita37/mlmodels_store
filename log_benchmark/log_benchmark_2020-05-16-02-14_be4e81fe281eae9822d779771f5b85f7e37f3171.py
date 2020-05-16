
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fd5fc28efd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-16 02:14:21.006435
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-16 02:14:21.010885
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-16 02:14:21.014451
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-16 02:14:21.017913
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fd6082a6470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352333.5312
Epoch 2/10

1/1 [==============================] - 0s 100ms/step - loss: 217060.1875
Epoch 3/10

1/1 [==============================] - 0s 99ms/step - loss: 123451.4062
Epoch 4/10

1/1 [==============================] - 0s 97ms/step - loss: 60885.8867
Epoch 5/10

1/1 [==============================] - 0s 93ms/step - loss: 30319.2188
Epoch 6/10

1/1 [==============================] - 0s 91ms/step - loss: 17138.2109
Epoch 7/10

1/1 [==============================] - 0s 102ms/step - loss: 10878.2627
Epoch 8/10

1/1 [==============================] - 0s 100ms/step - loss: 7507.5986
Epoch 9/10

1/1 [==============================] - 0s 107ms/step - loss: 5549.5869
Epoch 10/10

1/1 [==============================] - 0s 99ms/step - loss: 4349.8315

  #### Inference Need return ypred, ytrue ######################### 
[[  2.1159813   -2.4540815   -1.0551748   -1.9571893   -1.0632966
   -0.9173534    0.9144751   -0.32865894   1.3711059    1.0483582
    0.37363958   0.8839195    0.13881683  -0.77027     -2.4377468
    1.762758     0.08499467  -2.430344     1.5233638    0.37553975
   -0.7927699    0.45647916   0.30048394  -2.850388     0.08759052
    0.12065133   0.5486146    0.56551886   0.8579942    1.2150388
   -0.85420495   0.8972292    0.94626284   2.2460225   -0.5364858
    1.6332128   -1.5831201    0.32244736   1.8823016   -0.52823466
   -1.0070425    1.121201    -0.33537734   0.3079112   -1.2175734
   -1.5006092    0.85700583   0.07470316  -2.5679946   -0.4370284
   -3.5861096   -0.45159486  -1.7478067    0.8270845   -0.87527925
    1.0903387   -0.11566842  -0.02687973  -0.5636636   -1.3797439
    0.37562704  12.263938     8.517457    11.105986    10.6733885
    9.149995     8.950536    11.680921     9.544787    11.541448
    7.3071094    9.904493    12.516893     9.090424    10.871102
   10.334993    10.303608    10.7166395   10.588219    11.283917
    8.676181    10.019215    10.005973    12.5941       9.488077
   10.714239    10.143479    13.192021    12.423137    11.027278
   11.1833515   12.120085     9.818808    10.368761    10.750886
   11.82222      8.792265     9.137034    10.167003    11.190316
    9.341568    11.749735    11.538381    10.361335    11.074316
   10.130798    10.96547      8.123087    10.603856     9.56121
   10.934975    12.2004175   11.0023775   10.408238     9.89762
    9.431445     9.316518    11.960617    11.615092     9.856272
   -0.31698954  -0.08910495  -0.13748586  -0.09981859  -0.9092351
    1.5896823   -0.22359377   0.13638976  -0.5519048   -3.1969554
   -1.1290374   -0.9390902    1.2671883    2.314211    -0.05985487
    0.29948336  -0.64686465   1.5384476   -0.14374314   0.14872134
    0.0274722   -1.5507964   -1.7413392   -0.5389692    0.14085937
    0.8564544    0.5123285    0.946707     2.3854861   -1.0717907
    0.8448733    1.505079     1.201775    -1.8882369   -1.0002472
    1.6190963   -1.1428742    0.7256589    1.0395517   -0.45354992
    1.5534002   -1.6224793   -0.5745686    1.3718464   -0.77764046
   -0.73018086  -2.72534      0.829006     1.7812147    0.9976702
    0.9545717   -2.1650739   -0.6015368    2.0403461    1.4573082
   -1.50033      1.123059     1.0640535    1.0029154    1.0497674
    0.47910237   0.2341823    2.9434962    0.3106509    1.2316682
    0.03130841   1.6776853    1.9312227    2.3757434    2.065845
    0.68151325   1.6064909    1.1342576    1.0601082    0.42463458
    1.417182     4.000934     0.45870113   0.73446023   0.5962446
    2.6194172    1.5211627    2.4637537    1.1012148    0.2064184
    0.48784804   0.49179488   0.09270191   1.0584202    0.48007983
    1.1762142    0.72180724   0.31647062   0.8540771    0.03623366
    0.52813786   0.53188896   2.6267927    3.3449206    0.7133255
    1.7141712    1.4163704    0.13808346   0.8610776    1.129692
    1.4109726    0.24215406   2.1558888    0.8738425    1.9958906
    0.18863589   0.3567217    0.38183576   0.57521963   0.8313011
    1.1682281    0.07214487   0.4340818    0.42848408   1.4147545
    0.36163157  13.364111    11.147707    12.642601     8.793579
    9.328405    12.081568    10.417517     9.656998    11.019362
   10.433811    13.135166     9.934519    11.946864    10.356796
   11.638268    10.2231655   10.081389     8.947462    10.588309
    7.711863     9.646548    10.03053     10.942772     7.0512133
   10.978683    10.386374    10.474969    11.09478     10.076614
   12.011632     8.324552    13.090745    12.724183    10.158178
   11.887621     9.15372     12.505248    10.071851    10.403156
    8.676474    12.55868     12.978218     9.408394    10.4175005
   11.36538     10.422773    12.249822    10.174468    13.962115
   10.534258    12.062918    11.630451     8.025931    10.61405
    9.57703     10.78014     12.00389     10.630814     8.41212
    3.6705608    0.8850173    3.113309     1.827754     0.9716161
    0.2679032    0.8162911    0.4451062    0.873454     2.5352
    1.231673     0.3379308    1.6294484    0.59905994   0.44061923
    2.0294743    0.26455516   0.35263848   0.11670512   2.9215813
    0.12797308   3.2326646    0.8473071    0.8215109    2.8297281
    2.6518197    0.2333175    3.4594507    2.3369832    0.57638335
    1.3476803    0.9241917    0.2708714    2.717061     1.4449542
    2.1970506    1.3539914    0.30478066   2.101336     3.5294404
    0.6438797    1.7373544    2.2705696    0.90814775   1.0044374
    1.0950121    1.4073856    2.5824206    0.301607     2.327417
    1.0629036    0.15397573   2.1133454    1.5619733    1.4858192
    3.2122817    1.1565537    0.4927354    0.20181954   1.3715743
   -4.5998735   11.548603   -14.2208605 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-16 02:14:31.357643
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.5431
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-16 02:14:31.361695
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8411.35
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-16 02:14:31.365129
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.2507
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-16 02:14:31.369178
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -752.296
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140556695339528
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140554166121024
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140554166121528
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140554166122032
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140554166122536
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140554166123040

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fd5e88ec470> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.395346
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.368198
grad_step = 000002, loss = 0.346587
grad_step = 000003, loss = 0.323961
grad_step = 000004, loss = 0.300742
grad_step = 000005, loss = 0.279595
grad_step = 000006, loss = 0.269484
grad_step = 000007, loss = 0.263120
grad_step = 000008, loss = 0.245708
grad_step = 000009, loss = 0.227824
grad_step = 000010, loss = 0.215142
grad_step = 000011, loss = 0.207891
grad_step = 000012, loss = 0.204274
grad_step = 000013, loss = 0.198116
grad_step = 000014, loss = 0.189653
grad_step = 000015, loss = 0.181285
grad_step = 000016, loss = 0.173427
grad_step = 000017, loss = 0.165221
grad_step = 000018, loss = 0.156501
grad_step = 000019, loss = 0.148096
grad_step = 000020, loss = 0.140808
grad_step = 000021, loss = 0.134113
grad_step = 000022, loss = 0.126812
grad_step = 000023, loss = 0.118739
grad_step = 000024, loss = 0.111334
grad_step = 000025, loss = 0.105290
grad_step = 000026, loss = 0.100088
grad_step = 000027, loss = 0.094777
grad_step = 000028, loss = 0.089116
grad_step = 000029, loss = 0.083654
grad_step = 000030, loss = 0.078747
grad_step = 000031, loss = 0.074163
grad_step = 000032, loss = 0.069693
grad_step = 000033, loss = 0.065292
grad_step = 000034, loss = 0.061046
grad_step = 000035, loss = 0.057036
grad_step = 000036, loss = 0.053177
grad_step = 000037, loss = 0.049432
grad_step = 000038, loss = 0.045929
grad_step = 000039, loss = 0.042672
grad_step = 000040, loss = 0.039629
grad_step = 000041, loss = 0.036683
grad_step = 000042, loss = 0.033758
grad_step = 000043, loss = 0.031044
grad_step = 000044, loss = 0.028659
grad_step = 000045, loss = 0.026430
grad_step = 000046, loss = 0.024295
grad_step = 000047, loss = 0.022278
grad_step = 000048, loss = 0.020337
grad_step = 000049, loss = 0.018540
grad_step = 000050, loss = 0.016982
grad_step = 000051, loss = 0.015542
grad_step = 000052, loss = 0.014116
grad_step = 000053, loss = 0.012826
grad_step = 000054, loss = 0.011709
grad_step = 000055, loss = 0.010647
grad_step = 000056, loss = 0.009633
grad_step = 000057, loss = 0.008743
grad_step = 000058, loss = 0.007975
grad_step = 000059, loss = 0.007282
grad_step = 000060, loss = 0.006626
grad_step = 000061, loss = 0.006029
grad_step = 000062, loss = 0.005515
grad_step = 000063, loss = 0.005052
grad_step = 000064, loss = 0.004645
grad_step = 000065, loss = 0.004295
grad_step = 000066, loss = 0.003969
grad_step = 000067, loss = 0.003686
grad_step = 000068, loss = 0.003456
grad_step = 000069, loss = 0.003254
grad_step = 000070, loss = 0.003081
grad_step = 000071, loss = 0.002934
grad_step = 000072, loss = 0.002813
grad_step = 000073, loss = 0.002717
grad_step = 000074, loss = 0.002632
grad_step = 000075, loss = 0.002560
grad_step = 000076, loss = 0.002508
grad_step = 000077, loss = 0.002469
grad_step = 000078, loss = 0.002437
grad_step = 000079, loss = 0.002404
grad_step = 000080, loss = 0.002381
grad_step = 000081, loss = 0.002369
grad_step = 000082, loss = 0.002353
grad_step = 000083, loss = 0.002338
grad_step = 000084, loss = 0.002327
grad_step = 000085, loss = 0.002318
grad_step = 000086, loss = 0.002310
grad_step = 000087, loss = 0.002301
grad_step = 000088, loss = 0.002293
grad_step = 000089, loss = 0.002286
grad_step = 000090, loss = 0.002279
grad_step = 000091, loss = 0.002270
grad_step = 000092, loss = 0.002263
grad_step = 000093, loss = 0.002255
grad_step = 000094, loss = 0.002246
grad_step = 000095, loss = 0.002237
grad_step = 000096, loss = 0.002228
grad_step = 000097, loss = 0.002220
grad_step = 000098, loss = 0.002211
grad_step = 000099, loss = 0.002202
grad_step = 000100, loss = 0.002193
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002185
grad_step = 000102, loss = 0.002176
grad_step = 000103, loss = 0.002168
grad_step = 000104, loss = 0.002160
grad_step = 000105, loss = 0.002152
grad_step = 000106, loss = 0.002145
grad_step = 000107, loss = 0.002138
grad_step = 000108, loss = 0.002131
grad_step = 000109, loss = 0.002125
grad_step = 000110, loss = 0.002119
grad_step = 000111, loss = 0.002113
grad_step = 000112, loss = 0.002108
grad_step = 000113, loss = 0.002103
grad_step = 000114, loss = 0.002098
grad_step = 000115, loss = 0.002093
grad_step = 000116, loss = 0.002089
grad_step = 000117, loss = 0.002084
grad_step = 000118, loss = 0.002080
grad_step = 000119, loss = 0.002075
grad_step = 000120, loss = 0.002071
grad_step = 000121, loss = 0.002066
grad_step = 000122, loss = 0.002062
grad_step = 000123, loss = 0.002057
grad_step = 000124, loss = 0.002053
grad_step = 000125, loss = 0.002048
grad_step = 000126, loss = 0.002043
grad_step = 000127, loss = 0.002038
grad_step = 000128, loss = 0.002033
grad_step = 000129, loss = 0.002027
grad_step = 000130, loss = 0.002022
grad_step = 000131, loss = 0.002016
grad_step = 000132, loss = 0.002010
grad_step = 000133, loss = 0.002004
grad_step = 000134, loss = 0.001997
grad_step = 000135, loss = 0.001990
grad_step = 000136, loss = 0.001984
grad_step = 000137, loss = 0.001977
grad_step = 000138, loss = 0.001970
grad_step = 000139, loss = 0.001963
grad_step = 000140, loss = 0.001955
grad_step = 000141, loss = 0.001947
grad_step = 000142, loss = 0.001941
grad_step = 000143, loss = 0.001932
grad_step = 000144, loss = 0.001924
grad_step = 000145, loss = 0.001918
grad_step = 000146, loss = 0.001909
grad_step = 000147, loss = 0.001899
grad_step = 000148, loss = 0.001891
grad_step = 000149, loss = 0.001883
grad_step = 000150, loss = 0.001873
grad_step = 000151, loss = 0.001866
grad_step = 000152, loss = 0.001859
grad_step = 000153, loss = 0.001851
grad_step = 000154, loss = 0.001841
grad_step = 000155, loss = 0.001832
grad_step = 000156, loss = 0.001825
grad_step = 000157, loss = 0.001818
grad_step = 000158, loss = 0.001809
grad_step = 000159, loss = 0.001802
grad_step = 000160, loss = 0.001803
grad_step = 000161, loss = 0.001807
grad_step = 000162, loss = 0.001815
grad_step = 000163, loss = 0.001833
grad_step = 000164, loss = 0.001837
grad_step = 000165, loss = 0.001796
grad_step = 000166, loss = 0.001753
grad_step = 000167, loss = 0.001734
grad_step = 000168, loss = 0.001733
grad_step = 000169, loss = 0.001756
grad_step = 000170, loss = 0.001794
grad_step = 000171, loss = 0.001846
grad_step = 000172, loss = 0.001848
grad_step = 000173, loss = 0.001795
grad_step = 000174, loss = 0.001704
grad_step = 000175, loss = 0.001690
grad_step = 000176, loss = 0.001746
grad_step = 000177, loss = 0.001767
grad_step = 000178, loss = 0.001729
grad_step = 000179, loss = 0.001680
grad_step = 000180, loss = 0.001660
grad_step = 000181, loss = 0.001692
grad_step = 000182, loss = 0.001719
grad_step = 000183, loss = 0.001714
grad_step = 000184, loss = 0.001695
grad_step = 000185, loss = 0.001666
grad_step = 000186, loss = 0.001633
grad_step = 000187, loss = 0.001625
grad_step = 000188, loss = 0.001628
grad_step = 000189, loss = 0.001636
grad_step = 000190, loss = 0.001659
grad_step = 000191, loss = 0.001698
grad_step = 000192, loss = 0.001750
grad_step = 000193, loss = 0.001791
grad_step = 000194, loss = 0.001798
grad_step = 000195, loss = 0.001693
grad_step = 000196, loss = 0.001606
grad_step = 000197, loss = 0.001591
grad_step = 000198, loss = 0.001640
grad_step = 000199, loss = 0.001689
grad_step = 000200, loss = 0.001665
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001599
grad_step = 000202, loss = 0.001566
grad_step = 000203, loss = 0.001573
grad_step = 000204, loss = 0.001602
grad_step = 000205, loss = 0.001633
grad_step = 000206, loss = 0.001639
grad_step = 000207, loss = 0.001603
grad_step = 000208, loss = 0.001569
grad_step = 000209, loss = 0.001542
grad_step = 000210, loss = 0.001527
grad_step = 000211, loss = 0.001534
grad_step = 000212, loss = 0.001553
grad_step = 000213, loss = 0.001574
grad_step = 000214, loss = 0.001603
grad_step = 000215, loss = 0.001645
grad_step = 000216, loss = 0.001666
grad_step = 000217, loss = 0.001678
grad_step = 000218, loss = 0.001648
grad_step = 000219, loss = 0.001581
grad_step = 000220, loss = 0.001509
grad_step = 000221, loss = 0.001486
grad_step = 000222, loss = 0.001511
grad_step = 000223, loss = 0.001552
grad_step = 000224, loss = 0.001583
grad_step = 000225, loss = 0.001576
grad_step = 000226, loss = 0.001545
grad_step = 000227, loss = 0.001497
grad_step = 000228, loss = 0.001464
grad_step = 000229, loss = 0.001453
grad_step = 000230, loss = 0.001461
grad_step = 000231, loss = 0.001485
grad_step = 000232, loss = 0.001524
grad_step = 000233, loss = 0.001590
grad_step = 000234, loss = 0.001680
grad_step = 000235, loss = 0.001712
grad_step = 000236, loss = 0.001648
grad_step = 000237, loss = 0.001525
grad_step = 000238, loss = 0.001436
grad_step = 000239, loss = 0.001429
grad_step = 000240, loss = 0.001476
grad_step = 000241, loss = 0.001527
grad_step = 000242, loss = 0.001537
grad_step = 000243, loss = 0.001488
grad_step = 000244, loss = 0.001425
grad_step = 000245, loss = 0.001399
grad_step = 000246, loss = 0.001409
grad_step = 000247, loss = 0.001436
grad_step = 000248, loss = 0.001472
grad_step = 000249, loss = 0.001502
grad_step = 000250, loss = 0.001492
grad_step = 000251, loss = 0.001464
grad_step = 000252, loss = 0.001429
grad_step = 000253, loss = 0.001392
grad_step = 000254, loss = 0.001363
grad_step = 000255, loss = 0.001356
grad_step = 000256, loss = 0.001365
grad_step = 000257, loss = 0.001380
grad_step = 000258, loss = 0.001407
grad_step = 000259, loss = 0.001454
grad_step = 000260, loss = 0.001514
grad_step = 000261, loss = 0.001577
grad_step = 000262, loss = 0.001560
grad_step = 000263, loss = 0.001505
grad_step = 000264, loss = 0.001387
grad_step = 000265, loss = 0.001322
grad_step = 000266, loss = 0.001319
grad_step = 000267, loss = 0.001357
grad_step = 000268, loss = 0.001415
grad_step = 000269, loss = 0.001423
grad_step = 000270, loss = 0.001416
grad_step = 000271, loss = 0.001359
grad_step = 000272, loss = 0.001314
grad_step = 000273, loss = 0.001292
grad_step = 000274, loss = 0.001280
grad_step = 000275, loss = 0.001277
grad_step = 000276, loss = 0.001276
grad_step = 000277, loss = 0.001282
grad_step = 000278, loss = 0.001301
grad_step = 000279, loss = 0.001349
grad_step = 000280, loss = 0.001418
grad_step = 000281, loss = 0.001576
grad_step = 000282, loss = 0.001631
grad_step = 000283, loss = 0.001632
grad_step = 000284, loss = 0.001427
grad_step = 000285, loss = 0.001277
grad_step = 000286, loss = 0.001261
grad_step = 000287, loss = 0.001321
grad_step = 000288, loss = 0.001386
grad_step = 000289, loss = 0.001351
grad_step = 000290, loss = 0.001321
grad_step = 000291, loss = 0.001227
grad_step = 000292, loss = 0.001223
grad_step = 000293, loss = 0.001282
grad_step = 000294, loss = 0.001303
grad_step = 000295, loss = 0.001337
grad_step = 000296, loss = 0.001332
grad_step = 000297, loss = 0.001313
grad_step = 000298, loss = 0.001261
grad_step = 000299, loss = 0.001229
grad_step = 000300, loss = 0.001216
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001201
grad_step = 000302, loss = 0.001179
grad_step = 000303, loss = 0.001173
grad_step = 000304, loss = 0.001178
grad_step = 000305, loss = 0.001170
grad_step = 000306, loss = 0.001158
grad_step = 000307, loss = 0.001163
grad_step = 000308, loss = 0.001183
grad_step = 000309, loss = 0.001226
grad_step = 000310, loss = 0.001285
grad_step = 000311, loss = 0.001482
grad_step = 000312, loss = 0.001578
grad_step = 000313, loss = 0.001663
grad_step = 000314, loss = 0.001326
grad_step = 000315, loss = 0.001139
grad_step = 000316, loss = 0.001260
grad_step = 000317, loss = 0.001419
grad_step = 000318, loss = 0.001453
grad_step = 000319, loss = 0.001143
grad_step = 000320, loss = 0.001545
grad_step = 000321, loss = 0.002127
grad_step = 000322, loss = 0.001503
grad_step = 000323, loss = 0.001977
grad_step = 000324, loss = 0.002578
grad_step = 000325, loss = 0.001818
grad_step = 000326, loss = 0.002079
grad_step = 000327, loss = 0.001709
grad_step = 000328, loss = 0.001934
grad_step = 000329, loss = 0.001622
grad_step = 000330, loss = 0.001836
grad_step = 000331, loss = 0.001625
grad_step = 000332, loss = 0.001646
grad_step = 000333, loss = 0.001681
grad_step = 000334, loss = 0.001482
grad_step = 000335, loss = 0.001635
grad_step = 000336, loss = 0.001478
grad_step = 000337, loss = 0.001535
grad_step = 000338, loss = 0.001473
grad_step = 000339, loss = 0.001474
grad_step = 000340, loss = 0.001432
grad_step = 000341, loss = 0.001454
grad_step = 000342, loss = 0.001385
grad_step = 000343, loss = 0.001433
grad_step = 000344, loss = 0.001373
grad_step = 000345, loss = 0.001395
grad_step = 000346, loss = 0.001356
grad_step = 000347, loss = 0.001357
grad_step = 000348, loss = 0.001360
grad_step = 000349, loss = 0.001317
grad_step = 000350, loss = 0.001346
grad_step = 000351, loss = 0.001305
grad_step = 000352, loss = 0.001315
grad_step = 000353, loss = 0.001296
grad_step = 000354, loss = 0.001297
grad_step = 000355, loss = 0.001277
grad_step = 000356, loss = 0.001278
grad_step = 000357, loss = 0.001259
grad_step = 000358, loss = 0.001265
grad_step = 000359, loss = 0.001237
grad_step = 000360, loss = 0.001250
grad_step = 000361, loss = 0.001222
grad_step = 000362, loss = 0.001230
grad_step = 000363, loss = 0.001204
grad_step = 000364, loss = 0.001212
grad_step = 000365, loss = 0.001188
grad_step = 000366, loss = 0.001192
grad_step = 000367, loss = 0.001171
grad_step = 000368, loss = 0.001173
grad_step = 000369, loss = 0.001154
grad_step = 000370, loss = 0.001153
grad_step = 000371, loss = 0.001135
grad_step = 000372, loss = 0.001132
grad_step = 000373, loss = 0.001119
grad_step = 000374, loss = 0.001110
grad_step = 000375, loss = 0.001099
grad_step = 000376, loss = 0.001092
grad_step = 000377, loss = 0.001080
grad_step = 000378, loss = 0.001072
grad_step = 000379, loss = 0.001062
grad_step = 000380, loss = 0.001053
grad_step = 000381, loss = 0.001045
grad_step = 000382, loss = 0.001036
grad_step = 000383, loss = 0.001032
grad_step = 000384, loss = 0.001040
grad_step = 000385, loss = 0.001103
grad_step = 000386, loss = 0.001213
grad_step = 000387, loss = 0.001360
grad_step = 000388, loss = 0.001170
grad_step = 000389, loss = 0.001077
grad_step = 000390, loss = 0.001071
grad_step = 000391, loss = 0.001059
grad_step = 000392, loss = 0.001068
grad_step = 000393, loss = 0.001076
grad_step = 000394, loss = 0.001049
grad_step = 000395, loss = 0.001004
grad_step = 000396, loss = 0.001009
grad_step = 000397, loss = 0.001060
grad_step = 000398, loss = 0.001051
grad_step = 000399, loss = 0.001001
grad_step = 000400, loss = 0.001004
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001048
grad_step = 000402, loss = 0.001027
grad_step = 000403, loss = 0.001015
grad_step = 000404, loss = 0.001031
grad_step = 000405, loss = 0.001069
grad_step = 000406, loss = 0.001048
grad_step = 000407, loss = 0.001023
grad_step = 000408, loss = 0.000978
grad_step = 000409, loss = 0.000950
grad_step = 000410, loss = 0.000925
grad_step = 000411, loss = 0.000936
grad_step = 000412, loss = 0.000973
grad_step = 000413, loss = 0.000964
grad_step = 000414, loss = 0.000945
grad_step = 000415, loss = 0.000917
grad_step = 000416, loss = 0.000907
grad_step = 000417, loss = 0.000905
grad_step = 000418, loss = 0.000903
grad_step = 000419, loss = 0.000915
grad_step = 000420, loss = 0.000937
grad_step = 000421, loss = 0.000987
grad_step = 000422, loss = 0.000999
grad_step = 000423, loss = 0.001060
grad_step = 000424, loss = 0.000978
grad_step = 000425, loss = 0.000946
grad_step = 000426, loss = 0.000882
grad_step = 000427, loss = 0.000862
grad_step = 000428, loss = 0.000869
grad_step = 000429, loss = 0.000894
grad_step = 000430, loss = 0.000950
grad_step = 000431, loss = 0.000942
grad_step = 000432, loss = 0.000954
grad_step = 000433, loss = 0.000899
grad_step = 000434, loss = 0.000866
grad_step = 000435, loss = 0.000843
grad_step = 000436, loss = 0.000835
grad_step = 000437, loss = 0.000842
grad_step = 000438, loss = 0.000857
grad_step = 000439, loss = 0.000882
grad_step = 000440, loss = 0.000885
grad_step = 000441, loss = 0.000914
grad_step = 000442, loss = 0.000891
grad_step = 000443, loss = 0.000898
grad_step = 000444, loss = 0.000857
grad_step = 000445, loss = 0.000844
grad_step = 000446, loss = 0.000821
grad_step = 000447, loss = 0.000812
grad_step = 000448, loss = 0.000805
grad_step = 000449, loss = 0.000803
grad_step = 000450, loss = 0.000801
grad_step = 000451, loss = 0.000806
grad_step = 000452, loss = 0.000809
grad_step = 000453, loss = 0.000832
grad_step = 000454, loss = 0.000845
grad_step = 000455, loss = 0.000899
grad_step = 000456, loss = 0.000884
grad_step = 000457, loss = 0.000911
grad_step = 000458, loss = 0.000863
grad_step = 000459, loss = 0.000844
grad_step = 000460, loss = 0.000806
grad_step = 000461, loss = 0.000781
grad_step = 000462, loss = 0.000765
grad_step = 000463, loss = 0.000758
grad_step = 000464, loss = 0.000757
grad_step = 000465, loss = 0.000756
grad_step = 000466, loss = 0.000752
grad_step = 000467, loss = 0.000745
grad_step = 000468, loss = 0.000740
grad_step = 000469, loss = 0.000737
grad_step = 000470, loss = 0.000737
grad_step = 000471, loss = 0.000742
grad_step = 000472, loss = 0.000753
grad_step = 000473, loss = 0.000797
grad_step = 000474, loss = 0.000863
grad_step = 000475, loss = 0.001026
grad_step = 000476, loss = 0.000942
grad_step = 000477, loss = 0.000927
grad_step = 000478, loss = 0.000807
grad_step = 000479, loss = 0.000756
grad_step = 000480, loss = 0.000732
grad_step = 000481, loss = 0.000729
grad_step = 000482, loss = 0.000762
grad_step = 000483, loss = 0.000793
grad_step = 000484, loss = 0.000823
grad_step = 000485, loss = 0.000787
grad_step = 000486, loss = 0.000747
grad_step = 000487, loss = 0.000708
grad_step = 000488, loss = 0.000694
grad_step = 000489, loss = 0.000716
grad_step = 000490, loss = 0.000747
grad_step = 000491, loss = 0.000792
grad_step = 000492, loss = 0.000792
grad_step = 000493, loss = 0.000812
grad_step = 000494, loss = 0.000760
grad_step = 000495, loss = 0.000746
grad_step = 000496, loss = 0.000702
grad_step = 000497, loss = 0.000681
grad_step = 000498, loss = 0.000666
grad_step = 000499, loss = 0.000662
grad_step = 000500, loss = 0.000668
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000680
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

  date_run                              2020-05-16 02:14:54.801021
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.205432
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-16 02:14:54.807777
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0956105
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-16 02:14:54.815918
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.124086
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-16 02:14:54.822338
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.452836
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
0   2020-05-16 02:14:21.006435  ...    mean_absolute_error
1   2020-05-16 02:14:21.010885  ...     mean_squared_error
2   2020-05-16 02:14:21.014451  ...  median_absolute_error
3   2020-05-16 02:14:21.017913  ...               r2_score
4   2020-05-16 02:14:31.357643  ...    mean_absolute_error
5   2020-05-16 02:14:31.361695  ...     mean_squared_error
6   2020-05-16 02:14:31.365129  ...  median_absolute_error
7   2020-05-16 02:14:31.369178  ...               r2_score
8   2020-05-16 02:14:54.801021  ...    mean_absolute_error
9   2020-05-16 02:14:54.807777  ...     mean_squared_error
10  2020-05-16 02:14:54.815918  ...  median_absolute_error
11  2020-05-16 02:14:54.822338  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f11c735afd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:30, 321876.11it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 415713.82it/s]  9%|▉         | 876544/9912422 [00:00<00:15, 575891.05it/s] 31%|███       | 3039232/9912422 [00:00<00:08, 811694.76it/s] 59%|█████▊    | 5799936/9912422 [00:00<00:03, 1142300.48it/s] 89%|████████▉ | 8847360/9912422 [00:00<00:00, 1600822.80it/s]9920512it [00:00, 10206654.81it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 152065.02it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:04, 321548.28it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 414256.08it/s] 26%|██▋       | 434176/1648877 [00:00<00:02, 536361.03it/s]1654784it [00:00, 2802208.59it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 55673.78it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1179d5be48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1176ba80b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1179d5be48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f11792e20b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1176b22550> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1176b07c18> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1179d5be48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f11792a06d8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f1176b22550> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f11c7364ef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f2aa3cf71d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=ef40e9f6da243766f710ba558787101fb682c9dfc02254985f209ace99e96335
  Stored in directory: /tmp/pip-ephem-wheel-cache-2ul41q9i/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f2a99e62048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 43s
   57344/17464789 [..............................] - ETA: 37s
   90112/17464789 [..............................] - ETA: 35s
  212992/17464789 [..............................] - ETA: 19s
  319488/17464789 [..............................] - ETA: 16s
  524288/17464789 [..............................] - ETA: 11s
  958464/17464789 [>.............................] - ETA: 7s 
 1818624/17464789 [==>...........................] - ETA: 4s
 3506176/17464789 [=====>........................] - ETA: 2s
 6094848/17464789 [=========>....................] - ETA: 1s
 8978432/17464789 [==============>...............] - ETA: 0s
11665408/17464789 [===================>..........] - ETA: 0s
14417920/17464789 [=======================>......] - ETA: 0s
17268736/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-16 02:16:28.769015: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 02:16:28.774610: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-16 02:16:28.774862: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e9daa8e430 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 02:16:28.774882: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.5900 - accuracy: 0.5050
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6743 - accuracy: 0.4995
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6360 - accuracy: 0.5020 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6820 - accuracy: 0.4990
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6590 - accuracy: 0.5005
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6579 - accuracy: 0.5006
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6935 - accuracy: 0.4983
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6428 - accuracy: 0.5016
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7096 - accuracy: 0.4972
11000/25000 [============>.................] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
12000/25000 [=============>................] - ETA: 4s - loss: 7.6590 - accuracy: 0.5005
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6831 - accuracy: 0.4989
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6863 - accuracy: 0.4987
15000/25000 [=================>............] - ETA: 3s - loss: 7.6871 - accuracy: 0.4987
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6532 - accuracy: 0.5009
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6684 - accuracy: 0.4999
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6598 - accuracy: 0.5004
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6689 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6535 - accuracy: 0.5009
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6401 - accuracy: 0.5017
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6493 - accuracy: 0.5011
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
25000/25000 [==============================] - 10s 381us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-16 02:16:46.009313
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-16 02:16:46.009313  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:04<129:39:32, 1.85kB/s].vector_cache/glove.6B.zip:   0%|          | 303k/862M [00:04<90:45:20, 2.64kB/s]  .vector_cache/glove.6B.zip:   1%|          | 4.69M/862M [00:04<63:12:27, 3.77kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.3M/862M [00:04<43:41:58, 5.38kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.5M/862M [00:04<30:21:57, 7.69kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.1M/862M [00:04<21:00:52, 11.0kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.5M/862M [00:05<14:32:36, 15.7kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.7M/862M [00:05<10:03:17, 22.4kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:05<7:02:47, 31.9kB/s] .vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:06<4:55:19, 45.5kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:07<12:19:07, 18.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:07<8:36:41, 26.0kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:08<11:28:56, 19.5kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:08<8:01:35, 27.8kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:09<11:17:21, 19.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:09<7:53:30, 28.2kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:10<11:07:02, 20.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:10<7:46:17, 28.5kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:11<11:04:39, 20.0kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:11<7:44:38, 28.6kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:12<11:00:48, 20.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:12<7:41:57, 28.7kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:13<10:51:47, 20.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:13<7:35:37, 29.0kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:14<10:57:25, 20.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:14<7:39:33, 28.7kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:15<10:55:14, 20.1kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:15<7:38:03, 28.7kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:16<10:47:52, 20.3kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:16<7:32:52, 28.9kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:17<10:53:22, 20.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:17<7:36:48, 28.6kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:18<10:28:57, 20.8kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:18<7:19:39, 29.6kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:19<10:40:55, 20.3kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:19<7:28:01, 29.0kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:20<10:40:26, 20.3kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:20<7:27:40, 28.9kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:21<10:45:02, 20.1kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:21<7:30:53, 28.7kB/s] .vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:22<10:40:46, 20.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:22<7:27:54, 28.8kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:23<10:40:55, 20.1kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:23<7:28:00, 28.7kB/s] .vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:24<10:39:29, 20.1kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:24<7:27:06, 28.7kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:25<10:10:09, 21.0kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:25<7:06:30, 30.0kB/s] .vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:26<10:25:17, 20.4kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:26<7:17:09, 29.2kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:27<10:06:09, 21.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:27<7:03:43, 30.0kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:28<10:18:56, 20.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<7:12:39, 29.3kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<10:18:59, 20.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:29<7:12:39, 29.2kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:30<10:26:55, 20.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<7:18:11, 28.8kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<10:25:57, 20.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:31<7:17:32, 28.7kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:32<10:17:53, 20.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<7:11:52, 29.0kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<10:22:17, 20.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:33<7:14:57, 28.7kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<10:20:00, 20.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<7:13:22, 28.8kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<10:12:32, 20.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:35<7:08:07, 29.0kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:36<10:17:12, 20.1kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<7:11:23, 28.7kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<10:14:55, 20.2kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:37<7:09:49, 28.8kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:38<10:05:53, 20.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<7:03:27, 29.1kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<10:11:32, 20.2kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:39<7:07:30, 28.8kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:40<9:50:04, 20.8kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<6:52:24, 29.7kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<10:02:25, 20.3kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:41<7:01:03, 29.0kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:42<10:00:54, 20.3kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<6:59:58, 29.0kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<10:03:11, 20.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:43<7:01:35, 28.8kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:44<9:56:26, 20.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<6:56:56, 29.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<9:34:24, 21.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:45<6:41:27, 30.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:46<9:48:02, 20.5kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<6:51:00, 29.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<9:45:25, 20.6kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:47<6:49:07, 29.4kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:48<9:53:57, 20.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<6:55:06, 28.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<9:49:32, 20.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:49<6:51:59, 29.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:50<9:53:35, 20.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<6:54:49, 28.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:51<9:51:16, 20.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:51<6:53:13, 28.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:52<9:41:59, 20.4kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<6:46:42, 29.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:53<9:47:15, 20.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:53<6:50:28, 28.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:54<9:26:12, 20.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<6:35:46, 29.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:55<9:16:54, 21.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:55<6:29:11, 30.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:56<9:33:57, 20.4kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<6:41:06, 29.2kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:57<9:36:22, 20.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:57<6:42:53, 29.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:58<9:07:53, 21.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<6:22:52, 30.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<9:25:55, 20.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:59<6:35:29, 29.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:00<9:28:41, 20.4kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:00<6:37:27, 29.1kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:01<9:14:10, 20.9kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:01<6:27:15, 29.8kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:02<9:24:59, 20.4kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:02<6:34:48, 29.1kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:03<9:27:48, 20.2kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:03<6:36:47, 28.9kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:04<9:21:36, 20.4kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:04<6:32:25, 29.1kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:05<9:27:05, 20.1kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:05<6:36:16, 28.7kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:06<9:25:21, 20.1kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<6:35:04, 28.7kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:07<9:16:58, 20.4kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:07<6:29:10, 29.1kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:08<9:21:59, 20.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:08<6:32:41, 28.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:09<9:18:40, 20.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<6:30:21, 28.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<9:19:40, 20.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:10<6:31:03, 28.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:11<9:16:30, 20.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:11<6:28:51, 28.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<9:08:57, 20.4kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:12<6:23:33, 29.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:13<9:14:37, 20.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:13<6:27:31, 28.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<9:11:29, 20.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:14<6:25:21, 28.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<9:04:36, 20.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:15<6:20:30, 29.0kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:16<9:08:36, 20.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<6:23:18, 28.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:17<9:07:10, 20.1kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:17<6:22:18, 28.7kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:18<8:59:28, 20.3kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:18<6:16:54, 29.0kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<9:02:48, 20.1kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:19<6:19:15, 28.7kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:20<8:53:14, 20.4kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:20<6:12:33, 29.2kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:21<8:57:59, 20.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<6:15:51, 28.8kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<8:57:59, 20.1kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:22<6:15:52, 28.7kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<8:50:43, 20.3kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<6:10:46, 29.0kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<8:54:26, 20.1kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:24<6:13:21, 28.7kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:25<8:54:00, 20.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<6:13:05, 28.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<8:45:20, 20.3kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:26<6:07:00, 29.0kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:27<8:49:29, 20.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<6:09:54, 28.7kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<8:47:34, 20.1kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<6:08:35, 28.7kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:29<8:39:48, 20.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<6:03:07, 29.0kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:30<8:44:32, 20.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:30<6:06:25, 28.7kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:31<8:43:23, 20.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<6:05:39, 28.6kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<8:34:25, 20.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:32<5:59:21, 29.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:33<8:35:07, 20.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<5:59:50, 28.9kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<8:34:54, 20.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:34<5:59:43, 28.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<8:28:35, 20.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<5:55:15, 29.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<8:33:48, 20.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:36<5:58:54, 28.7kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:37<8:32:09, 20.1kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<5:57:46, 28.7kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:38<8:24:23, 20.4kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:38<5:52:18, 29.0kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:39<8:28:37, 20.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<5:55:16, 28.7kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:40<8:26:58, 20.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:40<5:54:08, 28.7kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:41<8:19:33, 20.3kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:41<5:48:55, 29.0kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<8:23:58, 20.1kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:42<5:52:01, 28.7kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:43<8:22:24, 20.1kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:43<5:51:01, 28.6kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:44<7:54:21, 21.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:44<5:31:19, 30.2kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:45<8:07:34, 20.6kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:45<5:40:37, 29.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:46<7:54:02, 21.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:46<5:31:09, 30.0kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:47<7:49:48, 21.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<5:28:08, 30.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<8:02:42, 20.5kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:48<5:37:08, 29.3kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:49<8:06:38, 20.3kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<5:39:54, 29.0kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:50<8:00:46, 20.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:50<5:35:46, 29.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:51<8:05:19, 20.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:51<5:38:57, 28.8kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:52<8:05:18, 20.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:52<5:38:57, 28.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:53<7:58:51, 20.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:53<5:34:25, 29.0kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:54<8:02:05, 20.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:54<5:36:41, 28.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:55<7:57:51, 20.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:55<5:33:43, 28.9kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:56<7:59:47, 20.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:56<5:35:03, 28.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:57<7:58:15, 20.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<5:34:00, 28.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:58<7:49:33, 20.4kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:58<5:27:54, 29.1kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:59<7:52:59, 20.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<5:30:18, 28.8kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<7:52:22, 20.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:00<5:29:53, 28.7kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:01<7:44:25, 20.4kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:01<5:24:18, 29.1kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:02<7:48:20, 20.1kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:02<5:27:03, 28.7kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:03<7:46:26, 20.1kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:03<5:25:44, 28.7kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:04<7:39:18, 20.4kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:04<5:20:47, 29.1kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:05<7:26:56, 20.9kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:05<5:12:05, 29.8kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:06<7:36:13, 20.4kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:06<5:18:34, 29.0kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:07<7:37:21, 20.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<5:19:22, 28.9kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:08<7:31:36, 20.4kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:08<5:15:20, 29.1kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:09<7:35:50, 20.1kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<5:18:17, 28.7kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:10<7:31:18, 20.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:10<5:15:06, 28.9kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:11<7:33:26, 20.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<5:16:35, 28.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:12<7:31:42, 20.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:12<5:15:23, 28.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:13<7:24:51, 20.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<5:10:35, 29.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:14<7:28:33, 20.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<5:13:10, 28.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:15<7:26:37, 20.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:15<5:11:50, 28.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:16<7:18:09, 20.4kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:16<5:05:54, 29.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:17<7:20:55, 20.2kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<5:07:49, 28.8kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:18<7:20:12, 20.1kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:18<5:07:20, 28.7kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:19<7:13:43, 20.4kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:19<5:02:47, 29.1kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:20<7:14:56, 20.2kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:20<5:03:42, 28.9kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:21<6:59:34, 20.9kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:21<4:52:54, 29.8kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:22<7:07:40, 20.4kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<4:58:33, 29.1kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:23<7:08:07, 20.3kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:23<4:58:53, 29.0kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<7:02:59, 20.5kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:24<4:55:16, 29.2kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:25<7:06:26, 20.2kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:25<4:57:41, 28.8kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:26<7:05:13, 20.2kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:26<4:56:51, 28.8kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:27<6:58:34, 20.4kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<4:52:10, 29.2kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:28<7:01:39, 20.2kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<4:54:19, 28.8kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:29<7:01:07, 20.1kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<4:53:58, 28.7kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:30<6:53:58, 20.4kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<4:48:57, 29.1kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:31<6:56:56, 20.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<4:51:04, 28.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:32<6:41:19, 20.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<4:40:07, 29.8kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:33<6:48:57, 20.4kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:33<4:45:26, 29.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:34<6:50:27, 20.2kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<4:46:29, 28.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<6:45:34, 20.4kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:35<4:43:03, 29.1kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:36<6:49:11, 20.1kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<4:45:38, 28.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:37<6:34:46, 20.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:37<4:35:34, 29.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:38<6:27:03, 21.1kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<4:30:08, 30.1kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:39<6:36:49, 20.5kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:39<4:37:00, 29.2kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:40<6:24:08, 21.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<4:28:05, 30.1kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:41<6:34:06, 20.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:41<4:35:02, 29.2kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:42<6:35:19, 20.3kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<4:35:54, 29.0kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<6:29:34, 20.5kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:43<4:31:51, 29.3kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:44<6:33:28, 20.2kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<4:34:35, 28.9kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<6:28:45, 20.4kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:45<4:31:16, 29.1kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:46<6:31:22, 20.2kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:46<4:33:06, 28.8kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:47<6:29:49, 20.1kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:47<4:32:01, 28.7kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:48<6:24:59, 20.3kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<4:28:37, 29.0kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:49<6:27:17, 20.1kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:49<4:30:14, 28.7kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:50<6:25:32, 20.1kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:50<4:29:01, 28.7kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:51<6:19:08, 20.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:51<4:24:35, 29.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:52<6:08:30, 20.8kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<4:17:07, 29.7kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<6:15:50, 20.3kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:53<4:22:17, 29.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:54<6:01:56, 21.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:54<4:12:31, 30.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<6:09:43, 20.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:55<4:17:57, 29.2kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:56<6:10:25, 20.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<4:18:25, 29.0kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:57<6:13:36, 20.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:57<4:20:38, 28.7kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:58<6:12:11, 20.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<4:19:39, 28.6kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:59<6:08:50, 20.2kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:59<4:17:17, 28.8kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:00<6:09:12, 20.0kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:00<4:17:36, 28.6kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:01<5:53:58, 20.8kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:01<4:06:55, 29.7kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<5:59:43, 20.4kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:02<4:10:56, 29.1kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<6:00:40, 20.2kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:03<4:11:36, 28.9kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:04<5:55:55, 20.4kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:04<4:08:15, 29.1kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<5:59:18, 20.1kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:05<4:10:37, 28.7kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:06<5:58:19, 20.1kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<4:09:56, 28.6kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<5:51:22, 20.4kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<4:05:04, 29.1kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<5:53:47, 20.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:08<4:06:45, 28.7kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:09<5:52:06, 20.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<4:05:35, 28.7kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<5:45:45, 20.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<4:01:08, 29.1kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<5:48:38, 20.1kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<4:03:09, 28.7kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<5:43:59, 20.3kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<3:59:53, 29.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<5:45:58, 20.1kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<4:01:16, 28.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:14<5:42:56, 20.1kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:14<3:59:13, 28.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<5:25:54, 21.1kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:15<3:47:16, 30.1kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:16<5:33:37, 20.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:16<3:52:39, 29.3kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:17<5:34:26, 20.3kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<3:53:13, 29.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:18<5:31:32, 20.4kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:18<3:51:10, 29.1kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<5:32:55, 20.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<3:52:09, 28.9kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:20<5:29:57, 20.3kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<3:50:03, 29.0kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:21<5:31:32, 20.1kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<3:51:09, 28.7kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:22<5:29:38, 20.1kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:22<3:49:53, 28.7kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<5:13:26, 21.0kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<3:38:32, 30.0kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:24<5:20:58, 20.4kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:24<3:43:49, 29.2kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:25<5:10:29, 21.0kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:25<3:36:30, 30.0kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:26<5:04:54, 21.3kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:26<3:32:34, 30.4kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:27<5:14:05, 20.6kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:27<3:38:57, 29.3kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<5:15:45, 20.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:28<3:40:07, 29.0kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:29<5:12:15, 20.5kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<3:37:40, 29.2kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:30<5:13:21, 20.3kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:30<3:38:27, 28.9kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:31<5:07:30, 20.5kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:31<3:34:21, 29.3kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:32<5:09:09, 20.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:32<3:35:30, 29.0kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:33<5:02:05, 20.7kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:33<3:30:33, 29.5kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:34<5:04:05, 20.4kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:34<3:31:58, 29.1kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:35<4:59:40, 20.6kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:35<3:28:51, 29.4kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:36<5:02:15, 20.3kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:36<3:30:51, 29.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:36<2:28:38, 41.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:37<3:35:19, 28.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:37<2:30:07, 40.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<4:19:21, 23.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:38<3:00:48, 33.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:39<4:34:49, 22.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:39<3:11:32, 31.3kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<4:45:01, 21.1kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:40<3:18:37, 30.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:41<4:50:51, 20.5kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<3:22:41, 29.3kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<4:47:13, 20.7kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<3:20:08, 29.5kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:43<4:50:02, 20.3kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:43<3:22:08, 29.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:44<4:35:06, 21.3kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:44<3:11:41, 30.4kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:45<4:41:41, 20.7kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:45<3:16:16, 29.5kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<4:39:32, 20.7kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:46<3:14:47, 29.6kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:47<4:32:41, 21.1kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<3:09:58, 30.1kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<4:38:40, 20.5kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:48<3:14:08, 29.3kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:49<4:35:49, 20.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:49<3:12:08, 29.4kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<4:38:28, 20.3kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:50<3:13:59, 29.0kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:51<4:33:54, 20.5kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<3:10:47, 29.3kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<4:35:41, 20.2kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:52<3:12:02, 28.9kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:53<4:30:51, 20.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:53<3:08:39, 29.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:54<4:31:40, 20.3kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<3:09:13, 28.9kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<4:27:11, 20.5kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:55<3:06:05, 29.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<4:28:08, 20.3kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<3:06:45, 29.0kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<4:23:45, 20.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:57<3:03:43, 29.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<4:14:59, 21.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:58<2:57:34, 30.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:59<4:18:51, 20.6kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:59<3:01:40, 29.3kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:59<2:37:31, 33.8kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<1:50:03, 48.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<2:09:11, 41.0kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<1:30:03, 58.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:01<3:20:27, 26.3kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:01<2:19:40, 37.5kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<3:37:25, 24.1kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:02<2:31:24, 34.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:03<3:57:37, 21.9kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:03<2:45:27, 31.2kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<4:03:27, 21.2kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<2:49:31, 30.2kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<3:59:56, 21.4kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:05<2:47:02, 30.5kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<4:04:47, 20.8kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<2:50:24, 29.7kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:07<4:07:44, 20.4kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:07<2:52:29, 29.1kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<3:55:59, 21.3kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<2:44:16, 30.4kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:09<4:01:50, 20.6kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:09<2:48:20, 29.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:10<3:59:33, 20.7kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<2:46:43, 29.5kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:11<4:01:01, 20.4kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:11<2:47:47, 29.1kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<3:49:32, 21.3kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:12<2:39:44, 30.3kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:13<3:54:57, 20.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:13<2:43:30, 29.4kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:14<3:52:38, 20.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<2:41:55, 29.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:15<3:43:38, 21.4kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:15<2:35:38, 30.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:16<3:41:18, 21.4kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<2:33:59, 30.6kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:17<3:47:46, 20.7kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:17<2:38:28, 29.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:18<3:45:04, 20.8kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:18<2:36:36, 29.6kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:19<3:39:38, 21.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:19<2:32:48, 30.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:20<3:42:15, 20.7kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<2:34:36, 29.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:21<3:43:52, 20.4kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:21<2:35:44, 29.1kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<3:39:54, 20.6kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<2:32:56, 29.4kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:23<3:41:05, 20.3kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:23<2:33:46, 29.0kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<3:37:40, 20.5kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<2:31:22, 29.3kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:25<3:38:28, 20.3kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:25<2:31:58, 28.9kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:26<3:25:38, 21.4kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:26<2:23:01, 30.5kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:27<3:23:52, 21.4kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:27<2:21:47, 30.5kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:28<3:21:32, 21.5kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<2:20:08, 30.6kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:29<3:26:20, 20.8kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:29<2:23:27, 29.7kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:30<3:24:59, 20.8kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<2:22:31, 29.6kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:31<3:19:47, 21.1kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:31<2:18:52, 30.1kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:32<3:23:51, 20.5kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<2:21:42, 29.3kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:33<3:20:43, 20.7kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<2:19:30, 29.5kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:34<3:22:38, 20.3kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<2:20:50, 29.0kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:35<3:17:53, 20.6kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:35<2:17:32, 29.4kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:36<3:11:06, 21.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:36<2:12:48, 30.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:37<3:12:52, 20.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:37<2:14:00, 29.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:38<3:14:47, 20.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:38<2:15:20, 29.1kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<3:11:28, 20.6kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:39<2:13:02, 29.3kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:40<3:04:42, 21.1kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<2:08:19, 30.2kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:41<3:07:42, 20.6kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:41<2:10:23, 29.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:42<3:05:33, 20.7kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:42<2:08:52, 29.5kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:43<3:06:17, 20.4kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:43<2:09:23, 29.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:44<3:02:07, 20.7kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:44<2:06:30, 29.5kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<2:56:33, 21.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:45<2:02:35, 30.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:46<2:59:24, 20.6kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:46<2:04:34, 29.4kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<2:57:42, 20.6kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:47<2:03:22, 29.4kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<2:58:12, 20.3kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:48<2:03:42, 29.0kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<2:54:50, 20.5kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:49<2:01:21, 29.3kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:50<2:54:58, 20.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:50<2:01:28, 29.0kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:51<2:45:26, 21.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:51<1:54:50, 30.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<2:42:03, 21.5kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:52<1:52:27, 30.7kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<2:46:08, 20.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:53<1:55:17, 29.6kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:54<2:44:12, 20.8kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:54<1:53:55, 29.7kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:55<2:44:29, 20.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<1:54:06, 29.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:56<2:44:46, 20.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:56<1:54:17, 29.0kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<2:41:10, 20.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:57<1:51:46, 29.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:58<2:41:27, 20.3kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:58<1:51:58, 28.9kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<2:36:57, 20.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<1:48:49, 29.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<2:37:13, 20.4kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:00<1:49:00, 29.1kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<2:33:40, 20.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<1:46:31, 29.4kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:02<2:34:10, 20.3kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:02<1:46:52, 29.0kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<2:30:01, 20.7kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<1:43:58, 29.5kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:04<2:30:46, 20.3kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:04<1:44:29, 29.0kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<2:27:07, 20.6kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<1:41:56, 29.4kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:06<2:27:39, 20.3kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:06<1:42:18, 28.9kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<2:19:06, 21.3kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<1:36:21, 30.4kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:08<2:20:33, 20.8kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:08<1:37:20, 29.7kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<2:21:45, 20.4kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<1:38:10, 29.1kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:10<2:18:48, 20.6kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:10<1:36:06, 29.4kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<2:19:00, 20.3kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<1:36:14, 28.9kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:12<2:15:45, 20.5kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:12<1:33:57, 29.3kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<2:15:00, 20.4kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<1:33:25, 29.1kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:14<2:14:38, 20.2kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:14<1:33:09, 28.8kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<2:11:09, 20.4kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:15<1:30:45, 29.2kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:16<2:05:50, 21.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:16<1:27:02, 30.0kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<2:07:04, 20.5kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:17<1:27:53, 29.3kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:18<2:04:52, 20.6kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<1:26:20, 29.4kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<2:05:28, 20.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<1:26:44, 28.9kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:20<2:01:14, 20.7kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:20<1:23:47, 29.5kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<2:01:38, 20.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:21<1:24:04, 29.0kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:22<1:58:42, 20.5kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:22<1:22:01, 29.3kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:23<1:53:58, 21.1kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:23<1:18:43, 30.1kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:24<1:55:20, 20.5kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:24<1:19:39, 29.3kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:25<1:52:44, 20.7kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:25<1:17:50, 29.5kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<1:52:57, 20.3kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:26<1:17:58, 29.0kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:27<1:50:11, 20.5kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:27<1:16:02, 29.3kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:28<1:49:39, 20.3kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:28<1:15:40, 29.0kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:29<1:43:08, 21.2kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:29<1:11:08, 30.3kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:30<1:44:33, 20.6kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<1:12:06, 29.4kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:31<1:42:13, 20.8kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:31<1:10:28, 29.6kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:32<1:42:20, 20.4kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:32<1:10:32, 29.1kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:33<1:39:40, 20.6kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:33<1:08:40, 29.4kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:34<1:38:42, 20.4kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<1:07:59, 29.2kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:35<1:38:13, 20.2kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:35<1:07:38, 28.8kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:36<1:35:10, 20.5kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:36<1:05:31, 29.2kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<1:30:50, 21.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:37<1:02:30, 30.0kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:38<1:31:46, 20.5kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:38<1:03:08, 29.2kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<1:29:25, 20.6kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:39<1:01:29, 29.4kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:40<1:29:02, 20.3kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:40<1:01:12, 29.0kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<1:26:04, 20.6kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:41<59:09, 29.4kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:42<1:25:03, 20.4kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<58:25, 29.1kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:43<1:21:15, 20.9kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:43<55:47, 29.9kB/s]  .vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:44<1:21:16, 20.5kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:44<55:46, 29.3kB/s]  .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<1:18:54, 20.7kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:45<54:07, 29.5kB/s]  .vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:46<1:18:07, 20.4kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<53:33, 29.2kB/s]  .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:47<1:17:20, 20.2kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:47<53:00, 28.8kB/s]  .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<1:12:03, 21.2kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<49:21, 30.2kB/s]  .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:49<1:09:46, 21.4kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:49<47:46, 30.5kB/s]  .vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<1:10:22, 20.7kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<48:09, 29.5kB/s]  .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:51<1:08:46, 20.7kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:51<47:01, 29.5kB/s]  .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:52<1:08:06, 20.4kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<46:32, 29.1kB/s]  .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:53<1:05:43, 20.6kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:53<44:52, 29.4kB/s]  .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:54<1:04:49, 20.3kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<44:14, 29.0kB/s]  .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:55<1:02:25, 20.6kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:55<42:33, 29.3kB/s]  .vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:56<1:01:39, 20.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<42:00, 28.9kB/s]  .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:57<59:20, 20.4kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:57<40:24, 29.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<58:17, 20.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<39:39, 28.8kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:59<53:41, 21.3kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:59<36:29, 30.4kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<51:37, 21.5kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<35:02, 30.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:01<51:28, 20.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:01<34:54, 29.7kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<50:39, 20.5kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:02<34:18, 29.2kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:03<48:48, 20.6kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:03<33:01, 29.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<47:45, 20.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:04<32:16, 28.9kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:05<45:23, 20.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:05<30:37, 29.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:06<42:34, 21.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<28:40, 30.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<41:58, 20.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:07<28:14, 29.3kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:08<40:01, 20.7kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:08<26:52, 29.5kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:09<39:04, 20.3kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<26:11, 29.0kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<35:12, 21.5kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:10<23:33, 30.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:11<34:47, 20.8kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:11<23:13, 29.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<33:11, 20.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<22:05, 29.6kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:13<30:58, 21.1kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:13<20:33, 30.1kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<29:58, 20.6kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<19:49, 29.5kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:15<28:44, 20.3kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:15<18:56, 29.0kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<26:45, 20.5kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<17:34, 29.3kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:17<25:15, 20.3kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:17<16:30, 29.0kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<23:20, 20.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<15:10, 29.3kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:19<21:49, 20.3kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:19<13:17, 29.0kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:21<09:22, 40.0kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:21<06:28, 56.9kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:21<04:11, 81.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<10:29, 32.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<06:36, 46.1kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:23<12:17, 24.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:23<07:37, 35.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<11:58, 22.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:24<07:18, 32.1kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:25<11:01, 21.3kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:25<06:34, 30.3kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<09:29, 21.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:26<05:29, 30.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:27<08:00, 20.5kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:27<03:42, 29.3kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:29<02:21, 40.6kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:29<01:36, 57.6kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:29<00:43, 81.9kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<01:48, 32.8kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:31, 46.7kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:31<00:59, 25.0kB/s].vector_cache/glove.6B.zip: 862MB [06:31, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 727/400000 [00:00<00:54, 7264.20it/s]  0%|          | 1462/400000 [00:00<00:54, 7288.93it/s]  1%|          | 2146/400000 [00:00<00:55, 7145.71it/s]  1%|          | 2812/400000 [00:00<00:56, 6991.41it/s]  1%|          | 3543/400000 [00:00<00:55, 7083.97it/s]  1%|          | 4280/400000 [00:00<00:55, 7166.46it/s]  1%|          | 4982/400000 [00:00<00:55, 7119.80it/s]  1%|▏         | 5717/400000 [00:00<00:54, 7186.12it/s]  2%|▏         | 6478/400000 [00:00<00:53, 7306.14it/s]  2%|▏         | 7214/400000 [00:01<00:53, 7321.57it/s]  2%|▏         | 7964/400000 [00:01<00:53, 7371.68it/s]  2%|▏         | 8734/400000 [00:01<00:52, 7466.54it/s]  2%|▏         | 9496/400000 [00:01<00:51, 7511.36it/s]  3%|▎         | 10241/400000 [00:01<00:53, 7292.93it/s]  3%|▎         | 10972/400000 [00:01<00:53, 7298.04it/s]  3%|▎         | 11716/400000 [00:01<00:52, 7338.60it/s]  3%|▎         | 12449/400000 [00:01<00:53, 7301.85it/s]  3%|▎         | 13189/400000 [00:01<00:52, 7329.03it/s]  3%|▎         | 13947/400000 [00:01<00:52, 7401.56it/s]  4%|▎         | 14687/400000 [00:02<00:52, 7373.08it/s]  4%|▍         | 15440/400000 [00:02<00:51, 7417.21it/s]  4%|▍         | 16214/400000 [00:02<00:51, 7510.64it/s]  4%|▍         | 16966/400000 [00:02<00:53, 7205.87it/s]  4%|▍         | 17690/400000 [00:02<00:54, 7059.23it/s]  5%|▍         | 18399/400000 [00:02<00:55, 6815.36it/s]  5%|▍         | 19085/400000 [00:02<00:56, 6782.12it/s]  5%|▍         | 19815/400000 [00:02<00:54, 6928.91it/s]  5%|▌         | 20563/400000 [00:02<00:53, 7084.07it/s]  5%|▌         | 21310/400000 [00:02<00:52, 7193.38it/s]  6%|▌         | 22032/400000 [00:03<00:52, 7185.77it/s]  6%|▌         | 22753/400000 [00:03<00:54, 6957.80it/s]  6%|▌         | 23452/400000 [00:03<00:55, 6750.50it/s]  6%|▌         | 24157/400000 [00:03<00:54, 6835.55it/s]  6%|▌         | 24907/400000 [00:03<00:53, 7019.60it/s]  6%|▋         | 25632/400000 [00:03<00:52, 7085.31it/s]  7%|▋         | 26369/400000 [00:03<00:52, 7167.49it/s]  7%|▋         | 27088/400000 [00:03<00:52, 7149.44it/s]  7%|▋         | 27825/400000 [00:03<00:51, 7213.87it/s]  7%|▋         | 28548/400000 [00:03<00:51, 7164.24it/s]  7%|▋         | 29268/400000 [00:04<00:51, 7174.76it/s]  8%|▊         | 30027/400000 [00:04<00:50, 7293.94it/s]  8%|▊         | 30764/400000 [00:04<00:50, 7314.68it/s]  8%|▊         | 31497/400000 [00:04<00:51, 7142.15it/s]  8%|▊         | 32213/400000 [00:04<00:51, 7121.87it/s]  8%|▊         | 32932/400000 [00:04<00:51, 7141.99it/s]  8%|▊         | 33675/400000 [00:04<00:50, 7223.79it/s]  9%|▊         | 34417/400000 [00:04<00:50, 7280.42it/s]  9%|▉         | 35146/400000 [00:04<00:52, 6928.22it/s]  9%|▉         | 35843/400000 [00:05<00:54, 6715.56it/s]  9%|▉         | 36519/400000 [00:05<00:54, 6632.24it/s]  9%|▉         | 37265/400000 [00:05<00:52, 6859.26it/s]  9%|▉         | 37993/400000 [00:05<00:51, 6978.34it/s] 10%|▉         | 38749/400000 [00:05<00:50, 7140.36it/s] 10%|▉         | 39467/400000 [00:05<00:53, 6737.70it/s] 10%|█         | 40148/400000 [00:05<00:54, 6654.19it/s] 10%|█         | 40875/400000 [00:05<00:52, 6826.49it/s] 10%|█         | 41563/400000 [00:05<00:53, 6731.95it/s] 11%|█         | 42240/400000 [00:05<00:53, 6641.02it/s] 11%|█         | 42932/400000 [00:06<00:53, 6722.10it/s] 11%|█         | 43644/400000 [00:06<00:52, 6835.15it/s] 11%|█         | 44404/400000 [00:06<00:50, 7047.32it/s] 11%|█▏        | 45112/400000 [00:06<00:50, 7027.44it/s] 11%|█▏        | 45817/400000 [00:06<00:51, 6816.07it/s] 12%|█▏        | 46502/400000 [00:06<00:53, 6659.92it/s] 12%|█▏        | 47171/400000 [00:06<00:54, 6482.19it/s] 12%|█▏        | 47823/400000 [00:06<00:55, 6343.49it/s] 12%|█▏        | 48465/400000 [00:06<00:55, 6364.40it/s] 12%|█▏        | 49166/400000 [00:06<00:53, 6545.08it/s] 12%|█▏        | 49913/400000 [00:07<00:51, 6795.54it/s] 13%|█▎        | 50679/400000 [00:07<00:49, 7033.09it/s] 13%|█▎        | 51388/400000 [00:07<00:50, 6857.05it/s] 13%|█▎        | 52133/400000 [00:07<00:49, 7023.65it/s] 13%|█▎        | 52882/400000 [00:07<00:48, 7154.56it/s] 13%|█▎        | 53638/400000 [00:07<00:47, 7269.42it/s] 14%|█▎        | 54368/400000 [00:07<00:48, 7138.23it/s] 14%|█▍        | 55093/400000 [00:07<00:48, 7169.90it/s] 14%|█▍        | 55845/400000 [00:07<00:47, 7270.95it/s] 14%|█▍        | 56592/400000 [00:08<00:46, 7328.95it/s] 14%|█▍        | 57346/400000 [00:08<00:46, 7390.61it/s] 15%|█▍        | 58087/400000 [00:08<00:47, 7134.44it/s] 15%|█▍        | 58804/400000 [00:08<00:49, 6889.45it/s] 15%|█▍        | 59497/400000 [00:08<00:51, 6666.79it/s] 15%|█▌        | 60168/400000 [00:08<00:51, 6552.28it/s] 15%|█▌        | 60827/400000 [00:08<00:52, 6402.99it/s] 15%|█▌        | 61471/400000 [00:08<00:53, 6360.80it/s] 16%|█▌        | 62117/400000 [00:08<00:52, 6389.30it/s] 16%|█▌        | 62758/400000 [00:08<00:54, 6230.68it/s] 16%|█▌        | 63510/400000 [00:09<00:51, 6568.14it/s] 16%|█▌        | 64234/400000 [00:09<00:49, 6756.09it/s] 16%|█▌        | 64925/400000 [00:09<00:49, 6800.69it/s] 16%|█▋        | 65610/400000 [00:09<00:49, 6783.07it/s] 17%|█▋        | 66347/400000 [00:09<00:48, 6948.93it/s] 17%|█▋        | 67045/400000 [00:09<00:48, 6828.80it/s] 17%|█▋        | 67731/400000 [00:09<00:49, 6723.34it/s] 17%|█▋        | 68430/400000 [00:09<00:48, 6798.58it/s] 17%|█▋        | 69133/400000 [00:09<00:48, 6864.82it/s] 17%|█▋        | 69866/400000 [00:09<00:47, 6997.45it/s] 18%|█▊        | 70568/400000 [00:10<00:47, 6999.85it/s] 18%|█▊        | 71270/400000 [00:10<00:47, 6979.14it/s] 18%|█▊        | 72005/400000 [00:10<00:46, 7084.96it/s] 18%|█▊        | 72733/400000 [00:10<00:45, 7141.99it/s] 18%|█▊        | 73448/400000 [00:10<00:47, 6946.39it/s] 19%|█▊        | 74145/400000 [00:10<00:48, 6710.18it/s] 19%|█▊        | 74819/400000 [00:10<00:50, 6416.37it/s] 19%|█▉        | 75514/400000 [00:10<00:49, 6566.50it/s] 19%|█▉        | 76302/400000 [00:10<00:46, 6909.88it/s] 19%|█▉        | 77027/400000 [00:11<00:46, 7008.37it/s] 19%|█▉        | 77734/400000 [00:11<00:47, 6838.34it/s] 20%|█▉        | 78423/400000 [00:11<00:48, 6644.00it/s] 20%|█▉        | 79093/400000 [00:11<00:49, 6432.60it/s] 20%|█▉        | 79743/400000 [00:11<00:49, 6450.18it/s] 20%|██        | 80433/400000 [00:11<00:48, 6578.63it/s] 20%|██        | 81176/400000 [00:11<00:46, 6811.35it/s] 20%|██        | 81866/400000 [00:11<00:46, 6836.00it/s] 21%|██        | 82567/400000 [00:11<00:46, 6886.58it/s] 21%|██        | 83319/400000 [00:11<00:44, 7063.29it/s] 21%|██        | 84028/400000 [00:12<00:45, 6914.18it/s] 21%|██        | 84769/400000 [00:12<00:44, 7053.99it/s] 21%|██▏       | 85477/400000 [00:12<00:44, 7051.33it/s] 22%|██▏       | 86202/400000 [00:12<00:44, 7108.21it/s] 22%|██▏       | 86915/400000 [00:12<00:44, 6995.41it/s] 22%|██▏       | 87616/400000 [00:12<00:46, 6695.35it/s] 22%|██▏       | 88290/400000 [00:12<00:47, 6598.95it/s] 22%|██▏       | 88953/400000 [00:12<00:47, 6517.56it/s] 22%|██▏       | 89608/400000 [00:12<00:47, 6525.58it/s] 23%|██▎       | 90306/400000 [00:12<00:46, 6655.19it/s] 23%|██▎       | 91010/400000 [00:13<00:45, 6765.60it/s] 23%|██▎       | 91744/400000 [00:13<00:44, 6925.87it/s] 23%|██▎       | 92439/400000 [00:13<00:45, 6793.49it/s] 23%|██▎       | 93121/400000 [00:13<00:45, 6712.78it/s] 23%|██▎       | 93816/400000 [00:13<00:45, 6781.93it/s] 24%|██▎       | 94558/400000 [00:13<00:43, 6960.79it/s] 24%|██▍       | 95257/400000 [00:13<00:44, 6903.26it/s] 24%|██▍       | 95985/400000 [00:13<00:43, 7009.69it/s] 24%|██▍       | 96705/400000 [00:13<00:42, 7063.60it/s] 24%|██▍       | 97461/400000 [00:14<00:41, 7204.78it/s] 25%|██▍       | 98208/400000 [00:14<00:41, 7282.27it/s] 25%|██▍       | 98938/400000 [00:14<00:42, 7049.12it/s] 25%|██▍       | 99646/400000 [00:14<00:44, 6707.26it/s] 25%|██▌       | 100322/400000 [00:14<00:45, 6544.18it/s] 25%|██▌       | 100981/400000 [00:14<00:47, 6264.84it/s] 25%|██▌       | 101658/400000 [00:14<00:46, 6407.69it/s] 26%|██▌       | 102331/400000 [00:14<00:45, 6498.97it/s] 26%|██▌       | 103075/400000 [00:14<00:43, 6754.65it/s] 26%|██▌       | 103833/400000 [00:14<00:42, 6982.09it/s] 26%|██▌       | 104558/400000 [00:15<00:41, 7059.17it/s] 26%|██▋       | 105269/400000 [00:15<00:42, 6978.10it/s] 26%|██▋       | 105970/400000 [00:15<00:43, 6817.39it/s] 27%|██▋       | 106655/400000 [00:15<00:43, 6737.01it/s] 27%|██▋       | 107331/400000 [00:15<00:43, 6653.65it/s] 27%|██▋       | 107999/400000 [00:15<00:44, 6601.27it/s] 27%|██▋       | 108689/400000 [00:15<00:43, 6687.35it/s] 27%|██▋       | 109434/400000 [00:15<00:42, 6897.08it/s] 28%|██▊       | 110174/400000 [00:15<00:41, 7040.17it/s] 28%|██▊       | 110925/400000 [00:15<00:40, 7173.18it/s] 28%|██▊       | 111645/400000 [00:16<00:40, 7127.64it/s] 28%|██▊       | 112360/400000 [00:16<00:41, 6871.05it/s] 28%|██▊       | 113105/400000 [00:16<00:40, 7033.79it/s] 28%|██▊       | 113828/400000 [00:16<00:40, 7089.55it/s] 29%|██▊       | 114602/400000 [00:16<00:39, 7270.70it/s] 29%|██▉       | 115338/400000 [00:16<00:39, 7297.03it/s] 29%|██▉       | 116070/400000 [00:16<00:40, 7086.86it/s] 29%|██▉       | 116782/400000 [00:16<00:41, 6877.13it/s] 29%|██▉       | 117473/400000 [00:16<00:42, 6707.05it/s] 30%|██▉       | 118147/400000 [00:17<00:43, 6496.00it/s] 30%|██▉       | 118801/400000 [00:17<00:44, 6372.05it/s] 30%|██▉       | 119442/400000 [00:17<00:44, 6309.60it/s] 30%|███       | 120076/400000 [00:17<00:45, 6180.00it/s] 30%|███       | 120733/400000 [00:17<00:44, 6291.49it/s] 30%|███       | 121394/400000 [00:17<00:43, 6383.13it/s] 31%|███       | 122057/400000 [00:17<00:43, 6454.50it/s] 31%|███       | 122722/400000 [00:17<00:42, 6509.67it/s] 31%|███       | 123398/400000 [00:17<00:42, 6581.89it/s] 31%|███       | 124070/400000 [00:17<00:41, 6620.60it/s] 31%|███       | 124818/400000 [00:18<00:40, 6856.01it/s] 31%|███▏      | 125588/400000 [00:18<00:38, 7074.29it/s] 32%|███▏      | 126338/400000 [00:18<00:38, 7195.03it/s] 32%|███▏      | 127073/400000 [00:18<00:37, 7240.15it/s] 32%|███▏      | 127813/400000 [00:18<00:37, 7285.33it/s] 32%|███▏      | 128545/400000 [00:18<00:37, 7295.12it/s] 32%|███▏      | 129276/400000 [00:18<00:37, 7182.08it/s] 32%|███▏      | 129996/400000 [00:18<00:38, 6975.54it/s] 33%|███▎      | 130696/400000 [00:18<00:39, 6763.96it/s] 33%|███▎      | 131376/400000 [00:18<00:40, 6711.24it/s] 33%|███▎      | 132050/400000 [00:19<00:40, 6617.68it/s] 33%|███▎      | 132714/400000 [00:19<00:40, 6582.81it/s] 33%|███▎      | 133410/400000 [00:19<00:39, 6690.88it/s] 34%|███▎      | 134081/400000 [00:19<00:41, 6407.98it/s] 34%|███▎      | 134780/400000 [00:19<00:40, 6570.88it/s] 34%|███▍      | 135470/400000 [00:19<00:39, 6664.88it/s] 34%|███▍      | 136165/400000 [00:19<00:39, 6745.98it/s] 34%|███▍      | 136913/400000 [00:19<00:37, 6950.51it/s] 34%|███▍      | 137633/400000 [00:19<00:37, 7022.04it/s] 35%|███▍      | 138338/400000 [00:20<00:38, 6850.24it/s] 35%|███▍      | 139026/400000 [00:20<00:38, 6724.34it/s] 35%|███▍      | 139737/400000 [00:20<00:38, 6834.07it/s] 35%|███▌      | 140474/400000 [00:20<00:37, 6984.20it/s] 35%|███▌      | 141186/400000 [00:20<00:36, 7023.39it/s] 35%|███▌      | 141920/400000 [00:20<00:36, 7114.10it/s] 36%|███▌      | 142653/400000 [00:20<00:35, 7177.44it/s] 36%|███▌      | 143392/400000 [00:20<00:35, 7238.70it/s] 36%|███▌      | 144130/400000 [00:20<00:35, 7280.32it/s] 36%|███▌      | 144859/400000 [00:20<00:35, 7172.96it/s] 36%|███▋      | 145584/400000 [00:21<00:35, 7194.90it/s] 37%|███▋      | 146305/400000 [00:21<00:36, 6963.92it/s] 37%|███▋      | 147046/400000 [00:21<00:35, 7091.59it/s] 37%|███▋      | 147789/400000 [00:21<00:35, 7187.22it/s] 37%|███▋      | 148522/400000 [00:21<00:34, 7229.41it/s] 37%|███▋      | 149247/400000 [00:21<00:34, 7213.77it/s] 38%|███▊      | 150022/400000 [00:21<00:33, 7366.62it/s] 38%|███▊      | 150761/400000 [00:21<00:33, 7354.38it/s] 38%|███▊      | 151544/400000 [00:21<00:33, 7479.52it/s] 38%|███▊      | 152309/400000 [00:21<00:32, 7529.06it/s] 38%|███▊      | 153063/400000 [00:22<00:32, 7487.87it/s] 38%|███▊      | 153840/400000 [00:22<00:32, 7567.87it/s] 39%|███▊      | 154598/400000 [00:22<00:33, 7368.92it/s] 39%|███▉      | 155337/400000 [00:22<00:33, 7261.41it/s] 39%|███▉      | 156088/400000 [00:22<00:33, 7333.81it/s] 39%|███▉      | 156890/400000 [00:22<00:32, 7526.55it/s] 39%|███▉      | 157667/400000 [00:22<00:31, 7595.05it/s] 40%|███▉      | 158429/400000 [00:22<00:31, 7578.99it/s] 40%|███▉      | 159189/400000 [00:22<00:32, 7420.03it/s] 40%|███▉      | 159933/400000 [00:22<00:32, 7380.92it/s] 40%|████      | 160673/400000 [00:23<00:34, 7017.40it/s] 40%|████      | 161380/400000 [00:23<00:35, 6653.14it/s] 41%|████      | 162053/400000 [00:23<00:35, 6611.12it/s] 41%|████      | 162720/400000 [00:23<00:36, 6472.28it/s] 41%|████      | 163372/400000 [00:23<00:37, 6321.49it/s] 41%|████      | 164008/400000 [00:23<00:38, 6148.28it/s] 41%|████      | 164627/400000 [00:23<00:40, 5842.75it/s] 41%|████▏     | 165255/400000 [00:23<00:39, 5966.01it/s] 41%|████▏     | 165857/400000 [00:23<00:39, 5890.81it/s] 42%|████▏     | 166479/400000 [00:24<00:39, 5985.48it/s] 42%|████▏     | 167196/400000 [00:24<00:36, 6297.12it/s] 42%|████▏     | 167914/400000 [00:24<00:35, 6538.31it/s] 42%|████▏     | 168649/400000 [00:24<00:34, 6760.30it/s] 42%|████▏     | 169344/400000 [00:24<00:33, 6814.21it/s] 43%|████▎     | 170034/400000 [00:24<00:33, 6837.46it/s] 43%|████▎     | 170790/400000 [00:24<00:32, 7037.52it/s] 43%|████▎     | 171553/400000 [00:24<00:31, 7203.34it/s] 43%|████▎     | 172318/400000 [00:24<00:31, 7330.63it/s] 43%|████▎     | 173055/400000 [00:24<00:31, 7266.75it/s] 43%|████▎     | 173784/400000 [00:25<00:31, 7252.80it/s] 44%|████▎     | 174511/400000 [00:25<00:31, 7095.94it/s] 44%|████▍     | 175223/400000 [00:25<00:32, 6898.52it/s] 44%|████▍     | 175916/400000 [00:25<00:33, 6755.33it/s] 44%|████▍     | 176594/400000 [00:25<00:33, 6642.24it/s] 44%|████▍     | 177261/400000 [00:25<00:34, 6465.92it/s] 44%|████▍     | 177911/400000 [00:25<00:35, 6316.95it/s] 45%|████▍     | 178568/400000 [00:25<00:34, 6390.28it/s] 45%|████▍     | 179237/400000 [00:25<00:34, 6475.65it/s] 45%|████▍     | 179966/400000 [00:26<00:32, 6699.49it/s] 45%|████▌     | 180667/400000 [00:26<00:32, 6787.53it/s] 45%|████▌     | 181353/400000 [00:26<00:32, 6807.36it/s] 46%|████▌     | 182063/400000 [00:26<00:31, 6890.13it/s] 46%|████▌     | 182782/400000 [00:26<00:31, 6969.57it/s] 46%|████▌     | 183527/400000 [00:26<00:30, 7106.83it/s] 46%|████▌     | 184263/400000 [00:26<00:30, 7179.45it/s] 46%|████▌     | 184996/400000 [00:26<00:29, 7222.22it/s] 46%|████▋     | 185720/400000 [00:26<00:31, 6716.05it/s] 47%|████▋     | 186443/400000 [00:26<00:31, 6859.95it/s] 47%|████▋     | 187161/400000 [00:27<00:30, 6949.58it/s] 47%|████▋     | 187891/400000 [00:27<00:30, 7049.52it/s] 47%|████▋     | 188662/400000 [00:27<00:29, 7234.23it/s] 47%|████▋     | 189417/400000 [00:27<00:28, 7324.36it/s] 48%|████▊     | 190153/400000 [00:27<00:29, 7189.32it/s] 48%|████▊     | 190875/400000 [00:27<00:29, 7168.30it/s] 48%|████▊     | 191641/400000 [00:27<00:28, 7307.03it/s] 48%|████▊     | 192374/400000 [00:27<00:29, 7121.18it/s] 48%|████▊     | 193155/400000 [00:27<00:28, 7312.13it/s] 48%|████▊     | 193890/400000 [00:27<00:28, 7109.52it/s] 49%|████▊     | 194609/400000 [00:28<00:28, 7131.51it/s] 49%|████▉     | 195367/400000 [00:28<00:28, 7258.80it/s] 49%|████▉     | 196136/400000 [00:28<00:27, 7382.10it/s] 49%|████▉     | 196877/400000 [00:28<00:27, 7346.61it/s] 49%|████▉     | 197614/400000 [00:28<00:27, 7230.48it/s] 50%|████▉     | 198361/400000 [00:28<00:27, 7299.62it/s] 50%|████▉     | 199112/400000 [00:28<00:27, 7359.08it/s] 50%|████▉     | 199870/400000 [00:28<00:26, 7423.64it/s] 50%|█████     | 200614/400000 [00:28<00:27, 7284.96it/s] 50%|█████     | 201354/400000 [00:28<00:27, 7316.27it/s] 51%|█████     | 202108/400000 [00:29<00:26, 7380.95it/s] 51%|█████     | 202881/400000 [00:29<00:26, 7481.65it/s] 51%|█████     | 203630/400000 [00:29<00:26, 7394.81it/s] 51%|█████     | 204374/400000 [00:29<00:26, 7405.03it/s] 51%|█████▏    | 205116/400000 [00:29<00:26, 7308.17it/s] 51%|█████▏    | 205848/400000 [00:29<00:26, 7295.61it/s] 52%|█████▏    | 206579/400000 [00:29<00:26, 7286.79it/s] 52%|█████▏    | 207309/400000 [00:29<00:26, 7153.27it/s] 52%|█████▏    | 208047/400000 [00:29<00:26, 7217.27it/s] 52%|█████▏    | 208773/400000 [00:30<00:26, 7227.82it/s] 52%|█████▏    | 209497/400000 [00:30<00:27, 6978.19it/s] 53%|█████▎    | 210198/400000 [00:30<00:27, 6917.58it/s] 53%|█████▎    | 210892/400000 [00:30<00:28, 6726.03it/s] 53%|█████▎    | 211567/400000 [00:30<00:28, 6636.30it/s] 53%|█████▎    | 212233/400000 [00:30<00:28, 6631.08it/s] 53%|█████▎    | 212964/400000 [00:30<00:27, 6819.17it/s] 53%|█████▎    | 213666/400000 [00:30<00:27, 6877.90it/s] 54%|█████▎    | 214383/400000 [00:30<00:26, 6961.23it/s] 54%|█████▍    | 215105/400000 [00:30<00:26, 7032.31it/s] 54%|█████▍    | 215828/400000 [00:31<00:25, 7087.76it/s] 54%|█████▍    | 216589/400000 [00:31<00:25, 7236.03it/s] 54%|█████▍    | 217315/400000 [00:31<00:26, 6977.80it/s] 55%|█████▍    | 218040/400000 [00:31<00:25, 7057.09it/s] 55%|█████▍    | 218774/400000 [00:31<00:25, 7137.80it/s] 55%|█████▍    | 219514/400000 [00:31<00:25, 7212.20it/s] 55%|█████▌    | 220249/400000 [00:31<00:24, 7251.54it/s] 55%|█████▌    | 220983/400000 [00:31<00:24, 7275.34it/s] 55%|█████▌    | 221731/400000 [00:31<00:24, 7334.13it/s] 56%|█████▌    | 222485/400000 [00:31<00:24, 7394.64it/s] 56%|█████▌    | 223226/400000 [00:32<00:24, 7275.35it/s] 56%|█████▌    | 223955/400000 [00:32<00:24, 7222.29it/s] 56%|█████▌    | 224678/400000 [00:32<00:24, 7157.90it/s] 56%|█████▋    | 225421/400000 [00:32<00:24, 7232.76it/s] 57%|█████▋    | 226145/400000 [00:32<00:24, 7033.45it/s] 57%|█████▋    | 226879/400000 [00:32<00:24, 7119.92it/s] 57%|█████▋    | 227593/400000 [00:32<00:24, 7060.56it/s] 57%|█████▋    | 228349/400000 [00:32<00:23, 7203.05it/s] 57%|█████▋    | 229143/400000 [00:32<00:23, 7408.79it/s] 57%|█████▋    | 229887/400000 [00:32<00:23, 7386.72it/s] 58%|█████▊    | 230629/400000 [00:33<00:22, 7394.81it/s] 58%|█████▊    | 231405/400000 [00:33<00:22, 7500.02it/s] 58%|█████▊    | 232157/400000 [00:33<00:22, 7347.31it/s] 58%|█████▊    | 232894/400000 [00:33<00:22, 7280.01it/s] 58%|█████▊    | 233624/400000 [00:33<00:22, 7275.64it/s] 59%|█████▊    | 234370/400000 [00:33<00:22, 7328.49it/s] 59%|█████▉    | 235105/400000 [00:33<00:22, 7334.79it/s] 59%|█████▉    | 235908/400000 [00:33<00:21, 7528.36it/s] 59%|█████▉    | 236707/400000 [00:33<00:21, 7659.64it/s] 59%|█████▉    | 237505/400000 [00:33<00:20, 7752.79it/s] 60%|█████▉    | 238299/400000 [00:34<00:20, 7806.03it/s] 60%|█████▉    | 239081/400000 [00:34<00:20, 7704.75it/s] 60%|█████▉    | 239853/400000 [00:34<00:21, 7533.24it/s] 60%|██████    | 240608/400000 [00:34<00:21, 7536.94it/s] 60%|██████    | 241363/400000 [00:34<00:21, 7499.18it/s] 61%|██████    | 242114/400000 [00:34<00:21, 7475.83it/s] 61%|██████    | 242883/400000 [00:34<00:20, 7537.22it/s] 61%|██████    | 243654/400000 [00:34<00:20, 7587.21it/s] 61%|██████    | 244414/400000 [00:34<00:20, 7475.69it/s] 61%|██████▏   | 245163/400000 [00:35<00:21, 7155.54it/s] 61%|██████▏   | 245882/400000 [00:35<00:22, 6890.75it/s] 62%|██████▏   | 246576/400000 [00:35<00:22, 6729.40it/s] 62%|██████▏   | 247253/400000 [00:35<00:23, 6605.20it/s] 62%|██████▏   | 247917/400000 [00:35<00:23, 6490.67it/s] 62%|██████▏   | 248623/400000 [00:35<00:22, 6649.51it/s] 62%|██████▏   | 249291/400000 [00:35<00:22, 6555.25it/s] 62%|██████▏   | 249949/400000 [00:35<00:22, 6541.37it/s] 63%|██████▎   | 250609/400000 [00:35<00:22, 6558.76it/s] 63%|██████▎   | 251267/400000 [00:35<00:23, 6410.67it/s] 63%|██████▎   | 251917/400000 [00:36<00:23, 6436.05it/s] 63%|██████▎   | 252648/400000 [00:36<00:22, 6673.36it/s] 63%|██████▎   | 253319/400000 [00:36<00:22, 6484.32it/s] 63%|██████▎   | 253985/400000 [00:36<00:22, 6535.19it/s] 64%|██████▎   | 254648/400000 [00:36<00:22, 6561.13it/s] 64%|██████▍   | 255308/400000 [00:36<00:22, 6572.21it/s] 64%|██████▍   | 255967/400000 [00:36<00:22, 6440.17it/s] 64%|██████▍   | 256677/400000 [00:36<00:21, 6623.97it/s] 64%|██████▍   | 257352/400000 [00:36<00:21, 6660.66it/s] 65%|██████▍   | 258100/400000 [00:37<00:20, 6885.10it/s] 65%|██████▍   | 258834/400000 [00:37<00:20, 7014.77it/s] 65%|██████▍   | 259539/400000 [00:37<00:20, 6811.55it/s] 65%|██████▌   | 260224/400000 [00:37<00:20, 6670.12it/s] 65%|██████▌   | 260910/400000 [00:37<00:20, 6724.09it/s] 65%|██████▌   | 261656/400000 [00:37<00:19, 6927.68it/s] 66%|██████▌   | 262413/400000 [00:37<00:19, 7108.40it/s] 66%|██████▌   | 263128/400000 [00:37<00:19, 6878.33it/s] 66%|██████▌   | 263820/400000 [00:37<00:20, 6688.10it/s] 66%|██████▌   | 264518/400000 [00:37<00:20, 6772.86it/s] 66%|██████▋   | 265199/400000 [00:38<00:20, 6574.11it/s] 66%|██████▋   | 265876/400000 [00:38<00:20, 6630.06it/s] 67%|██████▋   | 266543/400000 [00:38<00:20, 6639.91it/s] 67%|██████▋   | 267213/400000 [00:38<00:19, 6657.66it/s] 67%|██████▋   | 267928/400000 [00:38<00:19, 6797.24it/s] 67%|██████▋   | 268611/400000 [00:38<00:19, 6806.12it/s] 67%|██████▋   | 269344/400000 [00:38<00:18, 6954.22it/s] 68%|██████▊   | 270060/400000 [00:38<00:18, 7012.33it/s] 68%|██████▊   | 270806/400000 [00:38<00:18, 7138.75it/s] 68%|██████▊   | 271544/400000 [00:38<00:17, 7209.35it/s] 68%|██████▊   | 272267/400000 [00:39<00:17, 7203.67it/s] 68%|██████▊   | 272989/400000 [00:39<00:17, 7168.41it/s] 68%|██████▊   | 273707/400000 [00:39<00:17, 7104.51it/s] 69%|██████▊   | 274419/400000 [00:39<00:17, 7087.84it/s] 69%|██████▉   | 275164/400000 [00:39<00:17, 7191.42it/s] 69%|██████▉   | 275884/400000 [00:39<00:17, 7141.24it/s] 69%|██████▉   | 276635/400000 [00:39<00:17, 7246.76it/s] 69%|██████▉   | 277361/400000 [00:39<00:17, 6983.09it/s] 70%|██████▉   | 278073/400000 [00:39<00:17, 7022.03it/s] 70%|██████▉   | 278782/400000 [00:39<00:17, 7039.88it/s] 70%|██████▉   | 279492/400000 [00:40<00:17, 7055.17it/s] 70%|███████   | 280199/400000 [00:40<00:17, 6699.01it/s] 70%|███████   | 280904/400000 [00:40<00:17, 6799.04it/s] 70%|███████   | 281644/400000 [00:40<00:16, 6967.29it/s] 71%|███████   | 282400/400000 [00:40<00:16, 7134.12it/s] 71%|███████   | 283150/400000 [00:40<00:16, 7239.29it/s] 71%|███████   | 283882/400000 [00:40<00:15, 7260.73it/s] 71%|███████   | 284616/400000 [00:40<00:15, 7282.03it/s] 71%|███████▏  | 285368/400000 [00:40<00:15, 7350.26it/s] 72%|███████▏  | 286131/400000 [00:40<00:15, 7429.77it/s] 72%|███████▏  | 286885/400000 [00:41<00:15, 7460.81it/s] 72%|███████▏  | 287632/400000 [00:41<00:16, 7017.05it/s] 72%|███████▏  | 288340/400000 [00:41<00:16, 6732.00it/s] 72%|███████▏  | 289020/400000 [00:41<00:16, 6658.81it/s] 72%|███████▏  | 289691/400000 [00:41<00:16, 6612.63it/s] 73%|███████▎  | 290368/400000 [00:41<00:16, 6657.21it/s] 73%|███████▎  | 291037/400000 [00:41<00:16, 6639.85it/s] 73%|███████▎  | 291735/400000 [00:41<00:16, 6737.80it/s] 73%|███████▎  | 292411/400000 [00:41<00:16, 6669.69it/s] 73%|███████▎  | 293115/400000 [00:42<00:15, 6775.50it/s] 73%|███████▎  | 293824/400000 [00:42<00:15, 6866.45it/s] 74%|███████▎  | 294512/400000 [00:42<00:15, 6862.23it/s] 74%|███████▍  | 295253/400000 [00:42<00:14, 7016.91it/s] 74%|███████▍  | 295957/400000 [00:42<00:14, 6996.85it/s] 74%|███████▍  | 296676/400000 [00:42<00:14, 7049.61it/s] 74%|███████▍  | 297424/400000 [00:42<00:14, 7173.13it/s] 75%|███████▍  | 298143/400000 [00:42<00:14, 7079.77it/s] 75%|███████▍  | 298897/400000 [00:42<00:14, 7211.35it/s] 75%|███████▍  | 299671/400000 [00:42<00:13, 7361.59it/s] 75%|███████▌  | 300409/400000 [00:43<00:13, 7354.73it/s] 75%|███████▌  | 301146/400000 [00:43<00:13, 7332.86it/s] 75%|███████▌  | 301881/400000 [00:43<00:13, 7258.64it/s] 76%|███████▌  | 302608/400000 [00:43<00:13, 7082.10it/s] 76%|███████▌  | 303342/400000 [00:43<00:13, 7156.93it/s] 76%|███████▌  | 304059/400000 [00:43<00:13, 7119.03it/s] 76%|███████▌  | 304772/400000 [00:43<00:13, 7085.11it/s] 76%|███████▋  | 305482/400000 [00:43<00:13, 7053.88it/s] 77%|███████▋  | 306219/400000 [00:43<00:13, 7144.52it/s] 77%|███████▋  | 306935/400000 [00:43<00:13, 7114.05it/s] 77%|███████▋  | 307665/400000 [00:44<00:12, 7168.15it/s] 77%|███████▋  | 308394/400000 [00:44<00:12, 7195.63it/s] 77%|███████▋  | 309114/400000 [00:44<00:12, 7174.80it/s] 77%|███████▋  | 309832/400000 [00:44<00:12, 7074.79it/s] 78%|███████▊  | 310569/400000 [00:44<00:12, 7159.64it/s] 78%|███████▊  | 311286/400000 [00:44<00:12, 7114.26it/s] 78%|███████▊  | 312039/400000 [00:44<00:12, 7232.18it/s] 78%|███████▊  | 312791/400000 [00:44<00:11, 7314.78it/s] 78%|███████▊  | 313540/400000 [00:44<00:11, 7364.84it/s] 79%|███████▊  | 314328/400000 [00:44<00:11, 7511.80it/s] 79%|███████▉  | 315084/400000 [00:45<00:11, 7523.54it/s] 79%|███████▉  | 315838/400000 [00:45<00:11, 7447.89it/s] 79%|███████▉  | 316584/400000 [00:45<00:11, 7366.00it/s] 79%|███████▉  | 317322/400000 [00:45<00:11, 7220.17it/s] 80%|███████▉  | 318046/400000 [00:45<00:11, 7079.51it/s] 80%|███████▉  | 318756/400000 [00:45<00:11, 7065.61it/s] 80%|███████▉  | 319464/400000 [00:45<00:11, 6821.48it/s] 80%|████████  | 320149/400000 [00:45<00:11, 6802.65it/s] 80%|████████  | 320843/400000 [00:45<00:11, 6841.70it/s] 80%|████████  | 321586/400000 [00:46<00:11, 7007.63it/s] 81%|████████  | 322346/400000 [00:46<00:10, 7174.09it/s] 81%|████████  | 323066/400000 [00:46<00:11, 6989.88it/s] 81%|████████  | 323768/400000 [00:46<00:10, 6978.21it/s] 81%|████████  | 324468/400000 [00:46<00:11, 6752.68it/s] 81%|████████▏ | 325169/400000 [00:46<00:10, 6827.75it/s] 81%|████████▏ | 325923/400000 [00:46<00:10, 7024.62it/s] 82%|████████▏ | 326680/400000 [00:46<00:10, 7178.60it/s] 82%|████████▏ | 327421/400000 [00:46<00:10, 7246.06it/s] 82%|████████▏ | 328161/400000 [00:46<00:09, 7291.37it/s] 82%|████████▏ | 328913/400000 [00:47<00:09, 7357.24it/s] 82%|████████▏ | 329656/400000 [00:47<00:09, 7374.88it/s] 83%|████████▎ | 330398/400000 [00:47<00:09, 7385.59it/s] 83%|████████▎ | 331138/400000 [00:47<00:09, 7295.20it/s] 83%|████████▎ | 331869/400000 [00:47<00:09, 7159.09it/s] 83%|████████▎ | 332610/400000 [00:47<00:09, 7230.75it/s] 83%|████████▎ | 333371/400000 [00:47<00:09, 7340.12it/s] 84%|████████▎ | 334121/400000 [00:47<00:08, 7384.99it/s] 84%|████████▎ | 334861/400000 [00:47<00:09, 7023.32it/s] 84%|████████▍ | 335568/400000 [00:47<00:09, 6831.71it/s] 84%|████████▍ | 336256/400000 [00:48<00:09, 6808.09it/s] 84%|████████▍ | 336960/400000 [00:48<00:09, 6875.79it/s] 84%|████████▍ | 337694/400000 [00:48<00:08, 7008.49it/s] 85%|████████▍ | 338410/400000 [00:48<00:08, 7050.76it/s] 85%|████████▍ | 339160/400000 [00:48<00:08, 7179.54it/s] 85%|████████▍ | 339900/400000 [00:48<00:08, 7243.94it/s] 85%|████████▌ | 340626/400000 [00:48<00:08, 6886.95it/s] 85%|████████▌ | 341386/400000 [00:48<00:08, 7085.28it/s] 86%|████████▌ | 342121/400000 [00:48<00:08, 7162.20it/s] 86%|████████▌ | 342873/400000 [00:48<00:07, 7265.60it/s] 86%|████████▌ | 343604/400000 [00:49<00:07, 7278.55it/s] 86%|████████▌ | 344368/400000 [00:49<00:07, 7381.99it/s] 86%|████████▋ | 345108/400000 [00:49<00:07, 7306.29it/s] 86%|████████▋ | 345841/400000 [00:49<00:07, 7290.63it/s] 87%|████████▋ | 346572/400000 [00:49<00:07, 7077.65it/s] 87%|████████▋ | 347282/400000 [00:49<00:07, 7009.42it/s] 87%|████████▋ | 348022/400000 [00:49<00:07, 7120.59it/s] 87%|████████▋ | 348748/400000 [00:49<00:07, 7161.40it/s] 87%|████████▋ | 349466/400000 [00:49<00:07, 7165.07it/s] 88%|████████▊ | 350184/400000 [00:50<00:07, 7074.58it/s] 88%|████████▊ | 350893/400000 [00:50<00:07, 6914.81it/s] 88%|████████▊ | 351629/400000 [00:50<00:06, 7041.16it/s] 88%|████████▊ | 352335/400000 [00:50<00:06, 6951.24it/s] 88%|████████▊ | 353032/400000 [00:50<00:06, 6950.92it/s] 88%|████████▊ | 353753/400000 [00:50<00:06, 7025.07it/s] 89%|████████▊ | 354491/400000 [00:50<00:06, 7125.52it/s] 89%|████████▉ | 355221/400000 [00:50<00:06, 7173.25it/s] 89%|████████▉ | 355940/400000 [00:50<00:06, 7178.13it/s] 89%|████████▉ | 356660/400000 [00:50<00:06, 7183.76it/s] 89%|████████▉ | 357387/400000 [00:51<00:05, 7207.19it/s] 90%|████████▉ | 358108/400000 [00:51<00:05, 7110.87it/s] 90%|████████▉ | 358820/400000 [00:51<00:06, 6726.68it/s] 90%|████████▉ | 359498/400000 [00:51<00:06, 6600.64it/s] 90%|█████████ | 360162/400000 [00:51<00:06, 6599.16it/s] 90%|█████████ | 360853/400000 [00:51<00:05, 6671.03it/s] 90%|█████████ | 361606/400000 [00:51<00:05, 6906.21it/s] 91%|█████████ | 362328/400000 [00:51<00:05, 6996.70it/s] 91%|█████████ | 363070/400000 [00:51<00:05, 7116.99it/s] 91%|█████████ | 363811/400000 [00:51<00:05, 7201.35it/s] 91%|█████████ | 364556/400000 [00:52<00:04, 7273.12it/s] 91%|█████████▏| 365288/400000 [00:52<00:04, 7285.02it/s] 92%|█████████▏| 366018/400000 [00:52<00:04, 7159.60it/s] 92%|█████████▏| 366747/400000 [00:52<00:04, 7195.79it/s] 92%|█████████▏| 367468/400000 [00:52<00:04, 7069.17it/s] 92%|█████████▏| 368202/400000 [00:52<00:04, 7145.83it/s] 92%|█████████▏| 368918/400000 [00:52<00:04, 7032.17it/s] 92%|█████████▏| 369633/400000 [00:52<00:04, 7066.75it/s] 93%|█████████▎| 370341/400000 [00:52<00:04, 6999.83it/s] 93%|█████████▎| 371078/400000 [00:52<00:04, 7105.57it/s] 93%|█████████▎| 371817/400000 [00:53<00:03, 7187.90it/s] 93%|█████████▎| 372558/400000 [00:53<00:03, 7251.87it/s] 93%|█████████▎| 373301/400000 [00:53<00:03, 7303.24it/s] 94%|█████████▎| 374034/400000 [00:53<00:03, 7308.46it/s] 94%|█████████▎| 374766/400000 [00:53<00:03, 6964.59it/s] 94%|█████████▍| 375467/400000 [00:53<00:03, 6800.38it/s] 94%|█████████▍| 376240/400000 [00:53<00:03, 7053.56it/s] 94%|█████████▍| 376981/400000 [00:53<00:03, 7156.63it/s] 94%|█████████▍| 377740/400000 [00:53<00:03, 7279.38it/s] 95%|█████████▍| 378497/400000 [00:54<00:02, 7362.58it/s] 95%|█████████▍| 379236/400000 [00:54<00:02, 7301.07it/s] 95%|█████████▍| 379968/400000 [00:54<00:02, 7213.09it/s] 95%|█████████▌| 380725/400000 [00:54<00:02, 7314.87it/s] 95%|█████████▌| 381464/400000 [00:54<00:02, 7334.41it/s] 96%|█████████▌| 382199/400000 [00:54<00:02, 6928.61it/s] 96%|█████████▌| 382898/400000 [00:54<00:02, 6884.26it/s] 96%|█████████▌| 383616/400000 [00:54<00:02, 6968.12it/s] 96%|█████████▌| 384351/400000 [00:54<00:02, 7076.46it/s] 96%|█████████▋| 385072/400000 [00:54<00:02, 7115.69it/s] 96%|█████████▋| 385874/400000 [00:55<00:01, 7364.04it/s] 97%|█████████▋| 386668/400000 [00:55<00:01, 7525.80it/s] 97%|█████████▋| 387424/400000 [00:55<00:01, 7349.95it/s] 97%|█████████▋| 388217/400000 [00:55<00:01, 7513.48it/s] 97%|█████████▋| 388987/400000 [00:55<00:01, 7566.17it/s] 97%|█████████▋| 389791/400000 [00:55<00:01, 7700.06it/s] 98%|█████████▊| 390569/400000 [00:55<00:01, 7722.38it/s] 98%|█████████▊| 391359/400000 [00:55<00:01, 7774.00it/s] 98%|█████████▊| 392144/400000 [00:55<00:01, 7796.35it/s] 98%|█████████▊| 392925/400000 [00:55<00:00, 7790.65it/s] 98%|█████████▊| 393712/400000 [00:56<00:00, 7802.22it/s] 99%|█████████▊| 394493/400000 [00:56<00:00, 7630.58it/s] 99%|█████████▉| 395258/400000 [00:56<00:00, 7520.37it/s] 99%|█████████▉| 396012/400000 [00:56<00:00, 7475.66it/s] 99%|█████████▉| 396761/400000 [00:56<00:00, 7114.49it/s] 99%|█████████▉| 397477/400000 [00:56<00:00, 6941.54it/s]100%|█████████▉| 398176/400000 [00:56<00:00, 6784.02it/s]100%|█████████▉| 398858/400000 [00:56<00:00, 6698.59it/s]100%|█████████▉| 399552/400000 [00:56<00:00, 6768.14it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7020.81it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc7fd937940> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01131927327536718 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011377460382455169 	 Accuracy: 49

  model saves at 49% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15967 out of table with 15966 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15967 out of table with 15966 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-16 02:26:06.853498: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 02:26:06.858056: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-16 02:26:06.858196: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56374a770ae0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 02:26:06.858213: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc8094a8fd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6206 - accuracy: 0.5030
 2000/25000 [=>............................] - ETA: 10s - loss: 7.5746 - accuracy: 0.5060
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5286 - accuracy: 0.5090 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5938 - accuracy: 0.5048
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6728 - accuracy: 0.4996
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7126 - accuracy: 0.4970
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6097 - accuracy: 0.5037
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6072 - accuracy: 0.5039
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5900 - accuracy: 0.5050
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6145 - accuracy: 0.5034
11000/25000 [============>.................] - ETA: 4s - loss: 7.6304 - accuracy: 0.5024
12000/25000 [=============>................] - ETA: 4s - loss: 7.6411 - accuracy: 0.5017
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6407 - accuracy: 0.5017
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
15000/25000 [=================>............] - ETA: 3s - loss: 7.6441 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6455 - accuracy: 0.5014
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6585 - accuracy: 0.5005
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6573 - accuracy: 0.5006
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6674 - accuracy: 0.4999
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6835 - accuracy: 0.4989
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6739 - accuracy: 0.4995
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6799 - accuracy: 0.4991
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6813 - accuracy: 0.4990
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6730 - accuracy: 0.4996
25000/25000 [==============================] - 10s 390us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fc779911ef0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fc75d870f60> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4384 - crf_viterbi_accuracy: 0.1600 - val_loss: 1.3358 - val_crf_viterbi_accuracy: 0.1867

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

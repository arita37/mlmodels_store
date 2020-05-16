
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7ff09d62df28> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-16 03:14:07.950120
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-16 03:14:07.953949
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-16 03:14:07.957043
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-16 03:14:07.960145
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7ff0a93f7438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352182.7188
Epoch 2/10

1/1 [==============================] - 0s 107ms/step - loss: 243687.2812
Epoch 3/10

1/1 [==============================] - 0s 103ms/step - loss: 137006.2500
Epoch 4/10

1/1 [==============================] - 0s 93ms/step - loss: 67069.8359
Epoch 5/10

1/1 [==============================] - 0s 94ms/step - loss: 34212.4180
Epoch 6/10

1/1 [==============================] - 0s 95ms/step - loss: 19342.8066
Epoch 7/10

1/1 [==============================] - 0s 99ms/step - loss: 12117.0625
Epoch 8/10

1/1 [==============================] - 0s 97ms/step - loss: 8283.8086
Epoch 9/10

1/1 [==============================] - 0s 100ms/step - loss: 6073.4307
Epoch 10/10

1/1 [==============================] - 0s 100ms/step - loss: 4721.6387

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.05488688 -1.3391075   0.6220261  -0.76536596 -0.9028383  -1.8889699
   0.9943508  -1.3086432  -1.1912898   0.37933287 -0.60268426  0.5787569
   0.19187206 -0.3198494  -0.29814547  0.6072407   1.0431776   0.41404337
  -0.7315286  -1.6329201   0.40344104 -0.9011258  -0.9630568   0.25898644
   0.35170072 -0.60673475 -0.7264832   1.8072009  -1.0462286  -1.0876276
   1.814932    0.7318737   0.47273406  0.25313085  1.4956245  -1.5868379
  -2.1293328   1.9814881   0.20680523  0.15362817 -2.4232497  -0.90573835
   0.83602023  0.03941882  0.05341465  1.7627797  -2.9045653  -1.4344525
   0.35278308  0.38833514  0.96826094  0.34118998  0.53188896 -1.7521033
   1.5181895   0.75686574 -0.370277    1.122678   -0.33781248 -0.19502658
  -0.43105045 11.378111    9.558251    9.133248    8.124091   10.09978
  10.857276   11.322348   10.074376    9.925037   10.235458   10.238582
   9.683413    9.861658   11.052914    9.158547   10.679146    9.662427
   9.032086    9.94426    13.024384   11.362994    9.896378   11.400418
   8.726619    9.662847   10.041997   11.163236   10.018409    8.787823
  10.543313   11.966724    9.805159    8.458659   10.296293    8.718993
  10.364289    9.348053    9.104461   10.017185    9.609412    8.761024
   9.782296   10.21694    11.676038   10.057514   10.616713    9.64638
  12.0421505  10.717193   10.987307    8.695366   10.739832   10.888479
   9.392458    7.724158   11.586829    9.492951   10.410232    9.054919
   1.4928073   1.3192055  -0.3598219  -2.5458064   0.9350376  -0.93143106
   0.25589666 -0.3527098   1.3781339   1.2165604  -0.9708697   0.7528995
  -0.2376911  -0.60847366  0.30889374 -2.4655142  -0.86625934 -0.42068172
  -1.2146412  -2.5267222  -0.45610857  1.6617204  -0.25982085  0.01869828
   0.4127645   0.9192656  -0.49027202 -0.8988857  -0.14749299  2.3860445
  -1.0348494  -1.1369965  -1.4603913   0.8766029   2.1031623  -0.5796423
   0.791617    0.15113696  0.777539    2.812381   -0.7442761   0.10122851
   0.879159    0.639148    1.0409495   1.5412928  -0.03864309 -0.08076078
  -1.7299868  -0.41167822  0.5828351  -0.8465115  -1.4600048  -1.6407115
   0.88056934 -1.7597581  -1.1802981   0.23251247 -0.3746097  -0.02546376
   2.0967927   2.6611862   1.0014462   1.3380113   1.46175     0.77109945
   1.7395768   1.9334263   0.65855277  0.3335355   0.3951475   2.5029335
   0.19643778  2.403812    2.1329675   0.48668104  2.6957636   0.8971473
   2.892445    1.6143668   0.2702664   0.8153125   1.75027     1.5016501
   0.91613483  3.2103982   0.08042514  1.8265796   0.13651979  0.36853933
   0.14595062  0.11418527  1.6035566   3.6781793   2.8670006   2.4964795
   1.6292844   1.673657    1.86622     0.6234108   0.8953945   3.0056615
   1.1856918   2.1201873   0.8704773   0.3546089   1.5089645   0.6276452
   1.8195008   1.0010788   0.96359444  0.19267792  0.41213173  0.23922229
   0.72510695  1.5584393   1.3709378   2.6808672   1.048275    1.1216056
   0.68154013 10.8809      9.867015   11.543428    8.005051   11.330741
   8.383604    9.547167   11.417428   10.842671    8.8601055   9.476228
   9.72118     8.633955    8.047986   12.038047    9.838467    9.916359
  10.907546   10.994544   10.489984    9.559067    9.749144    9.738915
  11.284936   10.795359    9.583323   11.29359     9.977997    8.996104
  10.632126   11.112535   10.463227   11.566115    9.477123    8.760928
  10.94159     9.161169   10.864851    9.613228   10.251767   10.295398
  11.293149   10.736195    9.579175   10.21594     9.542999    7.5609374
   8.190672    9.7329855  12.314019    8.3125725   7.8199697   9.10005
  10.545458    8.779735   11.045131    9.889095   10.83612    10.6967325
   2.1713095   3.243318    0.18433607  0.5533933   1.1673899   0.66500604
   0.59589577  1.5482215   1.8155773   1.0039672   2.0201325   0.10069865
   0.550136    0.06613141  2.7920666   0.15217209  0.27112865  0.9512087
   1.4278885   0.09034461  1.0039494   2.5454135   0.7341881   0.8607985
   0.29461604  1.1346923   1.2623734   3.0043325   0.6741085   0.9395936
   0.31033212  0.26650476  0.2248922   1.9485365   2.5584016   1.7796209
   3.6771388   1.203078    1.2617227   1.0024976   0.17046845  0.12620091
   1.607055    0.16809952  0.9011948   0.31907696  2.727248    0.20136708
   0.96232736  3.3836002   1.6595724   2.6831656   1.7397798   0.95004433
   0.58558553  0.9969864   1.9704039   2.1258678   1.5949202   1.7856437
  -7.5265665   7.088711   -7.667123  ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-16 03:14:18.455932
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.0584
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-16 03:14:18.460462
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8499.83
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-16 03:14:18.464405
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    92.392
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-16 03:14:18.469601
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -760.22
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140671066998152
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140668537881096
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140668537881600
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140668537882104
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140668537882608
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140668537883112

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7ff096dc3588> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.524018
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.500134
grad_step = 000002, loss = 0.478980
grad_step = 000003, loss = 0.457332
grad_step = 000004, loss = 0.434978
grad_step = 000005, loss = 0.408352
grad_step = 000006, loss = 0.385332
grad_step = 000007, loss = 0.369851
grad_step = 000008, loss = 0.355622
grad_step = 000009, loss = 0.336279
grad_step = 000010, loss = 0.317166
grad_step = 000011, loss = 0.297249
grad_step = 000012, loss = 0.279434
grad_step = 000013, loss = 0.264405
grad_step = 000014, loss = 0.253072
grad_step = 000015, loss = 0.245849
grad_step = 000016, loss = 0.237564
grad_step = 000017, loss = 0.226828
grad_step = 000018, loss = 0.215088
grad_step = 000019, loss = 0.203241
grad_step = 000020, loss = 0.192665
grad_step = 000021, loss = 0.182959
grad_step = 000022, loss = 0.173114
grad_step = 000023, loss = 0.164125
grad_step = 000024, loss = 0.155648
grad_step = 000025, loss = 0.147251
grad_step = 000026, loss = 0.138897
grad_step = 000027, loss = 0.130352
grad_step = 000028, loss = 0.122208
grad_step = 000029, loss = 0.115191
grad_step = 000030, loss = 0.108791
grad_step = 000031, loss = 0.102309
grad_step = 000032, loss = 0.095717
grad_step = 000033, loss = 0.089171
grad_step = 000034, loss = 0.083069
grad_step = 000035, loss = 0.077604
grad_step = 000036, loss = 0.072481
grad_step = 000037, loss = 0.067519
grad_step = 000038, loss = 0.062718
grad_step = 000039, loss = 0.058162
grad_step = 000040, loss = 0.053924
grad_step = 000041, loss = 0.049977
grad_step = 000042, loss = 0.046304
grad_step = 000043, loss = 0.042832
grad_step = 000044, loss = 0.039531
grad_step = 000045, loss = 0.036313
grad_step = 000046, loss = 0.033188
grad_step = 000047, loss = 0.030401
grad_step = 000048, loss = 0.027932
grad_step = 000049, loss = 0.025535
grad_step = 000050, loss = 0.023259
grad_step = 000051, loss = 0.021189
grad_step = 000052, loss = 0.019266
grad_step = 000053, loss = 0.017443
grad_step = 000054, loss = 0.015754
grad_step = 000055, loss = 0.014208
grad_step = 000056, loss = 0.012815
grad_step = 000057, loss = 0.011567
grad_step = 000058, loss = 0.010438
grad_step = 000059, loss = 0.009436
grad_step = 000060, loss = 0.008519
grad_step = 000061, loss = 0.007680
grad_step = 000062, loss = 0.006945
grad_step = 000063, loss = 0.006311
grad_step = 000064, loss = 0.005751
grad_step = 000065, loss = 0.005250
grad_step = 000066, loss = 0.004823
grad_step = 000067, loss = 0.004453
grad_step = 000068, loss = 0.004108
grad_step = 000069, loss = 0.003817
grad_step = 000070, loss = 0.003581
grad_step = 000071, loss = 0.003367
grad_step = 000072, loss = 0.003174
grad_step = 000073, loss = 0.003004
grad_step = 000074, loss = 0.002862
grad_step = 000075, loss = 0.002744
grad_step = 000076, loss = 0.002643
grad_step = 000077, loss = 0.002557
grad_step = 000078, loss = 0.002474
grad_step = 000079, loss = 0.002408
grad_step = 000080, loss = 0.002356
grad_step = 000081, loss = 0.002308
grad_step = 000082, loss = 0.002268
grad_step = 000083, loss = 0.002239
grad_step = 000084, loss = 0.002216
grad_step = 000085, loss = 0.002193
grad_step = 000086, loss = 0.002174
grad_step = 000087, loss = 0.002157
grad_step = 000088, loss = 0.002143
grad_step = 000089, loss = 0.002134
grad_step = 000090, loss = 0.002125
grad_step = 000091, loss = 0.002114
grad_step = 000092, loss = 0.002104
grad_step = 000093, loss = 0.002095
grad_step = 000094, loss = 0.002084
grad_step = 000095, loss = 0.002074
grad_step = 000096, loss = 0.002063
grad_step = 000097, loss = 0.002052
grad_step = 000098, loss = 0.002041
grad_step = 000099, loss = 0.002029
grad_step = 000100, loss = 0.002019
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002014
grad_step = 000102, loss = 0.002031
grad_step = 000103, loss = 0.002046
grad_step = 000104, loss = 0.002029
grad_step = 000105, loss = 0.001966
grad_step = 000106, loss = 0.001951
grad_step = 000107, loss = 0.001977
grad_step = 000108, loss = 0.001968
grad_step = 000109, loss = 0.001926
grad_step = 000110, loss = 0.001900
grad_step = 000111, loss = 0.001912
grad_step = 000112, loss = 0.001926
grad_step = 000113, loss = 0.001897
grad_step = 000114, loss = 0.001863
grad_step = 000115, loss = 0.001854
grad_step = 000116, loss = 0.001865
grad_step = 000117, loss = 0.001867
grad_step = 000118, loss = 0.001845
grad_step = 000119, loss = 0.001820
grad_step = 000120, loss = 0.001809
grad_step = 000121, loss = 0.001812
grad_step = 000122, loss = 0.001818
grad_step = 000123, loss = 0.001816
grad_step = 000124, loss = 0.001808
grad_step = 000125, loss = 0.001788
grad_step = 000126, loss = 0.001770
grad_step = 000127, loss = 0.001756
grad_step = 000128, loss = 0.001750
grad_step = 000129, loss = 0.001749
grad_step = 000130, loss = 0.001752
grad_step = 000131, loss = 0.001760
grad_step = 000132, loss = 0.001768
grad_step = 000133, loss = 0.001784
grad_step = 000134, loss = 0.001772
grad_step = 000135, loss = 0.001746
grad_step = 000136, loss = 0.001703
grad_step = 000137, loss = 0.001685
grad_step = 000138, loss = 0.001695
grad_step = 000139, loss = 0.001707
grad_step = 000140, loss = 0.001705
grad_step = 000141, loss = 0.001680
grad_step = 000142, loss = 0.001655
grad_step = 000143, loss = 0.001640
grad_step = 000144, loss = 0.001640
grad_step = 000145, loss = 0.001651
grad_step = 000146, loss = 0.001669
grad_step = 000147, loss = 0.001701
grad_step = 000148, loss = 0.001713
grad_step = 000149, loss = 0.001708
grad_step = 000150, loss = 0.001648
grad_step = 000151, loss = 0.001598
grad_step = 000152, loss = 0.001595
grad_step = 000153, loss = 0.001624
grad_step = 000154, loss = 0.001651
grad_step = 000155, loss = 0.001637
grad_step = 000156, loss = 0.001602
grad_step = 000157, loss = 0.001568
grad_step = 000158, loss = 0.001565
grad_step = 000159, loss = 0.001587
grad_step = 000160, loss = 0.001607
grad_step = 000161, loss = 0.001618
grad_step = 000162, loss = 0.001597
grad_step = 000163, loss = 0.001570
grad_step = 000164, loss = 0.001543
grad_step = 000165, loss = 0.001537
grad_step = 000166, loss = 0.001548
grad_step = 000167, loss = 0.001561
grad_step = 000168, loss = 0.001574
grad_step = 000169, loss = 0.001571
grad_step = 000170, loss = 0.001560
grad_step = 000171, loss = 0.001537
grad_step = 000172, loss = 0.001521
grad_step = 000173, loss = 0.001514
grad_step = 000174, loss = 0.001517
grad_step = 000175, loss = 0.001526
grad_step = 000176, loss = 0.001532
grad_step = 000177, loss = 0.001538
grad_step = 000178, loss = 0.001537
grad_step = 000179, loss = 0.001536
grad_step = 000180, loss = 0.001524
grad_step = 000181, loss = 0.001511
grad_step = 000182, loss = 0.001498
grad_step = 000183, loss = 0.001488
grad_step = 000184, loss = 0.001484
grad_step = 000185, loss = 0.001485
grad_step = 000186, loss = 0.001489
grad_step = 000187, loss = 0.001493
grad_step = 000188, loss = 0.001501
grad_step = 000189, loss = 0.001511
grad_step = 000190, loss = 0.001527
grad_step = 000191, loss = 0.001536
grad_step = 000192, loss = 0.001546
grad_step = 000193, loss = 0.001528
grad_step = 000194, loss = 0.001501
grad_step = 000195, loss = 0.001468
grad_step = 000196, loss = 0.001454
grad_step = 000197, loss = 0.001462
grad_step = 000198, loss = 0.001478
grad_step = 000199, loss = 0.001488
grad_step = 000200, loss = 0.001483
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001469
grad_step = 000202, loss = 0.001451
grad_step = 000203, loss = 0.001438
grad_step = 000204, loss = 0.001434
grad_step = 000205, loss = 0.001436
grad_step = 000206, loss = 0.001443
grad_step = 000207, loss = 0.001452
grad_step = 000208, loss = 0.001466
grad_step = 000209, loss = 0.001478
grad_step = 000210, loss = 0.001494
grad_step = 000211, loss = 0.001493
grad_step = 000212, loss = 0.001484
grad_step = 000213, loss = 0.001455
grad_step = 000214, loss = 0.001427
grad_step = 000215, loss = 0.001409
grad_step = 000216, loss = 0.001410
grad_step = 000217, loss = 0.001420
grad_step = 000218, loss = 0.001431
grad_step = 000219, loss = 0.001438
grad_step = 000220, loss = 0.001432
grad_step = 000221, loss = 0.001418
grad_step = 000222, loss = 0.001402
grad_step = 000223, loss = 0.001391
grad_step = 000224, loss = 0.001388
grad_step = 000225, loss = 0.001392
grad_step = 000226, loss = 0.001399
grad_step = 000227, loss = 0.001407
grad_step = 000228, loss = 0.001418
grad_step = 000229, loss = 0.001426
grad_step = 000230, loss = 0.001438
grad_step = 000231, loss = 0.001439
grad_step = 000232, loss = 0.001436
grad_step = 000233, loss = 0.001417
grad_step = 000234, loss = 0.001396
grad_step = 000235, loss = 0.001375
grad_step = 000236, loss = 0.001363
grad_step = 000237, loss = 0.001362
grad_step = 000238, loss = 0.001369
grad_step = 000239, loss = 0.001379
grad_step = 000240, loss = 0.001387
grad_step = 000241, loss = 0.001394
grad_step = 000242, loss = 0.001396
grad_step = 000243, loss = 0.001400
grad_step = 000244, loss = 0.001397
grad_step = 000245, loss = 0.001392
grad_step = 000246, loss = 0.001379
grad_step = 000247, loss = 0.001365
grad_step = 000248, loss = 0.001350
grad_step = 000249, loss = 0.001339
grad_step = 000250, loss = 0.001334
grad_step = 000251, loss = 0.001334
grad_step = 000252, loss = 0.001337
grad_step = 000253, loss = 0.001344
grad_step = 000254, loss = 0.001356
grad_step = 000255, loss = 0.001374
grad_step = 000256, loss = 0.001410
grad_step = 000257, loss = 0.001457
grad_step = 000258, loss = 0.001535
grad_step = 000259, loss = 0.001549
grad_step = 000260, loss = 0.001517
grad_step = 000261, loss = 0.001390
grad_step = 000262, loss = 0.001324
grad_step = 000263, loss = 0.001361
grad_step = 000264, loss = 0.001407
grad_step = 000265, loss = 0.001395
grad_step = 000266, loss = 0.001350
grad_step = 000267, loss = 0.001343
grad_step = 000268, loss = 0.001350
grad_step = 000269, loss = 0.001338
grad_step = 000270, loss = 0.001335
grad_step = 000271, loss = 0.001344
grad_step = 000272, loss = 0.001340
grad_step = 000273, loss = 0.001318
grad_step = 000274, loss = 0.001305
grad_step = 000275, loss = 0.001316
grad_step = 000276, loss = 0.001326
grad_step = 000277, loss = 0.001317
grad_step = 000278, loss = 0.001297
grad_step = 000279, loss = 0.001290
grad_step = 000280, loss = 0.001301
grad_step = 000281, loss = 0.001308
grad_step = 000282, loss = 0.001301
grad_step = 000283, loss = 0.001285
grad_step = 000284, loss = 0.001277
grad_step = 000285, loss = 0.001281
grad_step = 000286, loss = 0.001287
grad_step = 000287, loss = 0.001287
grad_step = 000288, loss = 0.001280
grad_step = 000289, loss = 0.001274
grad_step = 000290, loss = 0.001274
grad_step = 000291, loss = 0.001282
grad_step = 000292, loss = 0.001291
grad_step = 000293, loss = 0.001305
grad_step = 000294, loss = 0.001325
grad_step = 000295, loss = 0.001370
grad_step = 000296, loss = 0.001426
grad_step = 000297, loss = 0.001522
grad_step = 000298, loss = 0.001524
grad_step = 000299, loss = 0.001484
grad_step = 000300, loss = 0.001345
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001276
grad_step = 000302, loss = 0.001305
grad_step = 000303, loss = 0.001341
grad_step = 000304, loss = 0.001325
grad_step = 000305, loss = 0.001290
grad_step = 000306, loss = 0.001294
grad_step = 000307, loss = 0.001306
grad_step = 000308, loss = 0.001281
grad_step = 000309, loss = 0.001260
grad_step = 000310, loss = 0.001269
grad_step = 000311, loss = 0.001288
grad_step = 000312, loss = 0.001281
grad_step = 000313, loss = 0.001251
grad_step = 000314, loss = 0.001237
grad_step = 000315, loss = 0.001251
grad_step = 000316, loss = 0.001266
grad_step = 000317, loss = 0.001259
grad_step = 000318, loss = 0.001237
grad_step = 000319, loss = 0.001224
grad_step = 000320, loss = 0.001230
grad_step = 000321, loss = 0.001241
grad_step = 000322, loss = 0.001242
grad_step = 000323, loss = 0.001233
grad_step = 000324, loss = 0.001221
grad_step = 000325, loss = 0.001215
grad_step = 000326, loss = 0.001218
grad_step = 000327, loss = 0.001223
grad_step = 000328, loss = 0.001223
grad_step = 000329, loss = 0.001219
grad_step = 000330, loss = 0.001211
grad_step = 000331, loss = 0.001205
grad_step = 000332, loss = 0.001202
grad_step = 000333, loss = 0.001203
grad_step = 000334, loss = 0.001205
grad_step = 000335, loss = 0.001206
grad_step = 000336, loss = 0.001205
grad_step = 000337, loss = 0.001203
grad_step = 000338, loss = 0.001200
grad_step = 000339, loss = 0.001197
grad_step = 000340, loss = 0.001197
grad_step = 000341, loss = 0.001199
grad_step = 000342, loss = 0.001210
grad_step = 000343, loss = 0.001233
grad_step = 000344, loss = 0.001283
grad_step = 000345, loss = 0.001356
grad_step = 000346, loss = 0.001483
grad_step = 000347, loss = 0.001540
grad_step = 000348, loss = 0.001549
grad_step = 000349, loss = 0.001385
grad_step = 000350, loss = 0.001264
grad_step = 000351, loss = 0.001263
grad_step = 000352, loss = 0.001312
grad_step = 000353, loss = 0.001309
grad_step = 000354, loss = 0.001234
grad_step = 000355, loss = 0.001219
grad_step = 000356, loss = 0.001262
grad_step = 000357, loss = 0.001266
grad_step = 000358, loss = 0.001225
grad_step = 000359, loss = 0.001187
grad_step = 000360, loss = 0.001207
grad_step = 000361, loss = 0.001236
grad_step = 000362, loss = 0.001213
grad_step = 000363, loss = 0.001176
grad_step = 000364, loss = 0.001177
grad_step = 000365, loss = 0.001204
grad_step = 000366, loss = 0.001202
grad_step = 000367, loss = 0.001168
grad_step = 000368, loss = 0.001151
grad_step = 000369, loss = 0.001169
grad_step = 000370, loss = 0.001185
grad_step = 000371, loss = 0.001174
grad_step = 000372, loss = 0.001153
grad_step = 000373, loss = 0.001150
grad_step = 000374, loss = 0.001161
grad_step = 000375, loss = 0.001161
grad_step = 000376, loss = 0.001147
grad_step = 000377, loss = 0.001137
grad_step = 000378, loss = 0.001141
grad_step = 000379, loss = 0.001149
grad_step = 000380, loss = 0.001146
grad_step = 000381, loss = 0.001139
grad_step = 000382, loss = 0.001136
grad_step = 000383, loss = 0.001141
grad_step = 000384, loss = 0.001145
grad_step = 000385, loss = 0.001144
grad_step = 000386, loss = 0.001143
grad_step = 000387, loss = 0.001148
grad_step = 000388, loss = 0.001160
grad_step = 000389, loss = 0.001179
grad_step = 000390, loss = 0.001195
grad_step = 000391, loss = 0.001224
grad_step = 000392, loss = 0.001241
grad_step = 000393, loss = 0.001264
grad_step = 000394, loss = 0.001238
grad_step = 000395, loss = 0.001201
grad_step = 000396, loss = 0.001144
grad_step = 000397, loss = 0.001112
grad_step = 000398, loss = 0.001116
grad_step = 000399, loss = 0.001140
grad_step = 000400, loss = 0.001157
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001144
grad_step = 000402, loss = 0.001121
grad_step = 000403, loss = 0.001104
grad_step = 000404, loss = 0.001105
grad_step = 000405, loss = 0.001117
grad_step = 000406, loss = 0.001122
grad_step = 000407, loss = 0.001118
grad_step = 000408, loss = 0.001106
grad_step = 000409, loss = 0.001097
grad_step = 000410, loss = 0.001095
grad_step = 000411, loss = 0.001097
grad_step = 000412, loss = 0.001099
grad_step = 000413, loss = 0.001099
grad_step = 000414, loss = 0.001097
grad_step = 000415, loss = 0.001094
grad_step = 000416, loss = 0.001091
grad_step = 000417, loss = 0.001089
grad_step = 000418, loss = 0.001086
grad_step = 000419, loss = 0.001083
grad_step = 000420, loss = 0.001080
grad_step = 000421, loss = 0.001078
grad_step = 000422, loss = 0.001078
grad_step = 000423, loss = 0.001079
grad_step = 000424, loss = 0.001081
grad_step = 000425, loss = 0.001083
grad_step = 000426, loss = 0.001084
grad_step = 000427, loss = 0.001084
grad_step = 000428, loss = 0.001085
grad_step = 000429, loss = 0.001089
grad_step = 000430, loss = 0.001095
grad_step = 000431, loss = 0.001106
grad_step = 000432, loss = 0.001120
grad_step = 000433, loss = 0.001143
grad_step = 000434, loss = 0.001159
grad_step = 000435, loss = 0.001180
grad_step = 000436, loss = 0.001175
grad_step = 000437, loss = 0.001162
grad_step = 000438, loss = 0.001125
grad_step = 000439, loss = 0.001090
grad_step = 000440, loss = 0.001070
grad_step = 000441, loss = 0.001069
grad_step = 000442, loss = 0.001079
grad_step = 000443, loss = 0.001082
grad_step = 000444, loss = 0.001074
grad_step = 000445, loss = 0.001062
grad_step = 000446, loss = 0.001059
grad_step = 000447, loss = 0.001065
grad_step = 000448, loss = 0.001071
grad_step = 000449, loss = 0.001070
grad_step = 000450, loss = 0.001060
grad_step = 000451, loss = 0.001048
grad_step = 000452, loss = 0.001040
grad_step = 000453, loss = 0.001040
grad_step = 000454, loss = 0.001044
grad_step = 000455, loss = 0.001049
grad_step = 000456, loss = 0.001050
grad_step = 000457, loss = 0.001047
grad_step = 000458, loss = 0.001042
grad_step = 000459, loss = 0.001037
grad_step = 000460, loss = 0.001034
grad_step = 000461, loss = 0.001032
grad_step = 000462, loss = 0.001031
grad_step = 000463, loss = 0.001033
grad_step = 000464, loss = 0.001036
grad_step = 000465, loss = 0.001041
grad_step = 000466, loss = 0.001049
grad_step = 000467, loss = 0.001062
grad_step = 000468, loss = 0.001078
grad_step = 000469, loss = 0.001109
grad_step = 000470, loss = 0.001144
grad_step = 000471, loss = 0.001200
grad_step = 000472, loss = 0.001233
grad_step = 000473, loss = 0.001259
grad_step = 000474, loss = 0.001204
grad_step = 000475, loss = 0.001123
grad_step = 000476, loss = 0.001051
grad_step = 000477, loss = 0.001035
grad_step = 000478, loss = 0.001066
grad_step = 000479, loss = 0.001087
grad_step = 000480, loss = 0.001077
grad_step = 000481, loss = 0.001051
grad_step = 000482, loss = 0.001047
grad_step = 000483, loss = 0.001059
grad_step = 000484, loss = 0.001059
grad_step = 000485, loss = 0.001036
grad_step = 000486, loss = 0.001011
grad_step = 000487, loss = 0.001008
grad_step = 000488, loss = 0.001025
grad_step = 000489, loss = 0.001042
grad_step = 000490, loss = 0.001042
grad_step = 000491, loss = 0.001025
grad_step = 000492, loss = 0.001006
grad_step = 000493, loss = 0.000996
grad_step = 000494, loss = 0.000998
grad_step = 000495, loss = 0.001008
grad_step = 000496, loss = 0.001016
grad_step = 000497, loss = 0.001019
grad_step = 000498, loss = 0.001014
grad_step = 000499, loss = 0.001006
grad_step = 000500, loss = 0.000996
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000988
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

  date_run                              2020-05-16 03:14:41.304701
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.246336
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-16 03:14:41.310801
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.167543
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-16 03:14:41.318687
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.122036
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-16 03:14:41.324477
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.54588
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
0   2020-05-16 03:14:07.950120  ...    mean_absolute_error
1   2020-05-16 03:14:07.953949  ...     mean_squared_error
2   2020-05-16 03:14:07.957043  ...  median_absolute_error
3   2020-05-16 03:14:07.960145  ...               r2_score
4   2020-05-16 03:14:18.455932  ...    mean_absolute_error
5   2020-05-16 03:14:18.460462  ...     mean_squared_error
6   2020-05-16 03:14:18.464405  ...  median_absolute_error
7   2020-05-16 03:14:18.469601  ...               r2_score
8   2020-05-16 03:14:41.304701  ...    mean_absolute_error
9   2020-05-16 03:14:41.310801  ...     mean_squared_error
10  2020-05-16 03:14:41.318687  ...  median_absolute_error
11  2020-05-16 03:14:41.324477  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2fb7560fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 312396.76it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 404147.18it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 559089.77it/s] 30%|███       | 3006464/9912422 [00:00<00:08, 787828.72it/s] 57%|█████▋    | 5693440/9912422 [00:00<00:03, 1107883.32it/s] 87%|████████▋ | 8650752/9912422 [00:01<00:00, 1552574.82it/s]9920512it [00:01, 9743846.15it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 123212.69it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 312556.37it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 403100.29it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 558359.07it/s]1654784it [00:00, 2818794.58it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 46914.86it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2f69f62e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2f695920b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2f69f62e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2f694e80b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2f66d28550> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2f66d0e710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2f69f62e48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2f694a66d8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided None 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2f66d28550> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f2fb756aef0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc46c7991d0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=c39089ab1bdaa12f9f7e106919f7d19c16052f37c9ccc56485098c2a76f709c7
  Stored in directory: /tmp/pip-ephem-wheel-cache-85_ey229/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc404594710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 46s
   57344/17464789 [..............................] - ETA: 39s
   90112/17464789 [..............................] - ETA: 37s
  180224/17464789 [..............................] - ETA: 24s
  335872/17464789 [..............................] - ETA: 16s
  663552/17464789 [>.............................] - ETA: 9s 
 1327104/17464789 [=>............................] - ETA: 5s
 2654208/17464789 [===>..........................] - ETA: 2s
 5275648/17464789 [========>.....................] - ETA: 1s
 8159232/17464789 [=============>................] - ETA: 0s
10977280/17464789 [=================>............] - ETA: 0s
13778944/17464789 [======================>.......] - ETA: 0s
16678912/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-16 03:16:13.849627: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 03:16:13.853980: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-16 03:16:13.854130: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5620fe04e330 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 03:16:13.854147: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6053 - accuracy: 0.5040
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7126 - accuracy: 0.4970
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7637 - accuracy: 0.4937 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6053 - accuracy: 0.5040
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6237 - accuracy: 0.5028
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6768 - accuracy: 0.4993
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6841 - accuracy: 0.4989
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6532 - accuracy: 0.5009
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6700 - accuracy: 0.4998
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6988 - accuracy: 0.4979
11000/25000 [============>.................] - ETA: 4s - loss: 7.7182 - accuracy: 0.4966
12000/25000 [=============>................] - ETA: 4s - loss: 7.6986 - accuracy: 0.4979
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7173 - accuracy: 0.4967
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7050 - accuracy: 0.4975
15000/25000 [=================>............] - ETA: 3s - loss: 7.6809 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6944 - accuracy: 0.4982
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6964 - accuracy: 0.4981
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6990 - accuracy: 0.4979
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6789 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6688 - accuracy: 0.4999
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6624 - accuracy: 0.5003
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6615 - accuracy: 0.5003
25000/25000 [==============================] - 9s 367us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-16 03:16:30.462235
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-16 03:16:30.462235  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<13:35:01, 17.6kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:00<9:31:24, 25.1kB/s]  .vector_cache/glove.6B.zip:   1%|          | 6.56M/862M [00:00<6:37:13, 35.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.6M/862M [00:00<4:35:09, 51.3kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.5M/862M [00:00<3:10:24, 73.2kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.2M/862M [00:00<2:11:56, 105kB/s] .vector_cache/glove.6B.zip:   5%|▍         | 41.3M/862M [00:01<1:31:37, 149kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.9M/862M [00:01<1:03:35, 213kB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.3M/862M [00:01<44:35, 302kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 53.3M/862M [00:02<6:02:27, 37.2kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<4:13:36, 53.0kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<8:49:45, 25.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<6:10:27, 36.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<10:01:42, 22.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:04<7:00:43, 31.8kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<10:23:43, 21.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:05<7:16:02, 30.6kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<10:47:34, 20.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:06<7:32:48, 29.4kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:07<10:31:05, 21.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:07<7:21:16, 30.1kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:08<10:27:04, 21.2kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:08<7:18:22, 30.2kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<10:44:55, 20.5kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:09<7:30:55, 29.3kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<10:27:45, 21.0kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:10<7:18:55, 30.0kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<10:24:17, 21.1kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.0M/862M [00:11<7:16:16, 30.1kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:11<5:08:31, 42.6kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<8:40:44, 25.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:12<6:03:56, 36.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:12<4:18:08, 50.7kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:13<8:04:33, 27.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.1M/862M [00:13<5:38:41, 38.6kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:13<3:59:39, 54.5kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<7:48:51, 27.9kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.0M/862M [00:14<5:27:48, 39.8kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:14<3:51:17, 56.3kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<7:38:41, 28.4kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:15<5:20:34, 40.5kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:15<3:47:43, 57.1kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:16<7:41:26, 28.2kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.4M/862M [00:16<5:22:31, 40.2kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:16<3:48:30, 56.7kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<7:38:55, 28.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:17<5:20:41, 40.3kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:17<3:50:19, 56.1kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<7:43:15, 27.9kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:18<5:23:47, 39.8kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:18<3:49:12, 56.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:19<7:37:43, 28.2kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:19<5:19:54, 40.2kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:19<3:46:46, 56.7kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:20<7:35:20, 28.2kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:20<5:18:08, 40.3kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:20<3:49:26, 55.9kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:21<7:15:27, 29.4kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:21<5:04:14, 42.0kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:21<3:48:21, 56.0kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<7:16:23, 29.3kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:22<5:05:07, 41.8kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:22<3:35:13, 59.2kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:23<7:24:59, 28.6kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:23<5:11:01, 40.9kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:23<3:40:21, 57.7kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<7:29:28, 28.3kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<5:14:00, 40.4kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<4:08:41, 51.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<7:49:02, 27.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:25<5:27:42, 38.6kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:25<3:54:18, 54.0kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<7:45:24, 27.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<5:25:08, 38.8kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<3:54:45, 53.7kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<7:37:47, 27.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:27<5:19:48, 39.3kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:27<3:51:59, 54.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<7:42:21, 27.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<5:23:03, 38.8kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<3:50:28, 54.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<7:32:44, 27.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:29<5:16:30, 39.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:29<3:43:12, 56.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<7:32:49, 27.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<5:16:19, 39.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<3:50:06, 54.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<7:38:58, 27.2kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:31<5:21:00, 38.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<8:44:12, 23.7kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<6:06:07, 33.9kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<4:35:09, 45.1kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<8:07:19, 25.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<5:40:46, 36.3kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<8:56:33, 23.0kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:34<6:15:09, 32.9kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<9:34:07, 21.5kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:35<6:41:12, 30.7kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:35<4:42:57, 43.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<7:59:37, 25.6kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<5:35:23, 36.5kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<8:49:56, 23.1kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<6:10:30, 33.0kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<9:13:36, 22.1kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<6:26:48, 31.5kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<4:33:26, 44.6kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<8:01:35, 25.3kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<5:36:44, 36.1kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<8:48:52, 23.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<6:09:45, 32.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<9:07:23, 22.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<6:22:22, 31.6kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<4:32:50, 44.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<7:37:34, 26.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<5:19:36, 37.7kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<3:52:51, 51.7kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<7:30:17, 26.8kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<5:14:28, 38.2kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<4:07:03, 48.6kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<7:40:53, 26.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<5:21:52, 37.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<4:39:06, 42.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<8:02:43, 24.8kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<5:37:08, 35.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<4:09:33, 47.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<7:39:19, 26.0kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:46<5:20:50, 37.1kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:46<3:50:20, 51.7kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<7:23:30, 26.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<5:09:48, 38.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<3:41:58, 53.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<7:19:00, 27.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<5:06:59, 38.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<8:19:36, 23.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<5:49:00, 33.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<4:07:58, 47.6kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<7:34:02, 26.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<5:17:09, 37.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<3:47:20, 51.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<7:19:12, 26.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<5:07:07, 38.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<8:14:08, 23.7kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<5:45:13, 33.9kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<4:04:31, 47.8kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<7:37:44, 25.6kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<5:19:35, 36.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<3:46:24, 51.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<2:39:18, 73.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:55<1:51:51, 104kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<5:38:21, 34.3kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:56<3:56:13, 48.9kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<2:48:26, 68.4kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<1:58:47, 97.0kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<1:23:34, 137kB/s] .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<5:10:39, 37.0kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<3:36:59, 52.8kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<2:34:29, 73.9kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<1:49:02, 105kB/s] .vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<1:16:45, 148kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<5:02:08, 37.7kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<3:30:55, 53.8kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<2:30:49, 75.0kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<1:46:25, 106kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:04<1:14:56, 151kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<4:59:32, 37.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:05<3:29:10, 53.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<2:29:10, 75.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<1:45:15, 106kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<1:14:06, 151kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<4:56:54, 37.6kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:08<3:27:23, 53.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<2:27:38, 75.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<1:44:09, 107kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<1:13:20, 151kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<4:55:46, 37.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<3:26:30, 53.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<2:27:19, 74.7kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 203M/862M [01:13<1:43:57, 106kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<1:13:11, 150kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<4:51:50, 37.6kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<3:23:57, 53.6kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:14<2:22:28, 76.5kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<2:03:39, 88.1kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<1:27:22, 125kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:16<1:01:37, 176kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<4:37:56, 39.1kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<3:14:12, 55.8kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:17<2:15:42, 79.6kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<1:54:12, 94.5kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<1:20:43, 134kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:19<56:58, 189kB/s]  .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<4:35:17, 39.1kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:20<3:12:27, 55.7kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:20<2:14:22, 79.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<2:00:20, 88.8kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<1:25:02, 126kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:22<1:00:03, 177kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<4:15:49, 41.6kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:23<2:58:44, 59.4kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<2:07:20, 83.1kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<1:29:55, 118kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:25<1:03:26, 166kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<4:17:57, 40.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:26<3:00:13, 58.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:26<2:05:51, 83.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<3:42:17, 47.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<2:36:22, 66.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:28<1:49:44, 95.1kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<5:04:57, 34.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<3:32:59, 48.9kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<2:31:11, 68.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<1:46:35, 97.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<1:14:58, 138kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<4:39:43, 36.9kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:32<3:15:21, 52.7kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<2:18:53, 73.9kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<1:37:58, 105kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:34<1:08:57, 148kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<4:35:50, 37.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<3:12:38, 52.9kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<2:16:55, 74.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<1:36:35, 105kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<1:07:59, 149kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<4:28:31, 37.7kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:38<3:07:33, 53.8kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<2:13:17, 75.5kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<1:34:02, 107kB/s] .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:40<1:06:11, 151kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<4:27:47, 37.4kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:41<3:07:00, 53.4kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<2:13:01, 74.8kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<1:33:52, 106kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<1:06:03, 150kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<4:26:26, 37.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<3:06:03, 53.1kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<2:12:14, 74.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<1:33:17, 105kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<1:05:38, 149kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<4:22:42, 37.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<3:03:31, 53.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:47<2:08:07, 76.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<1:51:19, 87.5kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<1:18:39, 124kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:49<55:25, 175kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<4:11:38, 38.6kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<2:55:41, 55.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<2:05:04, 77.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<1:28:15, 109kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:52<1:02:08, 155kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<4:12:12, 38.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<2:56:10, 54.3kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:53<2:02:56, 77.5kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<11:40:39, 13.6kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<8:10:50, 19.4kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<5:43:05, 27.7kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<7:27:14, 21.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<5:12:47, 30.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:56<3:38:49, 43.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:56<2:33:10, 61.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<1:52:53, 83.5kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<1:19:43, 118kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:58<55:58, 168kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:58<40:37, 231kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<3:22:26, 46.4kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<2:21:51, 66.1kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [01:59<1:39:25, 94.1kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [01:59<1:09:42, 134kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [01:59<48:59, 190kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<1:27:08, 107kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<1:01:42, 151kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<43:21, 214kB/s]  .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:01<31:49, 292kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<3:16:03, 47.4kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:02<2:17:22, 67.5kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<1:36:15, 96.1kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:02<1:07:28, 137kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<51:33, 179kB/s]  .vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:04<36:49, 250kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<25:56, 354kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:04<20:00, 459kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<3:07:02, 49.1kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<2:11:02, 69.9kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:05<1:31:44, 99.7kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:05<1:04:17, 142kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<52:55, 172kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<37:45, 241kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<26:33, 342kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:07<18:49, 482kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<17:02, 531kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<12:38, 714kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<09:02, 997kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:09<08:05, 1.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<3:03:32, 49.1kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<2:08:33, 69.9kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<1:29:58, 99.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:10<1:03:03, 142kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<53:52, 166kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<38:23, 233kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:12<27:01, 330kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:12<19:05, 466kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<17:12, 515kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<12:44, 696kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:14<09:06, 971kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<08:00, 1.10MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<3:02:38, 48.4kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<2:07:51, 68.9kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:15<1:29:26, 98.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:15<1:02:37, 140kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<1:24:35, 104kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<1:00:28, 145kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<42:27, 206kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:17<29:52, 292kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:17<21:04, 413kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:19<35:19, 246kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<25:24, 342kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<17:55, 483kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:19<14:26, 600kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:20<3:04:54, 46.8kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<2:09:23, 66.8kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:20<1:30:32, 95.2kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:20<1:03:22, 136kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<1:00:27, 142kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<43:03, 199kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:22<30:15, 283kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:22<21:17, 401kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<20:07, 423kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<15:10, 561kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<10:46, 788kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:24<09:03, 937kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<2:47:55, 50.5kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<1:57:30, 72.0kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:25<1:22:11, 103kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<1:00:18, 140kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:27<42:54, 196kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<30:07, 278kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:27<21:10, 395kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<19:59, 417kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<14:41, 568kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:29<10:24, 798kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:29<10:44, 774kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<3:04:10, 45.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:30<2:08:47, 64.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:30<1:29:59, 91.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<1:06:48, 123kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<47:18, 174kB/s]  .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<33:11, 247kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:32<23:17, 351kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<28:02, 291kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<20:16, 403kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<14:18, 569kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<14:28, 562kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<2:46:07, 49.0kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:35<1:56:16, 69.8kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:35<1:21:17, 99.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<59:35, 135kB/s]   .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<42:21, 190kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:37<29:43, 270kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:37<20:52, 383kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<23:24, 342kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<17:00, 469kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:39<12:00, 663kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:39<17:48, 447kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<4:14:35, 31.3kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<2:58:17, 44.6kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:40<2:04:31, 63.6kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:40<1:27:00, 90.7kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<1:35:29, 82.6kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<1:07:32, 117kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<47:18, 166kB/s]  .vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:42<33:10, 236kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<27:17, 287kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<20:15, 386kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<14:20, 544kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:44<10:10, 764kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<10:15, 756kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<07:52, 985kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:46<05:40, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:46<04:07, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<06:12, 1.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<04:56, 1.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<03:39, 2.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:48<02:42, 2.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<05:28, 1.39MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<04:49, 1.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<03:32, 2.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:50<02:37, 2.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<05:16, 1.43MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<04:18, 1.75MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<03:10, 2.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:52<02:21, 3.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<07:24, 1.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<05:56, 1.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<04:18, 1.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:54<03:08, 2.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<06:56, 1.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<05:43, 1.29MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<04:08, 1.78MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:56<03:01, 2.43MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<07:48, 940kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<06:03, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<04:21, 1.68MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:58<03:10, 2.29MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<11:32, 630kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<08:54, 816kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<06:23, 1.13MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:00<05:02, 1.43MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<3:29:24, 34.5kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<2:26:38, 49.2kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:01<1:42:17, 70.2kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:02<1:11:51, 99.6kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<7:10:09, 16.6kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<5:01:35, 23.7kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:03<3:30:28, 33.8kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:03<2:26:51, 48.3kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<1:49:52, 64.5kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<1:17:23, 91.5kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:05<54:04, 130kB/s]   .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:05<37:48, 186kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<58:58, 119kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<41:58, 167kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<29:29, 237kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:07<20:48, 335kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:07<14:45, 472kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<18:16, 380kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<13:26, 517kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<09:38, 719kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<06:56, 996kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<05:04, 1.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<06:41, 1.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:15, 1.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:57, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<02:59, 2.28MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:17, 2.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<04:33, 1.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<03:59, 1.70MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:00, 2.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:17, 2.95MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:23<55:12, 123kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:23<39:09, 173kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:23<27:30, 245kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:23<19:22, 347kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:23<13:42, 489kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:24<16:46, 399kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:24<12:03, 555kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:24<08:35, 776kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:25<06:09, 1.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:30<14:36, 454kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:30<10:45, 616kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:30<07:39, 862kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:30<05:30, 1.20MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:39<17:47, 369kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:39<12:58, 505kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:39<09:12, 709kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:39<06:34, 989kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:55<33:08, 196kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:55<23:41, 274kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:55<16:40, 387kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [04:02<39:00, 165kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [04:02<27:47, 232kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [04:02<19:53, 324kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [04:02<14:01, 457kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [04:02<09:54, 645kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [04:03<15:41, 407kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [04:03<11:06, 573kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [04:03<07:51, 806kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [04:04<11:15, 563kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [04:04<11:09, 567kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [04:04<07:57, 792kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [04:05<05:42, 1.10MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [04:05<04:08, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [04:13<47:29, 132kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [04:13<33:42, 186kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [04:13<23:40, 263kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [04:14<16:39, 373kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [04:16<14:49, 418kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [04:16<10:57, 564kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [04:16<07:50, 787kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [04:16<05:35, 1.10MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [04:18<06:10, 992kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [04:18<04:45, 1.29MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [04:18<03:29, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [04:18<02:34, 2.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [04:20<04:03, 1.49MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [04:20<03:31, 1.71MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [04:20<02:36, 2.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [04:20<01:56, 3.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [04:26<11:36, 516kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [04:27<08:35, 695kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [04:27<06:06, 974kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [04:27<04:22, 1.35MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [04:41<54:31, 109kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [04:41<38:36, 153kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [04:41<26:59, 218kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [04:41<18:54, 309kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [05:44<37:32:15, 2.60kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [05:44<26:15:12, 3.71kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [05:44<18:18:25, 5.30kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [05:44<12:45:09, 7.57kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [05:50<9:11:36, 10.5kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [05:50<6:26:05, 14.9kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [05:50<4:28:51, 21.3kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [06:02<3:18:53, 28.7kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [06:02<2:19:30, 40.9kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [06:02<1:37:13, 58.3kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [06:13<1:17:44, 72.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [06:13<54:49, 103kB/s]   .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [06:13<38:14, 146kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [06:13<28:13, 198kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [06:14<1:32:24, 60.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [06:14<1:04:27, 86.4kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [06:14<44:59, 123kB/s]   .vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [06:17<38:48, 142kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [06:17<27:34, 200kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [06:17<19:16, 285kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [06:49<45:04, 121kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [06:49<31:57, 171kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [06:49<22:21, 243kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [06:49<15:38, 345kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [06:52<2:42:58, 33.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [06:52<1:54:23, 47.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [06:52<1:19:42, 67.2kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [07:03<1:04:46, 82.1kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [07:03<45:43, 116kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [07:03<31:53, 166kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [07:03<22:17, 236kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [07:05<2:05:43, 41.8kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [07:05<1:28:18, 59.4kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [07:05<1:01:33, 84.7kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [07:06<42:55, 121kB/s]   .vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [07:11<1:37:47, 53.0kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [07:11<1:08:46, 75.3kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [07:11<47:55, 107kB/s]   .vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [07:24<44:06, 116kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [07:24<31:14, 163kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [07:24<21:50, 233kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [07:24<15:16, 330kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [07:31<3:05:57, 27.1kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [07:31<2:10:20, 38.7kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [07:31<1:30:55, 55.2kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [07:31<1:03:19, 78.7kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [07:35<58:21, 85.3kB/s]  .vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [07:35<41:16, 120kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [07:35<28:48, 172kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [07:38<22:43, 216kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [07:38<16:17, 301kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [07:38<11:23, 427kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [07:54<21:38, 224kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [07:54<15:32, 311kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [07:54<10:52, 441kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [07:57<10:15, 465kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [07:57<07:30, 635kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [07:57<05:18, 893kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [08:00<05:53, 799kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [08:00<04:29, 1.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [08:00<03:10, 1.46MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [08:04<05:36, 826kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [08:04<04:16, 1.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [08:04<03:05, 1.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [08:04<02:15, 2.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [08:12<13:08, 347kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [08:12<09:33, 476kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [08:13<06:44, 672kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [08:13<04:48, 938kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [08:54<1:26:16, 52.1kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [08:54<1:00:36, 74.1kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [08:54<42:18, 106kB/s]   .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [08:54<29:30, 150kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [09:10<58:50, 75.2kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [09:10<41:28, 107kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [09:11<28:54, 152kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [09:11<20:13, 216kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [09:12<20:18, 215kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [09:13<14:33, 299kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [09:13<10:11, 424kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [09:13<07:11, 598kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [09:14<11:19, 379kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [09:14<08:15, 518kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [09:15<05:49, 731kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [09:15<04:07, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [09:16<09:35, 440kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [09:16<07:02, 599kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [09:17<04:57, 842kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [09:18<04:42, 882kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [09:18<03:37, 1.14MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [09:18<02:35, 1.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [09:20<03:02, 1.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [09:20<02:27, 1.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [09:20<01:46, 2.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [09:22<02:23, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [09:22<02:00, 2.00MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [09:22<01:27, 2.74MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [09:24<02:14, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [09:24<01:52, 2.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [09:24<01:21, 2.87MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [09:26<02:12, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [09:26<01:51, 2.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [09:26<01:21, 2.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [09:29<02:59, 1.27MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [09:29<02:24, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [09:29<01:43, 2.18MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [09:31<02:14, 1.66MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [09:31<01:52, 1.99MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [09:31<01:21, 2.73MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [09:33<02:04, 1.77MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [09:33<01:44, 2.10MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [09:33<01:16, 2.86MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [09:36<02:30, 1.44MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [09:36<02:02, 1.76MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [09:36<01:28, 2.41MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [10:03<16:44, 211kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [10:03<11:59, 294kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [10:03<08:22, 417kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [10:08<09:01, 384kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [10:08<06:35, 525kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [10:08<04:37, 741kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [10:12<05:49, 583kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [10:12<04:20, 779kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [10:12<03:03, 1.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [10:14<03:22, 987kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [10:14<02:37, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [10:14<01:52, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [10:27<07:56, 410kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [10:27<05:48, 559kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [10:27<04:04, 789kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [10:29<03:44, 852kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [10:29<03:03, 1.04MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [10:29<02:12, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [10:29<01:35, 1.98MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [10:31<02:50, 1.10MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [10:31<02:16, 1.37MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [10:31<01:38, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [10:31<01:10, 2.59MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [10:33<06:50, 446kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [10:33<05:04, 600kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [10:33<03:35, 843kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [10:33<02:31, 1.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [10:35<05:49, 512kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [10:35<04:25, 674kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [10:35<03:07, 946kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [10:35<02:12, 1.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [10:37<05:51, 497kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [10:37<04:19, 671kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [10:37<03:02, 944kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [10:39<02:54, 980kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [10:39<02:16, 1.25MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [10:39<01:38, 1.72MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [10:39<01:10, 2.37MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [10:41<06:52, 404kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [10:41<05:04, 546kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [10:41<03:34, 768kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [10:41<02:30, 1.08MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [10:43<06:37, 408kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [10:43<04:51, 556kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [10:43<03:24, 784kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [10:43<02:24, 1.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [10:45<06:25, 411kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [10:45<04:41, 560kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [10:45<03:17, 790kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [10:47<03:01, 852kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [10:47<02:19, 1.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [10:47<01:38, 1.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [10:49<01:52, 1.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [10:49<01:30, 1.65MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [10:49<01:05, 2.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [10:51<01:26, 1.68MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [10:51<01:11, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [10:51<00:52, 2.74MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [10:51<00:38, 3.70MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [10:53<11:31, 205kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [10:53<08:14, 286kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [10:53<05:43, 406kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [10:55<04:39, 492kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [10:55<03:25, 669kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [10:55<02:24, 940kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [10:55<01:41, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [10:57<14:57, 149kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [10:57<10:36, 209kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [10:57<07:21, 297kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [10:59<05:43, 376kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [10:59<04:10, 515kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [10:59<02:54, 728kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [11:01<02:38, 789kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [11:01<02:00, 1.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [11:01<01:24, 1.44MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [11:03<01:37, 1.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [11:03<01:18, 1.55MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [11:03<00:55, 2.14MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [11:05<01:17, 1.51MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [11:05<01:02, 1.86MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [11:05<00:45, 2.54MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [11:07<01:00, 1.87MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [11:07<00:53, 2.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [11:07<00:39, 2.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [11:09<00:55, 1.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [11:09<00:54, 1.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [11:09<00:39, 2.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [11:11<00:53, 1.94MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [11:11<00:48, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [11:11<00:35, 2.91MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [11:12<00:50, 2.01MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [11:13<00:43, 2.30MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [11:13<00:31, 3.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [11:14<00:45, 2.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [11:15<00:45, 2.13MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [11:15<00:32, 2.90MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [11:16<00:46, 1.97MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [11:17<00:39, 2.30MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [11:17<00:29, 3.09MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [11:17<00:22, 4.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [11:19<01:15, 1.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [11:19<01:04, 1.36MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [11:19<00:46, 1.87MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [11:19<00:33, 2.52MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [11:21<01:01, 1.36MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [11:21<00:51, 1.62MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [11:21<00:37, 2.22MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [11:21<00:26, 2.99MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [11:22<01:18, 1.02MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [11:23<01:06, 1.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [11:23<00:47, 1.64MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [11:23<00:34, 2.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [11:24<01:11, 1.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [11:25<01:00, 1.25MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [11:25<00:42, 1.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [11:25<00:30, 2.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [11:26<01:35, 753kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [11:27<01:13, 964kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [11:27<00:52, 1.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [11:27<00:36, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [11:28<01:37, 693kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [11:28<01:14, 896kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [11:29<00:52, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [11:29<00:36, 1.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [11:30<01:44, 607kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [11:30<01:21, 772kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [11:31<00:56, 1.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [11:31<00:39, 1.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [11:32<01:27, 676kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [11:32<01:05, 900kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [11:33<00:45, 1.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [11:33<00:31, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [11:34<01:29, 616kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [11:34<01:08, 800kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [11:35<00:47, 1.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [11:35<00:32, 1.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [11:36<02:22, 359kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [11:36<01:44, 487kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [11:36<01:11, 687kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [11:37<00:48, 966kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [11:38<01:53, 414kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [11:38<01:25, 546kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [11:38<00:58, 769kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [11:39<00:39, 1.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [11:40<02:31, 281kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [11:40<01:49, 387kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [11:40<01:14, 548kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [11:41<00:50, 773kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [11:42<03:35, 179kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [11:42<02:32, 250kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [11:42<01:42, 356kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [11:42<01:08, 503kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [11:44<02:15, 255kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [11:44<01:39, 346kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [11:44<01:06, 490kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [11:44<00:43, 693kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [11:46<02:19, 218kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [11:46<01:38, 304kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [11:46<01:05, 431kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [11:48<00:50, 523kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [11:48<00:36, 702kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [11:48<00:24, 986kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [11:50<00:21, 1.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [11:50<00:16, 1.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [11:50<00:11, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [11:52<00:12, 1.49MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [11:52<00:09, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [11:52<00:06, 2.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [11:54<00:07, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [11:54<00:06, 2.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [11:54<00:04, 2.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [11:56<00:05, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [11:56<00:04, 2.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [11:56<00:02, 2.78MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [11:56<00:01, 3.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [11:58<02:50, 32.6kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [11:58<01:53, 46.4kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [11:58<00:53, 66.2kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [11:58<00:15, 94.5kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [12:00<00:28, 51.1kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [12:00<00:15, 72.4kB/s].vector_cache/glove.6B.zip: 862MB [12:00, 1.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 776/400000 [00:00<00:51, 7755.11it/s]  0%|          | 1532/400000 [00:00<00:51, 7693.51it/s]  1%|          | 2207/400000 [00:00<00:53, 7382.22it/s]  1%|          | 2900/400000 [00:00<00:54, 7238.57it/s]  1%|          | 3584/400000 [00:00<00:55, 7111.32it/s]  1%|          | 4245/400000 [00:00<00:56, 6952.57it/s]  1%|▏         | 5006/400000 [00:00<00:55, 7135.19it/s]  1%|▏         | 5749/400000 [00:00<00:54, 7220.92it/s]  2%|▏         | 6513/400000 [00:00<00:53, 7341.49it/s]  2%|▏         | 7274/400000 [00:01<00:52, 7417.67it/s]  2%|▏         | 7996/400000 [00:01<00:53, 7289.56it/s]  2%|▏         | 8764/400000 [00:01<00:52, 7400.06it/s]  2%|▏         | 9517/400000 [00:01<00:52, 7437.50it/s]  3%|▎         | 10279/400000 [00:01<00:52, 7489.49it/s]  3%|▎         | 11047/400000 [00:01<00:51, 7544.45it/s]  3%|▎         | 11818/400000 [00:01<00:51, 7592.08it/s]  3%|▎         | 12583/400000 [00:01<00:50, 7609.05it/s]  3%|▎         | 13343/400000 [00:01<00:51, 7485.52it/s]  4%|▎         | 14092/400000 [00:01<00:52, 7409.66it/s]  4%|▎         | 14833/400000 [00:02<00:53, 7243.82it/s]  4%|▍         | 15583/400000 [00:02<00:52, 7317.38it/s]  4%|▍         | 16329/400000 [00:02<00:52, 7357.12it/s]  4%|▍         | 17072/400000 [00:02<00:51, 7376.11it/s]  4%|▍         | 17813/400000 [00:02<00:51, 7383.10it/s]  5%|▍         | 18567/400000 [00:02<00:51, 7428.96it/s]  5%|▍         | 19311/400000 [00:02<00:51, 7419.97it/s]  5%|▌         | 20066/400000 [00:02<00:50, 7456.97it/s]  5%|▌         | 20819/400000 [00:02<00:50, 7476.20it/s]  5%|▌         | 21581/400000 [00:02<00:50, 7516.49it/s]  6%|▌         | 22333/400000 [00:03<00:50, 7421.85it/s]  6%|▌         | 23104/400000 [00:03<00:50, 7494.78it/s]  6%|▌         | 23878/400000 [00:03<00:49, 7565.43it/s]  6%|▌         | 24640/400000 [00:03<00:49, 7574.74it/s]  6%|▋         | 25410/400000 [00:03<00:49, 7609.34it/s]  7%|▋         | 26181/400000 [00:03<00:48, 7636.99it/s]  7%|▋         | 26945/400000 [00:03<00:49, 7572.33it/s]  7%|▋         | 27703/400000 [00:03<00:49, 7558.46it/s]  7%|▋         | 28460/400000 [00:03<00:49, 7461.29it/s]  7%|▋         | 29210/400000 [00:03<00:49, 7471.41it/s]  7%|▋         | 29983/400000 [00:04<00:49, 7546.17it/s]  8%|▊         | 30773/400000 [00:04<00:48, 7648.00it/s]  8%|▊         | 31539/400000 [00:04<00:48, 7640.69it/s]  8%|▊         | 32308/400000 [00:04<00:48, 7653.20it/s]  8%|▊         | 33097/400000 [00:04<00:47, 7717.13it/s]  8%|▊         | 33886/400000 [00:04<00:47, 7766.30it/s]  9%|▊         | 34678/400000 [00:04<00:46, 7809.78it/s]  9%|▉         | 35462/400000 [00:04<00:46, 7817.92it/s]  9%|▉         | 36244/400000 [00:04<00:47, 7718.98it/s]  9%|▉         | 37027/400000 [00:04<00:46, 7749.74it/s]  9%|▉         | 37803/400000 [00:05<00:46, 7740.26it/s] 10%|▉         | 38587/400000 [00:05<00:46, 7769.39it/s] 10%|▉         | 39376/400000 [00:05<00:46, 7803.94it/s] 10%|█         | 40157/400000 [00:05<00:46, 7784.88it/s] 10%|█         | 40936/400000 [00:05<00:46, 7689.20it/s] 10%|█         | 41726/400000 [00:05<00:46, 7748.60it/s] 11%|█         | 42502/400000 [00:05<00:46, 7729.70it/s] 11%|█         | 43276/400000 [00:05<00:47, 7477.34it/s] 11%|█         | 44026/400000 [00:05<00:49, 7258.55it/s] 11%|█         | 44755/400000 [00:05<00:49, 7155.00it/s] 11%|█▏        | 45473/400000 [00:06<00:50, 7081.03it/s] 12%|█▏        | 46247/400000 [00:06<00:48, 7266.35it/s] 12%|█▏        | 47012/400000 [00:06<00:47, 7376.97it/s] 12%|█▏        | 47769/400000 [00:06<00:47, 7433.04it/s] 12%|█▏        | 48544/400000 [00:06<00:46, 7525.08it/s] 12%|█▏        | 49312/400000 [00:06<00:46, 7570.28it/s] 13%|█▎        | 50071/400000 [00:06<00:46, 7541.10it/s] 13%|█▎        | 50826/400000 [00:06<00:47, 7341.74it/s] 13%|█▎        | 51562/400000 [00:06<00:48, 7171.14it/s] 13%|█▎        | 52282/400000 [00:07<00:48, 7099.05it/s] 13%|█▎        | 52994/400000 [00:07<00:49, 7065.76it/s] 13%|█▎        | 53732/400000 [00:07<00:48, 7156.32it/s] 14%|█▎        | 54493/400000 [00:07<00:47, 7284.03it/s] 14%|█▍        | 55251/400000 [00:07<00:46, 7369.88it/s] 14%|█▍        | 56029/400000 [00:07<00:45, 7488.28it/s] 14%|█▍        | 56806/400000 [00:07<00:45, 7569.26it/s] 14%|█▍        | 57567/400000 [00:07<00:45, 7580.31it/s] 15%|█▍        | 58334/400000 [00:07<00:44, 7606.68it/s] 15%|█▍        | 59096/400000 [00:07<00:44, 7601.84it/s] 15%|█▍        | 59872/400000 [00:08<00:44, 7648.33it/s] 15%|█▌        | 60647/400000 [00:08<00:44, 7676.82it/s] 15%|█▌        | 61415/400000 [00:08<00:45, 7500.77it/s] 16%|█▌        | 62167/400000 [00:08<00:46, 7298.31it/s] 16%|█▌        | 62899/400000 [00:08<00:47, 7109.59it/s] 16%|█▌        | 63654/400000 [00:08<00:46, 7235.71it/s] 16%|█▌        | 64426/400000 [00:08<00:45, 7373.83it/s] 16%|█▋        | 65166/400000 [00:08<00:45, 7365.02it/s] 16%|█▋        | 65905/400000 [00:08<00:47, 7086.72it/s] 17%|█▋        | 66618/400000 [00:08<00:47, 6984.14it/s] 17%|█▋        | 67320/400000 [00:09<00:47, 6968.00it/s] 17%|█▋        | 68019/400000 [00:09<00:48, 6896.75it/s] 17%|█▋        | 68724/400000 [00:09<00:47, 6940.57it/s] 17%|█▋        | 69478/400000 [00:09<00:46, 7107.91it/s] 18%|█▊        | 70243/400000 [00:09<00:45, 7259.91it/s] 18%|█▊        | 70972/400000 [00:09<00:46, 7150.77it/s] 18%|█▊        | 71689/400000 [00:09<00:47, 6974.40it/s] 18%|█▊        | 72389/400000 [00:09<00:47, 6836.15it/s] 18%|█▊        | 73146/400000 [00:09<00:46, 7040.80it/s] 18%|█▊        | 73897/400000 [00:09<00:45, 7173.62it/s] 19%|█▊        | 74618/400000 [00:10<00:46, 7069.79it/s] 19%|█▉        | 75328/400000 [00:10<00:46, 6971.83it/s] 19%|█▉        | 76028/400000 [00:10<00:46, 6932.96it/s] 19%|█▉        | 76736/400000 [00:10<00:46, 6975.49it/s] 19%|█▉        | 77490/400000 [00:10<00:45, 7135.39it/s] 20%|█▉        | 78246/400000 [00:10<00:44, 7256.55it/s] 20%|█▉        | 78984/400000 [00:10<00:44, 7292.35it/s] 20%|█▉        | 79736/400000 [00:10<00:43, 7358.49it/s] 20%|██        | 80473/400000 [00:10<00:44, 7196.97it/s] 20%|██        | 81213/400000 [00:10<00:43, 7254.60it/s] 20%|██        | 81978/400000 [00:11<00:43, 7366.81it/s] 21%|██        | 82732/400000 [00:11<00:42, 7417.78it/s] 21%|██        | 83488/400000 [00:11<00:42, 7458.85it/s] 21%|██        | 84248/400000 [00:11<00:42, 7500.61it/s] 21%|██▏       | 85012/400000 [00:11<00:41, 7540.66it/s] 21%|██▏       | 85771/400000 [00:11<00:41, 7554.85it/s] 22%|██▏       | 86535/400000 [00:11<00:41, 7578.06it/s] 22%|██▏       | 87294/400000 [00:11<00:42, 7406.56it/s] 22%|██▏       | 88036/400000 [00:11<00:43, 7181.50it/s] 22%|██▏       | 88757/400000 [00:12<00:44, 7048.80it/s] 22%|██▏       | 89473/400000 [00:12<00:43, 7080.57it/s] 23%|██▎       | 90233/400000 [00:12<00:42, 7227.22it/s] 23%|██▎       | 91007/400000 [00:12<00:41, 7371.42it/s] 23%|██▎       | 91759/400000 [00:12<00:41, 7404.16it/s] 23%|██▎       | 92501/400000 [00:12<00:42, 7191.00it/s] 23%|██▎       | 93223/400000 [00:12<00:43, 7122.45it/s] 23%|██▎       | 93938/400000 [00:12<00:43, 7024.78it/s] 24%|██▎       | 94643/400000 [00:12<00:45, 6642.61it/s] 24%|██▍       | 95374/400000 [00:12<00:44, 6828.69it/s] 24%|██▍       | 96065/400000 [00:13<00:44, 6850.81it/s] 24%|██▍       | 96754/400000 [00:13<00:44, 6834.75it/s] 24%|██▍       | 97440/400000 [00:13<00:45, 6720.05it/s] 25%|██▍       | 98179/400000 [00:13<00:43, 6907.71it/s] 25%|██▍       | 98916/400000 [00:13<00:42, 7028.60it/s] 25%|██▍       | 99622/400000 [00:13<00:43, 6900.14it/s] 25%|██▌       | 100333/400000 [00:13<00:43, 6961.48it/s] 25%|██▌       | 101072/400000 [00:13<00:42, 7083.86it/s] 25%|██▌       | 101783/400000 [00:13<00:43, 6864.76it/s] 26%|██▌       | 102473/400000 [00:13<00:44, 6706.47it/s] 26%|██▌       | 103147/400000 [00:14<00:44, 6651.32it/s] 26%|██▌       | 103824/400000 [00:14<00:44, 6685.97it/s] 26%|██▌       | 104495/400000 [00:14<00:45, 6515.91it/s] 26%|██▋       | 105149/400000 [00:14<00:46, 6380.10it/s] 26%|██▋       | 105790/400000 [00:14<00:47, 6191.41it/s] 27%|██▋       | 106456/400000 [00:14<00:46, 6321.35it/s] 27%|██▋       | 107182/400000 [00:14<00:44, 6574.90it/s] 27%|██▋       | 107914/400000 [00:14<00:43, 6781.74it/s] 27%|██▋       | 108597/400000 [00:14<00:43, 6702.20it/s] 27%|██▋       | 109292/400000 [00:15<00:42, 6773.48it/s] 28%|██▊       | 110011/400000 [00:15<00:42, 6890.12it/s] 28%|██▊       | 110756/400000 [00:15<00:41, 7042.59it/s] 28%|██▊       | 111525/400000 [00:15<00:39, 7223.72it/s] 28%|██▊       | 112307/400000 [00:15<00:38, 7390.69it/s] 28%|██▊       | 113050/400000 [00:15<00:39, 7337.73it/s] 28%|██▊       | 113813/400000 [00:15<00:38, 7420.73it/s] 29%|██▊       | 114557/400000 [00:15<00:38, 7382.71it/s] 29%|██▉       | 115297/400000 [00:15<00:39, 7173.94it/s] 29%|██▉       | 116055/400000 [00:15<00:38, 7289.99it/s] 29%|██▉       | 116808/400000 [00:16<00:38, 7321.07it/s] 29%|██▉       | 117559/400000 [00:16<00:38, 7375.97it/s] 30%|██▉       | 118298/400000 [00:16<00:39, 7187.25it/s] 30%|██▉       | 119061/400000 [00:16<00:38, 7314.20it/s] 30%|██▉       | 119832/400000 [00:16<00:37, 7426.30it/s] 30%|███       | 120598/400000 [00:16<00:37, 7494.02it/s] 30%|███       | 121375/400000 [00:16<00:36, 7572.73it/s] 31%|███       | 122134/400000 [00:16<00:37, 7459.09it/s] 31%|███       | 122882/400000 [00:16<00:38, 7289.68it/s] 31%|███       | 123613/400000 [00:16<00:38, 7158.87it/s] 31%|███       | 124331/400000 [00:17<00:39, 7053.06it/s] 31%|███▏      | 125038/400000 [00:17<00:39, 7035.66it/s] 31%|███▏      | 125818/400000 [00:17<00:37, 7246.24it/s] 32%|███▏      | 126595/400000 [00:17<00:36, 7394.80it/s] 32%|███▏      | 127354/400000 [00:17<00:36, 7451.55it/s] 32%|███▏      | 128126/400000 [00:17<00:36, 7527.61it/s] 32%|███▏      | 128905/400000 [00:17<00:35, 7594.76it/s] 32%|███▏      | 129692/400000 [00:17<00:35, 7675.19it/s] 33%|███▎      | 130461/400000 [00:17<00:36, 7453.12it/s] 33%|███▎      | 131211/400000 [00:17<00:36, 7466.02it/s] 33%|███▎      | 131967/400000 [00:18<00:35, 7493.79it/s] 33%|███▎      | 132733/400000 [00:18<00:35, 7541.69it/s] 33%|███▎      | 133501/400000 [00:18<00:35, 7581.10it/s] 34%|███▎      | 134266/400000 [00:18<00:34, 7601.36it/s] 34%|███▍      | 135037/400000 [00:18<00:34, 7632.27it/s] 34%|███▍      | 135801/400000 [00:18<00:34, 7614.77it/s] 34%|███▍      | 136563/400000 [00:18<00:35, 7441.43it/s] 34%|███▍      | 137309/400000 [00:18<00:35, 7408.22it/s] 35%|███▍      | 138051/400000 [00:18<00:35, 7399.20it/s] 35%|███▍      | 138798/400000 [00:19<00:35, 7417.56it/s] 35%|███▍      | 139555/400000 [00:19<00:34, 7458.95it/s] 35%|███▌      | 140322/400000 [00:19<00:34, 7519.22it/s] 35%|███▌      | 141075/400000 [00:19<00:35, 7218.04it/s] 35%|███▌      | 141845/400000 [00:19<00:35, 7355.91it/s] 36%|███▌      | 142613/400000 [00:19<00:34, 7447.48it/s] 36%|███▌      | 143376/400000 [00:19<00:34, 7498.68it/s] 36%|███▌      | 144143/400000 [00:19<00:33, 7548.19it/s] 36%|███▌      | 144912/400000 [00:19<00:33, 7588.29it/s] 36%|███▋      | 145677/400000 [00:19<00:33, 7606.41it/s] 37%|███▋      | 146439/400000 [00:20<00:34, 7327.74it/s] 37%|███▋      | 147220/400000 [00:20<00:33, 7465.03it/s] 37%|███▋      | 147984/400000 [00:20<00:33, 7513.37it/s] 37%|███▋      | 148758/400000 [00:20<00:33, 7578.57it/s] 37%|███▋      | 149545/400000 [00:20<00:32, 7661.75it/s] 38%|███▊      | 150313/400000 [00:20<00:32, 7606.89it/s] 38%|███▊      | 151075/400000 [00:20<00:33, 7507.72it/s] 38%|███▊      | 151839/400000 [00:20<00:32, 7546.69it/s] 38%|███▊      | 152612/400000 [00:20<00:32, 7600.31it/s] 38%|███▊      | 153388/400000 [00:20<00:32, 7646.62it/s] 39%|███▊      | 154154/400000 [00:21<00:32, 7545.99it/s] 39%|███▊      | 154934/400000 [00:21<00:32, 7619.79it/s] 39%|███▉      | 155697/400000 [00:21<00:33, 7402.44it/s] 39%|███▉      | 156440/400000 [00:21<00:32, 7393.32it/s] 39%|███▉      | 157181/400000 [00:21<00:33, 7335.05it/s] 39%|███▉      | 157942/400000 [00:21<00:32, 7413.98it/s] 40%|███▉      | 158710/400000 [00:21<00:32, 7491.31it/s] 40%|███▉      | 159469/400000 [00:21<00:31, 7520.02it/s] 40%|████      | 160235/400000 [00:21<00:31, 7561.30it/s] 40%|████      | 161017/400000 [00:21<00:31, 7634.65it/s] 40%|████      | 161781/400000 [00:22<00:31, 7628.92it/s] 41%|████      | 162547/400000 [00:22<00:31, 7637.12it/s] 41%|████      | 163317/400000 [00:22<00:30, 7653.13it/s] 41%|████      | 164088/400000 [00:22<00:30, 7668.97it/s] 41%|████      | 164856/400000 [00:22<00:31, 7383.30it/s] 41%|████▏     | 165620/400000 [00:22<00:31, 7457.12it/s] 42%|████▏     | 166397/400000 [00:22<00:30, 7546.03it/s] 42%|████▏     | 167174/400000 [00:22<00:30, 7610.45it/s] 42%|████▏     | 167939/400000 [00:22<00:30, 7621.21it/s] 42%|████▏     | 168713/400000 [00:22<00:30, 7655.40it/s] 42%|████▏     | 169480/400000 [00:23<00:30, 7658.79it/s] 43%|████▎     | 170253/400000 [00:23<00:29, 7678.69it/s] 43%|████▎     | 171038/400000 [00:23<00:29, 7728.95it/s] 43%|████▎     | 171812/400000 [00:23<00:30, 7523.14it/s] 43%|████▎     | 172571/400000 [00:23<00:30, 7541.38it/s] 43%|████▎     | 173329/400000 [00:23<00:30, 7552.84it/s] 44%|████▎     | 174085/400000 [00:23<00:30, 7397.84it/s] 44%|████▎     | 174845/400000 [00:23<00:30, 7456.87it/s] 44%|████▍     | 175620/400000 [00:23<00:29, 7542.09it/s] 44%|████▍     | 176393/400000 [00:23<00:29, 7596.10it/s] 44%|████▍     | 177154/400000 [00:24<00:29, 7550.11it/s] 44%|████▍     | 177920/400000 [00:24<00:29, 7582.39it/s] 45%|████▍     | 178679/400000 [00:24<00:30, 7373.72it/s] 45%|████▍     | 179444/400000 [00:24<00:29, 7443.87it/s] 45%|████▌     | 180202/400000 [00:24<00:29, 7482.74it/s] 45%|████▌     | 180964/400000 [00:24<00:29, 7520.75it/s] 45%|████▌     | 181717/400000 [00:24<00:29, 7519.79it/s] 46%|████▌     | 182493/400000 [00:24<00:28, 7587.91it/s] 46%|████▌     | 183253/400000 [00:24<00:28, 7573.51it/s] 46%|████▌     | 184037/400000 [00:25<00:28, 7650.66it/s] 46%|████▌     | 184803/400000 [00:25<00:28, 7547.85it/s] 46%|████▋     | 185559/400000 [00:25<00:28, 7419.27it/s] 47%|████▋     | 186312/400000 [00:25<00:28, 7451.48it/s] 47%|████▋     | 187087/400000 [00:25<00:28, 7537.65it/s] 47%|████▋     | 187865/400000 [00:25<00:27, 7606.29it/s] 47%|████▋     | 188636/400000 [00:25<00:27, 7635.80it/s] 47%|████▋     | 189401/400000 [00:25<00:27, 7635.17it/s] 48%|████▊     | 190165/400000 [00:25<00:27, 7557.52it/s] 48%|████▊     | 190922/400000 [00:25<00:28, 7352.19it/s] 48%|████▊     | 191659/400000 [00:26<00:29, 7162.90it/s] 48%|████▊     | 192378/400000 [00:26<00:29, 7046.57it/s] 48%|████▊     | 193107/400000 [00:26<00:29, 7117.83it/s] 48%|████▊     | 193875/400000 [00:26<00:28, 7276.80it/s] 49%|████▊     | 194648/400000 [00:26<00:27, 7405.82it/s] 49%|████▉     | 195422/400000 [00:26<00:27, 7500.85it/s] 49%|████▉     | 196194/400000 [00:26<00:26, 7564.95it/s] 49%|████▉     | 196967/400000 [00:26<00:26, 7611.34it/s] 49%|████▉     | 197730/400000 [00:26<00:26, 7543.30it/s] 50%|████▉     | 198496/400000 [00:26<00:26, 7576.09it/s] 50%|████▉     | 199255/400000 [00:27<00:26, 7576.40it/s] 50%|█████     | 200018/400000 [00:27<00:26, 7590.05it/s] 50%|█████     | 200792/400000 [00:27<00:26, 7632.35it/s] 50%|█████     | 201571/400000 [00:27<00:25, 7677.80it/s] 51%|█████     | 202351/400000 [00:27<00:25, 7713.68it/s] 51%|█████     | 203127/400000 [00:27<00:25, 7726.93it/s] 51%|█████     | 203900/400000 [00:27<00:25, 7663.02it/s] 51%|█████     | 204667/400000 [00:27<00:25, 7552.96it/s] 51%|█████▏    | 205431/400000 [00:27<00:25, 7578.18it/s] 52%|█████▏    | 206206/400000 [00:27<00:25, 7628.63it/s] 52%|█████▏    | 206970/400000 [00:28<00:25, 7620.61it/s] 52%|█████▏    | 207733/400000 [00:28<00:25, 7619.09it/s] 52%|█████▏    | 208500/400000 [00:28<00:25, 7631.89it/s] 52%|█████▏    | 209280/400000 [00:28<00:24, 7681.32it/s] 53%|█████▎    | 210049/400000 [00:28<00:24, 7665.91it/s] 53%|█████▎    | 210816/400000 [00:28<00:24, 7638.25it/s] 53%|█████▎    | 211580/400000 [00:28<00:24, 7613.67it/s] 53%|█████▎    | 212357/400000 [00:28<00:24, 7659.79it/s] 53%|█████▎    | 213124/400000 [00:28<00:24, 7633.63it/s] 53%|█████▎    | 213898/400000 [00:28<00:24, 7663.98it/s] 54%|█████▎    | 214675/400000 [00:29<00:24, 7692.96it/s] 54%|█████▍    | 215445/400000 [00:29<00:24, 7687.71it/s] 54%|█████▍    | 216214/400000 [00:29<00:24, 7573.83it/s] 54%|█████▍    | 216972/400000 [00:29<00:24, 7567.31it/s] 54%|█████▍    | 217745/400000 [00:29<00:23, 7613.59it/s] 55%|█████▍    | 218520/400000 [00:29<00:23, 7652.72it/s] 55%|█████▍    | 219290/400000 [00:29<00:23, 7664.99it/s] 55%|█████▌    | 220072/400000 [00:29<00:23, 7707.74it/s] 55%|█████▌    | 220843/400000 [00:29<00:23, 7699.47it/s] 55%|█████▌    | 221614/400000 [00:29<00:24, 7374.09it/s] 56%|█████▌    | 222355/400000 [00:30<00:24, 7308.73it/s] 56%|█████▌    | 223120/400000 [00:30<00:23, 7406.86it/s] 56%|█████▌    | 223943/400000 [00:30<00:23, 7635.54it/s] 56%|█████▌    | 224802/400000 [00:30<00:22, 7896.66it/s] 56%|█████▋    | 225656/400000 [00:30<00:21, 8078.83it/s] 57%|█████▋    | 226469/400000 [00:30<00:22, 7769.67it/s] 57%|█████▋    | 227269/400000 [00:30<00:22, 7836.14it/s] 57%|█████▋    | 228078/400000 [00:30<00:21, 7909.36it/s] 57%|█████▋    | 228884/400000 [00:30<00:21, 7952.55it/s] 57%|█████▋    | 229682/400000 [00:31<00:22, 7669.41it/s] 58%|█████▊    | 230470/400000 [00:31<00:21, 7728.54it/s] 58%|█████▊    | 231246/400000 [00:31<00:22, 7389.54it/s] 58%|█████▊    | 231990/400000 [00:31<00:23, 7145.53it/s] 58%|█████▊    | 232710/400000 [00:31<00:23, 7023.74it/s] 58%|█████▊    | 233484/400000 [00:31<00:23, 7222.67it/s] 59%|█████▊    | 234244/400000 [00:31<00:22, 7330.04it/s] 59%|█████▊    | 234991/400000 [00:31<00:22, 7368.13it/s] 59%|█████▉    | 235789/400000 [00:31<00:21, 7539.70it/s] 59%|█████▉    | 236632/400000 [00:31<00:20, 7784.69it/s] 59%|█████▉    | 237466/400000 [00:32<00:20, 7941.30it/s] 60%|█████▉    | 238264/400000 [00:32<00:21, 7686.39it/s] 60%|█████▉    | 239067/400000 [00:32<00:20, 7784.14it/s] 60%|█████▉    | 239870/400000 [00:32<00:20, 7853.82it/s] 60%|██████    | 240658/400000 [00:32<00:20, 7601.95it/s] 60%|██████    | 241520/400000 [00:32<00:20, 7879.53it/s] 61%|██████    | 242313/400000 [00:32<00:20, 7836.16it/s] 61%|██████    | 243101/400000 [00:32<00:20, 7476.68it/s] 61%|██████    | 243855/400000 [00:32<00:21, 7098.96it/s] 61%|██████    | 244609/400000 [00:33<00:21, 7223.51it/s] 61%|██████▏   | 245338/400000 [00:33<00:21, 7233.23it/s] 62%|██████▏   | 246114/400000 [00:33<00:20, 7383.02it/s] 62%|██████▏   | 246897/400000 [00:33<00:20, 7511.54it/s] 62%|██████▏   | 247653/400000 [00:33<00:20, 7525.87it/s] 62%|██████▏   | 248446/400000 [00:33<00:19, 7640.36it/s] 62%|██████▏   | 249241/400000 [00:33<00:19, 7730.51it/s] 63%|██████▎   | 250059/400000 [00:33<00:19, 7858.78it/s] 63%|██████▎   | 250847/400000 [00:33<00:19, 7608.78it/s] 63%|██████▎   | 251661/400000 [00:33<00:19, 7760.69it/s] 63%|██████▎   | 252511/400000 [00:34<00:18, 7965.53it/s] 63%|██████▎   | 253311/400000 [00:34<00:18, 7883.52it/s] 64%|██████▎   | 254102/400000 [00:34<00:19, 7606.70it/s] 64%|██████▎   | 254867/400000 [00:34<00:19, 7478.81it/s] 64%|██████▍   | 255619/400000 [00:34<00:19, 7407.55it/s] 64%|██████▍   | 256474/400000 [00:34<00:18, 7716.44it/s] 64%|██████▍   | 257359/400000 [00:34<00:17, 8024.54it/s] 65%|██████▍   | 258169/400000 [00:34<00:18, 7515.18it/s] 65%|██████▍   | 258958/400000 [00:34<00:18, 7623.46it/s] 65%|██████▍   | 259813/400000 [00:34<00:17, 7879.11it/s] 65%|██████▌   | 260650/400000 [00:35<00:17, 8018.63it/s] 65%|██████▌   | 261459/400000 [00:35<00:18, 7571.34it/s] 66%|██████▌   | 262226/400000 [00:35<00:18, 7355.20it/s] 66%|██████▌   | 262988/400000 [00:35<00:18, 7430.93it/s] 66%|██████▌   | 263834/400000 [00:35<00:17, 7710.61it/s] 66%|██████▌   | 264612/400000 [00:35<00:18, 7326.76it/s] 66%|██████▋   | 265354/400000 [00:35<00:19, 7065.63it/s] 67%|██████▋   | 266170/400000 [00:35<00:18, 7360.35it/s] 67%|██████▋   | 266981/400000 [00:35<00:17, 7568.72it/s] 67%|██████▋   | 267774/400000 [00:36<00:17, 7672.59it/s] 67%|██████▋   | 268547/400000 [00:36<00:17, 7518.20it/s] 67%|██████▋   | 269371/400000 [00:36<00:16, 7720.29it/s] 68%|██████▊   | 270156/400000 [00:36<00:16, 7757.30it/s] 68%|██████▊   | 270936/400000 [00:36<00:17, 7506.59it/s] 68%|██████▊   | 271691/400000 [00:36<00:17, 7338.21it/s] 68%|██████▊   | 272429/400000 [00:36<00:17, 7192.63it/s] 68%|██████▊   | 273220/400000 [00:36<00:17, 7392.93it/s] 69%|██████▊   | 274085/400000 [00:36<00:16, 7727.73it/s] 69%|██████▊   | 274865/400000 [00:36<00:16, 7631.01it/s] 69%|██████▉   | 275633/400000 [00:37<00:16, 7513.79it/s] 69%|██████▉   | 276436/400000 [00:37<00:16, 7659.99it/s] 69%|██████▉   | 277206/400000 [00:37<00:16, 7511.77it/s] 70%|██████▉   | 278027/400000 [00:37<00:15, 7707.90it/s] 70%|██████▉   | 278840/400000 [00:37<00:15, 7827.85it/s] 70%|██████▉   | 279629/400000 [00:37<00:15, 7845.57it/s] 70%|███████   | 280416/400000 [00:37<00:15, 7642.07it/s] 70%|███████   | 281189/400000 [00:37<00:15, 7666.20it/s] 71%|███████   | 282006/400000 [00:37<00:15, 7808.98it/s] 71%|███████   | 282795/400000 [00:38<00:14, 7830.07it/s] 71%|███████   | 283580/400000 [00:38<00:15, 7746.87it/s] 71%|███████   | 284356/400000 [00:38<00:15, 7679.56it/s] 71%|███████▏  | 285140/400000 [00:38<00:14, 7724.53it/s] 71%|███████▏  | 285947/400000 [00:38<00:14, 7824.15it/s] 72%|███████▏  | 286838/400000 [00:38<00:13, 8119.27it/s] 72%|███████▏  | 287704/400000 [00:38<00:13, 8273.54it/s] 72%|███████▏  | 288535/400000 [00:38<00:13, 8281.92it/s] 72%|███████▏  | 289366/400000 [00:38<00:13, 8236.66it/s] 73%|███████▎  | 290192/400000 [00:38<00:14, 7727.67it/s] 73%|███████▎  | 290973/400000 [00:39<00:14, 7398.84it/s] 73%|███████▎  | 291722/400000 [00:39<00:15, 7156.56it/s] 73%|███████▎  | 292446/400000 [00:39<00:15, 6885.26it/s] 73%|███████▎  | 293142/400000 [00:39<00:15, 6732.47it/s] 73%|███████▎  | 293902/400000 [00:39<00:15, 6968.96it/s] 74%|███████▎  | 294686/400000 [00:39<00:14, 7207.29it/s] 74%|███████▍  | 295414/400000 [00:39<00:14, 7072.62it/s] 74%|███████▍  | 296127/400000 [00:39<00:15, 6848.62it/s] 74%|███████▍  | 296817/400000 [00:39<00:15, 6654.82it/s] 74%|███████▍  | 297488/400000 [00:40<00:15, 6670.73it/s] 75%|███████▍  | 298289/400000 [00:40<00:14, 7021.67it/s] 75%|███████▍  | 298999/400000 [00:40<00:14, 6960.84it/s] 75%|███████▍  | 299701/400000 [00:40<00:14, 6725.79it/s] 75%|███████▌  | 300486/400000 [00:40<00:14, 7025.33it/s] 75%|███████▌  | 301196/400000 [00:40<00:14, 7035.56it/s] 75%|███████▌  | 301970/400000 [00:40<00:13, 7230.48it/s] 76%|███████▌  | 302698/400000 [00:40<00:13, 7182.63it/s] 76%|███████▌  | 303441/400000 [00:40<00:13, 7253.90it/s] 76%|███████▌  | 304307/400000 [00:40<00:12, 7623.84it/s] 76%|███████▋  | 305077/400000 [00:41<00:12, 7623.26it/s] 76%|███████▋  | 305845/400000 [00:41<00:12, 7576.10it/s] 77%|███████▋  | 306623/400000 [00:41<00:12, 7635.08it/s] 77%|███████▋  | 307390/400000 [00:41<00:12, 7619.02it/s] 77%|███████▋  | 308194/400000 [00:41<00:11, 7739.79it/s] 77%|███████▋  | 308970/400000 [00:41<00:11, 7688.84it/s] 77%|███████▋  | 309751/400000 [00:41<00:11, 7724.58it/s] 78%|███████▊  | 310567/400000 [00:41<00:11, 7848.46it/s] 78%|███████▊  | 311356/400000 [00:41<00:11, 7856.50it/s] 78%|███████▊  | 312143/400000 [00:41<00:11, 7675.50it/s] 78%|███████▊  | 312934/400000 [00:42<00:11, 7742.23it/s] 78%|███████▊  | 313710/400000 [00:42<00:11, 7714.31it/s] 79%|███████▊  | 314505/400000 [00:42<00:10, 7780.29it/s] 79%|███████▉  | 315291/400000 [00:42<00:10, 7802.59it/s] 79%|███████▉  | 316072/400000 [00:42<00:10, 7690.00it/s] 79%|███████▉  | 316842/400000 [00:42<00:10, 7655.27it/s] 79%|███████▉  | 317642/400000 [00:42<00:10, 7754.68it/s] 80%|███████▉  | 318500/400000 [00:42<00:10, 7983.31it/s] 80%|███████▉  | 319301/400000 [00:42<00:10, 7925.47it/s] 80%|████████  | 320096/400000 [00:42<00:10, 7679.49it/s] 80%|████████  | 320867/400000 [00:43<00:10, 7505.79it/s] 80%|████████  | 321656/400000 [00:43<00:10, 7615.06it/s] 81%|████████  | 322454/400000 [00:43<00:10, 7718.46it/s] 81%|████████  | 323259/400000 [00:43<00:09, 7813.82it/s] 81%|████████  | 324053/400000 [00:43<00:09, 7850.98it/s] 81%|████████  | 324840/400000 [00:43<00:09, 7596.24it/s] 81%|████████▏ | 325605/400000 [00:43<00:09, 7610.43it/s] 82%|████████▏ | 326464/400000 [00:43<00:09, 7878.34it/s] 82%|████████▏ | 327256/400000 [00:43<00:09, 7886.78it/s] 82%|████████▏ | 328092/400000 [00:43<00:08, 8022.20it/s] 82%|████████▏ | 328897/400000 [00:44<00:09, 7857.52it/s] 82%|████████▏ | 329686/400000 [00:44<00:09, 7589.25it/s] 83%|████████▎ | 330449/400000 [00:44<00:09, 7300.58it/s] 83%|████████▎ | 331184/400000 [00:44<00:09, 7048.52it/s] 83%|████████▎ | 331922/400000 [00:44<00:09, 7143.10it/s] 83%|████████▎ | 332689/400000 [00:44<00:09, 7282.80it/s] 83%|████████▎ | 333421/400000 [00:44<00:09, 7250.11it/s] 84%|████████▎ | 334149/400000 [00:44<00:09, 7042.62it/s] 84%|████████▎ | 334857/400000 [00:44<00:09, 7023.80it/s] 84%|████████▍ | 335594/400000 [00:45<00:09, 7123.08it/s] 84%|████████▍ | 336313/400000 [00:45<00:08, 7141.66it/s] 84%|████████▍ | 337061/400000 [00:45<00:08, 7237.37it/s] 84%|████████▍ | 337843/400000 [00:45<00:08, 7369.37it/s] 85%|████████▍ | 338582/400000 [00:45<00:08, 7373.36it/s] 85%|████████▍ | 339394/400000 [00:45<00:07, 7581.32it/s] 85%|████████▌ | 340155/400000 [00:45<00:08, 7328.76it/s] 85%|████████▌ | 340892/400000 [00:45<00:08, 6802.10it/s] 85%|████████▌ | 341688/400000 [00:45<00:08, 7112.20it/s] 86%|████████▌ | 342510/400000 [00:45<00:07, 7410.83it/s] 86%|████████▌ | 343262/400000 [00:46<00:07, 7242.38it/s] 86%|████████▌ | 343995/400000 [00:46<00:07, 7233.86it/s] 86%|████████▌ | 344767/400000 [00:46<00:07, 7371.68it/s] 86%|████████▋ | 345541/400000 [00:46<00:07, 7472.55it/s] 87%|████████▋ | 346331/400000 [00:46<00:07, 7594.05it/s] 87%|████████▋ | 347094/400000 [00:46<00:06, 7573.98it/s] 87%|████████▋ | 347854/400000 [00:46<00:07, 7441.27it/s] 87%|████████▋ | 348601/400000 [00:46<00:06, 7414.42it/s] 87%|████████▋ | 349344/400000 [00:46<00:07, 7195.75it/s] 88%|████████▊ | 350145/400000 [00:47<00:06, 7420.36it/s] 88%|████████▊ | 351027/400000 [00:47<00:06, 7790.31it/s] 88%|████████▊ | 351832/400000 [00:47<00:06, 7863.78it/s] 88%|████████▊ | 352624/400000 [00:47<00:06, 7664.45it/s] 88%|████████▊ | 353460/400000 [00:47<00:05, 7858.26it/s] 89%|████████▊ | 354332/400000 [00:47<00:05, 8097.13it/s] 89%|████████▉ | 355207/400000 [00:47<00:05, 8282.48it/s] 89%|████████▉ | 356040/400000 [00:47<00:05, 8164.18it/s] 89%|████████▉ | 356860/400000 [00:47<00:05, 7904.48it/s] 89%|████████▉ | 357695/400000 [00:47<00:05, 8032.41it/s] 90%|████████▉ | 358555/400000 [00:48<00:05, 8194.10it/s] 90%|████████▉ | 359436/400000 [00:48<00:04, 8367.67it/s] 90%|█████████ | 360296/400000 [00:48<00:04, 8435.66it/s] 90%|█████████ | 361165/400000 [00:48<00:04, 8508.13it/s] 91%|█████████ | 362018/400000 [00:48<00:04, 8171.11it/s] 91%|█████████ | 362840/400000 [00:48<00:04, 7794.60it/s] 91%|█████████ | 363716/400000 [00:48<00:04, 8059.47it/s] 91%|█████████ | 364555/400000 [00:48<00:04, 8154.54it/s] 91%|█████████▏| 365426/400000 [00:48<00:04, 8311.48it/s] 92%|█████████▏| 366270/400000 [00:48<00:04, 8349.57it/s] 92%|█████████▏| 367132/400000 [00:49<00:03, 8428.77it/s] 92%|█████████▏| 367978/400000 [00:49<00:03, 8185.38it/s] 92%|█████████▏| 368800/400000 [00:49<00:03, 8026.02it/s] 92%|█████████▏| 369632/400000 [00:49<00:03, 8111.38it/s] 93%|█████████▎| 370475/400000 [00:49<00:03, 8203.60it/s] 93%|█████████▎| 371339/400000 [00:49<00:03, 8328.81it/s] 93%|█████████▎| 372216/400000 [00:49<00:03, 8455.96it/s] 93%|█████████▎| 373066/400000 [00:49<00:03, 8468.15it/s] 93%|█████████▎| 373915/400000 [00:49<00:03, 7926.70it/s] 94%|█████████▎| 374716/400000 [00:50<00:03, 7319.09it/s] 94%|█████████▍| 375463/400000 [00:50<00:03, 7224.29it/s] 94%|█████████▍| 376348/400000 [00:50<00:03, 7645.60it/s] 94%|█████████▍| 377231/400000 [00:50<00:02, 7964.18it/s] 95%|█████████▍| 378109/400000 [00:50<00:02, 8190.69it/s] 95%|█████████▍| 378940/400000 [00:50<00:02, 8224.44it/s] 95%|█████████▍| 379796/400000 [00:50<00:02, 8322.20it/s] 95%|█████████▌| 380668/400000 [00:50<00:02, 8436.79it/s] 95%|█████████▌| 381517/400000 [00:50<00:02, 8151.71it/s] 96%|█████████▌| 382338/400000 [00:51<00:02, 7455.69it/s] 96%|█████████▌| 383126/400000 [00:51<00:02, 7577.36it/s] 96%|█████████▌| 383996/400000 [00:51<00:02, 7881.66it/s] 96%|█████████▌| 384879/400000 [00:51<00:01, 8142.24it/s] 96%|█████████▋| 385704/400000 [00:51<00:01, 8029.88it/s] 97%|█████████▋| 386558/400000 [00:51<00:01, 8175.55it/s] 97%|█████████▋| 387382/400000 [00:51<00:01, 8111.44it/s] 97%|█████████▋| 388225/400000 [00:51<00:01, 8203.02it/s] 97%|█████████▋| 389081/400000 [00:51<00:01, 8306.10it/s] 97%|█████████▋| 389915/400000 [00:51<00:01, 8302.86it/s] 98%|█████████▊| 390748/400000 [00:52<00:01, 7833.75it/s] 98%|█████████▊| 391539/400000 [00:52<00:01, 7326.14it/s] 98%|█████████▊| 392283/400000 [00:52<00:01, 7062.69it/s] 98%|█████████▊| 393116/400000 [00:52<00:00, 7400.41it/s] 98%|█████████▊| 393926/400000 [00:52<00:00, 7595.34it/s] 99%|█████████▊| 394695/400000 [00:52<00:00, 7451.47it/s] 99%|█████████▉| 395448/400000 [00:52<00:00, 7174.71it/s] 99%|█████████▉| 396221/400000 [00:52<00:00, 7331.60it/s] 99%|█████████▉| 397046/400000 [00:52<00:00, 7584.39it/s] 99%|█████████▉| 397811/400000 [00:52<00:00, 7583.03it/s]100%|█████████▉| 398651/400000 [00:53<00:00, 7810.30it/s]100%|█████████▉| 399528/400000 [00:53<00:00, 8073.94it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7511.58it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f62f3e5da58> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01131455956269402 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.010703669543250348 	 Accuracy: 76

  model saves at 76% accuracy 

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
2020-05-16 03:31:13.114459: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 03:31:13.119004: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397220000 Hz
2020-05-16 03:31:13.119181: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5570ef9c4cd0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 03:31:13.119198: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f62f7859e80> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.7586 - accuracy: 0.4940
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6666 - accuracy: 0.5000 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6411 - accuracy: 0.5017
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6114 - accuracy: 0.5036
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6155 - accuracy: 0.5033
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6360 - accuracy: 0.5020
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6398 - accuracy: 0.5017
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6394 - accuracy: 0.5018
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
11000/25000 [============>.................] - ETA: 4s - loss: 7.6680 - accuracy: 0.4999
12000/25000 [=============>................] - ETA: 3s - loss: 7.6730 - accuracy: 0.4996
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6560 - accuracy: 0.5007
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6644 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 2s - loss: 7.6799 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6676 - accuracy: 0.4999
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6576 - accuracy: 0.5006
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6394 - accuracy: 0.5018
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6473 - accuracy: 0.5013
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6452 - accuracy: 0.5014
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6498 - accuracy: 0.5011
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6534 - accuracy: 0.5009
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6634 - accuracy: 0.5002
25000/25000 [==============================] - 9s 356us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f624aec2668> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f6270ea4160> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 991ms/step - loss: 1.3728 - crf_viterbi_accuracy: 0.0267 - val_loss: 1.3027 - val_crf_viterbi_accuracy: 0.0000e+00

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

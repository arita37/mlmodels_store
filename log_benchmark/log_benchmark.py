
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '69b309ad857428cc5a734b8afd99842edf9b2a42', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/69b309ad857428cc5a734b8afd99842edf9b2a42

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7ff83d909e80> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-20 04:14:40.771602
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-20 04:14:40.777004
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-20 04:14:40.780742
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-20 04:14:40.784811
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7ff8496d3358> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356275.1250
Epoch 2/10

1/1 [==============================] - 0s 122ms/step - loss: 247679.1875
Epoch 3/10

1/1 [==============================] - 0s 107ms/step - loss: 153059.4531
Epoch 4/10

1/1 [==============================] - 0s 105ms/step - loss: 87808.2969
Epoch 5/10

1/1 [==============================] - 0s 105ms/step - loss: 50308.3711
Epoch 6/10

1/1 [==============================] - 0s 130ms/step - loss: 30393.0996
Epoch 7/10

1/1 [==============================] - 0s 104ms/step - loss: 19568.9355
Epoch 8/10

1/1 [==============================] - 0s 105ms/step - loss: 13396.2871
Epoch 9/10

1/1 [==============================] - 0s 107ms/step - loss: 9637.6807
Epoch 10/10

1/1 [==============================] - 0s 106ms/step - loss: 7366.3608

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.08578902  0.3398828   1.1533302  -0.15230295 -0.12908632  1.4561757
   1.2054875   0.35186976  0.4214359   1.4961929  -0.380776    0.924358
  -1.191607   -0.88177407  0.31543267  0.666868    0.3639558  -0.6656794
  -1.546481   -0.03486743  0.59906864 -0.5511316  -0.81232417 -0.6475199
   1.0708243  -1.2116097  -1.1879519  -1.4805729   0.9258907  -0.7864071
   0.02382186 -0.69599533  1.1507258   0.3021662   0.3007241   0.1636635
  -0.44792286  0.43476775 -1.6315327   0.664472   -0.01610412 -0.43299526
   2.0234272  -0.27239734 -0.84789157 -0.03445205  0.923607    0.3665066
   0.03040123 -0.9427828   1.0587765   0.3668631  -0.57196784 -1.2636735
   0.38338587  0.47094378 -0.18459523  0.7115534   1.0012715   0.83879375
   0.08493373  8.6630945   8.489023    7.790143    8.245792    6.579951
   6.9550295   8.271502    7.438119    7.939581    8.29262     7.184819
   8.216586    8.23116     7.4443026   7.928667    6.8152914   6.739134
   7.2668934   8.444289    7.805358    6.3405137   6.448969    7.382406
   7.3285093   6.900135    9.791288    7.777872    7.289386    8.145676
   7.6316743   7.8992887   8.104193    8.1288805   8.521859    7.8738475
   7.8248944   7.677412    7.4440837   8.0147      7.7449045   7.4520645
   7.819584    7.276929    7.805997    8.061792    7.7246003   6.3867717
   7.8674874   6.997418    8.042226    5.621865    7.7476277   8.272856
   8.199145    6.019612    8.007061    8.47958     8.640753    5.481789
   0.02331895 -0.01519883  1.1878734   0.9933523  -0.69196916  0.650191
  -0.01483122 -1.6503003  -0.4365412   1.3357595   0.2478643   0.7624557
   0.6475395   0.2365528  -1.1197015   0.04543009 -0.749338   -1.6177361
   0.03056373  0.04283816  0.2669049  -0.70698327  0.47268385  0.9447164
  -1.0559456  -0.13667114  0.03089499 -0.36823738 -1.0795766   0.32827777
   1.2074511  -1.1243762  -1.2208276  -0.2971846  -0.24321072 -0.7915438
  -0.20722234 -0.67002916 -1.8936939   0.22164169  0.27672336  2.1358204
   0.8770272  -0.7005142   0.7185636  -0.15500742 -1.3584034   0.6758714
   0.32834908 -1.3229055  -1.2857759  -1.0707469  -0.4011563  -0.7827173
  -1.2683473  -0.1342882  -1.9692055  -0.4145585   0.13322166  1.3935144
   0.42025495  1.5438628   0.19119197  0.78008515  0.4593985   0.75255835
   0.14441013  2.2530742   0.79649436  1.2727147   2.07354     0.6363764
   0.33742642  2.6046672   0.76265025  1.2741973   1.7872272   1.1580267
   1.4590021   1.1648206   1.6909955   1.8739152   0.5901503   1.083253
   2.2386553   0.20838511  0.3244868   0.6287286   0.62495637  1.136834
   2.022633    0.432576    0.36639357  0.61096275  1.9064856   1.4516909
   0.9725389   1.437266    0.49602693  2.1400778   1.2705376   0.7357255
   0.57850796  1.6186612   1.1104203   0.57535934  0.7797862   2.4833622
   1.4260885   0.22256738  0.7013722   1.0280584   0.18363273  1.1901628
   1.3599479   0.85084873  1.0488309   2.1429157   1.1951139   2.644192
   0.04101324  9.14401     7.7559347   8.456945    8.295976    7.5907946
   7.7431817   7.5058107   8.05849     7.5204616   8.557001    7.4581933
   7.8847127   8.660822    8.561937    7.766037    7.029397    7.9745297
   7.983852    7.0310335   6.5265126   8.504322    9.027438    7.6912417
   9.13324     8.07988     8.231619    8.011652    6.1299796   9.1484585
   8.872762    8.51719     9.523811    8.388928    7.0959754   7.6606746
   7.1083755   8.68934     8.113321    6.743719    6.822871    7.673247
   8.301182    7.0767846   9.066456    8.869811    7.3617454  10.231404
   7.0741262   8.7117605   8.449638    8.2342825   8.361051    8.690955
   7.8003144   8.4921055   7.6973777   7.748707    7.0410995   7.7552834
   0.193236    0.38985837  0.26934457  0.24254727  2.0833585   0.4967221
   1.7404611   0.2829067   0.796049    0.9597951   1.3474118   1.6071627
   2.8929033   1.1776986   2.1143334   2.5368652   1.0767155   0.3229046
   0.6349087   1.9363703   2.223174    1.9307225   1.4556882   0.20505345
   1.3469307   1.3664551   0.18326598  0.30160958  1.4395906   0.45436525
   2.4295733   0.3723644   1.3260727   0.28146124  0.41811472  1.6292483
   1.3478005   0.7620241   1.6717336   1.1588378   0.7368678   0.38795948
   0.8676557   1.8696423   0.25492156  2.3007607   1.4449687   0.18870002
   0.3704881   0.40132952  1.2909472   1.8494928   2.320324    1.5549881
   1.7106752   1.1233585   0.2430625   2.1507235   1.137217    2.2407455
  -6.0613117   8.245914   -6.4896584 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-20 04:14:50.473646
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.9646
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-20 04:14:50.478394
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8854.05
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-20 04:14:50.482372
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.4216
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-20 04:14:50.486338
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -791.943
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140703819346384
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140702860902920
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140702860903424
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140702860903928
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140702860904432
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140702860904936

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7ff845554dd8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.733894
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.699896
grad_step = 000002, loss = 0.677143
grad_step = 000003, loss = 0.654797
grad_step = 000004, loss = 0.630748
grad_step = 000005, loss = 0.603946
grad_step = 000006, loss = 0.576623
grad_step = 000007, loss = 0.552263
grad_step = 000008, loss = 0.535321
grad_step = 000009, loss = 0.519697
grad_step = 000010, loss = 0.503190
grad_step = 000011, loss = 0.484187
grad_step = 000012, loss = 0.467080
grad_step = 000013, loss = 0.454254
grad_step = 000014, loss = 0.444090
grad_step = 000015, loss = 0.433805
grad_step = 000016, loss = 0.422362
grad_step = 000017, loss = 0.410844
grad_step = 000018, loss = 0.400335
grad_step = 000019, loss = 0.390201
grad_step = 000020, loss = 0.379203
grad_step = 000021, loss = 0.367731
grad_step = 000022, loss = 0.356540
grad_step = 000023, loss = 0.345966
grad_step = 000024, loss = 0.336102
grad_step = 000025, loss = 0.326464
grad_step = 000026, loss = 0.316615
grad_step = 000027, loss = 0.306926
grad_step = 000028, loss = 0.297914
grad_step = 000029, loss = 0.288471
grad_step = 000030, loss = 0.278475
grad_step = 000031, loss = 0.268391
grad_step = 000032, loss = 0.258895
grad_step = 000033, loss = 0.250336
grad_step = 000034, loss = 0.243006
grad_step = 000035, loss = 0.235356
grad_step = 000036, loss = 0.227141
grad_step = 000037, loss = 0.219187
grad_step = 000038, loss = 0.211728
grad_step = 000039, loss = 0.204592
grad_step = 000040, loss = 0.197723
grad_step = 000041, loss = 0.190952
grad_step = 000042, loss = 0.184246
grad_step = 000043, loss = 0.177741
grad_step = 000044, loss = 0.171524
grad_step = 000045, loss = 0.165685
grad_step = 000046, loss = 0.160039
grad_step = 000047, loss = 0.154320
grad_step = 000048, loss = 0.148714
grad_step = 000049, loss = 0.143341
grad_step = 000050, loss = 0.138168
grad_step = 000051, loss = 0.133317
grad_step = 000052, loss = 0.128734
grad_step = 000053, loss = 0.124249
grad_step = 000054, loss = 0.119864
grad_step = 000055, loss = 0.115626
grad_step = 000056, loss = 0.111553
grad_step = 000057, loss = 0.107591
grad_step = 000058, loss = 0.103769
grad_step = 000059, loss = 0.100170
grad_step = 000060, loss = 0.096708
grad_step = 000061, loss = 0.093355
grad_step = 000062, loss = 0.090134
grad_step = 000063, loss = 0.087026
grad_step = 000064, loss = 0.084060
grad_step = 000065, loss = 0.081211
grad_step = 000066, loss = 0.078446
grad_step = 000067, loss = 0.075775
grad_step = 000068, loss = 0.073230
grad_step = 000069, loss = 0.070797
grad_step = 000070, loss = 0.068433
grad_step = 000071, loss = 0.066170
grad_step = 000072, loss = 0.063996
grad_step = 000073, loss = 0.061890
grad_step = 000074, loss = 0.059868
grad_step = 000075, loss = 0.057928
grad_step = 000076, loss = 0.056059
grad_step = 000077, loss = 0.054254
grad_step = 000078, loss = 0.052511
grad_step = 000079, loss = 0.050830
grad_step = 000080, loss = 0.049220
grad_step = 000081, loss = 0.047662
grad_step = 000082, loss = 0.046158
grad_step = 000083, loss = 0.044709
grad_step = 000084, loss = 0.043304
grad_step = 000085, loss = 0.041947
grad_step = 000086, loss = 0.040641
grad_step = 000087, loss = 0.039376
grad_step = 000088, loss = 0.038151
grad_step = 000089, loss = 0.036966
grad_step = 000090, loss = 0.035821
grad_step = 000091, loss = 0.034714
grad_step = 000092, loss = 0.033641
grad_step = 000093, loss = 0.032604
grad_step = 000094, loss = 0.031597
grad_step = 000095, loss = 0.030621
grad_step = 000096, loss = 0.029677
grad_step = 000097, loss = 0.028763
grad_step = 000098, loss = 0.027877
grad_step = 000099, loss = 0.027018
grad_step = 000100, loss = 0.026185
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.025377
grad_step = 000102, loss = 0.024594
grad_step = 000103, loss = 0.023835
grad_step = 000104, loss = 0.023096
grad_step = 000105, loss = 0.022381
grad_step = 000106, loss = 0.021686
grad_step = 000107, loss = 0.021013
grad_step = 000108, loss = 0.020359
grad_step = 000109, loss = 0.019724
grad_step = 000110, loss = 0.019110
grad_step = 000111, loss = 0.018519
grad_step = 000112, loss = 0.017946
grad_step = 000113, loss = 0.017377
grad_step = 000114, loss = 0.016822
grad_step = 000115, loss = 0.016296
grad_step = 000116, loss = 0.015787
grad_step = 000117, loss = 0.015285
grad_step = 000118, loss = 0.014794
grad_step = 000119, loss = 0.014326
grad_step = 000120, loss = 0.013875
grad_step = 000121, loss = 0.013430
grad_step = 000122, loss = 0.012996
grad_step = 000123, loss = 0.012580
grad_step = 000124, loss = 0.012179
grad_step = 000125, loss = 0.011788
grad_step = 000126, loss = 0.011405
grad_step = 000127, loss = 0.011033
grad_step = 000128, loss = 0.010677
grad_step = 000129, loss = 0.010332
grad_step = 000130, loss = 0.009999
grad_step = 000131, loss = 0.009674
grad_step = 000132, loss = 0.009357
grad_step = 000133, loss = 0.009049
grad_step = 000134, loss = 0.008752
grad_step = 000135, loss = 0.008465
grad_step = 000136, loss = 0.008188
grad_step = 000137, loss = 0.007920
grad_step = 000138, loss = 0.007663
grad_step = 000139, loss = 0.007420
grad_step = 000140, loss = 0.007191
grad_step = 000141, loss = 0.006971
grad_step = 000142, loss = 0.006753
grad_step = 000143, loss = 0.006527
grad_step = 000144, loss = 0.006298
grad_step = 000145, loss = 0.006088
grad_step = 000146, loss = 0.005909
grad_step = 000147, loss = 0.005747
grad_step = 000148, loss = 0.005582
grad_step = 000149, loss = 0.005412
grad_step = 000150, loss = 0.005246
grad_step = 000151, loss = 0.005094
grad_step = 000152, loss = 0.004956
grad_step = 000153, loss = 0.004824
grad_step = 000154, loss = 0.004694
grad_step = 000155, loss = 0.004566
grad_step = 000156, loss = 0.004444
grad_step = 000157, loss = 0.004328
grad_step = 000158, loss = 0.004216
grad_step = 000159, loss = 0.004107
grad_step = 000160, loss = 0.004000
grad_step = 000161, loss = 0.003896
grad_step = 000162, loss = 0.003799
grad_step = 000163, loss = 0.003708
grad_step = 000164, loss = 0.003623
grad_step = 000165, loss = 0.003545
grad_step = 000166, loss = 0.003481
grad_step = 000167, loss = 0.003440
grad_step = 000168, loss = 0.003422
grad_step = 000169, loss = 0.003410
grad_step = 000170, loss = 0.003325
grad_step = 000171, loss = 0.003191
grad_step = 000172, loss = 0.003078
grad_step = 000173, loss = 0.003037
grad_step = 000174, loss = 0.003023
grad_step = 000175, loss = 0.002956
grad_step = 000176, loss = 0.002851
grad_step = 000177, loss = 0.002764
grad_step = 000178, loss = 0.002745
grad_step = 000179, loss = 0.002751
grad_step = 000180, loss = 0.002710
grad_step = 000181, loss = 0.002623
grad_step = 000182, loss = 0.002538
grad_step = 000183, loss = 0.002501
grad_step = 000184, loss = 0.002495
grad_step = 000185, loss = 0.002477
grad_step = 000186, loss = 0.002431
grad_step = 000187, loss = 0.002370
grad_step = 000188, loss = 0.002327
grad_step = 000189, loss = 0.002310
grad_step = 000190, loss = 0.002308
grad_step = 000191, loss = 0.002307
grad_step = 000192, loss = 0.002288
grad_step = 000193, loss = 0.002263
grad_step = 000194, loss = 0.002236
grad_step = 000195, loss = 0.002221
grad_step = 000196, loss = 0.002228
grad_step = 000197, loss = 0.002240
grad_step = 000198, loss = 0.002241
grad_step = 000199, loss = 0.002217
grad_step = 000200, loss = 0.002172
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.002129
grad_step = 000202, loss = 0.002122
grad_step = 000203, loss = 0.002130
grad_step = 000204, loss = 0.002148
grad_step = 000205, loss = 0.002126
grad_step = 000206, loss = 0.002061
grad_step = 000207, loss = 0.001993
grad_step = 000208, loss = 0.001956
grad_step = 000209, loss = 0.001968
grad_step = 000210, loss = 0.002017
grad_step = 000211, loss = 0.002092
grad_step = 000212, loss = 0.002139
grad_step = 000213, loss = 0.002147
grad_step = 000214, loss = 0.002031
grad_step = 000215, loss = 0.001922
grad_step = 000216, loss = 0.001903
grad_step = 000217, loss = 0.001967
grad_step = 000218, loss = 0.001993
grad_step = 000219, loss = 0.001924
grad_step = 000220, loss = 0.001853
grad_step = 000221, loss = 0.001862
grad_step = 000222, loss = 0.001910
grad_step = 000223, loss = 0.001919
grad_step = 000224, loss = 0.001874
grad_step = 000225, loss = 0.001833
grad_step = 000226, loss = 0.001836
grad_step = 000227, loss = 0.001860
grad_step = 000228, loss = 0.001864
grad_step = 000229, loss = 0.001837
grad_step = 000230, loss = 0.001809
grad_step = 000231, loss = 0.001806
grad_step = 000232, loss = 0.001821
grad_step = 000233, loss = 0.001834
grad_step = 000234, loss = 0.001827
grad_step = 000235, loss = 0.001809
grad_step = 000236, loss = 0.001793
grad_step = 000237, loss = 0.001788
grad_step = 000238, loss = 0.001793
grad_step = 000239, loss = 0.001800
grad_step = 000240, loss = 0.001801
grad_step = 000241, loss = 0.001794
grad_step = 000242, loss = 0.001784
grad_step = 000243, loss = 0.001775
grad_step = 000244, loss = 0.001768
grad_step = 000245, loss = 0.001766
grad_step = 000246, loss = 0.001767
grad_step = 000247, loss = 0.001771
grad_step = 000248, loss = 0.001776
grad_step = 000249, loss = 0.001785
grad_step = 000250, loss = 0.001800
grad_step = 000251, loss = 0.001833
grad_step = 000252, loss = 0.001888
grad_step = 000253, loss = 0.001996
grad_step = 000254, loss = 0.002151
grad_step = 000255, loss = 0.002365
grad_step = 000256, loss = 0.002540
grad_step = 000257, loss = 0.002542
grad_step = 000258, loss = 0.002347
grad_step = 000259, loss = 0.002120
grad_step = 000260, loss = 0.001998
grad_step = 000261, loss = 0.002010
grad_step = 000262, loss = 0.002049
grad_step = 000263, loss = 0.002017
grad_step = 000264, loss = 0.001948
grad_step = 000265, loss = 0.001896
grad_step = 000266, loss = 0.001896
grad_step = 000267, loss = 0.001942
grad_step = 000268, loss = 0.001914
grad_step = 000269, loss = 0.001825
grad_step = 000270, loss = 0.001824
grad_step = 000271, loss = 0.001880
grad_step = 000272, loss = 0.001850
grad_step = 000273, loss = 0.001778
grad_step = 000274, loss = 0.001795
grad_step = 000275, loss = 0.001847
grad_step = 000276, loss = 0.001808
grad_step = 000277, loss = 0.001753
grad_step = 000278, loss = 0.001787
grad_step = 000279, loss = 0.001816
grad_step = 000280, loss = 0.001768
grad_step = 000281, loss = 0.001742
grad_step = 000282, loss = 0.001778
grad_step = 000283, loss = 0.001784
grad_step = 000284, loss = 0.001746
grad_step = 000285, loss = 0.001740
grad_step = 000286, loss = 0.001767
grad_step = 000287, loss = 0.001763
grad_step = 000288, loss = 0.001735
grad_step = 000289, loss = 0.001736
grad_step = 000290, loss = 0.001755
grad_step = 000291, loss = 0.001749
grad_step = 000292, loss = 0.001730
grad_step = 000293, loss = 0.001731
grad_step = 000294, loss = 0.001743
grad_step = 000295, loss = 0.001741
grad_step = 000296, loss = 0.001727
grad_step = 000297, loss = 0.001725
grad_step = 000298, loss = 0.001733
grad_step = 000299, loss = 0.001734
grad_step = 000300, loss = 0.001725
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001721
grad_step = 000302, loss = 0.001725
grad_step = 000303, loss = 0.001728
grad_step = 000304, loss = 0.001724
grad_step = 000305, loss = 0.001718
grad_step = 000306, loss = 0.001718
grad_step = 000307, loss = 0.001721
grad_step = 000308, loss = 0.001721
grad_step = 000309, loss = 0.001717
grad_step = 000310, loss = 0.001714
grad_step = 000311, loss = 0.001714
grad_step = 000312, loss = 0.001716
grad_step = 000313, loss = 0.001716
grad_step = 000314, loss = 0.001713
grad_step = 000315, loss = 0.001711
grad_step = 000316, loss = 0.001710
grad_step = 000317, loss = 0.001711
grad_step = 000318, loss = 0.001711
grad_step = 000319, loss = 0.001710
grad_step = 000320, loss = 0.001708
grad_step = 000321, loss = 0.001707
grad_step = 000322, loss = 0.001706
grad_step = 000323, loss = 0.001706
grad_step = 000324, loss = 0.001706
grad_step = 000325, loss = 0.001705
grad_step = 000326, loss = 0.001704
grad_step = 000327, loss = 0.001703
grad_step = 000328, loss = 0.001702
grad_step = 000329, loss = 0.001702
grad_step = 000330, loss = 0.001701
grad_step = 000331, loss = 0.001701
grad_step = 000332, loss = 0.001700
grad_step = 000333, loss = 0.001700
grad_step = 000334, loss = 0.001699
grad_step = 000335, loss = 0.001698
grad_step = 000336, loss = 0.001697
grad_step = 000337, loss = 0.001696
grad_step = 000338, loss = 0.001696
grad_step = 000339, loss = 0.001695
grad_step = 000340, loss = 0.001694
grad_step = 000341, loss = 0.001694
grad_step = 000342, loss = 0.001693
grad_step = 000343, loss = 0.001693
grad_step = 000344, loss = 0.001692
grad_step = 000345, loss = 0.001692
grad_step = 000346, loss = 0.001691
grad_step = 000347, loss = 0.001691
grad_step = 000348, loss = 0.001691
grad_step = 000349, loss = 0.001691
grad_step = 000350, loss = 0.001691
grad_step = 000351, loss = 0.001691
grad_step = 000352, loss = 0.001693
grad_step = 000353, loss = 0.001695
grad_step = 000354, loss = 0.001700
grad_step = 000355, loss = 0.001709
grad_step = 000356, loss = 0.001723
grad_step = 000357, loss = 0.001749
grad_step = 000358, loss = 0.001788
grad_step = 000359, loss = 0.001854
grad_step = 000360, loss = 0.001925
grad_step = 000361, loss = 0.002008
grad_step = 000362, loss = 0.002014
grad_step = 000363, loss = 0.001947
grad_step = 000364, loss = 0.001822
grad_step = 000365, loss = 0.001739
grad_step = 000366, loss = 0.001755
grad_step = 000367, loss = 0.001825
grad_step = 000368, loss = 0.001864
grad_step = 000369, loss = 0.001805
grad_step = 000370, loss = 0.001721
grad_step = 000371, loss = 0.001690
grad_step = 000372, loss = 0.001727
grad_step = 000373, loss = 0.001771
grad_step = 000374, loss = 0.001763
grad_step = 000375, loss = 0.001718
grad_step = 000376, loss = 0.001687
grad_step = 000377, loss = 0.001700
grad_step = 000378, loss = 0.001733
grad_step = 000379, loss = 0.001748
grad_step = 000380, loss = 0.001738
grad_step = 000381, loss = 0.001731
grad_step = 000382, loss = 0.001753
grad_step = 000383, loss = 0.001807
grad_step = 000384, loss = 0.001880
grad_step = 000385, loss = 0.001935
grad_step = 000386, loss = 0.001987
grad_step = 000387, loss = 0.002003
grad_step = 000388, loss = 0.001967
grad_step = 000389, loss = 0.001891
grad_step = 000390, loss = 0.001779
grad_step = 000391, loss = 0.001702
grad_step = 000392, loss = 0.001693
grad_step = 000393, loss = 0.001734
grad_step = 000394, loss = 0.001776
grad_step = 000395, loss = 0.001773
grad_step = 000396, loss = 0.001734
grad_step = 000397, loss = 0.001692
grad_step = 000398, loss = 0.001678
grad_step = 000399, loss = 0.001696
grad_step = 000400, loss = 0.001723
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001733
grad_step = 000402, loss = 0.001714
grad_step = 000403, loss = 0.001682
grad_step = 000404, loss = 0.001662
grad_step = 000405, loss = 0.001665
grad_step = 000406, loss = 0.001684
grad_step = 000407, loss = 0.001698
grad_step = 000408, loss = 0.001697
grad_step = 000409, loss = 0.001681
grad_step = 000410, loss = 0.001663
grad_step = 000411, loss = 0.001655
grad_step = 000412, loss = 0.001658
grad_step = 000413, loss = 0.001667
grad_step = 000414, loss = 0.001673
grad_step = 000415, loss = 0.001673
grad_step = 000416, loss = 0.001667
grad_step = 000417, loss = 0.001660
grad_step = 000418, loss = 0.001655
grad_step = 000419, loss = 0.001653
grad_step = 000420, loss = 0.001654
grad_step = 000421, loss = 0.001655
grad_step = 000422, loss = 0.001655
grad_step = 000423, loss = 0.001654
grad_step = 000424, loss = 0.001653
grad_step = 000425, loss = 0.001652
grad_step = 000426, loss = 0.001651
grad_step = 000427, loss = 0.001651
grad_step = 000428, loss = 0.001651
grad_step = 000429, loss = 0.001650
grad_step = 000430, loss = 0.001649
grad_step = 000431, loss = 0.001648
grad_step = 000432, loss = 0.001646
grad_step = 000433, loss = 0.001644
grad_step = 000434, loss = 0.001642
grad_step = 000435, loss = 0.001641
grad_step = 000436, loss = 0.001639
grad_step = 000437, loss = 0.001639
grad_step = 000438, loss = 0.001638
grad_step = 000439, loss = 0.001637
grad_step = 000440, loss = 0.001637
grad_step = 000441, loss = 0.001637
grad_step = 000442, loss = 0.001636
grad_step = 000443, loss = 0.001636
grad_step = 000444, loss = 0.001636
grad_step = 000445, loss = 0.001636
grad_step = 000446, loss = 0.001636
grad_step = 000447, loss = 0.001637
grad_step = 000448, loss = 0.001638
grad_step = 000449, loss = 0.001640
grad_step = 000450, loss = 0.001644
grad_step = 000451, loss = 0.001651
grad_step = 000452, loss = 0.001662
grad_step = 000453, loss = 0.001681
grad_step = 000454, loss = 0.001710
grad_step = 000455, loss = 0.001762
grad_step = 000456, loss = 0.001835
grad_step = 000457, loss = 0.001952
grad_step = 000458, loss = 0.002087
grad_step = 000459, loss = 0.002238
grad_step = 000460, loss = 0.002363
grad_step = 000461, loss = 0.002437
grad_step = 000462, loss = 0.002508
grad_step = 000463, loss = 0.002613
grad_step = 000464, loss = 0.002368
grad_step = 000465, loss = 0.002013
grad_step = 000466, loss = 0.001830
grad_step = 000467, loss = 0.001927
grad_step = 000468, loss = 0.002091
grad_step = 000469, loss = 0.001969
grad_step = 000470, loss = 0.001748
grad_step = 000471, loss = 0.001764
grad_step = 000472, loss = 0.001893
grad_step = 000473, loss = 0.001840
grad_step = 000474, loss = 0.001692
grad_step = 000475, loss = 0.001729
grad_step = 000476, loss = 0.001816
grad_step = 000477, loss = 0.001731
grad_step = 000478, loss = 0.001654
grad_step = 000479, loss = 0.001727
grad_step = 000480, loss = 0.001746
grad_step = 000481, loss = 0.001658
grad_step = 000482, loss = 0.001647
grad_step = 000483, loss = 0.001711
grad_step = 000484, loss = 0.001695
grad_step = 000485, loss = 0.001629
grad_step = 000486, loss = 0.001645
grad_step = 000487, loss = 0.001686
grad_step = 000488, loss = 0.001658
grad_step = 000489, loss = 0.001617
grad_step = 000490, loss = 0.001638
grad_step = 000491, loss = 0.001661
grad_step = 000492, loss = 0.001635
grad_step = 000493, loss = 0.001611
grad_step = 000494, loss = 0.001630
grad_step = 000495, loss = 0.001643
grad_step = 000496, loss = 0.001622
grad_step = 000497, loss = 0.001608
grad_step = 000498, loss = 0.001622
grad_step = 000499, loss = 0.001629
grad_step = 000500, loss = 0.001614
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001605
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

  date_run                              2020-05-20 04:15:16.553484
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.235375
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-20 04:15:16.559579
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.128298
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-20 04:15:16.567197
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.155554
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-20 04:15:16.573701
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.949527
metric_name                                             r2_score
Name: 11, dtype: object 

  


### Running {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepARTrainingNetwork: 26844
100%|██████████| 10/10 [00:03<00:00,  3.10it/s, avg_epoch_loss=5.25]
INFO:root:Epoch[0] Elapsed time 3.228 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.249061
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.249061489105225 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff8292bb320> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'deepar', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepFactorTrainingNetwork: 12466
100%|██████████| 10/10 [00:01<00:00,  6.11it/s, avg_epoch_loss=3.59e+3]
INFO:root:Epoch[0] Elapsed time 1.637 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=3590.403646
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 3590.4036458333335 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff790a0ef28> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'deepfactor', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU
INFO:gluonts.model.wavenet._estimator:Using dilation depth 10 and receptive field length 1024

  #### Fit  ####################################################### 
INFO:root:using training windows of length = 12
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in WaveNet: 97636
 30%|███       | 3/10 [00:14<00:32,  4.70s/it, avg_epoch_loss=6.94] 60%|██████    | 6/10 [00:26<00:17,  4.49s/it, avg_epoch_loss=6.91] 90%|█████████ | 9/10 [00:38<00:04,  4.37s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:42<00:00,  4.24s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 42.408 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.864835
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.864834547042847 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff790972588> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'wavenet', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in TransformerTrainingNetwork: 33911
100%|██████████| 10/10 [00:02<00:00,  4.80it/s, avg_epoch_loss=5.86]
INFO:root:Epoch[0] Elapsed time 2.084 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.862858
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.862857675552368 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff7742d1e48> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'transformer', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepStateTrainingNetwork: 28054
 10%|█         | 1/10 [02:26<22:00, 146.70s/it, avg_epoch_loss=0.412] 20%|██        | 2/10 [05:55<22:02, 165.29s/it, avg_epoch_loss=0.399] 30%|███       | 3/10 [09:19<20:39, 177.03s/it, avg_epoch_loss=0.39]  40%|████      | 4/10 [12:58<18:56, 189.42s/it, avg_epoch_loss=0.385] 50%|█████     | 5/10 [16:43<16:41, 200.35s/it, avg_epoch_loss=0.384] 60%|██████    | 6/10 [20:39<14:03, 210.87s/it, avg_epoch_loss=0.383] 70%|███████   | 7/10 [24:02<10:25, 208.55s/it, avg_epoch_loss=0.381] 80%|████████  | 8/10 [28:04<07:17, 218.53s/it, avg_epoch_loss=0.378] 90%|█████████ | 9/10 [31:59<03:43, 223.48s/it, avg_epoch_loss=0.376]100%|██████████| 10/10 [35:54<00:00, 227.09s/it, avg_epoch_loss=0.375]100%|██████████| 10/10 [35:55<00:00, 215.52s/it, avg_epoch_loss=0.375]
INFO:root:Epoch[0] Elapsed time 2155.191 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.374571
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3745713621377945 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff774285128> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_name': 'deepstate', 'model_uri': 'model_gluon.gluonts_model', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in GaussianProcessTrainingNetwork: 14
100%|██████████| 10/10 [00:01<00:00,  5.61it/s, avg_epoch_loss=415]
INFO:root:Epoch[0] Elapsed time 1.806 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=414.652022
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 414.65202175008733 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff774167a90> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
INFO:root:Using CPU

  #### Fit  ####################################################### 
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in SimpleFeedForwardTrainingNetwork: 20323
100%|██████████| 10/10 [00:00<00:00, 39.54it/s, avg_epoch_loss=5.14]
INFO:root:Epoch[0] Elapsed time 0.254 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.135198
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.135197591781616 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff790ca0470> <class 'mlmodels.model_gluon.gluonts_model.Model'>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.

  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}} 'Model' object is not iterable 

  


### Running {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': 'ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]} 

  #### Setup Model   ############################################## 

  {'model_pars': {'model_uri': 'model_gluon.gluonts_model', 'model_name': 'seq2seq', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_parallel_samples': 100, 'cardinality': [2], 'embedding_dimension': 10, 'decoder_mlp_layer': [5, 10, 5], 'decoder_mlp_static_dim': 10, 'quantiles': [0.1, 0.5, 0.9]}, '_comment': {'encoder': 'Seq2SeqEncoder', 'context_length': 'Optional[int] = None', 'scaler': 'Scaler = NOPScaler()'}}, 'data_pars': {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []}, 'compute_pars': {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}}, 'out_pars': {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_gluon/gluonts_seq2seq/', 'plot_prob': True, 'quantiles': [0.5]}} 1 validation error for MLPEncoderModel
layer_sizes
  field required (type=value_error.missing) 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/timeseries/test02/model_list.json 

                        date_run  ...            metric_name
0   2020-05-20 04:14:40.771602  ...    mean_absolute_error
1   2020-05-20 04:14:40.777004  ...     mean_squared_error
2   2020-05-20 04:14:40.780742  ...  median_absolute_error
3   2020-05-20 04:14:40.784811  ...               r2_score
4   2020-05-20 04:14:50.473646  ...    mean_absolute_error
5   2020-05-20 04:14:50.478394  ...     mean_squared_error
6   2020-05-20 04:14:50.482372  ...  median_absolute_error
7   2020-05-20 04:14:50.486338  ...               r2_score
8   2020-05-20 04:15:16.553484  ...    mean_absolute_error
9   2020-05-20 04:15:16.559579  ...     mean_squared_error
10  2020-05-20 04:15:16.567197  ...  median_absolute_error
11  2020-05-20 04:15:16.573701  ...               r2_score

[12 rows x 6 columns] 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
TypeError: 'Model' object is not iterable
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 120, in benchmark_run
    model     = module.Model(model_pars, data_pars, compute_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py", line 81, in __init__
    mpars['encoder'] = MLPEncoder()   #bug in seq2seq
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/gluonts/core/component.py", line 424, in init_wrapper
    model = PydanticModel(**{**nmargs, **kwargs})
  File "pydantic/main.py", line 283, in pydantic.main.BaseModel.__init__
pydantic.error_wrappers.ValidationError: 1 validation error for MLPEncoderModel
layer_sizes
  field required (type=value_error.missing)
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa275c89128> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa21d5c7940> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa22076bd30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa21d5c7940> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa275c89128> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa21d5c7940> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa22076bd30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa21d5c7940> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa275c89128> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa21d5c7940> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet50_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/benchmark.py", line 126, in benchmark_run
    model, session = module.fit(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 222, in fit
    train_iter, valid_iter = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/torchhub.py", line 190, in get_dataset
    loader = DataLoader(data_pars)
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa22076bd30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet50_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet50_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet50_2/'}} 'data_info' 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/cnn/mnist 

  Empty DataFrame
Columns: [date_run, model_uri, json, dataset_uri, metric, metric_name]
Index: [] 
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 236, in __init__
    self.data_info                = data_pars['data_info']
KeyError: 'data_info'
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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f5bad46e080> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=5b7eb8707e38fda9e11f90d9c373d42770a332b9427f82aed1264134766dcbc7
  Stored in directory: /tmp/pip-ephem-wheel-cache-hf2_osb0/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f5b4615da20> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1744896/17464789 [=>............................] - ETA: 0s
 4579328/17464789 [======>.......................] - ETA: 0s
 7856128/17464789 [============>.................] - ETA: 0s
11149312/17464789 [==================>...........] - ETA: 0s
14442496/17464789 [=======================>......] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-20 04:53:56.379824: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-20 04:53:56.393877: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-20 04:53:56.394105: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562285c2c970 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-20 04:53:56.394125: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 16s - loss: 7.4673 - accuracy: 0.5130
 2000/25000 [=>............................] - ETA: 11s - loss: 7.4596 - accuracy: 0.5135
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.4213 - accuracy: 0.5160 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.4405 - accuracy: 0.5148
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.4949 - accuracy: 0.5112
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.5618 - accuracy: 0.5068
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6294 - accuracy: 0.5024
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6494 - accuracy: 0.5011
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6411 - accuracy: 0.5017
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6804 - accuracy: 0.4991
11000/25000 [============>.................] - ETA: 4s - loss: 7.6834 - accuracy: 0.4989
12000/25000 [=============>................] - ETA: 4s - loss: 7.7216 - accuracy: 0.4964
13000/25000 [==============>...............] - ETA: 4s - loss: 7.7232 - accuracy: 0.4963
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6995 - accuracy: 0.4979
15000/25000 [=================>............] - ETA: 3s - loss: 7.6748 - accuracy: 0.4995
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6935 - accuracy: 0.4983
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7045 - accuracy: 0.4975
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7033 - accuracy: 0.4976
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6949 - accuracy: 0.4982
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7050 - accuracy: 0.4975
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6732 - accuracy: 0.4996
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6722 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6813 - accuracy: 0.4990
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6692 - accuracy: 0.4998
25000/25000 [==============================] - 10s 406us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-20 04:54:14.569598
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-20 04:54:14.569598  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:07:52, 11.3kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:01:23, 15.9kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:34:13, 22.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 877k/862M [00:01<7:24:29, 32.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.52M/862M [00:01<5:10:24, 46.1kB/s].vector_cache/glove.6B.zip:   1%|          | 9.36M/862M [00:01<3:35:52, 65.8kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.4M/862M [00:01<2:30:44, 94.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:01<1:44:55, 134kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:01<1:13:18, 191kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.9M/862M [00:01<51:05, 273kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.7M/862M [00:01<35:43, 388kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.0M/862M [00:01<24:55, 553kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.3M/862M [00:02<17:30, 784kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.5M/862M [00:02<12:15, 1.11MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.7M/862M [00:02<08:40, 1.57MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:02<06:06, 2.21MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<10:06, 1.34MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<08:57, 1.50MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:05<08:31, 1.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:05<06:31, 2.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<06:51, 1.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<06:28, 2.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:07<04:53, 2.73MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:07<03:34, 3.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<40:22, 329kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:08<31:20, 424kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:09<22:34, 588kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.7M/862M [00:09<15:57, 830kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<16:24, 805kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<12:58, 1.02MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:11<09:23, 1.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<09:25, 1.39MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<09:16, 1.42MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:13<07:09, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<07:06, 1.84MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:14<06:19, 2.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<04:45, 2.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<06:22, 2.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<07:06, 1.83MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:17<05:38, 2.30MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:17<04:06, 3.16MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<1:35:24, 136kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<1:08:03, 190kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:18<47:52, 270kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<36:26, 353kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<26:49, 480kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:20<19:04, 673kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<16:18, 785kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<12:42, 1.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:22<09:12, 1.39MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<09:26, 1.35MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:24<09:14, 1.38MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:24<07:06, 1.79MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:01, 1.81MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:13, 2.04MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<04:40, 2.71MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:13, 2.03MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:49, 1.84MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<05:20, 2.36MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<03:55, 3.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:34, 1.65MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<06:35, 1.90MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:53, 2.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:20, 1.97MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:44, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:19, 2.87MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:06, 2.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:05, 2.43MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<03:58, 3.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<03:40, 3.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:48, 2.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:18, 2.32MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<03:59, 3.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:40, 2.16MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<06:22, 1.92MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:58, 2.46MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<03:37, 3.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<09:49, 1.24MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:08, 1.50MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<05:57, 2.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:00, 1.73MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:14, 1.94MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:41, 2.58MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:55, 2.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:48, 1.77MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:26, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<03:55, 3.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<15:37, 767kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<12:13, 980kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<08:52, 1.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<08:48, 1.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<08:48, 1.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:01, 1.69MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<05:06, 2.32MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<08:20, 1.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:09, 1.66MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:15, 2.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<06:16, 1.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:41, 2.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:15, 2.76MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:33, 2.11MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:12, 2.25MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:54<03:54, 2.99MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:17, 2.20MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:16, 1.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:58, 2.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<03:36, 3.21MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<08:45, 1.32MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<07:24, 1.56MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<05:29, 2.10MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:21, 1.81MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:43, 2.01MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:18, 2.66MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<05:31, 2.07MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<06:25, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:07, 2.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:43, 3.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<14:39, 775kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<11:29, 988kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<08:19, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:18, 1.36MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<08:18, 1.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<06:20, 1.78MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<04:34, 2.45MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<08:05, 1.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<06:54, 1.62MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<05:05, 2.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<06:01, 1.85MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<05:26, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<04:06, 2.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<05:18, 2.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:01, 1.84MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<04:50, 2.29MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<03:30, 3.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<14:16, 771kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<11:12, 982kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<08:04, 1.36MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<08:03, 1.36MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<08:03, 1.36MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<06:14, 1.75MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<04:28, 2.43MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<10:18, 1.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<08:23, 1.30MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<06:09, 1.76MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:40, 1.62MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<05:50, 1.85MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<04:22, 2.46MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:25, 1.98MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<06:09, 1.74MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<04:49, 2.22MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<03:30, 3.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<07:39, 1.39MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:32, 1.63MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<04:49, 2.21MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<05:41, 1.86MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:19, 1.67MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:01, 2.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<03:38, 2.89MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<15:18, 688kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<11:52, 886kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<08:34, 1.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<08:16, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<08:07, 1.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<06:09, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<04:25, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<08:03, 1.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<06:47, 1.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:59, 2.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:44, 1.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<06:26, 1.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<04:59, 2.06MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<03:38, 2.81MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<06:03, 1.69MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<05:22, 1.91MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<03:59, 2.56MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:01, 2.02MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<05:47, 1.75MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:37, 2.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<03:20, 3.02MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<12:45, 792kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<10:01, 1.01MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<07:14, 1.39MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<07:15, 1.38MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<07:12, 1.39MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<05:34, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<04:00, 2.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<16:41, 597kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<12:32, 794kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<09:02, 1.10MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<08:28, 1.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<07:01, 1.41MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:07, 1.92MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<05:44, 1.71MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<05:05, 1.93MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<03:48, 2.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:50, 2.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<04:16, 2.28MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<03:11, 3.04MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:50<02:24, 4.03MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<13:25, 721kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<11:32, 839kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<08:36, 1.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<06:07, 1.57MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<14:46, 651kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<11:22, 846kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<08:09, 1.17MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<07:47, 1.23MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<07:35, 1.26MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<05:45, 1.66MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:09, 2.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<06:22, 1.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:28, 1.73MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:04, 2.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:57, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<05:33, 1.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<04:19, 2.17MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<03:09, 2.97MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<05:51, 1.59MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:08, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<03:50, 2.42MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<04:42, 1.97MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<05:15, 1.76MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<04:06, 2.25MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<02:58, 3.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<07:55, 1.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<06:31, 1.41MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<04:45, 1.93MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:23, 1.69MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:42, 1.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<04:25, 2.06MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<03:10, 2.86MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<13:36, 667kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<10:28, 865kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<07:33, 1.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<07:18, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<07:01, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<05:22, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:51, 2.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<24:09, 370kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<17:50, 500kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<12:39, 703kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<10:50, 817kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<09:27, 936kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<07:00, 1.26MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:15<05:00, 1.76MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<08:16, 1.06MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<06:45, 1.30MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<04:57, 1.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:22, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:36, 1.56MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:19, 2.01MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<03:06, 2.78MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<09:22, 923kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<07:28, 1.16MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<05:25, 1.59MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:43, 1.50MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:49, 1.47MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:31, 1.89MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<03:15, 2.61MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<27:56, 305kB/s] .vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<20:29, 415kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<14:32, 584kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<11:58, 705kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<10:10, 830kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<07:33, 1.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<05:21, 1.56MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<23:54, 350kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<17:38, 474kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<12:30, 667kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<10:32, 788kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<09:08, 908kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<06:49, 1.21MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:31<04:50, 1.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<27:09, 303kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<19:54, 414kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<14:04, 583kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<11:37, 703kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<09:51, 828kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<07:16, 1.12MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:35<05:10, 1.57MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<07:21, 1.10MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<06:02, 1.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:24, 1.83MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:49, 1.66MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:09, 1.56MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:02, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<02:55, 2.72MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<11:42, 680kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<09:03, 878kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<06:31, 1.22MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:16, 1.26MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<06:08, 1.29MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:43, 1.67MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<03:22, 2.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<09:11, 851kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<07:15, 1.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<05:16, 1.48MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:25, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:39, 1.67MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<03:25, 2.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:04, 1.88MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:40, 2.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<02:43, 2.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:39, 2.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:11, 1.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:19, 2.28MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<02:24, 3.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<19:51, 380kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<14:36, 516kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<10:23, 723kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<08:53, 841kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<07:48, 958kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<05:50, 1.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:55<04:09, 1.78MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<26:25, 280kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<19:16, 384kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<13:37, 541kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<11:06, 660kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<09:25, 779kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<06:59, 1.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<04:57, 1.47MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<12:15, 593kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<09:23, 774kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<06:44, 1.07MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<06:16, 1.15MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<05:59, 1.20MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<04:34, 1.57MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<03:17, 2.17MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<10:56, 652kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<08:24, 847kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<06:01, 1.18MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:47, 1.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:33, 1.27MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:15, 1.66MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<03:02, 2.30MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<23:22, 299kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<17:06, 408kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<12:07, 574kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<09:57, 695kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<07:46, 890kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<05:36, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<05:24, 1.27MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<04:29, 1.52MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:18, 2.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:52, 1.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<04:10, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<03:13, 2.09MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:22, 2.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:50, 1.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:24, 1.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:32, 2.63MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:13, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<02:59, 2.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:14, 2.96MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:00, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:34, 1.84MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:49, 2.33MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<02:03, 3.18MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<04:19, 1.50MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:42, 1.75MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:44, 2.37MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:21, 1.91MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:02, 2.11MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:16, 2.81MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<02:59, 2.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:29, 1.82MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:48, 2.27MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:27<02:01, 3.12MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<08:17, 760kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<06:28, 972kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:41, 1.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:38, 1.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<04:37, 1.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:31, 1.76MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:31<02:31, 2.44MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<05:21, 1.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<04:25, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:14, 1.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:35, 1.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:47, 1.60MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:55, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:07, 2.85MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:41, 1.28MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:56, 1.52MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<02:53, 2.08MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:19, 1.79MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:38, 1.63MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:52, 2.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<02:04, 2.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<07:44, 759kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<06:03, 969kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<04:21, 1.34MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<04:18, 1.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<04:18, 1.35MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:19, 1.74MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:23, 2.41MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<07:47, 737kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<06:04, 946kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<04:22, 1.31MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<04:21, 1.30MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<04:15, 1.33MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<03:13, 1.76MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:20, 2.41MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:34, 1.57MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<03:06, 1.80MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:17, 2.43MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<02:49, 1.97MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:12, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:32, 2.17MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<01:50, 2.98MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<07:06, 769kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<05:33, 982kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<04:00, 1.36MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:00, 1.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<03:57, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:59, 1.80MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:12, 2.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:56, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:38, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<01:59, 2.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:32, 2.07MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:21, 2.22MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<01:47, 2.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:22, 2.18MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:45, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:09, 2.39MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<01:33, 3.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<05:01, 1.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:05, 1.25MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<02:59, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:10, 1.59MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:45, 1.83MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<02:01, 2.47MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:33, 1.95MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:50, 1.75MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:12, 2.24MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<01:37, 3.05MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:55, 1.68MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:29, 1.97MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:51, 2.64MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<01:22, 3.52MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<05:37, 861kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<05:00, 967kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:43, 1.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<02:40, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:30, 1.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:59, 1.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:12, 2.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:33, 1.84MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:17, 2.05MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<01:42, 2.73MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:15, 2.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:37, 1.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:05, 2.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<01:29, 3.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<05:55, 770kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<04:34, 999kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<03:18, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:16, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:18, 1.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:33, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<01:49, 2.43MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<06:04, 729kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<04:43, 935kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<03:24, 1.29MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:19, 1.31MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:18, 1.32MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:31, 1.72MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:24<01:47, 2.39MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<04:37, 929kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<03:26, 1.24MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<02:29, 1.70MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:40, 1.58MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:49, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:13, 1.90MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<01:35, 2.62MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<05:32, 750kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<04:19, 959kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<03:07, 1.32MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<03:04, 1.33MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:36, 1.57MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<01:55, 2.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:12, 1.81MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:24, 1.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:52, 2.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:21, 2.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:39, 1.48MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:18, 1.71MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:41, 2.31MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:01, 1.91MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:16, 1.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:48, 2.14MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:38<01:17, 2.94MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<04:31, 842kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<03:30, 1.08MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:32, 1.49MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:40<01:49, 2.06MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<04:30, 830kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<03:56, 948kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:54, 1.28MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<02:04, 1.78MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<03:09, 1.16MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:36, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:54, 1.91MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:06, 1.71MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<02:14, 1.61MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:44, 2.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:46<01:15, 2.82MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<11:30, 307kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<08:25, 418kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<05:57, 587kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<04:52, 709kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<04:10, 827kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<03:06, 1.11MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<02:11, 1.55MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<04:42, 721kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<03:38, 928kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<02:36, 1.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:33, 1.30MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:29, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:55, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:21, 2.39MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<07:39, 425kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<05:41, 570kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<04:01, 799kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<03:30, 906kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<03:09, 1.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:22, 1.33MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:41, 1.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<04:57, 629kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<03:47, 818kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:43, 1.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:33, 1.19MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<02:03, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:31, 1.98MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:05, 2.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<03:51, 770kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<03:21, 884kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:30, 1.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:45, 1.65MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<04:19, 671kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<03:20, 865kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<02:24, 1.20MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:16, 1.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:08<02:13, 1.27MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:42, 1.65MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<01:12, 2.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:10<03:48, 726kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:57, 935kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:06, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<02:04, 1.30MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<02:01, 1.33MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:33, 1.73MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:05, 2.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<06:11, 425kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<04:33, 575kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:14, 802kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<02:47, 916kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<02:32, 1.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:54, 1.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<01:20, 1.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<03:57, 629kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<03:02, 816kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<02:09, 1.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<02:01, 1.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<01:55, 1.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:26, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<01:02, 2.29MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:39, 1.42MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:24, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:02, 2.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:13, 1.87MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:05, 2.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<00:48, 2.76MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:04, 2.07MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:58, 2.25MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:43, 2.99MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:59, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:09, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<00:55, 2.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<00:39, 3.13MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:58, 696kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:18, 898kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:38, 1.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:34, 1.27MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:32, 1.30MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:11, 1.68MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<00:50, 2.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<02:55, 660kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<02:15, 854kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:36, 1.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:30, 1.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:28, 1.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:07, 1.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<00:47, 2.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:37<02:34, 696kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<01:59, 898kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:25, 1.24MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:21, 1.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:19, 1.30MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:00, 1.68MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:42, 2.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<02:27, 672kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<01:54, 865kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:21, 1.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:16, 1.24MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<01:14, 1.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:57, 1.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:40, 2.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<02:18, 658kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<01:46, 852kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<01:15, 1.18MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<01:10, 1.23MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:58, 1.47MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:42, 1.99MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:47, 1.75MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:50, 1.63MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<00:39, 2.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:27, 2.86MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<03:27, 378kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<02:33, 509kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<01:47, 713kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:29, 833kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<01:18, 940kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:58, 1.25MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:40, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:43, 681kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:19, 881kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:55<00:56, 1.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:52, 1.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:51, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:39, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:27, 2.29MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<01:31, 677kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<01:10, 875kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [05:59<00:49, 1.21MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:46, 1.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:44, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:33, 1.69MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:23, 2.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:37, 1.41MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:32, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:23, 2.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:26, 1.87MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:29, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:22, 2.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<00:15, 2.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:53, 852kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:41, 1.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:29, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:28, 1.44MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:28, 1.43MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:21, 1.85MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<00:14, 2.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<02:01, 305kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<01:27, 419kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:59, 588kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:46, 710kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:39, 828kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:28, 1.11MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:18, 1.55MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:44, 648kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:33, 843kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:22, 1.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:20, 1.21MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:19, 1.27MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:14, 1.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:09, 2.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:14, 1.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.66MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:08, 2.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:09, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<00:04, 2.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.87MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:03, 2.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:03, 2.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:04, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.24MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:25<00:01, 3.06MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:05, 707kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 908kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.25MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<119:47:39,  1.08s/it]  0%|          | 723/400000 [00:01<83:42:33,  1.32it/s]  0%|          | 1440/400000 [00:01<58:29:45,  1.89it/s]  1%|          | 2110/400000 [00:01<40:52:59,  2.70it/s]  1%|          | 2826/400000 [00:01<28:34:17,  3.86it/s]  1%|          | 3534/400000 [00:01<19:58:08,  5.52it/s]  1%|          | 4245/400000 [00:01<13:57:28,  7.88it/s]  1%|          | 4979/400000 [00:01<9:45:24, 11.25it/s]   1%|▏         | 5646/400000 [00:01<6:49:23, 16.05it/s]  2%|▏         | 6398/400000 [00:01<4:46:17, 22.91it/s]  2%|▏         | 7129/400000 [00:02<3:20:17, 32.69it/s]  2%|▏         | 7850/400000 [00:02<2:20:13, 46.61it/s]  2%|▏         | 8563/400000 [00:02<1:38:15, 66.40it/s]  2%|▏         | 9272/400000 [00:02<1:08:56, 94.47it/s]  2%|▏         | 9988/400000 [00:02<48:26, 134.19it/s]   3%|▎         | 10704/400000 [00:02<34:07, 190.17it/s]  3%|▎         | 11449/400000 [00:02<24:05, 268.74it/s]  3%|▎         | 12191/400000 [00:02<17:05, 378.04it/s]  3%|▎         | 12939/400000 [00:02<12:12, 528.60it/s]  3%|▎         | 13678/400000 [00:02<08:47, 732.66it/s]  4%|▎         | 14413/400000 [00:03<06:24, 1002.72it/s]  4%|▍         | 15163/400000 [00:03<04:44, 1354.80it/s]  4%|▍         | 15924/400000 [00:03<03:33, 1797.99it/s]  4%|▍         | 16669/400000 [00:03<02:44, 2327.63it/s]  4%|▍         | 17413/400000 [00:03<02:10, 2929.26it/s]  5%|▍         | 18155/400000 [00:03<01:47, 3561.19it/s]  5%|▍         | 18890/400000 [00:03<01:35, 3974.21it/s]  5%|▍         | 19613/400000 [00:03<01:22, 4594.78it/s]  5%|▌         | 20359/400000 [00:03<01:13, 5192.53it/s]  5%|▌         | 21139/400000 [00:04<01:05, 5771.26it/s]  5%|▌         | 21880/400000 [00:04<01:01, 6181.12it/s]  6%|▌         | 22613/400000 [00:04<00:58, 6474.37it/s]  6%|▌         | 23345/400000 [00:04<00:56, 6693.72it/s]  6%|▌         | 24079/400000 [00:04<00:54, 6873.55it/s]  6%|▌         | 24814/400000 [00:04<00:53, 7008.57it/s]  6%|▋         | 25547/400000 [00:04<00:52, 7073.76it/s]  7%|▋         | 26309/400000 [00:04<00:51, 7227.73it/s]  7%|▋         | 27056/400000 [00:04<00:51, 7296.61it/s]  7%|▋         | 27798/400000 [00:04<00:52, 7108.64it/s]  7%|▋         | 28553/400000 [00:05<00:51, 7234.21it/s]  7%|▋         | 29284/400000 [00:05<00:51, 7240.64it/s]  8%|▊         | 30023/400000 [00:05<00:50, 7284.15it/s]  8%|▊         | 30755/400000 [00:05<00:50, 7272.84it/s]  8%|▊         | 31485/400000 [00:05<00:53, 6923.62it/s]  8%|▊         | 32203/400000 [00:05<00:52, 6994.81it/s]  8%|▊         | 32926/400000 [00:05<00:51, 7062.98it/s]  8%|▊         | 33657/400000 [00:05<00:51, 7133.49it/s]  9%|▊         | 34373/400000 [00:05<00:51, 7122.34it/s]  9%|▉         | 35090/400000 [00:05<00:51, 7133.36it/s]  9%|▉         | 35805/400000 [00:06<00:51, 7106.73it/s]  9%|▉         | 36517/400000 [00:06<00:51, 7027.08it/s]  9%|▉         | 37240/400000 [00:06<00:51, 7085.16it/s]  9%|▉         | 37967/400000 [00:06<00:50, 7135.98it/s] 10%|▉         | 38682/400000 [00:06<00:55, 6516.55it/s] 10%|▉         | 39431/400000 [00:06<00:53, 6779.31it/s] 10%|█         | 40152/400000 [00:06<00:52, 6902.47it/s] 10%|█         | 40891/400000 [00:06<00:51, 7040.85it/s] 10%|█         | 41602/400000 [00:06<00:51, 7026.60it/s] 11%|█         | 42340/400000 [00:06<00:50, 7126.32it/s] 11%|█         | 43077/400000 [00:07<00:49, 7194.39it/s] 11%|█         | 43799/400000 [00:07<00:49, 7197.84it/s] 11%|█         | 44542/400000 [00:07<00:48, 7265.86it/s] 11%|█▏        | 45270/400000 [00:07<00:49, 7113.46it/s] 11%|█▏        | 45997/400000 [00:07<00:49, 7157.32it/s] 12%|█▏        | 46730/400000 [00:07<00:49, 7206.45it/s] 12%|█▏        | 47452/400000 [00:07<00:49, 7153.65it/s] 12%|█▏        | 48201/400000 [00:07<00:48, 7249.04it/s] 12%|█▏        | 48927/400000 [00:07<00:49, 7108.04it/s] 12%|█▏        | 49655/400000 [00:08<00:48, 7157.95it/s] 13%|█▎        | 50390/400000 [00:08<00:48, 7212.65it/s] 13%|█▎        | 51114/400000 [00:08<00:48, 7219.60it/s] 13%|█▎        | 51840/400000 [00:08<00:48, 7230.60it/s] 13%|█▎        | 52564/400000 [00:08<00:48, 7117.37it/s] 13%|█▎        | 53277/400000 [00:08<00:48, 7118.68it/s] 13%|█▎        | 53997/400000 [00:08<00:48, 7140.90it/s] 14%|█▎        | 54712/400000 [00:08<00:48, 7126.54it/s] 14%|█▍        | 55462/400000 [00:08<00:47, 7232.69it/s] 14%|█▍        | 56189/400000 [00:08<00:48, 7122.53it/s] 14%|█▍        | 56930/400000 [00:09<00:47, 7205.61it/s] 14%|█▍        | 57652/400000 [00:09<00:48, 7014.66it/s] 15%|█▍        | 58356/400000 [00:09<00:49, 6959.32it/s] 15%|█▍        | 59092/400000 [00:09<00:48, 7074.39it/s] 15%|█▍        | 59801/400000 [00:09<00:48, 6986.10it/s] 15%|█▌        | 60523/400000 [00:09<00:48, 7052.82it/s] 15%|█▌        | 61263/400000 [00:09<00:47, 7152.09it/s] 16%|█▌        | 62001/400000 [00:09<00:46, 7218.76it/s] 16%|█▌        | 62753/400000 [00:09<00:46, 7306.05it/s] 16%|█▌        | 63485/400000 [00:09<00:47, 7140.61it/s] 16%|█▌        | 64229/400000 [00:10<00:46, 7224.89it/s] 16%|█▌        | 64984/400000 [00:10<00:45, 7317.43it/s] 16%|█▋        | 65736/400000 [00:10<00:45, 7376.85it/s] 17%|█▋        | 66487/400000 [00:10<00:44, 7416.02it/s] 17%|█▋        | 67230/400000 [00:10<00:46, 7198.52it/s] 17%|█▋        | 67952/400000 [00:10<00:46, 7185.09it/s] 17%|█▋        | 68692/400000 [00:10<00:45, 7247.94it/s] 17%|█▋        | 69426/400000 [00:10<00:45, 7272.97it/s] 18%|█▊        | 70188/400000 [00:10<00:44, 7373.29it/s] 18%|█▊        | 70927/400000 [00:10<00:45, 7294.01it/s] 18%|█▊        | 71695/400000 [00:11<00:44, 7404.57it/s] 18%|█▊        | 72450/400000 [00:11<00:44, 7443.39it/s] 18%|█▊        | 73196/400000 [00:11<00:44, 7358.31it/s] 18%|█▊        | 73933/400000 [00:11<00:44, 7281.11it/s] 19%|█▊        | 74662/400000 [00:11<00:45, 7111.18it/s] 19%|█▉        | 75381/400000 [00:11<00:45, 7130.28it/s] 19%|█▉        | 76100/400000 [00:11<00:45, 7146.28it/s] 19%|█▉        | 76816/400000 [00:11<00:45, 7134.79it/s] 19%|█▉        | 77556/400000 [00:11<00:44, 7210.36it/s] 20%|█▉        | 78297/400000 [00:11<00:44, 7268.77it/s] 20%|█▉        | 79025/400000 [00:12<00:44, 7161.81it/s] 20%|█▉        | 79751/400000 [00:12<00:44, 7189.17it/s] 20%|██        | 80487/400000 [00:12<00:44, 7238.04it/s] 20%|██        | 81260/400000 [00:12<00:43, 7375.53it/s] 21%|██        | 82004/400000 [00:12<00:43, 7394.19it/s] 21%|██        | 82762/400000 [00:12<00:42, 7447.74it/s] 21%|██        | 83523/400000 [00:12<00:42, 7494.57it/s] 21%|██        | 84290/400000 [00:12<00:41, 7545.44it/s] 21%|██▏       | 85059/400000 [00:12<00:41, 7588.06it/s] 21%|██▏       | 85819/400000 [00:12<00:41, 7582.29it/s] 22%|██▏       | 86578/400000 [00:13<00:41, 7552.70it/s] 22%|██▏       | 87339/400000 [00:13<00:41, 7568.66it/s] 22%|██▏       | 88097/400000 [00:13<00:41, 7560.31it/s] 22%|██▏       | 88867/400000 [00:13<00:40, 7599.21it/s] 22%|██▏       | 89628/400000 [00:13<00:41, 7500.46it/s] 23%|██▎       | 90379/400000 [00:13<00:42, 7226.75it/s] 23%|██▎       | 91136/400000 [00:13<00:42, 7324.26it/s] 23%|██▎       | 91871/400000 [00:13<00:42, 7248.02it/s] 23%|██▎       | 92598/400000 [00:13<00:42, 7216.40it/s] 23%|██▎       | 93321/400000 [00:14<00:42, 7135.45it/s] 24%|██▎       | 94069/400000 [00:14<00:42, 7235.33it/s] 24%|██▎       | 94794/400000 [00:14<00:42, 7225.21it/s] 24%|██▍       | 95518/400000 [00:14<00:42, 7177.00it/s] 24%|██▍       | 96237/400000 [00:14<00:42, 7161.03it/s] 24%|██▍       | 96954/400000 [00:14<00:42, 7077.33it/s] 24%|██▍       | 97690/400000 [00:14<00:42, 7158.41it/s] 25%|██▍       | 98429/400000 [00:14<00:41, 7224.61it/s] 25%|██▍       | 99159/400000 [00:14<00:41, 7246.18it/s] 25%|██▍       | 99885/400000 [00:14<00:41, 7212.01it/s] 25%|██▌       | 100607/400000 [00:15<00:41, 7154.47it/s] 25%|██▌       | 101356/400000 [00:15<00:41, 7249.54it/s] 26%|██▌       | 102089/400000 [00:15<00:40, 7269.90it/s] 26%|██▌       | 102830/400000 [00:15<00:40, 7308.48it/s] 26%|██▌       | 103562/400000 [00:15<00:40, 7242.69it/s] 26%|██▌       | 104300/400000 [00:15<00:40, 7281.30it/s] 26%|██▋       | 105029/400000 [00:15<00:40, 7262.93it/s] 26%|██▋       | 105780/400000 [00:15<00:40, 7330.70it/s] 27%|██▋       | 106514/400000 [00:15<00:40, 7192.58it/s] 27%|██▋       | 107235/400000 [00:15<00:40, 7196.64it/s] 27%|██▋       | 107956/400000 [00:16<00:40, 7125.69it/s] 27%|██▋       | 108697/400000 [00:16<00:40, 7207.73it/s] 27%|██▋       | 109451/400000 [00:16<00:39, 7303.03it/s] 28%|██▊       | 110194/400000 [00:16<00:39, 7340.61it/s] 28%|██▊       | 110929/400000 [00:16<00:40, 7078.57it/s] 28%|██▊       | 111640/400000 [00:16<00:41, 7023.28it/s] 28%|██▊       | 112369/400000 [00:16<00:40, 7101.08it/s] 28%|██▊       | 113081/400000 [00:16<00:40, 7101.74it/s] 28%|██▊       | 113813/400000 [00:16<00:39, 7164.37it/s] 29%|██▊       | 114531/400000 [00:16<00:39, 7137.53it/s] 29%|██▉       | 115246/400000 [00:17<00:40, 7032.34it/s] 29%|██▉       | 115964/400000 [00:17<00:40, 7074.45it/s] 29%|██▉       | 116681/400000 [00:17<00:39, 7102.02it/s] 29%|██▉       | 117414/400000 [00:17<00:39, 7167.29it/s] 30%|██▉       | 118132/400000 [00:17<00:39, 7134.88it/s] 30%|██▉       | 118846/400000 [00:17<00:40, 6979.77it/s] 30%|██▉       | 119578/400000 [00:17<00:39, 7077.14it/s] 30%|███       | 120287/400000 [00:17<00:39, 7036.62it/s] 30%|███       | 120992/400000 [00:17<00:39, 7025.20it/s] 30%|███       | 121696/400000 [00:17<00:39, 7027.11it/s] 31%|███       | 122400/400000 [00:18<00:40, 6885.63it/s] 31%|███       | 123128/400000 [00:18<00:39, 6997.30it/s] 31%|███       | 123869/400000 [00:18<00:38, 7115.59it/s] 31%|███       | 124615/400000 [00:18<00:38, 7213.84it/s] 31%|███▏      | 125338/400000 [00:18<00:38, 7111.37it/s] 32%|███▏      | 126103/400000 [00:18<00:37, 7263.32it/s] 32%|███▏      | 126910/400000 [00:18<00:36, 7487.46it/s] 32%|███▏      | 127662/400000 [00:18<00:36, 7439.46it/s] 32%|███▏      | 128413/400000 [00:18<00:36, 7458.14it/s] 32%|███▏      | 129176/400000 [00:18<00:36, 7507.66it/s] 32%|███▏      | 129929/400000 [00:19<00:35, 7513.21it/s] 33%|███▎      | 130692/400000 [00:19<00:35, 7546.51it/s] 33%|███▎      | 131448/400000 [00:19<00:35, 7528.30it/s] 33%|███▎      | 132202/400000 [00:19<00:36, 7433.47it/s] 33%|███▎      | 132946/400000 [00:19<00:35, 7431.41it/s] 33%|███▎      | 133692/400000 [00:19<00:35, 7438.54it/s] 34%|███▎      | 134437/400000 [00:19<00:35, 7439.01it/s] 34%|███▍      | 135182/400000 [00:19<00:35, 7439.93it/s] 34%|███▍      | 135927/400000 [00:19<00:35, 7436.34it/s] 34%|███▍      | 136671/400000 [00:20<00:35, 7346.05it/s] 34%|███▍      | 137406/400000 [00:20<00:36, 7145.71it/s] 35%|███▍      | 138122/400000 [00:20<00:36, 7147.00it/s] 35%|███▍      | 138845/400000 [00:20<00:36, 7169.09it/s] 35%|███▍      | 139585/400000 [00:20<00:35, 7234.47it/s] 35%|███▌      | 140310/400000 [00:20<00:36, 7171.34it/s] 35%|███▌      | 141055/400000 [00:20<00:35, 7251.13it/s] 35%|███▌      | 141781/400000 [00:20<00:35, 7209.63it/s] 36%|███▌      | 142526/400000 [00:20<00:35, 7278.61it/s] 36%|███▌      | 143273/400000 [00:20<00:35, 7334.30it/s] 36%|███▌      | 144007/400000 [00:21<00:35, 7310.48it/s] 36%|███▌      | 144739/400000 [00:21<00:36, 7070.80it/s] 36%|███▋      | 145476/400000 [00:21<00:35, 7157.19it/s] 37%|███▋      | 146213/400000 [00:21<00:35, 7219.37it/s] 37%|███▋      | 146949/400000 [00:21<00:34, 7258.94it/s] 37%|███▋      | 147676/400000 [00:21<00:35, 7108.46it/s] 37%|███▋      | 148389/400000 [00:21<00:36, 6943.71it/s] 37%|███▋      | 149096/400000 [00:21<00:35, 6979.47it/s] 37%|███▋      | 149822/400000 [00:21<00:35, 7060.59it/s] 38%|███▊      | 150530/400000 [00:21<00:35, 7058.82it/s] 38%|███▊      | 151270/400000 [00:22<00:34, 7155.29it/s] 38%|███▊      | 151987/400000 [00:22<00:34, 7114.48it/s] 38%|███▊      | 152733/400000 [00:22<00:34, 7213.70it/s] 38%|███▊      | 153474/400000 [00:22<00:33, 7268.93it/s] 39%|███▊      | 154202/400000 [00:22<00:35, 6990.11it/s] 39%|███▊      | 154915/400000 [00:22<00:34, 7030.77it/s] 39%|███▉      | 155621/400000 [00:22<00:35, 6896.31it/s] 39%|███▉      | 156330/400000 [00:22<00:35, 6952.55it/s] 39%|███▉      | 157058/400000 [00:22<00:34, 7045.28it/s] 39%|███▉      | 157785/400000 [00:22<00:34, 7109.70it/s] 40%|███▉      | 158503/400000 [00:23<00:33, 7129.66it/s] 40%|███▉      | 159217/400000 [00:23<00:34, 7044.57it/s] 40%|███▉      | 159970/400000 [00:23<00:33, 7182.03it/s] 40%|████      | 160732/400000 [00:23<00:32, 7307.05it/s] 40%|████      | 161471/400000 [00:23<00:32, 7330.56it/s] 41%|████      | 162206/400000 [00:23<00:32, 7282.63it/s] 41%|████      | 162936/400000 [00:23<00:33, 7123.04it/s] 41%|████      | 163657/400000 [00:23<00:33, 7148.50it/s] 41%|████      | 164385/400000 [00:23<00:32, 7186.33it/s] 41%|████▏     | 165120/400000 [00:23<00:32, 7234.22it/s] 41%|████▏     | 165844/400000 [00:24<00:32, 7160.84it/s] 42%|████▏     | 166564/400000 [00:24<00:32, 7171.36it/s] 42%|████▏     | 167293/400000 [00:24<00:32, 7205.98it/s] 42%|████▏     | 168044/400000 [00:24<00:31, 7294.00it/s] 42%|████▏     | 168797/400000 [00:24<00:31, 7361.76it/s] 42%|████▏     | 169534/400000 [00:24<00:31, 7296.24it/s] 43%|████▎     | 170265/400000 [00:24<00:31, 7260.52it/s] 43%|████▎     | 170992/400000 [00:24<00:31, 7202.42it/s] 43%|████▎     | 171713/400000 [00:24<00:31, 7154.76it/s] 43%|████▎     | 172440/400000 [00:25<00:31, 7186.87it/s] 43%|████▎     | 173159/400000 [00:25<00:31, 7176.64it/s] 43%|████▎     | 173896/400000 [00:25<00:31, 7231.80it/s] 44%|████▎     | 174657/400000 [00:25<00:30, 7339.83it/s] 44%|████▍     | 175423/400000 [00:25<00:30, 7431.46it/s] 44%|████▍     | 176167/400000 [00:25<00:30, 7402.81it/s] 44%|████▍     | 176908/400000 [00:25<00:30, 7380.30it/s] 44%|████▍     | 177647/400000 [00:25<00:31, 6966.80it/s] 45%|████▍     | 178354/400000 [00:25<00:31, 6995.73it/s] 45%|████▍     | 179080/400000 [00:25<00:31, 7070.65it/s] 45%|████▍     | 179790/400000 [00:26<00:31, 7075.27it/s] 45%|████▌     | 180500/400000 [00:26<00:31, 6952.62it/s] 45%|████▌     | 181198/400000 [00:26<00:31, 6939.10it/s] 45%|████▌     | 181905/400000 [00:26<00:31, 6976.74it/s] 46%|████▌     | 182651/400000 [00:26<00:30, 7112.99it/s] 46%|████▌     | 183404/400000 [00:26<00:29, 7231.11it/s] 46%|████▌     | 184145/400000 [00:26<00:29, 7282.73it/s] 46%|████▌     | 184875/400000 [00:26<00:30, 7152.10it/s] 46%|████▋     | 185592/400000 [00:26<00:30, 7081.28it/s] 47%|████▋     | 186302/400000 [00:26<00:30, 7024.11it/s] 47%|████▋     | 187027/400000 [00:27<00:30, 7088.18it/s] 47%|████▋     | 187737/400000 [00:27<00:30, 7016.41it/s] 47%|████▋     | 188440/400000 [00:27<00:30, 6868.47it/s] 47%|████▋     | 189129/400000 [00:27<00:30, 6835.06it/s] 47%|████▋     | 189839/400000 [00:27<00:30, 6911.46it/s] 48%|████▊     | 190531/400000 [00:27<00:30, 6893.53it/s] 48%|████▊     | 191244/400000 [00:27<00:29, 6961.41it/s] 48%|████▊     | 191941/400000 [00:27<00:30, 6875.86it/s] 48%|████▊     | 192630/400000 [00:27<00:30, 6878.58it/s] 48%|████▊     | 193338/400000 [00:27<00:29, 6936.22it/s] 49%|████▊     | 194050/400000 [00:28<00:29, 6989.10it/s] 49%|████▊     | 194767/400000 [00:28<00:29, 7039.77it/s] 49%|████▉     | 195472/400000 [00:28<00:29, 6829.01it/s] 49%|████▉     | 196204/400000 [00:28<00:29, 6965.52it/s] 49%|████▉     | 196924/400000 [00:28<00:28, 7033.25it/s] 49%|████▉     | 197673/400000 [00:28<00:28, 7163.44it/s] 50%|████▉     | 198391/400000 [00:28<00:28, 7161.14it/s] 50%|████▉     | 199109/400000 [00:28<00:29, 6896.00it/s] 50%|████▉     | 199832/400000 [00:28<00:28, 6992.63it/s] 50%|█████     | 200552/400000 [00:28<00:28, 7051.18it/s] 50%|█████     | 201296/400000 [00:29<00:27, 7162.31it/s] 51%|█████     | 202014/400000 [00:29<00:27, 7126.49it/s] 51%|█████     | 202743/400000 [00:29<00:27, 7173.24it/s] 51%|█████     | 203489/400000 [00:29<00:27, 7255.45it/s] 51%|█████     | 204216/400000 [00:29<00:27, 7152.49it/s] 51%|█████     | 204954/400000 [00:29<00:27, 7217.91it/s] 51%|█████▏    | 205690/400000 [00:29<00:26, 7257.87it/s] 52%|█████▏    | 206417/400000 [00:29<00:26, 7218.42it/s] 52%|█████▏    | 207140/400000 [00:29<00:26, 7175.78it/s] 52%|█████▏    | 207862/400000 [00:30<00:26, 7188.86it/s] 52%|█████▏    | 208582/400000 [00:30<00:26, 7154.39it/s] 52%|█████▏    | 209298/400000 [00:30<00:26, 7126.12it/s] 53%|█████▎    | 210011/400000 [00:30<00:27, 6983.86it/s] 53%|█████▎    | 210744/400000 [00:30<00:26, 7082.92it/s] 53%|█████▎    | 211480/400000 [00:30<00:26, 7163.04it/s] 53%|█████▎    | 212209/400000 [00:30<00:26, 7199.10it/s] 53%|█████▎    | 212930/400000 [00:30<00:26, 7044.87it/s] 53%|█████▎    | 213644/400000 [00:30<00:26, 7072.36it/s] 54%|█████▎    | 214379/400000 [00:30<00:25, 7151.52it/s] 54%|█████▍    | 215110/400000 [00:31<00:25, 7197.57it/s] 54%|█████▍    | 215840/400000 [00:31<00:25, 7226.74it/s] 54%|█████▍    | 216564/400000 [00:31<00:25, 7148.62it/s] 54%|█████▍    | 217280/400000 [00:31<00:25, 7040.95it/s] 55%|█████▍    | 218010/400000 [00:31<00:25, 7116.46it/s] 55%|█████▍    | 218723/400000 [00:31<00:25, 7050.45it/s] 55%|█████▍    | 219480/400000 [00:31<00:25, 7196.54it/s] 55%|█████▌    | 220201/400000 [00:31<00:25, 7169.53it/s] 55%|█████▌    | 220933/400000 [00:31<00:24, 7212.33it/s] 55%|█████▌    | 221655/400000 [00:31<00:24, 7181.30it/s] 56%|█████▌    | 222374/400000 [00:32<00:24, 7168.57it/s] 56%|█████▌    | 223092/400000 [00:32<00:24, 7123.68it/s] 56%|█████▌    | 223815/400000 [00:32<00:24, 7154.79it/s] 56%|█████▌    | 224531/400000 [00:32<00:24, 7094.47it/s] 56%|█████▋    | 225241/400000 [00:32<00:24, 7064.05it/s] 56%|█████▋    | 225974/400000 [00:32<00:24, 7139.85it/s] 57%|█████▋    | 226716/400000 [00:32<00:23, 7221.10it/s] 57%|█████▋    | 227439/400000 [00:32<00:24, 7174.21it/s] 57%|█████▋    | 228157/400000 [00:32<00:24, 7098.26it/s] 57%|█████▋    | 228911/400000 [00:32<00:23, 7223.77it/s] 57%|█████▋    | 229665/400000 [00:33<00:23, 7315.62it/s] 58%|█████▊    | 230406/400000 [00:33<00:23, 7341.94it/s] 58%|█████▊    | 231167/400000 [00:33<00:22, 7420.29it/s] 58%|█████▊    | 231942/400000 [00:33<00:22, 7515.74it/s] 58%|█████▊    | 232710/400000 [00:33<00:22, 7561.86it/s] 58%|█████▊    | 233471/400000 [00:33<00:21, 7574.91it/s] 59%|█████▊    | 234229/400000 [00:33<00:22, 7489.17it/s] 59%|█████▊    | 234985/400000 [00:33<00:21, 7508.35it/s] 59%|█████▉    | 235737/400000 [00:33<00:22, 7395.77it/s] 59%|█████▉    | 236478/400000 [00:33<00:22, 7235.38it/s] 59%|█████▉    | 237221/400000 [00:34<00:22, 7289.05it/s] 59%|█████▉    | 237951/400000 [00:34<00:22, 7211.21it/s] 60%|█████▉    | 238673/400000 [00:34<00:22, 7093.10it/s] 60%|█████▉    | 239384/400000 [00:34<00:23, 6975.91it/s] 60%|██████    | 240091/400000 [00:34<00:22, 7002.99it/s] 60%|██████    | 240820/400000 [00:34<00:22, 7085.79it/s] 60%|██████    | 241583/400000 [00:34<00:21, 7239.79it/s] 61%|██████    | 242316/400000 [00:34<00:21, 7265.88it/s] 61%|██████    | 243053/400000 [00:34<00:21, 7294.77it/s] 61%|██████    | 243803/400000 [00:34<00:21, 7354.81it/s] 61%|██████    | 244549/400000 [00:35<00:21, 7385.66it/s] 61%|██████▏   | 245293/400000 [00:35<00:20, 7399.60it/s] 62%|██████▏   | 246034/400000 [00:35<00:21, 7324.74it/s] 62%|██████▏   | 246767/400000 [00:35<00:21, 7246.72it/s] 62%|██████▏   | 247493/400000 [00:35<00:21, 7122.02it/s] 62%|██████▏   | 248207/400000 [00:35<00:21, 7042.93it/s] 62%|██████▏   | 248947/400000 [00:35<00:21, 7145.61it/s] 62%|██████▏   | 249697/400000 [00:35<00:20, 7248.02it/s] 63%|██████▎   | 250423/400000 [00:35<00:20, 7149.93it/s] 63%|██████▎   | 251141/400000 [00:36<00:20, 7157.30it/s] 63%|██████▎   | 251893/400000 [00:36<00:20, 7261.42it/s] 63%|██████▎   | 252620/400000 [00:36<00:20, 7168.80it/s] 63%|██████▎   | 253371/400000 [00:36<00:20, 7266.95it/s] 64%|██████▎   | 254120/400000 [00:36<00:19, 7331.52it/s] 64%|██████▎   | 254892/400000 [00:36<00:19, 7441.68it/s] 64%|██████▍   | 255638/400000 [00:36<00:19, 7384.56it/s] 64%|██████▍   | 256378/400000 [00:36<00:19, 7198.50it/s] 64%|██████▍   | 257109/400000 [00:36<00:19, 7230.99it/s] 64%|██████▍   | 257834/400000 [00:36<00:19, 7192.63it/s] 65%|██████▍   | 258555/400000 [00:37<00:19, 7110.19it/s] 65%|██████▍   | 259267/400000 [00:37<00:20, 7029.50it/s] 65%|██████▍   | 259971/400000 [00:37<00:20, 6987.39it/s] 65%|██████▌   | 260671/400000 [00:37<00:19, 6990.73it/s] 65%|██████▌   | 261371/400000 [00:37<00:20, 6840.52it/s] 66%|██████▌   | 262139/400000 [00:37<00:19, 7070.27it/s] 66%|██████▌   | 262901/400000 [00:37<00:18, 7226.06it/s] 66%|██████▌   | 263663/400000 [00:37<00:18, 7337.55it/s] 66%|██████▌   | 264444/400000 [00:37<00:18, 7470.62it/s] 66%|██████▋   | 265194/400000 [00:37<00:18, 7449.45it/s] 66%|██████▋   | 265941/400000 [00:38<00:18, 7338.38it/s] 67%|██████▋   | 266677/400000 [00:38<00:18, 7300.08it/s] 67%|██████▋   | 267409/400000 [00:38<00:18, 7263.36it/s] 67%|██████▋   | 268137/400000 [00:38<00:18, 7267.09it/s] 67%|██████▋   | 268865/400000 [00:38<00:18, 6987.69it/s] 67%|██████▋   | 269577/400000 [00:38<00:18, 7024.98it/s] 68%|██████▊   | 270309/400000 [00:38<00:18, 7109.05it/s] 68%|██████▊   | 271045/400000 [00:38<00:17, 7180.73it/s] 68%|██████▊   | 271773/400000 [00:38<00:17, 7209.45it/s] 68%|██████▊   | 272495/400000 [00:38<00:17, 7121.91it/s] 68%|██████▊   | 273234/400000 [00:39<00:17, 7199.49it/s] 68%|██████▊   | 273980/400000 [00:39<00:17, 7274.03it/s] 69%|██████▊   | 274709/400000 [00:39<00:17, 7097.72it/s] 69%|██████▉   | 275446/400000 [00:39<00:17, 7176.78it/s] 69%|██████▉   | 276170/400000 [00:39<00:17, 7195.18it/s] 69%|██████▉   | 276891/400000 [00:39<00:17, 7065.21it/s] 69%|██████▉   | 277599/400000 [00:39<00:17, 7013.15it/s] 70%|██████▉   | 278303/400000 [00:39<00:17, 7020.81it/s] 70%|██████▉   | 279057/400000 [00:39<00:16, 7167.22it/s] 70%|██████▉   | 279791/400000 [00:39<00:16, 7216.89it/s] 70%|███████   | 280514/400000 [00:40<00:16, 7170.90it/s] 70%|███████   | 281232/400000 [00:40<00:16, 7161.47it/s] 70%|███████   | 281949/400000 [00:40<00:16, 7160.33it/s] 71%|███████   | 282666/400000 [00:40<00:16, 7160.21it/s] 71%|███████   | 283383/400000 [00:40<00:16, 7091.35it/s] 71%|███████   | 284093/400000 [00:40<00:16, 7007.06it/s] 71%|███████   | 284795/400000 [00:40<00:16, 6988.21it/s] 71%|███████▏  | 285495/400000 [00:40<00:16, 6927.93it/s] 72%|███████▏  | 286193/400000 [00:40<00:16, 6942.43it/s] 72%|███████▏  | 286888/400000 [00:41<00:16, 6893.31it/s] 72%|███████▏  | 287602/400000 [00:41<00:16, 6965.07it/s] 72%|███████▏  | 288300/400000 [00:41<00:16, 6967.95it/s] 72%|███████▏  | 289020/400000 [00:41<00:15, 7031.26it/s] 72%|███████▏  | 289726/400000 [00:41<00:15, 7038.89it/s] 73%|███████▎  | 290431/400000 [00:41<00:15, 6919.05it/s] 73%|███████▎  | 291141/400000 [00:41<00:15, 6969.57it/s] 73%|███████▎  | 291882/400000 [00:41<00:15, 7094.70it/s] 73%|███████▎  | 292593/400000 [00:41<00:15, 7076.42it/s] 73%|███████▎  | 293302/400000 [00:41<00:15, 7079.85it/s] 74%|███████▎  | 294011/400000 [00:42<00:15, 6972.21it/s] 74%|███████▎  | 294723/400000 [00:42<00:15, 7013.78it/s] 74%|███████▍  | 295426/400000 [00:42<00:14, 7018.08it/s] 74%|███████▍  | 296141/400000 [00:42<00:14, 7049.65it/s] 74%|███████▍  | 296847/400000 [00:42<00:14, 7002.07it/s] 74%|███████▍  | 297548/400000 [00:42<00:15, 6755.05it/s] 75%|███████▍  | 298234/400000 [00:42<00:14, 6785.74it/s] 75%|███████▍  | 298945/400000 [00:42<00:14, 6876.83it/s] 75%|███████▍  | 299664/400000 [00:42<00:14, 6963.97it/s] 75%|███████▌  | 300380/400000 [00:42<00:14, 7020.70it/s] 75%|███████▌  | 301084/400000 [00:43<00:14, 6924.90it/s] 75%|███████▌  | 301783/400000 [00:43<00:14, 6942.83it/s] 76%|███████▌  | 302522/400000 [00:43<00:13, 7068.80it/s] 76%|███████▌  | 303230/400000 [00:43<00:13, 6976.86it/s] 76%|███████▌  | 303929/400000 [00:43<00:14, 6849.96it/s] 76%|███████▌  | 304644/400000 [00:43<00:13, 6935.19it/s] 76%|███████▋  | 305401/400000 [00:43<00:13, 7111.64it/s] 77%|███████▋  | 306160/400000 [00:43<00:12, 7246.14it/s] 77%|███████▋  | 306929/400000 [00:43<00:12, 7372.19it/s] 77%|███████▋  | 307669/400000 [00:43<00:12, 7273.98it/s] 77%|███████▋  | 308399/400000 [00:44<00:12, 7111.22it/s] 77%|███████▋  | 309152/400000 [00:44<00:12, 7230.58it/s] 77%|███████▋  | 309877/400000 [00:44<00:12, 7204.03it/s] 78%|███████▊  | 310599/400000 [00:44<00:12, 7207.34it/s] 78%|███████▊  | 311321/400000 [00:44<00:12, 7152.83it/s] 78%|███████▊  | 312038/400000 [00:44<00:12, 7045.41it/s] 78%|███████▊  | 312753/400000 [00:44<00:12, 7075.59it/s] 78%|███████▊  | 313462/400000 [00:44<00:12, 7058.65it/s] 79%|███████▊  | 314169/400000 [00:44<00:12, 7032.83it/s] 79%|███████▊  | 314873/400000 [00:44<00:12, 7004.52it/s] 79%|███████▉  | 315574/400000 [00:45<00:12, 6832.44it/s] 79%|███████▉  | 316311/400000 [00:45<00:11, 6983.55it/s] 79%|███████▉  | 317011/400000 [00:45<00:11, 6950.96it/s] 79%|███████▉  | 317741/400000 [00:45<00:11, 7050.28it/s] 80%|███████▉  | 318452/400000 [00:45<00:11, 7065.84it/s] 80%|███████▉  | 319160/400000 [00:45<00:11, 6901.80it/s] 80%|███████▉  | 319852/400000 [00:45<00:11, 6851.96it/s] 80%|████████  | 320557/400000 [00:45<00:11, 6907.79it/s] 80%|████████  | 321249/400000 [00:45<00:11, 6842.72it/s] 80%|████████  | 321965/400000 [00:46<00:11, 6934.23it/s] 81%|████████  | 322666/400000 [00:46<00:11, 6953.46it/s] 81%|████████  | 323362/400000 [00:46<00:11, 6883.97it/s] 81%|████████  | 324051/400000 [00:46<00:11, 6880.96it/s] 81%|████████  | 324752/400000 [00:46<00:10, 6915.61it/s] 81%|████████▏ | 325462/400000 [00:46<00:10, 6969.45it/s] 82%|████████▏ | 326160/400000 [00:46<00:10, 6924.00it/s] 82%|████████▏ | 326877/400000 [00:46<00:10, 6995.29it/s] 82%|████████▏ | 327602/400000 [00:46<00:10, 7069.56it/s] 82%|████████▏ | 328310/400000 [00:46<00:10, 7046.08it/s] 82%|████████▏ | 329020/400000 [00:47<00:10, 7059.79it/s] 82%|████████▏ | 329727/400000 [00:47<00:10, 6873.26it/s] 83%|████████▎ | 330425/400000 [00:47<00:10, 6904.60it/s] 83%|████████▎ | 331142/400000 [00:47<00:09, 6981.34it/s] 83%|████████▎ | 331841/400000 [00:47<00:09, 6922.71it/s] 83%|████████▎ | 332534/400000 [00:47<00:09, 6900.23it/s] 83%|████████▎ | 333225/400000 [00:47<00:09, 6817.14it/s] 83%|████████▎ | 333912/400000 [00:47<00:09, 6832.57it/s] 84%|████████▎ | 334641/400000 [00:47<00:09, 6961.59it/s] 84%|████████▍ | 335352/400000 [00:47<00:09, 7003.44it/s] 84%|████████▍ | 336054/400000 [00:48<00:09, 7006.91it/s] 84%|████████▍ | 336756/400000 [00:48<00:09, 6958.10it/s] 84%|████████▍ | 337505/400000 [00:48<00:08, 7107.44it/s] 85%|████████▍ | 338227/400000 [00:48<00:08, 7138.22it/s] 85%|████████▍ | 338942/400000 [00:48<00:08, 7058.49it/s] 85%|████████▍ | 339667/400000 [00:48<00:08, 7111.71it/s] 85%|████████▌ | 340379/400000 [00:48<00:08, 7092.67it/s] 85%|████████▌ | 341122/400000 [00:48<00:08, 7189.64it/s] 85%|████████▌ | 341842/400000 [00:48<00:08, 7150.55it/s] 86%|████████▌ | 342558/400000 [00:48<00:08, 6845.53it/s] 86%|████████▌ | 343284/400000 [00:49<00:08, 6963.05it/s] 86%|████████▌ | 344029/400000 [00:49<00:07, 7102.30it/s] 86%|████████▌ | 344769/400000 [00:49<00:07, 7188.79it/s] 86%|████████▋ | 345528/400000 [00:49<00:07, 7302.95it/s] 87%|████████▋ | 346261/400000 [00:49<00:07, 7252.62it/s] 87%|████████▋ | 347013/400000 [00:49<00:07, 7330.12it/s] 87%|████████▋ | 347776/400000 [00:49<00:07, 7417.01it/s] 87%|████████▋ | 348551/400000 [00:49<00:06, 7511.61it/s] 87%|████████▋ | 349333/400000 [00:49<00:06, 7600.78it/s] 88%|████████▊ | 350095/400000 [00:49<00:06, 7574.39it/s] 88%|████████▊ | 350854/400000 [00:50<00:06, 7447.35it/s] 88%|████████▊ | 351600/400000 [00:50<00:06, 7196.43it/s] 88%|████████▊ | 352333/400000 [00:50<00:06, 7233.43it/s] 88%|████████▊ | 353063/400000 [00:50<00:06, 7250.96it/s] 88%|████████▊ | 353790/400000 [00:50<00:06, 7253.14it/s] 89%|████████▊ | 354517/400000 [00:50<00:06, 7189.81it/s] 89%|████████▉ | 355237/400000 [00:50<00:06, 6920.00it/s] 89%|████████▉ | 355999/400000 [00:50<00:06, 7114.05it/s] 89%|████████▉ | 356759/400000 [00:50<00:05, 7251.43it/s] 89%|████████▉ | 357525/400000 [00:50<00:05, 7367.54it/s] 90%|████████▉ | 358278/400000 [00:51<00:05, 7413.07it/s] 90%|████████▉ | 359055/400000 [00:51<00:05, 7513.61it/s] 90%|████████▉ | 359828/400000 [00:51<00:05, 7575.05it/s] 90%|█████████ | 360605/400000 [00:51<00:05, 7631.97it/s] 90%|█████████ | 361370/400000 [00:51<00:05, 7599.96it/s] 91%|█████████ | 362138/400000 [00:51<00:04, 7621.45it/s] 91%|█████████ | 362913/400000 [00:51<00:04, 7659.43it/s] 91%|█████████ | 363680/400000 [00:51<00:04, 7650.55it/s] 91%|█████████ | 364447/400000 [00:51<00:04, 7654.37it/s] 91%|█████████▏| 365213/400000 [00:51<00:04, 7650.99it/s] 91%|█████████▏| 365985/400000 [00:52<00:04, 7670.16it/s] 92%|█████████▏| 366759/400000 [00:52<00:04, 7690.58it/s] 92%|█████████▏| 367529/400000 [00:52<00:04, 7487.93it/s] 92%|█████████▏| 368280/400000 [00:52<00:04, 7223.87it/s] 92%|█████████▏| 369006/400000 [00:52<00:04, 7203.73it/s] 92%|█████████▏| 369745/400000 [00:52<00:04, 7256.94it/s] 93%|█████████▎| 370482/400000 [00:52<00:04, 7289.77it/s] 93%|█████████▎| 371213/400000 [00:52<00:03, 7226.24it/s] 93%|█████████▎| 371937/400000 [00:52<00:03, 7158.83it/s] 93%|█████████▎| 372654/400000 [00:53<00:03, 7124.14it/s] 93%|█████████▎| 373392/400000 [00:53<00:03, 7198.96it/s] 94%|█████████▎| 374113/400000 [00:53<00:03, 7183.14it/s] 94%|█████████▎| 374832/400000 [00:53<00:03, 7133.59it/s] 94%|█████████▍| 375563/400000 [00:53<00:03, 7182.91it/s] 94%|█████████▍| 376282/400000 [00:53<00:03, 7183.59it/s] 94%|█████████▍| 377015/400000 [00:53<00:03, 7225.71it/s] 94%|█████████▍| 377738/400000 [00:53<00:03, 7206.62it/s] 95%|█████████▍| 378490/400000 [00:53<00:02, 7295.68it/s] 95%|█████████▍| 379234/400000 [00:53<00:02, 7337.66it/s] 95%|█████████▍| 379969/400000 [00:54<00:02, 7163.24it/s] 95%|█████████▌| 380710/400000 [00:54<00:02, 7233.58it/s] 95%|█████████▌| 381435/400000 [00:54<00:02, 7126.72it/s] 96%|█████████▌| 382176/400000 [00:54<00:02, 7207.40it/s] 96%|█████████▌| 382898/400000 [00:54<00:02, 7141.02it/s] 96%|█████████▌| 383613/400000 [00:54<00:02, 7089.69it/s] 96%|█████████▌| 384323/400000 [00:54<00:02, 7070.47it/s] 96%|█████████▋| 385032/400000 [00:54<00:02, 7074.20it/s] 96%|█████████▋| 385752/400000 [00:54<00:02, 7108.45it/s] 97%|█████████▋| 386511/400000 [00:54<00:01, 7245.35it/s] 97%|█████████▋| 387282/400000 [00:55<00:01, 7377.18it/s] 97%|█████████▋| 388029/400000 [00:55<00:01, 7402.26it/s] 97%|█████████▋| 388771/400000 [00:55<00:01, 7155.37it/s] 97%|█████████▋| 389489/400000 [00:55<00:01, 7066.02it/s] 98%|█████████▊| 390231/400000 [00:55<00:01, 7166.69it/s] 98%|█████████▊| 390961/400000 [00:55<00:01, 7205.88it/s] 98%|█████████▊| 391683/400000 [00:55<00:01, 7140.14it/s] 98%|█████████▊| 392399/400000 [00:55<00:01, 6951.11it/s] 98%|█████████▊| 393116/400000 [00:55<00:00, 7015.25it/s] 98%|█████████▊| 393862/400000 [00:55<00:00, 7141.05it/s] 99%|█████████▊| 394578/400000 [00:56<00:00, 7106.53it/s] 99%|█████████▉| 395323/400000 [00:56<00:00, 7206.09it/s] 99%|█████████▉| 396045/400000 [00:56<00:00, 7166.20it/s] 99%|█████████▉| 396763/400000 [00:56<00:00, 7139.46it/s] 99%|█████████▉| 397478/400000 [00:56<00:00, 7087.77it/s]100%|█████████▉| 398188/400000 [00:56<00:00, 6969.00it/s]100%|█████████▉| 398925/400000 [00:56<00:00, 7082.65it/s]100%|█████████▉| 399635/400000 [00:56<00:00, 6985.05it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7035.56it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f8aef03dd68> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011258869041530188 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.011674179879319308 	 Accuracy: 46

  model saves at 46% accuracy 

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
2020-05-20 05:03:31.175841: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-20 05:03:31.180706: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-20 05:03:31.180873: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561d30311460 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-20 05:03:31.180889: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f8a92a24080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 8.0653 - accuracy: 0.4740
 2000/25000 [=>............................] - ETA: 10s - loss: 7.9426 - accuracy: 0.4820
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.8251 - accuracy: 0.4897 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7433 - accuracy: 0.4950
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6636 - accuracy: 0.5002
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6794 - accuracy: 0.4992
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6469 - accuracy: 0.5013
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6436 - accuracy: 0.5015
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6070 - accuracy: 0.5039
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6743 - accuracy: 0.4995
11000/25000 [============>.................] - ETA: 4s - loss: 7.6583 - accuracy: 0.5005
12000/25000 [=============>................] - ETA: 4s - loss: 7.6768 - accuracy: 0.4993
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6784 - accuracy: 0.4992
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6863 - accuracy: 0.4987
15000/25000 [=================>............] - ETA: 3s - loss: 7.6717 - accuracy: 0.4997
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6570 - accuracy: 0.5006
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6567 - accuracy: 0.5006
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6794 - accuracy: 0.4992
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6763 - accuracy: 0.4994
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6827 - accuracy: 0.4990
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6783 - accuracy: 0.4992
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6553 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6634 - accuracy: 0.5002
25000/25000 [==============================] - 10s 401us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f8a47fd64e0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f8a8cc3f7b8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3321 - crf_viterbi_accuracy: 0.6533 - val_loss: 1.2728 - val_crf_viterbi_accuracy: 0.6800

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

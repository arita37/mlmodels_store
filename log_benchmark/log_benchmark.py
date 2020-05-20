
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f1998bbde80> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-20 12:13:25.999051
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-20 12:13:26.003500
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-20 12:13:26.007140
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-20 12:13:26.010633
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f19a49873c8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 1s 1s/step - loss: 354283.7500
Epoch 2/10

1/1 [==============================] - 0s 118ms/step - loss: 264686.4375
Epoch 3/10

1/1 [==============================] - 0s 107ms/step - loss: 161022.8125
Epoch 4/10

1/1 [==============================] - 0s 96ms/step - loss: 85889.8984
Epoch 5/10

1/1 [==============================] - 0s 105ms/step - loss: 44661.0312
Epoch 6/10

1/1 [==============================] - 0s 108ms/step - loss: 25051.9551
Epoch 7/10

1/1 [==============================] - 0s 129ms/step - loss: 15528.3301
Epoch 8/10

1/1 [==============================] - 0s 106ms/step - loss: 10464.2666
Epoch 9/10

1/1 [==============================] - 0s 106ms/step - loss: 7579.5615
Epoch 10/10

1/1 [==============================] - 0s 99ms/step - loss: 5803.6543

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.27742672e+00 -8.70338440e-01 -1.18036938e+00  4.46339965e-01
   4.29948509e-01 -1.59693718e+00  1.21475661e+00  6.09261334e-01
   2.88452506e-02  6.90194428e-01 -8.88067007e-01  9.60734367e-01
  -1.92367697e+00 -1.09980142e+00  8.92367601e-01  1.32997441e+00
  -1.76520908e+00  5.39214730e-01  1.55363154e+00 -1.57597494e+00
   3.25806051e-01  9.78198886e-01  4.22792405e-01 -1.98710012e+00
  -4.33351517e-01  1.21553600e-01  3.65874693e-02 -6.92007840e-02
  -1.06546521e+00 -1.34476364e-01  1.15351133e-01  8.35248768e-01
   6.54790401e-01  1.06144321e+00 -8.60459924e-01 -1.01898170e+00
  -2.13782489e-02 -3.73931587e-01  1.15385759e+00 -8.19415808e-01
  -6.86770439e-01 -9.35458660e-01 -1.04516268e+00 -5.08895397e-01
   9.14320588e-01  5.25777936e-02 -6.91475987e-01 -8.94623995e-03
   2.45413709e+00 -1.43071795e+00 -7.35371768e-01 -7.09619880e-01
   1.21344566e-01  1.31821251e+00 -1.12921667e+00  1.92069665e-01
   1.04014337e+00  1.93085194e-01  3.81969988e-01 -4.04846400e-01
  -1.20092764e-01  1.90076113e+00 -3.68768573e-01 -6.46613598e-01
  -9.84254241e-01  1.29429436e+00  2.37730026e-01  8.62014055e-01
   8.57137978e-01 -1.71348059e+00  7.27198958e-01  9.94052231e-01
  -3.86183083e-01  2.96464324e-01 -6.88809216e-01 -1.48570299e+00
   1.22950745e+00  4.26150620e-01 -1.00977612e+00 -1.58081722e+00
   4.43795443e-01 -4.23905164e-01  8.53275478e-01  8.34625185e-01
   6.91467583e-01 -1.54063344e+00  5.49490809e-01  7.21045807e-02
   1.18021083e+00 -1.32180572e-01 -1.30786467e+00  1.42361522e+00
   1.09047878e+00 -1.34884357e+00  3.49953353e-01  1.43131542e+00
   2.86877811e-01 -1.34610581e+00  2.33119026e-01 -4.27377462e-01
  -3.29604387e-01  1.11480713e+00  1.93446875e+00 -1.00720465e+00
  -1.78005075e+00  2.21033067e-01  1.13825417e+00 -1.60304606e+00
   3.61817718e-01 -1.67405629e+00 -1.00469148e+00 -1.69217885e-01
   1.84102869e+00 -1.49003401e-01  6.85154438e-01  6.22382700e-01
   1.53775394e+00  6.96260154e-01 -4.74710494e-01 -1.16615152e+00
   1.99308693e-01  1.04531279e+01  8.88622379e+00  8.80394554e+00
   8.90734768e+00  7.00168848e+00  8.82802010e+00  7.89741468e+00
   6.61134005e+00  1.07830334e+01  8.50550175e+00  8.47893810e+00
   7.71987391e+00  8.66910172e+00  9.39848614e+00  9.41254520e+00
   8.62559509e+00  8.62737274e+00  9.45478344e+00  7.47086334e+00
   9.76633739e+00  7.16067266e+00  8.86914635e+00  8.75697517e+00
   9.30250359e+00  9.81074333e+00  8.28788853e+00  7.81579065e+00
   8.75710583e+00  8.85330391e+00  9.17325115e+00  7.89167929e+00
   1.01415386e+01  6.93132877e+00  8.02370739e+00  8.11142445e+00
   8.61268044e+00  7.98514128e+00  9.39328766e+00  9.52318764e+00
   9.59488487e+00  8.39900017e+00  8.64990139e+00  1.06021404e+01
   9.03213596e+00  8.51941586e+00  8.55710506e+00  9.21444702e+00
   7.62008715e+00  1.00475769e+01  7.56392670e+00  8.53563690e+00
   9.13970661e+00  7.40333319e+00  8.63856411e+00  8.88773632e+00
   9.96545601e+00  6.39492702e+00  8.11784744e+00  9.32128620e+00
   3.21081591e+00  1.56905293e+00  4.79582191e-01  7.60210693e-01
   1.71659172e+00  1.95053875e-01  4.79596376e-01  1.52230477e+00
   1.54220438e+00  2.25283051e+00  2.39835405e+00  4.64129567e-01
   9.53404307e-01  1.49660695e+00  1.43814182e+00  1.62243509e+00
   1.66403079e+00  4.85636473e-01  6.23927474e-01  8.29572678e-01
   1.11899936e+00  1.79783702e+00  4.29024518e-01  5.54266155e-01
   4.18240070e-01  2.58956432e-01  1.46673608e+00  1.66684985e-01
   1.31115937e+00  1.71152067e+00  2.12133193e+00  7.46396601e-01
   1.85051322e+00  8.68637443e-01  5.28184235e-01  7.68713951e-01
   8.29020858e-01  2.52731085e-01  3.49708223e+00  7.53048658e-01
   1.59782052e+00  2.07907557e-01  6.51678324e-01  4.03647482e-01
   5.51127195e-01  8.74426782e-01  7.48392940e-01  4.75310922e-01
   1.87241626e+00  2.88333845e+00  2.55678296e-01  5.63676655e-01
   1.71021044e+00  1.69451511e+00  5.45802355e-01  2.40618849e+00
   2.02610016e-01  8.86438727e-01  1.45419347e+00  3.58550131e-01
   1.74996853e+00  1.95886016e-01  3.09758902e+00  1.40161943e+00
   5.76697707e-01  4.33923006e-01  3.68427634e-01  2.25068712e+00
   9.32599068e-01  6.70354724e-01  1.94198430e-01  1.67758763e-01
   1.31746304e+00  2.05005944e-01  2.10645628e+00  1.46670389e+00
   7.55706847e-01  3.20526123e-01  1.73458612e+00  2.36398315e+00
   2.85164261e+00  1.49750710e-01  9.06000316e-01  1.06312966e+00
   1.53205514e-01  1.97437787e+00  6.94120765e-01  8.64942133e-01
   1.62983644e+00  2.33248854e+00  4.41474319e-01  1.45467019e+00
   2.18171000e-01  1.17983413e+00  1.02061820e+00  1.51067734e+00
   7.63163149e-01  2.24803877e+00  2.07405615e+00  2.00555897e+00
   1.56337047e+00  3.61156106e-01  1.35522676e+00  1.74911880e+00
   5.38115442e-01  8.22701752e-01  6.30491674e-01  2.61275470e-01
   2.39742577e-01  1.43444467e+00  1.59473491e+00  2.18270350e+00
   2.12727165e+00  2.49299097e+00  7.27099478e-01  9.67821240e-01
   1.71919465e+00  1.20400858e+00  2.23680592e+00  1.34687400e+00
   9.04449821e-02  6.99878979e+00  7.85654449e+00  9.51785851e+00
   8.12893867e+00  7.91411686e+00  8.79157448e+00  9.28933811e+00
   9.39254475e+00  9.51078606e+00  9.84817600e+00  9.34107018e+00
   1.02206411e+01  9.44204807e+00  9.76898098e+00  9.16483593e+00
   9.35498333e+00  8.72769547e+00  9.99132347e+00  9.63207912e+00
   9.04247189e+00  9.55032921e+00  8.15296936e+00  7.89602423e+00
   8.55422878e+00  9.33648300e+00  9.60040855e+00  1.01160650e+01
   8.87816143e+00  7.84968519e+00  9.31975365e+00  9.05745316e+00
   9.21798325e+00  8.37532997e+00  8.63434696e+00  9.50680733e+00
   8.17494869e+00  8.28193569e+00  9.61985207e+00  8.87835979e+00
   7.89267206e+00  7.55996513e+00  8.22019577e+00  8.99029255e+00
   9.52686596e+00  9.16850471e+00  9.13497162e+00  8.97067738e+00
   1.01479912e+01  9.99946213e+00  9.45673943e+00  9.51796818e+00
   8.98272896e+00  8.22351170e+00  9.43573380e+00  8.80214977e+00
   8.30634308e+00  8.41915035e+00  8.96729183e+00  9.45947170e+00
  -8.30803108e+00 -7.71670723e+00  5.91115379e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-20 12:13:34.779797
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.4396
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-20 12:13:34.783584
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8748.87
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-20 12:13:34.786646
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.5465
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-20 12:13:34.789574
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -782.523
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139747571156416
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139746344215048
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139746344215552
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139746344216056
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139746344216560
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139746344217064

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f1984548e10> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.487680
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.452605
grad_step = 000002, loss = 0.423823
grad_step = 000003, loss = 0.393187
grad_step = 000004, loss = 0.360539
grad_step = 000005, loss = 0.331866
grad_step = 000006, loss = 0.318756
grad_step = 000007, loss = 0.313139
grad_step = 000008, loss = 0.296416
grad_step = 000009, loss = 0.278064
grad_step = 000010, loss = 0.265409
grad_step = 000011, loss = 0.257471
grad_step = 000012, loss = 0.250563
grad_step = 000013, loss = 0.242526
grad_step = 000014, loss = 0.232765
grad_step = 000015, loss = 0.221589
grad_step = 000016, loss = 0.210906
grad_step = 000017, loss = 0.202052
grad_step = 000018, loss = 0.193867
grad_step = 000019, loss = 0.185054
grad_step = 000020, loss = 0.176108
grad_step = 000021, loss = 0.167500
grad_step = 000022, loss = 0.159673
grad_step = 000023, loss = 0.152620
grad_step = 000024, loss = 0.145589
grad_step = 000025, loss = 0.138124
grad_step = 000026, loss = 0.130586
grad_step = 000027, loss = 0.123684
grad_step = 000028, loss = 0.117516
grad_step = 000029, loss = 0.111448
grad_step = 000030, loss = 0.105237
grad_step = 000031, loss = 0.099088
grad_step = 000032, loss = 0.093273
grad_step = 000033, loss = 0.087947
grad_step = 000034, loss = 0.082859
grad_step = 000035, loss = 0.077819
grad_step = 000036, loss = 0.072982
grad_step = 000037, loss = 0.068466
grad_step = 000038, loss = 0.064201
grad_step = 000039, loss = 0.060142
grad_step = 000040, loss = 0.056150
grad_step = 000041, loss = 0.052257
grad_step = 000042, loss = 0.048695
grad_step = 000043, loss = 0.045370
grad_step = 000044, loss = 0.042211
grad_step = 000045, loss = 0.039141
grad_step = 000046, loss = 0.036236
grad_step = 000047, loss = 0.033593
grad_step = 000048, loss = 0.031060
grad_step = 000049, loss = 0.028621
grad_step = 000050, loss = 0.026355
grad_step = 000051, loss = 0.024282
grad_step = 000052, loss = 0.022350
grad_step = 000053, loss = 0.020516
grad_step = 000054, loss = 0.018822
grad_step = 000055, loss = 0.017245
grad_step = 000056, loss = 0.015803
grad_step = 000057, loss = 0.014451
grad_step = 000058, loss = 0.013189
grad_step = 000059, loss = 0.012033
grad_step = 000060, loss = 0.010998
grad_step = 000061, loss = 0.010027
grad_step = 000062, loss = 0.009138
grad_step = 000063, loss = 0.008349
grad_step = 000064, loss = 0.007637
grad_step = 000065, loss = 0.006974
grad_step = 000066, loss = 0.006383
grad_step = 000067, loss = 0.005859
grad_step = 000068, loss = 0.005391
grad_step = 000069, loss = 0.004963
grad_step = 000070, loss = 0.004567
grad_step = 000071, loss = 0.004225
grad_step = 000072, loss = 0.003928
grad_step = 000073, loss = 0.003669
grad_step = 000074, loss = 0.003464
grad_step = 000075, loss = 0.003288
grad_step = 000076, loss = 0.003129
grad_step = 000077, loss = 0.002970
grad_step = 000078, loss = 0.002823
grad_step = 000079, loss = 0.002701
grad_step = 000080, loss = 0.002616
grad_step = 000081, loss = 0.002562
grad_step = 000082, loss = 0.002512
grad_step = 000083, loss = 0.002458
grad_step = 000084, loss = 0.002394
grad_step = 000085, loss = 0.002339
grad_step = 000086, loss = 0.002303
grad_step = 000087, loss = 0.002286
grad_step = 000088, loss = 0.002287
grad_step = 000089, loss = 0.002294
grad_step = 000090, loss = 0.002305
grad_step = 000091, loss = 0.002266
grad_step = 000092, loss = 0.002211
grad_step = 000093, loss = 0.002178
grad_step = 000094, loss = 0.002187
grad_step = 000095, loss = 0.002206
grad_step = 000096, loss = 0.002187
grad_step = 000097, loss = 0.002150
grad_step = 000098, loss = 0.002126
grad_step = 000099, loss = 0.002132
grad_step = 000100, loss = 0.002143
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002129
grad_step = 000102, loss = 0.002102
grad_step = 000103, loss = 0.002080
grad_step = 000104, loss = 0.002077
grad_step = 000105, loss = 0.002083
grad_step = 000106, loss = 0.002080
grad_step = 000107, loss = 0.002066
grad_step = 000108, loss = 0.002046
grad_step = 000109, loss = 0.002035
grad_step = 000110, loss = 0.002036
grad_step = 000111, loss = 0.002042
grad_step = 000112, loss = 0.002039
grad_step = 000113, loss = 0.002024
grad_step = 000114, loss = 0.002013
grad_step = 000115, loss = 0.002008
grad_step = 000116, loss = 0.002005
grad_step = 000117, loss = 0.001997
grad_step = 000118, loss = 0.001984
grad_step = 000119, loss = 0.001974
grad_step = 000120, loss = 0.001972
grad_step = 000121, loss = 0.001974
grad_step = 000122, loss = 0.001974
grad_step = 000123, loss = 0.001973
grad_step = 000124, loss = 0.001976
grad_step = 000125, loss = 0.001997
grad_step = 000126, loss = 0.002036
grad_step = 000127, loss = 0.002096
grad_step = 000128, loss = 0.002072
grad_step = 000129, loss = 0.002000
grad_step = 000130, loss = 0.001942
grad_step = 000131, loss = 0.001979
grad_step = 000132, loss = 0.002020
grad_step = 000133, loss = 0.001969
grad_step = 000134, loss = 0.001919
grad_step = 000135, loss = 0.001940
grad_step = 000136, loss = 0.001964
grad_step = 000137, loss = 0.001941
grad_step = 000138, loss = 0.001905
grad_step = 000139, loss = 0.001923
grad_step = 000140, loss = 0.001946
grad_step = 000141, loss = 0.001929
grad_step = 000142, loss = 0.001918
grad_step = 000143, loss = 0.001954
grad_step = 000144, loss = 0.002004
grad_step = 000145, loss = 0.002035
grad_step = 000146, loss = 0.002058
grad_step = 000147, loss = 0.002064
grad_step = 000148, loss = 0.002027
grad_step = 000149, loss = 0.001928
grad_step = 000150, loss = 0.001878
grad_step = 000151, loss = 0.001904
grad_step = 000152, loss = 0.001951
grad_step = 000153, loss = 0.001968
grad_step = 000154, loss = 0.001956
grad_step = 000155, loss = 0.001931
grad_step = 000156, loss = 0.001890
grad_step = 000157, loss = 0.001865
grad_step = 000158, loss = 0.001877
grad_step = 000159, loss = 0.001908
grad_step = 000160, loss = 0.001922
grad_step = 000161, loss = 0.001913
grad_step = 000162, loss = 0.001899
grad_step = 000163, loss = 0.001882
grad_step = 000164, loss = 0.001866
grad_step = 000165, loss = 0.001853
grad_step = 000166, loss = 0.001852
grad_step = 000167, loss = 0.001864
grad_step = 000168, loss = 0.001874
grad_step = 000169, loss = 0.001882
grad_step = 000170, loss = 0.001889
grad_step = 000171, loss = 0.001901
grad_step = 000172, loss = 0.001912
grad_step = 000173, loss = 0.001919
grad_step = 000174, loss = 0.001922
grad_step = 000175, loss = 0.001931
grad_step = 000176, loss = 0.001933
grad_step = 000177, loss = 0.001933
grad_step = 000178, loss = 0.001918
grad_step = 000179, loss = 0.001902
grad_step = 000180, loss = 0.001884
grad_step = 000181, loss = 0.001866
grad_step = 000182, loss = 0.001847
grad_step = 000183, loss = 0.001835
grad_step = 000184, loss = 0.001829
grad_step = 000185, loss = 0.001828
grad_step = 000186, loss = 0.001828
grad_step = 000187, loss = 0.001829
grad_step = 000188, loss = 0.001834
grad_step = 000189, loss = 0.001843
grad_step = 000190, loss = 0.001858
grad_step = 000191, loss = 0.001880
grad_step = 000192, loss = 0.001919
grad_step = 000193, loss = 0.001980
grad_step = 000194, loss = 0.002078
grad_step = 000195, loss = 0.002179
grad_step = 000196, loss = 0.002263
grad_step = 000197, loss = 0.002199
grad_step = 000198, loss = 0.002024
grad_step = 000199, loss = 0.001847
grad_step = 000200, loss = 0.001826
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001934
grad_step = 000202, loss = 0.002020
grad_step = 000203, loss = 0.001987
grad_step = 000204, loss = 0.001867
grad_step = 000205, loss = 0.001809
grad_step = 000206, loss = 0.001860
grad_step = 000207, loss = 0.001928
grad_step = 000208, loss = 0.001921
grad_step = 000209, loss = 0.001848
grad_step = 000210, loss = 0.001803
grad_step = 000211, loss = 0.001831
grad_step = 000212, loss = 0.001877
grad_step = 000213, loss = 0.001879
grad_step = 000214, loss = 0.001837
grad_step = 000215, loss = 0.001802
grad_step = 000216, loss = 0.001806
grad_step = 000217, loss = 0.001835
grad_step = 000218, loss = 0.001848
grad_step = 000219, loss = 0.001831
grad_step = 000220, loss = 0.001804
grad_step = 000221, loss = 0.001792
grad_step = 000222, loss = 0.001802
grad_step = 000223, loss = 0.001819
grad_step = 000224, loss = 0.001823
grad_step = 000225, loss = 0.001812
grad_step = 000226, loss = 0.001795
grad_step = 000227, loss = 0.001785
grad_step = 000228, loss = 0.001788
grad_step = 000229, loss = 0.001796
grad_step = 000230, loss = 0.001803
grad_step = 000231, loss = 0.001803
grad_step = 000232, loss = 0.001796
grad_step = 000233, loss = 0.001787
grad_step = 000234, loss = 0.001780
grad_step = 000235, loss = 0.001776
grad_step = 000236, loss = 0.001777
grad_step = 000237, loss = 0.001779
grad_step = 000238, loss = 0.001783
grad_step = 000239, loss = 0.001786
grad_step = 000240, loss = 0.001787
grad_step = 000241, loss = 0.001788
grad_step = 000242, loss = 0.001788
grad_step = 000243, loss = 0.001788
grad_step = 000244, loss = 0.001787
grad_step = 000245, loss = 0.001787
grad_step = 000246, loss = 0.001788
grad_step = 000247, loss = 0.001791
grad_step = 000248, loss = 0.001795
grad_step = 000249, loss = 0.001805
grad_step = 000250, loss = 0.001818
grad_step = 000251, loss = 0.001841
grad_step = 000252, loss = 0.001872
grad_step = 000253, loss = 0.001920
grad_step = 000254, loss = 0.001967
grad_step = 000255, loss = 0.002021
grad_step = 000256, loss = 0.002028
grad_step = 000257, loss = 0.001999
grad_step = 000258, loss = 0.001910
grad_step = 000259, loss = 0.001818
grad_step = 000260, loss = 0.001760
grad_step = 000261, loss = 0.001763
grad_step = 000262, loss = 0.001806
grad_step = 000263, loss = 0.001850
grad_step = 000264, loss = 0.001864
grad_step = 000265, loss = 0.001829
grad_step = 000266, loss = 0.001780
grad_step = 000267, loss = 0.001749
grad_step = 000268, loss = 0.001752
grad_step = 000269, loss = 0.001778
grad_step = 000270, loss = 0.001798
grad_step = 000271, loss = 0.001800
grad_step = 000272, loss = 0.001780
grad_step = 000273, loss = 0.001758
grad_step = 000274, loss = 0.001744
grad_step = 000275, loss = 0.001741
grad_step = 000276, loss = 0.001748
grad_step = 000277, loss = 0.001757
grad_step = 000278, loss = 0.001767
grad_step = 000279, loss = 0.001772
grad_step = 000280, loss = 0.001774
grad_step = 000281, loss = 0.001768
grad_step = 000282, loss = 0.001761
grad_step = 000283, loss = 0.001752
grad_step = 000284, loss = 0.001745
grad_step = 000285, loss = 0.001738
grad_step = 000286, loss = 0.001733
grad_step = 000287, loss = 0.001729
grad_step = 000288, loss = 0.001726
grad_step = 000289, loss = 0.001724
grad_step = 000290, loss = 0.001723
grad_step = 000291, loss = 0.001722
grad_step = 000292, loss = 0.001722
grad_step = 000293, loss = 0.001722
grad_step = 000294, loss = 0.001722
grad_step = 000295, loss = 0.001723
grad_step = 000296, loss = 0.001725
grad_step = 000297, loss = 0.001732
grad_step = 000298, loss = 0.001745
grad_step = 000299, loss = 0.001774
grad_step = 000300, loss = 0.001822
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001925
grad_step = 000302, loss = 0.002056
grad_step = 000303, loss = 0.002276
grad_step = 000304, loss = 0.002351
grad_step = 000305, loss = 0.002293
grad_step = 000306, loss = 0.001984
grad_step = 000307, loss = 0.001759
grad_step = 000308, loss = 0.001808
grad_step = 000309, loss = 0.001984
grad_step = 000310, loss = 0.002029
grad_step = 000311, loss = 0.001858
grad_step = 000312, loss = 0.001729
grad_step = 000313, loss = 0.001798
grad_step = 000314, loss = 0.001916
grad_step = 000315, loss = 0.001907
grad_step = 000316, loss = 0.001776
grad_step = 000317, loss = 0.001704
grad_step = 000318, loss = 0.001763
grad_step = 000319, loss = 0.001856
grad_step = 000320, loss = 0.001875
grad_step = 000321, loss = 0.001801
grad_step = 000322, loss = 0.001724
grad_step = 000323, loss = 0.001709
grad_step = 000324, loss = 0.001747
grad_step = 000325, loss = 0.001782
grad_step = 000326, loss = 0.001788
grad_step = 000327, loss = 0.001766
grad_step = 000328, loss = 0.001717
grad_step = 000329, loss = 0.001693
grad_step = 000330, loss = 0.001709
grad_step = 000331, loss = 0.001738
grad_step = 000332, loss = 0.001751
grad_step = 000333, loss = 0.001727
grad_step = 000334, loss = 0.001696
grad_step = 000335, loss = 0.001688
grad_step = 000336, loss = 0.001696
grad_step = 000337, loss = 0.001706
grad_step = 000338, loss = 0.001710
grad_step = 000339, loss = 0.001708
grad_step = 000340, loss = 0.001703
grad_step = 000341, loss = 0.001694
grad_step = 000342, loss = 0.001682
grad_step = 000343, loss = 0.001675
grad_step = 000344, loss = 0.001677
grad_step = 000345, loss = 0.001684
grad_step = 000346, loss = 0.001690
grad_step = 000347, loss = 0.001689
grad_step = 000348, loss = 0.001685
grad_step = 000349, loss = 0.001680
grad_step = 000350, loss = 0.001677
grad_step = 000351, loss = 0.001674
grad_step = 000352, loss = 0.001670
grad_step = 000353, loss = 0.001666
grad_step = 000354, loss = 0.001663
grad_step = 000355, loss = 0.001663
grad_step = 000356, loss = 0.001664
grad_step = 000357, loss = 0.001665
grad_step = 000358, loss = 0.001665
grad_step = 000359, loss = 0.001666
grad_step = 000360, loss = 0.001668
grad_step = 000361, loss = 0.001671
grad_step = 000362, loss = 0.001675
grad_step = 000363, loss = 0.001682
grad_step = 000364, loss = 0.001692
grad_step = 000365, loss = 0.001708
grad_step = 000366, loss = 0.001730
grad_step = 000367, loss = 0.001770
grad_step = 000368, loss = 0.001817
grad_step = 000369, loss = 0.001890
grad_step = 000370, loss = 0.001947
grad_step = 000371, loss = 0.002010
grad_step = 000372, loss = 0.002000
grad_step = 000373, loss = 0.001947
grad_step = 000374, loss = 0.001820
grad_step = 000375, loss = 0.001702
grad_step = 000376, loss = 0.001645
grad_step = 000377, loss = 0.001669
grad_step = 000378, loss = 0.001736
grad_step = 000379, loss = 0.001782
grad_step = 000380, loss = 0.001780
grad_step = 000381, loss = 0.001729
grad_step = 000382, loss = 0.001674
grad_step = 000383, loss = 0.001644
grad_step = 000384, loss = 0.001651
grad_step = 000385, loss = 0.001680
grad_step = 000386, loss = 0.001704
grad_step = 000387, loss = 0.001705
grad_step = 000388, loss = 0.001679
grad_step = 000389, loss = 0.001648
grad_step = 000390, loss = 0.001631
grad_step = 000391, loss = 0.001635
grad_step = 000392, loss = 0.001649
grad_step = 000393, loss = 0.001660
grad_step = 000394, loss = 0.001663
grad_step = 000395, loss = 0.001656
grad_step = 000396, loss = 0.001645
grad_step = 000397, loss = 0.001633
grad_step = 000398, loss = 0.001625
grad_step = 000399, loss = 0.001621
grad_step = 000400, loss = 0.001623
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001628
grad_step = 000402, loss = 0.001633
grad_step = 000403, loss = 0.001637
grad_step = 000404, loss = 0.001638
grad_step = 000405, loss = 0.001637
grad_step = 000406, loss = 0.001634
grad_step = 000407, loss = 0.001632
grad_step = 000408, loss = 0.001629
grad_step = 000409, loss = 0.001626
grad_step = 000410, loss = 0.001623
grad_step = 000411, loss = 0.001620
grad_step = 000412, loss = 0.001617
grad_step = 000413, loss = 0.001616
grad_step = 000414, loss = 0.001616
grad_step = 000415, loss = 0.001618
grad_step = 000416, loss = 0.001620
grad_step = 000417, loss = 0.001626
grad_step = 000418, loss = 0.001635
grad_step = 000419, loss = 0.001653
grad_step = 000420, loss = 0.001682
grad_step = 000421, loss = 0.001738
grad_step = 000422, loss = 0.001816
grad_step = 000423, loss = 0.001951
grad_step = 000424, loss = 0.002081
grad_step = 000425, loss = 0.002238
grad_step = 000426, loss = 0.002230
grad_step = 000427, loss = 0.002102
grad_step = 000428, loss = 0.001821
grad_step = 000429, loss = 0.001623
grad_step = 000430, loss = 0.001626
grad_step = 000431, loss = 0.001772
grad_step = 000432, loss = 0.001881
grad_step = 000433, loss = 0.001808
grad_step = 000434, loss = 0.001657
grad_step = 000435, loss = 0.001593
grad_step = 000436, loss = 0.001659
grad_step = 000437, loss = 0.001741
grad_step = 000438, loss = 0.001725
grad_step = 000439, loss = 0.001644
grad_step = 000440, loss = 0.001593
grad_step = 000441, loss = 0.001618
grad_step = 000442, loss = 0.001664
grad_step = 000443, loss = 0.001667
grad_step = 000444, loss = 0.001633
grad_step = 000445, loss = 0.001597
grad_step = 000446, loss = 0.001594
grad_step = 000447, loss = 0.001612
grad_step = 000448, loss = 0.001620
grad_step = 000449, loss = 0.001614
grad_step = 000450, loss = 0.001597
grad_step = 000451, loss = 0.001585
grad_step = 000452, loss = 0.001583
grad_step = 000453, loss = 0.001586
grad_step = 000454, loss = 0.001591
grad_step = 000455, loss = 0.001592
grad_step = 000456, loss = 0.001590
grad_step = 000457, loss = 0.001580
grad_step = 000458, loss = 0.001570
grad_step = 000459, loss = 0.001563
grad_step = 000460, loss = 0.001566
grad_step = 000461, loss = 0.001572
grad_step = 000462, loss = 0.001576
grad_step = 000463, loss = 0.001573
grad_step = 000464, loss = 0.001565
grad_step = 000465, loss = 0.001558
grad_step = 000466, loss = 0.001555
grad_step = 000467, loss = 0.001555
grad_step = 000468, loss = 0.001556
grad_step = 000469, loss = 0.001556
grad_step = 000470, loss = 0.001555
grad_step = 000471, loss = 0.001555
grad_step = 000472, loss = 0.001556
grad_step = 000473, loss = 0.001556
grad_step = 000474, loss = 0.001555
grad_step = 000475, loss = 0.001553
grad_step = 000476, loss = 0.001550
grad_step = 000477, loss = 0.001548
grad_step = 000478, loss = 0.001547
grad_step = 000479, loss = 0.001546
grad_step = 000480, loss = 0.001546
grad_step = 000481, loss = 0.001545
grad_step = 000482, loss = 0.001544
grad_step = 000483, loss = 0.001544
grad_step = 000484, loss = 0.001546
grad_step = 000485, loss = 0.001550
grad_step = 000486, loss = 0.001557
grad_step = 000487, loss = 0.001568
grad_step = 000488, loss = 0.001589
grad_step = 000489, loss = 0.001622
grad_step = 000490, loss = 0.001678
grad_step = 000491, loss = 0.001748
grad_step = 000492, loss = 0.001850
grad_step = 000493, loss = 0.001921
grad_step = 000494, loss = 0.001973
grad_step = 000495, loss = 0.001911
grad_step = 000496, loss = 0.001784
grad_step = 000497, loss = 0.001632
grad_step = 000498, loss = 0.001542
grad_step = 000499, loss = 0.001551
grad_step = 000500, loss = 0.001631
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001700
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

  date_run                              2020-05-20 12:13:57.416119
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.230015
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-20 12:13:57.421644
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.120153
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-20 12:13:57.427857
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.145015
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-20 12:13:57.432946
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.825764
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
100%|██████████| 10/10 [00:02<00:00,  3.96it/s, avg_epoch_loss=5.26]
INFO:root:Epoch[0] Elapsed time 2.528 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.256865
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.256864833831787 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f1998948320> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  7.75it/s, avg_epoch_loss=3.59e+3]
INFO:root:Epoch[0] Elapsed time 1.290 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=3590.403646
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 3590.4036458333335 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f193968a588> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
 30%|███       | 3/10 [00:11<00:27,  3.92s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:22<00:15,  3.85s/it, avg_epoch_loss=6.91] 90%|█████████ | 9/10 [00:33<00:03,  3.74s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:36<00:00,  3.66s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 36.650 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.864980
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.864979600906372 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f18d06a11d0> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  5.39it/s, avg_epoch_loss=5.81]
INFO:root:Epoch[0] Elapsed time 1.857 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.809819
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.809819030761719 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f18d06a9e10> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
 10%|█         | 1/10 [02:14<20:12, 134.67s/it, avg_epoch_loss=0.412] 20%|██        | 2/10 [05:27<20:17, 152.20s/it, avg_epoch_loss=0.399] 30%|███       | 3/10 [09:04<20:01, 171.62s/it, avg_epoch_loss=0.39]  40%|████      | 4/10 [12:49<18:45, 187.59s/it, avg_epoch_loss=0.385] 50%|█████     | 5/10 [16:16<16:07, 193.51s/it, avg_epoch_loss=0.384] 60%|██████    | 6/10 [20:03<13:33, 203.48s/it, avg_epoch_loss=0.383] 70%|███████   | 7/10 [23:56<10:37, 212.38s/it, avg_epoch_loss=0.381] 80%|████████  | 8/10 [27:39<07:10, 215.41s/it, avg_epoch_loss=0.378] 90%|█████████ | 9/10 [31:39<03:42, 222.76s/it, avg_epoch_loss=0.376]100%|██████████| 10/10 [35:00<00:00, 216.27s/it, avg_epoch_loss=0.375]100%|██████████| 10/10 [35:00<00:00, 210.04s/it, avg_epoch_loss=0.375]
INFO:root:Epoch[0] Elapsed time 2100.379 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.374571
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3745713621377945 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f18d052f160> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:01<00:00,  6.16it/s, avg_epoch_loss=415]
INFO:root:Epoch[0] Elapsed time 1.656 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=414.652022
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 414.65202175008733 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f193b2586a0> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
100%|██████████| 10/10 [00:00<00:00, 42.77it/s, avg_epoch_loss=5.12]
INFO:root:Epoch[0] Elapsed time 0.235 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.123024
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.123023939132691 (occurred at epoch 0)
INFO:root:End model training
>>>model:  <mlmodels.model_gluon.gluonts_model.Model object at 0x7f193b282588> <class 'mlmodels.model_gluon.gluonts_model.Model'>
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
0   2020-05-20 12:13:25.999051  ...    mean_absolute_error
1   2020-05-20 12:13:26.003500  ...     mean_squared_error
2   2020-05-20 12:13:26.007140  ...  median_absolute_error
3   2020-05-20 12:13:26.010633  ...               r2_score
4   2020-05-20 12:13:34.779797  ...    mean_absolute_error
5   2020-05-20 12:13:34.783584  ...     mean_squared_error
6   2020-05-20 12:13:34.786646  ...  median_absolute_error
7   2020-05-20 12:13:34.789574  ...               r2_score
8   2020-05-20 12:13:57.416119  ...    mean_absolute_error
9   2020-05-20 12:13:57.421644  ...     mean_squared_error
10  2020-05-20 12:13:57.427857  ...  median_absolute_error
11  2020-05-20 12:13:57.432946  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f56034f4d68> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet18', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet18/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet34/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f55aae32940> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f55adfd6d30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f55aae32940> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f56034f4d68> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': 'ztest/model_tch/torchhub/wide_resnet101_2/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f55aae32940> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f55adfd6d30> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f55aae32940> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f56034f4d68> <class 'mlmodels.model_tch.torchhub.Model'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} 'data_info' 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f55aae32940> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f55adfd6d30> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f202dedf080> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=585bcaeba7804ed19a018096be2f6c2bdcf76471edf30ff26c1a4948974ca278
  Stored in directory: /tmp/pip-ephem-wheel-cache-13_wo_rw/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f1fc5cdb5c0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1499136/17464789 [=>............................] - ETA: 0s
 5521408/17464789 [========>.....................] - ETA: 0s
12279808/17464789 [====================>.........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-20 12:51:29.981576: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-20 12:51:29.999549: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-20 12:51:29.999758: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e2e45a00d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-20 12:51:29.999794: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.7280 - accuracy: 0.4960
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8046 - accuracy: 0.4910 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7842 - accuracy: 0.4923
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7663 - accuracy: 0.4935
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6789 - accuracy: 0.4992
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7203 - accuracy: 0.4965
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7214 - accuracy: 0.4964
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7663 - accuracy: 0.4935
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7280 - accuracy: 0.4960
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7080 - accuracy: 0.4973
11000/25000 [============>.................] - ETA: 4s - loss: 7.7126 - accuracy: 0.4970
12000/25000 [=============>................] - ETA: 3s - loss: 7.6871 - accuracy: 0.4987
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6843 - accuracy: 0.4988
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
15000/25000 [=================>............] - ETA: 2s - loss: 7.7075 - accuracy: 0.4973
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6800 - accuracy: 0.4991
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6955 - accuracy: 0.4981
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6734 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6747 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6781 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6841 - accuracy: 0.4989
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6771 - accuracy: 0.4993
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6846 - accuracy: 0.4988
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 9s 355us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-20 12:51:45.403600
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-20 12:51:45.403600  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<47:54:43, 5.00kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<33:46:19, 7.09kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<23:41:31, 10.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:02<16:35:07, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 3.62M/862M [00:02<11:34:34, 20.6kB/s].vector_cache/glove.6B.zip:   1%|          | 7.75M/862M [00:02<8:03:57, 29.4kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.5M/862M [00:02<5:37:00, 42.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.3M/862M [00:02<3:54:39, 60.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.5M/862M [00:02<2:43:32, 85.7kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.6M/862M [00:02<1:54:02, 122kB/s] .vector_cache/glove.6B.zip:   4%|▎         | 30.2M/862M [00:02<1:19:28, 174kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.7M/862M [00:02<55:30, 249kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 37.4M/862M [00:02<38:47, 354kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.4M/862M [00:03<27:11, 504kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:03<19:04, 715kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.0M/862M [00:03<13:23, 1.01MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:03<09:25, 1.43MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:03<31:51, 424kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<24:06, 557kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:05<20:32, 654kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:06<15:10, 885kB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.9M/862M [00:06<10:48, 1.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:07<11:43, 1.14MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:07<09:43, 1.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:08<07:11, 1.86MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:09<07:51, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<08:20, 1.59MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.6M/862M [00:10<06:32, 2.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<04:44, 2.79MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:11<43:00, 308kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:11<31:28, 420kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:11<22:19, 591kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:13<18:38, 706kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.3M/862M [00:13<14:09, 929kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:13<10:15, 1.28MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:13<07:19, 1.79MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:15<54:32, 240kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:15<39:30, 331kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:15<27:52, 468kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:17<22:31, 578kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:17<17:05, 761kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:17<12:13, 1.06MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:19<11:37, 1.11MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:19<09:27, 1.37MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:19<06:56, 1.86MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:21<07:53, 1.63MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:21<06:50, 1.88MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:21<05:03, 2.54MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:23<06:32, 1.96MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:23<05:53, 2.17MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:23<04:27, 2.87MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:25<06:07, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:25<05:35, 2.28MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:25<04:10, 3.04MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<05:57, 2.13MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<05:26, 2.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<04:05, 3.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<05:51, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<05:23, 2.34MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:29<04:05, 3.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<05:54, 2.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<06:42, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<05:19, 2.35MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<05:44, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<05:20, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<04:00, 3.11MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<05:42, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<06:32, 1.89MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<05:06, 2.42MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<03:46, 3.27MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<07:08, 1.73MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<06:16, 1.97MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<04:42, 2.62MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<06:08, 2.00MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<06:49, 1.79MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:24, 2.26MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<03:54, 3.12MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:41<11:45:38, 17.3kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<8:14:59, 24.6kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<5:46:02, 35.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<4:04:20, 49.6kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<2:52:11, 70.4kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<2:00:34, 100kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<1:27:01, 139kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<1:02:06, 194kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<43:41, 275kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<33:19, 360kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<24:32, 488kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<17:27, 685kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<15:00, 794kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:49<12:56, 921kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:49<09:34, 1.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<06:50, 1.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<10:58, 1.08MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<08:53, 1.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<06:28, 1.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<07:16, 1.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:53<07:30, 1.57MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:53<05:46, 2.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<04:11, 2.80MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<09:06, 1.29MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<07:35, 1.54MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<05:36, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<06:38, 1.75MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<07:05, 1.64MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:57<05:33, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:44, 2.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:13, 2.21MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:59<03:57, 2.92MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:26, 2.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<06:12, 1.86MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<04:50, 2.37MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<03:31, 3.24MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<08:39, 1.32MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<07:15, 1.58MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<05:21, 2.13MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<06:23, 1.78MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:39, 2.01MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<04:11, 2.70MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<05:37, 2.01MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<05:05, 2.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<03:50, 2.93MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<05:21, 2.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<04:52, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<03:41, 3.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<05:13, 2.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<04:47, 2.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<03:38, 3.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<05:09, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<04:45, 2.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<03:36, 3.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<05:07, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<04:42, 2.34MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<03:34, 3.08MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<05:05, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<04:41, 2.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<03:33, 3.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<05:03, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<05:46, 1.88MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<04:30, 2.41MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<03:22, 3.22MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<05:30, 1.97MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<04:59, 2.17MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<03:46, 2.86MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<05:07, 2.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<04:41, 2.29MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<03:31, 3.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<05:02, 2.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<05:44, 1.86MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<04:34, 2.33MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<03:19, 3.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<32:47, 324kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<24:00, 442kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<17:01, 622kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<14:26, 730kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<12:23, 851kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<09:12, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<06:32, 1.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<42:21, 247kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<30:44, 341kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<21:44, 480kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<17:34, 592kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<13:21, 778kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<09:35, 1.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<09:13, 1.12MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<08:35, 1.20MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<06:32, 1.58MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:39, 2.20MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<15:44, 653kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<12:04, 850kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<08:41, 1.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<08:28, 1.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<07:38, 1.34MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:38<05:52, 1.73MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<04:19, 2.35MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<05:44, 1.76MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:40<05:05, 1.99MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<03:49, 2.65MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<05:01, 2.00MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<05:35, 1.80MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:42<04:26, 2.26MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<03:11, 3.13MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:44<38:39, 259kB/s] .vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:44<27:54, 358kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<19:45, 505kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<13:53, 714kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:46<49:44, 200kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<36:50, 269kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<26:12, 378kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<18:23, 537kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<18:47, 525kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:48<14:10, 695kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<10:09, 968kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<09:22, 1.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:50<07:33, 1.29MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<05:31, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<06:09, 1.58MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<05:18, 1.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<03:57, 2.45MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<05:02, 1.91MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:54<04:29, 2.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<03:23, 2.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<04:37, 2.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<04:13, 2.27MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<03:11, 2.99MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<04:28, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:58<04:06, 2.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:58<03:06, 3.04MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<04:24, 2.14MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:02, 2.33MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:00<03:04, 3.07MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<04:21, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:00, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:02<03:02, 3.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<04:19, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<03:58, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:04<02:59, 3.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<04:17, 2.16MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<04:53, 1.89MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:06<03:53, 2.37MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<04:12, 2.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<03:53, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<02:55, 3.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<04:10, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<03:42, 2.45MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<02:51, 3.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<02:05, 4.31MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<37:46, 239kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<27:20, 330kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<19:18, 466kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<15:35, 575kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<11:49, 758kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<08:29, 1.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<08:01, 1.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<06:31, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:46, 1.86MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<05:25, 1.63MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<04:42, 1.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<03:30, 2.51MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<04:23, 1.99MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<04:44, 1.85MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<03:44, 2.34MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<04:02, 2.15MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<03:47, 2.29MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<02:53, 3.00MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<03:54, 2.21MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<03:39, 2.36MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<02:46, 3.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:25<03:53, 2.20MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<03:37, 2.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<02:44, 3.10MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<03:53, 2.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<03:35, 2.36MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<02:43, 3.10MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<03:53, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:29<03:35, 2.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<02:41, 3.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<03:51, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<03:33, 2.34MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<02:41, 3.08MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<03:50, 2.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<04:23, 1.89MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<03:29, 2.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:31, 3.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<7:53:25, 17.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<5:31:58, 24.7kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<3:51:50, 35.3kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<2:43:28, 49.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<1:55:10, 70.6kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<1:20:34, 101kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<58:03, 139kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<41:26, 195kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<29:05, 276kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<22:10, 361kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:41<16:19, 490kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<11:35, 687kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<09:57, 796kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<07:46, 1.02MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<05:37, 1.40MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:45<05:46, 1.36MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<05:40, 1.38MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:45<04:19, 1.82MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:06, 2.51MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:47<06:26, 1.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<05:18, 1.47MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:47<03:52, 2.00MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:49<04:31, 1.71MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:49<04:45, 1.62MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<03:40, 2.10MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<02:39, 2.89MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<08:06, 944kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:51<06:28, 1.18MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:42, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<05:03, 1.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:53<04:19, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:12, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<04:00, 1.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:55<03:35, 2.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<02:41, 2.78MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<03:38, 2.04MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:57<03:18, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<02:29, 2.97MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<03:29, 2.12MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:59<03:12, 2.30MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:59<02:25, 3.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<03:25, 2.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:01<03:08, 2.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:01<02:22, 3.06MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<03:22, 2.15MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:50, 1.89MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:03<03:01, 2.40MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<02:10, 3.31MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<17:09, 418kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<12:45, 562kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:05<09:05, 786kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<07:59, 889kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<06:19, 1.12MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<04:35, 1.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<04:51, 1.45MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<04:07, 1.71MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:08<03:01, 2.31MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<03:45, 1.85MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:10<03:24, 2.05MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<02:44, 2.54MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:11<02:03, 3.38MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<03:25, 2.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<03:08, 2.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<02:21, 2.90MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<03:15, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<02:58, 2.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<02:15, 3.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<03:10, 2.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<02:54, 2.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:09, 3.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<01:35, 4.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<51:01, 131kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<37:04, 181kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<26:12, 255kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 463M/862M [03:18<18:19, 362kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<16:19, 406kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:20<12:00, 552kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<08:34, 770kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<06:02, 1.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<30:33, 215kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<22:02, 297kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<15:30, 421kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<12:20, 526kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<09:18, 697kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<06:39, 971kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<05:57, 1.08MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<06:26, 995kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:27<05:04, 1.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:39, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<04:15, 1.49MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<03:45, 1.68MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:29<02:49, 2.24MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<03:14, 1.93MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<03:37, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:31<02:52, 2.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:04, 3.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<05:19, 1.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<04:23, 1.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<03:13, 1.91MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:38, 1.68MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<03:11, 1.92MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<02:22, 2.56MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:36<03:03, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<03:23, 1.79MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<02:39, 2.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<01:55, 3.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<06:53, 870kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<05:26, 1.10MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:57, 1.51MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<04:06, 1.44MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<04:06, 1.44MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:08, 1.88MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:16, 2.58MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<04:17, 1.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<03:30, 1.67MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:35, 2.25MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<01:52, 3.09MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<24:35, 235kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<17:48, 325kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<12:33, 459kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<10:04, 567kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:46<07:38, 748kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<05:27, 1.04MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:48<05:08, 1.10MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<04:10, 1.35MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:03, 1.84MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:50<03:26, 1.62MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<02:58, 1.87MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<02:12, 2.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:52<02:50, 1.94MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<02:33, 2.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<01:55, 2.85MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<02:37, 2.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<02:57, 1.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<02:20, 2.32MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<02:29, 2.15MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<02:17, 2.34MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<01:43, 3.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<01:14, 4.25MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<29:08, 182kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<22:09, 239kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:59<15:52, 333kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<11:09, 471kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<09:01, 579kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<06:57, 751kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<05:00, 1.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<04:31, 1.14MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<03:48, 1.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<02:48, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<02:59, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<02:42, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:02, 2.48MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:06<02:27, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:18, 2.17MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<01:43, 2.88MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:08<02:15, 2.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:07, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<01:35, 3.08MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:11, 2.23MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:02, 2.39MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:32, 3.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:12, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<02:02, 2.35MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:30, 3.16MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<01:07, 4.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<57:03, 83.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:14<40:22, 117kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<28:12, 167kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:16<20:42, 226kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:16<14:52, 314kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<10:30, 442kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<07:20, 627kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<22:29, 205kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:18<16:06, 285kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<11:20, 403kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<07:57, 571kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:20<09:54, 458kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:20<07:23, 612kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<05:15, 856kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:22<04:42, 948kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<03:45, 1.19MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:42, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<02:55, 1.50MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<02:29, 1.76MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<01:50, 2.36MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<02:18, 1.88MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<02:03, 2.10MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<01:32, 2.78MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<02:04, 2.05MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<01:53, 2.24MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:24, 3.00MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<01:58, 2.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<01:44, 2.40MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:18, 3.19MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<00:57, 4.29MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<17:16, 239kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:32<12:55, 319kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<09:11, 446kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<06:26, 632kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<06:12, 652kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:34<04:46, 848kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<03:25, 1.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:36<03:18, 1.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:36<02:43, 1.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<01:58, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:38<02:18, 1.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<02:00, 1.94MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:30, 2.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:40<01:57, 1.97MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<01:41, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<01:18, 2.94MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<00:56, 4.00MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:42<14:57, 253kB/s] .vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:42<11:14, 336kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<08:02, 468kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<06:08, 604kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<04:41, 790kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<03:20, 1.10MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<03:09, 1.15MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<02:57, 1.23MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<02:12, 1.64MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:36, 2.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:14, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:48<01:55, 1.85MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:25, 2.47MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<01:48, 1.93MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:50<01:37, 2.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:13, 2.85MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:39, 2.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<01:30, 2.27MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<01:08, 3.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:35, 2.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<01:23, 2.41MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<01:03, 3.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<00:46, 4.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<12:58, 254kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<09:24, 350kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<06:36, 494kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:57<05:20, 605kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<04:03, 793kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:53, 1.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<02:45, 1.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<02:34, 1.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:55, 1.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:23, 2.23MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:01<02:00, 1.55MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:43, 1.79MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<01:16, 2.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:35, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:25, 2.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:03, 2.81MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:26, 2.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<01:18, 2.26MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<00:58, 3.01MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:21, 2.12MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<01:14, 2.31MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<00:56, 3.05MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:18, 2.14MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<01:12, 2.33MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<00:54, 3.07MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:11<01:16, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:10, 2.34MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<00:52, 3.08MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:13<01:14, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<01:08, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<00:51, 3.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<01:12, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<01:06, 2.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<00:50, 3.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<01:10, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:20, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<01:02, 2.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:45, 3.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:45, 1.41MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:29, 1.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:05, 2.23MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:19, 1.82MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<01:10, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<00:52, 2.72MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:23<01:09, 2.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:23<01:02, 2.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<00:46, 2.94MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<01:04, 2.11MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:25<00:59, 2.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<00:44, 3.02MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<01:01, 2.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:27<00:56, 2.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<00:42, 3.05MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<00:59, 2.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<00:52, 2.43MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<00:43, 2.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:31, 3.96MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:31<02:25, 854kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:54, 1.08MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:26, 1.43MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:00, 1.99MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<01:42, 1.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<01:23, 1.42MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:00, 1.93MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:09, 1.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<01:12, 1.59MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<00:56, 2.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:40, 2.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:25, 1.30MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<01:11, 1.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:52, 2.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:01, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<01:06, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:51, 2.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:36, 2.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:57, 877kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:41<01:33, 1.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<01:06, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:42<01:08, 1.44MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:58, 1.70MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:42, 2.28MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:51, 1.84MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:45, 2.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:34, 2.74MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:44, 2.03MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:46<00:40, 2.23MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:29, 2.98MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<00:41, 2.11MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:36, 2.40MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:29, 2.87MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:21, 3.93MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:50<01:45, 781kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:20, 1.02MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:58, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:40, 1.95MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<05:25, 242kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<04:03, 323kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<02:52, 451kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<01:57, 640kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<02:19, 535kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<01:43, 717kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:13, 999kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:50, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<05:18, 221kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<03:48, 306kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<02:38, 432kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<02:02, 539kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<01:32, 715kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:04, 995kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<00:58, 1.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:00<00:47, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:33, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:36, 1.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:31, 1.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:22, 2.47MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:04<00:28, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:04<00:30, 1.75MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:23, 2.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:16, 3.06MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:30, 1.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:26, 1.86MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:19, 2.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<00:23, 1.94MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<00:20, 2.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:14, 2.99MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:10, 4.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<02:54, 238kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<02:05, 329kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<01:25, 464kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<01:05, 573kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:49, 755kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:33, 1.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:30, 1.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:24, 1.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:16, 1.85MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:17, 1.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:15, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:10, 2.51MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:11, 2.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:17, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:14, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:09, 2.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:11, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:09, 1.99MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:06, 2.66MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:07, 2.08MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:08, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:06, 2.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:04, 3.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:05, 2.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 2.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:03, 2.94MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:01, 3.02MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.14MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.32MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 3.06MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<126:34:39,  1.14s/it]  0%|          | 781/400000 [00:01<88:26:09,  1.25it/s]  0%|          | 1567/400000 [00:01<61:47:14,  1.79it/s]  1%|          | 2421/400000 [00:01<43:09:44,  2.56it/s]  1%|          | 3290/400000 [00:01<30:09:05,  3.65it/s]  1%|          | 4161/400000 [00:01<21:03:48,  5.22it/s]  1%|          | 4921/400000 [00:01<14:43:13,  7.46it/s]  1%|▏         | 5668/400000 [00:01<10:17:21, 10.65it/s]  2%|▏         | 6534/400000 [00:01<7:11:25, 15.20it/s]   2%|▏         | 7426/400000 [00:02<5:01:32, 21.70it/s]  2%|▏         | 8314/400000 [00:02<3:30:49, 30.97it/s]  2%|▏         | 9173/400000 [00:02<2:27:28, 44.17it/s]  3%|▎         | 10012/400000 [00:02<1:43:15, 62.95it/s]  3%|▎         | 10860/400000 [00:02<1:12:21, 89.64it/s]  3%|▎         | 11743/400000 [00:02<50:45, 127.50it/s]   3%|▎         | 12591/400000 [00:02<35:40, 180.97it/s]  3%|▎         | 13490/400000 [00:02<25:07, 256.32it/s]  4%|▎         | 14352/400000 [00:02<17:47, 361.37it/s]  4%|▍         | 15227/400000 [00:02<12:38, 507.26it/s]  4%|▍         | 16095/400000 [00:03<09:03, 706.94it/s]  4%|▍         | 16972/400000 [00:03<06:32, 976.16it/s]  4%|▍         | 17838/400000 [00:03<04:47, 1328.83it/s]  5%|▍         | 18698/400000 [00:03<03:35, 1771.88it/s]  5%|▍         | 19540/400000 [00:03<02:44, 2308.62it/s]  5%|▌         | 20366/400000 [00:03<02:09, 2937.89it/s]  5%|▌         | 21237/400000 [00:03<01:43, 3666.88it/s]  6%|▌         | 22133/400000 [00:03<01:24, 4456.22it/s]  6%|▌         | 22987/400000 [00:03<01:12, 5191.60it/s]  6%|▌         | 23847/400000 [00:03<01:03, 5891.36it/s]  6%|▌         | 24741/400000 [00:04<00:57, 6561.63it/s]  6%|▋         | 25633/400000 [00:04<00:52, 7125.38it/s]  7%|▋         | 26507/400000 [00:04<00:49, 7535.49it/s]  7%|▋         | 27380/400000 [00:04<00:48, 7736.62it/s]  7%|▋         | 28239/400000 [00:04<00:47, 7843.18it/s]  7%|▋         | 29083/400000 [00:04<00:46, 7994.91it/s]  7%|▋         | 29926/400000 [00:04<00:45, 8119.48it/s]  8%|▊         | 30769/400000 [00:04<00:45, 8176.25it/s]  8%|▊         | 31608/400000 [00:04<00:45, 8103.67it/s]  8%|▊         | 32434/400000 [00:04<00:45, 8097.42it/s]  8%|▊         | 33255/400000 [00:05<00:45, 8088.59it/s]  9%|▊         | 34072/400000 [00:05<00:45, 8051.31it/s]  9%|▊         | 34927/400000 [00:05<00:44, 8192.21it/s]  9%|▉         | 35751/400000 [00:05<00:44, 8172.46it/s]  9%|▉         | 36631/400000 [00:05<00:43, 8348.73it/s]  9%|▉         | 37491/400000 [00:05<00:43, 8419.50it/s] 10%|▉         | 38349/400000 [00:05<00:42, 8464.47it/s] 10%|▉         | 39209/400000 [00:05<00:42, 8503.06it/s] 10%|█         | 40061/400000 [00:05<00:42, 8378.71it/s] 10%|█         | 40944/400000 [00:06<00:42, 8508.15it/s] 10%|█         | 41824/400000 [00:06<00:41, 8593.26it/s] 11%|█         | 42706/400000 [00:06<00:41, 8658.86it/s] 11%|█         | 43573/400000 [00:06<00:41, 8648.28it/s] 11%|█         | 44439/400000 [00:06<00:42, 8416.57it/s] 11%|█▏        | 45302/400000 [00:06<00:41, 8478.52it/s] 12%|█▏        | 46166/400000 [00:06<00:41, 8525.21it/s] 12%|█▏        | 47020/400000 [00:06<00:42, 8362.79it/s] 12%|█▏        | 47858/400000 [00:06<00:43, 8165.00it/s] 12%|█▏        | 48707/400000 [00:06<00:42, 8257.83it/s] 12%|█▏        | 49599/400000 [00:07<00:41, 8444.16it/s] 13%|█▎        | 50446/400000 [00:07<00:41, 8401.14it/s] 13%|█▎        | 51324/400000 [00:07<00:40, 8508.59it/s] 13%|█▎        | 52212/400000 [00:07<00:40, 8616.55it/s] 13%|█▎        | 53075/400000 [00:07<00:41, 8425.74it/s] 13%|█▎        | 53948/400000 [00:07<00:40, 8512.95it/s] 14%|█▎        | 54843/400000 [00:07<00:39, 8638.63it/s] 14%|█▍        | 55732/400000 [00:07<00:39, 8711.09it/s] 14%|█▍        | 56605/400000 [00:07<00:39, 8696.74it/s] 14%|█▍        | 57476/400000 [00:07<00:39, 8578.19it/s] 15%|█▍        | 58354/400000 [00:08<00:39, 8637.40it/s] 15%|█▍        | 59255/400000 [00:08<00:38, 8744.32it/s] 15%|█▌        | 60131/400000 [00:08<00:38, 8717.97it/s] 15%|█▌        | 61004/400000 [00:08<00:39, 8541.30it/s] 15%|█▌        | 61860/400000 [00:08<00:40, 8452.96it/s] 16%|█▌        | 62729/400000 [00:08<00:39, 8521.95it/s] 16%|█▌        | 63583/400000 [00:08<00:39, 8500.02it/s] 16%|█▌        | 64495/400000 [00:08<00:38, 8676.57it/s] 16%|█▋        | 65388/400000 [00:08<00:38, 8749.09it/s] 17%|█▋        | 66264/400000 [00:08<00:38, 8684.50it/s] 17%|█▋        | 67164/400000 [00:09<00:37, 8774.92it/s] 17%|█▋        | 68043/400000 [00:09<00:37, 8753.81it/s] 17%|█▋        | 68924/400000 [00:09<00:37, 8769.07it/s] 17%|█▋        | 69823/400000 [00:09<00:37, 8833.97it/s] 18%|█▊        | 70707/400000 [00:09<00:38, 8600.33it/s] 18%|█▊        | 71572/400000 [00:09<00:38, 8615.14it/s] 18%|█▊        | 72435/400000 [00:09<00:38, 8609.07it/s] 18%|█▊        | 73333/400000 [00:09<00:37, 8715.76it/s] 19%|█▊        | 74206/400000 [00:09<00:37, 8683.43it/s] 19%|█▉        | 75075/400000 [00:09<00:38, 8352.99it/s] 19%|█▉        | 75914/400000 [00:10<00:39, 8107.20it/s] 19%|█▉        | 76729/400000 [00:10<00:39, 8106.26it/s] 19%|█▉        | 77556/400000 [00:10<00:39, 8154.60it/s] 20%|█▉        | 78397/400000 [00:10<00:39, 8227.28it/s] 20%|█▉        | 79244/400000 [00:10<00:38, 8297.02it/s] 20%|██        | 80075/400000 [00:10<00:39, 8143.06it/s] 20%|██        | 80891/400000 [00:10<00:39, 8025.26it/s] 20%|██        | 81772/400000 [00:10<00:38, 8243.30it/s] 21%|██        | 82609/400000 [00:10<00:38, 8280.61it/s] 21%|██        | 83444/400000 [00:11<00:38, 8300.12it/s] 21%|██        | 84294/400000 [00:11<00:37, 8358.12it/s] 21%|██▏       | 85131/400000 [00:11<00:37, 8342.51it/s] 21%|██▏       | 85991/400000 [00:11<00:37, 8415.80it/s] 22%|██▏       | 86854/400000 [00:11<00:36, 8478.87it/s] 22%|██▏       | 87703/400000 [00:11<00:36, 8453.74it/s] 22%|██▏       | 88549/400000 [00:11<00:36, 8421.57it/s] 22%|██▏       | 89392/400000 [00:11<00:37, 8206.34it/s] 23%|██▎       | 90232/400000 [00:11<00:37, 8262.22it/s] 23%|██▎       | 91060/400000 [00:11<00:37, 8184.73it/s] 23%|██▎       | 91880/400000 [00:12<00:38, 8002.51it/s] 23%|██▎       | 92770/400000 [00:12<00:37, 8251.37it/s] 23%|██▎       | 93599/400000 [00:12<00:37, 8209.53it/s] 24%|██▎       | 94427/400000 [00:12<00:37, 8227.57it/s] 24%|██▍       | 95252/400000 [00:12<00:37, 8162.55it/s] 24%|██▍       | 96070/400000 [00:12<00:37, 8041.29it/s] 24%|██▍       | 96876/400000 [00:12<00:37, 8044.28it/s] 24%|██▍       | 97703/400000 [00:12<00:37, 8108.39it/s] 25%|██▍       | 98549/400000 [00:12<00:36, 8209.46it/s] 25%|██▍       | 99371/400000 [00:12<00:37, 8035.86it/s] 25%|██▌       | 100176/400000 [00:13<00:37, 7948.47it/s] 25%|██▌       | 100973/400000 [00:13<00:38, 7804.18it/s] 25%|██▌       | 101755/400000 [00:13<00:38, 7727.28it/s] 26%|██▌       | 102529/400000 [00:13<00:38, 7651.05it/s] 26%|██▌       | 103296/400000 [00:13<00:39, 7594.23it/s] 26%|██▌       | 104069/400000 [00:13<00:38, 7633.86it/s] 26%|██▌       | 104860/400000 [00:13<00:38, 7704.88it/s] 26%|██▋       | 105659/400000 [00:13<00:37, 7786.77it/s] 27%|██▋       | 106471/400000 [00:13<00:37, 7881.82it/s] 27%|██▋       | 107260/400000 [00:13<00:37, 7864.53it/s] 27%|██▋       | 108047/400000 [00:14<00:37, 7863.19it/s] 27%|██▋       | 108834/400000 [00:14<00:37, 7769.81it/s] 27%|██▋       | 109627/400000 [00:14<00:37, 7816.96it/s] 28%|██▊       | 110410/400000 [00:14<00:37, 7765.75it/s] 28%|██▊       | 111187/400000 [00:14<00:37, 7651.16it/s] 28%|██▊       | 111980/400000 [00:14<00:37, 7731.24it/s] 28%|██▊       | 112771/400000 [00:14<00:36, 7776.78it/s] 28%|██▊       | 113550/400000 [00:14<00:37, 7629.69it/s] 29%|██▊       | 114342/400000 [00:14<00:37, 7710.74it/s] 29%|██▉       | 115114/400000 [00:15<00:38, 7455.01it/s] 29%|██▉       | 115893/400000 [00:15<00:37, 7550.72it/s] 29%|██▉       | 116651/400000 [00:15<00:37, 7524.13it/s] 29%|██▉       | 117405/400000 [00:15<00:38, 7350.34it/s] 30%|██▉       | 118230/400000 [00:15<00:37, 7597.51it/s] 30%|██▉       | 119022/400000 [00:15<00:36, 7689.58it/s] 30%|██▉       | 119815/400000 [00:15<00:36, 7759.16it/s] 30%|███       | 120630/400000 [00:15<00:35, 7871.96it/s] 30%|███       | 121472/400000 [00:15<00:34, 8027.28it/s] 31%|███       | 122303/400000 [00:15<00:34, 8109.92it/s] 31%|███       | 123116/400000 [00:16<00:34, 8115.43it/s] 31%|███       | 123929/400000 [00:16<00:34, 8083.33it/s] 31%|███       | 124739/400000 [00:16<00:34, 8049.65it/s] 31%|███▏      | 125584/400000 [00:16<00:33, 8162.93it/s] 32%|███▏      | 126426/400000 [00:16<00:33, 8236.04it/s] 32%|███▏      | 127251/400000 [00:16<00:33, 8157.91it/s] 32%|███▏      | 128068/400000 [00:16<00:33, 8013.27it/s] 32%|███▏      | 128871/400000 [00:16<00:34, 7962.98it/s] 32%|███▏      | 129705/400000 [00:16<00:33, 8070.15it/s] 33%|███▎      | 130543/400000 [00:16<00:33, 8158.17it/s] 33%|███▎      | 131360/400000 [00:17<00:32, 8141.09it/s] 33%|███▎      | 132175/400000 [00:17<00:33, 8053.20it/s] 33%|███▎      | 132981/400000 [00:17<00:33, 8032.37it/s] 33%|███▎      | 133785/400000 [00:17<00:33, 8018.40it/s] 34%|███▎      | 134588/400000 [00:17<00:33, 7981.41it/s] 34%|███▍      | 135387/400000 [00:17<00:33, 7968.07it/s] 34%|███▍      | 136185/400000 [00:17<00:33, 7856.36it/s] 34%|███▍      | 136972/400000 [00:17<00:34, 7584.81it/s] 34%|███▍      | 137733/400000 [00:17<00:34, 7535.25it/s] 35%|███▍      | 138527/400000 [00:17<00:34, 7649.98it/s] 35%|███▍      | 139321/400000 [00:18<00:33, 7734.16it/s] 35%|███▌      | 140107/400000 [00:18<00:33, 7768.92it/s] 35%|███▌      | 140885/400000 [00:18<00:33, 7771.07it/s] 35%|███▌      | 141663/400000 [00:18<00:33, 7743.94it/s] 36%|███▌      | 142471/400000 [00:18<00:32, 7839.74it/s] 36%|███▌      | 143306/400000 [00:18<00:32, 7983.78it/s] 36%|███▌      | 144106/400000 [00:18<00:32, 7851.66it/s] 36%|███▌      | 144908/400000 [00:18<00:32, 7900.10it/s] 36%|███▋      | 145741/400000 [00:18<00:31, 8023.46it/s] 37%|███▋      | 146550/400000 [00:18<00:31, 8040.86it/s] 37%|███▋      | 147385/400000 [00:19<00:31, 8129.52it/s] 37%|███▋      | 148199/400000 [00:19<00:31, 7937.42it/s] 37%|███▋      | 148995/400000 [00:19<00:31, 7868.33it/s] 37%|███▋      | 149796/400000 [00:19<00:31, 7910.16it/s] 38%|███▊      | 150588/400000 [00:19<00:32, 7709.95it/s] 38%|███▊      | 151379/400000 [00:19<00:32, 7767.86it/s] 38%|███▊      | 152158/400000 [00:19<00:32, 7655.52it/s] 38%|███▊      | 152932/400000 [00:19<00:32, 7680.35it/s] 38%|███▊      | 153701/400000 [00:19<00:32, 7681.70it/s] 39%|███▊      | 154470/400000 [00:19<00:32, 7590.49it/s] 39%|███▉      | 155248/400000 [00:20<00:32, 7644.52it/s] 39%|███▉      | 156014/400000 [00:20<00:32, 7528.37it/s] 39%|███▉      | 156817/400000 [00:20<00:31, 7670.64it/s] 39%|███▉      | 157629/400000 [00:20<00:31, 7798.09it/s] 40%|███▉      | 158437/400000 [00:20<00:30, 7878.58it/s] 40%|███▉      | 159254/400000 [00:20<00:30, 7961.72it/s] 40%|████      | 160052/400000 [00:20<00:30, 7935.77it/s] 40%|████      | 160885/400000 [00:20<00:29, 8048.97it/s] 40%|████      | 161715/400000 [00:20<00:29, 8119.95it/s] 41%|████      | 162528/400000 [00:21<00:29, 8063.51it/s] 41%|████      | 163335/400000 [00:21<00:29, 8035.06it/s] 41%|████      | 164139/400000 [00:21<00:29, 7954.36it/s] 41%|████      | 164970/400000 [00:21<00:29, 8057.50it/s] 41%|████▏     | 165809/400000 [00:21<00:28, 8152.40it/s] 42%|████▏     | 166639/400000 [00:21<00:28, 8195.07it/s] 42%|████▏     | 167460/400000 [00:21<00:28, 8066.44it/s] 42%|████▏     | 168268/400000 [00:21<00:29, 7953.25it/s] 42%|████▏     | 169096/400000 [00:21<00:28, 8047.69it/s] 42%|████▏     | 169908/400000 [00:21<00:28, 8064.58it/s] 43%|████▎     | 170719/400000 [00:22<00:28, 8076.93it/s] 43%|████▎     | 171528/400000 [00:22<00:28, 8041.60it/s] 43%|████▎     | 172333/400000 [00:22<00:28, 7940.60it/s] 43%|████▎     | 173128/400000 [00:22<00:28, 7901.33it/s] 43%|████▎     | 173956/400000 [00:22<00:28, 8010.44it/s] 44%|████▎     | 174800/400000 [00:22<00:27, 8131.94it/s] 44%|████▍     | 175615/400000 [00:22<00:27, 8063.95it/s] 44%|████▍     | 176423/400000 [00:22<00:28, 7922.48it/s] 44%|████▍     | 177261/400000 [00:22<00:27, 8053.56it/s] 45%|████▍     | 178068/400000 [00:22<00:27, 7994.88it/s] 45%|████▍     | 178899/400000 [00:23<00:27, 8084.91it/s] 45%|████▍     | 179726/400000 [00:23<00:27, 8137.96it/s] 45%|████▌     | 180541/400000 [00:23<00:27, 7934.62it/s] 45%|████▌     | 181337/400000 [00:23<00:28, 7788.34it/s] 46%|████▌     | 182131/400000 [00:23<00:27, 7830.10it/s] 46%|████▌     | 182916/400000 [00:23<00:27, 7832.06it/s] 46%|████▌     | 183701/400000 [00:23<00:27, 7743.11it/s] 46%|████▌     | 184477/400000 [00:23<00:28, 7662.46it/s] 46%|████▋     | 185256/400000 [00:23<00:27, 7696.53it/s] 47%|████▋     | 186049/400000 [00:23<00:27, 7762.90it/s] 47%|████▋     | 186840/400000 [00:24<00:27, 7806.10it/s] 47%|████▋     | 187622/400000 [00:24<00:27, 7781.19it/s] 47%|████▋     | 188401/400000 [00:24<00:27, 7720.73it/s] 47%|████▋     | 189174/400000 [00:24<00:28, 7496.52it/s] 47%|████▋     | 189926/400000 [00:24<00:28, 7490.35it/s] 48%|████▊     | 190677/400000 [00:24<00:27, 7486.59it/s] 48%|████▊     | 191427/400000 [00:24<00:27, 7479.65it/s] 48%|████▊     | 192176/400000 [00:24<00:27, 7472.07it/s] 48%|████▊     | 192924/400000 [00:24<00:27, 7450.03it/s] 48%|████▊     | 193670/400000 [00:24<00:28, 7161.14it/s] 49%|████▊     | 194389/400000 [00:25<00:28, 7168.73it/s] 49%|████▉     | 195108/400000 [00:25<00:29, 7064.37it/s] 49%|████▉     | 195817/400000 [00:25<00:29, 7003.92it/s] 49%|████▉     | 196519/400000 [00:25<00:29, 6945.79it/s] 49%|████▉     | 197245/400000 [00:25<00:28, 7035.65it/s] 49%|████▉     | 197997/400000 [00:25<00:28, 7172.51it/s] 50%|████▉     | 198799/400000 [00:25<00:27, 7406.58it/s] 50%|████▉     | 199615/400000 [00:25<00:26, 7615.27it/s] 50%|█████     | 200380/400000 [00:25<00:26, 7513.09it/s] 50%|█████     | 201135/400000 [00:26<00:26, 7375.85it/s] 50%|█████     | 201876/400000 [00:26<00:27, 7151.56it/s] 51%|█████     | 202595/400000 [00:26<00:27, 7155.01it/s] 51%|█████     | 203313/400000 [00:26<00:27, 7114.11it/s] 51%|█████     | 204062/400000 [00:26<00:27, 7222.43it/s] 51%|█████     | 204838/400000 [00:26<00:26, 7373.47it/s] 51%|█████▏    | 205590/400000 [00:26<00:26, 7415.98it/s] 52%|█████▏    | 206333/400000 [00:26<00:26, 7328.77it/s] 52%|█████▏    | 207068/400000 [00:26<00:26, 7266.84it/s] 52%|█████▏    | 207825/400000 [00:26<00:26, 7353.81it/s] 52%|█████▏    | 208600/400000 [00:27<00:25, 7465.90it/s] 52%|█████▏    | 209354/400000 [00:27<00:25, 7479.57it/s] 53%|█████▎    | 210103/400000 [00:27<00:25, 7390.32it/s] 53%|█████▎    | 210865/400000 [00:27<00:25, 7455.61it/s] 53%|█████▎    | 211638/400000 [00:27<00:25, 7533.70it/s] 53%|█████▎    | 212409/400000 [00:27<00:24, 7581.34it/s] 53%|█████▎    | 213168/400000 [00:27<00:25, 7431.85it/s] 53%|█████▎    | 213935/400000 [00:27<00:24, 7501.29it/s] 54%|█████▎    | 214687/400000 [00:27<00:24, 7471.28it/s] 54%|█████▍    | 215473/400000 [00:27<00:24, 7581.98it/s] 54%|█████▍    | 216239/400000 [00:28<00:24, 7603.72it/s] 54%|█████▍    | 217000/400000 [00:28<00:24, 7460.30it/s] 54%|█████▍    | 217750/400000 [00:28<00:24, 7470.27it/s] 55%|█████▍    | 218498/400000 [00:28<00:24, 7471.24it/s] 55%|█████▍    | 219270/400000 [00:28<00:23, 7542.33it/s] 55%|█████▌    | 220032/400000 [00:28<00:23, 7562.71it/s] 55%|█████▌    | 220789/400000 [00:28<00:24, 7444.86it/s] 55%|█████▌    | 221535/400000 [00:28<00:24, 7332.85it/s] 56%|█████▌    | 222277/400000 [00:28<00:24, 7356.34it/s] 56%|█████▌    | 223014/400000 [00:28<00:24, 7321.72it/s] 56%|█████▌    | 223766/400000 [00:29<00:23, 7378.39it/s] 56%|█████▌    | 224505/400000 [00:29<00:23, 7369.87it/s] 56%|█████▋    | 225243/400000 [00:29<00:24, 7229.74it/s] 56%|█████▋    | 225967/400000 [00:29<00:24, 7195.63it/s] 57%|█████▋    | 226688/400000 [00:29<00:24, 7166.92it/s] 57%|█████▋    | 227484/400000 [00:29<00:23, 7386.52it/s] 57%|█████▋    | 228317/400000 [00:29<00:22, 7643.89it/s] 57%|█████▋    | 229101/400000 [00:29<00:22, 7700.32it/s] 57%|█████▋    | 229874/400000 [00:29<00:22, 7559.44it/s] 58%|█████▊    | 230633/400000 [00:29<00:22, 7562.59it/s] 58%|█████▊    | 231391/400000 [00:30<00:22, 7565.53it/s] 58%|█████▊    | 232161/400000 [00:30<00:22, 7603.97it/s] 58%|█████▊    | 232923/400000 [00:30<00:22, 7521.83it/s] 58%|█████▊    | 233725/400000 [00:30<00:21, 7662.95it/s] 59%|█████▊    | 234512/400000 [00:30<00:21, 7722.24it/s] 59%|█████▉    | 235287/400000 [00:30<00:21, 7728.82it/s] 59%|█████▉    | 236061/400000 [00:30<00:22, 7176.99it/s] 59%|█████▉    | 236825/400000 [00:30<00:22, 7308.65it/s] 59%|█████▉    | 237590/400000 [00:30<00:21, 7407.71it/s] 60%|█████▉    | 238372/400000 [00:31<00:21, 7525.37it/s] 60%|█████▉    | 239177/400000 [00:31<00:20, 7674.56it/s] 60%|█████▉    | 239948/400000 [00:31<00:20, 7675.66it/s] 60%|██████    | 240738/400000 [00:31<00:20, 7740.65it/s] 60%|██████    | 241514/400000 [00:31<00:20, 7676.54it/s] 61%|██████    | 242309/400000 [00:31<00:20, 7755.44it/s] 61%|██████    | 243086/400000 [00:31<00:20, 7728.33it/s] 61%|██████    | 243900/400000 [00:31<00:19, 7847.27it/s] 61%|██████    | 244706/400000 [00:31<00:19, 7908.09it/s] 61%|██████▏   | 245506/400000 [00:31<00:19, 7933.30it/s] 62%|██████▏   | 246351/400000 [00:32<00:19, 8080.68it/s] 62%|██████▏   | 247161/400000 [00:32<00:18, 8048.89it/s] 62%|██████▏   | 247967/400000 [00:32<00:18, 8044.09it/s] 62%|██████▏   | 248772/400000 [00:32<00:18, 7970.70it/s] 62%|██████▏   | 249596/400000 [00:32<00:18, 8047.26it/s] 63%|██████▎   | 250456/400000 [00:32<00:18, 8205.41it/s] 63%|██████▎   | 251285/400000 [00:32<00:18, 8230.14it/s] 63%|██████▎   | 252113/400000 [00:32<00:17, 8242.54it/s] 63%|██████▎   | 252938/400000 [00:32<00:18, 8131.46it/s] 63%|██████▎   | 253752/400000 [00:32<00:18, 7990.62it/s] 64%|██████▎   | 254553/400000 [00:33<00:18, 7910.53it/s] 64%|██████▍   | 255364/400000 [00:33<00:18, 7968.39it/s] 64%|██████▍   | 256162/400000 [00:33<00:18, 7873.87it/s] 64%|██████▍   | 256951/400000 [00:33<00:18, 7771.76it/s] 64%|██████▍   | 257735/400000 [00:33<00:18, 7790.20it/s] 65%|██████▍   | 258532/400000 [00:33<00:18, 7840.74it/s] 65%|██████▍   | 259349/400000 [00:33<00:17, 7934.98it/s] 65%|██████▌   | 260152/400000 [00:33<00:17, 7962.79it/s] 65%|██████▌   | 260962/400000 [00:33<00:17, 8002.46it/s] 65%|██████▌   | 261766/400000 [00:33<00:17, 8012.45it/s] 66%|██████▌   | 262596/400000 [00:34<00:16, 8096.03it/s] 66%|██████▌   | 263406/400000 [00:34<00:16, 8054.34it/s] 66%|██████▌   | 264212/400000 [00:34<00:17, 7788.83it/s] 66%|██████▌   | 264994/400000 [00:34<00:18, 7403.36it/s] 66%|██████▋   | 265747/400000 [00:34<00:18, 7439.02it/s] 67%|██████▋   | 266519/400000 [00:34<00:17, 7519.70it/s] 67%|██████▋   | 267275/400000 [00:34<00:17, 7530.52it/s] 67%|██████▋   | 268095/400000 [00:34<00:17, 7718.69it/s] 67%|██████▋   | 268895/400000 [00:34<00:16, 7800.46it/s] 67%|██████▋   | 269678/400000 [00:34<00:17, 7663.27it/s] 68%|██████▊   | 270494/400000 [00:35<00:16, 7804.27it/s] 68%|██████▊   | 271277/400000 [00:35<00:16, 7677.90it/s] 68%|██████▊   | 272047/400000 [00:35<00:16, 7683.95it/s] 68%|██████▊   | 272817/400000 [00:35<00:16, 7609.27it/s] 68%|██████▊   | 273623/400000 [00:35<00:16, 7738.96it/s] 69%|██████▊   | 274436/400000 [00:35<00:15, 7850.47it/s] 69%|██████▉   | 275249/400000 [00:35<00:15, 7932.23it/s] 69%|██████▉   | 276071/400000 [00:35<00:15, 8014.56it/s] 69%|██████▉   | 276874/400000 [00:35<00:15, 7924.75it/s] 69%|██████▉   | 277668/400000 [00:36<00:15, 7919.29it/s] 70%|██████▉   | 278461/400000 [00:36<00:15, 7904.37it/s] 70%|██████▉   | 279252/400000 [00:36<00:15, 7802.58it/s] 70%|███████   | 280033/400000 [00:36<00:15, 7764.79it/s] 70%|███████   | 280810/400000 [00:36<00:15, 7675.46it/s] 70%|███████   | 281588/400000 [00:36<00:15, 7704.44it/s] 71%|███████   | 282359/400000 [00:36<00:15, 7524.75it/s] 71%|███████   | 283154/400000 [00:36<00:15, 7647.24it/s] 71%|███████   | 284029/400000 [00:36<00:14, 7945.44it/s] 71%|███████   | 284828/400000 [00:36<00:14, 7923.23it/s] 71%|███████▏  | 285659/400000 [00:37<00:14, 8034.95it/s] 72%|███████▏  | 286507/400000 [00:37<00:13, 8162.53it/s] 72%|███████▏  | 287326/400000 [00:37<00:14, 7752.35it/s] 72%|███████▏  | 288112/400000 [00:37<00:14, 7782.50it/s] 72%|███████▏  | 288895/400000 [00:37<00:14, 7610.40it/s] 72%|███████▏  | 289660/400000 [00:37<00:14, 7535.63it/s] 73%|███████▎  | 290417/400000 [00:37<00:14, 7498.06it/s] 73%|███████▎  | 291239/400000 [00:37<00:14, 7699.12it/s] 73%|███████▎  | 292071/400000 [00:37<00:13, 7874.26it/s] 73%|███████▎  | 292878/400000 [00:37<00:13, 7930.83it/s] 73%|███████▎  | 293747/400000 [00:38<00:13, 8143.35it/s] 74%|███████▎  | 294583/400000 [00:38<00:12, 8205.60it/s] 74%|███████▍  | 295406/400000 [00:38<00:12, 8069.02it/s] 74%|███████▍  | 296215/400000 [00:38<00:12, 8048.93it/s] 74%|███████▍  | 297022/400000 [00:38<00:12, 7973.79it/s] 74%|███████▍  | 297849/400000 [00:38<00:12, 8058.61it/s] 75%|███████▍  | 298656/400000 [00:38<00:12, 7976.95it/s] 75%|███████▍  | 299455/400000 [00:38<00:12, 7916.00it/s] 75%|███████▌  | 300258/400000 [00:38<00:12, 7947.99it/s] 75%|███████▌  | 301054/400000 [00:38<00:13, 7498.80it/s] 75%|███████▌  | 301878/400000 [00:39<00:12, 7706.68it/s] 76%|███████▌  | 302670/400000 [00:39<00:12, 7767.10it/s] 76%|███████▌  | 303479/400000 [00:39<00:12, 7860.36it/s] 76%|███████▌  | 304310/400000 [00:39<00:11, 7987.47it/s] 76%|███████▋  | 305112/400000 [00:39<00:12, 7892.79it/s] 76%|███████▋  | 305936/400000 [00:39<00:11, 7993.20it/s] 77%|███████▋  | 306738/400000 [00:39<00:11, 7883.50it/s] 77%|███████▋  | 307560/400000 [00:39<00:11, 7978.43it/s] 77%|███████▋  | 308360/400000 [00:39<00:12, 7555.13it/s] 77%|███████▋  | 309134/400000 [00:40<00:11, 7608.88it/s] 77%|███████▋  | 309927/400000 [00:40<00:11, 7701.14it/s] 78%|███████▊  | 310712/400000 [00:40<00:11, 7742.05it/s] 78%|███████▊  | 311516/400000 [00:40<00:11, 7826.12it/s] 78%|███████▊  | 312331/400000 [00:40<00:11, 7919.21it/s] 78%|███████▊  | 313198/400000 [00:40<00:10, 8128.55it/s] 79%|███████▊  | 314029/400000 [00:40<00:10, 8179.72it/s] 79%|███████▊  | 314880/400000 [00:40<00:10, 8274.00it/s] 79%|███████▉  | 315709/400000 [00:40<00:10, 8172.10it/s] 79%|███████▉  | 316528/400000 [00:40<00:10, 7933.11it/s] 79%|███████▉  | 317385/400000 [00:41<00:10, 8112.25it/s] 80%|███████▉  | 318206/400000 [00:41<00:10, 8139.78it/s] 80%|███████▉  | 319043/400000 [00:41<00:09, 8205.88it/s] 80%|███████▉  | 319884/400000 [00:41<00:09, 8265.14it/s] 80%|████████  | 320712/400000 [00:41<00:09, 8244.69it/s] 80%|████████  | 321573/400000 [00:41<00:09, 8350.45it/s] 81%|████████  | 322409/400000 [00:41<00:09, 8256.15it/s] 81%|████████  | 323266/400000 [00:41<00:09, 8346.31it/s] 81%|████████  | 324102/400000 [00:41<00:09, 8340.51it/s] 81%|████████  | 324937/400000 [00:41<00:09, 7912.40it/s] 81%|████████▏ | 325734/400000 [00:42<00:09, 7900.62it/s] 82%|████████▏ | 326528/400000 [00:42<00:09, 7562.24it/s] 82%|████████▏ | 327305/400000 [00:42<00:09, 7621.63it/s] 82%|████████▏ | 328114/400000 [00:42<00:09, 7754.35it/s] 82%|████████▏ | 328897/400000 [00:42<00:09, 7775.08it/s] 82%|████████▏ | 329745/400000 [00:42<00:08, 7971.18it/s] 83%|████████▎ | 330586/400000 [00:42<00:08, 8097.47it/s] 83%|████████▎ | 331445/400000 [00:42<00:08, 8238.50it/s] 83%|████████▎ | 332275/400000 [00:42<00:08, 8254.81it/s] 83%|████████▎ | 333103/400000 [00:42<00:08, 8113.14it/s] 83%|████████▎ | 333954/400000 [00:43<00:08, 8225.43it/s] 84%|████████▎ | 334779/400000 [00:43<00:07, 8195.99it/s] 84%|████████▍ | 335600/400000 [00:43<00:08, 7994.69it/s] 84%|████████▍ | 336419/400000 [00:43<00:07, 8051.83it/s] 84%|████████▍ | 337258/400000 [00:43<00:07, 8148.05it/s] 85%|████████▍ | 338076/400000 [00:43<00:07, 8155.87it/s] 85%|████████▍ | 338898/400000 [00:43<00:07, 8174.94it/s] 85%|████████▍ | 339717/400000 [00:43<00:07, 8140.77it/s] 85%|████████▌ | 340537/400000 [00:43<00:07, 8156.85it/s] 85%|████████▌ | 341354/400000 [00:44<00:07, 8070.40it/s] 86%|████████▌ | 342217/400000 [00:44<00:07, 8229.36it/s] 86%|████████▌ | 343052/400000 [00:44<00:06, 8262.92it/s] 86%|████████▌ | 343880/400000 [00:44<00:06, 8110.15it/s] 86%|████████▌ | 344695/400000 [00:44<00:06, 8119.73it/s] 86%|████████▋ | 345508/400000 [00:44<00:06, 8036.34it/s] 87%|████████▋ | 346313/400000 [00:44<00:06, 8036.89it/s] 87%|████████▋ | 347142/400000 [00:44<00:06, 8109.31it/s] 87%|████████▋ | 347973/400000 [00:44<00:06, 8165.78it/s] 87%|████████▋ | 348791/400000 [00:44<00:06, 8005.32it/s] 87%|████████▋ | 349593/400000 [00:45<00:06, 7867.68it/s] 88%|████████▊ | 350404/400000 [00:45<00:06, 7937.98it/s] 88%|████████▊ | 351199/400000 [00:45<00:06, 7851.45it/s] 88%|████████▊ | 352003/400000 [00:45<00:06, 7903.74it/s] 88%|████████▊ | 352812/400000 [00:45<00:05, 7958.18it/s] 88%|████████▊ | 353627/400000 [00:45<00:05, 8012.81it/s] 89%|████████▊ | 354450/400000 [00:45<00:05, 8074.75it/s] 89%|████████▉ | 355292/400000 [00:45<00:05, 8173.20it/s] 89%|████████▉ | 356135/400000 [00:45<00:05, 8245.80it/s] 89%|████████▉ | 356961/400000 [00:45<00:05, 7969.36it/s] 89%|████████▉ | 357761/400000 [00:46<00:05, 7845.98it/s] 90%|████████▉ | 358581/400000 [00:46<00:05, 7946.92it/s] 90%|████████▉ | 359400/400000 [00:46<00:05, 8017.20it/s] 90%|█████████ | 360204/400000 [00:46<00:04, 7990.02it/s] 90%|█████████ | 361005/400000 [00:46<00:04, 7963.55it/s] 90%|█████████ | 361803/400000 [00:46<00:04, 7886.45it/s] 91%|█████████ | 362593/400000 [00:46<00:04, 7743.87it/s] 91%|█████████ | 363438/400000 [00:46<00:04, 7940.76it/s] 91%|█████████ | 364240/400000 [00:46<00:04, 7963.75it/s] 91%|█████████▏| 365048/400000 [00:46<00:04, 7996.07it/s] 91%|█████████▏| 365856/400000 [00:47<00:04, 8020.46it/s] 92%|█████████▏| 366712/400000 [00:47<00:04, 8172.25it/s] 92%|█████████▏| 367531/400000 [00:47<00:04, 7989.98it/s] 92%|█████████▏| 368343/400000 [00:47<00:03, 8027.85it/s] 92%|█████████▏| 369148/400000 [00:47<00:03, 7986.24it/s] 92%|█████████▏| 369948/400000 [00:47<00:03, 7868.11it/s] 93%|█████████▎| 370736/400000 [00:47<00:03, 7804.14it/s] 93%|█████████▎| 371522/400000 [00:47<00:03, 7818.36it/s] 93%|█████████▎| 372310/400000 [00:47<00:03, 7834.41it/s] 93%|█████████▎| 373094/400000 [00:47<00:03, 7532.48it/s] 93%|█████████▎| 373851/400000 [00:48<00:03, 7498.16it/s] 94%|█████████▎| 374608/400000 [00:48<00:03, 7517.76it/s] 94%|█████████▍| 375362/400000 [00:48<00:03, 7460.20it/s] 94%|█████████▍| 376141/400000 [00:48<00:03, 7554.98it/s] 94%|█████████▍| 376914/400000 [00:48<00:03, 7604.13it/s] 94%|█████████▍| 377690/400000 [00:48<00:02, 7649.25it/s] 95%|█████████▍| 378530/400000 [00:48<00:02, 7858.64it/s] 95%|█████████▍| 379318/400000 [00:48<00:02, 7604.95it/s] 95%|█████████▌| 380083/400000 [00:48<00:02, 7617.09it/s] 95%|█████████▌| 380872/400000 [00:49<00:02, 7696.72it/s] 95%|█████████▌| 381651/400000 [00:49<00:02, 7722.92it/s] 96%|█████████▌| 382435/400000 [00:49<00:02, 7756.40it/s] 96%|█████████▌| 383227/400000 [00:49<00:02, 7803.41it/s] 96%|█████████▌| 384022/400000 [00:49<00:02, 7845.97it/s] 96%|█████████▌| 384838/400000 [00:49<00:01, 7935.40it/s] 96%|█████████▋| 385633/400000 [00:49<00:01, 7919.30it/s] 97%|█████████▋| 386433/400000 [00:49<00:01, 7939.72it/s] 97%|█████████▋| 387228/400000 [00:49<00:01, 7930.13it/s] 97%|█████████▋| 388022/400000 [00:49<00:01, 7911.23it/s] 97%|█████████▋| 388814/400000 [00:50<00:01, 7897.80it/s] 97%|█████████▋| 389621/400000 [00:50<00:01, 7946.74it/s] 98%|█████████▊| 390420/400000 [00:50<00:01, 7957.34it/s] 98%|█████████▊| 391216/400000 [00:50<00:01, 7896.94it/s] 98%|█████████▊| 392015/400000 [00:50<00:01, 7921.84it/s] 98%|█████████▊| 392809/400000 [00:50<00:00, 7923.24it/s] 98%|█████████▊| 393615/400000 [00:50<00:00, 7962.73it/s] 99%|█████████▊| 394419/400000 [00:50<00:00, 7982.72it/s] 99%|█████████▉| 395218/400000 [00:50<00:00, 7733.68it/s] 99%|█████████▉| 396004/400000 [00:50<00:00, 7769.44it/s] 99%|█████████▉| 396783/400000 [00:51<00:00, 7714.36it/s] 99%|█████████▉| 397588/400000 [00:51<00:00, 7811.22it/s]100%|█████████▉| 398385/400000 [00:51<00:00, 7856.57it/s]100%|█████████▉| 399194/400000 [00:51<00:00, 7924.51it/s]100%|█████████▉| 399999/400000 [00:51<00:00, 7777.71it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f77f6c73d68> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01098463693913472 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011001522325751774 	 Accuracy: 55

  model saves at 55% accuracy 

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
2020-05-20 13:00:51.857641: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-20 13:00:51.861132: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294690000 Hz
2020-05-20 13:00:51.861308: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55eebd354ad0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-20 13:00:51.861323: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f779a4681d0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9273 - accuracy: 0.4830
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6436 - accuracy: 0.5015 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8046 - accuracy: 0.4910
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7433 - accuracy: 0.4950
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7280 - accuracy: 0.4960
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7356 - accuracy: 0.4955
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7192 - accuracy: 0.4966
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7414 - accuracy: 0.4951
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7245 - accuracy: 0.4962
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7050 - accuracy: 0.4975
11000/25000 [============>.................] - ETA: 4s - loss: 7.7294 - accuracy: 0.4959
12000/25000 [=============>................] - ETA: 4s - loss: 7.7075 - accuracy: 0.4973
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6572 - accuracy: 0.5006
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6579 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 3s - loss: 7.6554 - accuracy: 0.5007
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6618 - accuracy: 0.5003
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6441 - accuracy: 0.5015
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6598 - accuracy: 0.5004
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6642 - accuracy: 0.5002
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6636 - accuracy: 0.5002
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6652 - accuracy: 0.5001
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6569 - accuracy: 0.5006
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6613 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6494 - accuracy: 0.5011
25000/25000 [==============================] - 10s 380us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f774fad54e0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f775772ea90> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 951ms/step - loss: 1.9302 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.9400 - val_crf_viterbi_accuracy: 0.3333

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

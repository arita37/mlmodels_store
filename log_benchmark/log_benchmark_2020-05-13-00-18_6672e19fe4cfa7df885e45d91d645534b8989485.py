
  test_benchmark /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_benchmark', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_benchmark 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '6672e19fe4cfa7df885e45d91d645534b8989485', 'workflow': 'test_benchmark'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_benchmark

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/6672e19fe4cfa7df885e45d91d645534b8989485

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/6672e19fe4cfa7df885e45d91d645534b8989485

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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f6b17942fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 00:18:20.526987
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 00:18:20.530485
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 00:18:20.533740
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 00:18:20.536831
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f6b2370c518> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355707.3438
Epoch 2/10

1/1 [==============================] - 0s 97ms/step - loss: 289657.0938
Epoch 3/10

1/1 [==============================] - 0s 96ms/step - loss: 214438.5000
Epoch 4/10

1/1 [==============================] - 0s 96ms/step - loss: 140190.1094
Epoch 5/10

1/1 [==============================] - 0s 92ms/step - loss: 87413.0391
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 54584.0117
Epoch 7/10

1/1 [==============================] - 0s 91ms/step - loss: 35163.3906
Epoch 8/10

1/1 [==============================] - 0s 88ms/step - loss: 23666.8281
Epoch 9/10

1/1 [==============================] - 0s 89ms/step - loss: 16726.8496
Epoch 10/10

1/1 [==============================] - 0s 91ms/step - loss: 12359.4141

  #### Inference Need return ypred, ytrue ######################### 
[[-2.09832534e-01  5.13850403e+00  7.16141796e+00  5.62701988e+00
   5.85469627e+00  6.09553719e+00  6.13608027e+00  6.84292269e+00
   6.34140682e+00  4.37919474e+00  6.85072947e+00  4.93251801e+00
   5.88224316e+00  5.75855494e+00  6.20893049e+00  7.39087343e+00
   4.52676964e+00  4.33514595e+00  5.91639280e+00  3.72912359e+00
   5.48296404e+00  4.97638559e+00  4.93268061e+00  6.10902023e+00
   5.03562593e+00  5.11160851e+00  4.31522083e+00  4.40983486e+00
   5.54501295e+00  5.82576370e+00  6.11621666e+00  6.19439745e+00
   6.72306252e+00  5.81259871e+00  6.12746048e+00  6.42522192e+00
   5.25026798e+00  6.29010725e+00  6.62066174e+00  5.93136883e+00
   3.92129397e+00  4.75820732e+00  5.53588915e+00  4.06069088e+00
   5.86877060e+00  5.61603546e+00  4.52637100e+00  4.92729044e+00
   6.57287598e+00  6.01682425e+00  6.05417013e+00  5.52424192e+00
   5.90432978e+00  6.65635538e+00  5.29268551e+00  4.25574684e+00
   3.90966129e+00  4.60545444e+00  6.07006359e+00  4.28192091e+00
  -1.26000941e+00  1.66902840e-02 -1.05006957e+00  3.74583125e-01
   1.42965949e+00 -1.77958667e-01  1.04208565e+00 -5.27446389e-01
   1.43949032e-01  5.72850525e-01 -6.86048865e-01  4.81561273e-01
  -1.15523231e+00 -8.68934751e-01  1.67328227e+00  9.97427821e-01
   1.44239366e-01 -8.97814512e-01 -9.31150615e-01 -1.67584884e+00
  -2.59949148e-01  7.49439836e-01  5.59694946e-01 -7.38267481e-01
   1.40520322e+00 -7.60211706e-01  3.18936050e-01 -1.47931910e+00
   7.96912432e-01 -4.35371399e-01 -6.16912842e-02 -1.10477746e-01
   5.58597803e-01 -2.51128793e-01 -2.15058863e-01 -1.58892167e+00
  -2.48550773e-02  6.02429628e-01  8.27554226e-01  3.06867898e-01
  -3.06735098e-01  5.96255660e-02 -1.19444877e-01  1.32852602e+00
   1.81676507e+00 -7.08259284e-01  9.25944686e-01  1.00954771e+00
  -7.45983958e-01 -1.04717433e+00 -1.40742970e+00  1.01018381e+00
   1.45546532e+00 -5.28084576e-01  1.40237856e+00 -2.46113762e-01
   3.90091270e-01  1.00826740e-01 -9.66682732e-02  5.78081071e-01
   3.54490876e-01  1.40883565e+00  1.70340586e+00  1.15366280e+00
   9.99792933e-01  1.62545133e+00 -4.43705916e-01  4.55384791e-01
  -4.96218145e-01  4.77395475e-01  1.76108265e+00  2.90943682e-01
   1.61018461e-01 -3.91180396e-01  1.57002807e-01  7.78222442e-01
   8.54144394e-01 -1.13034151e-01  5.40381670e-03  4.50680614e-01
  -1.00149643e+00  7.02108443e-02  8.08724403e-01 -7.05924749e-01
  -1.20568824e+00 -4.19819057e-01  1.69374436e-01 -5.94188333e-01
   4.22091097e-01 -1.02021480e+00  9.12581265e-01 -8.81931603e-01
   3.08560431e-01  3.56534332e-01 -7.64462113e-01 -1.09374869e+00
   8.70274901e-01  7.62154311e-02 -1.74087584e-02 -9.74906266e-01
   7.27526009e-01 -5.43968797e-01  5.21708012e-01 -2.09216565e-01
  -8.07558358e-01 -3.88444304e-01 -3.38155359e-01  5.91256320e-02
   7.38238275e-01  5.74207246e-01 -1.72848463e-01  1.52018189e+00
   9.40188110e-01 -7.41187215e-01 -3.85132194e-01 -8.79931688e-01
   5.95391393e-02 -2.06410825e-01  2.49709010e-01 -7.88278639e-01
   7.58301020e-02  6.02710152e+00  7.51998520e+00  6.41308403e+00
   7.66906261e+00  6.05102444e+00  5.15328884e+00  7.08971262e+00
   6.34639406e+00  7.01289368e+00  5.56669998e+00  6.80224752e+00
   6.78260612e+00  5.76559448e+00  4.57541513e+00  4.71844292e+00
   4.90843058e+00  6.06028318e+00  6.15818167e+00  6.33005571e+00
   6.35123587e+00  7.11020708e+00  5.72736168e+00  6.39575005e+00
   6.97280836e+00  5.83762789e+00  7.00368071e+00  7.61116695e+00
   5.55457592e+00  6.14049244e+00  7.09846973e+00  5.67285395e+00
   5.34187841e+00  7.51370001e+00  7.43197680e+00  6.09844398e+00
   5.09685183e+00  5.23979664e+00  6.45909739e+00  6.69030094e+00
   5.18723583e+00  6.65139484e+00  5.92822981e+00  5.50446606e+00
   5.72641706e+00  5.84341192e+00  6.77007484e+00  7.81261444e+00
   6.73357439e+00  5.49251747e+00  5.62851572e+00  7.24810982e+00
   6.74215698e+00  6.13490820e+00  5.69034624e+00  6.25994205e+00
   5.53019667e+00  4.68216276e+00  6.66819286e+00  6.21847916e+00
   4.13608372e-01  1.67704499e+00  4.68563139e-01  2.16141522e-01
   4.56436753e-01  3.91318798e-01  2.00929189e+00  1.05512786e+00
   5.82272232e-01  1.55798173e+00  2.00768280e+00  1.17283845e+00
   2.41367102e+00  5.11570215e-01  2.71817207e-01  1.47596884e+00
   1.67316663e+00  2.30957329e-01  7.50216842e-01  7.02896237e-01
   9.82139051e-01  7.50086188e-01  1.34408927e+00  6.35910511e-01
   1.36415243e+00  1.33405447e+00  2.41111779e+00  8.01453352e-01
   1.58463490e+00  1.17351627e+00  2.61040270e-01  2.29742050e+00
   1.05431843e+00  2.98386991e-01  1.58525395e+00  1.28679156e+00
   2.13849282e+00  5.99843144e-01  4.84265447e-01  3.35245967e-01
   6.45815969e-01  1.28813982e+00  2.05357385e+00  5.92775404e-01
   5.84118009e-01  4.63234901e-01  6.79763496e-01  1.52457881e+00
   1.55433023e+00  3.98983300e-01  4.51848984e-01  2.64884043e+00
   2.84068704e-01  1.13286066e+00  7.69400895e-01  1.09649515e+00
   2.15480709e+00  4.37274337e-01  1.68339610e+00  2.75442123e+00
   1.54092956e+00  1.88892925e+00  1.38105989e+00  1.07977486e+00
   6.29167855e-01  4.31560814e-01  1.70925832e+00  2.73175764e+00
   1.87892640e+00  3.37783098e-01  1.63144612e+00  5.70767105e-01
   5.98038673e-01  1.54327893e+00  2.79307508e+00  1.85421610e+00
   1.06291032e+00  3.76561463e-01  2.79859543e-01  6.99990213e-01
   4.03985023e-01  2.71060467e-01  1.40846014e+00  1.29641891e+00
   9.53774929e-01  2.19279468e-01  1.41435111e+00  1.00374103e+00
   1.38323832e+00  1.57989502e+00  1.06869817e+00  1.56206274e+00
   1.44352138e+00  6.48191333e-01  5.24230778e-01  2.57112980e-01
   1.93339229e+00  5.75470984e-01  6.30877197e-01  8.89379501e-01
   1.78763568e-01  1.77585614e+00  2.81484842e-01  2.34043932e+00
   4.11452174e-01  7.74901211e-01  6.04267836e-01  9.75393832e-01
   1.81892347e+00  3.79195750e-01  1.76417971e+00  9.73552942e-01
   2.00352669e+00  4.44987297e-01  2.21887469e-01  2.35420203e+00
   1.06159747e+00  6.73941970e-01  7.55310237e-01  2.44828844e+00
   3.03074837e+00 -1.01449099e+01 -5.15179253e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 00:18:28.938855
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.4799
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 00:18:28.942785
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9328.16
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 00:18:28.945665
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.2671
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 00:18:28.948359
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -834.403
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140097574684264
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140096364790448
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140096364790952
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140096364791456
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140096364791960
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140096364792464

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f6b17942240> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.525702
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.489608
grad_step = 000002, loss = 0.460142
grad_step = 000003, loss = 0.424131
grad_step = 000004, loss = 0.386794
grad_step = 000005, loss = 0.356065
grad_step = 000006, loss = 0.344046
grad_step = 000007, loss = 0.331353
grad_step = 000008, loss = 0.308839
grad_step = 000009, loss = 0.286430
grad_step = 000010, loss = 0.270586
grad_step = 000011, loss = 0.259981
grad_step = 000012, loss = 0.250521
grad_step = 000013, loss = 0.240176
grad_step = 000014, loss = 0.229037
grad_step = 000015, loss = 0.217214
grad_step = 000016, loss = 0.205043
grad_step = 000017, loss = 0.193931
grad_step = 000018, loss = 0.184758
grad_step = 000019, loss = 0.176305
grad_step = 000020, loss = 0.167523
grad_step = 000021, loss = 0.158419
grad_step = 000022, loss = 0.149509
grad_step = 000023, loss = 0.141187
grad_step = 000024, loss = 0.133404
grad_step = 000025, loss = 0.126028
grad_step = 000026, loss = 0.118859
grad_step = 000027, loss = 0.111705
grad_step = 000028, loss = 0.104770
grad_step = 000029, loss = 0.098458
grad_step = 000030, loss = 0.092710
grad_step = 000031, loss = 0.087115
grad_step = 000032, loss = 0.081520
grad_step = 000033, loss = 0.076093
grad_step = 000034, loss = 0.070992
grad_step = 000035, loss = 0.066275
grad_step = 000036, loss = 0.061893
grad_step = 000037, loss = 0.057728
grad_step = 000038, loss = 0.053712
grad_step = 000039, loss = 0.049941
grad_step = 000040, loss = 0.046443
grad_step = 000041, loss = 0.043223
grad_step = 000042, loss = 0.040210
grad_step = 000043, loss = 0.037254
grad_step = 000044, loss = 0.034370
grad_step = 000045, loss = 0.031758
grad_step = 000046, loss = 0.029450
grad_step = 000047, loss = 0.027307
grad_step = 000048, loss = 0.025268
grad_step = 000049, loss = 0.023331
grad_step = 000050, loss = 0.021525
grad_step = 000051, loss = 0.019895
grad_step = 000052, loss = 0.018416
grad_step = 000053, loss = 0.017020
grad_step = 000054, loss = 0.015700
grad_step = 000055, loss = 0.014492
grad_step = 000056, loss = 0.013409
grad_step = 000057, loss = 0.012412
grad_step = 000058, loss = 0.011481
grad_step = 000059, loss = 0.010631
grad_step = 000060, loss = 0.009862
grad_step = 000061, loss = 0.009168
grad_step = 000062, loss = 0.008540
grad_step = 000063, loss = 0.007953
grad_step = 000064, loss = 0.007405
grad_step = 000065, loss = 0.006909
grad_step = 000066, loss = 0.006455
grad_step = 000067, loss = 0.006038
grad_step = 000068, loss = 0.005653
grad_step = 000069, loss = 0.005296
grad_step = 000070, loss = 0.004969
grad_step = 000071, loss = 0.004675
grad_step = 000072, loss = 0.004402
grad_step = 000073, loss = 0.004150
grad_step = 000074, loss = 0.003922
grad_step = 000075, loss = 0.003717
grad_step = 000076, loss = 0.003533
grad_step = 000077, loss = 0.003364
grad_step = 000078, loss = 0.003212
grad_step = 000079, loss = 0.003079
grad_step = 000080, loss = 0.002959
grad_step = 000081, loss = 0.002851
grad_step = 000082, loss = 0.002755
grad_step = 000083, loss = 0.002669
grad_step = 000084, loss = 0.002593
grad_step = 000085, loss = 0.002525
grad_step = 000086, loss = 0.002465
grad_step = 000087, loss = 0.002412
grad_step = 000088, loss = 0.002366
grad_step = 000089, loss = 0.002326
grad_step = 000090, loss = 0.002291
grad_step = 000091, loss = 0.002262
grad_step = 000092, loss = 0.002242
grad_step = 000093, loss = 0.002237
grad_step = 000094, loss = 0.002241
grad_step = 000095, loss = 0.002230
grad_step = 000096, loss = 0.002190
grad_step = 000097, loss = 0.002158
grad_step = 000098, loss = 0.002160
grad_step = 000099, loss = 0.002173
grad_step = 000100, loss = 0.002167
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002140
grad_step = 000102, loss = 0.002124
grad_step = 000103, loss = 0.002131
grad_step = 000104, loss = 0.002139
grad_step = 000105, loss = 0.002132
grad_step = 000106, loss = 0.002113
grad_step = 000107, loss = 0.002101
grad_step = 000108, loss = 0.002103
grad_step = 000109, loss = 0.002108
grad_step = 000110, loss = 0.002105
grad_step = 000111, loss = 0.002093
grad_step = 000112, loss = 0.002078
grad_step = 000113, loss = 0.002070
grad_step = 000114, loss = 0.002067
grad_step = 000115, loss = 0.002068
grad_step = 000116, loss = 0.002067
grad_step = 000117, loss = 0.002062
grad_step = 000118, loss = 0.002055
grad_step = 000119, loss = 0.002045
grad_step = 000120, loss = 0.002034
grad_step = 000121, loss = 0.002024
grad_step = 000122, loss = 0.002014
grad_step = 000123, loss = 0.002006
grad_step = 000124, loss = 0.001999
grad_step = 000125, loss = 0.001991
grad_step = 000126, loss = 0.001984
grad_step = 000127, loss = 0.001978
grad_step = 000128, loss = 0.001971
grad_step = 000129, loss = 0.001964
grad_step = 000130, loss = 0.001959
grad_step = 000131, loss = 0.001959
grad_step = 000132, loss = 0.001979
grad_step = 000133, loss = 0.002074
grad_step = 000134, loss = 0.002353
grad_step = 000135, loss = 0.002608
grad_step = 000136, loss = 0.002334
grad_step = 000137, loss = 0.001923
grad_step = 000138, loss = 0.002256
grad_step = 000139, loss = 0.002299
grad_step = 000140, loss = 0.001918
grad_step = 000141, loss = 0.002160
grad_step = 000142, loss = 0.002135
grad_step = 000143, loss = 0.001902
grad_step = 000144, loss = 0.002129
grad_step = 000145, loss = 0.002006
grad_step = 000146, loss = 0.001920
grad_step = 000147, loss = 0.002077
grad_step = 000148, loss = 0.001912
grad_step = 000149, loss = 0.001943
grad_step = 000150, loss = 0.002000
grad_step = 000151, loss = 0.001870
grad_step = 000152, loss = 0.001956
grad_step = 000153, loss = 0.001926
grad_step = 000154, loss = 0.001863
grad_step = 000155, loss = 0.001937
grad_step = 000156, loss = 0.001872
grad_step = 000157, loss = 0.001867
grad_step = 000158, loss = 0.001906
grad_step = 000159, loss = 0.001843
grad_step = 000160, loss = 0.001862
grad_step = 000161, loss = 0.001872
grad_step = 000162, loss = 0.001826
grad_step = 000163, loss = 0.001849
grad_step = 000164, loss = 0.001845
grad_step = 000165, loss = 0.001814
grad_step = 000166, loss = 0.001833
grad_step = 000167, loss = 0.001825
grad_step = 000168, loss = 0.001801
grad_step = 000169, loss = 0.001814
grad_step = 000170, loss = 0.001809
grad_step = 000171, loss = 0.001789
grad_step = 000172, loss = 0.001795
grad_step = 000173, loss = 0.001795
grad_step = 000174, loss = 0.001778
grad_step = 000175, loss = 0.001777
grad_step = 000176, loss = 0.001781
grad_step = 000177, loss = 0.001770
grad_step = 000178, loss = 0.001760
grad_step = 000179, loss = 0.001763
grad_step = 000180, loss = 0.001761
grad_step = 000181, loss = 0.001750
grad_step = 000182, loss = 0.001745
grad_step = 000183, loss = 0.001746
grad_step = 000184, loss = 0.001743
grad_step = 000185, loss = 0.001735
grad_step = 000186, loss = 0.001729
grad_step = 000187, loss = 0.001727
grad_step = 000188, loss = 0.001726
grad_step = 000189, loss = 0.001723
grad_step = 000190, loss = 0.001716
grad_step = 000191, loss = 0.001710
grad_step = 000192, loss = 0.001707
grad_step = 000193, loss = 0.001705
grad_step = 000194, loss = 0.001703
grad_step = 000195, loss = 0.001701
grad_step = 000196, loss = 0.001699
grad_step = 000197, loss = 0.001696
grad_step = 000198, loss = 0.001693
grad_step = 000199, loss = 0.001691
grad_step = 000200, loss = 0.001690
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001692
grad_step = 000202, loss = 0.001699
grad_step = 000203, loss = 0.001718
grad_step = 000204, loss = 0.001753
grad_step = 000205, loss = 0.001824
grad_step = 000206, loss = 0.001951
grad_step = 000207, loss = 0.002168
grad_step = 000208, loss = 0.002159
grad_step = 000209, loss = 0.001838
grad_step = 000210, loss = 0.001677
grad_step = 000211, loss = 0.001910
grad_step = 000212, loss = 0.001982
grad_step = 000213, loss = 0.001718
grad_step = 000214, loss = 0.001728
grad_step = 000215, loss = 0.001921
grad_step = 000216, loss = 0.001791
grad_step = 000217, loss = 0.001662
grad_step = 000218, loss = 0.001807
grad_step = 000219, loss = 0.001812
grad_step = 000220, loss = 0.001670
grad_step = 000221, loss = 0.001693
grad_step = 000222, loss = 0.001777
grad_step = 000223, loss = 0.001679
grad_step = 000224, loss = 0.001648
grad_step = 000225, loss = 0.001705
grad_step = 000226, loss = 0.001691
grad_step = 000227, loss = 0.001622
grad_step = 000228, loss = 0.001658
grad_step = 000229, loss = 0.001670
grad_step = 000230, loss = 0.001633
grad_step = 000231, loss = 0.001616
grad_step = 000232, loss = 0.001650
grad_step = 000233, loss = 0.001632
grad_step = 000234, loss = 0.001608
grad_step = 000235, loss = 0.001615
grad_step = 000236, loss = 0.001629
grad_step = 000237, loss = 0.001607
grad_step = 000238, loss = 0.001597
grad_step = 000239, loss = 0.001606
grad_step = 000240, loss = 0.001609
grad_step = 000241, loss = 0.001594
grad_step = 000242, loss = 0.001587
grad_step = 000243, loss = 0.001594
grad_step = 000244, loss = 0.001593
grad_step = 000245, loss = 0.001584
grad_step = 000246, loss = 0.001576
grad_step = 000247, loss = 0.001580
grad_step = 000248, loss = 0.001580
grad_step = 000249, loss = 0.001575
grad_step = 000250, loss = 0.001568
grad_step = 000251, loss = 0.001566
grad_step = 000252, loss = 0.001568
grad_step = 000253, loss = 0.001566
grad_step = 000254, loss = 0.001561
grad_step = 000255, loss = 0.001556
grad_step = 000256, loss = 0.001555
grad_step = 000257, loss = 0.001556
grad_step = 000258, loss = 0.001554
grad_step = 000259, loss = 0.001550
grad_step = 000260, loss = 0.001545
grad_step = 000261, loss = 0.001542
grad_step = 000262, loss = 0.001541
grad_step = 000263, loss = 0.001540
grad_step = 000264, loss = 0.001538
grad_step = 000265, loss = 0.001535
grad_step = 000266, loss = 0.001531
grad_step = 000267, loss = 0.001528
grad_step = 000268, loss = 0.001526
grad_step = 000269, loss = 0.001524
grad_step = 000270, loss = 0.001522
grad_step = 000271, loss = 0.001521
grad_step = 000272, loss = 0.001519
grad_step = 000273, loss = 0.001517
grad_step = 000274, loss = 0.001514
grad_step = 000275, loss = 0.001512
grad_step = 000276, loss = 0.001511
grad_step = 000277, loss = 0.001510
grad_step = 000278, loss = 0.001510
grad_step = 000279, loss = 0.001513
grad_step = 000280, loss = 0.001522
grad_step = 000281, loss = 0.001532
grad_step = 000282, loss = 0.001549
grad_step = 000283, loss = 0.001565
grad_step = 000284, loss = 0.001590
grad_step = 000285, loss = 0.001610
grad_step = 000286, loss = 0.001621
grad_step = 000287, loss = 0.001603
grad_step = 000288, loss = 0.001567
grad_step = 000289, loss = 0.001519
grad_step = 000290, loss = 0.001487
grad_step = 000291, loss = 0.001476
grad_step = 000292, loss = 0.001483
grad_step = 000293, loss = 0.001495
grad_step = 000294, loss = 0.001508
grad_step = 000295, loss = 0.001527
grad_step = 000296, loss = 0.001533
grad_step = 000297, loss = 0.001528
grad_step = 000298, loss = 0.001489
grad_step = 000299, loss = 0.001463
grad_step = 000300, loss = 0.001460
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001468
grad_step = 000302, loss = 0.001467
grad_step = 000303, loss = 0.001462
grad_step = 000304, loss = 0.001465
grad_step = 000305, loss = 0.001468
grad_step = 000306, loss = 0.001466
grad_step = 000307, loss = 0.001455
grad_step = 000308, loss = 0.001442
grad_step = 000309, loss = 0.001434
grad_step = 000310, loss = 0.001435
grad_step = 000311, loss = 0.001440
grad_step = 000312, loss = 0.001446
grad_step = 000313, loss = 0.001447
grad_step = 000314, loss = 0.001447
grad_step = 000315, loss = 0.001442
grad_step = 000316, loss = 0.001447
grad_step = 000317, loss = 0.001454
grad_step = 000318, loss = 0.001474
grad_step = 000319, loss = 0.001494
grad_step = 000320, loss = 0.001516
grad_step = 000321, loss = 0.001525
grad_step = 000322, loss = 0.001521
grad_step = 000323, loss = 0.001501
grad_step = 000324, loss = 0.001490
grad_step = 000325, loss = 0.001477
grad_step = 000326, loss = 0.001476
grad_step = 000327, loss = 0.001434
grad_step = 000328, loss = 0.001402
grad_step = 000329, loss = 0.001403
grad_step = 000330, loss = 0.001422
grad_step = 000331, loss = 0.001427
grad_step = 000332, loss = 0.001410
grad_step = 000333, loss = 0.001403
grad_step = 000334, loss = 0.001404
grad_step = 000335, loss = 0.001404
grad_step = 000336, loss = 0.001397
grad_step = 000337, loss = 0.001387
grad_step = 000338, loss = 0.001380
grad_step = 000339, loss = 0.001379
grad_step = 000340, loss = 0.001383
grad_step = 000341, loss = 0.001389
grad_step = 000342, loss = 0.001395
grad_step = 000343, loss = 0.001405
grad_step = 000344, loss = 0.001424
grad_step = 000345, loss = 0.001452
grad_step = 000346, loss = 0.001498
grad_step = 000347, loss = 0.001492
grad_step = 000348, loss = 0.001475
grad_step = 000349, loss = 0.001448
grad_step = 000350, loss = 0.001448
grad_step = 000351, loss = 0.001443
grad_step = 000352, loss = 0.001393
grad_step = 000353, loss = 0.001356
grad_step = 000354, loss = 0.001363
grad_step = 000355, loss = 0.001386
grad_step = 000356, loss = 0.001389
grad_step = 000357, loss = 0.001373
grad_step = 000358, loss = 0.001363
grad_step = 000359, loss = 0.001370
grad_step = 000360, loss = 0.001388
grad_step = 000361, loss = 0.001393
grad_step = 000362, loss = 0.001371
grad_step = 000363, loss = 0.001349
grad_step = 000364, loss = 0.001345
grad_step = 000365, loss = 0.001352
grad_step = 000366, loss = 0.001349
grad_step = 000367, loss = 0.001334
grad_step = 000368, loss = 0.001326
grad_step = 000369, loss = 0.001332
grad_step = 000370, loss = 0.001334
grad_step = 000371, loss = 0.001325
grad_step = 000372, loss = 0.001317
grad_step = 000373, loss = 0.001319
grad_step = 000374, loss = 0.001325
grad_step = 000375, loss = 0.001330
grad_step = 000376, loss = 0.001340
grad_step = 000377, loss = 0.001348
grad_step = 000378, loss = 0.001362
grad_step = 000379, loss = 0.001386
grad_step = 000380, loss = 0.001427
grad_step = 000381, loss = 0.001473
grad_step = 000382, loss = 0.001505
grad_step = 000383, loss = 0.001505
grad_step = 000384, loss = 0.001469
grad_step = 000385, loss = 0.001404
grad_step = 000386, loss = 0.001352
grad_step = 000387, loss = 0.001311
grad_step = 000388, loss = 0.001301
grad_step = 000389, loss = 0.001316
grad_step = 000390, loss = 0.001340
grad_step = 000391, loss = 0.001361
grad_step = 000392, loss = 0.001345
grad_step = 000393, loss = 0.001315
grad_step = 000394, loss = 0.001284
grad_step = 000395, loss = 0.001279
grad_step = 000396, loss = 0.001292
grad_step = 000397, loss = 0.001299
grad_step = 000398, loss = 0.001300
grad_step = 000399, loss = 0.001298
grad_step = 000400, loss = 0.001294
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001286
grad_step = 000402, loss = 0.001273
grad_step = 000403, loss = 0.001262
grad_step = 000404, loss = 0.001259
grad_step = 000405, loss = 0.001263
grad_step = 000406, loss = 0.001268
grad_step = 000407, loss = 0.001270
grad_step = 000408, loss = 0.001267
grad_step = 000409, loss = 0.001262
grad_step = 000410, loss = 0.001256
grad_step = 000411, loss = 0.001253
grad_step = 000412, loss = 0.001251
grad_step = 000413, loss = 0.001251
grad_step = 000414, loss = 0.001249
grad_step = 000415, loss = 0.001245
grad_step = 000416, loss = 0.001240
grad_step = 000417, loss = 0.001236
grad_step = 000418, loss = 0.001234
grad_step = 000419, loss = 0.001233
grad_step = 000420, loss = 0.001233
grad_step = 000421, loss = 0.001233
grad_step = 000422, loss = 0.001232
grad_step = 000423, loss = 0.001230
grad_step = 000424, loss = 0.001229
grad_step = 000425, loss = 0.001228
grad_step = 000426, loss = 0.001229
grad_step = 000427, loss = 0.001231
grad_step = 000428, loss = 0.001236
grad_step = 000429, loss = 0.001246
grad_step = 000430, loss = 0.001267
grad_step = 000431, loss = 0.001302
grad_step = 000432, loss = 0.001370
grad_step = 000433, loss = 0.001462
grad_step = 000434, loss = 0.001594
grad_step = 000435, loss = 0.001672
grad_step = 000436, loss = 0.001647
grad_step = 000437, loss = 0.001455
grad_step = 000438, loss = 0.001266
grad_step = 000439, loss = 0.001221
grad_step = 000440, loss = 0.001317
grad_step = 000441, loss = 0.001391
grad_step = 000442, loss = 0.001342
grad_step = 000443, loss = 0.001258
grad_step = 000444, loss = 0.001233
grad_step = 000445, loss = 0.001267
grad_step = 000446, loss = 0.001277
grad_step = 000447, loss = 0.001251
grad_step = 000448, loss = 0.001231
grad_step = 000449, loss = 0.001224
grad_step = 000450, loss = 0.001220
grad_step = 000451, loss = 0.001217
grad_step = 000452, loss = 0.001220
grad_step = 000453, loss = 0.001219
grad_step = 000454, loss = 0.001200
grad_step = 000455, loss = 0.001185
grad_step = 000456, loss = 0.001192
grad_step = 000457, loss = 0.001205
grad_step = 000458, loss = 0.001201
grad_step = 000459, loss = 0.001181
grad_step = 000460, loss = 0.001169
grad_step = 000461, loss = 0.001177
grad_step = 000462, loss = 0.001190
grad_step = 000463, loss = 0.001188
grad_step = 000464, loss = 0.001173
grad_step = 000465, loss = 0.001159
grad_step = 000466, loss = 0.001160
grad_step = 000467, loss = 0.001169
grad_step = 000468, loss = 0.001172
grad_step = 000469, loss = 0.001167
grad_step = 000470, loss = 0.001158
grad_step = 000471, loss = 0.001150
grad_step = 000472, loss = 0.001148
grad_step = 000473, loss = 0.001149
grad_step = 000474, loss = 0.001151
grad_step = 000475, loss = 0.001153
grad_step = 000476, loss = 0.001151
grad_step = 000477, loss = 0.001148
grad_step = 000478, loss = 0.001143
grad_step = 000479, loss = 0.001138
grad_step = 000480, loss = 0.001134
grad_step = 000481, loss = 0.001132
grad_step = 000482, loss = 0.001131
grad_step = 000483, loss = 0.001132
grad_step = 000484, loss = 0.001133
grad_step = 000485, loss = 0.001134
grad_step = 000486, loss = 0.001135
grad_step = 000487, loss = 0.001135
grad_step = 000488, loss = 0.001137
grad_step = 000489, loss = 0.001136
grad_step = 000490, loss = 0.001135
grad_step = 000491, loss = 0.001129
grad_step = 000492, loss = 0.001122
grad_step = 000493, loss = 0.001116
grad_step = 000494, loss = 0.001112
grad_step = 000495, loss = 0.001110
grad_step = 000496, loss = 0.001109
grad_step = 000497, loss = 0.001108
grad_step = 000498, loss = 0.001108
grad_step = 000499, loss = 0.001109
grad_step = 000500, loss = 0.001111
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001117
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

  date_run                              2020-05-13 00:18:46.543177
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.254173
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 00:18:46.548968
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.184354
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 00:18:46.555593
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.136687
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 00:18:46.560719
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.80132
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
0   2020-05-13 00:18:20.526987  ...    mean_absolute_error
1   2020-05-13 00:18:20.530485  ...     mean_squared_error
2   2020-05-13 00:18:20.533740  ...  median_absolute_error
3   2020-05-13 00:18:20.536831  ...               r2_score
4   2020-05-13 00:18:28.938855  ...    mean_absolute_error
5   2020-05-13 00:18:28.942785  ...     mean_squared_error
6   2020-05-13 00:18:28.945665  ...  median_absolute_error
7   2020-05-13 00:18:28.948359  ...               r2_score
8   2020-05-13 00:18:46.543177  ...    mean_absolute_error
9   2020-05-13 00:18:46.548968  ...     mean_squared_error
10  2020-05-13 00:18:46.555593  ...  median_absolute_error
11  2020-05-13 00:18:46.560719  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbeb0ff7fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███       | 3055616/9912422 [00:00<00:00, 30522568.69it/s]9920512it [00:00, 33552348.61it/s]                             
0it [00:00, ?it/s]32768it [00:00, 562472.19it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162471.99it/s]1654784it [00:00, 10923836.62it/s]                         
0it [00:00, ?it/s]8192it [00:00, 189803.45it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbe639f8eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbe630260f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbe639f8eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbe62f7f128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbe607ba518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbe607a6c88> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbe639f8eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbe62f3c748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbe607ba518> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fbe62df7550> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f0e44641240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=2ccfb4eabf0ffeb6ee72b0dba68ae2267ceb2df9bd5ee8acd6fd726ca3892595
  Stored in directory: /tmp/pip-ephem-wheel-cache-uiatntrb/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f0e3a7ac080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3784704/17464789 [=====>........................] - ETA: 0s
12533760/17464789 [====================>.........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 00:20:11.565292: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 00:20:11.569624: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-13 00:20:11.569786: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55642ce7d0e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 00:20:11.569801: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6513 - accuracy: 0.5010 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7842 - accuracy: 0.4923
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8123 - accuracy: 0.4905
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7525 - accuracy: 0.4944
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6922 - accuracy: 0.4983
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6754 - accuracy: 0.4994
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6858 - accuracy: 0.4988
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6768 - accuracy: 0.4993
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6099 - accuracy: 0.5037
11000/25000 [============>.................] - ETA: 3s - loss: 7.6206 - accuracy: 0.5030
12000/25000 [=============>................] - ETA: 3s - loss: 7.6193 - accuracy: 0.5031
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6124 - accuracy: 0.5035
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6075 - accuracy: 0.5039
15000/25000 [=================>............] - ETA: 2s - loss: 7.6022 - accuracy: 0.5042
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6264 - accuracy: 0.5026
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6432 - accuracy: 0.5015
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6573 - accuracy: 0.5006
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6642 - accuracy: 0.5002
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6582 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6644 - accuracy: 0.5001
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6708 - accuracy: 0.4997
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6640 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 7s 270us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 00:20:24.610744
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 00:20:24.610744  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:24:28, 10.7kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:55:09, 15.0kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:11:49, 21.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:50:45, 30.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:28:41, 43.5kB/s].vector_cache/glove.6B.zip:   1%|          | 8.92M/862M [00:01<3:48:45, 62.2kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:01<2:39:38, 88.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.9M/862M [00:01<1:51:13, 127kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.7M/862M [00:01<1:17:37, 181kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.2M/862M [00:01<54:07, 258kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.2M/862M [00:01<37:49, 367kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.4M/862M [00:02<26:26, 522kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.4M/862M [00:02<18:29, 743kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.9M/862M [00:02<13:00, 1.05MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:02<09:07, 1.49MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.6M/862M [00:02<06:28, 2.09MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:02<05:36, 2.41MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<05:49, 2.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<05:57, 2.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:05<04:37, 2.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<05:46, 2.32MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:06<07:10, 1.86MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.2M/862M [00:07<05:40, 2.35MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.3M/862M [00:07<04:08, 3.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<08:38, 1.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<07:25, 1.79MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.4M/862M [00:09<05:32, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<06:56, 1.91MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<06:12, 2.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:10<04:40, 2.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<06:21, 2.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<07:07, 1.85MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:12<05:39, 2.32MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:13<04:04, 3.21MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<1:12:16, 181kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<51:54, 252kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:14<36:36, 357kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<28:35, 455kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<22:40, 574kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.9M/862M [00:16<16:26, 791kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:16<11:37, 1.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:18<20:52, 621kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<15:57, 811kB/s].vector_cache/glove.6B.zip:  10%|█         | 87.0M/862M [00:18<11:28, 1.13MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<11:02, 1.17MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:20<10:28, 1.23MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:20<07:59, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:20<05:43, 2.24MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<1:26:05, 149kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<1:01:34, 208kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<43:21, 295kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:24<33:13, 384kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<24:32, 519kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.3M/862M [00:24<17:27, 728kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<15:12, 834kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<11:54, 1.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<08:38, 1.46MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<09:01, 1.40MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:52, 1.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<06:45, 1.86MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<04:51, 2.58MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<13:19, 941kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<10:36, 1.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<07:43, 1.62MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<08:19, 1.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<08:22, 1.49MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:29, 1.92MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:32, 1.90MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:51, 2.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:22, 2.83MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<05:56, 2.08MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:41, 1.84MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:17, 2.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<03:49, 3.20MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<11:51:02, 17.3kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<8:18:45, 24.6kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<5:48:42, 35.1kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<4:06:14, 49.5kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<2:53:32, 70.3kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<2:01:31, 100kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<1:27:41, 138kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<1:03:50, 190kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<45:11, 268kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<31:38, 381kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<36:22, 332kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<26:41, 452kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<18:57, 635kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<16:01, 748kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<13:46, 871kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<10:15, 1.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<07:18, 1.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<1:30:12, 132kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<1:04:21, 185kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<45:12, 263kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<34:20, 345kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<26:27, 448kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<19:02, 622kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<13:23, 880kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<27:12, 433kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<20:14, 582kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<14:24, 816kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<12:48, 915kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<10:09, 1.15MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<07:24, 1.58MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<07:53, 1.47MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<06:41, 1.74MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:57, 2.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:12, 1.87MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:43, 1.72MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:17, 2.19MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<05:33, 2.07MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:04, 2.27MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<03:48, 3.01MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:20, 2.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:04, 1.88MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<04:49, 2.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:12, 2.18MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<04:48, 2.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:03<03:38, 3.11MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:10, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<05:55, 1.91MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<04:37, 2.44MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:05<03:22, 3.33MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:18, 1.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:59, 1.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<05:10, 2.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:13, 1.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:43, 1.66MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:11, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<03:46, 2.94MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<07:59, 1.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<06:43, 1.65MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:58, 2.22MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:03, 1.82MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:30, 1.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:07, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<03:42, 2.96MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<42:08, 260kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:15<30:37, 358kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<21:37, 505kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<17:38, 618kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<14:33, 748kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<10:44, 1.01MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<09:14, 1.17MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:34, 1.43MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:33, 1.94MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:23, 1.68MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:40, 1.61MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:13, 2.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:22, 1.99MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:52, 2.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:38, 2.92MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:01, 2.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:40, 1.87MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<04:26, 2.38MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<03:14, 3.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<07:46, 1.36MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:31, 1.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:49, 2.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:49, 1.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<06:14, 1.68MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:49, 2.17MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<03:31, 2.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<07:18, 1.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:13, 1.67MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:37, 2.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<05:37, 1.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<04:59, 2.07MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:44, 2.75MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<05:01, 2.05MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:35, 2.24MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:25, 2.99MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:47, 2.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:25, 2.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:21, 3.04MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:44, 2.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:21, 2.32MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<03:18, 3.06MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:41, 2.15MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<04:19, 2.33MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<03:16, 3.06MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:39, 2.15MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<05:17, 1.89MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<04:13, 2.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<04:33, 2.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<04:12, 2.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<03:09, 3.14MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:45<02:20, 4.22MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<40:44, 242kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<30:32, 323kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<21:51, 451kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<16:48, 583kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<12:47, 765kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<09:11, 1.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<08:39, 1.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<08:10, 1.19MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<06:08, 1.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:26, 2.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<06:31, 1.48MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<05:34, 1.73MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<04:08, 2.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:08, 1.87MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:36, 2.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<03:26, 2.78MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:37, 2.06MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:12, 2.26MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:10, 2.98MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:27, 2.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:05, 2.31MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:05, 3.05MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:22, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:01, 2.33MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:00, 3.11MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:19, 2.15MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<03:58, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:00, 3.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:17, 2.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:56, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<02:58, 3.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:13, 2.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:54, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<02:57, 3.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:11, 2.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:52, 2.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<02:55, 3.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:11, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:46, 1.89MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:48, 2.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:06, 2.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:46, 2.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<02:52, 3.11MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:05, 2.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:41, 1.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:40, 2.42MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<02:39, 3.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<11:10, 791kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<08:44, 1.01MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<06:19, 1.39MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<06:27, 1.36MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:26, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:00, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<04:51, 1.79MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<05:10, 1.68MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:03, 2.14MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<04:13, 2.04MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<03:51, 2.24MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<02:54, 2.95MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<04:02, 2.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:34, 1.87MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<03:34, 2.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:24<02:35, 3.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<08:05, 1.05MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<06:31, 1.30MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:46, 1.77MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:18, 1.59MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<05:25, 1.55MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<04:14, 1.99MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:28<03:03, 2.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<1:01:47, 135kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<44:05, 189kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<30:59, 269kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<23:31, 352kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<18:09, 456kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<13:03, 633kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<09:11, 895kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<12:30, 658kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<09:34, 857kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<06:52, 1.19MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<06:42, 1.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<06:21, 1.28MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:50, 1.68MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:41, 1.73MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:07, 1.96MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:05, 2.61MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:01, 1.99MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:28, 1.79MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:31, 2.27MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:44, 2.12MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:25, 2.32MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<02:35, 3.05MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:39, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:21, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:33, 3.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:36, 2.16MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:03, 1.92MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:10, 2.45MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<02:19, 3.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<05:17, 1.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:30, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:47<03:20, 2.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:07, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:40, 2.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:45, 2.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:43, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:10, 1.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:14, 2.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<02:21, 3.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<06:56, 1.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:37, 1.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<04:07, 1.82MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<04:36, 1.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:00, 1.86MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:59, 2.49MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:49, 1.94MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:26, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:35, 2.84MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:31, 2.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:58, 1.85MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:09, 2.32MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<03:22, 2.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:05, 2.35MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:20, 3.08MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:19, 2.16MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:02, 2.36MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:18, 3.11MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:17, 2.16MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:45, 1.89MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<02:56, 2.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:08, 3.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:44, 1.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:43, 1.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<03:29, 2.02MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:03, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:33, 1.96MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:39, 2.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:29, 1.98MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:50, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:59, 2.31MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<02:12, 3.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:53, 1.76MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:26, 1.99MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:34, 2.64MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:23, 2.00MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:45, 1.81MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<02:58, 2.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:15<02:08, 3.14MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<49:54, 135kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<35:34, 188kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<24:58, 267kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<18:57, 351kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<13:56, 476kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<09:53, 668kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<08:25, 780kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<06:27, 1.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<04:44, 1.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<03:22, 1.93MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<50:45, 128kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<36:49, 177kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<26:03, 249kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<19:08, 336kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<14:02, 458kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<09:57, 643kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<08:25, 756kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<06:31, 975kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:42, 1.34MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:46, 1.32MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<04:37, 1.36MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:30, 1.79MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<02:31, 2.47MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<05:38, 1.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<04:34, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:21, 1.85MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:46, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:54, 1.58MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:59, 2.05MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:33<02:09, 2.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<08:50, 689kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<06:48, 895kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<04:54, 1.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<04:49, 1.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:59, 1.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<02:56, 2.04MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:27, 1.73MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:01, 1.96MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<02:15, 2.62MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:58, 1.98MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:16, 1.79MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:33, 2.29MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:40<01:50, 3.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<08:54, 654kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<06:49, 851kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<04:53, 1.18MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<04:45, 1.21MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<04:29, 1.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:24, 1.68MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:44<02:26, 2.34MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<06:28, 877kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<05:06, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<03:42, 1.52MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:53, 1.44MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<03:52, 1.45MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:56, 1.90MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:48<02:08, 2.61MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<03:43, 1.49MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:10, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:21, 2.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<02:54, 1.88MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<03:09, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:29, 2.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:36, 2.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<02:23, 2.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<01:48, 2.98MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:30, 2.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<02:50, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:15, 2.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:56<01:37, 3.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<20:12, 261kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<14:39, 359kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<10:20, 506kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<08:24, 618kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<06:56, 749kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<05:04, 1.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<03:35, 1.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<05:46, 890kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<04:33, 1.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<03:18, 1.54MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:29, 1.45MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:57, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:11, 2.30MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<02:42, 1.85MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:23, 2.08MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:47, 2.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:25, 2.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:10, 2.26MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:38, 2.98MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:17, 2.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<02:06, 2.31MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<01:35, 3.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<02:14, 2.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:32, 1.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:00, 2.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:09, 2.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<01:59, 2.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:30, 3.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:08, 2.18MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<01:58, 2.36MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:28, 3.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:07, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<01:56, 2.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:28, 3.09MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<02:05, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:54, 2.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<01:27, 3.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:03, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:21, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<01:50, 2.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:20, 3.28MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<03:01, 1.45MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:33, 1.71MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<01:53, 2.30MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:19, 1.85MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:30, 1.72MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:56, 2.22MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:24, 3.02MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<02:44, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:21, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:44, 2.41MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:11, 1.91MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:23, 1.74MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:53, 2.20MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:58, 2.09MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:48, 2.26MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:21, 2.99MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:53, 2.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:44, 2.31MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:18, 3.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:51, 2.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:06, 1.88MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<01:40, 2.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:47, 2.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:39, 2.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:15, 3.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:46, 2.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:02, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:34, 2.41MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<01:10, 3.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:55, 1.95MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:44, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:18, 2.85MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<01:45, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:36, 2.28MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:43<01:12, 3.03MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:42, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:45, 2.05MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:21, 2.67MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:35, 2.23MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:28, 2.40MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:07, 3.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:36, 2.18MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:28, 2.36MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:06, 3.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:34, 2.17MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:27, 2.35MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:05, 3.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:33, 2.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:25, 2.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:04, 3.09MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<01:31, 2.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:24, 2.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:03, 3.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:29, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:42, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:21, 2.37MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<00:57, 3.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<07:00, 449kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<05:13, 602kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<03:41, 843kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<03:16, 940kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:54, 1.05MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:11, 1.40MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:01<01:32, 1.95MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<22:48, 132kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<16:14, 185kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<11:20, 263kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<08:31, 345kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<06:15, 469kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<04:25, 658kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<03:43, 771kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<03:11, 899kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<02:22, 1.21MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<02:05, 1.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:44, 1.60MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:17, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:31, 1.80MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:37, 1.68MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:16, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:18, 2.05MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<01:10, 2.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<00:53, 2.97MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:13, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:07, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<00:50, 3.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:10, 2.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:02, 2.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<00:47, 3.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:07, 2.18MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:17, 1.91MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:01, 2.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:43, 3.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<09:13, 259kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<06:40, 357kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<04:41, 503kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<03:46, 615kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<03:06, 745kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<02:16, 1.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<01:36, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:48, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:29, 1.50MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<01:05, 2.05MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:15, 1.73MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:19, 1.64MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:01, 2.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:44, 2.91MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:38, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:22, 1.54MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<00:59, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:09, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:01, 2.00MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:45, 2.69MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:59, 2.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:05, 1.81MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:51, 2.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:36, 3.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<14:22, 133kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<10:13, 187kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<07:05, 265kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<05:18, 348kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<04:04, 451kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<02:55, 627kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<02:02, 882kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<02:03, 862kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:37, 1.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:09, 1.50MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:11, 1.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:11, 1.44MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:54, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:53, 1.85MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:46, 2.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:34, 2.79MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:45, 2.06MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:41, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:30, 2.99MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:42, 2.12MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:48, 1.87MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:37, 2.39MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:26, 3.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:52, 1.63MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:45, 1.88MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:33, 2.52MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:41, 1.95MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:37, 2.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:27, 2.87MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:37, 2.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:33, 2.30MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:25, 3.03MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:34, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:39, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:30, 2.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:21, 3.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<01:29, 779kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:09, 999kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:48, 1.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:48, 1.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:47, 1.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:35, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:33, 1.81MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:29, 2.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:21, 2.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:28, 2.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:25, 2.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:18, 2.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:25, 2.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:28, 1.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:22, 2.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:15, 3.17MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<06:00, 136kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<04:15, 190kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<02:53, 270kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<02:06, 354kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<01:37, 458kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:09, 633kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:51, 789kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:39, 1.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:09<00:27, 1.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:27, 1.35MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:26, 1.39MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:19, 1.80MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:13, 2.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:40, 808kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:31, 1.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<00:21, 1.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:20, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:17, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:11, 2.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:13, 1.80MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:08, 2.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:11, 1.81MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:08, 2.32MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:19<00:05, 3.19MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:17, 899kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:13, 1.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:09, 1.55MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:08, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.72MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:04, 2.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 2.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.77MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:01, 2.04MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.24MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.96MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 922/400000 [00:00<00:43, 9211.77it/s]  0%|          | 1860/400000 [00:00<00:42, 9260.61it/s]  1%|          | 2745/400000 [00:00<00:43, 9131.70it/s]  1%|          | 3691/400000 [00:00<00:42, 9225.25it/s]  1%|          | 4637/400000 [00:00<00:42, 9294.07it/s]  1%|▏         | 5525/400000 [00:00<00:43, 9162.63it/s]  2%|▏         | 6469/400000 [00:00<00:42, 9244.00it/s]  2%|▏         | 7404/400000 [00:00<00:42, 9273.72it/s]  2%|▏         | 8335/400000 [00:00<00:42, 9283.97it/s]  2%|▏         | 9267/400000 [00:01<00:42, 9292.39it/s]  3%|▎         | 10182/400000 [00:01<00:42, 9246.35it/s]  3%|▎         | 11089/400000 [00:01<00:42, 9177.08it/s]  3%|▎         | 12010/400000 [00:01<00:42, 9184.35it/s]  3%|▎         | 12920/400000 [00:01<00:42, 9084.93it/s]  3%|▎         | 13866/400000 [00:01<00:42, 9193.58it/s]  4%|▎         | 14782/400000 [00:01<00:42, 9126.06it/s]  4%|▍         | 15704/400000 [00:01<00:41, 9152.53it/s]  4%|▍         | 16667/400000 [00:01<00:41, 9290.09it/s]  4%|▍         | 17596/400000 [00:01<00:41, 9254.53it/s]  5%|▍         | 18569/400000 [00:02<00:40, 9390.03it/s]  5%|▍         | 19509/400000 [00:02<00:41, 9239.78it/s]  5%|▌         | 20455/400000 [00:02<00:40, 9304.22it/s]  5%|▌         | 21418/400000 [00:02<00:40, 9396.52it/s]  6%|▌         | 22376/400000 [00:02<00:39, 9450.09it/s]  6%|▌         | 23335/400000 [00:02<00:39, 9489.80it/s]  6%|▌         | 24285/400000 [00:02<00:39, 9394.22it/s]  6%|▋         | 25258/400000 [00:02<00:39, 9490.72it/s]  7%|▋         | 26208/400000 [00:02<00:39, 9464.90it/s]  7%|▋         | 27155/400000 [00:02<00:39, 9391.58it/s]  7%|▋         | 28134/400000 [00:03<00:39, 9505.85it/s]  7%|▋         | 29086/400000 [00:03<00:39, 9456.78it/s]  8%|▊         | 30033/400000 [00:03<00:39, 9431.07it/s]  8%|▊         | 30977/400000 [00:03<00:39, 9356.00it/s]  8%|▊         | 31913/400000 [00:03<00:40, 9117.73it/s]  8%|▊         | 32827/400000 [00:03<00:41, 8933.88it/s]  8%|▊         | 33723/400000 [00:03<00:41, 8879.98it/s]  9%|▊         | 34638/400000 [00:03<00:40, 8956.33it/s]  9%|▉         | 35535/400000 [00:03<00:40, 8910.76it/s]  9%|▉         | 36427/400000 [00:03<00:41, 8791.11it/s]  9%|▉         | 37319/400000 [00:04<00:41, 8827.86it/s] 10%|▉         | 38212/400000 [00:04<00:40, 8857.21it/s] 10%|▉         | 39108/400000 [00:04<00:40, 8887.48it/s] 10%|█         | 40006/400000 [00:04<00:40, 8914.74it/s] 10%|█         | 40898/400000 [00:04<00:40, 8812.14it/s] 10%|█         | 41780/400000 [00:04<00:40, 8790.25it/s] 11%|█         | 42675/400000 [00:04<00:40, 8837.33it/s] 11%|█         | 43660/400000 [00:04<00:39, 9118.49it/s] 11%|█         | 44683/400000 [00:04<00:37, 9425.28it/s] 11%|█▏        | 45630/400000 [00:04<00:38, 9142.44it/s] 12%|█▏        | 46614/400000 [00:05<00:37, 9338.61it/s] 12%|█▏        | 47553/400000 [00:05<00:38, 9194.77it/s] 12%|█▏        | 48477/400000 [00:05<00:38, 9188.92it/s] 12%|█▏        | 49431/400000 [00:05<00:37, 9290.46it/s] 13%|█▎        | 50460/400000 [00:05<00:36, 9568.97it/s] 13%|█▎        | 51421/400000 [00:05<00:36, 9571.29it/s] 13%|█▎        | 52381/400000 [00:05<00:36, 9570.42it/s] 13%|█▎        | 53348/400000 [00:05<00:36, 9599.53it/s] 14%|█▎        | 54310/400000 [00:05<00:36, 9440.31it/s] 14%|█▍        | 55256/400000 [00:05<00:37, 9271.61it/s] 14%|█▍        | 56185/400000 [00:06<00:37, 9080.90it/s] 14%|█▍        | 57096/400000 [00:06<00:38, 8942.90it/s] 14%|█▍        | 57993/400000 [00:06<00:38, 8904.05it/s] 15%|█▍        | 59006/400000 [00:06<00:36, 9237.23it/s] 15%|█▍        | 59944/400000 [00:06<00:36, 9277.64it/s] 15%|█▌        | 60875/400000 [00:06<00:36, 9286.99it/s] 15%|█▌        | 61825/400000 [00:06<00:36, 9348.06it/s] 16%|█▌        | 62786/400000 [00:06<00:35, 9424.80it/s] 16%|█▌        | 63753/400000 [00:06<00:35, 9494.12it/s] 16%|█▌        | 64712/400000 [00:07<00:35, 9521.07it/s] 16%|█▋        | 65701/400000 [00:07<00:34, 9626.40it/s] 17%|█▋        | 66671/400000 [00:07<00:34, 9646.16it/s] 17%|█▋        | 67670/400000 [00:07<00:34, 9745.67it/s] 17%|█▋        | 68654/400000 [00:07<00:33, 9764.12it/s] 17%|█▋        | 69631/400000 [00:07<00:33, 9730.51it/s] 18%|█▊        | 70605/400000 [00:07<00:33, 9722.09it/s] 18%|█▊        | 71581/400000 [00:07<00:33, 9731.78it/s] 18%|█▊        | 72556/400000 [00:07<00:33, 9736.16it/s] 18%|█▊        | 73530/400000 [00:07<00:33, 9696.82it/s] 19%|█▊        | 74500/400000 [00:08<00:33, 9674.29it/s] 19%|█▉        | 75468/400000 [00:08<00:33, 9588.54it/s] 19%|█▉        | 76428/400000 [00:08<00:34, 9465.19it/s] 19%|█▉        | 77376/400000 [00:08<00:34, 9451.66it/s] 20%|█▉        | 78322/400000 [00:08<00:34, 9418.89it/s] 20%|█▉        | 79293/400000 [00:08<00:33, 9504.25it/s] 20%|██        | 80244/400000 [00:08<00:33, 9477.28it/s] 20%|██        | 81193/400000 [00:08<00:34, 9343.49it/s] 21%|██        | 82183/400000 [00:08<00:33, 9501.24it/s] 21%|██        | 83135/400000 [00:08<00:33, 9452.35it/s] 21%|██        | 84122/400000 [00:09<00:32, 9573.80it/s] 21%|██▏       | 85130/400000 [00:09<00:32, 9717.57it/s] 22%|██▏       | 86130/400000 [00:09<00:32, 9800.26it/s] 22%|██▏       | 87112/400000 [00:09<00:32, 9766.82it/s] 22%|██▏       | 88090/400000 [00:09<00:32, 9742.66it/s] 22%|██▏       | 89065/400000 [00:09<00:32, 9692.39it/s] 23%|██▎       | 90035/400000 [00:09<00:33, 9211.72it/s] 23%|██▎       | 91010/400000 [00:09<00:32, 9365.47it/s] 23%|██▎       | 91991/400000 [00:09<00:32, 9493.62it/s] 23%|██▎       | 92964/400000 [00:09<00:32, 9562.51it/s] 23%|██▎       | 93923/400000 [00:10<00:32, 9451.28it/s] 24%|██▎       | 94871/400000 [00:10<00:32, 9298.18it/s] 24%|██▍       | 95803/400000 [00:10<00:33, 9200.58it/s] 24%|██▍       | 96738/400000 [00:10<00:32, 9243.88it/s] 24%|██▍       | 97685/400000 [00:10<00:32, 9307.72it/s] 25%|██▍       | 98617/400000 [00:10<00:32, 9295.68it/s] 25%|██▍       | 99548/400000 [00:10<00:32, 9207.44it/s] 25%|██▌       | 100470/400000 [00:10<00:33, 9045.97it/s] 25%|██▌       | 101376/400000 [00:10<00:33, 8836.35it/s] 26%|██▌       | 102264/400000 [00:10<00:33, 8848.88it/s] 26%|██▌       | 103151/400000 [00:11<00:34, 8654.57it/s] 26%|██▌       | 104019/400000 [00:11<00:34, 8487.82it/s] 26%|██▌       | 104886/400000 [00:11<00:34, 8540.72it/s] 26%|██▋       | 105798/400000 [00:11<00:33, 8705.18it/s] 27%|██▋       | 106702/400000 [00:11<00:33, 8798.58it/s] 27%|██▋       | 107639/400000 [00:11<00:32, 8962.12it/s] 27%|██▋       | 108564/400000 [00:11<00:32, 9044.46it/s] 27%|██▋       | 109470/400000 [00:11<00:32, 8966.73it/s] 28%|██▊       | 110395/400000 [00:11<00:32, 9049.19it/s] 28%|██▊       | 111346/400000 [00:11<00:31, 9182.06it/s] 28%|██▊       | 112266/400000 [00:12<00:31, 9181.77it/s] 28%|██▊       | 113211/400000 [00:12<00:30, 9260.58it/s] 29%|██▊       | 114168/400000 [00:12<00:30, 9350.71it/s] 29%|██▉       | 115119/400000 [00:12<00:30, 9396.19it/s] 29%|██▉       | 116092/400000 [00:12<00:29, 9491.05it/s] 29%|██▉       | 117042/400000 [00:12<00:30, 9423.57it/s] 29%|██▉       | 117985/400000 [00:12<00:30, 9363.33it/s] 30%|██▉       | 118922/400000 [00:12<00:30, 9341.40it/s] 30%|██▉       | 119857/400000 [00:12<00:30, 9225.92it/s] 30%|███       | 120781/400000 [00:13<00:30, 9125.33it/s] 30%|███       | 121706/400000 [00:13<00:30, 9162.08it/s] 31%|███       | 122650/400000 [00:13<00:30, 9242.75it/s] 31%|███       | 123575/400000 [00:13<00:30, 9159.89it/s] 31%|███       | 124530/400000 [00:13<00:29, 9271.91it/s] 31%|███▏      | 125458/400000 [00:13<00:29, 9236.20it/s] 32%|███▏      | 126405/400000 [00:13<00:29, 9303.70it/s] 32%|███▏      | 127344/400000 [00:13<00:29, 9326.41it/s] 32%|███▏      | 128329/400000 [00:13<00:28, 9475.06it/s] 32%|███▏      | 129278/400000 [00:13<00:28, 9424.45it/s] 33%|███▎      | 130225/400000 [00:14<00:28, 9435.04it/s] 33%|███▎      | 131169/400000 [00:14<00:28, 9356.10it/s] 33%|███▎      | 132106/400000 [00:14<00:28, 9359.88it/s] 33%|███▎      | 133043/400000 [00:14<00:29, 9102.88it/s] 33%|███▎      | 133956/400000 [00:14<00:29, 9062.21it/s] 34%|███▎      | 134882/400000 [00:14<00:29, 9118.89it/s] 34%|███▍      | 135816/400000 [00:14<00:28, 9181.57it/s] 34%|███▍      | 136735/400000 [00:14<00:28, 9136.58it/s] 34%|███▍      | 137660/400000 [00:14<00:28, 9169.18it/s] 35%|███▍      | 138616/400000 [00:14<00:28, 9282.86it/s] 35%|███▍      | 139608/400000 [00:15<00:27, 9463.86it/s] 35%|███▌      | 140558/400000 [00:15<00:27, 9473.20it/s] 35%|███▌      | 141507/400000 [00:15<00:27, 9376.11it/s] 36%|███▌      | 142468/400000 [00:15<00:27, 9443.06it/s] 36%|███▌      | 143414/400000 [00:15<00:27, 9258.42it/s] 36%|███▌      | 144383/400000 [00:15<00:27, 9382.61it/s] 36%|███▋      | 145323/400000 [00:15<00:27, 9293.87it/s] 37%|███▋      | 146254/400000 [00:15<00:27, 9157.69it/s] 37%|███▋      | 147172/400000 [00:15<00:28, 8942.91it/s] 37%|███▋      | 148069/400000 [00:15<00:28, 8812.29it/s] 37%|███▋      | 149016/400000 [00:16<00:27, 8997.90it/s] 37%|███▋      | 149997/400000 [00:16<00:27, 9225.91it/s] 38%|███▊      | 150923/400000 [00:16<00:27, 9195.22it/s] 38%|███▊      | 151852/400000 [00:16<00:26, 9220.01it/s] 38%|███▊      | 152810/400000 [00:16<00:26, 9322.68it/s] 38%|███▊      | 153793/400000 [00:16<00:26, 9468.86it/s] 39%|███▊      | 154763/400000 [00:16<00:25, 9535.03it/s] 39%|███▉      | 155725/400000 [00:16<00:25, 9558.10it/s] 39%|███▉      | 156682/400000 [00:16<00:25, 9463.80it/s] 39%|███▉      | 157630/400000 [00:16<00:25, 9455.47it/s] 40%|███▉      | 158577/400000 [00:17<00:25, 9287.50it/s] 40%|███▉      | 159518/400000 [00:17<00:25, 9321.50it/s] 40%|████      | 160451/400000 [00:17<00:25, 9320.72it/s] 40%|████      | 161399/400000 [00:17<00:25, 9366.75it/s] 41%|████      | 162422/400000 [00:17<00:24, 9608.22it/s] 41%|████      | 163418/400000 [00:17<00:24, 9710.17it/s] 41%|████      | 164478/400000 [00:17<00:23, 9959.56it/s] 41%|████▏     | 165513/400000 [00:17<00:23, 10071.53it/s] 42%|████▏     | 166523/400000 [00:17<00:23, 9789.65it/s]  42%|████▏     | 167506/400000 [00:17<00:23, 9799.89it/s] 42%|████▏     | 168497/400000 [00:18<00:23, 9831.91it/s] 42%|████▏     | 169492/400000 [00:18<00:23, 9866.10it/s] 43%|████▎     | 170480/400000 [00:18<00:23, 9675.99it/s] 43%|████▎     | 171450/400000 [00:18<00:23, 9610.69it/s] 43%|████▎     | 172413/400000 [00:18<00:23, 9525.82it/s] 43%|████▎     | 173367/400000 [00:18<00:23, 9451.87it/s] 44%|████▎     | 174314/400000 [00:18<00:24, 9270.65it/s] 44%|████▍     | 175243/400000 [00:18<00:24, 9171.83it/s] 44%|████▍     | 176162/400000 [00:18<00:24, 9125.58it/s] 44%|████▍     | 177130/400000 [00:19<00:24, 9282.93it/s] 45%|████▍     | 178133/400000 [00:19<00:23, 9495.07it/s] 45%|████▍     | 179150/400000 [00:19<00:22, 9685.96it/s] 45%|████▌     | 180122/400000 [00:19<00:22, 9691.36it/s] 45%|████▌     | 181093/400000 [00:19<00:22, 9622.44it/s] 46%|████▌     | 182057/400000 [00:19<00:22, 9603.08it/s] 46%|████▌     | 183019/400000 [00:19<00:22, 9576.75it/s] 46%|████▌     | 184005/400000 [00:19<00:22, 9658.14it/s] 46%|████▋     | 185017/400000 [00:19<00:21, 9790.64it/s] 47%|████▋     | 186009/400000 [00:19<00:21, 9827.02it/s] 47%|████▋     | 187003/400000 [00:20<00:21, 9859.81it/s] 47%|████▋     | 187990/400000 [00:20<00:21, 9856.55it/s] 47%|████▋     | 188976/400000 [00:20<00:21, 9786.14it/s] 47%|████▋     | 189955/400000 [00:20<00:21, 9781.28it/s] 48%|████▊     | 190937/400000 [00:20<00:21, 9792.64it/s] 48%|████▊     | 191917/400000 [00:20<00:21, 9785.07it/s] 48%|████▊     | 192896/400000 [00:20<00:21, 9744.37it/s] 48%|████▊     | 193878/400000 [00:20<00:21, 9765.93it/s] 49%|████▊     | 194855/400000 [00:20<00:21, 9532.29it/s] 49%|████▉     | 195810/400000 [00:20<00:21, 9326.40it/s] 49%|████▉     | 196751/400000 [00:21<00:21, 9348.52it/s] 49%|████▉     | 197763/400000 [00:21<00:21, 9565.73it/s] 50%|████▉     | 198722/400000 [00:21<00:21, 9430.29it/s] 50%|████▉     | 199667/400000 [00:21<00:21, 9296.64it/s] 50%|█████     | 200599/400000 [00:21<00:21, 9298.68it/s] 50%|█████     | 201554/400000 [00:21<00:21, 9371.88it/s] 51%|█████     | 202541/400000 [00:21<00:20, 9514.90it/s] 51%|█████     | 203522/400000 [00:21<00:20, 9599.70it/s] 51%|█████     | 204487/400000 [00:21<00:20, 9612.78it/s] 51%|█████▏    | 205460/400000 [00:21<00:20, 9646.08it/s] 52%|█████▏    | 206469/400000 [00:22<00:19, 9774.19it/s] 52%|█████▏    | 207460/400000 [00:22<00:19, 9811.90it/s] 52%|█████▏    | 208442/400000 [00:22<00:19, 9681.28it/s] 52%|█████▏    | 209411/400000 [00:22<00:20, 9465.93it/s] 53%|█████▎    | 210362/400000 [00:22<00:20, 9477.84it/s] 53%|█████▎    | 211341/400000 [00:22<00:19, 9567.96it/s] 53%|█████▎    | 212328/400000 [00:22<00:19, 9655.83it/s] 53%|█████▎    | 213334/400000 [00:22<00:19, 9772.49it/s] 54%|█████▎    | 214330/400000 [00:22<00:18, 9827.96it/s] 54%|█████▍    | 215314/400000 [00:22<00:18, 9801.15it/s] 54%|█████▍    | 216295/400000 [00:23<00:19, 9618.30it/s] 54%|█████▍    | 217307/400000 [00:23<00:18, 9762.68it/s] 55%|█████▍    | 218285/400000 [00:23<00:18, 9767.60it/s] 55%|█████▍    | 219263/400000 [00:23<00:18, 9742.34it/s] 55%|█████▌    | 220238/400000 [00:23<00:18, 9546.29it/s] 55%|█████▌    | 221194/400000 [00:23<00:19, 9387.50it/s] 56%|█████▌    | 222135/400000 [00:23<00:19, 9179.43it/s] 56%|█████▌    | 223056/400000 [00:23<00:19, 9115.95it/s] 56%|█████▌    | 223970/400000 [00:23<00:19, 9034.08it/s] 56%|█████▌    | 224875/400000 [00:24<00:19, 8986.34it/s] 56%|█████▋    | 225805/400000 [00:24<00:19, 9077.64it/s] 57%|█████▋    | 226739/400000 [00:24<00:18, 9152.87it/s] 57%|█████▋    | 227664/400000 [00:24<00:18, 9181.36it/s] 57%|█████▋    | 228651/400000 [00:24<00:18, 9374.72it/s] 57%|█████▋    | 229590/400000 [00:24<00:18, 9366.04it/s] 58%|█████▊    | 230569/400000 [00:24<00:17, 9487.48it/s] 58%|█████▊    | 231535/400000 [00:24<00:17, 9536.32it/s] 58%|█████▊    | 232511/400000 [00:24<00:17, 9599.78it/s] 58%|█████▊    | 233489/400000 [00:24<00:17, 9650.79it/s] 59%|█████▊    | 234455/400000 [00:25<00:17, 9513.03it/s] 59%|█████▉    | 235408/400000 [00:25<00:17, 9494.75it/s] 59%|█████▉    | 236447/400000 [00:25<00:16, 9743.86it/s] 59%|█████▉    | 237445/400000 [00:25<00:16, 9812.37it/s] 60%|█████▉    | 238428/400000 [00:25<00:16, 9777.30it/s] 60%|█████▉    | 239407/400000 [00:25<00:16, 9654.90it/s] 60%|██████    | 240388/400000 [00:25<00:16, 9698.87it/s] 60%|██████    | 241372/400000 [00:25<00:16, 9739.36it/s] 61%|██████    | 242347/400000 [00:25<00:16, 9523.12it/s] 61%|██████    | 243301/400000 [00:25<00:16, 9376.95it/s] 61%|██████    | 244241/400000 [00:26<00:16, 9367.68it/s] 61%|██████▏   | 245192/400000 [00:26<00:16, 9408.49it/s] 62%|██████▏   | 246179/400000 [00:26<00:16, 9539.29it/s] 62%|██████▏   | 247134/400000 [00:26<00:16, 9138.96it/s] 62%|██████▏   | 248088/400000 [00:26<00:16, 9255.30it/s] 62%|██████▏   | 249050/400000 [00:26<00:16, 9361.64it/s] 62%|██████▏   | 249989/400000 [00:26<00:16, 9079.21it/s] 63%|██████▎   | 250904/400000 [00:26<00:16, 9099.57it/s] 63%|██████▎   | 251817/400000 [00:26<00:16, 8732.64it/s] 63%|██████▎   | 252753/400000 [00:26<00:16, 8911.69it/s] 63%|██████▎   | 253649/400000 [00:27<00:16, 8904.47it/s] 64%|██████▎   | 254543/400000 [00:27<00:16, 8745.86it/s] 64%|██████▍   | 255458/400000 [00:27<00:16, 8862.93it/s] 64%|██████▍   | 256347/400000 [00:27<00:16, 8841.79it/s] 64%|██████▍   | 257257/400000 [00:27<00:16, 8916.86it/s] 65%|██████▍   | 258150/400000 [00:27<00:15, 8884.73it/s] 65%|██████▍   | 259066/400000 [00:27<00:15, 8964.16it/s] 65%|██████▌   | 260045/400000 [00:27<00:15, 9195.15it/s] 65%|██████▌   | 261028/400000 [00:27<00:14, 9375.08it/s] 65%|██████▌   | 261975/400000 [00:27<00:14, 9401.30it/s] 66%|██████▌   | 262917/400000 [00:28<00:14, 9291.20it/s] 66%|██████▌   | 263860/400000 [00:28<00:14, 9329.51it/s] 66%|██████▌   | 264802/400000 [00:28<00:14, 9353.81it/s] 66%|██████▋   | 265776/400000 [00:28<00:14, 9465.61it/s] 67%|██████▋   | 266724/400000 [00:28<00:14, 9392.41it/s] 67%|██████▋   | 267664/400000 [00:28<00:14, 9323.16it/s] 67%|██████▋   | 268634/400000 [00:28<00:13, 9431.88it/s] 67%|██████▋   | 269592/400000 [00:28<00:13, 9473.67it/s] 68%|██████▊   | 270609/400000 [00:28<00:13, 9671.44it/s] 68%|██████▊   | 271578/400000 [00:28<00:13, 9661.37it/s] 68%|██████▊   | 272546/400000 [00:29<00:13, 9625.15it/s] 68%|██████▊   | 273535/400000 [00:29<00:13, 9700.01it/s] 69%|██████▊   | 274519/400000 [00:29<00:12, 9741.53it/s] 69%|██████▉   | 275494/400000 [00:29<00:12, 9643.21it/s] 69%|██████▉   | 276459/400000 [00:29<00:12, 9569.73it/s] 69%|██████▉   | 277417/400000 [00:29<00:12, 9551.10it/s] 70%|██████▉   | 278373/400000 [00:29<00:13, 9320.79it/s] 70%|██████▉   | 279309/400000 [00:29<00:12, 9330.23it/s] 70%|███████   | 280296/400000 [00:29<00:12, 9485.40it/s] 70%|███████   | 281300/400000 [00:30<00:12, 9644.04it/s] 71%|███████   | 282266/400000 [00:30<00:12, 9602.57it/s] 71%|███████   | 283228/400000 [00:30<00:12, 9485.32it/s] 71%|███████   | 284178/400000 [00:30<00:12, 9381.33it/s] 71%|███████▏  | 285140/400000 [00:30<00:12, 9449.85it/s] 72%|███████▏  | 286087/400000 [00:30<00:12, 9455.49it/s] 72%|███████▏  | 287034/400000 [00:30<00:12, 9356.95it/s] 72%|███████▏  | 288012/400000 [00:30<00:11, 9478.21it/s] 72%|███████▏  | 288961/400000 [00:30<00:11, 9378.05it/s] 72%|███████▏  | 289909/400000 [00:30<00:11, 9406.70it/s] 73%|███████▎  | 290851/400000 [00:31<00:11, 9350.54it/s] 73%|███████▎  | 291787/400000 [00:31<00:11, 9270.21it/s] 73%|███████▎  | 292740/400000 [00:31<00:11, 9344.48it/s] 73%|███████▎  | 293675/400000 [00:31<00:11, 9322.67it/s] 74%|███████▎  | 294608/400000 [00:31<00:11, 9254.60it/s] 74%|███████▍  | 295544/400000 [00:31<00:11, 9285.08it/s] 74%|███████▍  | 296473/400000 [00:31<00:11, 9255.19it/s] 74%|███████▍  | 297453/400000 [00:31<00:10, 9410.07it/s] 75%|███████▍  | 298395/400000 [00:31<00:10, 9371.20it/s] 75%|███████▍  | 299333/400000 [00:31<00:10, 9228.74it/s] 75%|███████▌  | 300318/400000 [00:32<00:10, 9406.68it/s] 75%|███████▌  | 301265/400000 [00:32<00:10, 9423.85it/s] 76%|███████▌  | 302266/400000 [00:32<00:10, 9591.77it/s] 76%|███████▌  | 303254/400000 [00:32<00:10, 9674.57it/s] 76%|███████▌  | 304223/400000 [00:32<00:10, 9569.88it/s] 76%|███████▋  | 305225/400000 [00:32<00:09, 9699.17it/s] 77%|███████▋  | 306197/400000 [00:32<00:09, 9645.63it/s] 77%|███████▋  | 307163/400000 [00:32<00:09, 9438.69it/s] 77%|███████▋  | 308122/400000 [00:32<00:09, 9482.49it/s] 77%|███████▋  | 309072/400000 [00:32<00:09, 9456.56it/s] 78%|███████▊  | 310019/400000 [00:33<00:09, 9446.28it/s] 78%|███████▊  | 310965/400000 [00:33<00:09, 9305.88it/s] 78%|███████▊  | 311897/400000 [00:33<00:09, 9229.45it/s] 78%|███████▊  | 312821/400000 [00:33<00:09, 9199.83it/s] 78%|███████▊  | 313742/400000 [00:33<00:09, 9166.88it/s] 79%|███████▊  | 314670/400000 [00:33<00:09, 9198.25it/s] 79%|███████▉  | 315591/400000 [00:33<00:09, 9088.25it/s] 79%|███████▉  | 316501/400000 [00:33<00:09, 9020.84it/s] 79%|███████▉  | 317480/400000 [00:33<00:08, 9236.12it/s] 80%|███████▉  | 318441/400000 [00:33<00:08, 9344.89it/s] 80%|███████▉  | 319402/400000 [00:34<00:08, 9422.52it/s] 80%|████████  | 320363/400000 [00:34<00:08, 9475.15it/s] 80%|████████  | 321312/400000 [00:34<00:08, 9398.01it/s] 81%|████████  | 322273/400000 [00:34<00:08, 9459.60it/s] 81%|████████  | 323220/400000 [00:34<00:08, 9370.44it/s] 81%|████████  | 324158/400000 [00:34<00:08, 9294.72it/s] 81%|████████▏ | 325089/400000 [00:34<00:08, 9221.02it/s] 82%|████████▏ | 326012/400000 [00:34<00:08, 9063.80it/s] 82%|████████▏ | 326920/400000 [00:34<00:08, 9010.33it/s] 82%|████████▏ | 327841/400000 [00:34<00:07, 9067.00it/s] 82%|████████▏ | 328813/400000 [00:35<00:07, 9251.62it/s] 82%|████████▏ | 329772/400000 [00:35<00:07, 9350.49it/s] 83%|████████▎ | 330728/400000 [00:35<00:07, 9410.03it/s] 83%|████████▎ | 331678/400000 [00:35<00:07, 9434.65it/s] 83%|████████▎ | 332623/400000 [00:35<00:07, 9425.23it/s] 83%|████████▎ | 333594/400000 [00:35<00:06, 9506.94it/s] 84%|████████▎ | 334553/400000 [00:35<00:06, 9530.28it/s] 84%|████████▍ | 335507/400000 [00:35<00:06, 9419.49it/s] 84%|████████▍ | 336450/400000 [00:35<00:06, 9307.63it/s] 84%|████████▍ | 337393/400000 [00:36<00:06, 9343.67it/s] 85%|████████▍ | 338352/400000 [00:36<00:06, 9414.49it/s] 85%|████████▍ | 339294/400000 [00:36<00:06, 9200.15it/s] 85%|████████▌ | 340216/400000 [00:36<00:06, 9076.58it/s] 85%|████████▌ | 341126/400000 [00:36<00:06, 9024.03it/s] 86%|████████▌ | 342037/400000 [00:36<00:06, 9049.17it/s] 86%|████████▌ | 342994/400000 [00:36<00:06, 9198.67it/s] 86%|████████▌ | 343925/400000 [00:36<00:06, 9229.66it/s] 86%|████████▌ | 344872/400000 [00:36<00:05, 9298.39it/s] 86%|████████▋ | 345803/400000 [00:36<00:05, 9285.04it/s] 87%|████████▋ | 346732/400000 [00:37<00:05, 9259.93it/s] 87%|████████▋ | 347659/400000 [00:37<00:05, 9153.01it/s] 87%|████████▋ | 348575/400000 [00:37<00:05, 9099.92it/s] 87%|████████▋ | 349540/400000 [00:37<00:05, 9257.55it/s] 88%|████████▊ | 350489/400000 [00:37<00:05, 9324.54it/s] 88%|████████▊ | 351429/400000 [00:37<00:05, 9345.21it/s] 88%|████████▊ | 352365/400000 [00:37<00:05, 9325.78it/s] 88%|████████▊ | 353298/400000 [00:37<00:05, 9301.36it/s] 89%|████████▊ | 354259/400000 [00:37<00:04, 9391.62it/s] 89%|████████▉ | 355211/400000 [00:37<00:04, 9429.47it/s] 89%|████████▉ | 356177/400000 [00:38<00:04, 9494.89it/s] 89%|████████▉ | 357127/400000 [00:38<00:04, 9454.99it/s] 90%|████████▉ | 358073/400000 [00:38<00:04, 9371.86it/s] 90%|████████▉ | 359011/400000 [00:38<00:04, 9254.98it/s] 90%|████████▉ | 359996/400000 [00:38<00:04, 9424.48it/s] 90%|█████████ | 360971/400000 [00:38<00:04, 9519.81it/s] 90%|█████████ | 361927/400000 [00:38<00:03, 9530.92it/s] 91%|█████████ | 362881/400000 [00:38<00:03, 9314.26it/s] 91%|█████████ | 363860/400000 [00:38<00:03, 9449.85it/s] 91%|█████████ | 364807/400000 [00:38<00:03, 9446.98it/s] 91%|█████████▏| 365790/400000 [00:39<00:03, 9556.30it/s] 92%|█████████▏| 366811/400000 [00:39<00:03, 9741.31it/s] 92%|█████████▏| 367805/400000 [00:39<00:03, 9798.43it/s] 92%|█████████▏| 368804/400000 [00:39<00:03, 9854.44it/s] 92%|█████████▏| 369791/400000 [00:39<00:03, 9678.40it/s] 93%|█████████▎| 370761/400000 [00:39<00:03, 9654.57it/s] 93%|█████████▎| 371762/400000 [00:39<00:02, 9758.47it/s] 93%|█████████▎| 372739/400000 [00:39<00:02, 9652.89it/s] 93%|█████████▎| 373706/400000 [00:39<00:02, 9576.30it/s] 94%|█████████▎| 374665/400000 [00:39<00:02, 9524.10it/s] 94%|█████████▍| 375625/400000 [00:40<00:02, 9544.60it/s] 94%|█████████▍| 376580/400000 [00:40<00:02, 9510.64it/s] 94%|█████████▍| 377532/400000 [00:40<00:02, 9355.27it/s] 95%|█████████▍| 378469/400000 [00:40<00:02, 9291.40it/s] 95%|█████████▍| 379404/400000 [00:40<00:02, 9308.02it/s] 95%|█████████▌| 380349/400000 [00:40<00:02, 9348.66it/s] 95%|█████████▌| 381285/400000 [00:40<00:02, 9308.03it/s] 96%|█████████▌| 382217/400000 [00:40<00:01, 9196.46it/s] 96%|█████████▌| 383138/400000 [00:40<00:01, 9147.75it/s] 96%|█████████▌| 384061/400000 [00:40<00:01, 9172.18it/s] 96%|█████████▋| 385005/400000 [00:41<00:01, 9249.81it/s] 96%|█████████▋| 385931/400000 [00:41<00:01, 9248.96it/s] 97%|█████████▋| 386857/400000 [00:41<00:01, 9222.83it/s] 97%|█████████▋| 387804/400000 [00:41<00:01, 9292.83it/s] 97%|█████████▋| 388736/400000 [00:41<00:01, 9299.12it/s] 97%|█████████▋| 389667/400000 [00:41<00:01, 9228.75it/s] 98%|█████████▊| 390599/400000 [00:41<00:01, 9255.89it/s] 98%|█████████▊| 391539/400000 [00:41<00:00, 9297.99it/s] 98%|█████████▊| 392469/400000 [00:41<00:00, 9281.86it/s] 98%|█████████▊| 393398/400000 [00:41<00:00, 9172.68it/s] 99%|█████████▊| 394343/400000 [00:42<00:00, 9251.80it/s] 99%|█████████▉| 395277/400000 [00:42<00:00, 9275.13it/s] 99%|█████████▉| 396241/400000 [00:42<00:00, 9380.09it/s] 99%|█████████▉| 397196/400000 [00:42<00:00, 9428.17it/s]100%|█████████▉| 398144/400000 [00:42<00:00, 9442.30it/s]100%|█████████▉| 399089/400000 [00:42<00:00, 9418.93it/s]100%|█████████▉| 399999/400000 [00:42<00:00, 9371.09it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ff42caee518> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011133442620862707 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.01136111455617541 	 Accuracy: 47

  model saves at 47% accuracy 

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
2020-05-13 00:29:21.762048: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 00:29:21.766840: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095245000 Hz
2020-05-13 00:29:21.767033: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5571fc7ddb40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 00:29:21.767056: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ff3d5ebe0f0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4980 - accuracy: 0.5110
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6666 - accuracy: 0.5000 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7177 - accuracy: 0.4967
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8046 - accuracy: 0.4910
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7065 - accuracy: 0.4974
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7228 - accuracy: 0.4963
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7411 - accuracy: 0.4951
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7663 - accuracy: 0.4935
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7160 - accuracy: 0.4968
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7188 - accuracy: 0.4966
11000/25000 [============>.................] - ETA: 3s - loss: 7.7405 - accuracy: 0.4952
12000/25000 [=============>................] - ETA: 3s - loss: 7.7292 - accuracy: 0.4959
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6772 - accuracy: 0.4993
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
15000/25000 [=================>............] - ETA: 2s - loss: 7.6850 - accuracy: 0.4988
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7212 - accuracy: 0.4964
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7162 - accuracy: 0.4968
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7075 - accuracy: 0.4973
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6900 - accuracy: 0.4985
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6858 - accuracy: 0.4988
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6812 - accuracy: 0.4990
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6866 - accuracy: 0.4987
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 7s 283us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7ff38da7a710> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7ff3d7f93eb8> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4557 - crf_viterbi_accuracy: 0.3333 - val_loss: 1.3565 - val_crf_viterbi_accuracy: 0.3333

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

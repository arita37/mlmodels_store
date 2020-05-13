
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fcbc9745fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 01:13:15.524949
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 01:13:15.528413
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 01:13:15.531575
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 01:13:15.534830
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fcbd550f4a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356177.7188
Epoch 2/10

1/1 [==============================] - 0s 109ms/step - loss: 276892.8438
Epoch 3/10

1/1 [==============================] - 0s 109ms/step - loss: 202110.3594
Epoch 4/10

1/1 [==============================] - 0s 104ms/step - loss: 135999.3281
Epoch 5/10

1/1 [==============================] - 0s 100ms/step - loss: 88424.6641
Epoch 6/10

1/1 [==============================] - 0s 102ms/step - loss: 57401.0078
Epoch 7/10

1/1 [==============================] - 0s 97ms/step - loss: 38198.4609
Epoch 8/10

1/1 [==============================] - 0s 102ms/step - loss: 26298.4707
Epoch 9/10

1/1 [==============================] - 0s 100ms/step - loss: 18830.8984
Epoch 10/10

1/1 [==============================] - 0s 101ms/step - loss: 14014.0342

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.1650367e+00  7.8455442e-01 -5.5686420e-01  4.1689590e-01
  -5.4887807e-01  8.6807549e-02 -8.3599740e-01  3.0150458e-01
  -2.3556669e-01  4.5132113e-01  3.3710223e-01 -2.4500895e-01
  -5.9755087e-01 -2.9532909e-03  1.3811711e+00 -1.0661736e+00
  -5.1627487e-02 -1.3790073e+00  9.5392597e-01  1.0507405e-02
   8.0437946e-01  7.3318851e-01  3.2924175e-01 -2.8696185e-01
  -1.1700795e+00 -4.1675851e-01  7.2674471e-01 -5.4283822e-01
   2.0124602e-01 -8.2335061e-01 -2.0422548e-02 -5.7652605e-01
   1.7005333e-01  1.9341595e-01 -2.6314825e-02  5.2247036e-01
   9.8800516e-01 -1.1348680e+00  3.4704036e-01  6.3784307e-01
   4.6516299e-02  1.7336583e-01  3.0461136e-01 -9.5720088e-01
   5.4699981e-01  2.9387349e-01  7.4887228e-01  1.5647858e+00
   4.1103250e-01 -6.4059395e-01  3.7925443e-01  8.2784355e-02
  -1.3294835e+00 -4.2824885e-01  4.6847603e-01 -2.4993846e-01
   1.4042742e+00  5.4309964e-01  6.5990090e-01  1.1948166e+00
  -1.3036778e+00 -1.9010594e-01 -9.1726053e-01 -6.4727122e-01
   1.7578150e-01 -6.8618104e-02  1.5745181e-01 -2.9438686e-01
   4.2326930e-01  6.4539069e-01 -1.7187357e-01  6.9222873e-01
   7.2406304e-01 -2.3480469e-01 -2.8665435e-01  1.1708913e+00
  -7.2292127e-02  1.1123273e+00  7.8557009e-01  4.0150592e-01
   6.2140197e-01 -6.3353807e-01  6.1394817e-01 -3.7658754e-01
   3.4039214e-01  1.7014277e-01  8.2705820e-01  2.2473764e-01
  -3.4651530e-01  1.0802937e+00  2.5915259e-01  6.6696501e-01
   3.7930366e-01  3.8957426e-01 -1.6703153e-01  4.4095737e-01
  -1.0256560e+00  2.8612208e-01 -9.5766598e-01  2.1446553e-01
   3.4370810e-01  1.9990426e-01  1.5370786e-02  3.7789285e-01
   8.8665080e-01  6.5502989e-01  4.3676403e-01  6.4485002e-01
  -3.5991659e-03 -1.4726691e+00 -7.2295725e-01  1.1640756e+00
   9.5097399e-01  1.5645987e-01 -1.4316440e-02  6.3292712e-01
  -2.6082736e-01  2.5183153e-01  4.2790937e-01  2.8256243e-01
  -2.7097568e-02  6.1954961e+00  4.4159555e+00  5.6226616e+00
   5.3165550e+00  4.3833585e+00  5.2362628e+00  5.4539924e+00
   5.2541561e+00  5.5155396e+00  5.0594053e+00  4.6109438e+00
   6.0311694e+00  4.6769586e+00  5.1302538e+00  3.9770975e+00
   3.8952353e+00  3.8037856e+00  5.0461006e+00  5.8386488e+00
   3.7471247e+00  4.6786547e+00  3.7399428e+00  5.6864395e+00
   4.3373904e+00  5.7486877e+00  4.6126614e+00  4.7104630e+00
   5.0267062e+00  5.0325704e+00  5.2388186e+00  5.3943219e+00
   5.3355308e+00  5.2530947e+00  5.4509411e+00  6.1975560e+00
   5.2622204e+00  5.3158531e+00  5.5943522e+00  4.4084821e+00
   5.2890368e+00  4.7426348e+00  5.2341123e+00  4.5340161e+00
   6.5549846e+00  6.0423403e+00  4.7995491e+00  5.3732424e+00
   4.9109135e+00  4.9506845e+00  5.7450347e+00  4.8342533e+00
   4.5860658e+00  5.0847940e+00  5.3638201e+00  4.7417488e+00
   6.0519128e+00  4.9471598e+00  6.1396160e+00  5.9230185e+00
   1.1729841e+00  6.7668009e-01  6.3913506e-01  2.3609934e+00
   2.1714287e+00  2.1784735e+00  1.3192582e+00  9.1416883e-01
   5.4724634e-01  2.4619842e-01  5.8105081e-01  1.4376802e+00
   4.6482825e-01  2.1158538e+00  7.8998262e-01  2.2737794e+00
   1.1875608e+00  1.3824755e+00  6.6888559e-01  2.5991511e-01
   1.3164425e+00  4.9386907e-01  8.1606358e-01  1.6205612e+00
   1.7978723e+00  4.1319203e-01  6.3520569e-01  1.0675988e+00
   1.1154834e+00  1.4254884e+00  3.2537282e-01  1.0349557e+00
   2.6210195e-01  2.3872763e-01  6.9913661e-01  9.9338257e-01
   9.1414076e-01  9.3375015e-01  7.0332360e-01  1.3932011e+00
   2.1945195e+00  1.5870626e+00  8.7042987e-01  5.0155216e-01
   2.0625215e+00  7.4741936e-01  8.8537514e-01  9.7263634e-01
   1.0092704e+00  8.1143355e-01  1.3203965e+00  6.5169322e-01
   2.0186365e-01  1.5983740e+00  8.9453638e-01  7.0769382e-01
   8.0879849e-01  6.2122345e-01  9.0824151e-01  1.1490965e+00
   1.7987908e+00  6.2164313e-01  3.3772093e-01  9.5286715e-01
   1.0774288e+00  3.0302745e-01  8.6877120e-01  1.6127133e+00
   1.0539842e+00  4.6639699e-01  5.2102286e-01  7.5010544e-01
   2.4287143e+00  3.9065659e-01  5.3693062e-01  3.5053593e-01
   1.3365216e+00  1.1201208e+00  4.9958014e-01  7.9271287e-01
   1.2660792e+00  2.3077005e-01  5.9409034e-01  1.2597984e+00
   3.5186708e-01  3.1709337e-01  9.3769205e-01  9.3779939e-01
   1.8382596e+00  1.6435368e+00  3.6780554e-01  1.8082066e+00
   1.1547617e+00  1.6362793e+00  1.1332638e+00  9.9328268e-01
   1.4957716e+00  1.1366147e+00  2.2276640e+00  1.1321641e+00
   4.6474570e-01  2.5030088e+00  6.0224104e-01  4.3499923e-01
   2.4758520e+00  1.7490425e+00  1.2136192e+00  5.9843546e-01
   1.8977510e+00  7.5579441e-01  1.6122028e+00  1.2977837e+00
   1.6330076e+00  1.2166421e+00  1.5498567e+00  4.7335172e-01
   8.4541333e-01  1.7837453e+00  1.0359492e+00  8.5983956e-01
   2.7137637e-02  6.1775475e+00  5.7076268e+00  5.9598141e+00
   6.7028933e+00  5.9156690e+00  6.8324661e+00  6.1736932e+00
   6.0974078e+00  6.4579754e+00  5.8875213e+00  5.8864927e+00
   6.2107158e+00  5.2147999e+00  5.9433188e+00  5.0187736e+00
   5.9689717e+00  5.9338088e+00  5.0892677e+00  6.4568644e+00
   6.3851566e+00  5.4448476e+00  6.0505247e+00  5.5057607e+00
   5.3134317e+00  6.4715953e+00  5.9535480e+00  5.3550510e+00
   4.7429276e+00  6.2669139e+00  4.6671290e+00  6.0052333e+00
   4.5674763e+00  4.4751382e+00  4.7897964e+00  6.8438969e+00
   5.5686612e+00  5.6792297e+00  6.2011604e+00  5.1313205e+00
   6.6732607e+00  6.5214653e+00  5.2326202e+00  6.8593044e+00
   5.2787085e+00  4.9365420e+00  5.4607668e+00  6.5068979e+00
   5.7163200e+00  5.2082381e+00  4.8934865e+00  5.8303862e+00
   5.6521592e+00  5.8903451e+00  5.7560987e+00  6.0438995e+00
   6.3982892e+00  6.0139070e+00  6.0182090e+00  6.0594573e+00
  -6.0211573e+00 -3.0136094e+00  6.4054914e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 01:13:25.861914
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.5021
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 01:13:25.868164
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9333.38
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 01:13:25.871703
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.3516
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 01:13:25.875510
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   -834.87
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140512875787880
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140510363046576
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140510363047080
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140510363047584
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140510363048088
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140510363048592

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fcbc2e7c588> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.474098
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.451193
grad_step = 000002, loss = 0.437700
grad_step = 000003, loss = 0.427162
grad_step = 000004, loss = 0.417034
grad_step = 000005, loss = 0.409076
grad_step = 000006, loss = 0.403127
grad_step = 000007, loss = 0.394269
grad_step = 000008, loss = 0.385460
grad_step = 000009, loss = 0.377056
grad_step = 000010, loss = 0.368932
grad_step = 000011, loss = 0.360609
grad_step = 000012, loss = 0.352245
grad_step = 000013, loss = 0.344100
grad_step = 000014, loss = 0.336359
grad_step = 000015, loss = 0.328932
grad_step = 000016, loss = 0.321348
grad_step = 000017, loss = 0.313606
grad_step = 000018, loss = 0.305936
grad_step = 000019, loss = 0.298395
grad_step = 000020, loss = 0.290913
grad_step = 000021, loss = 0.283426
grad_step = 000022, loss = 0.276108
grad_step = 000023, loss = 0.268895
grad_step = 000024, loss = 0.261660
grad_step = 000025, loss = 0.254379
grad_step = 000026, loss = 0.247274
grad_step = 000027, loss = 0.240306
grad_step = 000028, loss = 0.233434
grad_step = 000029, loss = 0.226666
grad_step = 000030, loss = 0.219999
grad_step = 000031, loss = 0.213428
grad_step = 000032, loss = 0.206920
grad_step = 000033, loss = 0.200542
grad_step = 000034, loss = 0.194344
grad_step = 000035, loss = 0.188262
grad_step = 000036, loss = 0.182270
grad_step = 000037, loss = 0.176437
grad_step = 000038, loss = 0.170756
grad_step = 000039, loss = 0.165166
grad_step = 000040, loss = 0.159726
grad_step = 000041, loss = 0.154436
grad_step = 000042, loss = 0.149249
grad_step = 000043, loss = 0.144206
grad_step = 000044, loss = 0.139308
grad_step = 000045, loss = 0.134541
grad_step = 000046, loss = 0.129890
grad_step = 000047, loss = 0.125369
grad_step = 000048, loss = 0.120959
grad_step = 000049, loss = 0.116683
grad_step = 000050, loss = 0.112557
grad_step = 000051, loss = 0.108562
grad_step = 000052, loss = 0.104766
grad_step = 000053, loss = 0.101051
grad_step = 000054, loss = 0.097330
grad_step = 000055, loss = 0.093603
grad_step = 000056, loss = 0.090263
grad_step = 000057, loss = 0.086997
grad_step = 000058, loss = 0.083596
grad_step = 000059, loss = 0.080521
grad_step = 000060, loss = 0.077567
grad_step = 000061, loss = 0.074496
grad_step = 000062, loss = 0.071703
grad_step = 000063, loss = 0.069045
grad_step = 000064, loss = 0.066261
grad_step = 000065, loss = 0.063726
grad_step = 000066, loss = 0.061322
grad_step = 000067, loss = 0.058830
grad_step = 000068, loss = 0.056511
grad_step = 000069, loss = 0.054320
grad_step = 000070, loss = 0.052137
grad_step = 000071, loss = 0.050009
grad_step = 000072, loss = 0.047987
grad_step = 000073, loss = 0.046078
grad_step = 000074, loss = 0.044179
grad_step = 000075, loss = 0.042310
grad_step = 000076, loss = 0.040563
grad_step = 000077, loss = 0.038890
grad_step = 000078, loss = 0.037259
grad_step = 000079, loss = 0.035677
grad_step = 000080, loss = 0.034125
grad_step = 000081, loss = 0.032638
grad_step = 000082, loss = 0.031236
grad_step = 000083, loss = 0.029888
grad_step = 000084, loss = 0.028593
grad_step = 000085, loss = 0.027364
grad_step = 000086, loss = 0.026179
grad_step = 000087, loss = 0.025031
grad_step = 000088, loss = 0.023928
grad_step = 000089, loss = 0.022827
grad_step = 000090, loss = 0.021744
grad_step = 000091, loss = 0.020704
grad_step = 000092, loss = 0.019729
grad_step = 000093, loss = 0.018815
grad_step = 000094, loss = 0.017966
grad_step = 000095, loss = 0.017181
grad_step = 000096, loss = 0.016456
grad_step = 000097, loss = 0.015788
grad_step = 000098, loss = 0.015139
grad_step = 000099, loss = 0.014430
grad_step = 000100, loss = 0.013607
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.012848
grad_step = 000102, loss = 0.012279
grad_step = 000103, loss = 0.011823
grad_step = 000104, loss = 0.011340
grad_step = 000105, loss = 0.010754
grad_step = 000106, loss = 0.010153
grad_step = 000107, loss = 0.009661
grad_step = 000108, loss = 0.009281
grad_step = 000109, loss = 0.008930
grad_step = 000110, loss = 0.008529
grad_step = 000111, loss = 0.008075
grad_step = 000112, loss = 0.007653
grad_step = 000113, loss = 0.007316
grad_step = 000114, loss = 0.007043
grad_step = 000115, loss = 0.006784
grad_step = 000116, loss = 0.006507
grad_step = 000117, loss = 0.006208
grad_step = 000118, loss = 0.005903
grad_step = 000119, loss = 0.005619
grad_step = 000120, loss = 0.005374
grad_step = 000121, loss = 0.005169
grad_step = 000122, loss = 0.004990
grad_step = 000123, loss = 0.004831
grad_step = 000124, loss = 0.004692
grad_step = 000125, loss = 0.004569
grad_step = 000126, loss = 0.004456
grad_step = 000127, loss = 0.004326
grad_step = 000128, loss = 0.004153
grad_step = 000129, loss = 0.003932
grad_step = 000130, loss = 0.003716
grad_step = 000131, loss = 0.003552
grad_step = 000132, loss = 0.003453
grad_step = 000133, loss = 0.003399
grad_step = 000134, loss = 0.003361
grad_step = 000135, loss = 0.003309
grad_step = 000136, loss = 0.003219
grad_step = 000137, loss = 0.003093
grad_step = 000138, loss = 0.002954
grad_step = 000139, loss = 0.002838
grad_step = 000140, loss = 0.002762
grad_step = 000141, loss = 0.002722
grad_step = 000142, loss = 0.002704
grad_step = 000143, loss = 0.002689
grad_step = 000144, loss = 0.002667
grad_step = 000145, loss = 0.002624
grad_step = 000146, loss = 0.002569
grad_step = 000147, loss = 0.002506
grad_step = 000148, loss = 0.002437
grad_step = 000149, loss = 0.002374
grad_step = 000150, loss = 0.002322
grad_step = 000151, loss = 0.002285
grad_step = 000152, loss = 0.002260
grad_step = 000153, loss = 0.002241
grad_step = 000154, loss = 0.002229
grad_step = 000155, loss = 0.002225
grad_step = 000156, loss = 0.002237
grad_step = 000157, loss = 0.002280
grad_step = 000158, loss = 0.002360
grad_step = 000159, loss = 0.002470
grad_step = 000160, loss = 0.002561
grad_step = 000161, loss = 0.002543
grad_step = 000162, loss = 0.002349
grad_step = 000163, loss = 0.002132
grad_step = 000164, loss = 0.002080
grad_step = 000165, loss = 0.002186
grad_step = 000166, loss = 0.002290
grad_step = 000167, loss = 0.002250
grad_step = 000168, loss = 0.002112
grad_step = 000169, loss = 0.002039
grad_step = 000170, loss = 0.002089
grad_step = 000171, loss = 0.002165
grad_step = 000172, loss = 0.002149
grad_step = 000173, loss = 0.002060
grad_step = 000174, loss = 0.002014
grad_step = 000175, loss = 0.002049
grad_step = 000176, loss = 0.002095
grad_step = 000177, loss = 0.002081
grad_step = 000178, loss = 0.002027
grad_step = 000179, loss = 0.001997
grad_step = 000180, loss = 0.002014
grad_step = 000181, loss = 0.002043
grad_step = 000182, loss = 0.002042
grad_step = 000183, loss = 0.002010
grad_step = 000184, loss = 0.001985
grad_step = 000185, loss = 0.001982
grad_step = 000186, loss = 0.001997
grad_step = 000187, loss = 0.002009
grad_step = 000188, loss = 0.002007
grad_step = 000189, loss = 0.001991
grad_step = 000190, loss = 0.001974
grad_step = 000191, loss = 0.001965
grad_step = 000192, loss = 0.001968
grad_step = 000193, loss = 0.001975
grad_step = 000194, loss = 0.001980
grad_step = 000195, loss = 0.001978
grad_step = 000196, loss = 0.001969
grad_step = 000197, loss = 0.001959
grad_step = 000198, loss = 0.001951
grad_step = 000199, loss = 0.001947
grad_step = 000200, loss = 0.001946
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001948
grad_step = 000202, loss = 0.001951
grad_step = 000203, loss = 0.001953
grad_step = 000204, loss = 0.001954
grad_step = 000205, loss = 0.001955
grad_step = 000206, loss = 0.001956
grad_step = 000207, loss = 0.001957
grad_step = 000208, loss = 0.001960
grad_step = 000209, loss = 0.001964
grad_step = 000210, loss = 0.001972
grad_step = 000211, loss = 0.001986
grad_step = 000212, loss = 0.002007
grad_step = 000213, loss = 0.002036
grad_step = 000214, loss = 0.002074
grad_step = 000215, loss = 0.002111
grad_step = 000216, loss = 0.002135
grad_step = 000217, loss = 0.002120
grad_step = 000218, loss = 0.002065
grad_step = 000219, loss = 0.001985
grad_step = 000220, loss = 0.001922
grad_step = 000221, loss = 0.001902
grad_step = 000222, loss = 0.001924
grad_step = 000223, loss = 0.001965
grad_step = 000224, loss = 0.001999
grad_step = 000225, loss = 0.002005
grad_step = 000226, loss = 0.001978
grad_step = 000227, loss = 0.001935
grad_step = 000228, loss = 0.001898
grad_step = 000229, loss = 0.001887
grad_step = 000230, loss = 0.001899
grad_step = 000231, loss = 0.001921
grad_step = 000232, loss = 0.001937
grad_step = 000233, loss = 0.001937
grad_step = 000234, loss = 0.001921
grad_step = 000235, loss = 0.001898
grad_step = 000236, loss = 0.001879
grad_step = 000237, loss = 0.001871
grad_step = 000238, loss = 0.001874
grad_step = 000239, loss = 0.001882
grad_step = 000240, loss = 0.001891
grad_step = 000241, loss = 0.001896
grad_step = 000242, loss = 0.001897
grad_step = 000243, loss = 0.001893
grad_step = 000244, loss = 0.001886
grad_step = 000245, loss = 0.001878
grad_step = 000246, loss = 0.001870
grad_step = 000247, loss = 0.001864
grad_step = 000248, loss = 0.001861
grad_step = 000249, loss = 0.001861
grad_step = 000250, loss = 0.001867
grad_step = 000251, loss = 0.001880
grad_step = 000252, loss = 0.001903
grad_step = 000253, loss = 0.001935
grad_step = 000254, loss = 0.001976
grad_step = 000255, loss = 0.001999
grad_step = 000256, loss = 0.001982
grad_step = 000257, loss = 0.001918
grad_step = 000258, loss = 0.001863
grad_step = 000259, loss = 0.001855
grad_step = 000260, loss = 0.001893
grad_step = 000261, loss = 0.001938
grad_step = 000262, loss = 0.001932
grad_step = 000263, loss = 0.001905
grad_step = 000264, loss = 0.001888
grad_step = 000265, loss = 0.001914
grad_step = 000266, loss = 0.001966
grad_step = 000267, loss = 0.001979
grad_step = 000268, loss = 0.001961
grad_step = 000269, loss = 0.001910
grad_step = 000270, loss = 0.001886
grad_step = 000271, loss = 0.001891
grad_step = 000272, loss = 0.001882
grad_step = 000273, loss = 0.001854
grad_step = 000274, loss = 0.001819
grad_step = 000275, loss = 0.001815
grad_step = 000276, loss = 0.001840
grad_step = 000277, loss = 0.001859
grad_step = 000278, loss = 0.001857
grad_step = 000279, loss = 0.001835
grad_step = 000280, loss = 0.001822
grad_step = 000281, loss = 0.001825
grad_step = 000282, loss = 0.001829
grad_step = 000283, loss = 0.001823
grad_step = 000284, loss = 0.001808
grad_step = 000285, loss = 0.001797
grad_step = 000286, loss = 0.001797
grad_step = 000287, loss = 0.001805
grad_step = 000288, loss = 0.001812
grad_step = 000289, loss = 0.001813
grad_step = 000290, loss = 0.001809
grad_step = 000291, loss = 0.001803
grad_step = 000292, loss = 0.001800
grad_step = 000293, loss = 0.001800
grad_step = 000294, loss = 0.001804
grad_step = 000295, loss = 0.001809
grad_step = 000296, loss = 0.001813
grad_step = 000297, loss = 0.001816
grad_step = 000298, loss = 0.001817
grad_step = 000299, loss = 0.001819
grad_step = 000300, loss = 0.001821
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001828
grad_step = 000302, loss = 0.001842
grad_step = 000303, loss = 0.001862
grad_step = 000304, loss = 0.001895
grad_step = 000305, loss = 0.001922
grad_step = 000306, loss = 0.001952
grad_step = 000307, loss = 0.001937
grad_step = 000308, loss = 0.001899
grad_step = 000309, loss = 0.001829
grad_step = 000310, loss = 0.001778
grad_step = 000311, loss = 0.001768
grad_step = 000312, loss = 0.001794
grad_step = 000313, loss = 0.001831
grad_step = 000314, loss = 0.001842
grad_step = 000315, loss = 0.001833
grad_step = 000316, loss = 0.001813
grad_step = 000317, loss = 0.001824
grad_step = 000318, loss = 0.001865
grad_step = 000319, loss = 0.001938
grad_step = 000320, loss = 0.001968
grad_step = 000321, loss = 0.001968
grad_step = 000322, loss = 0.001877
grad_step = 000323, loss = 0.001788
grad_step = 000324, loss = 0.001749
grad_step = 000325, loss = 0.001779
grad_step = 000326, loss = 0.001831
grad_step = 000327, loss = 0.001838
grad_step = 000328, loss = 0.001800
grad_step = 000329, loss = 0.001750
grad_step = 000330, loss = 0.001740
grad_step = 000331, loss = 0.001767
grad_step = 000332, loss = 0.001791
grad_step = 000333, loss = 0.001786
grad_step = 000334, loss = 0.001755
grad_step = 000335, loss = 0.001732
grad_step = 000336, loss = 0.001733
grad_step = 000337, loss = 0.001750
grad_step = 000338, loss = 0.001760
grad_step = 000339, loss = 0.001751
grad_step = 000340, loss = 0.001734
grad_step = 000341, loss = 0.001722
grad_step = 000342, loss = 0.001723
grad_step = 000343, loss = 0.001732
grad_step = 000344, loss = 0.001737
grad_step = 000345, loss = 0.001733
grad_step = 000346, loss = 0.001723
grad_step = 000347, loss = 0.001714
grad_step = 000348, loss = 0.001712
grad_step = 000349, loss = 0.001715
grad_step = 000350, loss = 0.001719
grad_step = 000351, loss = 0.001719
grad_step = 000352, loss = 0.001715
grad_step = 000353, loss = 0.001710
grad_step = 000354, loss = 0.001706
grad_step = 000355, loss = 0.001704
grad_step = 000356, loss = 0.001706
grad_step = 000357, loss = 0.001710
grad_step = 000358, loss = 0.001718
grad_step = 000359, loss = 0.001730
grad_step = 000360, loss = 0.001757
grad_step = 000361, loss = 0.001814
grad_step = 000362, loss = 0.001947
grad_step = 000363, loss = 0.002157
grad_step = 000364, loss = 0.002487
grad_step = 000365, loss = 0.002471
grad_step = 000366, loss = 0.002133
grad_step = 000367, loss = 0.001716
grad_step = 000368, loss = 0.001854
grad_step = 000369, loss = 0.002099
grad_step = 000370, loss = 0.001838
grad_step = 000371, loss = 0.001724
grad_step = 000372, loss = 0.001945
grad_step = 000373, loss = 0.001831
grad_step = 000374, loss = 0.001699
grad_step = 000375, loss = 0.001864
grad_step = 000376, loss = 0.001786
grad_step = 000377, loss = 0.001703
grad_step = 000378, loss = 0.001806
grad_step = 000379, loss = 0.001742
grad_step = 000380, loss = 0.001701
grad_step = 000381, loss = 0.001767
grad_step = 000382, loss = 0.001714
grad_step = 000383, loss = 0.001694
grad_step = 000384, loss = 0.001742
grad_step = 000385, loss = 0.001693
grad_step = 000386, loss = 0.001694
grad_step = 000387, loss = 0.001722
grad_step = 000388, loss = 0.001681
grad_step = 000389, loss = 0.001689
grad_step = 000390, loss = 0.001706
grad_step = 000391, loss = 0.001675
grad_step = 000392, loss = 0.001681
grad_step = 000393, loss = 0.001694
grad_step = 000394, loss = 0.001669
grad_step = 000395, loss = 0.001673
grad_step = 000396, loss = 0.001682
grad_step = 000397, loss = 0.001664
grad_step = 000398, loss = 0.001665
grad_step = 000399, loss = 0.001673
grad_step = 000400, loss = 0.001660
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001657
grad_step = 000402, loss = 0.001664
grad_step = 000403, loss = 0.001656
grad_step = 000404, loss = 0.001650
grad_step = 000405, loss = 0.001655
grad_step = 000406, loss = 0.001652
grad_step = 000407, loss = 0.001645
grad_step = 000408, loss = 0.001647
grad_step = 000409, loss = 0.001647
grad_step = 000410, loss = 0.001642
grad_step = 000411, loss = 0.001639
grad_step = 000412, loss = 0.001641
grad_step = 000413, loss = 0.001639
grad_step = 000414, loss = 0.001635
grad_step = 000415, loss = 0.001634
grad_step = 000416, loss = 0.001634
grad_step = 000417, loss = 0.001631
grad_step = 000418, loss = 0.001628
grad_step = 000419, loss = 0.001628
grad_step = 000420, loss = 0.001627
grad_step = 000421, loss = 0.001625
grad_step = 000422, loss = 0.001622
grad_step = 000423, loss = 0.001622
grad_step = 000424, loss = 0.001621
grad_step = 000425, loss = 0.001619
grad_step = 000426, loss = 0.001617
grad_step = 000427, loss = 0.001615
grad_step = 000428, loss = 0.001615
grad_step = 000429, loss = 0.001613
grad_step = 000430, loss = 0.001612
grad_step = 000431, loss = 0.001610
grad_step = 000432, loss = 0.001609
grad_step = 000433, loss = 0.001608
grad_step = 000434, loss = 0.001608
grad_step = 000435, loss = 0.001608
grad_step = 000436, loss = 0.001610
grad_step = 000437, loss = 0.001617
grad_step = 000438, loss = 0.001633
grad_step = 000439, loss = 0.001664
grad_step = 000440, loss = 0.001727
grad_step = 000441, loss = 0.001813
grad_step = 000442, loss = 0.001954
grad_step = 000443, loss = 0.002025
grad_step = 000444, loss = 0.002051
grad_step = 000445, loss = 0.001883
grad_step = 000446, loss = 0.001704
grad_step = 000447, loss = 0.001634
grad_step = 000448, loss = 0.001671
grad_step = 000449, loss = 0.001729
grad_step = 000450, loss = 0.001711
grad_step = 000451, loss = 0.001677
grad_step = 000452, loss = 0.001679
grad_step = 000453, loss = 0.001679
grad_step = 000454, loss = 0.001652
grad_step = 000455, loss = 0.001603
grad_step = 000456, loss = 0.001600
grad_step = 000457, loss = 0.001643
grad_step = 000458, loss = 0.001658
grad_step = 000459, loss = 0.001625
grad_step = 000460, loss = 0.001578
grad_step = 000461, loss = 0.001577
grad_step = 000462, loss = 0.001608
grad_step = 000463, loss = 0.001617
grad_step = 000464, loss = 0.001597
grad_step = 000465, loss = 0.001574
grad_step = 000466, loss = 0.001573
grad_step = 000467, loss = 0.001583
grad_step = 000468, loss = 0.001584
grad_step = 000469, loss = 0.001577
grad_step = 000470, loss = 0.001570
grad_step = 000471, loss = 0.001569
grad_step = 000472, loss = 0.001567
grad_step = 000473, loss = 0.001562
grad_step = 000474, loss = 0.001560
grad_step = 000475, loss = 0.001562
grad_step = 000476, loss = 0.001563
grad_step = 000477, loss = 0.001559
grad_step = 000478, loss = 0.001551
grad_step = 000479, loss = 0.001547
grad_step = 000480, loss = 0.001548
grad_step = 000481, loss = 0.001551
grad_step = 000482, loss = 0.001551
grad_step = 000483, loss = 0.001547
grad_step = 000484, loss = 0.001542
grad_step = 000485, loss = 0.001539
grad_step = 000486, loss = 0.001539
grad_step = 000487, loss = 0.001539
grad_step = 000488, loss = 0.001537
grad_step = 000489, loss = 0.001535
grad_step = 000490, loss = 0.001533
grad_step = 000491, loss = 0.001532
grad_step = 000492, loss = 0.001532
grad_step = 000493, loss = 0.001531
grad_step = 000494, loss = 0.001530
grad_step = 000495, loss = 0.001528
grad_step = 000496, loss = 0.001526
grad_step = 000497, loss = 0.001524
grad_step = 000498, loss = 0.001524
grad_step = 000499, loss = 0.001523
grad_step = 000500, loss = 0.001522
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001521
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

  date_run                              2020-05-13 01:13:47.998382
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.210967
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 01:13:48.004085
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.104579
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 01:13:48.011502
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.139848
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 01:13:48.016504
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.589116
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
0   2020-05-13 01:13:15.524949  ...    mean_absolute_error
1   2020-05-13 01:13:15.528413  ...     mean_squared_error
2   2020-05-13 01:13:15.531575  ...  median_absolute_error
3   2020-05-13 01:13:15.534830  ...               r2_score
4   2020-05-13 01:13:25.861914  ...    mean_absolute_error
5   2020-05-13 01:13:25.868164  ...     mean_squared_error
6   2020-05-13 01:13:25.871703  ...  median_absolute_error
7   2020-05-13 01:13:25.875510  ...               r2_score
8   2020-05-13 01:13:47.998382  ...    mean_absolute_error
9   2020-05-13 01:13:48.004085  ...     mean_squared_error
10  2020-05-13 01:13:48.011502  ...  median_absolute_error
11  2020-05-13 01:13:48.016504  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff4781909b0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 155887.63it/s] 44%|████▍     | 4382720/9912422 [00:00<00:24, 222352.07it/s]9920512it [00:00, 37500692.31it/s]                           
0it [00:00, ?it/s]32768it [00:00, 610527.75it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 455781.05it/s]1654784it [00:00, 11576789.18it/s]                         
0it [00:00, ?it/s]8192it [00:00, 205609.04it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff42ab3ee48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff42798b0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff42ab3ee48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff42798b048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff4279004a8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff42798b0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff42ab3ee48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff42798b048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff4279004a8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff478148e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f7e45e2d208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=17ad64509452f31bdca25949d480dcab6eae06d7c667d62b9d93936341d7c79b
  Stored in directory: /tmp/pip-ephem-wheel-cache-t6rdt4vl/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f7dddc28710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3235840/17464789 [====>.........................] - ETA: 0s
 7626752/17464789 [============>.................] - ETA: 0s
14876672/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 01:15:12.661617: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 01:15:12.665218: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 01:15:12.665412: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559b58f53010 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 01:15:12.665431: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5516 - accuracy: 0.5075 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.4724 - accuracy: 0.5127
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.3830 - accuracy: 0.5185
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.4734 - accuracy: 0.5126
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5312 - accuracy: 0.5088
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5812 - accuracy: 0.5056
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6168 - accuracy: 0.5033
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6206 - accuracy: 0.5030
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6452 - accuracy: 0.5014
11000/25000 [============>.................] - ETA: 4s - loss: 7.6318 - accuracy: 0.5023
12000/25000 [=============>................] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6631 - accuracy: 0.5002
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6458 - accuracy: 0.5014
15000/25000 [=================>............] - ETA: 3s - loss: 7.6738 - accuracy: 0.4995
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6590 - accuracy: 0.5005
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6702 - accuracy: 0.4998
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6641 - accuracy: 0.5002
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6594 - accuracy: 0.5005
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6582 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6601 - accuracy: 0.5004
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6736 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6706 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 9s 370us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 01:15:28.624349
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 01:15:28.624349  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<23:07:21, 10.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<16:25:17, 14.6kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:32:55, 20.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<8:05:30, 29.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.62M/862M [00:01<5:38:57, 42.2kB/s].vector_cache/glove.6B.zip:   1%|          | 7.46M/862M [00:01<3:56:19, 60.3kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:01<2:44:37, 86.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 16.8M/862M [00:01<1:54:42, 123kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.8M/862M [00:01<1:20:01, 175kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.7M/862M [00:01<55:46, 250kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 29.4M/862M [00:01<38:59, 356kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.3M/862M [00:02<27:13, 507kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.9M/862M [00:02<19:05, 720kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.8M/862M [00:02<13:21, 1.02MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.5M/862M [00:02<09:25, 1.44MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.9M/862M [00:02<06:39, 2.03MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.2M/862M [00:02<06:05, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<06:09, 2.18MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<06:14, 2.15MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:05<04:50, 2.77MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<05:53, 2.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:06<05:28, 2.44MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<04:06, 3.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<06:00, 2.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<05:33, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:09<04:13, 3.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<06:04, 2.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<06:58, 1.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:11<05:27, 2.42MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.7M/862M [00:11<04:00, 3.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<08:44, 1.50MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<07:29, 1.76MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.7M/862M [00:12<05:31, 2.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:14<06:56, 1.89MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:14<07:33, 1.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:14<05:50, 2.24MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:15<04:14, 3.07MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:16<10:18, 1.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:16<08:34, 1.52MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:16<06:19, 2.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:18<07:26, 1.74MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<07:50, 1.65MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<06:02, 2.14MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.1M/862M [00:18<04:24, 2.93MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<08:48, 1.46MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<07:28, 1.72MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:20<05:33, 2.31MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:22<06:53, 1.86MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<07:27, 1.72MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.3M/862M [00:22<05:51, 2.19MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:22<04:14, 3.00MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<48:42, 262kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<35:22, 360kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<24:59, 509kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<20:26, 620kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<16:52, 751kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<12:22, 1.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<08:47, 1.43MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<12:13:11, 17.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<8:34:17, 24.5kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<5:59:32, 35.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<4:13:53, 49.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<2:58:56, 70.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<2:05:19, 99.8kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<1:30:24, 138kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<1:04:31, 193kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<45:23, 274kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<34:37, 358kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<26:45, 463kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<19:20, 640kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<13:36, 906kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<12:03:33, 17.0kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<8:27:31, 24.3kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<5:54:49, 34.7kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<4:10:31, 49.0kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<2:56:31, 69.4kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<2:03:37, 99.0kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<1:29:10, 137kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<1:04:53, 188kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<45:59, 265kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<34:01, 356kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<25:04, 484kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<17:49, 679kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<15:15, 791kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<13:08, 917kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<09:47, 1.23MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:46, 1.37MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:23, 1.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:27, 2.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:36, 1.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:09, 1.66MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:37, 2.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<04:04, 2.91MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<1:27:28, 136kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<1:02:26, 190kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<43:55, 269kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<33:24, 353kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<25:47, 457kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<18:37, 632kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<14:53, 787kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<11:38, 1.01MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<08:24, 1.39MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:35, 1.36MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:13, 1.61MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:17, 2.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:25, 1.80MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:52, 1.68MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<05:23, 2.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:37, 2.05MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<05:09, 2.23MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<03:52, 2.96MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:22, 2.13MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<04:45, 2.40MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<03:35, 3.18MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<02:39, 4.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<47:39, 239kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<34:28, 330kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<24:22, 465kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<19:41, 574kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<16:04, 703kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<11:44, 962kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<08:21, 1.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<11:03, 1.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<08:41, 1.29MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:37, 1.69MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<04:44, 2.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<48:15, 231kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<34:54, 320kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:09<24:39, 452kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<19:50, 560kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<14:49, 748kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<10:36, 1.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<07:33, 1.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<46:55, 235kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<33:57, 325kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<23:59, 459kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<19:20, 567kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<14:38, 748kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<10:30, 1.04MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:54, 1.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:09, 1.19MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<06:53, 1.58MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:17<04:56, 2.19MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<10:49, 1.00MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<08:40, 1.25MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<06:20, 1.70MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:56, 1.55MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<07:04, 1.52MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:24, 1.99MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:59, 2.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:52, 1.82MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<05:11, 2.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:54, 2.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<05:12, 2.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<04:43, 2.24MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<03:34, 2.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<04:59, 2.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:34, 2.30MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<03:25, 3.07MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<04:54, 2.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:29, 2.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:21, 3.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:31<04:49, 2.16MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:30, 1.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<04:18, 2.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<03:08, 3.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<09:22, 1.10MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<07:37, 1.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:35, 1.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<06:18, 1.63MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<06:31, 1.57MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:05, 2.02MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<05:12, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:40, 2.18MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:31, 2.88MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<04:50, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<05:28, 1.85MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<04:16, 2.37MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:05, 3.27MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<19:37, 513kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<14:47, 681kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<10:35, 949kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<09:43, 1.03MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<07:40, 1.30MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<05:38, 1.77MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<04:03, 2.45MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<42:50, 232kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<30:59, 320kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<21:53, 452kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<17:36, 560kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<13:19, 740kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<09:33, 1.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<08:59, 1.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<08:18, 1.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<06:18, 1.55MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<05:58, 1.63MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:11, 1.87MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<03:51, 2.51MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:57, 1.95MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:27, 2.17MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<03:21, 2.87MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<04:36, 2.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:12, 1.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:07, 2.32MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<04:24, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:04, 2.33MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:03, 3.11MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:19, 2.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<04:57, 1.91MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<03:56, 2.39MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<02:50, 3.31MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<1:04:31, 145kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<46:07, 203kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<32:26, 288kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<24:50, 375kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<19:17, 483kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<13:57, 666kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<09:50, 941kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<15:17, 605kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<11:39, 793kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<08:22, 1.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<07:59, 1.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<06:31, 1.41MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<04:47, 1.91MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:29, 1.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:42, 1.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:23, 2.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:08<03:10, 2.85MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<09:44, 928kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<07:43, 1.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<05:37, 1.60MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<06:01, 1.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<05:07, 1.75MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:45, 2.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<04:45, 1.87MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<05:08, 1.73MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:58, 2.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<02:54, 3.05MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<05:38, 1.57MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:51, 1.82MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<03:36, 2.44MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:28, 1.96MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<04:47, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:45, 2.33MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:20<04:03, 2.14MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:39, 2.37MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<02:48, 3.09MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<03:50, 2.25MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<04:31, 1.90MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:33, 2.42MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<02:36, 3.29MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<05:36, 1.52MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<04:47, 1.78MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<03:34, 2.39MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:28, 1.89MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:00, 2.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<03:00, 2.81MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:05, 2.06MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<04:34, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<03:34, 2.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<02:37, 3.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:54, 1.70MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:18, 1.94MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:13, 2.58MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:11, 1.98MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:37, 1.79MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:38, 2.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<03:52, 2.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:33, 2.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<02:41, 3.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:47, 2.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:28, 2.34MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<02:38, 3.08MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:44, 2.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:26, 2.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<02:36, 3.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:42, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<03:25, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<02:35, 3.08MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:40, 2.16MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<03:17, 2.41MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<02:29, 3.17MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<03:35, 2.19MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<04:07, 1.91MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<03:16, 2.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<02:22, 3.30MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<7:33:03, 17.2kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<5:17:33, 24.5kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<3:41:45, 35.0kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<2:36:19, 49.5kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<1:50:59, 69.6kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<1:17:54, 99.1kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<54:21, 141kB/s]   .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<42:20, 181kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<30:23, 252kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<21:22, 357kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<16:41, 455kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<13:13, 574kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<09:37, 788kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<06:46, 1.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<7:20:26, 17.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<5:08:49, 24.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<3:35:37, 34.8kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<2:31:57, 49.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<1:47:51, 69.1kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<1:15:44, 98.3kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<53:50, 137kB/s]   .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<38:18, 193kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<27:04, 272kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<18:54, 387kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<42:42, 171kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<31:22, 233kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<22:14, 329kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<15:37, 466kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<13:34, 534kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<10:15, 707kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<07:19, 987kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<06:45, 1.06MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<06:11, 1.16MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<04:38, 1.54MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<03:19, 2.14MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<06:06, 1.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:01, 1.42MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<03:40, 1.92MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:13, 1.67MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:24, 1.60MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:23, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:07<02:29, 2.81MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:55, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<03:27, 2.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:34, 2.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:09<01:53, 3.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<28:44, 240kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<20:47, 332kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<14:40, 468kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<11:50, 578kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<08:59, 761kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<06:26, 1.06MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<06:05, 1.11MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:56, 1.37MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<03:37, 1.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:06, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<04:14, 1.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:15, 2.05MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<02:22, 2.80MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<04:11, 1.58MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:37, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:40, 2.47MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:23, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:43, 1.76MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:56, 2.23MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:07, 2.08MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<03:30, 1.85MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<02:45, 2.36MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<02:01, 3.19MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:40, 1.75MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<03:13, 1.99MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:23, 2.68MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<03:09, 2.01MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<03:31, 1.81MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<02:43, 2.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:00, 3.14MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:29, 1.80MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:05, 2.03MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:19, 2.70MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<03:04, 2.03MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<02:46, 2.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:05, 2.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:55, 2.11MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<02:40, 2.30MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:00, 3.04MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:51, 2.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:36, 2.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<01:58, 3.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:47, 2.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<02:34, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<01:56, 3.08MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<02:45, 2.15MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<02:32, 2.34MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<01:53, 3.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<02:43, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<03:06, 1.89MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<02:25, 2.42MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<01:46, 3.29MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<03:44, 1.55MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:13, 1.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:24, 2.41MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:44<03:00, 1.91MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:16, 1.75MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<02:34, 2.22MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:45<01:51, 3.06MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<5:28:07, 17.3kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<3:50:00, 24.6kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<2:40:25, 35.2kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<1:52:52, 49.7kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<1:20:07, 70.0kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<56:14, 99.4kB/s]  .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<39:53, 139kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<28:27, 194kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<19:57, 276kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<15:09, 361kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<11:08, 490kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<07:54, 688kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<06:46, 797kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<05:50, 924kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<04:20, 1.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<03:03, 1.74MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<19:33, 273kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<14:12, 375kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<10:02, 528kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<08:12, 641kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<06:49, 771kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<05:01, 1.04MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<04:22, 1.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<04:06, 1.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:06, 1.67MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:13, 2.31MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:56, 1.30MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<03:16, 1.56MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<02:24, 2.11MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:51, 1.77MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:02, 1.66MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:21, 2.14MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:41, 2.94MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<37:04, 135kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<26:26, 189kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:06<18:32, 267kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<14:02, 351kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<10:18, 477kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<07:18, 669kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<06:13, 779kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<04:50, 1.00MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<03:29, 1.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<03:33, 1.34MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:58, 1.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:10, 2.19MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:38, 1.79MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:48, 1.68MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:10, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<01:34, 2.97MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<03:30, 1.33MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:55, 1.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:08, 2.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<02:33, 1.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:43, 1.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:08, 2.14MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:12, 2.04MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:00, 2.24MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:31, 2.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:05, 2.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:23, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:51, 2.38MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:22, 3.21MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:22, 1.85MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:06, 2.07MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:34, 2.75MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:06, 2.04MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:21, 1.83MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<01:49, 2.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:26<01:18, 3.23MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<05:32, 764kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<04:19, 978kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<03:06, 1.35MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<03:08, 1.33MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:37, 1.58MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:56, 2.14MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:18, 1.78MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<02:27, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:54, 2.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:22, 2.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<05:14, 768kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<04:05, 983kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:57, 1.35MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<02:58, 1.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:53, 1.37MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<02:13, 1.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:09, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<01:54, 2.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:25, 2.70MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:53, 2.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<01:42, 2.23MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:17, 2.95MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:46, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:37, 2.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:13, 3.03MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:43, 2.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<01:58, 1.87MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:32, 2.39MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<01:07, 3.23MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:01, 1.79MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:47, 2.01MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:19, 2.71MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:45, 2.02MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:57, 1.82MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:32, 2.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:37, 2.14MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<01:29, 2.32MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:07, 3.05MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:34, 2.16MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:48, 1.89MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<01:25, 2.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:31, 2.19MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<01:24, 2.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:03, 3.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:30, 2.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:23, 2.36MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:01, 3.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:28, 2.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:41, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:18, 2.43MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<00:57, 3.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:01, 1.54MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<01:44, 1.80MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:17, 2.41MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:36, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:26, 2.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:04, 2.82MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:26, 2.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:18, 2.28MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<00:59, 3.00MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:22, 2.13MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:15, 2.32MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<00:56, 3.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:20, 2.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:30, 1.89MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:10, 2.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<00:51, 3.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:52, 1.49MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:36, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:10, 2.35MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:27, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:34, 1.73MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:13, 2.23MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<00:52, 3.06MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<02:34, 1.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:04, 1.28MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:30, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:38, 1.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:40, 1.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:16, 2.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:54, 2.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:24, 1.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:56, 1.30MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:23, 1.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:32, 1.59MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:35, 1.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:13, 1.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:52, 2.74MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<2:18:16, 17.2kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<1:36:46, 24.6kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<1:07:03, 35.1kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<46:45, 49.5kB/s]  .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<33:09, 69.8kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<23:10, 99.2kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<16:02, 141kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<11:57, 188kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<08:34, 261kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<05:58, 370kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<04:38, 470kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<03:41, 590kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<02:39, 814kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<01:51, 1.14MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<02:08, 982kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:43, 1.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<01:14, 1.67MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:20, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:08, 1.79MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:49, 2.42MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:02, 1.89MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<01:07, 1.75MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:53, 2.21MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<00:54, 2.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:49, 2.28MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:37, 3.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:51, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:47, 2.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:34, 3.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:51, 2.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<00:57, 1.83MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:45, 2.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:31, 3.19MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<07:27, 228kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<05:22, 315kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<03:44, 445kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:56, 553kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<02:13, 731kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:34, 1.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:26, 1.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<01:19, 1.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<00:59, 1.55MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:41, 2.17MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<03:13, 463kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<02:24, 618kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:41, 866kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:29, 959kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:19, 1.07MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:59, 1.42MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:41, 1.98MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<1:18:56, 17.2kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<55:07, 24.5kB/s]  .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<37:53, 34.9kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<26:05, 49.3kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<18:18, 70.0kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<12:34, 99.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<08:50, 138kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<06:23, 190kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<04:29, 268kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<03:04, 381kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<02:34, 448kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:54, 601kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<01:19, 839kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:09, 936kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:01, 1.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:45, 1.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:40, 1.51MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:34, 1.76MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:24, 2.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:30, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:32, 1.73MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:25, 2.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:17, 3.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<08:43, 100kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<06:08, 142kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<04:12, 201kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<02:50, 287kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<03:29, 231kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<02:30, 319kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<01:43, 451kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:19, 559kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:59, 738kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:41, 1.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:27, 1.45MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<05:19, 126kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<03:50, 174kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<02:40, 245kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<01:49, 347kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<01:21, 445kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:59, 596kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:41, 832kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:34, 930kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:30, 1.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:22, 1.38MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:18, 1.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:11, 2.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:12, 1.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:10, 2.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:09, 2.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:08, 2.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:05, 3.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:07, 2.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:06, 2.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:04, 3.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 1.89MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:03, 2.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:02, 2.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:01, 3.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.19MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.36MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 3.15MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 882/400000 [00:00<00:45, 8812.40it/s]  0%|          | 1724/400000 [00:00<00:45, 8690.43it/s]  1%|          | 2651/400000 [00:00<00:44, 8855.95it/s]  1%|          | 3567/400000 [00:00<00:44, 8943.47it/s]  1%|          | 4498/400000 [00:00<00:43, 9048.05it/s]  1%|▏         | 5390/400000 [00:00<00:43, 9007.23it/s]  2%|▏         | 6253/400000 [00:00<00:44, 8889.83it/s]  2%|▏         | 7182/400000 [00:00<00:43, 9004.76it/s]  2%|▏         | 8198/400000 [00:00<00:42, 9321.38it/s]  2%|▏         | 9145/400000 [00:01<00:41, 9364.00it/s]  3%|▎         | 10104/400000 [00:01<00:41, 9428.53it/s]  3%|▎         | 11051/400000 [00:01<00:41, 9437.08it/s]  3%|▎         | 12062/400000 [00:01<00:40, 9627.99it/s]  3%|▎         | 13045/400000 [00:01<00:39, 9686.51it/s]  4%|▎         | 14044/400000 [00:01<00:39, 9775.25it/s]  4%|▍         | 15019/400000 [00:01<00:40, 9434.00it/s]  4%|▍         | 15997/400000 [00:01<00:40, 9533.32it/s]  4%|▍         | 16995/400000 [00:01<00:39, 9660.72it/s]  4%|▍         | 17996/400000 [00:01<00:39, 9761.08it/s]  5%|▍         | 18973/400000 [00:02<00:39, 9691.40it/s]  5%|▍         | 19943/400000 [00:02<00:39, 9660.96it/s]  5%|▌         | 20910/400000 [00:02<00:39, 9612.60it/s]  5%|▌         | 21917/400000 [00:02<00:38, 9745.02it/s]  6%|▌         | 22893/400000 [00:02<00:38, 9701.36it/s]  6%|▌         | 23881/400000 [00:02<00:38, 9754.14it/s]  6%|▌         | 24857/400000 [00:02<00:38, 9683.77it/s]  6%|▋         | 25826/400000 [00:02<00:38, 9673.59it/s]  7%|▋         | 26839/400000 [00:02<00:38, 9804.80it/s]  7%|▋         | 27846/400000 [00:02<00:37, 9881.16it/s]  7%|▋         | 28859/400000 [00:03<00:37, 9954.17it/s]  7%|▋         | 29855/400000 [00:03<00:38, 9716.70it/s]  8%|▊         | 30843/400000 [00:03<00:37, 9764.43it/s]  8%|▊         | 31839/400000 [00:03<00:37, 9820.45it/s]  8%|▊         | 32823/400000 [00:03<00:37, 9825.34it/s]  8%|▊         | 33816/400000 [00:03<00:37, 9855.45it/s]  9%|▊         | 34803/400000 [00:03<00:37, 9772.26it/s]  9%|▉         | 35781/400000 [00:03<00:38, 9579.17it/s]  9%|▉         | 36777/400000 [00:03<00:37, 9689.58it/s]  9%|▉         | 37748/400000 [00:03<00:38, 9518.81it/s] 10%|▉         | 38731/400000 [00:04<00:37, 9608.05it/s] 10%|▉         | 39694/400000 [00:04<00:37, 9613.38it/s] 10%|█         | 40657/400000 [00:04<00:37, 9585.00it/s] 10%|█         | 41650/400000 [00:04<00:37, 9684.14it/s] 11%|█         | 42647/400000 [00:04<00:36, 9765.39it/s] 11%|█         | 43646/400000 [00:04<00:36, 9830.93it/s] 11%|█         | 44630/400000 [00:04<00:36, 9740.95it/s] 11%|█▏        | 45605/400000 [00:04<00:36, 9610.11it/s] 12%|█▏        | 46612/400000 [00:04<00:36, 9740.88it/s] 12%|█▏        | 47606/400000 [00:04<00:35, 9798.96it/s] 12%|█▏        | 48608/400000 [00:05<00:35, 9863.83it/s] 12%|█▏        | 49596/400000 [00:05<00:36, 9685.19it/s] 13%|█▎        | 50566/400000 [00:05<00:36, 9548.11it/s] 13%|█▎        | 51523/400000 [00:05<00:36, 9487.11it/s] 13%|█▎        | 52473/400000 [00:05<00:36, 9489.82it/s] 13%|█▎        | 53448/400000 [00:05<00:36, 9566.39it/s] 14%|█▎        | 54418/400000 [00:05<00:35, 9605.00it/s] 14%|█▍        | 55379/400000 [00:05<00:36, 9484.40it/s] 14%|█▍        | 56356/400000 [00:05<00:35, 9566.42it/s] 14%|█▍        | 57365/400000 [00:05<00:35, 9713.60it/s] 15%|█▍        | 58338/400000 [00:06<00:35, 9579.24it/s] 15%|█▍        | 59300/400000 [00:06<00:35, 9589.13it/s] 15%|█▌        | 60260/400000 [00:06<00:35, 9483.70it/s] 15%|█▌        | 61268/400000 [00:06<00:35, 9654.11it/s] 16%|█▌        | 62260/400000 [00:06<00:34, 9731.89it/s] 16%|█▌        | 63270/400000 [00:06<00:34, 9838.67it/s] 16%|█▌        | 64255/400000 [00:06<00:34, 9787.19it/s] 16%|█▋        | 65251/400000 [00:06<00:34, 9836.03it/s] 17%|█▋        | 66269/400000 [00:06<00:33, 9934.25it/s] 17%|█▋        | 67280/400000 [00:06<00:33, 9984.83it/s] 17%|█▋        | 68280/400000 [00:07<00:33, 9914.50it/s] 17%|█▋        | 69272/400000 [00:07<00:34, 9574.57it/s] 18%|█▊        | 70244/400000 [00:07<00:34, 9615.78it/s] 18%|█▊        | 71213/400000 [00:07<00:34, 9637.79it/s] 18%|█▊        | 72221/400000 [00:07<00:33, 9763.83it/s] 18%|█▊        | 73223/400000 [00:07<00:33, 9837.64it/s] 19%|█▊        | 74208/400000 [00:07<00:33, 9748.15it/s] 19%|█▉        | 75190/400000 [00:07<00:33, 9768.48it/s] 19%|█▉        | 76168/400000 [00:07<00:33, 9656.32it/s] 19%|█▉        | 77150/400000 [00:08<00:33, 9702.81it/s] 20%|█▉        | 78141/400000 [00:08<00:32, 9763.28it/s] 20%|█▉        | 79118/400000 [00:08<00:33, 9597.95it/s] 20%|██        | 80085/400000 [00:08<00:33, 9617.72it/s] 20%|██        | 81048/400000 [00:08<00:33, 9619.16it/s] 21%|██        | 82011/400000 [00:08<00:33, 9523.52it/s] 21%|██        | 82964/400000 [00:08<00:35, 8894.19it/s] 21%|██        | 83863/400000 [00:08<00:36, 8705.38it/s] 21%|██        | 84761/400000 [00:08<00:35, 8785.36it/s] 21%|██▏       | 85645/400000 [00:08<00:35, 8782.56it/s] 22%|██▏       | 86531/400000 [00:09<00:35, 8804.78it/s] 22%|██▏       | 87423/400000 [00:09<00:35, 8837.78it/s] 22%|██▏       | 88309/400000 [00:09<00:35, 8701.13it/s] 22%|██▏       | 89201/400000 [00:09<00:35, 8764.25it/s] 23%|██▎       | 90079/400000 [00:09<00:35, 8692.26it/s] 23%|██▎       | 90950/400000 [00:09<00:35, 8668.09it/s] 23%|██▎       | 91818/400000 [00:09<00:35, 8627.90it/s] 23%|██▎       | 92687/400000 [00:09<00:35, 8644.95it/s] 23%|██▎       | 93682/400000 [00:09<00:34, 8996.35it/s] 24%|██▎       | 94621/400000 [00:09<00:33, 9110.72it/s] 24%|██▍       | 95536/400000 [00:10<00:33, 8973.36it/s] 24%|██▍       | 96436/400000 [00:10<00:34, 8889.67it/s] 24%|██▍       | 97328/400000 [00:10<00:34, 8684.34it/s] 25%|██▍       | 98199/400000 [00:10<00:34, 8683.94it/s] 25%|██▍       | 99070/400000 [00:10<00:35, 8562.37it/s] 25%|██▍       | 99928/400000 [00:10<00:35, 8397.80it/s] 25%|██▌       | 100770/400000 [00:10<00:36, 8304.95it/s] 25%|██▌       | 101603/400000 [00:10<00:37, 8057.03it/s] 26%|██▌       | 102451/400000 [00:10<00:36, 8176.44it/s] 26%|██▌       | 103281/400000 [00:11<00:36, 8213.03it/s] 26%|██▌       | 104104/400000 [00:11<00:36, 8170.12it/s] 26%|██▌       | 104951/400000 [00:11<00:35, 8256.52it/s] 26%|██▋       | 105780/400000 [00:11<00:35, 8263.24it/s] 27%|██▋       | 106634/400000 [00:11<00:35, 8342.22it/s] 27%|██▋       | 107637/400000 [00:11<00:33, 8785.53it/s] 27%|██▋       | 108602/400000 [00:11<00:32, 9027.15it/s] 27%|██▋       | 109530/400000 [00:11<00:31, 9099.44it/s] 28%|██▊       | 110445/400000 [00:11<00:32, 8974.67it/s] 28%|██▊       | 111373/400000 [00:11<00:31, 9063.31it/s] 28%|██▊       | 112305/400000 [00:12<00:31, 9136.27it/s] 28%|██▊       | 113259/400000 [00:12<00:30, 9253.40it/s] 29%|██▊       | 114196/400000 [00:12<00:30, 9286.74it/s] 29%|██▉       | 115126/400000 [00:12<00:31, 9091.75it/s] 29%|██▉       | 116046/400000 [00:12<00:31, 9122.99it/s] 29%|██▉       | 116981/400000 [00:12<00:30, 9188.56it/s] 29%|██▉       | 117901/400000 [00:12<00:31, 8841.76it/s] 30%|██▉       | 118841/400000 [00:12<00:31, 8997.30it/s] 30%|██▉       | 119744/400000 [00:12<00:31, 9005.09it/s] 30%|███       | 120703/400000 [00:12<00:30, 9171.44it/s] 30%|███       | 121623/400000 [00:13<00:30, 9098.69it/s] 31%|███       | 122535/400000 [00:13<00:30, 9068.48it/s] 31%|███       | 123448/400000 [00:13<00:30, 9084.85it/s] 31%|███       | 124358/400000 [00:13<00:30, 8900.11it/s] 31%|███▏      | 125250/400000 [00:13<00:31, 8853.02it/s] 32%|███▏      | 126147/400000 [00:13<00:30, 8887.24it/s] 32%|███▏      | 127050/400000 [00:13<00:30, 8925.67it/s] 32%|███▏      | 127965/400000 [00:13<00:30, 8990.49it/s] 32%|███▏      | 128865/400000 [00:13<00:30, 8823.80it/s] 32%|███▏      | 129799/400000 [00:13<00:30, 8970.71it/s] 33%|███▎      | 130733/400000 [00:14<00:29, 9077.18it/s] 33%|███▎      | 131642/400000 [00:14<00:29, 8980.15it/s] 33%|███▎      | 132542/400000 [00:14<00:30, 8739.14it/s] 33%|███▎      | 133419/400000 [00:14<00:30, 8675.59it/s] 34%|███▎      | 134333/400000 [00:14<00:30, 8809.52it/s] 34%|███▍      | 135216/400000 [00:14<00:30, 8673.02it/s] 34%|███▍      | 136085/400000 [00:14<00:30, 8660.90it/s] 34%|███▍      | 136995/400000 [00:14<00:29, 8786.78it/s] 34%|███▍      | 137875/400000 [00:14<00:30, 8671.19it/s] 35%|███▍      | 138784/400000 [00:14<00:29, 8791.62it/s] 35%|███▍      | 139690/400000 [00:15<00:29, 8868.11it/s] 35%|███▌      | 140603/400000 [00:15<00:29, 8942.43it/s] 35%|███▌      | 141499/400000 [00:15<00:29, 8913.79it/s] 36%|███▌      | 142392/400000 [00:15<00:29, 8739.21it/s] 36%|███▌      | 143309/400000 [00:15<00:28, 8861.12it/s] 36%|███▌      | 144272/400000 [00:15<00:28, 9076.06it/s] 36%|███▋      | 145229/400000 [00:15<00:27, 9218.65it/s] 37%|███▋      | 146153/400000 [00:15<00:28, 9059.90it/s] 37%|███▋      | 147062/400000 [00:15<00:28, 8950.46it/s] 37%|███▋      | 148018/400000 [00:16<00:27, 9122.66it/s] 37%|███▋      | 148985/400000 [00:16<00:27, 9279.94it/s] 37%|███▋      | 149924/400000 [00:16<00:26, 9310.73it/s] 38%|███▊      | 150871/400000 [00:16<00:26, 9354.64it/s] 38%|███▊      | 151808/400000 [00:16<00:26, 9273.29it/s] 38%|███▊      | 152790/400000 [00:16<00:26, 9429.29it/s] 38%|███▊      | 153787/400000 [00:16<00:25, 9583.07it/s] 39%|███▊      | 154766/400000 [00:16<00:25, 9643.38it/s] 39%|███▉      | 155774/400000 [00:16<00:25, 9768.62it/s] 39%|███▉      | 156753/400000 [00:16<00:25, 9669.18it/s] 39%|███▉      | 157725/400000 [00:17<00:25, 9683.81it/s] 40%|███▉      | 158725/400000 [00:17<00:24, 9775.33it/s] 40%|███▉      | 159721/400000 [00:17<00:24, 9829.13it/s] 40%|████      | 160705/400000 [00:17<00:24, 9696.47it/s] 40%|████      | 161676/400000 [00:17<00:25, 9268.38it/s] 41%|████      | 162674/400000 [00:17<00:25, 9469.85it/s] 41%|████      | 163645/400000 [00:17<00:24, 9538.59it/s] 41%|████      | 164633/400000 [00:17<00:24, 9636.77it/s] 41%|████▏     | 165647/400000 [00:17<00:23, 9780.15it/s] 42%|████▏     | 166628/400000 [00:17<00:24, 9721.11it/s] 42%|████▏     | 167650/400000 [00:18<00:23, 9865.27it/s] 42%|████▏     | 168671/400000 [00:18<00:23, 9964.83it/s] 42%|████▏     | 169669/400000 [00:18<00:23, 9918.63it/s] 43%|████▎     | 170662/400000 [00:18<00:23, 9893.71it/s] 43%|████▎     | 171653/400000 [00:18<00:23, 9686.62it/s] 43%|████▎     | 172666/400000 [00:18<00:23, 9814.97it/s] 43%|████▎     | 173677/400000 [00:18<00:22, 9899.48it/s] 44%|████▎     | 174672/400000 [00:18<00:22, 9913.18it/s] 44%|████▍     | 175688/400000 [00:18<00:22, 9985.33it/s] 44%|████▍     | 176688/400000 [00:18<00:22, 9806.90it/s] 44%|████▍     | 177670/400000 [00:19<00:22, 9678.81it/s] 45%|████▍     | 178671/400000 [00:19<00:22, 9773.60it/s] 45%|████▍     | 179682/400000 [00:19<00:22, 9869.91it/s] 45%|████▌     | 180707/400000 [00:19<00:21, 9979.31it/s] 45%|████▌     | 181706/400000 [00:19<00:22, 9827.69it/s] 46%|████▌     | 182710/400000 [00:19<00:21, 9889.66it/s] 46%|████▌     | 183700/400000 [00:19<00:22, 9761.32it/s] 46%|████▌     | 184717/400000 [00:19<00:21, 9878.00it/s] 46%|████▋     | 185728/400000 [00:19<00:21, 9945.88it/s] 47%|████▋     | 186724/400000 [00:19<00:21, 9828.06it/s] 47%|████▋     | 187716/400000 [00:20<00:21, 9855.05it/s] 47%|████▋     | 188745/400000 [00:20<00:21, 9981.22it/s] 47%|████▋     | 189744/400000 [00:20<00:21, 9981.49it/s] 48%|████▊     | 190774/400000 [00:20<00:20, 10072.16it/s] 48%|████▊     | 191782/400000 [00:20<00:21, 9742.91it/s]  48%|████▊     | 192760/400000 [00:20<00:21, 9473.98it/s] 48%|████▊     | 193768/400000 [00:20<00:21, 9646.81it/s] 49%|████▊     | 194785/400000 [00:20<00:20, 9796.65it/s] 49%|████▉     | 195768/400000 [00:20<00:20, 9748.94it/s] 49%|████▉     | 196745/400000 [00:20<00:21, 9527.79it/s] 49%|████▉     | 197701/400000 [00:21<00:21, 9535.59it/s] 50%|████▉     | 198669/400000 [00:21<00:21, 9575.88it/s] 50%|████▉     | 199628/400000 [00:21<00:21, 9540.78it/s] 50%|█████     | 200619/400000 [00:21<00:20, 9646.28it/s] 50%|█████     | 201585/400000 [00:21<00:20, 9622.17it/s] 51%|█████     | 202555/400000 [00:21<00:20, 9643.47it/s] 51%|█████     | 203573/400000 [00:21<00:20, 9797.78it/s] 51%|█████     | 204583/400000 [00:21<00:19, 9884.30it/s] 51%|█████▏    | 205590/400000 [00:21<00:19, 9938.11it/s] 52%|█████▏    | 206585/400000 [00:22<00:19, 9794.18it/s] 52%|█████▏    | 207580/400000 [00:22<00:19, 9837.79it/s] 52%|█████▏    | 208565/400000 [00:22<00:19, 9756.79it/s] 52%|█████▏    | 209542/400000 [00:22<00:19, 9719.13it/s] 53%|█████▎    | 210548/400000 [00:22<00:19, 9818.67it/s] 53%|█████▎    | 211531/400000 [00:22<00:19, 9742.44it/s] 53%|█████▎    | 212521/400000 [00:22<00:19, 9786.23it/s] 53%|█████▎    | 213537/400000 [00:22<00:18, 9895.18it/s] 54%|█████▎    | 214545/400000 [00:22<00:18, 9948.63it/s] 54%|█████▍    | 215543/400000 [00:22<00:18, 9957.29it/s] 54%|█████▍    | 216540/400000 [00:23<00:18, 9764.23it/s] 54%|█████▍    | 217536/400000 [00:23<00:18, 9821.49it/s] 55%|█████▍    | 218540/400000 [00:23<00:18, 9884.03it/s] 55%|█████▍    | 219566/400000 [00:23<00:18, 9993.54it/s] 55%|█████▌    | 220588/400000 [00:23<00:17, 10059.78it/s] 55%|█████▌    | 221595/400000 [00:23<00:18, 9773.02it/s]  56%|█████▌    | 222599/400000 [00:23<00:18, 9850.75it/s] 56%|█████▌    | 223586/400000 [00:23<00:18, 9561.06it/s] 56%|█████▌    | 224582/400000 [00:23<00:18, 9676.04it/s] 56%|█████▋    | 225601/400000 [00:23<00:17, 9824.24it/s] 57%|█████▋    | 226586/400000 [00:24<00:17, 9771.61it/s] 57%|█████▋    | 227592/400000 [00:24<00:17, 9854.70it/s] 57%|█████▋    | 228579/400000 [00:24<00:17, 9779.02it/s] 57%|█████▋    | 229594/400000 [00:24<00:17, 9886.33it/s] 58%|█████▊    | 230614/400000 [00:24<00:16, 9975.83it/s] 58%|█████▊    | 231613/400000 [00:24<00:17, 9822.30it/s] 58%|█████▊    | 232597/400000 [00:24<00:17, 9695.91it/s] 58%|█████▊    | 233586/400000 [00:24<00:17, 9751.31it/s] 59%|█████▊    | 234580/400000 [00:24<00:16, 9805.54it/s] 59%|█████▉    | 235582/400000 [00:24<00:16, 9866.62it/s] 59%|█████▉    | 236570/400000 [00:25<00:16, 9778.72it/s] 59%|█████▉    | 237593/400000 [00:25<00:16, 9907.46it/s] 60%|█████▉    | 238585/400000 [00:25<00:16, 9731.33it/s] 60%|█████▉    | 239587/400000 [00:25<00:16, 9813.85it/s] 60%|██████    | 240597/400000 [00:25<00:16, 9895.50it/s] 60%|██████    | 241588/400000 [00:25<00:16, 9778.60it/s] 61%|██████    | 242567/400000 [00:25<00:16, 9693.16it/s] 61%|██████    | 243538/400000 [00:25<00:16, 9663.04it/s] 61%|██████    | 244505/400000 [00:25<00:16, 9648.49it/s] 61%|██████▏   | 245471/400000 [00:25<00:16, 9649.57it/s] 62%|██████▏   | 246437/400000 [00:26<00:16, 9212.67it/s] 62%|██████▏   | 247363/400000 [00:26<00:16, 9117.42it/s] 62%|██████▏   | 248279/400000 [00:26<00:16, 9041.82it/s] 62%|██████▏   | 249229/400000 [00:26<00:16, 9172.29it/s] 63%|██████▎   | 250149/400000 [00:26<00:16, 9069.12it/s] 63%|██████▎   | 251074/400000 [00:26<00:16, 9120.80it/s] 63%|██████▎   | 252081/400000 [00:26<00:15, 9385.65it/s] 63%|██████▎   | 253087/400000 [00:26<00:15, 9575.12it/s] 64%|██████▎   | 254048/400000 [00:26<00:15, 9365.82it/s] 64%|██████▍   | 255029/400000 [00:27<00:15, 9493.78it/s] 64%|██████▍   | 256007/400000 [00:27<00:15, 9575.62it/s] 64%|██████▍   | 257007/400000 [00:27<00:14, 9698.22it/s] 64%|██████▍   | 257979/400000 [00:27<00:14, 9667.57it/s] 65%|██████▍   | 258948/400000 [00:27<00:14, 9619.37it/s] 65%|██████▍   | 259911/400000 [00:27<00:14, 9356.42it/s] 65%|██████▌   | 260849/400000 [00:27<00:14, 9332.43it/s] 65%|██████▌   | 261800/400000 [00:27<00:14, 9384.43it/s] 66%|██████▌   | 262740/400000 [00:27<00:14, 9336.74it/s] 66%|██████▌   | 263675/400000 [00:27<00:14, 9278.29it/s] 66%|██████▌   | 264604/400000 [00:28<00:14, 9196.17it/s] 66%|██████▋   | 265533/400000 [00:28<00:14, 9223.24it/s] 67%|██████▋   | 266474/400000 [00:28<00:14, 9276.08it/s] 67%|██████▋   | 267443/400000 [00:28<00:14, 9394.85it/s] 67%|██████▋   | 268384/400000 [00:28<00:14, 9391.89it/s] 67%|██████▋   | 269324/400000 [00:28<00:14, 9174.89it/s] 68%|██████▊   | 270243/400000 [00:28<00:14, 9156.79it/s] 68%|██████▊   | 271247/400000 [00:28<00:13, 9402.65it/s] 68%|██████▊   | 272249/400000 [00:28<00:13, 9578.92it/s] 68%|██████▊   | 273267/400000 [00:28<00:12, 9751.36it/s] 69%|██████▊   | 274245/400000 [00:29<00:12, 9731.88it/s] 69%|██████▉   | 275225/400000 [00:29<00:12, 9750.48it/s] 69%|██████▉   | 276225/400000 [00:29<00:12, 9822.79it/s] 69%|██████▉   | 277238/400000 [00:29<00:12, 9912.47it/s] 70%|██████▉   | 278261/400000 [00:29<00:12, 10003.27it/s] 70%|██████▉   | 279278/400000 [00:29<00:12, 10050.85it/s] 70%|███████   | 280284/400000 [00:29<00:12, 9610.08it/s]  70%|███████   | 281299/400000 [00:29<00:12, 9764.45it/s] 71%|███████   | 282296/400000 [00:29<00:11, 9824.30it/s] 71%|███████   | 283305/400000 [00:29<00:11, 9901.22it/s] 71%|███████   | 284298/400000 [00:30<00:12, 9565.92it/s] 71%|███████▏  | 285259/400000 [00:30<00:11, 9567.84it/s] 72%|███████▏  | 286242/400000 [00:30<00:11, 9643.45it/s] 72%|███████▏  | 287209/400000 [00:30<00:11, 9648.54it/s] 72%|███████▏  | 288178/400000 [00:30<00:11, 9659.58it/s] 72%|███████▏  | 289145/400000 [00:30<00:11, 9583.49it/s] 73%|███████▎  | 290105/400000 [00:30<00:11, 9218.46it/s] 73%|███████▎  | 291035/400000 [00:30<00:11, 9240.52it/s] 73%|███████▎  | 291969/400000 [00:30<00:11, 9268.95it/s] 73%|███████▎  | 292957/400000 [00:30<00:11, 9442.26it/s] 73%|███████▎  | 293904/400000 [00:31<00:11, 9382.04it/s] 74%|███████▎  | 294844/400000 [00:31<00:11, 9304.20it/s] 74%|███████▍  | 295776/400000 [00:31<00:11, 9081.68it/s] 74%|███████▍  | 296687/400000 [00:31<00:11, 9057.05it/s] 74%|███████▍  | 297686/400000 [00:31<00:10, 9316.56it/s] 75%|███████▍  | 298621/400000 [00:31<00:11, 9206.45it/s] 75%|███████▍  | 299544/400000 [00:31<00:11, 9106.74it/s] 75%|███████▌  | 300516/400000 [00:31<00:10, 9281.41it/s] 75%|███████▌  | 301479/400000 [00:31<00:10, 9381.37it/s] 76%|███████▌  | 302439/400000 [00:32<00:10, 9443.48it/s] 76%|███████▌  | 303420/400000 [00:32<00:10, 9548.93it/s] 76%|███████▌  | 304377/400000 [00:32<00:10, 9413.50it/s] 76%|███████▋  | 305325/400000 [00:32<00:10, 9431.57it/s] 77%|███████▋  | 306306/400000 [00:32<00:09, 9539.33it/s] 77%|███████▋  | 307307/400000 [00:32<00:09, 9675.62it/s] 77%|███████▋  | 308276/400000 [00:32<00:09, 9639.51it/s] 77%|███████▋  | 309241/400000 [00:32<00:09, 9252.16it/s] 78%|███████▊  | 310182/400000 [00:32<00:09, 9297.41it/s] 78%|███████▊  | 311187/400000 [00:32<00:09, 9508.78it/s] 78%|███████▊  | 312142/400000 [00:33<00:09, 9463.62it/s] 78%|███████▊  | 313091/400000 [00:33<00:09, 9458.37it/s] 79%|███████▊  | 314039/400000 [00:33<00:09, 9116.01it/s] 79%|███████▉  | 315021/400000 [00:33<00:09, 9315.61it/s] 79%|███████▉  | 315957/400000 [00:33<00:09, 9208.39it/s] 79%|███████▉  | 316881/400000 [00:33<00:09, 9179.37it/s] 79%|███████▉  | 317801/400000 [00:33<00:09, 9132.15it/s] 80%|███████▉  | 318745/400000 [00:33<00:08, 9220.88it/s] 80%|███████▉  | 319725/400000 [00:33<00:08, 9385.08it/s] 80%|████████  | 320682/400000 [00:33<00:08, 9437.91it/s] 80%|████████  | 321627/400000 [00:34<00:08, 9409.07it/s] 81%|████████  | 322569/400000 [00:34<00:08, 9235.44it/s] 81%|████████  | 323494/400000 [00:34<00:08, 9225.67it/s] 81%|████████  | 324469/400000 [00:34<00:08, 9376.17it/s] 81%|████████▏ | 325456/400000 [00:34<00:07, 9517.00it/s] 82%|████████▏ | 326420/400000 [00:34<00:07, 9551.71it/s] 82%|████████▏ | 327377/400000 [00:34<00:07, 9356.26it/s] 82%|████████▏ | 328315/400000 [00:34<00:07, 9242.46it/s] 82%|████████▏ | 329277/400000 [00:34<00:07, 9348.71it/s] 83%|████████▎ | 330217/400000 [00:34<00:07, 9362.98it/s] 83%|████████▎ | 331187/400000 [00:35<00:07, 9460.08it/s] 83%|████████▎ | 332134/400000 [00:35<00:07, 9265.98it/s] 83%|████████▎ | 333063/400000 [00:35<00:07, 9059.49it/s] 84%|████████▎ | 334018/400000 [00:35<00:07, 9199.08it/s] 84%|████████▎ | 334968/400000 [00:35<00:07, 9284.97it/s] 84%|████████▍ | 335922/400000 [00:35<00:06, 9358.92it/s] 84%|████████▍ | 336860/400000 [00:35<00:06, 9270.57it/s] 84%|████████▍ | 337789/400000 [00:35<00:06, 9268.38it/s] 85%|████████▍ | 338717/400000 [00:35<00:06, 9267.76it/s] 85%|████████▍ | 339656/400000 [00:35<00:06, 9303.56it/s] 85%|████████▌ | 340595/400000 [00:36<00:06, 9325.98it/s] 85%|████████▌ | 341528/400000 [00:36<00:06, 9239.55it/s] 86%|████████▌ | 342453/400000 [00:36<00:06, 8921.14it/s] 86%|████████▌ | 343348/400000 [00:36<00:06, 8843.77it/s] 86%|████████▌ | 344248/400000 [00:36<00:06, 8889.92it/s] 86%|████████▋ | 345175/400000 [00:36<00:06, 8999.84it/s] 87%|████████▋ | 346077/400000 [00:36<00:06, 8973.41it/s] 87%|████████▋ | 346977/400000 [00:36<00:05, 8980.98it/s] 87%|████████▋ | 347898/400000 [00:36<00:05, 9048.10it/s] 87%|████████▋ | 348812/400000 [00:37<00:05, 9074.80it/s] 87%|████████▋ | 349753/400000 [00:37<00:05, 9169.88it/s] 88%|████████▊ | 350671/400000 [00:37<00:05, 9038.86it/s] 88%|████████▊ | 351576/400000 [00:37<00:05, 9015.60it/s] 88%|████████▊ | 352495/400000 [00:37<00:05, 9064.59it/s] 88%|████████▊ | 353402/400000 [00:37<00:05, 9038.61it/s] 89%|████████▊ | 354348/400000 [00:37<00:04, 9158.56it/s] 89%|████████▉ | 355279/400000 [00:37<00:04, 9200.34it/s] 89%|████████▉ | 356200/400000 [00:37<00:04, 9107.31it/s] 89%|████████▉ | 357112/400000 [00:37<00:04, 8893.25it/s] 90%|████████▉ | 358061/400000 [00:38<00:04, 9062.14it/s] 90%|████████▉ | 359015/400000 [00:38<00:04, 9199.01it/s] 90%|████████▉ | 359937/400000 [00:38<00:04, 9188.36it/s] 90%|█████████ | 360858/400000 [00:38<00:04, 9062.29it/s] 90%|█████████ | 361800/400000 [00:38<00:04, 9166.38it/s] 91%|█████████ | 362748/400000 [00:38<00:04, 9255.05it/s] 91%|█████████ | 363675/400000 [00:38<00:03, 9104.13it/s] 91%|█████████ | 364587/400000 [00:38<00:03, 8917.48it/s] 91%|█████████▏| 365481/400000 [00:38<00:03, 8855.38it/s] 92%|█████████▏| 366419/400000 [00:38<00:03, 9005.39it/s] 92%|█████████▏| 367376/400000 [00:39<00:03, 9164.36it/s] 92%|█████████▏| 368309/400000 [00:39<00:03, 9211.84it/s] 92%|█████████▏| 369232/400000 [00:39<00:03, 9162.28it/s] 93%|█████████▎| 370150/400000 [00:39<00:03, 9045.70it/s] 93%|█████████▎| 371056/400000 [00:39<00:03, 8793.07it/s] 93%|█████████▎| 371975/400000 [00:39<00:03, 8907.68it/s] 93%|█████████▎| 372901/400000 [00:39<00:03, 9009.11it/s] 93%|█████████▎| 373826/400000 [00:39<00:02, 9078.09it/s] 94%|█████████▎| 374736/400000 [00:39<00:02, 9063.33it/s] 94%|█████████▍| 375683/400000 [00:39<00:02, 9180.79it/s] 94%|█████████▍| 376619/400000 [00:40<00:02, 9232.83it/s] 94%|█████████▍| 377545/400000 [00:40<00:02, 9236.43it/s] 95%|█████████▍| 378470/400000 [00:40<00:02, 9190.51it/s] 95%|█████████▍| 379390/400000 [00:40<00:02, 8859.48it/s] 95%|█████████▌| 380319/400000 [00:40<00:02, 8983.77it/s] 95%|█████████▌| 381256/400000 [00:40<00:02, 9095.43it/s] 96%|█████████▌| 382173/400000 [00:40<00:01, 9115.73it/s] 96%|█████████▌| 383098/400000 [00:40<00:01, 9152.38it/s] 96%|█████████▌| 384015/400000 [00:40<00:01, 9150.72it/s] 96%|█████████▋| 385008/400000 [00:40<00:01, 9370.01it/s] 96%|█████████▋| 385947/400000 [00:41<00:01, 9131.05it/s] 97%|█████████▋| 386912/400000 [00:41<00:01, 9278.61it/s] 97%|█████████▋| 387843/400000 [00:41<00:01, 9202.97it/s] 97%|█████████▋| 388766/400000 [00:41<00:01, 9188.36it/s] 97%|█████████▋| 389728/400000 [00:41<00:01, 9311.99it/s] 98%|█████████▊| 390661/400000 [00:41<00:01, 9270.75it/s] 98%|█████████▊| 391590/400000 [00:41<00:00, 9175.68it/s] 98%|█████████▊| 392509/400000 [00:41<00:00, 9137.41it/s] 98%|█████████▊| 393424/400000 [00:41<00:00, 9000.25it/s] 99%|█████████▊| 394325/400000 [00:42<00:00, 8816.84it/s] 99%|█████████▉| 395247/400000 [00:42<00:00, 8933.22it/s] 99%|█████████▉| 396151/400000 [00:42<00:00, 8962.59it/s] 99%|█████████▉| 397049/400000 [00:42<00:00, 8891.35it/s] 99%|█████████▉| 397993/400000 [00:42<00:00, 9047.09it/s]100%|█████████▉| 398948/400000 [00:42<00:00, 9191.01it/s]100%|█████████▉| 399869/400000 [00:42<00:00, 8866.32it/s]100%|█████████▉| 399999/400000 [00:42<00:00, 9379.41it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f07946a2630> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011150102161713082 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011046171985741045 	 Accuracy: 64

  model saves at 64% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15845 out of table with 15799 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15845 out of table with 15799 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-13 01:24:19.139683: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 01:24:19.144127: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 01:24:19.144304: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558f3fd14600 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 01:24:19.144318: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f073bb93198> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4596 - accuracy: 0.5135 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.4724 - accuracy: 0.5127
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5363 - accuracy: 0.5085
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6176 - accuracy: 0.5032
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6308 - accuracy: 0.5023
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6863 - accuracy: 0.4987
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6973 - accuracy: 0.4980
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6837 - accuracy: 0.4989
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6712 - accuracy: 0.4997
11000/25000 [============>.................] - ETA: 4s - loss: 7.6457 - accuracy: 0.5014
12000/25000 [=============>................] - ETA: 4s - loss: 7.6142 - accuracy: 0.5034
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6147 - accuracy: 0.5034
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5965 - accuracy: 0.5046
15000/25000 [=================>............] - ETA: 3s - loss: 7.5736 - accuracy: 0.5061
16000/25000 [==================>...........] - ETA: 2s - loss: 7.5430 - accuracy: 0.5081
17000/25000 [===================>..........] - ETA: 2s - loss: 7.5800 - accuracy: 0.5056
18000/25000 [====================>.........] - ETA: 2s - loss: 7.5917 - accuracy: 0.5049
19000/25000 [=====================>........] - ETA: 1s - loss: 7.5908 - accuracy: 0.5049
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6045 - accuracy: 0.5041
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6111 - accuracy: 0.5036
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6269 - accuracy: 0.5026
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6373 - accuracy: 0.5019
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
25000/25000 [==============================] - 9s 369us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f06e84318d0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f06f621a160> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 924ms/step - loss: 1.3305 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.3784 - val_crf_viterbi_accuracy: 0.6533

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

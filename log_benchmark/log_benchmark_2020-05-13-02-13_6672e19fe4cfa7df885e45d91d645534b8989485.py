
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f5df8de3fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 02:14:08.966213
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 02:14:08.970952
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 02:14:08.975380
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 02:14:08.979094
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f5e04bad438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 349360.7188
Epoch 2/10

1/1 [==============================] - 0s 104ms/step - loss: 267443.1562
Epoch 3/10

1/1 [==============================] - 0s 103ms/step - loss: 194049.5625
Epoch 4/10

1/1 [==============================] - 0s 103ms/step - loss: 135203.0000
Epoch 5/10

1/1 [==============================] - 0s 105ms/step - loss: 91961.0312
Epoch 6/10

1/1 [==============================] - 0s 105ms/step - loss: 62921.5430
Epoch 7/10

1/1 [==============================] - 0s 101ms/step - loss: 44137.2422
Epoch 8/10

1/1 [==============================] - 0s 101ms/step - loss: 32121.9062
Epoch 9/10

1/1 [==============================] - 0s 102ms/step - loss: 24309.6738
Epoch 10/10

1/1 [==============================] - 0s 101ms/step - loss: 19041.1152

  #### Inference Need return ypred, ytrue ######################### 
[[-1.38539270e-01  3.89985824e+00  4.33200645e+00  2.63479114e+00
   2.72035861e+00  4.61594009e+00  4.95144892e+00  5.35103750e+00
   3.34625864e+00  4.32652664e+00  3.69628406e+00  5.17751455e+00
   3.28614593e+00  3.70244217e+00  3.22524047e+00  3.93919325e+00
   5.27946043e+00  4.08940125e+00  4.02649403e+00  2.72361398e+00
   4.00466394e+00  4.72478628e+00  3.28139257e+00  4.92688322e+00
   3.30164123e+00  4.77324820e+00  3.08837652e+00  4.77882195e+00
   3.24536514e+00  4.59563446e+00  4.09135675e+00  5.03611803e+00
   3.63385391e+00  4.70222282e+00  3.27824521e+00  2.66783690e+00
   4.37238741e+00  5.16979408e+00  3.02768636e+00  3.01523113e+00
   2.95650148e+00  3.53374815e+00  4.36154079e+00  3.71350050e+00
   2.63981342e+00  4.12849522e+00  3.89260483e+00  4.68190432e+00
   4.26706886e+00  4.18730688e+00  4.56242609e+00  5.16087723e+00
   4.32373714e+00  4.26547623e+00  3.41607714e+00  4.67508841e+00
   3.96854234e+00  3.61887765e+00  4.44465017e+00  5.34347391e+00
  -3.16336781e-01  8.24339211e-01  1.27437711e+00  2.29597509e-01
  -8.89909744e-01 -3.40005934e-01 -7.72029757e-01  9.48225796e-01
   1.14068413e+00 -5.08167207e-01  4.54760313e-01 -8.74315858e-01
  -1.26612151e+00 -1.34279728e+00 -4.37751323e-01  5.93742728e-01
   1.04356217e+00 -2.60700017e-01  2.12627396e-01 -1.03137565e+00
   5.04576802e-01  2.97964662e-01  5.54453969e-01 -1.06641412e+00
  -1.34604621e+00  7.14000687e-02 -8.53197217e-01 -4.38528389e-01
   9.49779868e-01 -1.20144880e+00 -1.19141057e-01 -1.38365313e-01
   4.88420248e-01  1.22636473e+00 -2.33157739e-01  5.36143541e-01
   1.21628857e+00 -9.15561497e-01  2.78141677e-01  3.05786014e-01
   4.97733563e-01  1.13476849e+00  5.95986903e-01 -9.87933397e-01
  -5.45836568e-01 -7.28505015e-01 -5.44789255e-01 -1.10324454e+00
  -8.44108224e-01  1.32933557e+00  8.40390444e-01  4.51501042e-01
   9.52718139e-01 -7.62190998e-01  2.18393981e-01 -1.06057273e-02
   6.74233437e-01  2.12188140e-01 -1.18546581e+00 -2.89721608e-01
  -4.05160338e-01  8.47252131e-01 -3.13672066e-01 -1.36659539e+00
   4.75512713e-01  8.82676303e-01 -1.32645118e+00 -1.63299814e-01
  -5.44707358e-01 -9.74536717e-01 -5.20467851e-03  1.03674579e+00
   1.11982930e+00 -6.15576029e-01  2.62544543e-01  7.21093595e-01
   7.04217553e-01 -1.16598451e+00  1.23539306e-01  8.58652949e-01
  -8.97319376e-01 -1.01187074e+00  1.36233044e+00  1.93373278e-01
   1.24645114e+00 -2.15445071e-01  2.47428313e-01 -1.04489672e+00
   1.30792654e+00 -7.86309600e-01 -7.60096848e-01 -9.88761634e-02
  -4.62588787e-01 -4.07818347e-01 -7.08125651e-01 -8.31773162e-01
   9.57373604e-02  1.31162822e+00 -7.17251122e-01  5.68194270e-01
  -1.10873306e+00  1.25088656e+00  9.76094604e-01  1.21282339e+00
   1.24816582e-01 -8.50935653e-02 -5.74702807e-02 -2.00706899e-01
   1.03069479e-02 -9.40281212e-01  1.25206426e-01 -8.73359740e-01
   1.37256849e+00  3.55186254e-01  1.10748613e+00 -3.19499195e-01
  -6.40111864e-01  7.46732593e-01  1.03789377e+00  1.22630405e+00
   4.00201082e-02  3.67234612e+00  4.39750814e+00  5.43349838e+00
   4.11088896e+00  5.43597317e+00  4.98467684e+00  5.57493114e+00
   5.53447390e+00  5.31680536e+00  4.93676376e+00  3.78443241e+00
   5.60486174e+00  3.55318785e+00  5.69397497e+00  5.68514299e+00
   4.83398056e+00  4.80651474e+00  5.14093876e+00  3.59537649e+00
   3.70535994e+00  5.11854839e+00  5.71500206e+00  5.07048178e+00
   4.24752426e+00  5.60357857e+00  4.30989599e+00  5.95188618e+00
   5.78917694e+00  5.91521072e+00  5.21408749e+00  4.58317137e+00
   5.42789030e+00  3.41511106e+00  6.00184870e+00  5.74591160e+00
   5.21690750e+00  5.91625404e+00  4.39621782e+00  4.37615061e+00
   5.95315552e+00  3.49629879e+00  5.81757450e+00  5.91794348e+00
   5.44361830e+00  4.32747936e+00  5.06093121e+00  5.89459324e+00
   4.56912756e+00  4.50543070e+00  5.92548513e+00  3.70137119e+00
   5.19114161e+00  4.65756416e+00  5.14201546e+00  5.26891232e+00
   5.13710260e+00  5.54456520e+00  3.40500164e+00  5.07732582e+00
   2.51948476e-01  6.38359070e-01  2.09372449e+00  2.18139029e+00
   5.40009797e-01  1.54913020e+00  1.10908234e+00  1.53454161e+00
   1.95990992e+00  1.81726027e+00  2.15699291e+00  2.25167632e+00
   2.11588240e+00  6.25097096e-01  2.58020997e-01  5.96495509e-01
   1.66838133e+00  9.84986961e-01  8.63398135e-01  2.22669077e+00
   1.44739842e+00  1.19939399e+00  1.59918106e+00  2.76857436e-01
   2.23028779e+00  1.51987529e+00  4.26470757e-01  1.30804479e+00
   6.94845438e-01  1.10864270e+00  4.10994411e-01  9.59167361e-01
   1.32999873e+00  3.31477106e-01  1.23880804e+00  1.93317866e+00
   2.04722548e+00  2.32908249e+00  3.63063812e-01  7.12146163e-01
   1.88685262e+00  7.12813973e-01  3.70652437e-01  2.01664114e+00
   3.25410187e-01  1.98001766e+00  4.32533205e-01  2.81979084e-01
   5.82392335e-01  1.69769478e+00  5.94190240e-01  8.18773508e-01
   1.88891292e+00  1.50061750e+00  3.57832611e-01  2.83390522e-01
   2.87410080e-01  1.45514691e+00  1.52842009e+00  1.94668150e+00
   5.38050592e-01  2.05311728e+00  2.05395079e+00  4.57725644e-01
   2.28331351e+00  6.27213895e-01  4.72800732e-01  9.91837919e-01
   1.24878693e+00  1.21858335e+00  2.96675920e-01  1.48504663e+00
   2.23526359e+00  3.12275290e-01  8.07146907e-01  2.27914810e+00
   1.43520689e+00  2.81157196e-01  1.27726436e+00  4.88959074e-01
   1.82325578e+00  1.84733582e+00  2.09142494e+00  2.04661131e+00
   2.92859674e-01  3.46436322e-01  8.07871640e-01  5.27279556e-01
   1.75327229e+00  2.22417021e+00  1.48366809e+00  3.89123201e-01
   5.58248758e-01  3.38199258e-01  4.74490762e-01  1.62890744e+00
   6.16521001e-01  4.47668374e-01  8.23284686e-01  1.51423502e+00
   1.16035497e+00  1.64041829e+00  1.71830726e+00  5.12058198e-01
   3.59696150e-01  1.53166509e+00  1.32013762e+00  9.90220606e-01
   1.26395011e+00  1.46210194e+00  1.10574579e+00  8.14790666e-01
   6.80240512e-01  1.68418813e+00  9.33309436e-01  2.13500953e+00
   1.63844275e+00  3.17768574e-01  1.58175004e+00  2.88762927e-01
   1.85161400e+00 -4.78264713e+00 -4.04973078e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 02:14:18.428316
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.6821
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 02:14:18.432732
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9560.66
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 02:14:18.436384
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.1193
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 02:14:18.440130
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -855.225
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140041224868304
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140040015110832
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140040015111336
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140040015111840
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140040015112344
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140040015112848

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f5df253afd0> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.470574
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.433891
grad_step = 000002, loss = 0.406275
grad_step = 000003, loss = 0.374874
grad_step = 000004, loss = 0.341784
grad_step = 000005, loss = 0.311670
grad_step = 000006, loss = 0.290026
grad_step = 000007, loss = 0.277303
grad_step = 000008, loss = 0.262526
grad_step = 000009, loss = 0.245715
grad_step = 000010, loss = 0.231611
grad_step = 000011, loss = 0.222679
grad_step = 000012, loss = 0.215313
grad_step = 000013, loss = 0.207648
grad_step = 000014, loss = 0.198317
grad_step = 000015, loss = 0.189275
grad_step = 000016, loss = 0.181324
grad_step = 000017, loss = 0.173620
grad_step = 000018, loss = 0.165206
grad_step = 000019, loss = 0.156488
grad_step = 000020, loss = 0.148623
grad_step = 000021, loss = 0.141909
grad_step = 000022, loss = 0.135580
grad_step = 000023, loss = 0.129011
grad_step = 000024, loss = 0.122205
grad_step = 000025, loss = 0.115530
grad_step = 000026, loss = 0.109244
grad_step = 000027, loss = 0.103207
grad_step = 000028, loss = 0.097335
grad_step = 000029, loss = 0.091569
grad_step = 000030, loss = 0.086140
grad_step = 000031, loss = 0.081166
grad_step = 000032, loss = 0.076370
grad_step = 000033, loss = 0.071570
grad_step = 000034, loss = 0.067023
grad_step = 000035, loss = 0.062924
grad_step = 000036, loss = 0.058990
grad_step = 000037, loss = 0.055050
grad_step = 000038, loss = 0.051215
grad_step = 000039, loss = 0.047595
grad_step = 000040, loss = 0.044199
grad_step = 000041, loss = 0.041069
grad_step = 000042, loss = 0.038163
grad_step = 000043, loss = 0.035419
grad_step = 000044, loss = 0.032854
grad_step = 000045, loss = 0.030446
grad_step = 000046, loss = 0.028118
grad_step = 000047, loss = 0.025870
grad_step = 000048, loss = 0.023768
grad_step = 000049, loss = 0.021864
grad_step = 000050, loss = 0.020135
grad_step = 000051, loss = 0.018523
grad_step = 000052, loss = 0.017001
grad_step = 000053, loss = 0.015559
grad_step = 000054, loss = 0.014224
grad_step = 000055, loss = 0.013013
grad_step = 000056, loss = 0.011891
grad_step = 000057, loss = 0.010863
grad_step = 000058, loss = 0.009930
grad_step = 000059, loss = 0.009082
grad_step = 000060, loss = 0.008314
grad_step = 000061, loss = 0.007605
grad_step = 000062, loss = 0.006960
grad_step = 000063, loss = 0.006388
grad_step = 000064, loss = 0.005877
grad_step = 000065, loss = 0.005402
grad_step = 000066, loss = 0.004985
grad_step = 000067, loss = 0.004634
grad_step = 000068, loss = 0.004318
grad_step = 000069, loss = 0.004033
grad_step = 000070, loss = 0.003781
grad_step = 000071, loss = 0.003561
grad_step = 000072, loss = 0.003366
grad_step = 000073, loss = 0.003200
grad_step = 000074, loss = 0.003056
grad_step = 000075, loss = 0.002931
grad_step = 000076, loss = 0.002825
grad_step = 000077, loss = 0.002736
grad_step = 000078, loss = 0.002654
grad_step = 000079, loss = 0.002586
grad_step = 000080, loss = 0.002529
grad_step = 000081, loss = 0.002481
grad_step = 000082, loss = 0.002441
grad_step = 000083, loss = 0.002407
grad_step = 000084, loss = 0.002378
grad_step = 000085, loss = 0.002352
grad_step = 000086, loss = 0.002331
grad_step = 000087, loss = 0.002314
grad_step = 000088, loss = 0.002299
grad_step = 000089, loss = 0.002287
grad_step = 000090, loss = 0.002275
grad_step = 000091, loss = 0.002264
grad_step = 000092, loss = 0.002257
grad_step = 000093, loss = 0.002250
grad_step = 000094, loss = 0.002243
grad_step = 000095, loss = 0.002237
grad_step = 000096, loss = 0.002230
grad_step = 000097, loss = 0.002224
grad_step = 000098, loss = 0.002218
grad_step = 000099, loss = 0.002212
grad_step = 000100, loss = 0.002206
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002199
grad_step = 000102, loss = 0.002193
grad_step = 000103, loss = 0.002187
grad_step = 000104, loss = 0.002180
grad_step = 000105, loss = 0.002173
grad_step = 000106, loss = 0.002166
grad_step = 000107, loss = 0.002159
grad_step = 000108, loss = 0.002153
grad_step = 000109, loss = 0.002146
grad_step = 000110, loss = 0.002140
grad_step = 000111, loss = 0.002134
grad_step = 000112, loss = 0.002128
grad_step = 000113, loss = 0.002122
grad_step = 000114, loss = 0.002116
grad_step = 000115, loss = 0.002110
grad_step = 000116, loss = 0.002104
grad_step = 000117, loss = 0.002099
grad_step = 000118, loss = 0.002094
grad_step = 000119, loss = 0.002089
grad_step = 000120, loss = 0.002085
grad_step = 000121, loss = 0.002083
grad_step = 000122, loss = 0.002078
grad_step = 000123, loss = 0.002069
grad_step = 000124, loss = 0.002065
grad_step = 000125, loss = 0.002064
grad_step = 000126, loss = 0.002059
grad_step = 000127, loss = 0.002052
grad_step = 000128, loss = 0.002052
grad_step = 000129, loss = 0.002050
grad_step = 000130, loss = 0.002042
grad_step = 000131, loss = 0.002032
grad_step = 000132, loss = 0.002029
grad_step = 000133, loss = 0.002029
grad_step = 000134, loss = 0.002025
grad_step = 000135, loss = 0.002018
grad_step = 000136, loss = 0.002011
grad_step = 000137, loss = 0.002006
grad_step = 000138, loss = 0.002003
grad_step = 000139, loss = 0.001999
grad_step = 000140, loss = 0.001993
grad_step = 000141, loss = 0.001986
grad_step = 000142, loss = 0.001979
grad_step = 000143, loss = 0.001976
grad_step = 000144, loss = 0.001977
grad_step = 000145, loss = 0.001990
grad_step = 000146, loss = 0.002024
grad_step = 000147, loss = 0.002007
grad_step = 000148, loss = 0.001990
grad_step = 000149, loss = 0.001985
grad_step = 000150, loss = 0.001993
grad_step = 000151, loss = 0.001979
grad_step = 000152, loss = 0.001931
grad_step = 000153, loss = 0.001950
grad_step = 000154, loss = 0.001985
grad_step = 000155, loss = 0.001938
grad_step = 000156, loss = 0.001981
grad_step = 000157, loss = 0.002028
grad_step = 000158, loss = 0.001941
grad_step = 000159, loss = 0.002005
grad_step = 000160, loss = 0.001985
grad_step = 000161, loss = 0.001932
grad_step = 000162, loss = 0.001958
grad_step = 000163, loss = 0.001899
grad_step = 000164, loss = 0.001922
grad_step = 000165, loss = 0.001902
grad_step = 000166, loss = 0.001898
grad_step = 000167, loss = 0.001917
grad_step = 000168, loss = 0.001912
grad_step = 000169, loss = 0.001943
grad_step = 000170, loss = 0.001939
grad_step = 000171, loss = 0.001932
grad_step = 000172, loss = 0.001890
grad_step = 000173, loss = 0.001861
grad_step = 000174, loss = 0.001849
grad_step = 000175, loss = 0.001861
grad_step = 000176, loss = 0.001902
grad_step = 000177, loss = 0.001924
grad_step = 000178, loss = 0.001869
grad_step = 000179, loss = 0.001844
grad_step = 000180, loss = 0.001813
grad_step = 000181, loss = 0.001818
grad_step = 000182, loss = 0.001855
grad_step = 000183, loss = 0.001867
grad_step = 000184, loss = 0.001880
grad_step = 000185, loss = 0.001865
grad_step = 000186, loss = 0.001824
grad_step = 000187, loss = 0.001799
grad_step = 000188, loss = 0.001775
grad_step = 000189, loss = 0.001780
grad_step = 000190, loss = 0.001791
grad_step = 000191, loss = 0.001794
grad_step = 000192, loss = 0.001796
grad_step = 000193, loss = 0.001788
grad_step = 000194, loss = 0.001779
grad_step = 000195, loss = 0.001778
grad_step = 000196, loss = 0.001765
grad_step = 000197, loss = 0.001754
grad_step = 000198, loss = 0.001750
grad_step = 000199, loss = 0.001739
grad_step = 000200, loss = 0.001737
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001734
grad_step = 000202, loss = 0.001731
grad_step = 000203, loss = 0.001736
grad_step = 000204, loss = 0.001755
grad_step = 000205, loss = 0.001790
grad_step = 000206, loss = 0.001876
grad_step = 000207, loss = 0.002026
grad_step = 000208, loss = 0.002075
grad_step = 000209, loss = 0.001975
grad_step = 000210, loss = 0.001738
grad_step = 000211, loss = 0.001737
grad_step = 000212, loss = 0.001888
grad_step = 000213, loss = 0.001871
grad_step = 000214, loss = 0.001773
grad_step = 000215, loss = 0.001712
grad_step = 000216, loss = 0.001736
grad_step = 000217, loss = 0.001815
grad_step = 000218, loss = 0.001775
grad_step = 000219, loss = 0.001693
grad_step = 000220, loss = 0.001691
grad_step = 000221, loss = 0.001751
grad_step = 000222, loss = 0.001764
grad_step = 000223, loss = 0.001675
grad_step = 000224, loss = 0.001658
grad_step = 000225, loss = 0.001667
grad_step = 000226, loss = 0.001699
grad_step = 000227, loss = 0.001710
grad_step = 000228, loss = 0.001673
grad_step = 000229, loss = 0.001647
grad_step = 000230, loss = 0.001630
grad_step = 000231, loss = 0.001635
grad_step = 000232, loss = 0.001654
grad_step = 000233, loss = 0.001652
grad_step = 000234, loss = 0.001642
grad_step = 000235, loss = 0.001621
grad_step = 000236, loss = 0.001611
grad_step = 000237, loss = 0.001611
grad_step = 000238, loss = 0.001612
grad_step = 000239, loss = 0.001617
grad_step = 000240, loss = 0.001627
grad_step = 000241, loss = 0.001633
grad_step = 000242, loss = 0.001634
grad_step = 000243, loss = 0.001628
grad_step = 000244, loss = 0.001621
grad_step = 000245, loss = 0.001618
grad_step = 000246, loss = 0.001618
grad_step = 000247, loss = 0.001619
grad_step = 000248, loss = 0.001617
grad_step = 000249, loss = 0.001615
grad_step = 000250, loss = 0.001614
grad_step = 000251, loss = 0.001617
grad_step = 000252, loss = 0.001626
grad_step = 000253, loss = 0.001641
grad_step = 000254, loss = 0.001658
grad_step = 000255, loss = 0.001673
grad_step = 000256, loss = 0.001687
grad_step = 000257, loss = 0.001699
grad_step = 000258, loss = 0.001700
grad_step = 000259, loss = 0.001689
grad_step = 000260, loss = 0.001632
grad_step = 000261, loss = 0.001574
grad_step = 000262, loss = 0.001541
grad_step = 000263, loss = 0.001543
grad_step = 000264, loss = 0.001562
grad_step = 000265, loss = 0.001570
grad_step = 000266, loss = 0.001580
grad_step = 000267, loss = 0.001586
grad_step = 000268, loss = 0.001593
grad_step = 000269, loss = 0.001578
grad_step = 000270, loss = 0.001550
grad_step = 000271, loss = 0.001517
grad_step = 000272, loss = 0.001505
grad_step = 000273, loss = 0.001510
grad_step = 000274, loss = 0.001521
grad_step = 000275, loss = 0.001526
grad_step = 000276, loss = 0.001518
grad_step = 000277, loss = 0.001508
grad_step = 000278, loss = 0.001499
grad_step = 000279, loss = 0.001500
grad_step = 000280, loss = 0.001517
grad_step = 000281, loss = 0.001569
grad_step = 000282, loss = 0.001634
grad_step = 000283, loss = 0.001747
grad_step = 000284, loss = 0.001739
grad_step = 000285, loss = 0.001741
grad_step = 000286, loss = 0.001752
grad_step = 000287, loss = 0.001719
grad_step = 000288, loss = 0.001544
grad_step = 000289, loss = 0.001467
grad_step = 000290, loss = 0.001570
grad_step = 000291, loss = 0.001612
grad_step = 000292, loss = 0.001498
grad_step = 000293, loss = 0.001444
grad_step = 000294, loss = 0.001497
grad_step = 000295, loss = 0.001574
grad_step = 000296, loss = 0.001547
grad_step = 000297, loss = 0.001474
grad_step = 000298, loss = 0.001446
grad_step = 000299, loss = 0.001472
grad_step = 000300, loss = 0.001487
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001459
grad_step = 000302, loss = 0.001412
grad_step = 000303, loss = 0.001410
grad_step = 000304, loss = 0.001438
grad_step = 000305, loss = 0.001444
grad_step = 000306, loss = 0.001421
grad_step = 000307, loss = 0.001393
grad_step = 000308, loss = 0.001397
grad_step = 000309, loss = 0.001412
grad_step = 000310, loss = 0.001404
grad_step = 000311, loss = 0.001382
grad_step = 000312, loss = 0.001368
grad_step = 000313, loss = 0.001372
grad_step = 000314, loss = 0.001388
grad_step = 000315, loss = 0.001402
grad_step = 000316, loss = 0.001420
grad_step = 000317, loss = 0.001445
grad_step = 000318, loss = 0.001507
grad_step = 000319, loss = 0.001577
grad_step = 000320, loss = 0.001646
grad_step = 000321, loss = 0.001598
grad_step = 000322, loss = 0.001438
grad_step = 000323, loss = 0.001336
grad_step = 000324, loss = 0.001366
grad_step = 000325, loss = 0.001436
grad_step = 000326, loss = 0.001442
grad_step = 000327, loss = 0.001398
grad_step = 000328, loss = 0.001349
grad_step = 000329, loss = 0.001317
grad_step = 000330, loss = 0.001297
grad_step = 000331, loss = 0.001312
grad_step = 000332, loss = 0.001349
grad_step = 000333, loss = 0.001380
grad_step = 000334, loss = 0.001400
grad_step = 000335, loss = 0.001361
grad_step = 000336, loss = 0.001326
grad_step = 000337, loss = 0.001306
grad_step = 000338, loss = 0.001284
grad_step = 000339, loss = 0.001264
grad_step = 000340, loss = 0.001261
grad_step = 000341, loss = 0.001270
grad_step = 000342, loss = 0.001288
grad_step = 000343, loss = 0.001300
grad_step = 000344, loss = 0.001301
grad_step = 000345, loss = 0.001299
grad_step = 000346, loss = 0.001304
grad_step = 000347, loss = 0.001326
grad_step = 000348, loss = 0.001313
grad_step = 000349, loss = 0.001300
grad_step = 000350, loss = 0.001250
grad_step = 000351, loss = 0.001225
grad_step = 000352, loss = 0.001219
grad_step = 000353, loss = 0.001218
grad_step = 000354, loss = 0.001214
grad_step = 000355, loss = 0.001214
grad_step = 000356, loss = 0.001223
grad_step = 000357, loss = 0.001244
grad_step = 000358, loss = 0.001290
grad_step = 000359, loss = 0.001328
grad_step = 000360, loss = 0.001388
grad_step = 000361, loss = 0.001350
grad_step = 000362, loss = 0.001313
grad_step = 000363, loss = 0.001228
grad_step = 000364, loss = 0.001179
grad_step = 000365, loss = 0.001160
grad_step = 000366, loss = 0.001165
grad_step = 000367, loss = 0.001178
grad_step = 000368, loss = 0.001193
grad_step = 000369, loss = 0.001228
grad_step = 000370, loss = 0.001259
grad_step = 000371, loss = 0.001350
grad_step = 000372, loss = 0.001238
grad_step = 000373, loss = 0.001170
grad_step = 000374, loss = 0.001130
grad_step = 000375, loss = 0.001157
grad_step = 000376, loss = 0.001260
grad_step = 000377, loss = 0.001264
grad_step = 000378, loss = 0.001287
grad_step = 000379, loss = 0.001193
grad_step = 000380, loss = 0.001159
grad_step = 000381, loss = 0.001156
grad_step = 000382, loss = 0.001149
grad_step = 000383, loss = 0.001199
grad_step = 000384, loss = 0.001208
grad_step = 000385, loss = 0.001227
grad_step = 000386, loss = 0.001208
grad_step = 000387, loss = 0.001137
grad_step = 000388, loss = 0.001098
grad_step = 000389, loss = 0.001136
grad_step = 000390, loss = 0.001121
grad_step = 000391, loss = 0.001091
grad_step = 000392, loss = 0.001059
grad_step = 000393, loss = 0.001052
grad_step = 000394, loss = 0.001064
grad_step = 000395, loss = 0.001069
grad_step = 000396, loss = 0.001061
grad_step = 000397, loss = 0.001057
grad_step = 000398, loss = 0.001107
grad_step = 000399, loss = 0.001172
grad_step = 000400, loss = 0.001347
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001271
grad_step = 000402, loss = 0.001181
grad_step = 000403, loss = 0.001059
grad_step = 000404, loss = 0.001012
grad_step = 000405, loss = 0.001009
grad_step = 000406, loss = 0.001065
grad_step = 000407, loss = 0.001181
grad_step = 000408, loss = 0.001160
grad_step = 000409, loss = 0.001210
grad_step = 000410, loss = 0.001076
grad_step = 000411, loss = 0.001008
grad_step = 000412, loss = 0.000974
grad_step = 000413, loss = 0.001026
grad_step = 000414, loss = 0.001228
grad_step = 000415, loss = 0.001112
grad_step = 000416, loss = 0.001016
grad_step = 000417, loss = 0.000975
grad_step = 000418, loss = 0.000990
grad_step = 000419, loss = 0.001038
grad_step = 000420, loss = 0.001057
grad_step = 000421, loss = 0.001117
grad_step = 000422, loss = 0.001024
grad_step = 000423, loss = 0.000955
grad_step = 000424, loss = 0.000960
grad_step = 000425, loss = 0.001023
grad_step = 000426, loss = 0.001186
grad_step = 000427, loss = 0.001093
grad_step = 000428, loss = 0.000976
grad_step = 000429, loss = 0.000927
grad_step = 000430, loss = 0.000954
grad_step = 000431, loss = 0.001089
grad_step = 000432, loss = 0.001078
grad_step = 000433, loss = 0.001006
grad_step = 000434, loss = 0.000908
grad_step = 000435, loss = 0.000900
grad_step = 000436, loss = 0.000986
grad_step = 000437, loss = 0.001027
grad_step = 000438, loss = 0.001047
grad_step = 000439, loss = 0.000919
grad_step = 000440, loss = 0.000879
grad_step = 000441, loss = 0.000867
grad_step = 000442, loss = 0.000886
grad_step = 000443, loss = 0.000953
grad_step = 000444, loss = 0.000927
grad_step = 000445, loss = 0.000923
grad_step = 000446, loss = 0.000864
grad_step = 000447, loss = 0.000835
grad_step = 000448, loss = 0.000828
grad_step = 000449, loss = 0.000835
grad_step = 000450, loss = 0.000844
grad_step = 000451, loss = 0.000853
grad_step = 000452, loss = 0.000921
grad_step = 000453, loss = 0.000895
grad_step = 000454, loss = 0.000921
grad_step = 000455, loss = 0.000855
grad_step = 000456, loss = 0.000823
grad_step = 000457, loss = 0.000806
grad_step = 000458, loss = 0.000802
grad_step = 000459, loss = 0.000797
grad_step = 000460, loss = 0.000809
grad_step = 000461, loss = 0.000825
grad_step = 000462, loss = 0.000920
grad_step = 000463, loss = 0.000891
grad_step = 000464, loss = 0.000968
grad_step = 000465, loss = 0.000847
grad_step = 000466, loss = 0.000782
grad_step = 000467, loss = 0.000808
grad_step = 000468, loss = 0.000930
grad_step = 000469, loss = 0.001267
grad_step = 000470, loss = 0.000968
grad_step = 000471, loss = 0.000926
grad_step = 000472, loss = 0.001426
grad_step = 000473, loss = 0.001104
grad_step = 000474, loss = 0.000930
grad_step = 000475, loss = 0.001167
grad_step = 000476, loss = 0.001027
grad_step = 000477, loss = 0.001091
grad_step = 000478, loss = 0.001236
grad_step = 000479, loss = 0.001153
grad_step = 000480, loss = 0.001256
grad_step = 000481, loss = 0.001216
grad_step = 000482, loss = 0.001398
grad_step = 000483, loss = 0.001261
grad_step = 000484, loss = 0.001296
grad_step = 000485, loss = 0.001316
grad_step = 000486, loss = 0.001046
grad_step = 000487, loss = 0.001057
grad_step = 000488, loss = 0.001050
grad_step = 000489, loss = 0.000922
grad_step = 000490, loss = 0.000934
grad_step = 000491, loss = 0.001013
grad_step = 000492, loss = 0.000770
grad_step = 000493, loss = 0.000948
grad_step = 000494, loss = 0.000858
grad_step = 000495, loss = 0.000789
grad_step = 000496, loss = 0.000830
grad_step = 000497, loss = 0.000787
grad_step = 000498, loss = 0.000749
grad_step = 000499, loss = 0.000811
grad_step = 000500, loss = 0.000710
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000873
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

  date_run                              2020-05-13 02:14:42.521021
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.178424
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 02:14:42.528465
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0749643
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 02:14:42.536232
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.111381
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 02:14:42.542311
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -0.13911
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
0   2020-05-13 02:14:08.966213  ...    mean_absolute_error
1   2020-05-13 02:14:08.970952  ...     mean_squared_error
2   2020-05-13 02:14:08.975380  ...  median_absolute_error
3   2020-05-13 02:14:08.979094  ...               r2_score
4   2020-05-13 02:14:18.428316  ...    mean_absolute_error
5   2020-05-13 02:14:18.432732  ...     mean_squared_error
6   2020-05-13 02:14:18.436384  ...  median_absolute_error
7   2020-05-13 02:14:18.440130  ...               r2_score
8   2020-05-13 02:14:42.521021  ...    mean_absolute_error
9   2020-05-13 02:14:42.528465  ...     mean_squared_error
10  2020-05-13 02:14:42.536232  ...  median_absolute_error
11  2020-05-13 02:14:42.542311  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2f663e9b0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 35%|███▍      | 3448832/9912422 [00:00<00:00, 34275474.43it/s]9920512it [00:00, 34920132.97it/s]                             
0it [00:00, ?it/s]32768it [00:00, 468441.81it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 162561.92it/s]1654784it [00:00, 11683798.71it/s]                         
0it [00:00, ?it/s]8192it [00:00, 192082.62it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2a8fece48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2a5e390b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2a8fece48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2a5e39048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2a5dae4a8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2a5e390b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2a8fece48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2a5e39048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2a5dae4a8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fa2f663e9b0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ff9d7e1b208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=76061050ead0da6b8681985bcf7756de6a37b8bb3452ffca993ff8dc6a2949a6
  Stored in directory: /tmp/pip-ephem-wheel-cache-hhbpikpp/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ff9cdf85080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2211840/17464789 [==>...........................] - ETA: 0s
 6930432/17464789 [==========>...................] - ETA: 0s
11894784/17464789 [===================>..........] - ETA: 0s
17203200/17464789 [============================>.] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 02:16:10.991676: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 02:16:10.996528: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 02:16:10.996746: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55966eb24f80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 02:16:10.996763: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 8.1113 - accuracy: 0.4710
 2000/25000 [=>............................] - ETA: 10s - loss: 7.9043 - accuracy: 0.4845
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8864 - accuracy: 0.4857 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.8085 - accuracy: 0.4908
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7341 - accuracy: 0.4956
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7228 - accuracy: 0.4963
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7192 - accuracy: 0.4966
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.7088 - accuracy: 0.4972
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6598 - accuracy: 0.5004
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6390 - accuracy: 0.5018
11000/25000 [============>.................] - ETA: 4s - loss: 7.6541 - accuracy: 0.5008
12000/25000 [=============>................] - ETA: 4s - loss: 7.6436 - accuracy: 0.5015
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6820 - accuracy: 0.4990
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6568 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 3s - loss: 7.6472 - accuracy: 0.5013
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6398 - accuracy: 0.5017
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6531 - accuracy: 0.5009
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6453 - accuracy: 0.5014
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6545 - accuracy: 0.5008
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6620 - accuracy: 0.5003
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6761 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6638 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6633 - accuracy: 0.5002
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6558 - accuracy: 0.5007
25000/25000 [==============================] - 10s 397us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 02:16:28.530957
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 02:16:28.530957  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:03<92:10:52, 2.60kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:03<64:45:19, 3.70kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:03<45:22:31, 5.28kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:03<31:45:06, 7.53kB/s].vector_cache/glove.6B.zip:   0%|          | 3.51M/862M [00:03<22:09:41, 10.8kB/s].vector_cache/glove.6B.zip:   1%|          | 5.42M/862M [00:03<15:28:56, 15.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.36M/862M [00:03<10:47:24, 22.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.7M/862M [00:03<7:30:59, 31.4kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 16.0M/862M [00:04<5:14:59, 44.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.6M/862M [00:04<3:39:42, 63.9kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.8M/862M [00:04<2:33:07, 91.3kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.4M/862M [00:04<1:46:49, 130kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 32.6M/862M [00:04<1:14:24, 186kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.0M/862M [00:04<51:54, 265kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 41.2M/862M [00:04<36:15, 377kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.5M/862M [00:04<25:20, 537kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.7M/862M [00:04<17:44, 763kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:05<13:22, 1.01MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.0M/862M [00:05<09:33, 1.41MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.7M/862M [00:07<10:52, 1.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:07<12:39, 1.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.3M/862M [00:07<10:06, 1.33MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.0M/862M [00:07<07:22, 1.81MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:09<08:29, 1.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:09<08:20, 1.60MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:09<06:26, 2.07MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:11<06:43, 1.97MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:11<07:06, 1.87MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.1M/862M [00:11<05:30, 2.41MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:11<03:58, 3.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:13<24:32, 538kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:13<19:33, 676kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.2M/862M [00:13<14:16, 924kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:15<12:10, 1.08MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:15<10:53, 1.21MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:15<08:12, 1.60MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:17<07:56, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:17<07:54, 1.65MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:17<06:04, 2.15MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:19<06:26, 2.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:19<06:52, 1.89MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.6M/862M [00:19<05:21, 2.42MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:19<03:54, 3.32MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:21<12:16, 1.05MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:21<10:56, 1.18MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.7M/862M [00:21<08:14, 1.57MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:23<07:54, 1.63MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:23<07:53, 1.63MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.8M/862M [00:23<06:00, 2.14MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:23<04:21, 2.94MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:25<11:29, 1.11MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:25<10:28, 1.22MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:25<07:54, 1.62MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:27<07:39, 1.66MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:27<07:44, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:27<05:54, 2.15MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<04:16, 2.97MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<15:18, 828kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:29<13:05, 967kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:29<09:43, 1.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<08:54, 1.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:31<08:35, 1.47MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:31<06:34, 1.91MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<06:41, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<09:22, 1.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:33<07:34, 1.65MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:33<05:41, 2.20MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<06:15, 1.99MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:35<06:37, 1.88MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:35<05:12, 2.39MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:37<05:43, 2.17MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:37<06:14, 1.98MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:37<04:52, 2.54MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<03:32, 3.48MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<15:16, 807kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<15:17, 806kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<11:51, 1.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:39<08:34, 1.43MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<08:59, 1.36MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<08:35, 1.43MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:41<06:33, 1.87MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<06:37, 1.84MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<09:14, 1.32MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:43<07:37, 1.60MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:43<05:34, 2.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<06:54, 1.76MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<07:00, 1.73MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:45<05:27, 2.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<05:50, 2.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<06:18, 1.91MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:47<04:57, 2.43MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<05:28, 2.19MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:49<05:59, 2.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:49<04:43, 2.53MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<05:18, 2.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:51<07:50, 1.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:51<06:36, 1.80MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:51<04:51, 2.44MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<06:20, 1.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<06:34, 1.80MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:53<05:08, 2.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<05:33, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<06:01, 1.96MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:55<04:44, 2.48MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:55<03:25, 3.41MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<48:06, 243kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<35:46, 327kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<25:28, 459kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:57<17:56, 649kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<18:23, 633kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<14:57, 777kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:59<10:59, 1.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:59<07:47, 1.49MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<43:05, 268kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<32:14, 359kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:00<23:03, 501kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<18:00, 638kB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:02<14:40, 783kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:02<10:46, 1.06MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<09:26, 1.21MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<10:52, 1.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:04<08:40, 1.32MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:05<06:17, 1.81MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<07:10, 1.59MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<07:08, 1.59MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:06<05:30, 2.06MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<05:44, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<06:07, 1.84MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:08<04:43, 2.38MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<03:25, 3.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<14:04, 797kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<14:02, 799kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<10:52, 1.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:10<07:51, 1.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<08:15, 1.35MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<07:47, 1.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:12<05:52, 1.90MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<04:13, 2.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<12:21, 897kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<10:40, 1.04MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:14<07:57, 1.39MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<07:24, 1.49MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<07:10, 1.54MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<05:26, 2.02MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<03:55, 2.79MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<13:03, 838kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:18<11:06, 985kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:18<08:15, 1.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<07:35, 1.43MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<09:22, 1.16MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<07:28, 1.46MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:20<05:28, 1.98MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<06:29, 1.66MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<06:30, 1.66MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:22<05:02, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<05:19, 2.02MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<07:44, 1.39MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<06:25, 1.67MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:24<04:41, 2.28MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<05:53, 1.81MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<06:02, 1.76MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:26<04:42, 2.26MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<05:04, 2.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<05:31, 1.92MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<04:16, 2.47MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<03:06, 3.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<13:43, 768kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<13:33, 776kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:30<10:22, 1.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:30<07:29, 1.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<07:49, 1.34MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<07:22, 1.42MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:32<05:35, 1.87MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<04:00, 2.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<41:12, 252kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<32:43, 318kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<23:51, 435kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:34<16:54, 613kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<14:22, 718kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<11:55, 866kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:36<08:48, 1.17MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<07:51, 1.31MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<07:25, 1.38MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:38<05:39, 1.81MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<05:39, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<05:51, 1.74MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:40<04:34, 2.23MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<04:52, 2.07MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<05:18, 1.90MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:42<04:06, 2.46MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:42<02:59, 3.37MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:44<10:22, 969kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:44<09:08, 1.10MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:44<06:50, 1.47MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:46<06:26, 1.55MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:46<08:16, 1.21MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:46<06:43, 1.48MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:46<04:55, 2.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<05:54, 1.68MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<05:55, 1.68MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:48<04:35, 2.16MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<04:51, 2.03MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:50<05:10, 1.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:50<04:03, 2.42MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<04:28, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:52<04:54, 1.99MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:52<03:48, 2.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<02:46, 3.50MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:53<08:53, 1.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:54<09:35, 1.01MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:54<07:28, 1.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:54<05:26, 1.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:55<06:12, 1.55MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:55<06:04, 1.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:56<04:37, 2.08MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:56<03:20, 2.87MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<10:41, 895kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<09:12, 1.04MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:58<06:52, 1.39MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<06:23, 1.49MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<06:11, 1.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:00<04:45, 1.99MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<04:54, 1.92MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<05:07, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:02<04:01, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<04:22, 2.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<06:33, 1.43MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:03<05:27, 1.72MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:04<04:01, 2.32MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<05:07, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:05<05:19, 1.75MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:05<04:08, 2.24MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<04:26, 2.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<06:36, 1.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<05:21, 1.72MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:08<03:57, 2.32MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<05:01, 1.83MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:09<05:09, 1.77MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:09<04:01, 2.27MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<04:20, 2.10MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<04:42, 1.93MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<03:42, 2.45MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<04:05, 2.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<06:16, 1.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:13<05:05, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:13<03:43, 2.42MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<04:50, 1.85MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<05:03, 1.77MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:15<03:53, 2.30MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<02:48, 3.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:17<19:29, 456kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:17<16:57, 524kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<12:34, 705kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:17<08:59, 984kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<08:26, 1.05MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:19<07:29, 1.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:19<05:38, 1.56MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:21<05:23, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<05:08, 1.70MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<03:58, 2.20MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<03:01, 2.88MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<04:20, 2.00MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:23<04:35, 1.89MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:23<03:33, 2.44MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<02:34, 3.34MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<09:46, 880kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<08:23, 1.02MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:25<06:15, 1.37MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<05:47, 1.47MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<07:15, 1.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:27<05:45, 1.48MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:27<04:10, 2.04MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:29<04:57, 1.71MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:29<05:04, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:29<03:55, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:31<04:08, 2.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:31<04:28, 1.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:31<03:30, 2.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:33<03:50, 2.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:33<04:14, 1.96MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:33<03:17, 2.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<02:23, 3.46MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:35<12:05, 684kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<09:59, 826kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:35<07:21, 1.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:37<06:30, 1.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:37<06:01, 1.36MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:37<04:35, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:39<04:33, 1.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:39<04:39, 1.74MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:39<03:34, 2.27MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<02:35, 3.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<07:37, 1.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<06:47, 1.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:41<05:06, 1.57MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<04:54, 1.63MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<04:53, 1.63MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:43<03:46, 2.11MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<03:58, 2.00MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:45<04:12, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:45<03:18, 2.39MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:47<03:37, 2.17MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:47<03:56, 1.99MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:47<03:06, 2.52MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<02:15, 3.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:49<18:27, 422kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:49<14:19, 543kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:49<10:21, 750kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:50<08:30, 907kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:51<08:50, 872kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:51<06:47, 1.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:51<04:54, 1.57MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:52<05:07, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:53<05:01, 1.52MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:53<03:51, 1.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:54<03:57, 1.92MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:55<04:08, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:55<03:14, 2.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:56<03:30, 2.14MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:57<03:48, 1.97MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:57<03:00, 2.49MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<03:20, 2.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<03:40, 2.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:59<02:54, 2.55MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:00<03:15, 2.26MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:00<03:36, 2.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:01<02:51, 2.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:02<03:13, 2.27MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:02<03:34, 2.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:03<02:46, 2.62MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:03<02:01, 3.57MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<06:07, 1.18MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<05:37, 1.29MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<04:15, 1.69MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:06<04:09, 1.72MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<04:12, 1.70MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<03:13, 2.22MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:07<02:19, 3.06MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:08<13:59, 507kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:08<11:03, 641kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:08<08:02, 879kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<06:46, 1.04MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<06:00, 1.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:10<04:30, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:11<03:12, 2.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<17:20, 401kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:12<13:26, 518kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:12<09:39, 719kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<06:50, 1.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<07:50, 879kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<06:47, 1.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:14<05:03, 1.36MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:16<04:39, 1.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:16<04:32, 1.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:16<03:28, 1.96MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:18<03:33, 1.90MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:18<05:01, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:18<04:03, 1.66MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:18<02:59, 2.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:20<03:43, 1.80MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:20<03:48, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<02:55, 2.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<02:07, 3.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:22<05:09, 1.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:22<04:47, 1.38MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:22<03:36, 1.83MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<02:36, 2.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:24<05:08, 1.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:24<04:46, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:24<03:35, 1.82MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<02:35, 2.50MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:26<04:58, 1.30MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:26<05:54, 1.10MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:26<04:39, 1.39MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:26<03:25, 1.88MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:28<03:56, 1.62MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:28<03:58, 1.61MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:28<03:01, 2.11MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<02:11, 2.91MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<05:45, 1.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<06:23, 990kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<05:04, 1.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:30<03:39, 1.72MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:32<04:06, 1.52MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:32<04:00, 1.56MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:32<03:03, 2.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:32<02:12, 2.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:34<06:33, 945kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:34<05:42, 1.08MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:34<04:16, 1.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<03:02, 2.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:36<12:40, 484kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:36<09:59, 614kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:36<07:15, 843kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:38<06:03, 1.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<05:19, 1.14MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<03:59, 1.51MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<03:47, 1.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<04:53, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:40<03:53, 1.54MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:40<02:49, 2.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<03:25, 1.73MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:42<03:29, 1.70MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:42<02:40, 2.21MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:42<01:57, 3.00MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:44<03:40, 1.60MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:44<03:37, 1.61MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:44<02:46, 2.11MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<01:58, 2.93MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<39:28, 147kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<28:39, 202kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:46<20:16, 285kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<15:01, 380kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<11:32, 495kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:48<08:17, 688kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:48<05:50, 970kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<07:40, 736kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<06:23, 883kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:50<04:43, 1.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<04:12, 1.32MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<03:59, 1.40MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:52<03:02, 1.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:53<03:02, 1.82MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<03:09, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<02:27, 2.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:55<02:36, 2.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<03:54, 1.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<03:10, 1.71MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:56<02:19, 2.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:57<02:48, 1.91MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<02:40, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<02:02, 2.62MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<02:27, 2.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<02:38, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:00<02:04, 2.56MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:01<02:20, 2.24MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:01<02:32, 2.06MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:02<01:58, 2.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:02<01:25, 3.61MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:03<09:02, 571kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:03<07:15, 712kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<05:15, 978kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:04<03:42, 1.38MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:05<40:26, 126kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:05<29:10, 175kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<20:33, 247kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<14:22, 351kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:07<12:20, 408kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:07<09:32, 527kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:07<06:52, 729kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:09<05:36, 884kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:09<04:47, 1.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:09<03:32, 1.40MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:10<02:31, 1.95MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:11<05:43, 856kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:11<04:51, 1.01MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:11<03:34, 1.37MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<02:32, 1.91MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:13<06:47, 710kB/s] .vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:13<05:35, 862kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:13<04:05, 1.18MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:53, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:15<06:53, 690kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:15<06:28, 735kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:15<04:57, 959kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<03:32, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:17<03:40, 1.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:17<03:25, 1.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:17<02:33, 1.82MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<01:49, 2.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:19<06:15, 737kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:19<05:11, 889kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:19<03:47, 1.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:41, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:21<05:08, 886kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:21<04:22, 1.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:21<03:15, 1.39MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:23<03:01, 1.48MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:23<02:53, 1.55MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:23<02:11, 2.03MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:25<02:16, 1.94MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:25<03:07, 1.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:25<02:34, 1.70MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<01:52, 2.33MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:27<02:25, 1.79MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:27<02:27, 1.77MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:27<01:52, 2.30MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<01:20, 3.18MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:29<13:00, 328kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:29<10:35, 403kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:29<07:44, 551kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:29<05:28, 774kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<04:52, 861kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<04:08, 1.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:31<03:03, 1.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:33<02:49, 1.46MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:33<03:24, 1.21MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<02:45, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:33<01:59, 2.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:35<02:25, 1.67MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:35<02:24, 1.69MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:35<01:49, 2.21MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:35<01:19, 3.04MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:37<04:03, 986kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<03:31, 1.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<02:36, 1.53MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:37<01:51, 2.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<05:52, 670kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:39<04:48, 815kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:39<03:31, 1.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<03:05, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<03:31, 1.09MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<02:48, 1.37MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<02:02, 1.88MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:43<02:24, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:43<02:23, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:43<01:48, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:43<01:17, 2.89MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:45<05:35, 666kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:45<04:33, 816kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:45<03:20, 1.11MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<02:55, 1.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:47<03:24, 1.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:47<02:42, 1.35MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:47<01:57, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<02:17, 1.56MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:49<02:14, 1.60MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<01:41, 2.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:49<01:12, 2.90MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:51<03:14, 1.08MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:51<03:30, 1.00MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:51<02:46, 1.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:51<01:59, 1.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:53<02:15, 1.52MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:53<02:11, 1.57MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:53<01:39, 2.06MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:53<01:11, 2.84MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 660M/862M [04:55<04:02, 835kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:55<03:24, 988kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:55<02:31, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<02:18, 1.43MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:57<02:12, 1.50MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:57<01:40, 1.96MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<01:42, 1.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:59<01:45, 1.84MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:59<01:20, 2.40MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<00:58, 3.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:00<02:04, 1.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<02:38, 1.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:01<02:07, 1.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<01:32, 2.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:02<01:52, 1.65MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:03<01:51, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:03<01:24, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:04<01:29, 2.03MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:05<01:35, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:05<01:14, 2.42MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:06<01:21, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<02:03, 1.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:07<01:42, 1.73MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<01:14, 2.36MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:08<01:36, 1.80MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:09<01:37, 1.78MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:09<01:14, 2.33MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:09<00:53, 3.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:10<02:44, 1.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<02:24, 1.17MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:11<01:46, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:11<01:15, 2.19MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:12<03:36, 764kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:12<03:00, 916kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:13<02:11, 1.25MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<01:34, 1.73MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:14<02:03, 1.31MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:14<02:26, 1.10MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:15<01:54, 1.40MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:15<01:23, 1.91MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:16<01:38, 1.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:16<01:35, 1.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:17<01:13, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:18<01:16, 2.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:18<01:21, 1.88MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:19<01:02, 2.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:19<00:44, 3.34MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:20<05:05, 486kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:20<04:26, 559kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:20<03:19, 744kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:21<02:20, 1.04MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:22<02:14, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:22<02:00, 1.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:22<01:29, 1.60MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:24<01:25, 1.64MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:24<01:48, 1.29MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:24<01:28, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:25<01:03, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:26<01:20, 1.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:26<01:19, 1.70MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:26<01:00, 2.23MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:27<00:43, 3.05MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:28<01:44, 1.27MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:28<01:35, 1.38MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:28<01:11, 1.83MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:29<00:50, 2.53MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:30<01:58, 1.08MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:30<01:44, 1.22MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<01:17, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:32<01:14, 1.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:32<01:13, 1.68MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:32<00:56, 2.17MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:34<00:59, 2.03MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<01:02, 1.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<00:48, 2.47MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:34<00:34, 3.36MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:36<01:27, 1.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:36<01:21, 1.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<01:01, 1.86MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:38<01:01, 1.83MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:38<01:02, 1.80MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:38<00:47, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:33, 3.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<04:28, 400kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<03:45, 477kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:40<02:46, 641kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:40<01:56, 900kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:42<01:47, 965kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:42<01:32, 1.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<01:08, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:44<01:03, 1.56MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:44<01:01, 1.60MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:44<00:46, 2.11MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:44<00:32, 2.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<01:52, 845kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:46<01:34, 998kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<01:09, 1.35MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:46<00:48, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<02:11, 691kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:48<01:48, 838kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:48<01:18, 1.15MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<00:54, 1.61MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:50<01:52, 771kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:50<01:34, 920kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<01:08, 1.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<01:00, 1.36MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:52<00:57, 1.44MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<00:42, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:52<00:29, 2.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<03:42, 353kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<03:02, 429kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:54<02:14, 581kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<01:33, 817kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<01:22, 897kB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:56<01:11, 1.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<00:52, 1.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:35, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<01:53, 618kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:58<01:31, 764kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<01:06, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<00:55, 1.19MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:00<01:01, 1.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:00<00:47, 1.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<00:33, 1.88MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:37, 1.63MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:02<00:37, 1.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:02<00:27, 2.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<00:19, 2.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:04<00:46, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<00:42, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:04<00:31, 1.78MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:06<00:30, 1.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:06<00:30, 1.76MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:23, 2.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:15, 3.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<01:00, 816kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:50, 969kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:36, 1.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<00:25, 1.84MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:54, 830kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:10<00:46, 977kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:33, 1.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:22, 1.85MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:43, 956kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:12<00:37, 1.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:27, 1.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:12<00:18, 2.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:41, 894kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:14<00:35, 1.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:25, 1.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<00:17, 1.94MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:15<00:27, 1.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:16<00:23, 1.38MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<00:17, 1.84MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:12, 2.44MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:16, 1.74MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:18<00:21, 1.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:18<00:17, 1.66MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:11, 2.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:19<00:13, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:13, 1.84MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:10, 2.35MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:21<00:09, 2.13MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:22<00:10, 1.99MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:07, 2.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:04, 3.53MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:23<16:26, 16.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<11:23, 23.9kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<07:33, 34.2kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:24<04:32, 48.8kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:25<03:06, 66.9kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<02:10, 94.0kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<01:24, 133kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:27<00:45, 185kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:27<00:32, 252kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:20, 354kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:29<00:09, 466kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:29<00:06, 596kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:03, 821kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:31<00:00, 977kB/s].vector_cache/glove.6B.zip: 862MB [06:31, 2.20MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 722/400000 [00:00<00:55, 7212.18it/s]  0%|          | 1468/400000 [00:00<00:54, 7282.23it/s]  1%|          | 2239/400000 [00:00<00:53, 7403.77it/s]  1%|          | 2994/400000 [00:00<00:53, 7446.32it/s]  1%|          | 3751/400000 [00:00<00:52, 7482.47it/s]  1%|          | 4511/400000 [00:00<00:52, 7516.92it/s]  1%|▏         | 5253/400000 [00:00<00:52, 7486.34it/s]  1%|▏         | 5974/400000 [00:00<00:53, 7398.55it/s]  2%|▏         | 6739/400000 [00:00<00:52, 7470.38it/s]  2%|▏         | 7534/400000 [00:01<00:51, 7607.07it/s]  2%|▏         | 8277/400000 [00:01<00:51, 7548.20it/s]  2%|▏         | 9044/400000 [00:01<00:51, 7583.11it/s]  2%|▏         | 9793/400000 [00:01<00:51, 7513.21it/s]  3%|▎         | 10555/400000 [00:01<00:51, 7542.04it/s]  3%|▎         | 11332/400000 [00:01<00:51, 7606.81it/s]  3%|▎         | 12100/400000 [00:01<00:50, 7626.04it/s]  3%|▎         | 12861/400000 [00:01<00:51, 7550.62it/s]  3%|▎         | 13638/400000 [00:01<00:50, 7614.14it/s]  4%|▎         | 14425/400000 [00:01<00:50, 7688.87it/s]  4%|▍         | 15194/400000 [00:02<00:50, 7627.78it/s]  4%|▍         | 15993/400000 [00:02<00:49, 7732.30it/s]  4%|▍         | 16767/400000 [00:02<00:50, 7635.84it/s]  4%|▍         | 17540/400000 [00:02<00:49, 7662.83it/s]  5%|▍         | 18318/400000 [00:02<00:49, 7695.33it/s]  5%|▍         | 19097/400000 [00:02<00:49, 7722.64it/s]  5%|▍         | 19870/400000 [00:02<00:49, 7639.21it/s]  5%|▌         | 20635/400000 [00:02<00:50, 7484.76it/s]  5%|▌         | 21385/400000 [00:02<00:50, 7430.74it/s]  6%|▌         | 22132/400000 [00:02<00:50, 7437.78it/s]  6%|▌         | 22890/400000 [00:03<00:50, 7479.40it/s]  6%|▌         | 23682/400000 [00:03<00:49, 7603.59it/s]  6%|▌         | 24444/400000 [00:03<00:49, 7554.95it/s]  6%|▋         | 25223/400000 [00:03<00:49, 7622.25it/s]  6%|▋         | 25986/400000 [00:03<00:50, 7424.16it/s]  7%|▋         | 26778/400000 [00:03<00:49, 7563.93it/s]  7%|▋         | 27580/400000 [00:03<00:48, 7694.22it/s]  7%|▋         | 28352/400000 [00:03<00:48, 7661.86it/s]  7%|▋         | 29120/400000 [00:03<00:48, 7662.85it/s]  7%|▋         | 29913/400000 [00:03<00:47, 7739.02it/s]  8%|▊         | 30702/400000 [00:04<00:47, 7782.49it/s]  8%|▊         | 31501/400000 [00:04<00:46, 7842.97it/s]  8%|▊         | 32286/400000 [00:04<00:47, 7788.24it/s]  8%|▊         | 33070/400000 [00:04<00:47, 7800.89it/s]  8%|▊         | 33851/400000 [00:04<00:46, 7799.16it/s]  9%|▊         | 34637/400000 [00:04<00:46, 7813.55it/s]  9%|▉         | 35419/400000 [00:04<00:48, 7553.21it/s]  9%|▉         | 36177/400000 [00:04<00:48, 7442.95it/s]  9%|▉         | 36974/400000 [00:04<00:47, 7590.39it/s]  9%|▉         | 37758/400000 [00:04<00:47, 7662.79it/s] 10%|▉         | 38526/400000 [00:05<00:47, 7593.61it/s] 10%|▉         | 39287/400000 [00:05<00:47, 7590.47it/s] 10%|█         | 40049/400000 [00:05<00:47, 7598.61it/s] 10%|█         | 40832/400000 [00:05<00:46, 7666.55it/s] 10%|█         | 41622/400000 [00:05<00:46, 7733.05it/s] 11%|█         | 42413/400000 [00:05<00:45, 7783.29it/s] 11%|█         | 43192/400000 [00:05<00:45, 7783.27it/s] 11%|█         | 43971/400000 [00:05<00:46, 7656.84it/s] 11%|█         | 44738/400000 [00:05<00:46, 7629.85it/s] 11%|█▏        | 45509/400000 [00:05<00:46, 7652.43it/s] 12%|█▏        | 46275/400000 [00:06<00:46, 7653.94it/s] 12%|█▏        | 47041/400000 [00:06<00:46, 7643.46it/s] 12%|█▏        | 47806/400000 [00:06<00:46, 7522.45it/s] 12%|█▏        | 48559/400000 [00:06<00:47, 7464.80it/s] 12%|█▏        | 49306/400000 [00:06<00:47, 7454.18it/s] 13%|█▎        | 50052/400000 [00:06<00:47, 7352.38it/s] 13%|█▎        | 50800/400000 [00:06<00:47, 7389.52it/s] 13%|█▎        | 51540/400000 [00:06<00:47, 7297.38it/s] 13%|█▎        | 52282/400000 [00:06<00:47, 7333.31it/s] 13%|█▎        | 53043/400000 [00:06<00:46, 7412.71it/s] 13%|█▎        | 53785/400000 [00:07<00:46, 7370.50it/s] 14%|█▎        | 54525/400000 [00:07<00:46, 7376.71it/s] 14%|█▍        | 55263/400000 [00:07<00:46, 7366.43it/s] 14%|█▍        | 56040/400000 [00:07<00:45, 7480.75it/s] 14%|█▍        | 56824/400000 [00:07<00:45, 7583.77it/s] 14%|█▍        | 57584/400000 [00:07<00:46, 7401.90it/s] 15%|█▍        | 58352/400000 [00:07<00:45, 7482.16it/s] 15%|█▍        | 59102/400000 [00:07<00:46, 7329.63it/s] 15%|█▍        | 59853/400000 [00:07<00:46, 7382.25it/s] 15%|█▌        | 60599/400000 [00:08<00:45, 7404.14it/s] 15%|█▌        | 61367/400000 [00:08<00:45, 7482.65it/s] 16%|█▌        | 62122/400000 [00:08<00:45, 7502.22it/s] 16%|█▌        | 62873/400000 [00:08<00:45, 7456.92it/s] 16%|█▌        | 63646/400000 [00:08<00:44, 7532.51it/s] 16%|█▌        | 64427/400000 [00:08<00:44, 7613.03it/s] 16%|█▋        | 65209/400000 [00:08<00:43, 7672.34it/s] 16%|█▋        | 65994/400000 [00:08<00:43, 7724.40it/s] 17%|█▋        | 66767/400000 [00:08<00:43, 7586.73it/s] 17%|█▋        | 67543/400000 [00:08<00:43, 7636.40it/s] 17%|█▋        | 68308/400000 [00:09<00:43, 7622.41it/s] 17%|█▋        | 69087/400000 [00:09<00:43, 7671.65it/s] 17%|█▋        | 69858/400000 [00:09<00:42, 7682.19it/s] 18%|█▊        | 70627/400000 [00:09<00:43, 7603.08it/s] 18%|█▊        | 71388/400000 [00:09<00:43, 7584.71it/s] 18%|█▊        | 72165/400000 [00:09<00:42, 7637.18it/s] 18%|█▊        | 72930/400000 [00:09<00:42, 7626.56it/s] 18%|█▊        | 73693/400000 [00:09<00:42, 7594.53it/s] 19%|█▊        | 74453/400000 [00:09<00:43, 7551.19it/s] 19%|█▉        | 75212/400000 [00:09<00:42, 7561.45it/s] 19%|█▉        | 75969/400000 [00:10<00:43, 7524.69it/s] 19%|█▉        | 76739/400000 [00:10<00:42, 7574.79it/s] 19%|█▉        | 77502/400000 [00:10<00:42, 7589.39it/s] 20%|█▉        | 78262/400000 [00:10<00:43, 7477.73it/s] 20%|█▉        | 79013/400000 [00:10<00:42, 7485.24it/s] 20%|█▉        | 79776/400000 [00:10<00:42, 7525.52it/s] 20%|██        | 80538/400000 [00:10<00:42, 7552.09it/s] 20%|██        | 81307/400000 [00:10<00:41, 7592.75it/s] 21%|██        | 82067/400000 [00:10<00:42, 7533.71it/s] 21%|██        | 82828/400000 [00:10<00:41, 7553.91it/s] 21%|██        | 83596/400000 [00:11<00:41, 7591.15it/s] 21%|██        | 84378/400000 [00:11<00:41, 7655.67it/s] 21%|██▏       | 85156/400000 [00:11<00:40, 7691.47it/s] 21%|██▏       | 85926/400000 [00:11<00:41, 7495.00it/s] 22%|██▏       | 86694/400000 [00:11<00:41, 7548.94it/s] 22%|██▏       | 87482/400000 [00:11<00:40, 7641.75it/s] 22%|██▏       | 88255/400000 [00:11<00:40, 7666.80it/s] 22%|██▏       | 89037/400000 [00:11<00:40, 7709.56it/s] 22%|██▏       | 89809/400000 [00:11<00:40, 7633.12it/s] 23%|██▎       | 90590/400000 [00:11<00:40, 7685.26it/s] 23%|██▎       | 91377/400000 [00:12<00:39, 7737.68it/s] 23%|██▎       | 92159/400000 [00:12<00:39, 7759.59it/s] 23%|██▎       | 92958/400000 [00:12<00:39, 7825.08it/s] 23%|██▎       | 93741/400000 [00:12<00:39, 7771.60it/s] 24%|██▎       | 94542/400000 [00:12<00:38, 7839.07it/s] 24%|██▍       | 95327/400000 [00:12<00:39, 7810.81it/s] 24%|██▍       | 96141/400000 [00:12<00:38, 7904.89it/s] 24%|██▍       | 96932/400000 [00:12<00:38, 7885.69it/s] 24%|██▍       | 97721/400000 [00:12<00:38, 7773.44it/s] 25%|██▍       | 98499/400000 [00:12<00:38, 7735.22it/s] 25%|██▍       | 99273/400000 [00:13<00:39, 7709.80it/s] 25%|██▌       | 100045/400000 [00:13<00:39, 7549.83it/s] 25%|██▌       | 100827/400000 [00:13<00:39, 7625.48it/s] 25%|██▌       | 101591/400000 [00:13<00:39, 7596.43it/s] 26%|██▌       | 102393/400000 [00:13<00:38, 7717.79it/s] 26%|██▌       | 103178/400000 [00:13<00:38, 7754.85it/s] 26%|██▌       | 103968/400000 [00:13<00:37, 7797.49it/s] 26%|██▌       | 104755/400000 [00:13<00:37, 7818.49it/s] 26%|██▋       | 105538/400000 [00:13<00:37, 7757.35it/s] 27%|██▋       | 106323/400000 [00:13<00:37, 7781.86it/s] 27%|██▋       | 107132/400000 [00:14<00:37, 7871.18it/s] 27%|██▋       | 107933/400000 [00:14<00:36, 7908.32it/s] 27%|██▋       | 108732/400000 [00:14<00:36, 7930.06it/s] 27%|██▋       | 109526/400000 [00:14<00:37, 7815.39it/s] 28%|██▊       | 110309/400000 [00:14<00:37, 7797.20it/s] 28%|██▊       | 111092/400000 [00:14<00:37, 7805.10it/s] 28%|██▊       | 111873/400000 [00:14<00:38, 7442.85it/s] 28%|██▊       | 112636/400000 [00:14<00:38, 7497.08it/s] 28%|██▊       | 113389/400000 [00:14<00:39, 7344.37it/s] 29%|██▊       | 114160/400000 [00:15<00:38, 7448.19it/s] 29%|██▊       | 114939/400000 [00:15<00:37, 7546.20it/s] 29%|██▉       | 115730/400000 [00:15<00:37, 7651.54it/s] 29%|██▉       | 116530/400000 [00:15<00:36, 7750.69it/s] 29%|██▉       | 117310/400000 [00:15<00:36, 7763.24it/s] 30%|██▉       | 118088/400000 [00:15<00:37, 7617.34it/s] 30%|██▉       | 118858/400000 [00:15<00:36, 7639.87it/s] 30%|██▉       | 119644/400000 [00:15<00:36, 7702.33it/s] 30%|███       | 120428/400000 [00:15<00:36, 7742.03it/s] 30%|███       | 121203/400000 [00:15<00:36, 7715.32it/s] 30%|███       | 121975/400000 [00:16<00:36, 7656.88it/s] 31%|███       | 122764/400000 [00:16<00:35, 7723.57it/s] 31%|███       | 123537/400000 [00:16<00:36, 7634.14it/s] 31%|███       | 124301/400000 [00:16<00:36, 7609.35it/s] 31%|███▏      | 125063/400000 [00:16<00:36, 7573.49it/s] 31%|███▏      | 125821/400000 [00:16<00:36, 7509.85it/s] 32%|███▏      | 126573/400000 [00:16<00:36, 7406.98it/s] 32%|███▏      | 127331/400000 [00:16<00:36, 7454.52it/s] 32%|███▏      | 128105/400000 [00:16<00:36, 7536.97it/s] 32%|███▏      | 128860/400000 [00:16<00:36, 7496.89it/s] 32%|███▏      | 129644/400000 [00:17<00:35, 7596.62it/s] 33%|███▎      | 130417/400000 [00:17<00:35, 7632.98it/s] 33%|███▎      | 131196/400000 [00:17<00:35, 7679.01it/s] 33%|███▎      | 131965/400000 [00:17<00:35, 7583.94it/s] 33%|███▎      | 132724/400000 [00:17<00:35, 7574.37it/s] 33%|███▎      | 133500/400000 [00:17<00:34, 7628.48it/s] 34%|███▎      | 134282/400000 [00:17<00:34, 7684.47it/s] 34%|███▍      | 135051/400000 [00:17<00:34, 7642.55it/s] 34%|███▍      | 135834/400000 [00:17<00:34, 7696.77it/s] 34%|███▍      | 136604/400000 [00:17<00:34, 7653.05it/s] 34%|███▍      | 137370/400000 [00:18<00:34, 7577.81it/s] 35%|███▍      | 138139/400000 [00:18<00:34, 7608.95it/s] 35%|███▍      | 138901/400000 [00:18<00:34, 7598.10it/s] 35%|███▍      | 139662/400000 [00:18<00:34, 7555.49it/s] 35%|███▌      | 140418/400000 [00:18<00:34, 7472.81it/s] 35%|███▌      | 141173/400000 [00:18<00:34, 7494.17it/s] 35%|███▌      | 141948/400000 [00:18<00:34, 7566.84it/s] 36%|███▌      | 142706/400000 [00:18<00:34, 7416.77it/s] 36%|███▌      | 143454/400000 [00:18<00:34, 7433.67it/s] 36%|███▌      | 144199/400000 [00:18<00:34, 7335.22it/s] 36%|███▌      | 144963/400000 [00:19<00:34, 7422.37it/s] 36%|███▋      | 145745/400000 [00:19<00:33, 7535.90it/s] 37%|███▋      | 146532/400000 [00:19<00:33, 7632.22it/s] 37%|███▋      | 147333/400000 [00:19<00:32, 7740.23it/s] 37%|███▋      | 148109/400000 [00:19<00:32, 7697.73it/s] 37%|███▋      | 148880/400000 [00:19<00:32, 7664.69it/s] 37%|███▋      | 149648/400000 [00:19<00:32, 7666.04it/s] 38%|███▊      | 150416/400000 [00:19<00:33, 7543.68it/s] 38%|███▊      | 151200/400000 [00:19<00:32, 7626.91it/s] 38%|███▊      | 151987/400000 [00:19<00:32, 7696.98it/s] 38%|███▊      | 152778/400000 [00:20<00:31, 7757.15it/s] 38%|███▊      | 153592/400000 [00:20<00:31, 7866.30it/s] 39%|███▊      | 154409/400000 [00:20<00:30, 7953.18it/s] 39%|███▉      | 155223/400000 [00:20<00:30, 8006.43it/s] 39%|███▉      | 156025/400000 [00:20<00:30, 7952.81it/s] 39%|███▉      | 156821/400000 [00:20<00:30, 7929.28it/s] 39%|███▉      | 157615/400000 [00:20<00:31, 7779.72it/s] 40%|███▉      | 158394/400000 [00:20<00:31, 7728.92it/s] 40%|███▉      | 159168/400000 [00:20<00:31, 7695.74it/s] 40%|███▉      | 159947/400000 [00:20<00:31, 7720.90it/s] 40%|████      | 160744/400000 [00:21<00:30, 7792.46it/s] 40%|████      | 161525/400000 [00:21<00:30, 7794.67it/s] 41%|████      | 162318/400000 [00:21<00:30, 7834.65it/s] 41%|████      | 163102/400000 [00:21<00:30, 7782.86it/s] 41%|████      | 163881/400000 [00:21<00:30, 7749.79it/s] 41%|████      | 164666/400000 [00:21<00:30, 7779.11it/s] 41%|████▏     | 165458/400000 [00:21<00:29, 7819.82it/s] 42%|████▏     | 166241/400000 [00:21<00:29, 7803.22it/s] 42%|████▏     | 167046/400000 [00:21<00:29, 7875.02it/s] 42%|████▏     | 167834/400000 [00:21<00:29, 7786.02it/s] 42%|████▏     | 168614/400000 [00:22<00:29, 7776.88it/s] 42%|████▏     | 169410/400000 [00:22<00:29, 7830.84it/s] 43%|████▎     | 170194/400000 [00:22<00:29, 7809.98it/s] 43%|████▎     | 170976/400000 [00:22<00:29, 7809.07it/s] 43%|████▎     | 171758/400000 [00:22<00:29, 7757.29it/s] 43%|████▎     | 172534/400000 [00:22<00:29, 7753.85it/s] 43%|████▎     | 173315/400000 [00:22<00:29, 7768.23it/s] 44%|████▎     | 174109/400000 [00:22<00:28, 7817.29it/s] 44%|████▎     | 174891/400000 [00:22<00:28, 7786.94it/s] 44%|████▍     | 175670/400000 [00:23<00:28, 7754.05it/s] 44%|████▍     | 176446/400000 [00:23<00:29, 7633.53it/s] 44%|████▍     | 177217/400000 [00:23<00:29, 7653.86it/s] 45%|████▍     | 178003/400000 [00:23<00:28, 7713.62it/s] 45%|████▍     | 178809/400000 [00:23<00:28, 7812.42it/s] 45%|████▍     | 179591/400000 [00:23<00:28, 7733.61it/s] 45%|████▌     | 180365/400000 [00:23<00:28, 7611.15it/s] 45%|████▌     | 181153/400000 [00:23<00:28, 7688.84it/s] 45%|████▌     | 181923/400000 [00:23<00:28, 7640.21it/s] 46%|████▌     | 182709/400000 [00:23<00:28, 7703.76it/s] 46%|████▌     | 183480/400000 [00:24<00:28, 7684.31it/s] 46%|████▌     | 184274/400000 [00:24<00:27, 7759.24it/s] 46%|████▋     | 185051/400000 [00:24<00:27, 7732.28it/s] 46%|████▋     | 185844/400000 [00:24<00:27, 7789.62it/s] 47%|████▋     | 186640/400000 [00:24<00:27, 7838.69it/s] 47%|████▋     | 187425/400000 [00:24<00:27, 7766.02it/s] 47%|████▋     | 188202/400000 [00:24<00:27, 7686.72it/s] 47%|████▋     | 188972/400000 [00:24<00:27, 7619.58it/s] 47%|████▋     | 189735/400000 [00:24<00:27, 7580.27it/s] 48%|████▊     | 190495/400000 [00:24<00:27, 7585.39it/s] 48%|████▊     | 191254/400000 [00:25<00:27, 7550.96it/s] 48%|████▊     | 192010/400000 [00:25<00:27, 7528.09it/s] 48%|████▊     | 192771/400000 [00:25<00:27, 7550.49it/s] 48%|████▊     | 193527/400000 [00:25<00:27, 7548.27it/s] 49%|████▊     | 194300/400000 [00:25<00:27, 7601.19it/s] 49%|████▉     | 195071/400000 [00:25<00:26, 7631.90it/s] 49%|████▉     | 195835/400000 [00:25<00:26, 7619.38it/s] 49%|████▉     | 196610/400000 [00:25<00:26, 7657.95it/s] 49%|████▉     | 197380/400000 [00:25<00:26, 7668.89it/s] 50%|████▉     | 198147/400000 [00:25<00:26, 7638.83it/s] 50%|████▉     | 198919/400000 [00:26<00:26, 7662.66it/s] 50%|████▉     | 199690/400000 [00:26<00:26, 7674.69it/s] 50%|█████     | 200490/400000 [00:26<00:25, 7768.95it/s] 50%|█████     | 201268/400000 [00:26<00:25, 7649.58it/s] 51%|█████     | 202034/400000 [00:26<00:25, 7639.97it/s] 51%|█████     | 202816/400000 [00:26<00:25, 7689.27it/s] 51%|█████     | 203586/400000 [00:26<00:25, 7617.30it/s] 51%|█████     | 204365/400000 [00:26<00:25, 7668.25it/s] 51%|█████▏    | 205133/400000 [00:26<00:25, 7659.59it/s] 51%|█████▏    | 205912/400000 [00:26<00:25, 7696.78it/s] 52%|█████▏    | 206703/400000 [00:27<00:24, 7756.55it/s] 52%|█████▏    | 207479/400000 [00:27<00:25, 7665.14it/s] 52%|█████▏    | 208247/400000 [00:27<00:25, 7666.26it/s] 52%|█████▏    | 209014/400000 [00:27<00:25, 7608.23it/s] 52%|█████▏    | 209796/400000 [00:27<00:24, 7669.21it/s] 53%|█████▎    | 210603/400000 [00:27<00:24, 7785.04it/s] 53%|█████▎    | 211383/400000 [00:27<00:24, 7736.06it/s] 53%|█████▎    | 212178/400000 [00:27<00:24, 7797.91it/s] 53%|█████▎    | 212981/400000 [00:27<00:23, 7864.86it/s] 53%|█████▎    | 213768/400000 [00:27<00:23, 7864.34it/s] 54%|█████▎    | 214555/400000 [00:28<00:23, 7786.53it/s] 54%|█████▍    | 215335/400000 [00:28<00:24, 7578.96it/s] 54%|█████▍    | 216095/400000 [00:28<00:24, 7557.26it/s] 54%|█████▍    | 216871/400000 [00:28<00:24, 7616.69it/s] 54%|█████▍    | 217641/400000 [00:28<00:23, 7639.35it/s] 55%|█████▍    | 218434/400000 [00:28<00:23, 7721.33it/s] 55%|█████▍    | 219216/400000 [00:28<00:23, 7747.00it/s] 55%|█████▌    | 220011/400000 [00:28<00:23, 7806.29it/s] 55%|█████▌    | 220793/400000 [00:28<00:23, 7623.34it/s] 55%|█████▌    | 221559/400000 [00:28<00:23, 7634.23it/s] 56%|█████▌    | 222324/400000 [00:29<00:23, 7527.95it/s] 56%|█████▌    | 223078/400000 [00:29<00:23, 7437.11it/s] 56%|█████▌    | 223834/400000 [00:29<00:23, 7471.32it/s] 56%|█████▌    | 224631/400000 [00:29<00:23, 7612.36it/s] 56%|█████▋    | 225418/400000 [00:29<00:22, 7686.23it/s] 57%|█████▋    | 226196/400000 [00:29<00:22, 7712.93it/s] 57%|█████▋    | 226968/400000 [00:29<00:22, 7594.52it/s] 57%|█████▋    | 227729/400000 [00:29<00:23, 7486.32it/s] 57%|█████▋    | 228500/400000 [00:29<00:22, 7550.78it/s] 57%|█████▋    | 229256/400000 [00:30<00:22, 7535.09it/s] 58%|█████▊    | 230011/400000 [00:30<00:22, 7487.05it/s] 58%|█████▊    | 230761/400000 [00:30<00:22, 7460.85it/s] 58%|█████▊    | 231519/400000 [00:30<00:22, 7495.32it/s] 58%|█████▊    | 232272/400000 [00:30<00:22, 7504.99it/s] 58%|█████▊    | 233035/400000 [00:30<00:22, 7540.45it/s] 58%|█████▊    | 233792/400000 [00:30<00:22, 7547.87it/s] 59%|█████▊    | 234547/400000 [00:30<00:22, 7520.37it/s] 59%|█████▉    | 235330/400000 [00:30<00:21, 7609.52it/s] 59%|█████▉    | 236107/400000 [00:30<00:21, 7653.65it/s] 59%|█████▉    | 236873/400000 [00:31<00:21, 7619.47it/s] 59%|█████▉    | 237636/400000 [00:31<00:21, 7549.11it/s] 60%|█████▉    | 238392/400000 [00:31<00:21, 7435.98it/s] 60%|█████▉    | 239159/400000 [00:31<00:21, 7503.07it/s] 60%|█████▉    | 239913/400000 [00:31<00:21, 7511.90it/s] 60%|██████    | 240665/400000 [00:31<00:21, 7453.79it/s] 60%|██████    | 241428/400000 [00:31<00:21, 7504.68it/s] 61%|██████    | 242179/400000 [00:31<00:21, 7414.85it/s] 61%|██████    | 242930/400000 [00:31<00:21, 7442.84it/s] 61%|██████    | 243702/400000 [00:31<00:20, 7523.77it/s] 61%|██████    | 244488/400000 [00:32<00:20, 7619.30it/s] 61%|██████▏   | 245251/400000 [00:32<00:20, 7604.59it/s] 62%|██████▏   | 246012/400000 [00:32<00:20, 7565.98it/s] 62%|██████▏   | 246803/400000 [00:32<00:19, 7664.63it/s] 62%|██████▏   | 247592/400000 [00:32<00:19, 7729.71it/s] 62%|██████▏   | 248366/400000 [00:32<00:19, 7725.62it/s] 62%|██████▏   | 249139/400000 [00:32<00:19, 7637.43it/s] 62%|██████▏   | 249904/400000 [00:32<00:20, 7474.29it/s] 63%|██████▎   | 250653/400000 [00:32<00:20, 7464.33it/s] 63%|██████▎   | 251444/400000 [00:32<00:19, 7590.96it/s] 63%|██████▎   | 252234/400000 [00:33<00:19, 7680.63it/s] 63%|██████▎   | 253004/400000 [00:33<00:19, 7642.09it/s] 63%|██████▎   | 253769/400000 [00:33<00:19, 7576.12it/s] 64%|██████▎   | 254540/400000 [00:33<00:19, 7613.99it/s] 64%|██████▍   | 255302/400000 [00:33<00:19, 7508.91it/s] 64%|██████▍   | 256054/400000 [00:33<00:19, 7508.52it/s] 64%|██████▍   | 256832/400000 [00:33<00:18, 7587.30it/s] 64%|██████▍   | 257592/400000 [00:33<00:18, 7572.53it/s] 65%|██████▍   | 258371/400000 [00:33<00:18, 7635.34it/s] 65%|██████▍   | 259152/400000 [00:33<00:18, 7686.17it/s] 65%|██████▍   | 259931/400000 [00:34<00:18, 7713.36it/s] 65%|██████▌   | 260703/400000 [00:34<00:18, 7698.41it/s] 65%|██████▌   | 261474/400000 [00:34<00:18, 7592.93it/s] 66%|██████▌   | 262266/400000 [00:34<00:17, 7685.87it/s] 66%|██████▌   | 263043/400000 [00:34<00:17, 7709.69it/s] 66%|██████▌   | 263815/400000 [00:34<00:17, 7660.42it/s] 66%|██████▌   | 264582/400000 [00:34<00:17, 7561.43it/s] 66%|██████▋   | 265339/400000 [00:34<00:18, 7447.36it/s] 67%|██████▋   | 266124/400000 [00:34<00:17, 7563.36it/s] 67%|██████▋   | 266917/400000 [00:34<00:17, 7668.85it/s] 67%|██████▋   | 267703/400000 [00:35<00:17, 7725.12it/s] 67%|██████▋   | 268486/400000 [00:35<00:16, 7754.54it/s] 67%|██████▋   | 269263/400000 [00:35<00:17, 7590.12it/s] 68%|██████▊   | 270040/400000 [00:35<00:17, 7641.77it/s] 68%|██████▊   | 270825/400000 [00:35<00:16, 7701.39it/s] 68%|██████▊   | 271630/400000 [00:35<00:16, 7802.31it/s] 68%|██████▊   | 272412/400000 [00:35<00:16, 7786.75it/s] 68%|██████▊   | 273192/400000 [00:35<00:16, 7635.31it/s] 68%|██████▊   | 273965/400000 [00:35<00:16, 7659.84it/s] 69%|██████▊   | 274732/400000 [00:35<00:16, 7646.45it/s] 69%|██████▉   | 275507/400000 [00:36<00:16, 7676.28it/s] 69%|██████▉   | 276276/400000 [00:36<00:16, 7642.60it/s] 69%|██████▉   | 277041/400000 [00:36<00:16, 7453.21it/s] 69%|██████▉   | 277788/400000 [00:36<00:16, 7350.53it/s] 70%|██████▉   | 278525/400000 [00:36<00:16, 7195.82it/s] 70%|██████▉   | 279279/400000 [00:36<00:16, 7294.46it/s] 70%|███████   | 280032/400000 [00:36<00:16, 7361.47it/s] 70%|███████   | 280770/400000 [00:36<00:16, 7307.70it/s] 70%|███████   | 281533/400000 [00:36<00:16, 7398.83it/s] 71%|███████   | 282282/400000 [00:37<00:15, 7424.60it/s] 71%|███████   | 283060/400000 [00:37<00:15, 7526.18it/s] 71%|███████   | 283814/400000 [00:37<00:15, 7417.36it/s] 71%|███████   | 284557/400000 [00:37<00:15, 7396.57it/s] 71%|███████▏  | 285338/400000 [00:37<00:15, 7514.52it/s] 72%|███████▏  | 286092/400000 [00:37<00:15, 7520.39it/s] 72%|███████▏  | 286852/400000 [00:37<00:15, 7542.02it/s] 72%|███████▏  | 287607/400000 [00:37<00:14, 7501.64it/s] 72%|███████▏  | 288358/400000 [00:37<00:14, 7471.50it/s] 72%|███████▏  | 289138/400000 [00:37<00:14, 7564.32it/s] 72%|███████▏  | 289918/400000 [00:38<00:14, 7631.99it/s] 73%|███████▎  | 290695/400000 [00:38<00:14, 7670.26it/s] 73%|███████▎  | 291473/400000 [00:38<00:14, 7700.32it/s] 73%|███████▎  | 292244/400000 [00:38<00:14, 7617.66it/s] 73%|███████▎  | 293007/400000 [00:38<00:14, 7546.27it/s] 73%|███████▎  | 293767/400000 [00:38<00:14, 7561.50it/s] 74%|███████▎  | 294524/400000 [00:38<00:13, 7540.48it/s] 74%|███████▍  | 295298/400000 [00:38<00:13, 7597.92it/s] 74%|███████▍  | 296059/400000 [00:38<00:13, 7512.80it/s] 74%|███████▍  | 296860/400000 [00:38<00:13, 7654.04it/s] 74%|███████▍  | 297652/400000 [00:39<00:13, 7730.52it/s] 75%|███████▍  | 298444/400000 [00:39<00:13, 7784.13it/s] 75%|███████▍  | 299241/400000 [00:39<00:12, 7837.21it/s] 75%|███████▌  | 300026/400000 [00:39<00:12, 7725.46it/s] 75%|███████▌  | 300800/400000 [00:39<00:12, 7721.91it/s] 75%|███████▌  | 301573/400000 [00:39<00:12, 7714.02it/s] 76%|███████▌  | 302356/400000 [00:39<00:12, 7745.64it/s] 76%|███████▌  | 303131/400000 [00:39<00:12, 7680.06it/s] 76%|███████▌  | 303900/400000 [00:39<00:13, 7309.90it/s] 76%|███████▌  | 304671/400000 [00:39<00:12, 7423.68it/s] 76%|███████▋  | 305447/400000 [00:40<00:12, 7521.18it/s] 77%|███████▋  | 306226/400000 [00:40<00:12, 7599.86it/s] 77%|███████▋  | 307003/400000 [00:40<00:12, 7648.93it/s] 77%|███████▋  | 307770/400000 [00:40<00:12, 7597.73it/s] 77%|███████▋  | 308531/400000 [00:40<00:12, 7535.69it/s] 77%|███████▋  | 309294/400000 [00:40<00:11, 7561.48it/s] 78%|███████▊  | 310051/400000 [00:40<00:11, 7553.50it/s] 78%|███████▊  | 310817/400000 [00:40<00:11, 7585.12it/s] 78%|███████▊  | 311576/400000 [00:40<00:11, 7501.66it/s] 78%|███████▊  | 312336/400000 [00:40<00:11, 7528.81it/s] 78%|███████▊  | 313111/400000 [00:41<00:11, 7592.72it/s] 78%|███████▊  | 313899/400000 [00:41<00:11, 7675.86it/s] 79%|███████▊  | 314695/400000 [00:41<00:10, 7758.56it/s] 79%|███████▉  | 315472/400000 [00:41<00:11, 7634.47it/s] 79%|███████▉  | 316237/400000 [00:41<00:11, 7505.59it/s] 79%|███████▉  | 316989/400000 [00:41<00:11, 7440.13it/s] 79%|███████▉  | 317755/400000 [00:41<00:10, 7503.22it/s] 80%|███████▉  | 318534/400000 [00:41<00:10, 7585.76it/s] 80%|███████▉  | 319294/400000 [00:41<00:10, 7481.63it/s] 80%|████████  | 320054/400000 [00:41<00:10, 7515.25it/s] 80%|████████  | 320832/400000 [00:42<00:10, 7590.81it/s] 80%|████████  | 321602/400000 [00:42<00:10, 7622.98it/s] 81%|████████  | 322384/400000 [00:42<00:10, 7678.98it/s] 81%|████████  | 323153/400000 [00:42<00:10, 7568.06it/s] 81%|████████  | 323924/400000 [00:42<00:09, 7608.10it/s] 81%|████████  | 324693/400000 [00:42<00:09, 7631.28it/s] 81%|████████▏ | 325459/400000 [00:42<00:09, 7639.24it/s] 82%|████████▏ | 326224/400000 [00:42<00:09, 7548.49it/s] 82%|████████▏ | 326985/400000 [00:42<00:09, 7565.18it/s] 82%|████████▏ | 327780/400000 [00:42<00:09, 7675.95it/s] 82%|████████▏ | 328549/400000 [00:43<00:09, 7526.38it/s] 82%|████████▏ | 329303/400000 [00:43<00:09, 7517.58it/s] 83%|████████▎ | 330076/400000 [00:43<00:09, 7578.16it/s] 83%|████████▎ | 330835/400000 [00:43<00:09, 7552.64it/s] 83%|████████▎ | 331599/400000 [00:43<00:09, 7577.97it/s] 83%|████████▎ | 332393/400000 [00:43<00:08, 7682.21it/s] 83%|████████▎ | 333173/400000 [00:43<00:08, 7716.88it/s] 83%|████████▎ | 333946/400000 [00:43<00:08, 7500.93it/s] 84%|████████▎ | 334698/400000 [00:43<00:08, 7472.26it/s] 84%|████████▍ | 335474/400000 [00:44<00:08, 7555.53it/s] 84%|████████▍ | 336251/400000 [00:44<00:08, 7618.19it/s] 84%|████████▍ | 337039/400000 [00:44<00:08, 7692.78it/s] 84%|████████▍ | 337810/400000 [00:44<00:08, 7646.46it/s] 85%|████████▍ | 338576/400000 [00:44<00:08, 7562.34it/s] 85%|████████▍ | 339355/400000 [00:44<00:07, 7627.29it/s] 85%|████████▌ | 340137/400000 [00:44<00:07, 7682.42it/s] 85%|████████▌ | 340918/400000 [00:44<00:07, 7718.79it/s] 85%|████████▌ | 341691/400000 [00:44<00:07, 7695.62it/s] 86%|████████▌ | 342461/400000 [00:44<00:07, 7618.18it/s] 86%|████████▌ | 343235/400000 [00:45<00:07, 7653.23it/s] 86%|████████▌ | 344001/400000 [00:45<00:07, 7613.42it/s] 86%|████████▌ | 344790/400000 [00:45<00:07, 7694.07it/s] 86%|████████▋ | 345583/400000 [00:45<00:07, 7760.01it/s] 87%|████████▋ | 346370/400000 [00:45<00:06, 7791.68it/s] 87%|████████▋ | 347172/400000 [00:45<00:06, 7856.83it/s] 87%|████████▋ | 347959/400000 [00:45<00:06, 7792.89it/s] 87%|████████▋ | 348739/400000 [00:45<00:06, 7772.07it/s] 87%|████████▋ | 349524/400000 [00:45<00:06, 7793.83it/s] 88%|████████▊ | 350304/400000 [00:45<00:06, 7782.26it/s] 88%|████████▊ | 351112/400000 [00:46<00:06, 7868.52it/s] 88%|████████▊ | 351910/400000 [00:46<00:06, 7900.78it/s] 88%|████████▊ | 352701/400000 [00:46<00:06, 7774.84it/s] 88%|████████▊ | 353480/400000 [00:46<00:05, 7764.52it/s] 89%|████████▊ | 354257/400000 [00:46<00:05, 7695.38it/s] 89%|████████▉ | 355027/400000 [00:46<00:05, 7519.67it/s] 89%|████████▉ | 355787/400000 [00:46<00:05, 7543.00it/s] 89%|████████▉ | 356543/400000 [00:46<00:05, 7383.79it/s] 89%|████████▉ | 357296/400000 [00:46<00:05, 7426.51it/s] 90%|████████▉ | 358040/400000 [00:46<00:05, 7380.52it/s] 90%|████████▉ | 358814/400000 [00:47<00:05, 7483.65it/s] 90%|████████▉ | 359564/400000 [00:47<00:05, 7454.38it/s] 90%|█████████ | 360330/400000 [00:47<00:05, 7513.04it/s] 90%|█████████ | 361084/400000 [00:47<00:05, 7516.58it/s] 90%|█████████ | 361837/400000 [00:47<00:05, 7516.20it/s] 91%|█████████ | 362601/400000 [00:47<00:04, 7552.02it/s] 91%|█████████ | 363361/400000 [00:47<00:04, 7565.94it/s] 91%|█████████ | 364145/400000 [00:47<00:04, 7645.39it/s] 91%|█████████ | 364910/400000 [00:47<00:04, 7618.08it/s] 91%|█████████▏| 365673/400000 [00:47<00:04, 7577.55it/s] 92%|█████████▏| 366462/400000 [00:48<00:04, 7666.18it/s] 92%|█████████▏| 367230/400000 [00:48<00:04, 7614.60it/s] 92%|█████████▏| 367992/400000 [00:48<00:04, 7579.89it/s] 92%|█████████▏| 368751/400000 [00:48<00:04, 7561.85it/s] 92%|█████████▏| 369508/400000 [00:48<00:04, 7505.76it/s] 93%|█████████▎| 370274/400000 [00:48<00:03, 7550.47it/s] 93%|█████████▎| 371040/400000 [00:48<00:03, 7581.54it/s] 93%|█████████▎| 371814/400000 [00:48<00:03, 7625.72it/s] 93%|█████████▎| 372583/400000 [00:48<00:03, 7643.06it/s] 93%|█████████▎| 373348/400000 [00:48<00:03, 7552.19it/s] 94%|█████████▎| 374118/400000 [00:49<00:03, 7594.22it/s] 94%|█████████▎| 374919/400000 [00:49<00:03, 7712.80it/s] 94%|█████████▍| 375702/400000 [00:49<00:03, 7747.45it/s] 94%|█████████▍| 376491/400000 [00:49<00:03, 7788.52it/s] 94%|█████████▍| 377271/400000 [00:49<00:02, 7735.14it/s] 95%|█████████▍| 378045/400000 [00:49<00:02, 7534.50it/s] 95%|█████████▍| 378835/400000 [00:49<00:02, 7638.55it/s] 95%|█████████▍| 379602/400000 [00:49<00:02, 7646.15it/s] 95%|█████████▌| 380369/400000 [00:49<00:02, 7652.34it/s] 95%|█████████▌| 381135/400000 [00:49<00:02, 7569.33it/s] 95%|█████████▌| 381893/400000 [00:50<00:02, 7448.94it/s] 96%|█████████▌| 382663/400000 [00:50<00:02, 7522.14it/s] 96%|█████████▌| 383426/400000 [00:50<00:02, 7550.62it/s] 96%|█████████▌| 384187/400000 [00:50<00:02, 7566.62it/s] 96%|█████████▌| 384945/400000 [00:50<00:01, 7528.22it/s] 96%|█████████▋| 385699/400000 [00:50<00:01, 7525.15it/s] 97%|█████████▋| 386473/400000 [00:50<00:01, 7586.35it/s] 97%|█████████▋| 387232/400000 [00:50<00:01, 7583.03it/s] 97%|█████████▋| 388008/400000 [00:50<00:01, 7632.50it/s] 97%|█████████▋| 388772/400000 [00:51<00:01, 7559.89it/s] 97%|█████████▋| 389541/400000 [00:51<00:01, 7597.71it/s] 98%|█████████▊| 390302/400000 [00:51<00:01, 7599.61it/s] 98%|█████████▊| 391066/400000 [00:51<00:01, 7610.25it/s] 98%|█████████▊| 391842/400000 [00:51<00:01, 7653.77it/s] 98%|█████████▊| 392608/400000 [00:51<00:00, 7523.69it/s] 98%|█████████▊| 393361/400000 [00:51<00:00, 7504.54it/s] 99%|█████████▊| 394143/400000 [00:51<00:00, 7594.71it/s] 99%|█████████▊| 394904/400000 [00:51<00:00, 7275.84it/s] 99%|█████████▉| 395635/400000 [00:51<00:00, 7272.91it/s] 99%|█████████▉| 396365/400000 [00:52<00:00, 7235.02it/s] 99%|█████████▉| 397147/400000 [00:52<00:00, 7399.93it/s] 99%|█████████▉| 397925/400000 [00:52<00:00, 7508.55it/s]100%|█████████▉| 398708/400000 [00:52<00:00, 7600.17it/s]100%|█████████▉| 399491/400000 [00:52<00:00, 7666.60it/s]100%|█████████▉| 399999/400000 [00:52<00:00, 7619.14it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb8cb241a90> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011743053090418868 	 Accuracy: 48
Train Epoch: 1 	 Loss: 0.011370479661884116 	 Accuracy: 50

  model saves at 50% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15783 out of table with 15654 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15783 out of table with 15654 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-13 02:25:44.990586: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 02:25:44.996491: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 02:25:44.997453: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ba0cbeaac0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 02:25:44.997473: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fb8746d3c50> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.6666 - accuracy: 0.5000
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7740 - accuracy: 0.4930
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.8200 - accuracy: 0.4900 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.8890 - accuracy: 0.4855
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8384 - accuracy: 0.4888
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8736 - accuracy: 0.4865
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.8200 - accuracy: 0.4900
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.7701 - accuracy: 0.4933
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7876 - accuracy: 0.4921
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7479 - accuracy: 0.4947
11000/25000 [============>.................] - ETA: 4s - loss: 7.7335 - accuracy: 0.4956
12000/25000 [=============>................] - ETA: 4s - loss: 7.7254 - accuracy: 0.4962
13000/25000 [==============>...............] - ETA: 4s - loss: 7.7221 - accuracy: 0.4964
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7082 - accuracy: 0.4973
15000/25000 [=================>............] - ETA: 3s - loss: 7.6809 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6628 - accuracy: 0.5002
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6450 - accuracy: 0.5014
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6607 - accuracy: 0.5004
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6537 - accuracy: 0.5008
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6613 - accuracy: 0.5003
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6622 - accuracy: 0.5003
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6604 - accuracy: 0.5004
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6606 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6609 - accuracy: 0.5004
25000/25000 [==============================] - 10s 406us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fb82b0fb8d0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fb8d2997160> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.8047 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.8287 - val_crf_viterbi_accuracy: 0.0000e+00

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

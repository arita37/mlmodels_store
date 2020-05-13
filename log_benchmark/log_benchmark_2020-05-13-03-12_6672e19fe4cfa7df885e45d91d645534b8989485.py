
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fa2a80c7fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 03:12:25.225936
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 03:12:25.229909
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 03:12:25.233248
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 03:12:25.236492
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fa2b3e91470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354005.8750
Epoch 2/10

1/1 [==============================] - 0s 101ms/step - loss: 262911.8125
Epoch 3/10

1/1 [==============================] - 0s 96ms/step - loss: 171695.6562
Epoch 4/10

1/1 [==============================] - 0s 91ms/step - loss: 102996.3984
Epoch 5/10

1/1 [==============================] - 0s 94ms/step - loss: 59866.5156
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 35674.8828
Epoch 7/10

1/1 [==============================] - 0s 99ms/step - loss: 22437.0430
Epoch 8/10

1/1 [==============================] - 0s 104ms/step - loss: 14934.1914
Epoch 9/10

1/1 [==============================] - 0s 100ms/step - loss: 10436.6611
Epoch 10/10

1/1 [==============================] - 0s 99ms/step - loss: 7687.0151

  #### Inference Need return ypred, ytrue ######################### 
[[-2.21250474e-01  2.62701273e-01 -3.72629523e-01  4.49131787e-01
  -1.25597215e+00  1.49490380e+00 -8.00297201e-01 -8.59576762e-01
  -4.78532523e-01 -4.08881903e-01  3.12082559e-01  1.15087545e+00
  -1.14960074e+00  8.22547972e-01 -1.23304367e+00 -7.22232103e-01
  -5.09267926e-01 -1.49518818e-01 -1.69952065e-01 -1.96245313e-01
  -4.53849137e-01 -4.30102050e-01 -4.01814431e-02 -2.16862559e-01
   1.71563625e-01  8.21327806e-01  1.19387472e+00  8.98478746e-01
   3.35344672e-03 -1.35659993e+00 -7.63040364e-01  9.23210084e-01
   1.82564944e-01 -2.80378759e-02 -1.10438561e+00  6.29468799e-01
  -6.90583229e-01  2.64093041e-01 -3.20498884e-01  2.41994560e-01
   7.32114732e-01 -4.13427949e-01 -6.17634773e-01  3.01048487e-01
   1.39664173e-01 -2.33109444e-01 -2.53062993e-01  5.39498091e-01
  -8.52073967e-01 -3.94246459e-01  2.16745958e-01  1.83396637e+00
  -1.05100107e+00 -1.31133139e+00  1.46563649e-02  1.45682621e+00
   6.94817185e-01 -3.57041866e-01 -3.76335561e-01 -1.76531148e+00
   1.11288846e-01  3.87162179e-01  4.27210361e-01 -1.57874775e+00
   9.69253004e-01 -7.24910498e-01  9.67924237e-01  1.28438845e-01
  -6.61390424e-01 -2.56893009e-01  6.06202424e-01 -1.49864650e+00
  -6.83750570e-01 -2.72454321e-02 -2.33131349e-01  3.49735200e-01
   4.41648960e-02 -1.40319705e+00  7.76056647e-02  1.67956293e-01
  -5.93605280e-01  9.64885950e-02  9.30588603e-01  2.22862303e-01
   5.44830918e-01 -7.30424047e-01  6.69690788e-01  5.12233615e-01
  -3.55088264e-01  4.04770076e-01  3.67159545e-01 -1.03043616e+00
   3.35895002e-01 -1.43749440e+00  1.14157259e+00  1.44790697e+00
  -1.05036616e+00  8.59466136e-01  9.35083389e-01  3.34927648e-01
  -1.14783311e+00 -1.31017685e+00 -1.92747489e-02 -2.48215109e-01
   3.10259819e-01 -3.21156293e-01 -2.88164318e-01  6.83056891e-01
   1.03182673e+00 -3.04782271e-01 -1.35679632e-01 -1.41304329e-01
   1.10040665e-01  1.14949036e+00 -2.06557989e-01 -2.16592014e-01
   1.25153875e+00 -4.32730913e-01 -1.26635742e+00 -4.47982192e-01
  -3.57185215e-01  6.54681015e+00  8.11462593e+00  7.59021664e+00
   7.24674034e+00  7.49189568e+00  8.26864147e+00  7.57386589e+00
   9.38369083e+00  7.88679314e+00  8.09127140e+00  8.19641304e+00
   8.99211121e+00  8.42025280e+00  6.41616917e+00  7.11148930e+00
   7.18268538e+00  8.77819443e+00  7.09678745e+00  7.11987114e+00
   8.88096428e+00  7.92357922e+00  7.71338463e+00  6.83675337e+00
   7.42300320e+00  6.88091516e+00  6.83506441e+00  7.68835020e+00
   8.28858566e+00  7.54997206e+00  8.83504963e+00  7.02755642e+00
   6.63351059e+00  7.39607191e+00  7.89175940e+00  8.85781860e+00
   8.51552105e+00  8.07919979e+00  6.85615635e+00  6.00275326e+00
   7.55228710e+00  7.90544415e+00  7.84847546e+00  7.71385622e+00
   7.50294495e+00  5.85218430e+00  6.72725105e+00  8.07396698e+00
   8.37081146e+00  8.24172497e+00  7.77594614e+00  5.95887709e+00
   8.36602020e+00  7.84527254e+00  7.14838552e+00  7.83454561e+00
   8.60954475e+00  7.65082598e+00  7.50792313e+00  7.14558125e+00
   1.02584600e+00  3.83899689e-01  5.72070897e-01  4.18713450e-01
   4.62403178e-01  1.72965956e+00  2.72891784e+00  8.06988716e-01
   3.22938681e-01  1.33097875e+00  1.56653619e+00  1.34530008e+00
   9.47444499e-01  1.05529439e+00  8.00796211e-01  1.51811028e+00
   1.85445130e-01  7.99179077e-01  5.07315397e-01  7.53774047e-01
   1.65970516e+00  1.01030064e+00  4.60051715e-01  1.72599554e-01
   1.66934443e+00  1.11308134e+00  2.15803957e+00  1.13925588e+00
   2.15323472e+00  2.96130359e-01  7.09932566e-01  1.23394978e+00
   1.98255658e+00  4.85363662e-01  2.59476364e-01  1.45472896e+00
   9.03954327e-01  1.32799566e+00  8.96625876e-01  1.08917785e+00
   1.64745319e+00  4.88287091e-01  1.74863052e+00  2.09376431e+00
   7.50023246e-01  1.00567079e+00  1.02269816e+00  1.78647912e+00
   8.58715653e-01  5.01406252e-01  1.03346157e+00  4.19211030e-01
   1.42841506e+00  2.77827978e+00  2.27016687e+00  1.60101783e+00
   2.08512163e+00  1.51581573e+00  8.03747177e-01  2.51015902e-01
   1.96342087e+00  8.61241162e-01  4.41523850e-01  2.96884775e-01
   2.01349688e+00  1.41803074e+00  1.20726812e+00  2.56055737e+00
   6.82113886e-01  1.26735044e+00  3.89380097e-01  2.42574215e+00
   1.98754334e+00  4.16517675e-01  4.56719279e-01  4.91315484e-01
   2.05159187e+00  1.43058908e+00  5.15042841e-01  1.52918422e+00
   1.79703331e+00  4.72849607e-01  2.32273793e+00  5.32545745e-01
   1.53494406e+00  4.58895624e-01  2.25094175e+00  6.47016823e-01
   2.96590352e+00  1.36184347e+00  6.25513434e-01  1.19055510e+00
   8.53857636e-01  8.00243139e-01  2.88833976e-01  4.69720960e-01
   1.11038232e+00  2.21468163e+00  1.06862044e+00  1.71247208e+00
   1.93983400e+00  6.93289995e-01  8.38320374e-01  1.48956871e+00
   4.62544918e-01  8.88154387e-01  2.35304952e-01  4.19636250e-01
   8.98067653e-01  1.30072248e+00  3.60654175e-01  1.49769664e+00
   6.85123444e-01  2.00412226e+00  7.52571583e-01  1.64905643e+00
   1.26274586e+00  1.44926429e+00  1.38197231e+00  2.16608500e+00
   8.69099498e-02  7.01969624e+00  8.14453220e+00  7.64085293e+00
   7.34332085e+00  8.47126102e+00  7.53934145e+00  8.27306461e+00
   8.07293320e+00  8.21633625e+00  8.16429329e+00  7.82531404e+00
   7.98559952e+00  7.66206837e+00  8.09127235e+00  7.27309608e+00
   7.40960789e+00  6.40171862e+00  9.06711864e+00  7.53455305e+00
   7.46309328e+00  7.74587202e+00  6.76499224e+00  8.00881767e+00
   7.83085251e+00  8.83275509e+00  7.76019573e+00  7.79308462e+00
   7.87166691e+00  7.65722752e+00  8.88831902e+00  8.29419899e+00
   9.25366592e+00  8.82243824e+00  6.88888454e+00  8.21964073e+00
   8.16391945e+00  8.87992668e+00  7.51116371e+00  8.20088863e+00
   7.32888317e+00  7.28269863e+00  7.35760403e+00  6.86158800e+00
   8.54706669e+00  8.13716888e+00  8.43349743e+00  6.99262285e+00
   8.45642757e+00  7.03275633e+00  8.32760239e+00  8.37780857e+00
   7.29582214e+00  7.99895287e+00  8.07053185e+00  7.79158688e+00
   8.73712444e+00  7.48411465e+00  7.57522392e+00  9.18197441e+00
  -7.02896738e+00 -4.12109947e+00  8.22983360e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 03:12:35.202372
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.3935
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 03:12:35.206836
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8931.24
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 03:12:35.210687
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.0014
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 03:12:35.214528
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -798.856
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140336221614712
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140333709165232
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140333709165736
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140333709166240
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140333709166744
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140333709167248

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fa2afd10f60> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.492979
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.473828
grad_step = 000002, loss = 0.462020
grad_step = 000003, loss = 0.451102
grad_step = 000004, loss = 0.441370
grad_step = 000005, loss = 0.431865
grad_step = 000006, loss = 0.420374
grad_step = 000007, loss = 0.408865
grad_step = 000008, loss = 0.397468
grad_step = 000009, loss = 0.384744
grad_step = 000010, loss = 0.372051
grad_step = 000011, loss = 0.358987
grad_step = 000012, loss = 0.345867
grad_step = 000013, loss = 0.331964
grad_step = 000014, loss = 0.318237
grad_step = 000015, loss = 0.304844
grad_step = 000016, loss = 0.292110
grad_step = 000017, loss = 0.279853
grad_step = 000018, loss = 0.268466
grad_step = 000019, loss = 0.257614
grad_step = 000020, loss = 0.245969
grad_step = 000021, loss = 0.234056
grad_step = 000022, loss = 0.222026
grad_step = 000023, loss = 0.210084
grad_step = 000024, loss = 0.198476
grad_step = 000025, loss = 0.186187
grad_step = 000026, loss = 0.174197
grad_step = 000027, loss = 0.162755
grad_step = 000028, loss = 0.152195
grad_step = 000029, loss = 0.142234
grad_step = 000030, loss = 0.132445
grad_step = 000031, loss = 0.122891
grad_step = 000032, loss = 0.113611
grad_step = 000033, loss = 0.104832
grad_step = 000034, loss = 0.096284
grad_step = 000035, loss = 0.088190
grad_step = 000036, loss = 0.080596
grad_step = 000037, loss = 0.073477
grad_step = 000038, loss = 0.066850
grad_step = 000039, loss = 0.060507
grad_step = 000040, loss = 0.054608
grad_step = 000041, loss = 0.049079
grad_step = 000042, loss = 0.043981
grad_step = 000043, loss = 0.039332
grad_step = 000044, loss = 0.035135
grad_step = 000045, loss = 0.031274
grad_step = 000046, loss = 0.027779
grad_step = 000047, loss = 0.024610
grad_step = 000048, loss = 0.021739
grad_step = 000049, loss = 0.019100
grad_step = 000050, loss = 0.016771
grad_step = 000051, loss = 0.014741
grad_step = 000052, loss = 0.012960
grad_step = 000053, loss = 0.011387
grad_step = 000054, loss = 0.010042
grad_step = 000055, loss = 0.008874
grad_step = 000056, loss = 0.007859
grad_step = 000057, loss = 0.006983
grad_step = 000058, loss = 0.006254
grad_step = 000059, loss = 0.005656
grad_step = 000060, loss = 0.005190
grad_step = 000061, loss = 0.004831
grad_step = 000062, loss = 0.004526
grad_step = 000063, loss = 0.004230
grad_step = 000064, loss = 0.003929
grad_step = 000065, loss = 0.003696
grad_step = 000066, loss = 0.003555
grad_step = 000067, loss = 0.003461
grad_step = 000068, loss = 0.003363
grad_step = 000069, loss = 0.003259
grad_step = 000070, loss = 0.003162
grad_step = 000071, loss = 0.003077
grad_step = 000072, loss = 0.003005
grad_step = 000073, loss = 0.002955
grad_step = 000074, loss = 0.002910
grad_step = 000075, loss = 0.002849
grad_step = 000076, loss = 0.002779
grad_step = 000077, loss = 0.002722
grad_step = 000078, loss = 0.002676
grad_step = 000079, loss = 0.002632
grad_step = 000080, loss = 0.002582
grad_step = 000081, loss = 0.002530
grad_step = 000082, loss = 0.002489
grad_step = 000083, loss = 0.002456
grad_step = 000084, loss = 0.002419
grad_step = 000085, loss = 0.002375
grad_step = 000086, loss = 0.002333
grad_step = 000087, loss = 0.002299
grad_step = 000088, loss = 0.002274
grad_step = 000089, loss = 0.002251
grad_step = 000090, loss = 0.002225
grad_step = 000091, loss = 0.002201
grad_step = 000092, loss = 0.002182
grad_step = 000093, loss = 0.002173
grad_step = 000094, loss = 0.002175
grad_step = 000095, loss = 0.002197
grad_step = 000096, loss = 0.002242
grad_step = 000097, loss = 0.002310
grad_step = 000098, loss = 0.002268
grad_step = 000099, loss = 0.002194
grad_step = 000100, loss = 0.002183
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002227
grad_step = 000102, loss = 0.002209
grad_step = 000103, loss = 0.002132
grad_step = 000104, loss = 0.002141
grad_step = 000105, loss = 0.002156
grad_step = 000106, loss = 0.002092
grad_step = 000107, loss = 0.002043
grad_step = 000108, loss = 0.002084
grad_step = 000109, loss = 0.002123
grad_step = 000110, loss = 0.002072
grad_step = 000111, loss = 0.002020
grad_step = 000112, loss = 0.002040
grad_step = 000113, loss = 0.002065
grad_step = 000114, loss = 0.002031
grad_step = 000115, loss = 0.001991
grad_step = 000116, loss = 0.001998
grad_step = 000117, loss = 0.002019
grad_step = 000118, loss = 0.002006
grad_step = 000119, loss = 0.001978
grad_step = 000120, loss = 0.001976
grad_step = 000121, loss = 0.001995
grad_step = 000122, loss = 0.002002
grad_step = 000123, loss = 0.001989
grad_step = 000124, loss = 0.001986
grad_step = 000125, loss = 0.002021
grad_step = 000126, loss = 0.002083
grad_step = 000127, loss = 0.002135
grad_step = 000128, loss = 0.002147
grad_step = 000129, loss = 0.002092
grad_step = 000130, loss = 0.002024
grad_step = 000131, loss = 0.001954
grad_step = 000132, loss = 0.001939
grad_step = 000133, loss = 0.001986
grad_step = 000134, loss = 0.002026
grad_step = 000135, loss = 0.001998
grad_step = 000136, loss = 0.001919
grad_step = 000137, loss = 0.001902
grad_step = 000138, loss = 0.001941
grad_step = 000139, loss = 0.001954
grad_step = 000140, loss = 0.001930
grad_step = 000141, loss = 0.001913
grad_step = 000142, loss = 0.001924
grad_step = 000143, loss = 0.001925
grad_step = 000144, loss = 0.001901
grad_step = 000145, loss = 0.001881
grad_step = 000146, loss = 0.001897
grad_step = 000147, loss = 0.001947
grad_step = 000148, loss = 0.002009
grad_step = 000149, loss = 0.002112
grad_step = 000150, loss = 0.002276
grad_step = 000151, loss = 0.002259
grad_step = 000152, loss = 0.002048
grad_step = 000153, loss = 0.001864
grad_step = 000154, loss = 0.002048
grad_step = 000155, loss = 0.002122
grad_step = 000156, loss = 0.001925
grad_step = 000157, loss = 0.001971
grad_step = 000158, loss = 0.002007
grad_step = 000159, loss = 0.001891
grad_step = 000160, loss = 0.001998
grad_step = 000161, loss = 0.001942
grad_step = 000162, loss = 0.001845
grad_step = 000163, loss = 0.001979
grad_step = 000164, loss = 0.001920
grad_step = 000165, loss = 0.001836
grad_step = 000166, loss = 0.001922
grad_step = 000167, loss = 0.001879
grad_step = 000168, loss = 0.001840
grad_step = 000169, loss = 0.001898
grad_step = 000170, loss = 0.001847
grad_step = 000171, loss = 0.001817
grad_step = 000172, loss = 0.001875
grad_step = 000173, loss = 0.001834
grad_step = 000174, loss = 0.001810
grad_step = 000175, loss = 0.001850
grad_step = 000176, loss = 0.001815
grad_step = 000177, loss = 0.001813
grad_step = 000178, loss = 0.001846
grad_step = 000179, loss = 0.001824
grad_step = 000180, loss = 0.001850
grad_step = 000181, loss = 0.001932
grad_step = 000182, loss = 0.002011
grad_step = 000183, loss = 0.002178
grad_step = 000184, loss = 0.002308
grad_step = 000185, loss = 0.002126
grad_step = 000186, loss = 0.001859
grad_step = 000187, loss = 0.001849
grad_step = 000188, loss = 0.001991
grad_step = 000189, loss = 0.001991
grad_step = 000190, loss = 0.001860
grad_step = 000191, loss = 0.001848
grad_step = 000192, loss = 0.001898
grad_step = 000193, loss = 0.001878
grad_step = 000194, loss = 0.001826
grad_step = 000195, loss = 0.001850
grad_step = 000196, loss = 0.001846
grad_step = 000197, loss = 0.001811
grad_step = 000198, loss = 0.001819
grad_step = 000199, loss = 0.001826
grad_step = 000200, loss = 0.001806
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001788
grad_step = 000202, loss = 0.001780
grad_step = 000203, loss = 0.001796
grad_step = 000204, loss = 0.001797
grad_step = 000205, loss = 0.001761
grad_step = 000206, loss = 0.001754
grad_step = 000207, loss = 0.001775
grad_step = 000208, loss = 0.001773
grad_step = 000209, loss = 0.001750
grad_step = 000210, loss = 0.001736
grad_step = 000211, loss = 0.001743
grad_step = 000212, loss = 0.001756
grad_step = 000213, loss = 0.001750
grad_step = 000214, loss = 0.001727
grad_step = 000215, loss = 0.001720
grad_step = 000216, loss = 0.001730
grad_step = 000217, loss = 0.001736
grad_step = 000218, loss = 0.001730
grad_step = 000219, loss = 0.001719
grad_step = 000220, loss = 0.001711
grad_step = 000221, loss = 0.001710
grad_step = 000222, loss = 0.001714
grad_step = 000223, loss = 0.001715
grad_step = 000224, loss = 0.001710
grad_step = 000225, loss = 0.001705
grad_step = 000226, loss = 0.001702
grad_step = 000227, loss = 0.001700
grad_step = 000228, loss = 0.001698
grad_step = 000229, loss = 0.001696
grad_step = 000230, loss = 0.001693
grad_step = 000231, loss = 0.001688
grad_step = 000232, loss = 0.001683
grad_step = 000233, loss = 0.001681
grad_step = 000234, loss = 0.001680
grad_step = 000235, loss = 0.001679
grad_step = 000236, loss = 0.001679
grad_step = 000237, loss = 0.001680
grad_step = 000238, loss = 0.001681
grad_step = 000239, loss = 0.001686
grad_step = 000240, loss = 0.001701
grad_step = 000241, loss = 0.001738
grad_step = 000242, loss = 0.001826
grad_step = 000243, loss = 0.002012
grad_step = 000244, loss = 0.002297
grad_step = 000245, loss = 0.002436
grad_step = 000246, loss = 0.002172
grad_step = 000247, loss = 0.001787
grad_step = 000248, loss = 0.001815
grad_step = 000249, loss = 0.002011
grad_step = 000250, loss = 0.001949
grad_step = 000251, loss = 0.001765
grad_step = 000252, loss = 0.001780
grad_step = 000253, loss = 0.001875
grad_step = 000254, loss = 0.001823
grad_step = 000255, loss = 0.001694
grad_step = 000256, loss = 0.001730
grad_step = 000257, loss = 0.001814
grad_step = 000258, loss = 0.001743
grad_step = 000259, loss = 0.001640
grad_step = 000260, loss = 0.001751
grad_step = 000261, loss = 0.001767
grad_step = 000262, loss = 0.001638
grad_step = 000263, loss = 0.001684
grad_step = 000264, loss = 0.001750
grad_step = 000265, loss = 0.001650
grad_step = 000266, loss = 0.001650
grad_step = 000267, loss = 0.001689
grad_step = 000268, loss = 0.001649
grad_step = 000269, loss = 0.001639
grad_step = 000270, loss = 0.001639
grad_step = 000271, loss = 0.001623
grad_step = 000272, loss = 0.001633
grad_step = 000273, loss = 0.001618
grad_step = 000274, loss = 0.001591
grad_step = 000275, loss = 0.001613
grad_step = 000276, loss = 0.001610
grad_step = 000277, loss = 0.001578
grad_step = 000278, loss = 0.001581
grad_step = 000279, loss = 0.001594
grad_step = 000280, loss = 0.001571
grad_step = 000281, loss = 0.001563
grad_step = 000282, loss = 0.001572
grad_step = 000283, loss = 0.001561
grad_step = 000284, loss = 0.001549
grad_step = 000285, loss = 0.001555
grad_step = 000286, loss = 0.001552
grad_step = 000287, loss = 0.001539
grad_step = 000288, loss = 0.001541
grad_step = 000289, loss = 0.001550
grad_step = 000290, loss = 0.001570
grad_step = 000291, loss = 0.001649
grad_step = 000292, loss = 0.001696
grad_step = 000293, loss = 0.001737
grad_step = 000294, loss = 0.001726
grad_step = 000295, loss = 0.001710
grad_step = 000296, loss = 0.001701
grad_step = 000297, loss = 0.001638
grad_step = 000298, loss = 0.001570
grad_step = 000299, loss = 0.001530
grad_step = 000300, loss = 0.001563
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001620
grad_step = 000302, loss = 0.001610
grad_step = 000303, loss = 0.001556
grad_step = 000304, loss = 0.001523
grad_step = 000305, loss = 0.001538
grad_step = 000306, loss = 0.001545
grad_step = 000307, loss = 0.001530
grad_step = 000308, loss = 0.001520
grad_step = 000309, loss = 0.001524
grad_step = 000310, loss = 0.001540
grad_step = 000311, loss = 0.001549
grad_step = 000312, loss = 0.001531
grad_step = 000313, loss = 0.001511
grad_step = 000314, loss = 0.001512
grad_step = 000315, loss = 0.001517
grad_step = 000316, loss = 0.001510
grad_step = 000317, loss = 0.001496
grad_step = 000318, loss = 0.001489
grad_step = 000319, loss = 0.001490
grad_step = 000320, loss = 0.001492
grad_step = 000321, loss = 0.001490
grad_step = 000322, loss = 0.001486
grad_step = 000323, loss = 0.001479
grad_step = 000324, loss = 0.001478
grad_step = 000325, loss = 0.001481
grad_step = 000326, loss = 0.001484
grad_step = 000327, loss = 0.001483
grad_step = 000328, loss = 0.001487
grad_step = 000329, loss = 0.001503
grad_step = 000330, loss = 0.001543
grad_step = 000331, loss = 0.001637
grad_step = 000332, loss = 0.001824
grad_step = 000333, loss = 0.002161
grad_step = 000334, loss = 0.002391
grad_step = 000335, loss = 0.002290
grad_step = 000336, loss = 0.001747
grad_step = 000337, loss = 0.001480
grad_step = 000338, loss = 0.001732
grad_step = 000339, loss = 0.001922
grad_step = 000340, loss = 0.001689
grad_step = 000341, loss = 0.001479
grad_step = 000342, loss = 0.001673
grad_step = 000343, loss = 0.001769
grad_step = 000344, loss = 0.001533
grad_step = 000345, loss = 0.001503
grad_step = 000346, loss = 0.001670
grad_step = 000347, loss = 0.001610
grad_step = 000348, loss = 0.001458
grad_step = 000349, loss = 0.001548
grad_step = 000350, loss = 0.001612
grad_step = 000351, loss = 0.001498
grad_step = 000352, loss = 0.001467
grad_step = 000353, loss = 0.001556
grad_step = 000354, loss = 0.001533
grad_step = 000355, loss = 0.001453
grad_step = 000356, loss = 0.001490
grad_step = 000357, loss = 0.001529
grad_step = 000358, loss = 0.001468
grad_step = 000359, loss = 0.001452
grad_step = 000360, loss = 0.001497
grad_step = 000361, loss = 0.001478
grad_step = 000362, loss = 0.001444
grad_step = 000363, loss = 0.001457
grad_step = 000364, loss = 0.001477
grad_step = 000365, loss = 0.001448
grad_step = 000366, loss = 0.001433
grad_step = 000367, loss = 0.001457
grad_step = 000368, loss = 0.001453
grad_step = 000369, loss = 0.001432
grad_step = 000370, loss = 0.001432
grad_step = 000371, loss = 0.001445
grad_step = 000372, loss = 0.001437
grad_step = 000373, loss = 0.001423
grad_step = 000374, loss = 0.001427
grad_step = 000375, loss = 0.001433
grad_step = 000376, loss = 0.001425
grad_step = 000377, loss = 0.001417
grad_step = 000378, loss = 0.001421
grad_step = 000379, loss = 0.001424
grad_step = 000380, loss = 0.001416
grad_step = 000381, loss = 0.001412
grad_step = 000382, loss = 0.001414
grad_step = 000383, loss = 0.001415
grad_step = 000384, loss = 0.001410
grad_step = 000385, loss = 0.001406
grad_step = 000386, loss = 0.001408
grad_step = 000387, loss = 0.001408
grad_step = 000388, loss = 0.001404
grad_step = 000389, loss = 0.001401
grad_step = 000390, loss = 0.001401
grad_step = 000391, loss = 0.001401
grad_step = 000392, loss = 0.001399
grad_step = 000393, loss = 0.001396
grad_step = 000394, loss = 0.001395
grad_step = 000395, loss = 0.001395
grad_step = 000396, loss = 0.001394
grad_step = 000397, loss = 0.001391
grad_step = 000398, loss = 0.001389
grad_step = 000399, loss = 0.001389
grad_step = 000400, loss = 0.001388
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001387
grad_step = 000402, loss = 0.001385
grad_step = 000403, loss = 0.001383
grad_step = 000404, loss = 0.001382
grad_step = 000405, loss = 0.001381
grad_step = 000406, loss = 0.001380
grad_step = 000407, loss = 0.001378
grad_step = 000408, loss = 0.001377
grad_step = 000409, loss = 0.001375
grad_step = 000410, loss = 0.001374
grad_step = 000411, loss = 0.001373
grad_step = 000412, loss = 0.001372
grad_step = 000413, loss = 0.001370
grad_step = 000414, loss = 0.001369
grad_step = 000415, loss = 0.001368
grad_step = 000416, loss = 0.001366
grad_step = 000417, loss = 0.001365
grad_step = 000418, loss = 0.001364
grad_step = 000419, loss = 0.001362
grad_step = 000420, loss = 0.001361
grad_step = 000421, loss = 0.001360
grad_step = 000422, loss = 0.001358
grad_step = 000423, loss = 0.001357
grad_step = 000424, loss = 0.001356
grad_step = 000425, loss = 0.001355
grad_step = 000426, loss = 0.001355
grad_step = 000427, loss = 0.001355
grad_step = 000428, loss = 0.001355
grad_step = 000429, loss = 0.001356
grad_step = 000430, loss = 0.001359
grad_step = 000431, loss = 0.001363
grad_step = 000432, loss = 0.001370
grad_step = 000433, loss = 0.001377
grad_step = 000434, loss = 0.001385
grad_step = 000435, loss = 0.001386
grad_step = 000436, loss = 0.001385
grad_step = 000437, loss = 0.001373
grad_step = 000438, loss = 0.001359
grad_step = 000439, loss = 0.001345
grad_step = 000440, loss = 0.001336
grad_step = 000441, loss = 0.001334
grad_step = 000442, loss = 0.001337
grad_step = 000443, loss = 0.001342
grad_step = 000444, loss = 0.001345
grad_step = 000445, loss = 0.001346
grad_step = 000446, loss = 0.001342
grad_step = 000447, loss = 0.001337
grad_step = 000448, loss = 0.001330
grad_step = 000449, loss = 0.001324
grad_step = 000450, loss = 0.001319
grad_step = 000451, loss = 0.001316
grad_step = 000452, loss = 0.001315
grad_step = 000453, loss = 0.001316
grad_step = 000454, loss = 0.001317
grad_step = 000455, loss = 0.001320
grad_step = 000456, loss = 0.001325
grad_step = 000457, loss = 0.001333
grad_step = 000458, loss = 0.001348
grad_step = 000459, loss = 0.001371
grad_step = 000460, loss = 0.001413
grad_step = 000461, loss = 0.001462
grad_step = 000462, loss = 0.001530
grad_step = 000463, loss = 0.001565
grad_step = 000464, loss = 0.001571
grad_step = 000465, loss = 0.001501
grad_step = 000466, loss = 0.001395
grad_step = 000467, loss = 0.001312
grad_step = 000468, loss = 0.001297
grad_step = 000469, loss = 0.001342
grad_step = 000470, loss = 0.001398
grad_step = 000471, loss = 0.001425
grad_step = 000472, loss = 0.001389
grad_step = 000473, loss = 0.001334
grad_step = 000474, loss = 0.001289
grad_step = 000475, loss = 0.001290
grad_step = 000476, loss = 0.001322
grad_step = 000477, loss = 0.001341
grad_step = 000478, loss = 0.001333
grad_step = 000479, loss = 0.001307
grad_step = 000480, loss = 0.001288
grad_step = 000481, loss = 0.001291
grad_step = 000482, loss = 0.001311
grad_step = 000483, loss = 0.001336
grad_step = 000484, loss = 0.001347
grad_step = 000485, loss = 0.001345
grad_step = 000486, loss = 0.001330
grad_step = 000487, loss = 0.001315
grad_step = 000488, loss = 0.001310
grad_step = 000489, loss = 0.001312
grad_step = 000490, loss = 0.001320
grad_step = 000491, loss = 0.001315
grad_step = 000492, loss = 0.001298
grad_step = 000493, loss = 0.001273
grad_step = 000494, loss = 0.001255
grad_step = 000495, loss = 0.001251
grad_step = 000496, loss = 0.001256
grad_step = 000497, loss = 0.001264
grad_step = 000498, loss = 0.001269
grad_step = 000499, loss = 0.001271
grad_step = 000500, loss = 0.001271
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001271
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

  date_run                              2020-05-13 03:12:53.742656
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.245074
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 03:12:53.748268
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.16685
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 03:12:53.755434
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.139975
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 03:12:53.760516
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.53534
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
0   2020-05-13 03:12:25.225936  ...    mean_absolute_error
1   2020-05-13 03:12:25.229909  ...     mean_squared_error
2   2020-05-13 03:12:25.233248  ...  median_absolute_error
3   2020-05-13 03:12:25.236492  ...               r2_score
4   2020-05-13 03:12:35.202372  ...    mean_absolute_error
5   2020-05-13 03:12:35.206836  ...     mean_squared_error
6   2020-05-13 03:12:35.210687  ...  median_absolute_error
7   2020-05-13 03:12:35.214528  ...               r2_score
8   2020-05-13 03:12:53.742656  ...    mean_absolute_error
9   2020-05-13 03:12:53.748268  ...     mean_squared_error
10  2020-05-13 03:12:53.755434  ...  median_absolute_error
11  2020-05-13 03:12:53.760516  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff3333a2d68> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███       | 3096576/9912422 [00:00<00:00, 30113147.29it/s]9920512it [00:00, 37249813.27it/s]                             
0it [00:00, ?it/s]32768it [00:00, 686189.50it/s]
0it [00:00, ?it/s]  4%|▍         | 73728/1648877 [00:00<00:02, 732985.17it/s]1654784it [00:00, 9459661.49it/s]                          
0it [00:00, ?it/s]8192it [00:00, 242799.27it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff2e5d5deb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff2e538d0f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff2e5d5deb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff2e52e3128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff2e2b1e518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff2e2b09780> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff2e5d5deb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff2e52a0748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff2e2b1e518> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff2e515b588> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9fffa88240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=d2ff603cb507f1e2e73d7c06fe2b32e0b94f7c57a903bcc3f40afc04c540a247
  Stored in directory: /tmp/pip-ephem-wheel-cache-pbrj4ec9/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f9ff5e0e080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 2s
 1851392/17464789 [==>...........................] - ETA: 0s
 8847360/17464789 [==============>...............] - ETA: 0s
16408576/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 03:14:20.332754: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 03:14:20.337347: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-13 03:14:20.337495: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560cd8c1bf00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 03:14:20.337508: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5440 - accuracy: 0.5080
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6283 - accuracy: 0.5025 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7177 - accuracy: 0.4967
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6743 - accuracy: 0.4995
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7464 - accuracy: 0.4948
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7152 - accuracy: 0.4968
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7433 - accuracy: 0.4950
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7433 - accuracy: 0.4950
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7058 - accuracy: 0.4974
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6758 - accuracy: 0.4994
11000/25000 [============>.................] - ETA: 3s - loss: 7.6750 - accuracy: 0.4995
12000/25000 [=============>................] - ETA: 3s - loss: 7.6781 - accuracy: 0.4992
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6737 - accuracy: 0.4995
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6622 - accuracy: 0.5003
15000/25000 [=================>............] - ETA: 2s - loss: 7.6441 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6455 - accuracy: 0.5014
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6486 - accuracy: 0.5012
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6777 - accuracy: 0.4993
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6690 - accuracy: 0.4998
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6735 - accuracy: 0.4996
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6733 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 7s 284us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 03:14:34.037220
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 03:14:34.037220  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:28:25, 11.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:16:01, 15.7kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:44:29, 22.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:31:38, 31.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:15:21, 45.4kB/s].vector_cache/glove.6B.zip:   1%|          | 9.18M/862M [00:01<3:39:24, 64.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.0M/862M [00:01<2:33:00, 92.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.6M/862M [00:01<1:46:36, 132kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.0M/862M [00:01<1:14:21, 188kB/s].vector_cache/glove.6B.zip:   3%|▎         | 27.1M/862M [00:01<51:48, 269kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 30.5M/862M [00:01<36:15, 382kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.3M/862M [00:02<25:19, 544kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.4M/862M [00:02<17:44, 773kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.3M/862M [00:02<12:25, 1.10MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.1M/862M [00:02<08:46, 1.55MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<06:41, 2.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<06:35, 2.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<06:30, 2.06MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.9M/862M [00:05<04:57, 2.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<06:00, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<05:38, 2.37MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<04:18, 3.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:08<06:00, 2.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<06:56, 1.91MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:09<05:30, 2.41MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.3M/862M [00:09<03:59, 3.31MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<12:52, 1.03MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<10:24, 1.27MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:11<07:37, 1.73MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<08:22, 1.57MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:12<07:12, 1.82MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:13<05:22, 2.44MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<06:50, 1.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<06:08, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:15<04:37, 2.82MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<06:18, 2.06MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<05:45, 2.26MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:16<04:21, 2.98MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:18<06:06, 2.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<05:36, 2.30MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:18<04:12, 3.06MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<05:59, 2.15MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:20<06:51, 1.88MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.5M/862M [00:20<05:21, 2.40MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:21<03:56, 3.26MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<08:23, 1.53MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:22<07:13, 1.77MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.6M/862M [00:22<05:22, 2.38MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<06:44, 1.89MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:24<07:20, 1.73MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.7M/862M [00:24<05:41, 2.23MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<04:09, 3.05MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<08:22, 1.51MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:10, 1.77MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<05:17, 2.39MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:38, 1.90MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:57, 2.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<04:29, 2.80MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:05, 2.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<05:33, 2.25MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<04:12, 2.97MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:52, 2.12MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:41, 1.86MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<05:14, 2.37MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<03:48, 3.25MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<11:04, 1.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<09:01, 1.37MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<06:37, 1.86MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:30, 1.64MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:49, 1.58MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:00, 2.05MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<04:21, 2.82MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<09:44, 1.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:06, 1.51MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<05:56, 2.06MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:00, 1.74MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<06:09, 1.98MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<04:36, 2.63MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:05, 1.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:29, 2.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:08, 2.92MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:44, 2.10MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:16, 2.28MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<03:59, 3.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:37, 2.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:09, 2.32MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<03:54, 3.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:33, 2.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<06:20, 1.88MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:03, 2.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<05:26, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:13, 2.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<04:36, 2.57MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:05, 2.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<03:38, 3.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<03:15, 3.63MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<02:55, 4.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<08:11, 1.44MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<08:49, 1.33MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:04, 1.66MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:31, 2.13MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<04:36, 2.55MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<03:47, 3.09MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<03:09, 3.70MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<19:53, 589kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<19:12, 610kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<14:31, 806kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<10:36, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<08:16, 1.41MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<06:10, 1.89MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<05:04, 2.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<03:57, 2.94MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<19:26, 599kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<18:02, 645kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<13:48, 843kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<10:06, 1.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<07:29, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<05:56, 1.95MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<04:30, 2.57MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<22:48, 508kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<20:05, 576kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<15:07, 765kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<11:05, 1.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<08:12, 1.40MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<06:09, 1.87MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<13:05, 879kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<13:14, 869kB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:00<10:16, 1.12MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<07:34, 1.51MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<05:48, 1.97MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<04:22, 2.61MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<16:34, 690kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<15:38, 731kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<11:54, 960kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<08:47, 1.30MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<06:33, 1.74MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<04:57, 2.29MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<26:53, 423kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<22:20, 509kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<16:34, 685kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<12:01, 943kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<08:49, 1.28MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<06:31, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<42:56, 263kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<33:48, 334kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<24:35, 459kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<17:37, 639kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<12:42, 885kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<09:14, 1.21MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<1:35:18, 118kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<1:10:11, 160kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<50:00, 224kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<35:23, 317kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<25:05, 446kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<21:19, 523kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<22:55, 487kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<17:47, 627kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<12:58, 859kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<09:28, 1.17MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<06:59, 1.59MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<12:02, 921kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<11:52, 934kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<09:11, 1.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:47, 1.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<05:10, 2.14MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<07:25, 1.49MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<08:38, 1.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:55, 1.59MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:11, 2.12MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<03:57, 2.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:12, 1.52MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<11:55, 918kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<09:59, 1.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<07:27, 1.47MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<05:29, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<04:10, 2.61MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<11:40, 932kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<11:00, 988kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<08:24, 1.29MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<06:08, 1.76MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<04:33, 2.37MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<15:46, 686kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<16:23, 660kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<12:50, 841kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<09:22, 1.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<06:46, 1.59MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<09:22, 1.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:54, 1.21MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:49, 1.57MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<05:01, 2.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<06:42, 1.59MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<09:07, 1.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<07:25, 1.44MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<05:29, 1.94MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<04:02, 2.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<29:20, 361kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<24:35, 431kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<18:13, 582kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<13:00, 813kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<11:25, 923kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<11:15, 936kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<08:44, 1.20MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<06:21, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<07:12, 1.45MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<08:05, 1.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:19, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<04:46, 2.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<03:27, 3.01MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<38:50, 268kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<30:03, 346kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<21:44, 478kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<15:21, 674kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<14:15, 724kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<12:49, 805kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<09:34, 1.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<06:53, 1.49MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<07:24, 1.38MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<07:43, 1.33MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:56, 1.73MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<04:17, 2.38MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:05, 1.44MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<07:16, 1.40MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<05:35, 1.82MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<04:04, 2.49MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:06, 1.66MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<06:27, 1.57MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<05:03, 2.00MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:39<03:39, 2.75MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<13:39, 736kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<11:38, 863kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<08:38, 1.16MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:41<06:08, 1.63MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<15:36, 640kB/s] .vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<12:49, 779kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<09:26, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<06:43, 1.48MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<12:56, 766kB/s] .vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<7:27:29, 22.2kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<5:13:55, 31.6kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [01:45<3:39:18, 45.0kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<2:35:39, 63.3kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<1:52:25, 87.6kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<1:19:31, 124kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<55:42, 176kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<41:33, 235kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<31:25, 311kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<22:33, 433kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<15:51, 613kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<19:06, 508kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<14:55, 650kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<10:48, 896kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<09:15, 1.04MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<08:28, 1.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<06:23, 1.51MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<04:35, 2.09MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<07:30, 1.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<06:38, 1.44MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:59, 1.91MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:14, 1.81MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<05:33, 1.71MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<04:16, 2.22MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<03:07, 3.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:09, 1.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:10, 1.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:46, 1.97MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:52, 1.92MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:12, 1.80MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<04:03, 2.30MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<02:55, 3.18MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<32:23, 287kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<24:26, 380kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<17:31, 529kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<13:43, 672kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<11:26, 806kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<08:23, 1.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<05:59, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<08:02, 1.14MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<07:22, 1.24MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:35, 1.63MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:24, 1.68MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<05:30, 1.65MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<04:13, 2.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:03, 2.95MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<08:57, 1.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<07:59, 1.13MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<06:01, 1.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<05:40, 1.58MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<05:44, 1.56MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:24, 2.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:32, 1.95MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<04:52, 1.82MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<03:50, 2.31MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:07, 2.14MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<04:34, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<03:37, 2.43MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<03:57, 2.21MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:30, 1.94MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<03:34, 2.44MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<03:55, 2.21MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<04:28, 1.94MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<03:32, 2.44MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<03:53, 2.21MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:22, 1.96MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<03:25, 2.51MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<02:29, 3.43MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<07:09, 1.19MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<06:37, 1.29MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<04:59, 1.71MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:34, 2.37MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<10:26, 811kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<09:01, 939kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<06:42, 1.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<06:03, 1.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:50, 1.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<04:28, 1.87MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:30, 1.85MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<04:45, 1.75MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<03:43, 2.23MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<03:57, 2.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<04:21, 1.90MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:26, 2.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:45, 2.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<04:15, 1.93MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<03:22, 2.43MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<03:41, 2.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<04:08, 1.96MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<03:13, 2.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<02:21, 3.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:45, 1.40MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<05:33, 1.45MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<04:14, 1.90MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:38<03:04, 2.60MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<05:21, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<05:20, 1.49MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<04:07, 1.94MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<03:12, 2.47MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<6:00:45, 22.0kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<4:12:57, 31.3kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<2:56:41, 44.7kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<2:04:53, 62.9kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<1:30:24, 86.9kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<1:03:55, 123kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<44:47, 175kB/s]  .vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<33:03, 236kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<24:40, 315kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<17:37, 441kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<13:31, 570kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<10:56, 705kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<08:01, 960kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<06:50, 1.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<06:14, 1.22MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:41, 1.62MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<03:21, 2.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<08:38, 878kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<07:32, 1.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<05:38, 1.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:09, 1.45MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:02, 1.49MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<03:53, 1.93MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<03:56, 1.89MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<04:10, 1.78MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<03:16, 2.26MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:56<02:21, 3.13MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<32:19, 228kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<24:04, 306kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<17:10, 428kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<12:04, 607kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<12:09, 601kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<09:58, 732kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<07:19, 996kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<06:17, 1.15MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<05:47, 1.25MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:24, 1.64MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<04:14, 1.69MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<04:21, 1.64MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:23, 2.11MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:31, 2.01MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:53, 1.83MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:03, 2.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:17, 2.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:43, 1.89MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<02:56, 2.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<03:11, 2.18MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:33, 1.95MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<02:46, 2.51MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:02, 3.40MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:23, 1.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:23, 1.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:24, 2.02MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:29, 1.95MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<03:47, 1.80MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<02:58, 2.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:11, 2.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:28, 1.94MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:44, 2.45MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:01, 2.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:23, 1.97MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<02:42, 2.46MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:58, 2.23MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:20, 1.98MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:38, 2.51MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<01:54, 3.45MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<12:19, 532kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<09:55, 659kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<07:15, 900kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<05:07, 1.27MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<07:52, 824kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<06:47, 954kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<05:03, 1.28MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:34, 1.40MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:25, 1.45MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:23, 1.88MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:24, 1.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:35, 1.77MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<02:49, 2.25MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<02:59, 2.10MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:20, 1.88MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:37, 2.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:51, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<03:11, 1.95MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<02:31, 2.45MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:31<01:48, 3.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<20:05, 306kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<15:13, 403kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<10:53, 562kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:33<07:40, 793kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<08:10, 743kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<06:52, 883kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<05:05, 1.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<03:47, 1.58MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<4:35:57, 21.7kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<3:13:25, 31.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<2:15:05, 44.2kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<1:35:09, 62.3kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<1:08:50, 86.1kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<48:37, 122kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<34:02, 173kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<24:56, 235kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<18:33, 316kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<13:14, 441kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<10:09, 571kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<08:12, 706kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<06:00, 962kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<05:06, 1.12MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<04:39, 1.23MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:30, 1.63MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<02:29, 2.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<07:43, 733kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<06:31, 866kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:50, 1.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:16, 1.31MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:03, 1.38MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:05, 1.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:03, 1.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:11, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:29, 2.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:38, 2.07MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:56, 1.85MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:18, 2.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:29, 2.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:49, 1.91MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<02:13, 2.41MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:25, 2.19MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:42, 1.96MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:08, 2.46MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:21, 2.23MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:39, 1.97MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:06, 2.48MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:19, 2.23MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:39, 1.95MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:03, 2.50MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<01:31, 3.35MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:49, 1.81MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:00, 1.70MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:18, 2.20MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<01:42, 2.95MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:40, 1.89MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<02:49, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:12, 2.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:21, 2.11MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:35, 1.91MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:01, 2.44MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<01:28, 3.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<05:27, 898kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<04:47, 1.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:35, 1.36MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:17, 1.47MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:13, 1.50MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:26, 1.97MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:46, 2.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:23, 1.41MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<03:16, 1.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:31, 1.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<02:31, 1.86MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:39, 1.76MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:05, 2.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:12, 2.10MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:27, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:56, 2.38MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:05, 2.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:20, 1.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:48, 2.50MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:18<01:19, 3.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:37, 1.24MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<03:23, 1.32MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:33, 1.75MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:20<01:49, 2.42MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:51, 1.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<04:32, 972kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:36, 1.22MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<02:36, 1.68MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:50, 1.53MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:48, 1.55MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:10, 1.99MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:12, 1.94MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:23, 1.79MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:52, 2.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<01:59, 2.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:12, 1.91MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:44, 2.41MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<01:53, 2.19MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:07, 1.96MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<01:40, 2.46MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:21, 3.01MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<2:57:51, 22.9kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<2:04:37, 32.7kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<1:27:01, 46.6kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<1:00:58, 65.8kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<44:12, 90.7kB/s]  .vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<31:17, 128kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<21:50, 182kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<16:06, 245kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<12:01, 328kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<08:34, 457kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<06:33, 590kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<05:19, 726kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:54, 987kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<03:19, 1.14MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<03:03, 1.25MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:18, 1.64MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:12, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<02:17, 1.63MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:46, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:50, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:44<01:59, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:32, 2.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:07, 3.23MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:34, 1.40MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<02:28, 1.45MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:54, 1.88MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:53, 1.86MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<02:01, 1.74MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:34, 2.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:39, 2.08MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<01:50, 1.87MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<01:27, 2.37MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:33, 2.17MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:45, 1.94MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:21, 2.48MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:00, 3.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:51, 1.78MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<01:56, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:30, 2.18MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:04, 3.02MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<08:38, 377kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<06:41, 486kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<04:47, 675kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<03:22, 950kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<03:51, 827kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<03:17, 965kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:26, 1.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:11, 1.42MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<02:07, 1.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:38, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:37, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<01:45, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:22, 2.21MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<00:59, 3.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<03:11, 934kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<02:49, 1.06MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<02:06, 1.41MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:55, 1.51MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:54, 1.53MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:26, 2.00MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:03, 2.71MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:37, 1.75MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:40, 1.69MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:18, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:21, 2.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:29, 1.85MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:09, 2.39MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<00:50, 3.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:33, 1.74MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:37, 1.67MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<01:15, 2.14MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:18, 2.03MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:24, 1.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:06, 2.36MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:11, 2.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:19, 1.95MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:02, 2.45MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:07, 2.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:16, 1.95MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:59, 2.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:17<00:43, 3.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:41, 1.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:39, 1.47MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<01:15, 1.91MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:15, 1.88MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:19, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<01:02, 2.26MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:21<00:44, 3.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<45:03, 50.9kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<31:55, 71.7kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<22:19, 102kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:23<15:25, 145kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<11:44, 189kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<08:38, 257kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<06:06, 361kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<04:16, 511kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<03:06, 696kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<1:39:59, 21.6kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<1:09:58, 30.7kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<48:23, 43.8kB/s]  .vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<33:51, 61.7kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<24:29, 85.2kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<17:17, 120kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<11:59, 171kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<08:44, 231kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<06:29, 310kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<04:36, 435kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:31<03:10, 617kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<03:41, 528kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<02:57, 659kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<02:08, 900kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:46, 1.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:36, 1.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:12, 1.55MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:07, 1.62MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<01:08, 1.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:37<00:52, 2.06MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:37<00:36, 2.86MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<04:22, 398kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<03:23, 513kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<02:26, 707kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:56, 865kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<01:40, 997kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<01:13, 1.35MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:52, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:07, 1.42MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:06, 1.44MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:51, 1.85MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:37, 2.50MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:48, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:52, 1.77MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:40, 2.25MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:41, 2.10MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:41, 2.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:31, 2.77MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:37, 2.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:41, 2.01MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:32, 2.54MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:35, 2.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:39, 2.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:31, 2.53MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:33, 2.25MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<00:33, 2.25MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:25, 2.90MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:31, 2.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:34, 2.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<00:26, 2.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:29, 2.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:32, 2.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:25, 2.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:27, 2.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:28, 2.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:25, 2.46MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:18, 3.27MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:26, 2.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:28, 2.04MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:22, 2.56MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:16, 3.40MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:26, 2.05MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:28, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:22, 2.45MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:23, 2.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:25, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:19, 2.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:20, 2.25MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:22, 2.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:17, 2.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:18, 2.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:19, 2.18MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:16, 2.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:12, 3.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:18, 2.08MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:19, 1.94MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:15, 2.47MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:15, 2.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:17, 2.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:13, 2.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:12<00:08, 3.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<01:14, 409kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:56, 531kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:39, 733kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:29, 888kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:25, 1.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:18, 1.39MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:14, 1.48MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:14, 1.54MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:10, 2.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:06, 2.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:14, 1.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:13, 1.36MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:09, 1.78MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:07, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:07, 1.75MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:22<00:05, 2.25MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:04, 2.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 1.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:03, 2.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:01, 3.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<04:22, 21.6kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<02:43, 30.8kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:59, 44.0kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:24, 61.1kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:16, 84.8kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:07, 120kB/s] .vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 853/400000 [00:00<00:46, 8523.69it/s]  0%|          | 1697/400000 [00:00<00:46, 8496.58it/s]  1%|          | 2544/400000 [00:00<00:46, 8487.47it/s]  1%|          | 3290/400000 [00:00<00:48, 8150.06it/s]  1%|          | 4094/400000 [00:00<00:48, 8115.43it/s]  1%|          | 4964/400000 [00:00<00:47, 8280.35it/s]  1%|▏         | 5837/400000 [00:00<00:46, 8408.40it/s]  2%|▏         | 6706/400000 [00:00<00:46, 8490.80it/s]  2%|▏         | 7582/400000 [00:00<00:45, 8564.28it/s]  2%|▏         | 8450/400000 [00:01<00:45, 8598.09it/s]  2%|▏         | 9316/400000 [00:01<00:45, 8614.48it/s]  3%|▎         | 10162/400000 [00:01<00:47, 8225.70it/s]  3%|▎         | 11061/400000 [00:01<00:46, 8438.90it/s]  3%|▎         | 11901/400000 [00:01<00:46, 8371.95it/s]  3%|▎         | 12736/400000 [00:01<00:47, 8188.96it/s]  3%|▎         | 13615/400000 [00:01<00:46, 8360.40it/s]  4%|▎         | 14481/400000 [00:01<00:45, 8445.70it/s]  4%|▍         | 15343/400000 [00:01<00:45, 8495.82it/s]  4%|▍         | 16234/400000 [00:01<00:44, 8615.82it/s]  4%|▍         | 17143/400000 [00:02<00:43, 8750.24it/s]  5%|▍         | 18029/400000 [00:02<00:43, 8781.24it/s]  5%|▍         | 19025/400000 [00:02<00:41, 9102.35it/s]  5%|▍         | 19939/400000 [00:02<00:42, 9025.16it/s]  5%|▌         | 20845/400000 [00:02<00:42, 8944.04it/s]  5%|▌         | 21742/400000 [00:02<00:42, 8877.01it/s]  6%|▌         | 22632/400000 [00:02<00:43, 8691.43it/s]  6%|▌         | 23517/400000 [00:02<00:43, 8735.78it/s]  6%|▌         | 24392/400000 [00:02<00:43, 8588.47it/s]  6%|▋         | 25325/400000 [00:02<00:42, 8797.73it/s]  7%|▋         | 26246/400000 [00:03<00:41, 8916.85it/s]  7%|▋         | 27140/400000 [00:03<00:42, 8847.01it/s]  7%|▋         | 28027/400000 [00:03<00:42, 8825.60it/s]  7%|▋         | 28911/400000 [00:03<00:42, 8700.38it/s]  7%|▋         | 29785/400000 [00:03<00:42, 8710.13it/s]  8%|▊         | 30663/400000 [00:03<00:42, 8728.94it/s]  8%|▊         | 31542/400000 [00:03<00:42, 8745.05it/s]  8%|▊         | 32417/400000 [00:03<00:42, 8738.64it/s]  8%|▊         | 33292/400000 [00:03<00:42, 8675.93it/s]  9%|▊         | 34168/400000 [00:03<00:42, 8700.68it/s]  9%|▉         | 35054/400000 [00:04<00:41, 8745.12it/s]  9%|▉         | 35929/400000 [00:04<00:41, 8680.70it/s]  9%|▉         | 36798/400000 [00:04<00:41, 8655.84it/s]  9%|▉         | 37664/400000 [00:04<00:41, 8639.84it/s] 10%|▉         | 38529/400000 [00:04<00:41, 8614.47it/s] 10%|▉         | 39391/400000 [00:04<00:42, 8472.44it/s] 10%|█         | 40239/400000 [00:04<00:42, 8438.04it/s] 10%|█         | 41110/400000 [00:04<00:42, 8515.86it/s] 10%|█         | 41981/400000 [00:04<00:41, 8570.43it/s] 11%|█         | 42839/400000 [00:04<00:42, 8470.78it/s] 11%|█         | 43687/400000 [00:05<00:42, 8350.66it/s] 11%|█         | 44563/400000 [00:05<00:41, 8467.90it/s] 11%|█▏        | 45411/400000 [00:05<00:42, 8441.52it/s] 12%|█▏        | 46284/400000 [00:05<00:41, 8524.50it/s] 12%|█▏        | 47149/400000 [00:05<00:41, 8561.00it/s] 12%|█▏        | 48022/400000 [00:05<00:40, 8609.41it/s] 12%|█▏        | 48908/400000 [00:05<00:40, 8681.87it/s] 12%|█▏        | 49777/400000 [00:05<00:40, 8650.58it/s] 13%|█▎        | 50644/400000 [00:05<00:40, 8656.10it/s] 13%|█▎        | 51510/400000 [00:05<00:41, 8481.10it/s] 13%|█▎        | 52360/400000 [00:06<00:41, 8281.50it/s] 13%|█▎        | 53190/400000 [00:06<00:42, 8240.20it/s] 14%|█▎        | 54036/400000 [00:06<00:41, 8303.31it/s] 14%|█▎        | 54886/400000 [00:06<00:41, 8361.05it/s] 14%|█▍        | 55749/400000 [00:06<00:40, 8438.68it/s] 14%|█▍        | 56621/400000 [00:06<00:40, 8519.89it/s] 14%|█▍        | 57499/400000 [00:06<00:39, 8595.62it/s] 15%|█▍        | 58360/400000 [00:06<00:41, 8305.64it/s] 15%|█▍        | 59194/400000 [00:06<00:41, 8271.19it/s] 15%|█▌        | 60023/400000 [00:07<00:41, 8211.04it/s] 15%|█▌        | 60848/400000 [00:07<00:41, 8222.39it/s] 15%|█▌        | 61672/400000 [00:07<00:41, 8206.93it/s] 16%|█▌        | 62494/400000 [00:07<00:41, 8156.47it/s] 16%|█▌        | 63311/400000 [00:07<00:41, 8153.91it/s] 16%|█▌        | 64158/400000 [00:07<00:40, 8244.47it/s] 16%|█▌        | 64999/400000 [00:07<00:40, 8291.24it/s] 16%|█▋        | 65840/400000 [00:07<00:40, 8324.98it/s] 17%|█▋        | 66718/400000 [00:07<00:39, 8454.96it/s] 17%|█▋        | 67577/400000 [00:07<00:39, 8493.00it/s] 17%|█▋        | 68427/400000 [00:08<00:39, 8307.31it/s] 17%|█▋        | 69300/400000 [00:08<00:39, 8428.16it/s] 18%|█▊        | 70170/400000 [00:08<00:38, 8507.50it/s] 18%|█▊        | 71044/400000 [00:08<00:38, 8575.21it/s] 18%|█▊        | 71930/400000 [00:08<00:37, 8656.97it/s] 18%|█▊        | 72801/400000 [00:08<00:37, 8671.02it/s] 18%|█▊        | 73669/400000 [00:08<00:37, 8606.48it/s] 19%|█▊        | 74549/400000 [00:08<00:37, 8660.90it/s] 19%|█▉        | 75427/400000 [00:08<00:37, 8696.02it/s] 19%|█▉        | 76298/400000 [00:08<00:37, 8697.66it/s] 19%|█▉        | 77169/400000 [00:09<00:37, 8688.77it/s] 20%|█▉        | 78044/400000 [00:09<00:36, 8706.57it/s] 20%|█▉        | 78924/400000 [00:09<00:36, 8731.39it/s] 20%|█▉        | 79803/400000 [00:09<00:36, 8748.54it/s] 20%|██        | 80678/400000 [00:09<00:36, 8718.98it/s] 20%|██        | 81550/400000 [00:09<00:36, 8638.62it/s] 21%|██        | 82415/400000 [00:09<00:37, 8542.66it/s] 21%|██        | 83289/400000 [00:09<00:36, 8598.39it/s] 21%|██        | 84150/400000 [00:09<00:36, 8556.67it/s] 21%|██▏       | 85016/400000 [00:09<00:36, 8585.06it/s] 21%|██▏       | 85875/400000 [00:10<00:36, 8584.26it/s] 22%|██▏       | 86753/400000 [00:10<00:36, 8641.96it/s] 22%|██▏       | 87630/400000 [00:10<00:35, 8678.02it/s] 22%|██▏       | 88504/400000 [00:10<00:35, 8696.12it/s] 22%|██▏       | 89383/400000 [00:10<00:35, 8721.92it/s] 23%|██▎       | 90256/400000 [00:10<00:35, 8718.13it/s] 23%|██▎       | 91128/400000 [00:10<00:35, 8689.83it/s] 23%|██▎       | 92006/400000 [00:10<00:35, 8715.38it/s] 23%|██▎       | 92883/400000 [00:10<00:35, 8728.98it/s] 23%|██▎       | 93756/400000 [00:10<00:35, 8713.21it/s] 24%|██▎       | 94628/400000 [00:11<00:35, 8526.70it/s] 24%|██▍       | 95496/400000 [00:11<00:35, 8568.84it/s] 24%|██▍       | 96374/400000 [00:11<00:35, 8628.71it/s] 24%|██▍       | 97238/400000 [00:11<00:35, 8564.82it/s] 25%|██▍       | 98101/400000 [00:11<00:35, 8583.37it/s] 25%|██▍       | 98960/400000 [00:11<00:35, 8518.66it/s] 25%|██▍       | 99813/400000 [00:11<00:35, 8480.41it/s] 25%|██▌       | 100664/400000 [00:11<00:35, 8487.38it/s] 25%|██▌       | 101531/400000 [00:11<00:34, 8540.97it/s] 26%|██▌       | 102399/400000 [00:11<00:34, 8581.96it/s] 26%|██▌       | 103263/400000 [00:12<00:34, 8596.51it/s] 26%|██▌       | 104149/400000 [00:12<00:34, 8671.83it/s] 26%|██▋       | 105017/400000 [00:12<00:34, 8493.53it/s] 26%|██▋       | 105876/400000 [00:12<00:34, 8520.84it/s] 27%|██▋       | 106729/400000 [00:12<00:34, 8428.79it/s] 27%|██▋       | 107593/400000 [00:12<00:34, 8490.45it/s] 27%|██▋       | 108461/400000 [00:12<00:34, 8544.67it/s] 27%|██▋       | 109326/400000 [00:12<00:33, 8573.92it/s] 28%|██▊       | 110205/400000 [00:12<00:33, 8637.09it/s] 28%|██▊       | 111070/400000 [00:12<00:34, 8286.72it/s] 28%|██▊       | 111902/400000 [00:13<00:35, 8082.36it/s] 28%|██▊       | 112747/400000 [00:13<00:35, 8187.45it/s] 28%|██▊       | 113597/400000 [00:13<00:34, 8277.83it/s] 29%|██▊       | 114452/400000 [00:13<00:34, 8356.32it/s] 29%|██▉       | 115308/400000 [00:13<00:33, 8413.81it/s] 29%|██▉       | 116163/400000 [00:13<00:33, 8448.99it/s] 29%|██▉       | 117009/400000 [00:13<00:33, 8367.73it/s] 29%|██▉       | 117872/400000 [00:13<00:33, 8442.22it/s] 30%|██▉       | 118754/400000 [00:13<00:32, 8551.14it/s] 30%|██▉       | 119620/400000 [00:13<00:32, 8583.44it/s] 30%|███       | 120498/400000 [00:14<00:32, 8639.48it/s] 30%|███       | 121363/400000 [00:14<00:32, 8624.32it/s] 31%|███       | 122226/400000 [00:14<00:32, 8570.55it/s] 31%|███       | 123084/400000 [00:14<00:32, 8417.52it/s] 31%|███       | 123953/400000 [00:14<00:32, 8494.95it/s] 31%|███       | 124812/400000 [00:14<00:32, 8522.94it/s] 31%|███▏      | 125677/400000 [00:14<00:32, 8559.43it/s] 32%|███▏      | 126534/400000 [00:14<00:32, 8512.85it/s] 32%|███▏      | 127386/400000 [00:14<00:32, 8508.07it/s] 32%|███▏      | 128238/400000 [00:15<00:32, 8435.78it/s] 32%|███▏      | 129082/400000 [00:15<00:32, 8373.51it/s] 32%|███▏      | 129948/400000 [00:15<00:31, 8457.42it/s] 33%|███▎      | 130813/400000 [00:15<00:31, 8511.77it/s] 33%|███▎      | 131669/400000 [00:15<00:31, 8523.82it/s] 33%|███▎      | 132527/400000 [00:15<00:31, 8540.22it/s] 33%|███▎      | 133382/400000 [00:15<00:31, 8535.10it/s] 34%|███▎      | 134236/400000 [00:15<00:31, 8501.59it/s] 34%|███▍      | 135096/400000 [00:15<00:31, 8528.64it/s] 34%|███▍      | 135961/400000 [00:15<00:30, 8564.19it/s] 34%|███▍      | 136849/400000 [00:16<00:30, 8654.97it/s] 34%|███▍      | 137715/400000 [00:16<00:30, 8651.38it/s] 35%|███▍      | 138581/400000 [00:16<00:30, 8645.79it/s] 35%|███▍      | 139446/400000 [00:16<00:30, 8623.59it/s] 35%|███▌      | 140309/400000 [00:16<00:30, 8596.27it/s] 35%|███▌      | 141169/400000 [00:16<00:30, 8588.43it/s] 36%|███▌      | 142028/400000 [00:16<00:30, 8558.54it/s] 36%|███▌      | 142884/400000 [00:16<00:30, 8434.52it/s] 36%|███▌      | 143733/400000 [00:16<00:30, 8450.70it/s] 36%|███▌      | 144597/400000 [00:16<00:30, 8505.53it/s] 36%|███▋      | 145457/400000 [00:17<00:29, 8533.00it/s] 37%|███▋      | 146316/400000 [00:17<00:29, 8549.75it/s] 37%|███▋      | 147174/400000 [00:17<00:29, 8556.33it/s] 37%|███▋      | 148032/400000 [00:17<00:29, 8562.49it/s] 37%|███▋      | 148889/400000 [00:17<00:29, 8431.70it/s] 37%|███▋      | 149760/400000 [00:17<00:29, 8511.35it/s] 38%|███▊      | 150612/400000 [00:17<00:29, 8488.12it/s] 38%|███▊      | 151491/400000 [00:17<00:28, 8574.12it/s] 38%|███▊      | 152364/400000 [00:17<00:28, 8620.02it/s] 38%|███▊      | 153249/400000 [00:17<00:28, 8685.86it/s] 39%|███▊      | 154134/400000 [00:18<00:28, 8731.86it/s] 39%|███▉      | 155008/400000 [00:18<00:28, 8628.52it/s] 39%|███▉      | 155872/400000 [00:18<00:28, 8566.50it/s] 39%|███▉      | 156768/400000 [00:18<00:28, 8678.45it/s] 39%|███▉      | 157637/400000 [00:18<00:28, 8642.62it/s] 40%|███▉      | 158525/400000 [00:18<00:27, 8710.88it/s] 40%|███▉      | 159420/400000 [00:18<00:27, 8779.39it/s] 40%|████      | 160299/400000 [00:18<00:27, 8772.74it/s] 40%|████      | 161177/400000 [00:18<00:28, 8396.80it/s] 41%|████      | 162054/400000 [00:18<00:27, 8504.47it/s] 41%|████      | 162932/400000 [00:19<00:27, 8584.49it/s] 41%|████      | 163820/400000 [00:19<00:27, 8669.95it/s] 41%|████      | 164701/400000 [00:19<00:27, 8710.86it/s] 41%|████▏     | 165596/400000 [00:19<00:26, 8780.33it/s] 42%|████▏     | 166476/400000 [00:19<00:26, 8769.47it/s] 42%|████▏     | 167373/400000 [00:19<00:26, 8827.29it/s] 42%|████▏     | 168257/400000 [00:19<00:26, 8771.58it/s] 42%|████▏     | 169139/400000 [00:19<00:26, 8783.50it/s] 43%|████▎     | 170018/400000 [00:19<00:26, 8732.07it/s] 43%|████▎     | 170906/400000 [00:19<00:26, 8773.63it/s] 43%|████▎     | 171784/400000 [00:20<00:26, 8593.47it/s] 43%|████▎     | 172645/400000 [00:20<00:26, 8516.93it/s] 43%|████▎     | 173509/400000 [00:20<00:26, 8551.07it/s] 44%|████▎     | 174389/400000 [00:20<00:26, 8623.15it/s] 44%|████▍     | 175266/400000 [00:20<00:25, 8664.96it/s] 44%|████▍     | 176145/400000 [00:20<00:25, 8699.84it/s] 44%|████▍     | 177023/400000 [00:20<00:25, 8723.12it/s] 44%|████▍     | 177896/400000 [00:20<00:25, 8700.58it/s] 45%|████▍     | 178769/400000 [00:20<00:25, 8707.94it/s] 45%|████▍     | 179644/400000 [00:20<00:25, 8719.45it/s] 45%|████▌     | 180517/400000 [00:21<00:25, 8716.16it/s] 45%|████▌     | 181398/400000 [00:21<00:25, 8743.03it/s] 46%|████▌     | 182273/400000 [00:21<00:24, 8710.70it/s] 46%|████▌     | 183153/400000 [00:21<00:24, 8736.16it/s] 46%|████▌     | 184033/400000 [00:21<00:24, 8752.86it/s] 46%|████▌     | 184949/400000 [00:21<00:24, 8869.36it/s] 46%|████▋     | 185837/400000 [00:21<00:24, 8857.77it/s] 47%|████▋     | 186724/400000 [00:21<00:24, 8658.52it/s] 47%|████▋     | 187592/400000 [00:21<00:24, 8584.52it/s] 47%|████▋     | 188452/400000 [00:21<00:24, 8567.09it/s] 47%|████▋     | 189310/400000 [00:22<00:24, 8492.95it/s] 48%|████▊     | 190174/400000 [00:22<00:24, 8535.74it/s] 48%|████▊     | 191029/400000 [00:22<00:24, 8528.02it/s] 48%|████▊     | 191908/400000 [00:22<00:24, 8604.20it/s] 48%|████▊     | 192785/400000 [00:22<00:23, 8650.58it/s] 48%|████▊     | 193655/400000 [00:22<00:23, 8664.58it/s] 49%|████▊     | 194529/400000 [00:22<00:23, 8684.84it/s] 49%|████▉     | 195398/400000 [00:22<00:23, 8684.24it/s] 49%|████▉     | 196278/400000 [00:22<00:23, 8717.88it/s] 49%|████▉     | 197163/400000 [00:22<00:23, 8756.99it/s] 50%|████▉     | 198039/400000 [00:23<00:23, 8747.26it/s] 50%|████▉     | 198914/400000 [00:23<00:23, 8733.45it/s] 50%|████▉     | 199790/400000 [00:23<00:22, 8738.95it/s] 50%|█████     | 200664/400000 [00:23<00:22, 8735.59it/s] 50%|█████     | 201538/400000 [00:23<00:22, 8718.63it/s] 51%|█████     | 202410/400000 [00:23<00:23, 8522.21it/s] 51%|█████     | 203264/400000 [00:23<00:23, 8449.81it/s] 51%|█████     | 204111/400000 [00:23<00:23, 8452.35it/s] 51%|█████▏    | 205017/400000 [00:23<00:22, 8625.86it/s] 51%|█████▏    | 205891/400000 [00:24<00:22, 8658.11it/s] 52%|█████▏    | 206773/400000 [00:24<00:22, 8705.42it/s] 52%|█████▏    | 207651/400000 [00:24<00:22, 8725.72it/s] 52%|█████▏    | 208525/400000 [00:24<00:22, 8670.83it/s] 52%|█████▏    | 209394/400000 [00:24<00:21, 8674.01it/s] 53%|█████▎    | 210262/400000 [00:24<00:21, 8674.98it/s] 53%|█████▎    | 211130/400000 [00:24<00:22, 8558.34it/s] 53%|█████▎    | 212008/400000 [00:24<00:21, 8622.89it/s] 53%|█████▎    | 212871/400000 [00:24<00:22, 8498.36it/s] 53%|█████▎    | 213722/400000 [00:24<00:22, 8395.30it/s] 54%|█████▎    | 214602/400000 [00:25<00:21, 8510.89it/s] 54%|█████▍    | 215489/400000 [00:25<00:21, 8613.65it/s] 54%|█████▍    | 216359/400000 [00:25<00:21, 8638.26it/s] 54%|█████▍    | 217224/400000 [00:25<00:21, 8366.05it/s] 55%|█████▍    | 218072/400000 [00:25<00:21, 8397.89it/s] 55%|█████▍    | 218951/400000 [00:25<00:21, 8509.89it/s] 55%|█████▍    | 219826/400000 [00:25<00:21, 8577.89it/s] 55%|█████▌    | 220702/400000 [00:25<00:20, 8629.04it/s] 55%|█████▌    | 221566/400000 [00:25<00:20, 8628.56it/s] 56%|█████▌    | 222453/400000 [00:25<00:20, 8698.80it/s] 56%|█████▌    | 223324/400000 [00:26<00:20, 8644.84it/s] 56%|█████▌    | 224195/400000 [00:26<00:20, 8663.61it/s] 56%|█████▋    | 225062/400000 [00:26<00:20, 8663.08it/s] 56%|█████▋    | 225929/400000 [00:26<00:20, 8657.03it/s] 57%|█████▋    | 226806/400000 [00:26<00:19, 8689.99it/s] 57%|█████▋    | 227685/400000 [00:26<00:19, 8717.01it/s] 57%|█████▋    | 228561/400000 [00:26<00:19, 8728.60it/s] 57%|█████▋    | 229434/400000 [00:26<00:19, 8714.56it/s] 58%|█████▊    | 230306/400000 [00:26<00:19, 8616.12it/s] 58%|█████▊    | 231168/400000 [00:26<00:19, 8523.36it/s] 58%|█████▊    | 232021/400000 [00:27<00:19, 8514.47it/s] 58%|█████▊    | 232887/400000 [00:27<00:19, 8555.30it/s] 58%|█████▊    | 233766/400000 [00:27<00:19, 8622.03it/s] 59%|█████▊    | 234629/400000 [00:27<00:19, 8622.40it/s] 59%|█████▉    | 235512/400000 [00:27<00:18, 8683.42it/s] 59%|█████▉    | 236381/400000 [00:27<00:18, 8683.85it/s] 59%|█████▉    | 237269/400000 [00:27<00:18, 8739.62it/s] 60%|█████▉    | 238158/400000 [00:27<00:18, 8782.02it/s] 60%|█████▉    | 239037/400000 [00:27<00:18, 8720.97it/s] 60%|█████▉    | 239915/400000 [00:27<00:18, 8736.06it/s] 60%|██████    | 240805/400000 [00:28<00:18, 8781.76it/s] 60%|██████    | 241710/400000 [00:28<00:17, 8858.02it/s] 61%|██████    | 242617/400000 [00:28<00:17, 8919.05it/s] 61%|██████    | 243510/400000 [00:28<00:17, 8801.16it/s] 61%|██████    | 244391/400000 [00:28<00:17, 8781.72it/s] 61%|██████▏   | 245270/400000 [00:28<00:17, 8773.95it/s] 62%|██████▏   | 246148/400000 [00:28<00:17, 8647.45it/s] 62%|██████▏   | 247014/400000 [00:28<00:17, 8535.71it/s] 62%|██████▏   | 247869/400000 [00:28<00:17, 8525.30it/s] 62%|██████▏   | 248736/400000 [00:28<00:17, 8567.79it/s] 62%|██████▏   | 249651/400000 [00:29<00:17, 8731.97it/s] 63%|██████▎   | 250587/400000 [00:29<00:16, 8909.26it/s] 63%|██████▎   | 251487/400000 [00:29<00:16, 8935.68it/s] 63%|██████▎   | 252382/400000 [00:29<00:16, 8824.36it/s] 63%|██████▎   | 253266/400000 [00:29<00:16, 8747.94it/s] 64%|██████▎   | 254142/400000 [00:29<00:16, 8619.69it/s] 64%|██████▍   | 255013/400000 [00:29<00:16, 8646.29it/s] 64%|██████▍   | 255888/400000 [00:29<00:16, 8675.04it/s] 64%|██████▍   | 256761/400000 [00:29<00:16, 8690.39it/s] 64%|██████▍   | 257641/400000 [00:29<00:16, 8721.81it/s] 65%|██████▍   | 258516/400000 [00:30<00:16, 8726.99it/s] 65%|██████▍   | 259399/400000 [00:30<00:16, 8755.61it/s] 65%|██████▌   | 260275/400000 [00:30<00:15, 8742.23it/s] 65%|██████▌   | 261150/400000 [00:30<00:15, 8704.56it/s] 66%|██████▌   | 262055/400000 [00:30<00:15, 8804.19it/s] 66%|██████▌   | 262953/400000 [00:30<00:15, 8854.70it/s] 66%|██████▌   | 263839/400000 [00:30<00:15, 8819.10it/s] 66%|██████▌   | 264722/400000 [00:30<00:15, 8743.58it/s] 66%|██████▋   | 265597/400000 [00:30<00:15, 8686.79it/s] 67%|██████▋   | 266466/400000 [00:30<00:15, 8667.37it/s] 67%|██████▋   | 267333/400000 [00:31<00:15, 8572.45it/s] 67%|██████▋   | 268213/400000 [00:31<00:15, 8635.87it/s] 67%|██████▋   | 269081/400000 [00:31<00:15, 8648.97it/s] 67%|██████▋   | 269947/400000 [00:31<00:15, 8609.82it/s] 68%|██████▊   | 270810/400000 [00:31<00:14, 8613.46it/s] 68%|██████▊   | 271672/400000 [00:31<00:15, 8510.87it/s] 68%|██████▊   | 272548/400000 [00:31<00:14, 8582.61it/s] 68%|██████▊   | 273422/400000 [00:31<00:14, 8626.80it/s] 69%|██████▊   | 274286/400000 [00:31<00:14, 8584.87it/s] 69%|██████▉   | 275145/400000 [00:32<00:14, 8523.17it/s] 69%|██████▉   | 276015/400000 [00:32<00:14, 8572.87it/s] 69%|██████▉   | 276890/400000 [00:32<00:14, 8624.47it/s] 69%|██████▉   | 277757/400000 [00:32<00:14, 8637.28it/s] 70%|██████▉   | 278621/400000 [00:32<00:14, 8628.97it/s] 70%|██████▉   | 279485/400000 [00:32<00:14, 8582.69it/s] 70%|███████   | 280344/400000 [00:32<00:13, 8561.61it/s] 70%|███████   | 281211/400000 [00:32<00:13, 8592.87it/s] 71%|███████   | 282080/400000 [00:32<00:13, 8619.74it/s] 71%|███████   | 282955/400000 [00:32<00:13, 8657.16it/s] 71%|███████   | 283832/400000 [00:33<00:13, 8688.66it/s] 71%|███████   | 284701/400000 [00:33<00:13, 8682.26it/s] 71%|███████▏  | 285570/400000 [00:33<00:13, 8433.93it/s] 72%|███████▏  | 286428/400000 [00:33<00:13, 8475.53it/s] 72%|███████▏  | 287300/400000 [00:33<00:13, 8545.07it/s] 72%|███████▏  | 288167/400000 [00:33<00:13, 8579.42it/s] 72%|███████▏  | 289036/400000 [00:33<00:12, 8611.13it/s] 72%|███████▏  | 289898/400000 [00:33<00:13, 8466.79it/s] 73%|███████▎  | 290746/400000 [00:33<00:12, 8421.21it/s] 73%|███████▎  | 291612/400000 [00:33<00:12, 8490.83it/s] 73%|███████▎  | 292480/400000 [00:34<00:12, 8545.96it/s] 73%|███████▎  | 293348/400000 [00:34<00:12, 8585.35it/s] 74%|███████▎  | 294223/400000 [00:34<00:12, 8633.30it/s] 74%|███████▍  | 295096/400000 [00:34<00:12, 8660.35it/s] 74%|███████▍  | 295963/400000 [00:34<00:12, 8608.69it/s] 74%|███████▍  | 296825/400000 [00:34<00:11, 8607.41it/s] 74%|███████▍  | 297686/400000 [00:34<00:11, 8531.37it/s] 75%|███████▍  | 298555/400000 [00:34<00:11, 8575.94it/s] 75%|███████▍  | 299427/400000 [00:34<00:11, 8617.76it/s] 75%|███████▌  | 300297/400000 [00:34<00:11, 8640.08it/s] 75%|███████▌  | 301162/400000 [00:35<00:11, 8580.40it/s] 76%|███████▌  | 302034/400000 [00:35<00:11, 8620.64it/s] 76%|███████▌  | 302897/400000 [00:35<00:11, 8495.44it/s] 76%|███████▌  | 303781/400000 [00:35<00:11, 8594.86it/s] 76%|███████▌  | 304648/400000 [00:35<00:11, 8616.43it/s] 76%|███████▋  | 305521/400000 [00:35<00:10, 8648.40it/s] 77%|███████▋  | 306404/400000 [00:35<00:10, 8701.62it/s] 77%|███████▋  | 307275/400000 [00:35<00:10, 8688.46it/s] 77%|███████▋  | 308145/400000 [00:35<00:10, 8413.47it/s] 77%|███████▋  | 308994/400000 [00:35<00:10, 8433.49it/s] 77%|███████▋  | 309839/400000 [00:36<00:10, 8425.89it/s] 78%|███████▊  | 310683/400000 [00:36<00:10, 8361.96it/s] 78%|███████▊  | 311521/400000 [00:36<00:10, 8358.53it/s] 78%|███████▊  | 312362/400000 [00:36<00:10, 8371.07it/s] 78%|███████▊  | 313200/400000 [00:36<00:10, 8239.09it/s] 79%|███████▊  | 314068/400000 [00:36<00:10, 8364.75it/s] 79%|███████▊  | 314930/400000 [00:36<00:10, 8436.76it/s] 79%|███████▉  | 315829/400000 [00:36<00:09, 8593.18it/s] 79%|███████▉  | 316735/400000 [00:36<00:09, 8727.76it/s] 79%|███████▉  | 317614/400000 [00:36<00:09, 8744.36it/s] 80%|███████▉  | 318511/400000 [00:37<00:09, 8808.42it/s] 80%|███████▉  | 319393/400000 [00:37<00:09, 8790.20it/s] 80%|████████  | 320273/400000 [00:37<00:09, 8717.23it/s] 80%|████████  | 321146/400000 [00:37<00:09, 8689.13it/s] 81%|████████  | 322017/400000 [00:37<00:08, 8692.71it/s] 81%|████████  | 322921/400000 [00:37<00:08, 8792.09it/s] 81%|████████  | 323828/400000 [00:37<00:08, 8873.53it/s] 81%|████████  | 324730/400000 [00:37<00:08, 8916.88it/s] 81%|████████▏ | 325623/400000 [00:37<00:08, 8749.94it/s] 82%|████████▏ | 326499/400000 [00:37<00:08, 8710.06it/s] 82%|████████▏ | 327371/400000 [00:38<00:08, 8641.47it/s] 82%|████████▏ | 328241/400000 [00:38<00:08, 8657.86it/s] 82%|████████▏ | 329145/400000 [00:38<00:08, 8768.88it/s] 83%|████████▎ | 330052/400000 [00:38<00:07, 8854.98it/s] 83%|████████▎ | 330962/400000 [00:38<00:07, 8925.89it/s] 83%|████████▎ | 331856/400000 [00:38<00:07, 8843.04it/s] 83%|████████▎ | 332750/400000 [00:38<00:07, 8869.65it/s] 83%|████████▎ | 333638/400000 [00:38<00:07, 8765.29it/s] 84%|████████▎ | 334516/400000 [00:38<00:07, 8703.95it/s] 84%|████████▍ | 335391/400000 [00:38<00:07, 8715.79it/s] 84%|████████▍ | 336277/400000 [00:39<00:07, 8757.46it/s] 84%|████████▍ | 337186/400000 [00:39<00:07, 8852.25it/s] 85%|████████▍ | 338089/400000 [00:39<00:06, 8903.09it/s] 85%|████████▍ | 338980/400000 [00:39<00:06, 8895.00it/s] 85%|████████▍ | 339870/400000 [00:39<00:06, 8817.59it/s] 85%|████████▌ | 340757/400000 [00:39<00:06, 8831.07it/s] 85%|████████▌ | 341652/400000 [00:39<00:06, 8863.93it/s] 86%|████████▌ | 342539/400000 [00:39<00:06, 8794.91it/s] 86%|████████▌ | 343419/400000 [00:39<00:06, 8764.35it/s] 86%|████████▌ | 344298/400000 [00:39<00:06, 8770.12it/s] 86%|████████▋ | 345176/400000 [00:40<00:06, 8769.45it/s] 87%|████████▋ | 346059/400000 [00:40<00:06, 8786.80it/s] 87%|████████▋ | 346938/400000 [00:40<00:06, 8781.55it/s] 87%|████████▋ | 347817/400000 [00:40<00:05, 8783.59it/s] 87%|████████▋ | 348696/400000 [00:40<00:05, 8594.99it/s] 87%|████████▋ | 349557/400000 [00:40<00:05, 8543.13it/s] 88%|████████▊ | 350413/400000 [00:40<00:05, 8376.86it/s] 88%|████████▊ | 351288/400000 [00:40<00:05, 8484.97it/s] 88%|████████▊ | 352152/400000 [00:40<00:05, 8529.51it/s] 88%|████████▊ | 353011/400000 [00:41<00:05, 8538.64it/s] 88%|████████▊ | 353866/400000 [00:41<00:05, 8517.38it/s] 89%|████████▊ | 354737/400000 [00:41<00:05, 8574.12it/s] 89%|████████▉ | 355613/400000 [00:41<00:05, 8628.01it/s] 89%|████████▉ | 356492/400000 [00:41<00:05, 8673.72it/s] 89%|████████▉ | 357360/400000 [00:41<00:04, 8627.71it/s] 90%|████████▉ | 358224/400000 [00:41<00:04, 8551.40it/s] 90%|████████▉ | 359088/400000 [00:41<00:04, 8577.59it/s] 90%|████████▉ | 359947/400000 [00:41<00:04, 8538.14it/s] 90%|█████████ | 360802/400000 [00:41<00:04, 8435.41it/s] 90%|█████████ | 361646/400000 [00:42<00:04, 8348.58it/s] 91%|█████████ | 362545/400000 [00:42<00:04, 8529.63it/s] 91%|█████████ | 363445/400000 [00:42<00:04, 8663.42it/s] 91%|█████████ | 364313/400000 [00:42<00:04, 8524.25it/s] 91%|█████████▏| 365167/400000 [00:42<00:04, 8387.98it/s] 92%|█████████▏| 366008/400000 [00:42<00:04, 8378.43it/s] 92%|█████████▏| 366878/400000 [00:42<00:03, 8470.19it/s] 92%|█████████▏| 367727/400000 [00:42<00:03, 8391.86it/s] 92%|█████████▏| 368568/400000 [00:42<00:03, 8346.49it/s] 92%|█████████▏| 369440/400000 [00:42<00:03, 8452.41it/s] 93%|█████████▎| 370311/400000 [00:43<00:03, 8527.98it/s] 93%|█████████▎| 371183/400000 [00:43<00:03, 8584.42it/s] 93%|█████████▎| 372043/400000 [00:43<00:03, 8533.99it/s] 93%|█████████▎| 372897/400000 [00:43<00:03, 8464.95it/s] 93%|█████████▎| 373767/400000 [00:43<00:03, 8533.34it/s] 94%|█████████▎| 374635/400000 [00:43<00:02, 8574.08it/s] 94%|█████████▍| 375506/400000 [00:43<00:02, 8612.73it/s] 94%|█████████▍| 376385/400000 [00:43<00:02, 8662.45it/s] 94%|█████████▍| 377259/400000 [00:43<00:02, 8683.16it/s] 95%|█████████▍| 378141/400000 [00:43<00:02, 8722.13it/s] 95%|█████████▍| 379014/400000 [00:44<00:02, 8654.85it/s] 95%|█████████▍| 379880/400000 [00:44<00:02, 8470.24it/s] 95%|█████████▌| 380747/400000 [00:44<00:02, 8528.14it/s] 95%|█████████▌| 381620/400000 [00:44<00:02, 8585.12it/s] 96%|█████████▌| 382482/400000 [00:44<00:02, 8591.95it/s] 96%|█████████▌| 383342/400000 [00:44<00:01, 8451.49it/s] 96%|█████████▌| 384223/400000 [00:44<00:01, 8554.86it/s] 96%|█████████▋| 385108/400000 [00:44<00:01, 8640.54it/s] 96%|█████████▋| 385973/400000 [00:44<00:01, 8641.93it/s] 97%|█████████▋| 386838/400000 [00:44<00:01, 8595.88it/s] 97%|█████████▋| 387699/400000 [00:45<00:01, 8598.64it/s] 97%|█████████▋| 388574/400000 [00:45<00:01, 8640.64it/s] 97%|█████████▋| 389446/400000 [00:45<00:01, 8662.29it/s] 98%|█████████▊| 390334/400000 [00:45<00:01, 8725.74it/s] 98%|█████████▊| 391233/400000 [00:45<00:00, 8801.61it/s] 98%|█████████▊| 392130/400000 [00:45<00:00, 8849.69it/s] 98%|█████████▊| 393016/400000 [00:45<00:00, 8705.29it/s] 98%|█████████▊| 393888/400000 [00:45<00:00, 8676.94it/s] 99%|█████████▊| 394757/400000 [00:45<00:00, 8520.10it/s] 99%|█████████▉| 395645/400000 [00:45<00:00, 8624.73it/s] 99%|█████████▉| 396547/400000 [00:46<00:00, 8737.47it/s] 99%|█████████▉| 397422/400000 [00:46<00:00, 8738.54it/s]100%|█████████▉| 398297/400000 [00:46<00:00, 8737.98it/s]100%|█████████▉| 399172/400000 [00:46<00:00, 8723.75it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8606.31it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f69516ffd30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011171260616810025 	 Accuracy: 49
Train Epoch: 1 	 Loss: 0.011058037895024022 	 Accuracy: 63

  model saves at 63% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15604 out of table with 15505 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15604 out of table with 15505 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-13 03:23:33.665251: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 03:23:33.669773: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-13 03:23:33.669902: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561952bd9cc0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 03:23:33.669915: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f695f973128> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.5286 - accuracy: 0.5090
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5746 - accuracy: 0.5060 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5900 - accuracy: 0.5050
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5785 - accuracy: 0.5058
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6053 - accuracy: 0.5040
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6436 - accuracy: 0.5015
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6228 - accuracy: 0.5029
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6130 - accuracy: 0.5035
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5831 - accuracy: 0.5054
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5762 - accuracy: 0.5059
11000/25000 [============>.................] - ETA: 3s - loss: 7.6234 - accuracy: 0.5028
12000/25000 [=============>................] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6579 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 2s - loss: 7.6625 - accuracy: 0.5003
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6542 - accuracy: 0.5008
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6414 - accuracy: 0.5016
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6479 - accuracy: 0.5012
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6521 - accuracy: 0.5009
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6567 - accuracy: 0.5006
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6360 - accuracy: 0.5020
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6485 - accuracy: 0.5012
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6606 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6545 - accuracy: 0.5008
25000/25000 [==============================] - 7s 279us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f68b5d3d8d0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f69545ff5c0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4572 - crf_viterbi_accuracy: 0.1200 - val_loss: 1.4848 - val_crf_viterbi_accuracy: 0.1733

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


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7fa1814eff60> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 04:14:17.615643
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 04:14:17.620192
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 04:14:17.623789
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 04:14:17.627881
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7fa18d2b9400> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354559.9375
Epoch 2/10

1/1 [==============================] - 0s 113ms/step - loss: 304526.8438
Epoch 3/10

1/1 [==============================] - 0s 104ms/step - loss: 232092.2500
Epoch 4/10

1/1 [==============================] - 0s 102ms/step - loss: 166013.9375
Epoch 5/10

1/1 [==============================] - 0s 105ms/step - loss: 114173.5547
Epoch 6/10

1/1 [==============================] - 0s 103ms/step - loss: 77149.8594
Epoch 7/10

1/1 [==============================] - 0s 106ms/step - loss: 53022.0664
Epoch 8/10

1/1 [==============================] - 0s 101ms/step - loss: 37818.9297
Epoch 9/10

1/1 [==============================] - 0s 109ms/step - loss: 28204.8828
Epoch 10/10

1/1 [==============================] - 0s 121ms/step - loss: 21803.5762

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.56199     0.32848462  0.29385963 -1.0434055   1.0229671   1.0269464
  -1.2249078  -0.4209598  -0.68178815 -0.5026692   1.3076515   1.3097088
   0.9855033  -0.350381   -0.13215938  0.8392058   0.8955416  -0.82981116
   0.74695253 -1.08853    -0.3950897   0.8680682  -0.54022336  0.14990644
   1.0992696  -0.4104593  -0.18635471  0.8060092  -0.95242983 -0.56867856
  -0.5839366   0.3806241  -0.10724846 -0.10981303 -0.7662405   0.7571814
  -1.2830087  -0.85578364  0.915861   -0.47398952  0.8277951   0.44194046
  -0.07110805  0.921558   -0.1946334  -1.2238504   0.5917422  -0.36178744
  -0.03829058  0.42672393  0.52546436  0.34169412 -1.0990639   1.1418173
   0.8728975  -0.6194843   0.48436666  0.14375146 -0.8741282  -0.23728381
  -1.0356674  -0.9896827   1.234024    0.22044273 -0.3205427   0.24131534
  -0.29707462  1.2444901   0.78867614  0.5585743  -0.66787314  0.57531077
  -0.81310415 -0.17733009  0.81776816 -1.0499493  -0.0774828   1.1603105
   0.7756842   0.09580754 -0.9919281   0.4032953   1.2986089   1.2666025
  -0.7414066   0.57795304  0.713939    0.99911743  0.57166106 -0.71828896
   1.2459706  -0.70854396  1.0489186   0.2488047   1.261711    1.3251795
  -0.05564739 -0.5696772  -0.714838   -0.0168571  -0.6852293   0.9836284
  -1.2999452   1.0188056   0.14059933 -0.48328733 -0.4769996  -1.1780586
  -0.9281076   0.4837364  -1.2256771  -0.67975295 -0.907446   -0.7722243
   0.5892681  -0.7331697  -0.27937603  1.0770688  -0.5878106  -0.33534148
   0.09240576  3.1752934   3.850356    2.532703    4.7147346   2.6026664
   3.6187296   2.6631246   3.100594    4.9724426   4.061751    4.5303087
   4.9925056   2.4898746   2.5269022   2.5672143   4.8376737   3.2635922
   2.655067    5.0198083   4.9403305   3.721691    3.3058183   4.8803616
   4.9668484   2.653316    5.120222    5.0346556   3.2132328   3.2860322
   3.229231    3.399201    3.4256334   4.8210864   4.67712     3.5258546
   4.5098553   3.8212922   3.0732172   2.6276634   5.102264    5.1203804
   3.7391715   2.7939074   4.5613427   2.9767883   5.081544    2.9471736
   2.9090872   2.743119    2.730352    2.7281952   4.4591184   3.3861053
   4.0616727   5.0919156   3.496881    3.4858642   5.030789    4.607797
   0.36962312  1.5413277   0.9632844   1.4724585   0.7718369   0.6201893
   0.94598037  1.6182094   1.9874675   0.5810652   0.85078263  0.38786072
   1.5836878   1.2107972   1.4414576   0.8964621   0.9113171   0.81076956
   1.3832208   0.6867356   1.378695    0.27720952  0.33294082  2.2017372
   0.9723714   1.0174562   2.2709439   1.6147016   0.4265554   1.5220442
   2.3080516   1.8126729   2.2123756   0.5861107   1.0500293   0.29503667
   0.31757808  0.5648594   0.8253467   1.8652754   1.7898922   1.4798764
   2.2102      0.29540396  1.9542093   1.3473687   1.356508    1.6675532
   1.493393    0.44997764  0.29549384  0.3317446   0.6515242   0.9060057
   0.7752869   0.50313735  0.4684589   1.9663367   2.171541    1.5206208
   0.27212763  0.4059887   2.0418406   0.43512315  1.9204413   0.88558567
   1.8797815   0.7162099   0.2910267   0.27826047  0.8566978   0.33469152
   1.3291671   0.69600534  1.1591064   0.7034272   0.6970346   0.4663996
   0.41783077  0.30922687  0.35600984  1.1205062   0.603763    0.36278355
   0.42557037  1.590364    0.4130727   1.5522792   1.7245116   2.1976888
   1.0094361   2.315202    0.31141806  1.4080005   1.0792556   0.3122096
   0.34627646  0.5943128   2.082277    1.6796968   0.43963903  1.9009193
   0.40429825  1.4306493   0.42857957  0.43372113  1.295634    0.92706865
   1.9507484   0.27960634  2.2047086   1.6957796   0.87849635  1.0251261
   2.1697917   1.3890355   1.2527893   0.5054653   1.4341704   0.6002219
   0.03054219  5.3068256   5.525484    4.243841    5.1153774   3.8618617
   4.74727     4.0526905   4.5859237   5.448649    5.4875274   4.077676
   3.9523935   4.383451    4.3547406   5.5574102   5.2296247   4.26018
   3.8065429   4.186182    4.518344    3.8421311   4.3390274   5.724662
   5.417275    3.5794525   5.6828294   4.81318     4.485288    5.4165573
   5.015076    4.625144    4.9459777   4.062696    4.670972    3.9896526
   3.6171365   5.6444364   3.8139558   3.6682134   5.6724706   3.6163492
   3.4502282   3.489368    4.7347937   4.785473    3.4263072   3.5315084
   5.7129674   5.65883     5.685576    4.210436    4.6527643   4.728849
   4.7164483   4.4046683   5.4787865   5.1248636   5.6216855   4.8315115
  -0.07331896 -3.5943089   0.19543913]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 04:14:27.293454
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.2537
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 04:14:27.297816
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9670.16
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 04:14:27.301998
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   98.2717
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 04:14:27.305818
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -865.032
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140331276911056
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140328764322368
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140328764322872
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140328764323376
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140328764323880
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140328764324384

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7fa18913aeb8> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.545097
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.514965
grad_step = 000002, loss = 0.492910
grad_step = 000003, loss = 0.471428
grad_step = 000004, loss = 0.448458
grad_step = 000005, loss = 0.422610
grad_step = 000006, loss = 0.396801
grad_step = 000007, loss = 0.376535
grad_step = 000008, loss = 0.362746
grad_step = 000009, loss = 0.344788
grad_step = 000010, loss = 0.325455
grad_step = 000011, loss = 0.308971
grad_step = 000012, loss = 0.297421
grad_step = 000013, loss = 0.287311
grad_step = 000014, loss = 0.277486
grad_step = 000015, loss = 0.266259
grad_step = 000016, loss = 0.254373
grad_step = 000017, loss = 0.243280
grad_step = 000018, loss = 0.232018
grad_step = 000019, loss = 0.220751
grad_step = 000020, loss = 0.210922
grad_step = 000021, loss = 0.201747
grad_step = 000022, loss = 0.191560
grad_step = 000023, loss = 0.181392
grad_step = 000024, loss = 0.172346
grad_step = 000025, loss = 0.163592
grad_step = 000026, loss = 0.155077
grad_step = 000027, loss = 0.146972
grad_step = 000028, loss = 0.138844
grad_step = 000029, loss = 0.130979
grad_step = 000030, loss = 0.123489
grad_step = 000031, loss = 0.116345
grad_step = 000032, loss = 0.109675
grad_step = 000033, loss = 0.103033
grad_step = 000034, loss = 0.096650
grad_step = 000035, loss = 0.090853
grad_step = 000036, loss = 0.085208
grad_step = 000037, loss = 0.079736
grad_step = 000038, loss = 0.074511
grad_step = 000039, loss = 0.069435
grad_step = 000040, loss = 0.064682
grad_step = 000041, loss = 0.060170
grad_step = 000042, loss = 0.055965
grad_step = 000043, loss = 0.051958
grad_step = 000044, loss = 0.048163
grad_step = 000045, loss = 0.044671
grad_step = 000046, loss = 0.041277
grad_step = 000047, loss = 0.038061
grad_step = 000048, loss = 0.035035
grad_step = 000049, loss = 0.032248
grad_step = 000050, loss = 0.029688
grad_step = 000051, loss = 0.027288
grad_step = 000052, loss = 0.025071
grad_step = 000053, loss = 0.023031
grad_step = 000054, loss = 0.021183
grad_step = 000055, loss = 0.019415
grad_step = 000056, loss = 0.017817
grad_step = 000057, loss = 0.016369
grad_step = 000058, loss = 0.015040
grad_step = 000059, loss = 0.013807
grad_step = 000060, loss = 0.012696
grad_step = 000061, loss = 0.011700
grad_step = 000062, loss = 0.010799
grad_step = 000063, loss = 0.009970
grad_step = 000064, loss = 0.009236
grad_step = 000065, loss = 0.008574
grad_step = 000066, loss = 0.007958
grad_step = 000067, loss = 0.007408
grad_step = 000068, loss = 0.006920
grad_step = 000069, loss = 0.006476
grad_step = 000070, loss = 0.006072
grad_step = 000071, loss = 0.005705
grad_step = 000072, loss = 0.005376
grad_step = 000073, loss = 0.005067
grad_step = 000074, loss = 0.004791
grad_step = 000075, loss = 0.004541
grad_step = 000076, loss = 0.004309
grad_step = 000077, loss = 0.004094
grad_step = 000078, loss = 0.003901
grad_step = 000079, loss = 0.003720
grad_step = 000080, loss = 0.003553
grad_step = 000081, loss = 0.003399
grad_step = 000082, loss = 0.003258
grad_step = 000083, loss = 0.003126
grad_step = 000084, loss = 0.003007
grad_step = 000085, loss = 0.002900
grad_step = 000086, loss = 0.002801
grad_step = 000087, loss = 0.002712
grad_step = 000088, loss = 0.002630
grad_step = 000089, loss = 0.002556
grad_step = 000090, loss = 0.002490
grad_step = 000091, loss = 0.002432
grad_step = 000092, loss = 0.002381
grad_step = 000093, loss = 0.002335
grad_step = 000094, loss = 0.002296
grad_step = 000095, loss = 0.002261
grad_step = 000096, loss = 0.002231
grad_step = 000097, loss = 0.002205
grad_step = 000098, loss = 0.002182
grad_step = 000099, loss = 0.002163
grad_step = 000100, loss = 0.002146
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002131
grad_step = 000102, loss = 0.002118
grad_step = 000103, loss = 0.002107
grad_step = 000104, loss = 0.002097
grad_step = 000105, loss = 0.002088
grad_step = 000106, loss = 0.002079
grad_step = 000107, loss = 0.002072
grad_step = 000108, loss = 0.002064
grad_step = 000109, loss = 0.002057
grad_step = 000110, loss = 0.002049
grad_step = 000111, loss = 0.002042
grad_step = 000112, loss = 0.002035
grad_step = 000113, loss = 0.002028
grad_step = 000114, loss = 0.002021
grad_step = 000115, loss = 0.002016
grad_step = 000116, loss = 0.002013
grad_step = 000117, loss = 0.002009
grad_step = 000118, loss = 0.002000
grad_step = 000119, loss = 0.001986
grad_step = 000120, loss = 0.001976
grad_step = 000121, loss = 0.001972
grad_step = 000122, loss = 0.001969
grad_step = 000123, loss = 0.001965
grad_step = 000124, loss = 0.001957
grad_step = 000125, loss = 0.001946
grad_step = 000126, loss = 0.001934
grad_step = 000127, loss = 0.001923
grad_step = 000128, loss = 0.001914
grad_step = 000129, loss = 0.001906
grad_step = 000130, loss = 0.001900
grad_step = 000131, loss = 0.001898
grad_step = 000132, loss = 0.001906
grad_step = 000133, loss = 0.001939
grad_step = 000134, loss = 0.002001
grad_step = 000135, loss = 0.002021
grad_step = 000136, loss = 0.001949
grad_step = 000137, loss = 0.001862
grad_step = 000138, loss = 0.001859
grad_step = 000139, loss = 0.001920
grad_step = 000140, loss = 0.001940
grad_step = 000141, loss = 0.001880
grad_step = 000142, loss = 0.001826
grad_step = 000143, loss = 0.001831
grad_step = 000144, loss = 0.001869
grad_step = 000145, loss = 0.001883
grad_step = 000146, loss = 0.001847
grad_step = 000147, loss = 0.001803
grad_step = 000148, loss = 0.001796
grad_step = 000149, loss = 0.001815
grad_step = 000150, loss = 0.001835
grad_step = 000151, loss = 0.001832
grad_step = 000152, loss = 0.001807
grad_step = 000153, loss = 0.001777
grad_step = 000154, loss = 0.001765
grad_step = 000155, loss = 0.001768
grad_step = 000156, loss = 0.001780
grad_step = 000157, loss = 0.001795
grad_step = 000158, loss = 0.001805
grad_step = 000159, loss = 0.001806
grad_step = 000160, loss = 0.001798
grad_step = 000161, loss = 0.001786
grad_step = 000162, loss = 0.001767
grad_step = 000163, loss = 0.001749
grad_step = 000164, loss = 0.001735
grad_step = 000165, loss = 0.001726
grad_step = 000166, loss = 0.001719
grad_step = 000167, loss = 0.001717
grad_step = 000168, loss = 0.001717
grad_step = 000169, loss = 0.001721
grad_step = 000170, loss = 0.001732
grad_step = 000171, loss = 0.001762
grad_step = 000172, loss = 0.001839
grad_step = 000173, loss = 0.001980
grad_step = 000174, loss = 0.002205
grad_step = 000175, loss = 0.002245
grad_step = 000176, loss = 0.002024
grad_step = 000177, loss = 0.001727
grad_step = 000178, loss = 0.001778
grad_step = 000179, loss = 0.001985
grad_step = 000180, loss = 0.001906
grad_step = 000181, loss = 0.001712
grad_step = 000182, loss = 0.001747
grad_step = 000183, loss = 0.001874
grad_step = 000184, loss = 0.001812
grad_step = 000185, loss = 0.001685
grad_step = 000186, loss = 0.001737
grad_step = 000187, loss = 0.001815
grad_step = 000188, loss = 0.001749
grad_step = 000189, loss = 0.001679
grad_step = 000190, loss = 0.001707
grad_step = 000191, loss = 0.001751
grad_step = 000192, loss = 0.001724
grad_step = 000193, loss = 0.001669
grad_step = 000194, loss = 0.001674
grad_step = 000195, loss = 0.001715
grad_step = 000196, loss = 0.001704
grad_step = 000197, loss = 0.001660
grad_step = 000198, loss = 0.001659
grad_step = 000199, loss = 0.001682
grad_step = 000200, loss = 0.001681
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001664
grad_step = 000202, loss = 0.001646
grad_step = 000203, loss = 0.001649
grad_step = 000204, loss = 0.001665
grad_step = 000205, loss = 0.001660
grad_step = 000206, loss = 0.001640
grad_step = 000207, loss = 0.001635
grad_step = 000208, loss = 0.001639
grad_step = 000209, loss = 0.001643
grad_step = 000210, loss = 0.001642
grad_step = 000211, loss = 0.001635
grad_step = 000212, loss = 0.001624
grad_step = 000213, loss = 0.001621
grad_step = 000214, loss = 0.001626
grad_step = 000215, loss = 0.001627
grad_step = 000216, loss = 0.001624
grad_step = 000217, loss = 0.001620
grad_step = 000218, loss = 0.001614
grad_step = 000219, loss = 0.001609
grad_step = 000220, loss = 0.001608
grad_step = 000221, loss = 0.001609
grad_step = 000222, loss = 0.001610
grad_step = 000223, loss = 0.001612
grad_step = 000224, loss = 0.001613
grad_step = 000225, loss = 0.001613
grad_step = 000226, loss = 0.001612
grad_step = 000227, loss = 0.001610
grad_step = 000228, loss = 0.001607
grad_step = 000229, loss = 0.001604
grad_step = 000230, loss = 0.001601
grad_step = 000231, loss = 0.001598
grad_step = 000232, loss = 0.001595
grad_step = 000233, loss = 0.001593
grad_step = 000234, loss = 0.001593
grad_step = 000235, loss = 0.001593
grad_step = 000236, loss = 0.001594
grad_step = 000237, loss = 0.001599
grad_step = 000238, loss = 0.001608
grad_step = 000239, loss = 0.001624
grad_step = 000240, loss = 0.001655
grad_step = 000241, loss = 0.001700
grad_step = 000242, loss = 0.001775
grad_step = 000243, loss = 0.001846
grad_step = 000244, loss = 0.001906
grad_step = 000245, loss = 0.001859
grad_step = 000246, loss = 0.001744
grad_step = 000247, loss = 0.001609
grad_step = 000248, loss = 0.001565
grad_step = 000249, loss = 0.001620
grad_step = 000250, loss = 0.001690
grad_step = 000251, loss = 0.001699
grad_step = 000252, loss = 0.001628
grad_step = 000253, loss = 0.001564
grad_step = 000254, loss = 0.001566
grad_step = 000255, loss = 0.001611
grad_step = 000256, loss = 0.001637
grad_step = 000257, loss = 0.001611
grad_step = 000258, loss = 0.001567
grad_step = 000259, loss = 0.001548
grad_step = 000260, loss = 0.001563
grad_step = 000261, loss = 0.001590
grad_step = 000262, loss = 0.001601
grad_step = 000263, loss = 0.001587
grad_step = 000264, loss = 0.001561
grad_step = 000265, loss = 0.001542
grad_step = 000266, loss = 0.001538
grad_step = 000267, loss = 0.001548
grad_step = 000268, loss = 0.001560
grad_step = 000269, loss = 0.001567
grad_step = 000270, loss = 0.001565
grad_step = 000271, loss = 0.001554
grad_step = 000272, loss = 0.001541
grad_step = 000273, loss = 0.001531
grad_step = 000274, loss = 0.001526
grad_step = 000275, loss = 0.001527
grad_step = 000276, loss = 0.001531
grad_step = 000277, loss = 0.001535
grad_step = 000278, loss = 0.001539
grad_step = 000279, loss = 0.001542
grad_step = 000280, loss = 0.001542
grad_step = 000281, loss = 0.001541
grad_step = 000282, loss = 0.001539
grad_step = 000283, loss = 0.001536
grad_step = 000284, loss = 0.001532
grad_step = 000285, loss = 0.001529
grad_step = 000286, loss = 0.001526
grad_step = 000287, loss = 0.001524
grad_step = 000288, loss = 0.001522
grad_step = 000289, loss = 0.001521
grad_step = 000290, loss = 0.001521
grad_step = 000291, loss = 0.001523
grad_step = 000292, loss = 0.001526
grad_step = 000293, loss = 0.001532
grad_step = 000294, loss = 0.001542
grad_step = 000295, loss = 0.001561
grad_step = 000296, loss = 0.001590
grad_step = 000297, loss = 0.001636
grad_step = 000298, loss = 0.001697
grad_step = 000299, loss = 0.001777
grad_step = 000300, loss = 0.001830
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001854
grad_step = 000302, loss = 0.001768
grad_step = 000303, loss = 0.001641
grad_step = 000304, loss = 0.001520
grad_step = 000305, loss = 0.001496
grad_step = 000306, loss = 0.001559
grad_step = 000307, loss = 0.001621
grad_step = 000308, loss = 0.001620
grad_step = 000309, loss = 0.001546
grad_step = 000310, loss = 0.001491
grad_step = 000311, loss = 0.001501
grad_step = 000312, loss = 0.001542
grad_step = 000313, loss = 0.001559
grad_step = 000314, loss = 0.001525
grad_step = 000315, loss = 0.001498
grad_step = 000316, loss = 0.001508
grad_step = 000317, loss = 0.001520
grad_step = 000318, loss = 0.001510
grad_step = 000319, loss = 0.001490
grad_step = 000320, loss = 0.001490
grad_step = 000321, loss = 0.001514
grad_step = 000322, loss = 0.001502
grad_step = 000323, loss = 0.001476
grad_step = 000324, loss = 0.001473
grad_step = 000325, loss = 0.001488
grad_step = 000326, loss = 0.001507
grad_step = 000327, loss = 0.001481
grad_step = 000328, loss = 0.001468
grad_step = 000329, loss = 0.001469
grad_step = 000330, loss = 0.001478
grad_step = 000331, loss = 0.001484
grad_step = 000332, loss = 0.001464
grad_step = 000333, loss = 0.001459
grad_step = 000334, loss = 0.001465
grad_step = 000335, loss = 0.001469
grad_step = 000336, loss = 0.001463
grad_step = 000337, loss = 0.001456
grad_step = 000338, loss = 0.001451
grad_step = 000339, loss = 0.001450
grad_step = 000340, loss = 0.001450
grad_step = 000341, loss = 0.001450
grad_step = 000342, loss = 0.001447
grad_step = 000343, loss = 0.001443
grad_step = 000344, loss = 0.001442
grad_step = 000345, loss = 0.001442
grad_step = 000346, loss = 0.001444
grad_step = 000347, loss = 0.001451
grad_step = 000348, loss = 0.001477
grad_step = 000349, loss = 0.001471
grad_step = 000350, loss = 0.001459
grad_step = 000351, loss = 0.001452
grad_step = 000352, loss = 0.001469
grad_step = 000353, loss = 0.001486
grad_step = 000354, loss = 0.001538
grad_step = 000355, loss = 0.001649
grad_step = 000356, loss = 0.001572
grad_step = 000357, loss = 0.001511
grad_step = 000358, loss = 0.001466
grad_step = 000359, loss = 0.001507
grad_step = 000360, loss = 0.001500
grad_step = 000361, loss = 0.001446
grad_step = 000362, loss = 0.001465
grad_step = 000363, loss = 0.001477
grad_step = 000364, loss = 0.001456
grad_step = 000365, loss = 0.001438
grad_step = 000366, loss = 0.001457
grad_step = 000367, loss = 0.001450
grad_step = 000368, loss = 0.001441
grad_step = 000369, loss = 0.001443
grad_step = 000370, loss = 0.001448
grad_step = 000371, loss = 0.001452
grad_step = 000372, loss = 0.001451
grad_step = 000373, loss = 0.001469
grad_step = 000374, loss = 0.001495
grad_step = 000375, loss = 0.001533
grad_step = 000376, loss = 0.001592
grad_step = 000377, loss = 0.001673
grad_step = 000378, loss = 0.001753
grad_step = 000379, loss = 0.001816
grad_step = 000380, loss = 0.001774
grad_step = 000381, loss = 0.001626
grad_step = 000382, loss = 0.001477
grad_step = 000383, loss = 0.001414
grad_step = 000384, loss = 0.001480
grad_step = 000385, loss = 0.001563
grad_step = 000386, loss = 0.001563
grad_step = 000387, loss = 0.001497
grad_step = 000388, loss = 0.001426
grad_step = 000389, loss = 0.001409
grad_step = 000390, loss = 0.001456
grad_step = 000391, loss = 0.001504
grad_step = 000392, loss = 0.001500
grad_step = 000393, loss = 0.001458
grad_step = 000394, loss = 0.001414
grad_step = 000395, loss = 0.001394
grad_step = 000396, loss = 0.001406
grad_step = 000397, loss = 0.001434
grad_step = 000398, loss = 0.001454
grad_step = 000399, loss = 0.001459
grad_step = 000400, loss = 0.001448
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001427
grad_step = 000402, loss = 0.001404
grad_step = 000403, loss = 0.001388
grad_step = 000404, loss = 0.001381
grad_step = 000405, loss = 0.001382
grad_step = 000406, loss = 0.001388
grad_step = 000407, loss = 0.001396
grad_step = 000408, loss = 0.001403
grad_step = 000409, loss = 0.001406
grad_step = 000410, loss = 0.001406
grad_step = 000411, loss = 0.001401
grad_step = 000412, loss = 0.001395
grad_step = 000413, loss = 0.001387
grad_step = 000414, loss = 0.001379
grad_step = 000415, loss = 0.001373
grad_step = 000416, loss = 0.001367
grad_step = 000417, loss = 0.001364
grad_step = 000418, loss = 0.001361
grad_step = 000419, loss = 0.001360
grad_step = 000420, loss = 0.001360
grad_step = 000421, loss = 0.001359
grad_step = 000422, loss = 0.001360
grad_step = 000423, loss = 0.001361
grad_step = 000424, loss = 0.001363
grad_step = 000425, loss = 0.001366
grad_step = 000426, loss = 0.001370
grad_step = 000427, loss = 0.001377
grad_step = 000428, loss = 0.001387
grad_step = 000429, loss = 0.001402
grad_step = 000430, loss = 0.001424
grad_step = 000431, loss = 0.001457
grad_step = 000432, loss = 0.001497
grad_step = 000433, loss = 0.001551
grad_step = 000434, loss = 0.001595
grad_step = 000435, loss = 0.001629
grad_step = 000436, loss = 0.001601
grad_step = 000437, loss = 0.001525
grad_step = 000438, loss = 0.001419
grad_step = 000439, loss = 0.001345
grad_step = 000440, loss = 0.001341
grad_step = 000441, loss = 0.001388
grad_step = 000442, loss = 0.001432
grad_step = 000443, loss = 0.001432
grad_step = 000444, loss = 0.001389
grad_step = 000445, loss = 0.001342
grad_step = 000446, loss = 0.001324
grad_step = 000447, loss = 0.001340
grad_step = 000448, loss = 0.001368
grad_step = 000449, loss = 0.001382
grad_step = 000450, loss = 0.001373
grad_step = 000451, loss = 0.001350
grad_step = 000452, loss = 0.001326
grad_step = 000453, loss = 0.001312
grad_step = 000454, loss = 0.001312
grad_step = 000455, loss = 0.001320
grad_step = 000456, loss = 0.001332
grad_step = 000457, loss = 0.001344
grad_step = 000458, loss = 0.001354
grad_step = 000459, loss = 0.001359
grad_step = 000460, loss = 0.001359
grad_step = 000461, loss = 0.001355
grad_step = 000462, loss = 0.001348
grad_step = 000463, loss = 0.001337
grad_step = 000464, loss = 0.001325
grad_step = 000465, loss = 0.001313
grad_step = 000466, loss = 0.001303
grad_step = 000467, loss = 0.001295
grad_step = 000468, loss = 0.001289
grad_step = 000469, loss = 0.001286
grad_step = 000470, loss = 0.001284
grad_step = 000471, loss = 0.001284
grad_step = 000472, loss = 0.001285
grad_step = 000473, loss = 0.001286
grad_step = 000474, loss = 0.001289
grad_step = 000475, loss = 0.001294
grad_step = 000476, loss = 0.001301
grad_step = 000477, loss = 0.001313
grad_step = 000478, loss = 0.001329
grad_step = 000479, loss = 0.001355
grad_step = 000480, loss = 0.001390
grad_step = 000481, loss = 0.001436
grad_step = 000482, loss = 0.001484
grad_step = 000483, loss = 0.001522
grad_step = 000484, loss = 0.001520
grad_step = 000485, loss = 0.001472
grad_step = 000486, loss = 0.001383
grad_step = 000487, loss = 0.001301
grad_step = 000488, loss = 0.001260
grad_step = 000489, loss = 0.001270
grad_step = 000490, loss = 0.001312
grad_step = 000491, loss = 0.001349
grad_step = 000492, loss = 0.001359
grad_step = 000493, loss = 0.001332
grad_step = 000494, loss = 0.001291
grad_step = 000495, loss = 0.001256
grad_step = 000496, loss = 0.001243
grad_step = 000497, loss = 0.001251
grad_step = 000498, loss = 0.001269
grad_step = 000499, loss = 0.001285
grad_step = 000500, loss = 0.001292
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001285
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

  date_run                              2020-05-13 04:14:51.511487
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.280771
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 04:14:51.517914
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.200103
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 04:14:51.526250
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.151918
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 04:14:51.532521
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.04064
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
0   2020-05-13 04:14:17.615643  ...    mean_absolute_error
1   2020-05-13 04:14:17.620192  ...     mean_squared_error
2   2020-05-13 04:14:17.623789  ...  median_absolute_error
3   2020-05-13 04:14:17.627881  ...               r2_score
4   2020-05-13 04:14:27.293454  ...    mean_absolute_error
5   2020-05-13 04:14:27.297816  ...     mean_squared_error
6   2020-05-13 04:14:27.301998  ...  median_absolute_error
7   2020-05-13 04:14:27.305818  ...               r2_score
8   2020-05-13 04:14:51.511487  ...    mean_absolute_error
9   2020-05-13 04:14:51.517914  ...     mean_squared_error
10  2020-05-13 04:14:51.526250  ...  median_absolute_error
11  2020-05-13 04:14:51.532521  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb681594cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 316604.23it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 408542.59it/s]  9%|▉         | 876544/9912422 [00:00<00:15, 565614.52it/s] 31%|███       | 3031040/9912422 [00:00<00:08, 797163.47it/s] 58%|█████▊    | 5783552/9912422 [00:00<00:03, 1121547.08it/s] 89%|████████▊ | 8790016/9912422 [00:01<00:00, 1570936.41it/s]9920512it [00:01, 9378587.87it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 143659.26it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 314488.34it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 406698.56it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 562923.41it/s]1654784it [00:00, 2812280.93it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 50254.47it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb633f4fe80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb63357c0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb633f4fe80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb6334d30f0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb630d0f4e0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb630cf9c50> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb633f4fe80> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb633491710> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb630d0f4e0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fb63357c128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f85ffda2208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=e835814d27379046753628020e5268989c5e10c0b138b7972fcc7c5da153c600
  Stored in directory: /tmp/pip-ephem-wheel-cache-c46_y7r7/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f8597b9e710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 45s
   57344/17464789 [..............................] - ETA: 38s
  106496/17464789 [..............................] - ETA: 31s
  212992/17464789 [..............................] - ETA: 20s
  385024/17464789 [..............................] - ETA: 14s
  786432/17464789 [>.............................] - ETA: 8s 
 1589248/17464789 [=>............................] - ETA: 4s
 3178496/17464789 [====>.........................] - ETA: 2s
 5980160/17464789 [=========>....................] - ETA: 1s
 8994816/17464789 [==============>...............] - ETA: 0s
12042240/17464789 [===================>..........] - ETA: 0s
14598144/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 04:16:24.883683: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 04:16:24.888497: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 04:16:24.888715: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56198f083650 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 04:16:24.888734: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5900 - accuracy: 0.5050
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6283 - accuracy: 0.5025
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6871 - accuracy: 0.4987 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6053 - accuracy: 0.5040
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6605 - accuracy: 0.5004
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6181 - accuracy: 0.5032
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6338 - accuracy: 0.5021
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6590 - accuracy: 0.5005
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6428 - accuracy: 0.5016
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6375 - accuracy: 0.5019
11000/25000 [============>.................] - ETA: 4s - loss: 7.6387 - accuracy: 0.5018
12000/25000 [=============>................] - ETA: 4s - loss: 7.6551 - accuracy: 0.5008
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6713 - accuracy: 0.4997
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6951 - accuracy: 0.4981
15000/25000 [=================>............] - ETA: 3s - loss: 7.6942 - accuracy: 0.4982
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6867 - accuracy: 0.4987
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6675 - accuracy: 0.4999
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6624 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6779 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6797 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6856 - accuracy: 0.4988
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6952 - accuracy: 0.4981
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6768 - accuracy: 0.4993
25000/25000 [==============================] - 10s 394us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 04:16:42.477887
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 04:16:42.477887  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<9:45:16, 24.6kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:00<6:50:39, 35.0kB/s] .vector_cache/glove.6B.zip:   1%|          | 5.69M/862M [00:00<4:45:47, 49.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.7M/862M [00:00<3:17:45, 71.3kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.5M/862M [00:00<2:17:02, 102kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 29.8M/862M [00:00<1:35:24, 145kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.7M/862M [00:00<1:06:06, 208kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.9M/862M [00:01<45:44, 296kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:01<32:37, 414kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:02<23:11, 580kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<8:43:51, 25.7kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:03<6:06:28, 36.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:04<9:22:52, 23.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:04<6:33:33, 34.0kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<10:18:37, 21.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:05<7:12:30, 30.8kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.7M/862M [00:06<10:40:32, 20.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:06<7:27:49, 29.7kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 63.8M/862M [00:07<10:43:02, 20.7kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:07<7:29:32, 29.5kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:08<10:53:25, 20.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:08<7:36:46, 29.0kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:09<10:55:24, 20.2kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:09<7:38:11, 28.8kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 70.1M/862M [00:10<10:47:46, 20.4kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:10<7:32:48, 29.1kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<10:55:36, 20.1kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:11<7:38:18, 28.7kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.3M/862M [00:12<10:50:03, 20.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:12<7:34:24, 28.8kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:13<10:50:27, 20.1kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:13<7:34:41, 28.7kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 78.5M/862M [00:14<10:48:09, 20.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:14<7:33:11, 28.7kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<10:17:08, 21.1kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:15<7:11:30, 30.1kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:16<10:11:51, 21.2kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:16<7:07:42, 30.3kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<10:30:53, 20.5kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:17<7:21:01, 29.3kB/s] .vector_cache/glove.6B.zip:  10%|█         | 86.9M/862M [00:18<10:35:42, 20.3kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:18<7:24:23, 29.0kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:19<10:29:46, 20.5kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.0M/862M [00:19<7:20:18, 29.2kB/s] .vector_cache/glove.6B.zip:  11%|█         | 91.1M/862M [00:20<10:12:51, 21.0kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:20<7:08:23, 29.9kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:21<10:28:02, 20.4kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:21<7:19:06, 29.1kB/s] .vector_cache/glove.6B.zip:  11%|█         | 95.2M/862M [00:22<10:03:17, 21.2kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:22<7:01:42, 30.2kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:23<10:20:33, 20.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:23<7:13:47, 29.3kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<10:20:52, 20.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:24<7:13:58, 29.2kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<10:27:44, 20.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:25<7:18:46, 28.8kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<10:26:26, 20.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:26<7:17:53, 28.8kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<10:19:28, 20.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:27<7:12:59, 29.0kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<10:24:37, 20.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<7:16:40, 28.7kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<10:04:13, 20.7kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:29<7:02:24, 29.6kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<9:54:01, 21.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<6:55:12, 30.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<10:10:07, 20.4kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:31<7:06:27, 29.2kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<10:11:51, 20.3kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<7:07:38, 29.0kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<10:16:15, 20.1kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:33<7:10:49, 28.7kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<9:49:19, 21.0kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:34<6:51:54, 29.9kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<10:03:38, 20.4kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:35<7:01:54, 29.1kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<10:06:45, 20.3kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:36<7:04:05, 28.9kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<10:00:33, 20.4kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:37<6:59:44, 29.1kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<10:04:42, 20.2kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<7:02:38, 28.8kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:39<10:00:36, 20.3kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:39<6:59:45, 29.0kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<10:03:54, 20.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<7:02:03, 28.7kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:41<10:02:02, 20.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:41<7:00:52, 28.7kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<9:32:49, 21.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<6:40:20, 30.1kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<9:48:39, 20.5kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<6:51:23, 29.2kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<9:52:43, 20.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<6:54:20, 28.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<9:26:13, 21.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:45<6:35:48, 30.2kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<9:21:05, 21.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:46<6:32:07, 30.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<9:40:24, 20.5kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:47<6:45:38, 29.3kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<9:40:04, 20.5kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:48<6:45:22, 29.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<9:45:29, 20.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:49<6:49:10, 28.8kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<9:42:55, 20.2kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<6:47:21, 28.9kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<9:45:44, 20.1kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:51<6:49:19, 28.7kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<9:42:37, 20.1kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<6:47:14, 28.7kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<9:15:22, 21.1kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<6:28:07, 30.1kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<9:27:57, 20.5kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<6:36:59, 29.3kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<9:09:17, 21.2kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:55<6:23:51, 30.2kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<9:25:01, 20.5kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:56<6:34:50, 29.3kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<9:28:02, 20.3kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<6:36:58, 29.0kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<9:22:09, 20.5kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:58<6:32:54, 29.2kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<9:05:03, 21.1kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [00:59<6:20:57, 30.1kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<8:59:00, 21.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:00<6:16:39, 30.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<9:15:18, 20.6kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:01<6:28:02, 29.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<9:19:28, 20.3kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<6:30:58, 29.0kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<9:12:21, 20.5kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:03<6:25:57, 29.3kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:04<9:19:46, 20.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:04<6:31:08, 28.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:05<9:19:03, 20.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:05<6:30:44, 28.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<8:50:56, 21.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:06<6:10:59, 30.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:07<9:06:30, 20.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:07<6:21:56, 29.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<8:48:41, 21.1kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:08<6:09:25, 30.2kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<9:01:13, 20.6kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:09<6:18:11, 29.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<9:00:43, 20.5kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:10<6:17:52, 29.3kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<8:48:18, 21.0kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:11<6:09:07, 29.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<9:00:46, 20.4kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:12<6:17:51, 29.1kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<8:54:47, 20.6kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:13<6:13:42, 29.3kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<8:43:07, 21.0kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<6:05:29, 29.9kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<8:55:42, 20.4kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:15<6:14:17, 29.1kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<8:53:14, 20.4kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:16<6:12:32, 29.2kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<8:58:00, 20.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:17<6:15:52, 28.8kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<8:56:13, 20.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:18<6:14:38, 28.8kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<8:54:25, 20.2kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<6:13:14, 28.7kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<4:22:09, 40.9kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:21<3:03:36, 58.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<5:51:37, 30.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<4:06:04, 43.1kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<2:52:50, 61.4kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<2:01:21, 87.2kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<5:03:29, 34.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<3:32:34, 49.5kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<2:29:25, 70.3kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:27<1:44:53, 99.9kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<5:09:52, 33.8kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:28<3:36:38, 48.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<6:57:55, 25.0kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:29<4:52:02, 35.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<7:46:51, 22.3kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:30<5:26:11, 31.8kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<8:03:38, 21.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<5:37:55, 30.6kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<8:03:21, 21.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:32<5:37:39, 30.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<8:20:29, 20.6kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:33<5:49:37, 29.4kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<8:23:33, 20.4kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:34<5:51:46, 29.1kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<8:19:11, 20.5kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<5:48:32, 29.2kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<4:04:25, 41.5kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:37<2:51:11, 59.1kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<5:50:28, 28.9kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:38<4:04:56, 41.2kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<7:06:49, 23.6kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<4:58:11, 33.6kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<3:29:14, 47.9kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:41<2:26:37, 68.1kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<5:26:41, 30.6kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<3:48:31, 43.4kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<2:40:39, 61.7kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:44<1:52:39, 87.7kB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<5:02:08, 32.7kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<3:31:25, 46.4kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<2:28:32, 66.0kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:47<1:44:13, 93.8kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<4:54:47, 33.2kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<3:26:17, 47.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<2:24:58, 66.9kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:50<1:41:47, 95.0kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<4:33:28, 35.4kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<3:11:26, 50.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<2:14:41, 71.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<1:34:30, 101kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<4:41:36, 34.0kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<3:17:04, 48.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<2:18:24, 68.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:56<1:37:08, 97.4kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<4:42:16, 33.5kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<3:17:30, 47.6kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<2:18:34, 67.7kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [01:59<1:37:15, 96.2kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<4:49:22, 32.3kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<3:22:24, 45.9kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<2:22:08, 65.3kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:02<1:39:48, 92.7kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<4:22:53, 35.2kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<3:03:58, 49.9kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<2:09:17, 71.0kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<1:30:48, 101kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<4:14:05, 36.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<2:57:53, 51.0kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<2:05:26, 72.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:08<1:27:24, 103kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<1:06:41, 135kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<46:56, 192kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<34:10, 262kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<24:16, 368kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:12<17:00, 522kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<17:32, 506kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<12:54, 687kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:14<09:26, 936kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<3:52:31, 38.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<2:42:45, 53.8kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<1:54:29, 76.5kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<1:20:53, 107kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<56:52, 153kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<40:59, 210kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<29:01, 297kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<21:34, 397kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<15:28, 552kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:23<11:15, 757kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:24<3:51:58, 36.7kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<2:42:17, 52.1kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:26<1:54:10, 73.9kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<1:20:36, 104kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:28<56:53, 147kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<40:48, 204kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<29:05, 285kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<21:28, 384kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<15:29, 531kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:32<11:12, 732kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<3:42:52, 36.8kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:33<2:34:52, 52.6kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<2:01:31, 66.9kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<1:25:17, 95.2kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<1:00:36, 133kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<42:54, 188kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<31:02, 258kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<21:59, 363kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<16:36, 477kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<11:58, 661kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:41<09:15, 852kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<1:55:25, 68.3kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<1:21:13, 96.3kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<57:39, 136kB/s]   .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:44<40:07, 193kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<42:24, 183kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<30:43, 252kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:46<21:25, 359kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<22:10, 346kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<16:29, 466kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:48<11:51, 645kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<3:16:09, 39.0kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<2:17:10, 55.2kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<1:36:30, 78.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:51<1:07:04, 112kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<1:40:54, 74.4kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<1:11:09, 105kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<50:30, 147kB/s]  .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<35:55, 207kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:55<25:28, 291kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<3:08:40, 39.2kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:56<2:10:54, 56.0kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<6:00:40, 20.3kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<4:13:09, 28.9kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:58<2:56:03, 41.3kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<2:07:37, 56.9kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<1:29:51, 80.7kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<1:03:26, 113kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<45:09, 159kB/s]  .vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:02<31:27, 227kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<26:41, 267kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<19:16, 369kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<14:25, 489kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<10:44, 656kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:06<07:31, 930kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<11:00, 635kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<08:27, 826kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:08<05:56, 1.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<08:27, 818kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<06:56, 997kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:10<05:06, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<3:25:14, 33.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:11<2:22:40, 47.8kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<1:43:09, 66.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<1:12:46, 93.5kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<51:27, 131kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<36:34, 184kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:15<25:25, 262kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<38:20, 174kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<27:29, 242kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:17<19:08, 345kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<22:28, 294kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:19<16:15, 406kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<12:15, 533kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<09:06, 717kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<07:17, 887kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<05:37, 1.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<04:51, 1.31MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<03:57, 1.61MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:25<03:01, 2.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<3:08:43, 33.7kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:26<2:10:53, 48.1kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:28<1:39:26, 63.2kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<1:10:02, 89.6kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<49:28, 126kB/s]   .vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<35:06, 177kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<25:15, 243kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<18:09, 338kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<13:29, 450kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<10:00, 606kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:34<06:59, 860kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<18:16, 329kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<13:17, 452kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<10:05, 588kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<07:32, 786kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<06:06, 961kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<04:45, 1.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:40<03:33, 1.64MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:41<2:54:21, 33.4kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:41<2:00:47, 47.7kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<1:32:38, 62.2kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<1:05:17, 88.1kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<46:03, 124kB/s]   .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<32:40, 174kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<23:27, 240kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<16:51, 333kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<12:30, 444kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<09:07, 607kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:49<06:21, 862kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<1:33:11, 58.8kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<1:05:31, 83.6kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<46:10, 117kB/s]   .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<32:40, 166kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<23:24, 228kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<16:50, 317kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:55<11:56, 444kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<2:43:13, 32.5kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:56<1:52:56, 46.4kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<1:27:53, 59.5kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:58<1:01:52, 84.5kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<43:34, 119kB/s]   .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<30:49, 167kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:00<21:22, 239kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<1:09:53, 72.9kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<49:17, 103kB/s]   .vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<34:50, 144kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<24:45, 203kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:06<17:52, 278kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<12:54, 384kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<09:39, 506kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<07:22, 663kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:08<05:08, 940kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<20:00, 241kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<14:23, 335kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<10:12, 468kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<2:27:13, 32.5kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:11<1:42:02, 46.3kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<1:13:23, 64.2kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<51:50, 90.8kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:13<35:49, 130kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<2:05:21, 37.0kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<1:28:19, 52.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:15<1:01:22, 74.9kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<44:11, 103kB/s]   .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<31:28, 145kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:17<21:46, 207kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<21:59, 205kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<15:54, 283kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:19<11:07, 402kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<09:00, 492kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<06:40, 663kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:21<04:41, 935kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<04:46, 915kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<04:06, 1.06MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<03:02, 1.43MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<02:19, 1.87MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<01:48, 2.40MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:23<01:27, 2.98MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:23<01:11, 3.61MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<05:37, 765kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<04:18, 997kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<03:14, 1.32MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:25<02:27, 1.74MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<01:53, 2.25MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:25<02:33, 1.66MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<1:25:22, 49.8kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:26<59:49, 70.9kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:26<41:57, 101kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:26<29:26, 143kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:26<20:42, 203kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:26<14:37, 287kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<13:37, 307kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<09:53, 422kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<07:04, 589kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:28<05:05, 814kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:28<03:42, 1.12MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:28<02:43, 1.51MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<05:57, 692kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<04:43, 870kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:30<03:26, 1.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:30<02:31, 1.61MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:30<01:53, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:30<01:27, 2.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<08:29, 477kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<06:18, 641kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:32<04:31, 889kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:32<03:18, 1.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:32<02:25, 1.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<03:47, 1.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<02:59, 1.33MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:34<02:12, 1.79MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:34<01:39, 2.38MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:34<01:15, 3.12MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<18:48, 208kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<13:30, 289kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<09:30, 408kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:36<06:43, 575kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:36<04:48, 800kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<15:09, 253kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<10:57, 350kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:38<07:45, 492kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:38<05:29, 691kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:38<03:55, 961kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<07:36, 496kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<05:37, 669kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<04:00, 934kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<03:13, 1.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<1:34:28, 39.5kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<1:05:58, 56.3kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:41<45:58, 80.3kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:41<32:07, 114kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<25:31, 143kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<18:07, 201kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<12:42, 286kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:43<08:55, 405kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<07:48, 460kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<05:43, 627kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<04:04, 876kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:45<02:55, 1.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<03:23, 1.04MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<02:37, 1.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:55, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:47<01:24, 2.46MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<02:37, 1.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<02:04, 1.66MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:49<01:32, 2.23MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:49<01:08, 2.98MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<02:30, 1.35MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<02:01, 1.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<01:29, 2.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:51<01:07, 2.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<02:17, 1.45MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:52, 1.76MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:53<01:22, 2.40MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:53<01:00, 3.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<03:42, 875kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<02:50, 1.14MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<02:03, 1.56MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:55<01:49, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<1:23:00, 38.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<58:05, 55.0kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<40:26, 78.5kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:56<28:09, 112kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<21:49, 144kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<15:28, 202kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<10:50, 287kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:58<07:35, 406kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<06:55, 443kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<05:13, 586kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<03:42, 821kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:00<02:37, 1.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<03:25, 874kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<02:39, 1.12MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<01:55, 1.54MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:02<01:23, 2.11MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<02:20, 1.25MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<02:02, 1.43MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:29, 1.95MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:04<01:05, 2.64MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<02:06, 1.36MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:52, 1.53MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<01:21, 2.08MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:06<00:59, 2.82MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<02:12, 1.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<01:45, 1.58MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<01:17, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:08<00:56, 2.90MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<02:13, 1.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:45, 1.54MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<01:16, 2.11MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:10<01:14, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<1:10:57, 37.8kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<49:31, 53.9kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:11<34:18, 76.9kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<24:41, 106kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<17:25, 150kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:13<12:06, 213kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:13<08:26, 302kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<09:23, 271kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<06:45, 375kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<04:43, 530kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:15<03:19, 747kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<05:48, 426kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<04:15, 579kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:17<02:58, 817kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<02:45, 872kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<02:16, 1.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:19<01:37, 1.47MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:19<01:09, 2.02MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<02:31, 925kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<01:57, 1.19MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:21<01:23, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:36, 1.41MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<01:17, 1.74MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:23<00:56, 2.39MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:14, 1.77MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<01:01, 2.13MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:25<00:44, 2.89MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:25<02:11, 988kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:26<59:56, 36.0kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<41:43, 51.3kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:26<28:43, 73.2kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:28<21:05, 99.0kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<14:54, 140kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:28<10:18, 199kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<07:41, 263kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<05:31, 364kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:30<03:49, 516kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<03:26, 566kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:32<02:33, 759kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:32<01:47, 1.07MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:51, 1.01MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<01:31, 1.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:34<01:04, 1.71MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:18, 1.39MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<01:03, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:36<00:45, 2.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:10, 1.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:02, 1.68MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:38<00:43, 2.33MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<01:03, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:51, 1.94MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<00:36, 2.67MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<01:56, 844kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<47:55, 34.1kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:41<33:07, 48.6kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<22:49, 68.5kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:43<16:01, 97.0kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:43<10:52, 138kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<08:44, 171kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<06:12, 240kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:45<04:12, 341kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<04:03, 351kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<02:56, 483kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:47<02:00, 685kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<02:08, 635kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<01:41, 804kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:49<01:09, 1.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<01:18, 984kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<01:01, 1.25MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:51<00:42, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:57, 1.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:46, 1.57MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:53<00:31, 2.20MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<02:39, 432kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<01:57, 585kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:55<01:22, 805kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<32:43, 33.9kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:56<22:13, 48.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<15:18, 67.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<10:43, 96.1kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [05:58<07:05, 137kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<09:43, 99.8kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<06:50, 141kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:00<04:34, 201kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<03:37, 248kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<02:35, 344kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<01:48, 458kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<01:19, 621kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:58, 783kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:44, 1.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:34, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:27, 1.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:23, 1.62MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:19, 1.95MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:10<00:14, 2.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<16:50, 34.7kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:11<10:57, 49.5kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<07:26, 69.1kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:13<05:11, 97.9kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:13<03:17, 140kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<02:31, 177kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<01:45, 248kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:15<01:05, 354kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<01:09, 323kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:49, 447kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:17<00:30, 634kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:30, 605kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:23, 781kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:19<00:13, 1.10MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:17, 839kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:12, 1.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:08, 1.26MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:06, 1.55MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:23<00:03, 2.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:07, 777kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:05, 987kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:25<00:02, 1.33MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<01:43, 34.6kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:26<00:10, 49.4kB/s].vector_cache/glove.6B.zip: 862MB [06:26, 2.23MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 806/400000 [00:00<00:49, 8055.23it/s]  0%|          | 1657/400000 [00:00<00:48, 8184.28it/s]  1%|          | 2526/400000 [00:00<00:47, 8328.35it/s]  1%|          | 3355/400000 [00:00<00:47, 8302.09it/s]  1%|          | 4121/400000 [00:00<00:48, 8097.92it/s]  1%|          | 4951/400000 [00:00<00:48, 8155.47it/s]  1%|▏         | 5805/400000 [00:00<00:47, 8266.20it/s]  2%|▏         | 6601/400000 [00:00<00:48, 8170.71it/s]  2%|▏         | 7486/400000 [00:00<00:46, 8362.87it/s]  2%|▏         | 8292/400000 [00:01<00:47, 8217.94it/s]  2%|▏         | 9113/400000 [00:01<00:47, 8212.92it/s]  2%|▏         | 9922/400000 [00:01<00:47, 8174.48it/s]  3%|▎         | 10814/400000 [00:01<00:46, 8382.73it/s]  3%|▎         | 11741/400000 [00:01<00:44, 8628.78it/s]  3%|▎         | 12602/400000 [00:01<00:45, 8586.66it/s]  3%|▎         | 13484/400000 [00:01<00:44, 8655.34it/s]  4%|▎         | 14389/400000 [00:01<00:43, 8769.71it/s]  4%|▍         | 15275/400000 [00:01<00:43, 8796.07it/s]  4%|▍         | 16155/400000 [00:01<00:44, 8696.19it/s]  4%|▍         | 17025/400000 [00:02<00:44, 8566.62it/s]  4%|▍         | 17883/400000 [00:02<00:45, 8463.52it/s]  5%|▍         | 18731/400000 [00:02<00:45, 8407.93it/s]  5%|▍         | 19584/400000 [00:02<00:45, 8443.31it/s]  5%|▌         | 20472/400000 [00:02<00:44, 8567.38it/s]  5%|▌         | 21358/400000 [00:02<00:43, 8652.40it/s]  6%|▌         | 22235/400000 [00:02<00:43, 8686.58it/s]  6%|▌         | 23105/400000 [00:02<00:44, 8509.10it/s]  6%|▌         | 23995/400000 [00:02<00:43, 8622.36it/s]  6%|▌         | 24869/400000 [00:02<00:43, 8656.91it/s]  6%|▋         | 25736/400000 [00:03<00:44, 8428.22it/s]  7%|▋         | 26581/400000 [00:03<00:44, 8416.69it/s]  7%|▋         | 27490/400000 [00:03<00:43, 8606.11it/s]  7%|▋         | 28377/400000 [00:03<00:42, 8680.87it/s]  7%|▋         | 29247/400000 [00:03<00:43, 8572.66it/s]  8%|▊         | 30106/400000 [00:03<00:43, 8547.47it/s]  8%|▊         | 30992/400000 [00:03<00:42, 8636.45it/s]  8%|▊         | 31871/400000 [00:03<00:42, 8677.45it/s]  8%|▊         | 32740/400000 [00:03<00:43, 8436.08it/s]  8%|▊         | 33607/400000 [00:03<00:43, 8502.80it/s]  9%|▊         | 34459/400000 [00:04<00:43, 8419.56it/s]  9%|▉         | 35322/400000 [00:04<00:43, 8479.86it/s]  9%|▉         | 36200/400000 [00:04<00:42, 8565.53it/s]  9%|▉         | 37108/400000 [00:04<00:41, 8711.77it/s] 10%|▉         | 38009/400000 [00:04<00:41, 8798.29it/s] 10%|▉         | 38890/400000 [00:04<00:42, 8557.07it/s] 10%|▉         | 39768/400000 [00:04<00:41, 8621.13it/s] 10%|█         | 40632/400000 [00:04<00:41, 8621.67it/s] 10%|█         | 41546/400000 [00:04<00:40, 8769.67it/s] 11%|█         | 42440/400000 [00:04<00:40, 8818.46it/s] 11%|█         | 43323/400000 [00:05<00:41, 8667.84it/s] 11%|█         | 44216/400000 [00:05<00:40, 8742.65it/s] 11%|█▏        | 45108/400000 [00:05<00:40, 8793.44it/s] 12%|█▏        | 46008/400000 [00:05<00:39, 8853.72it/s] 12%|█▏        | 46895/400000 [00:05<00:40, 8760.70it/s] 12%|█▏        | 47772/400000 [00:05<00:41, 8528.40it/s] 12%|█▏        | 48661/400000 [00:05<00:40, 8633.68it/s] 12%|█▏        | 49536/400000 [00:05<00:40, 8665.49it/s] 13%|█▎        | 50436/400000 [00:05<00:39, 8763.07it/s] 13%|█▎        | 51316/400000 [00:05<00:39, 8771.11it/s] 13%|█▎        | 52194/400000 [00:06<00:40, 8547.32it/s] 13%|█▎        | 53080/400000 [00:06<00:40, 8636.15it/s] 13%|█▎        | 53957/400000 [00:06<00:39, 8675.24it/s] 14%|█▎        | 54892/400000 [00:06<00:38, 8864.46it/s] 14%|█▍        | 55799/400000 [00:06<00:38, 8922.88it/s] 14%|█▍        | 56693/400000 [00:06<00:39, 8744.33it/s] 14%|█▍        | 57570/400000 [00:06<00:39, 8748.38it/s] 15%|█▍        | 58447/400000 [00:06<00:39, 8638.25it/s] 15%|█▍        | 59312/400000 [00:06<00:40, 8416.60it/s] 15%|█▌        | 60159/400000 [00:07<00:40, 8430.27it/s] 15%|█▌        | 61004/400000 [00:07<00:41, 8251.30it/s] 15%|█▌        | 61847/400000 [00:07<00:40, 8301.47it/s] 16%|█▌        | 62679/400000 [00:07<00:40, 8232.26it/s] 16%|█▌        | 63576/400000 [00:07<00:39, 8439.49it/s] 16%|█▌        | 64474/400000 [00:07<00:39, 8593.35it/s] 16%|█▋        | 65336/400000 [00:07<00:38, 8584.74it/s] 17%|█▋        | 66232/400000 [00:07<00:38, 8693.32it/s] 17%|█▋        | 67103/400000 [00:07<00:38, 8560.27it/s] 17%|█▋        | 68003/400000 [00:07<00:38, 8687.41it/s] 17%|█▋        | 68874/400000 [00:08<00:38, 8597.35it/s] 17%|█▋        | 69735/400000 [00:08<00:39, 8419.91it/s] 18%|█▊        | 70616/400000 [00:08<00:38, 8532.14it/s] 18%|█▊        | 71471/400000 [00:08<00:38, 8512.58it/s] 18%|█▊        | 72376/400000 [00:08<00:37, 8666.01it/s] 18%|█▊        | 73256/400000 [00:08<00:37, 8705.38it/s] 19%|█▊        | 74128/400000 [00:08<00:38, 8368.05it/s] 19%|█▊        | 74969/400000 [00:08<00:38, 8361.06it/s] 19%|█▉        | 75815/400000 [00:08<00:38, 8389.75it/s] 19%|█▉        | 76656/400000 [00:08<00:38, 8395.41it/s] 19%|█▉        | 77562/400000 [00:09<00:37, 8582.06it/s] 20%|█▉        | 78423/400000 [00:09<00:37, 8465.37it/s] 20%|█▉        | 79272/400000 [00:09<00:37, 8454.51it/s] 20%|██        | 80119/400000 [00:09<00:37, 8426.22it/s] 20%|██        | 80982/400000 [00:09<00:37, 8486.12it/s] 20%|██        | 81868/400000 [00:09<00:37, 8593.92it/s] 21%|██        | 82729/400000 [00:09<00:37, 8537.58it/s] 21%|██        | 83595/400000 [00:09<00:36, 8571.86it/s] 21%|██        | 84468/400000 [00:09<00:36, 8617.67it/s] 21%|██▏       | 85331/400000 [00:09<00:36, 8604.70it/s] 22%|██▏       | 86192/400000 [00:10<00:36, 8548.83it/s] 22%|██▏       | 87048/400000 [00:10<00:36, 8467.91it/s] 22%|██▏       | 87938/400000 [00:10<00:36, 8591.54it/s] 22%|██▏       | 88812/400000 [00:10<00:36, 8634.99it/s] 22%|██▏       | 89677/400000 [00:10<00:36, 8550.93it/s] 23%|██▎       | 90533/400000 [00:10<00:37, 8300.95it/s] 23%|██▎       | 91367/400000 [00:10<00:37, 8312.34it/s] 23%|██▎       | 92200/400000 [00:10<00:37, 8291.44it/s] 23%|██▎       | 93078/400000 [00:10<00:36, 8431.51it/s] 23%|██▎       | 93939/400000 [00:10<00:36, 8482.18it/s] 24%|██▎       | 94789/400000 [00:11<00:36, 8395.83it/s] 24%|██▍       | 95630/400000 [00:11<00:36, 8282.93it/s] 24%|██▍       | 96557/400000 [00:11<00:35, 8554.62it/s] 24%|██▍       | 97419/400000 [00:11<00:35, 8571.76it/s] 25%|██▍       | 98303/400000 [00:11<00:34, 8649.73it/s] 25%|██▍       | 99188/400000 [00:11<00:34, 8706.98it/s] 25%|██▌       | 100060/400000 [00:11<00:35, 8515.73it/s] 25%|██▌       | 100969/400000 [00:11<00:34, 8677.31it/s] 25%|██▌       | 101839/400000 [00:11<00:34, 8615.93it/s] 26%|██▌       | 102759/400000 [00:12<00:33, 8780.32it/s] 26%|██▌       | 103639/400000 [00:12<00:33, 8759.93it/s] 26%|██▌       | 104517/400000 [00:12<00:35, 8440.22it/s] 26%|██▋       | 105365/400000 [00:12<00:34, 8441.09it/s] 27%|██▋       | 106232/400000 [00:12<00:34, 8506.40it/s] 27%|██▋       | 107135/400000 [00:12<00:33, 8655.11it/s] 27%|██▋       | 108033/400000 [00:12<00:33, 8749.40it/s] 27%|██▋       | 108910/400000 [00:12<00:33, 8636.97it/s] 27%|██▋       | 109793/400000 [00:12<00:33, 8693.44it/s] 28%|██▊       | 110664/400000 [00:12<00:33, 8640.40it/s] 28%|██▊       | 111529/400000 [00:13<00:33, 8588.45it/s] 28%|██▊       | 112389/400000 [00:13<00:33, 8563.23it/s] 28%|██▊       | 113246/400000 [00:13<00:34, 8373.53it/s] 29%|██▊       | 114097/400000 [00:13<00:33, 8410.47it/s] 29%|██▊       | 114994/400000 [00:13<00:33, 8570.05it/s] 29%|██▉       | 115894/400000 [00:13<00:32, 8692.48it/s] 29%|██▉       | 116765/400000 [00:13<00:32, 8628.02it/s] 29%|██▉       | 117640/400000 [00:13<00:32, 8662.45it/s] 30%|██▉       | 118508/400000 [00:13<00:33, 8455.15it/s] 30%|██▉       | 119356/400000 [00:13<00:33, 8402.30it/s] 30%|███       | 120203/400000 [00:14<00:33, 8419.78it/s] 30%|███       | 121055/400000 [00:14<00:33, 8448.34it/s] 30%|███       | 121978/400000 [00:14<00:32, 8668.43it/s] 31%|███       | 122918/400000 [00:14<00:31, 8874.92it/s] 31%|███       | 123859/400000 [00:14<00:30, 9026.76it/s] 31%|███       | 124765/400000 [00:14<00:30, 8949.88it/s] 31%|███▏      | 125662/400000 [00:14<00:31, 8765.15it/s] 32%|███▏      | 126541/400000 [00:14<00:31, 8737.73it/s] 32%|███▏      | 127417/400000 [00:14<00:31, 8607.25it/s] 32%|███▏      | 128330/400000 [00:14<00:31, 8756.24it/s] 32%|███▏      | 129208/400000 [00:15<00:31, 8709.82it/s] 33%|███▎      | 130081/400000 [00:15<00:31, 8642.37it/s] 33%|███▎      | 130961/400000 [00:15<00:30, 8687.90it/s] 33%|███▎      | 131873/400000 [00:15<00:30, 8810.55it/s] 33%|███▎      | 132755/400000 [00:15<00:30, 8792.49it/s] 33%|███▎      | 133635/400000 [00:15<00:30, 8687.76it/s] 34%|███▎      | 134505/400000 [00:15<00:31, 8507.68it/s] 34%|███▍      | 135358/400000 [00:15<00:31, 8498.45it/s] 34%|███▍      | 136209/400000 [00:15<00:31, 8432.94it/s] 34%|███▍      | 137076/400000 [00:15<00:30, 8502.09it/s] 34%|███▍      | 137927/400000 [00:16<00:31, 8346.52it/s] 35%|███▍      | 138783/400000 [00:16<00:31, 8407.78it/s] 35%|███▍      | 139692/400000 [00:16<00:30, 8600.12it/s] 35%|███▌      | 140579/400000 [00:16<00:29, 8679.07it/s] 35%|███▌      | 141456/400000 [00:16<00:29, 8705.76it/s] 36%|███▌      | 142382/400000 [00:16<00:29, 8864.40it/s] 36%|███▌      | 143272/400000 [00:16<00:28, 8872.42it/s] 36%|███▌      | 144187/400000 [00:16<00:28, 8951.82it/s] 36%|███▋      | 145084/400000 [00:16<00:28, 8907.30it/s] 36%|███▋      | 145976/400000 [00:17<00:28, 8870.18it/s] 37%|███▋      | 146869/400000 [00:17<00:28, 8886.58it/s] 37%|███▋      | 147759/400000 [00:17<00:29, 8495.44it/s] 37%|███▋      | 148613/400000 [00:17<00:29, 8410.24it/s] 37%|███▋      | 149468/400000 [00:17<00:29, 8451.52it/s] 38%|███▊      | 150392/400000 [00:17<00:28, 8673.12it/s] 38%|███▊      | 151297/400000 [00:17<00:28, 8779.60it/s] 38%|███▊      | 152178/400000 [00:17<00:28, 8749.82it/s] 38%|███▊      | 153055/400000 [00:17<00:28, 8629.39it/s] 38%|███▊      | 153920/400000 [00:17<00:28, 8591.08it/s] 39%|███▊      | 154783/400000 [00:18<00:28, 8601.80it/s] 39%|███▉      | 155644/400000 [00:18<00:28, 8521.92it/s] 39%|███▉      | 156508/400000 [00:18<00:28, 8555.76it/s] 39%|███▉      | 157418/400000 [00:18<00:27, 8710.31it/s] 40%|███▉      | 158308/400000 [00:18<00:27, 8763.60it/s] 40%|███▉      | 159204/400000 [00:18<00:27, 8819.47it/s] 40%|████      | 160142/400000 [00:18<00:26, 8980.31it/s] 40%|████      | 161042/400000 [00:18<00:26, 8909.28it/s] 40%|████      | 161934/400000 [00:18<00:26, 8818.06it/s] 41%|████      | 162817/400000 [00:18<00:27, 8591.75it/s] 41%|████      | 163679/400000 [00:19<00:27, 8548.35it/s] 41%|████      | 164564/400000 [00:19<00:27, 8635.47it/s] 41%|████▏     | 165430/400000 [00:19<00:27, 8640.82it/s] 42%|████▏     | 166295/400000 [00:19<00:27, 8446.91it/s] 42%|████▏     | 167157/400000 [00:19<00:27, 8497.21it/s] 42%|████▏     | 168008/400000 [00:19<00:27, 8451.28it/s] 42%|████▏     | 168882/400000 [00:19<00:27, 8530.18it/s] 42%|████▏     | 169736/400000 [00:19<00:27, 8401.12it/s] 43%|████▎     | 170578/400000 [00:19<00:28, 8152.63it/s] 43%|████▎     | 171417/400000 [00:19<00:27, 8222.03it/s] 43%|████▎     | 172253/400000 [00:20<00:27, 8261.83it/s] 43%|████▎     | 173152/400000 [00:20<00:26, 8466.25it/s] 44%|████▎     | 174050/400000 [00:20<00:26, 8613.79it/s] 44%|████▎     | 174914/400000 [00:20<00:26, 8481.75it/s] 44%|████▍     | 175766/400000 [00:20<00:26, 8491.76it/s] 44%|████▍     | 176617/400000 [00:20<00:26, 8360.65it/s] 44%|████▍     | 177455/400000 [00:20<00:26, 8319.39it/s] 45%|████▍     | 178345/400000 [00:20<00:26, 8483.62it/s] 45%|████▍     | 179221/400000 [00:20<00:25, 8564.18it/s] 45%|████▌     | 180105/400000 [00:20<00:25, 8643.68it/s] 45%|████▌     | 180971/400000 [00:21<00:25, 8629.77it/s] 45%|████▌     | 181835/400000 [00:21<00:25, 8606.15it/s] 46%|████▌     | 182703/400000 [00:21<00:25, 8627.28it/s] 46%|████▌     | 183585/400000 [00:21<00:24, 8683.48it/s] 46%|████▌     | 184454/400000 [00:21<00:25, 8593.13it/s] 46%|████▋     | 185365/400000 [00:21<00:24, 8739.98it/s] 47%|████▋     | 186240/400000 [00:21<00:24, 8693.33it/s] 47%|████▋     | 187111/400000 [00:21<00:24, 8679.71it/s] 47%|████▋     | 187998/400000 [00:21<00:24, 8735.36it/s] 47%|████▋     | 188882/400000 [00:21<00:24, 8763.79it/s] 47%|████▋     | 189759/400000 [00:22<00:24, 8735.71it/s] 48%|████▊     | 190633/400000 [00:22<00:24, 8662.37it/s] 48%|████▊     | 191500/400000 [00:22<00:25, 8319.09it/s] 48%|████▊     | 192356/400000 [00:22<00:24, 8387.52it/s] 48%|████▊     | 193238/400000 [00:22<00:24, 8511.02it/s] 49%|████▊     | 194092/400000 [00:22<00:24, 8439.03it/s] 49%|████▉     | 195026/400000 [00:22<00:23, 8689.91it/s] 49%|████▉     | 195898/400000 [00:22<00:23, 8563.85it/s] 49%|████▉     | 196767/400000 [00:22<00:23, 8601.09it/s] 49%|████▉     | 197640/400000 [00:23<00:23, 8637.74it/s] 50%|████▉     | 198561/400000 [00:23<00:22, 8800.37it/s] 50%|████▉     | 199453/400000 [00:23<00:22, 8833.66it/s] 50%|█████     | 200338/400000 [00:23<00:22, 8690.66it/s] 50%|█████     | 201209/400000 [00:23<00:23, 8560.46it/s] 51%|█████     | 202067/400000 [00:23<00:23, 8496.61it/s] 51%|█████     | 202933/400000 [00:23<00:23, 8542.49it/s] 51%|█████     | 203789/400000 [00:23<00:23, 8477.32it/s] 51%|█████     | 204638/400000 [00:23<00:23, 8293.39it/s] 51%|█████▏    | 205469/400000 [00:23<00:23, 8253.63it/s] 52%|█████▏    | 206296/400000 [00:24<00:23, 8201.57it/s] 52%|█████▏    | 207212/400000 [00:24<00:22, 8466.90it/s] 52%|█████▏    | 208133/400000 [00:24<00:22, 8676.39it/s] 52%|█████▏    | 209004/400000 [00:24<00:22, 8618.24it/s] 52%|█████▏    | 209930/400000 [00:24<00:21, 8799.26it/s] 53%|█████▎    | 210832/400000 [00:24<00:21, 8861.88it/s] 53%|█████▎    | 211762/400000 [00:24<00:20, 8986.55it/s] 53%|█████▎    | 212663/400000 [00:24<00:21, 8890.05it/s] 53%|█████▎    | 213554/400000 [00:24<00:21, 8745.93it/s] 54%|█████▎    | 214431/400000 [00:24<00:21, 8441.17it/s] 54%|█████▍    | 215302/400000 [00:25<00:21, 8518.11it/s] 54%|█████▍    | 216176/400000 [00:25<00:21, 8580.92it/s] 54%|█████▍    | 217121/400000 [00:25<00:20, 8821.51it/s] 55%|█████▍    | 218007/400000 [00:25<00:20, 8769.84it/s] 55%|█████▍    | 218948/400000 [00:25<00:20, 8950.55it/s] 55%|█████▍    | 219865/400000 [00:25<00:19, 9014.06it/s] 55%|█████▌    | 220769/400000 [00:25<00:20, 8868.76it/s] 55%|█████▌    | 221658/400000 [00:25<00:20, 8652.32it/s] 56%|█████▌    | 222526/400000 [00:25<00:21, 8195.15it/s] 56%|█████▌    | 223437/400000 [00:26<00:20, 8448.96it/s] 56%|█████▌    | 224322/400000 [00:26<00:20, 8563.04it/s] 56%|█████▋    | 225237/400000 [00:26<00:20, 8730.74it/s] 57%|█████▋    | 226156/400000 [00:26<00:19, 8862.86it/s] 57%|█████▋    | 227046/400000 [00:26<00:19, 8722.22it/s] 57%|█████▋    | 227946/400000 [00:26<00:19, 8800.88it/s] 57%|█████▋    | 228849/400000 [00:26<00:19, 8867.46it/s] 57%|█████▋    | 229769/400000 [00:26<00:18, 8964.37it/s] 58%|█████▊    | 230691/400000 [00:26<00:18, 9037.98it/s] 58%|█████▊    | 231596/400000 [00:26<00:18, 8874.31it/s] 58%|█████▊    | 232485/400000 [00:27<00:18, 8858.21it/s] 58%|█████▊    | 233421/400000 [00:27<00:18, 9001.31it/s] 59%|█████▊    | 234323/400000 [00:27<00:18, 8946.70it/s] 59%|█████▉    | 235219/400000 [00:27<00:18, 8710.01it/s] 59%|█████▉    | 236093/400000 [00:27<00:19, 8459.79it/s] 59%|█████▉    | 236992/400000 [00:27<00:18, 8610.88it/s] 59%|█████▉    | 237907/400000 [00:27<00:18, 8764.80it/s] 60%|█████▉    | 238808/400000 [00:27<00:18, 8835.19it/s] 60%|█████▉    | 239731/400000 [00:27<00:17, 8949.92it/s] 60%|██████    | 240628/400000 [00:27<00:18, 8673.48it/s] 60%|██████    | 241499/400000 [00:28<00:18, 8352.47it/s] 61%|██████    | 242392/400000 [00:28<00:18, 8516.63it/s] 61%|██████    | 243263/400000 [00:28<00:18, 8572.39it/s] 61%|██████    | 244174/400000 [00:28<00:17, 8725.07it/s] 61%|██████▏   | 245050/400000 [00:28<00:18, 8572.86it/s] 61%|██████▏   | 245924/400000 [00:28<00:17, 8618.98it/s] 62%|██████▏   | 246844/400000 [00:28<00:17, 8784.02it/s] 62%|██████▏   | 247759/400000 [00:28<00:17, 8889.88it/s] 62%|██████▏   | 248660/400000 [00:28<00:16, 8923.23it/s] 62%|██████▏   | 249554/400000 [00:28<00:17, 8627.07it/s] 63%|██████▎   | 250420/400000 [00:29<00:17, 8512.43it/s] 63%|██████▎   | 251334/400000 [00:29<00:17, 8689.64it/s] 63%|██████▎   | 252281/400000 [00:29<00:16, 8908.60it/s] 63%|██████▎   | 253201/400000 [00:29<00:16, 8993.65it/s] 64%|██████▎   | 254103/400000 [00:29<00:16, 8806.05it/s] 64%|██████▍   | 255013/400000 [00:29<00:16, 8890.44it/s] 64%|██████▍   | 255905/400000 [00:29<00:16, 8809.47it/s] 64%|██████▍   | 256788/400000 [00:29<00:16, 8508.51it/s] 64%|██████▍   | 257643/400000 [00:29<00:16, 8491.85it/s] 65%|██████▍   | 258495/400000 [00:30<00:16, 8372.60it/s] 65%|██████▍   | 259335/400000 [00:30<00:16, 8318.07it/s] 65%|██████▌   | 260234/400000 [00:30<00:16, 8508.68it/s] 65%|██████▌   | 261160/400000 [00:30<00:15, 8719.97it/s] 66%|██████▌   | 262042/400000 [00:30<00:15, 8749.61it/s] 66%|██████▌   | 262919/400000 [00:30<00:15, 8608.95it/s] 66%|██████▌   | 263782/400000 [00:30<00:16, 8218.14it/s] 66%|██████▌   | 264609/400000 [00:30<00:17, 7761.86it/s] 66%|██████▋   | 265394/400000 [00:30<00:17, 7610.20it/s] 67%|██████▋   | 266162/400000 [00:30<00:17, 7612.33it/s] 67%|██████▋   | 266928/400000 [00:31<00:18, 7258.41it/s] 67%|██████▋   | 267704/400000 [00:31<00:17, 7400.76it/s] 67%|██████▋   | 268478/400000 [00:31<00:17, 7497.18it/s] 67%|██████▋   | 269358/400000 [00:31<00:16, 7843.38it/s] 68%|██████▊   | 270150/400000 [00:31<00:16, 7791.60it/s] 68%|██████▊   | 270952/400000 [00:31<00:16, 7857.92it/s] 68%|██████▊   | 271742/400000 [00:31<00:16, 7840.21it/s] 68%|██████▊   | 272663/400000 [00:31<00:15, 8206.13it/s] 68%|██████▊   | 273552/400000 [00:31<00:15, 8397.70it/s] 69%|██████▊   | 274410/400000 [00:32<00:14, 8450.86it/s] 69%|██████▉   | 275260/400000 [00:32<00:14, 8387.52it/s] 69%|██████▉   | 276164/400000 [00:32<00:14, 8571.13it/s] 69%|██████▉   | 277025/400000 [00:32<00:14, 8508.65it/s] 69%|██████▉   | 277879/400000 [00:32<00:14, 8448.30it/s] 70%|██████▉   | 278726/400000 [00:32<00:14, 8400.56it/s] 70%|██████▉   | 279622/400000 [00:32<00:14, 8559.53it/s] 70%|███████   | 280480/400000 [00:32<00:14, 8482.99it/s] 70%|███████   | 281330/400000 [00:32<00:14, 8188.74it/s] 71%|███████   | 282152/400000 [00:32<00:14, 8005.50it/s] 71%|███████   | 282956/400000 [00:33<00:14, 7862.28it/s] 71%|███████   | 283745/400000 [00:33<00:15, 7746.69it/s] 71%|███████   | 284528/400000 [00:33<00:14, 7770.53it/s] 71%|███████▏  | 285311/400000 [00:33<00:14, 7786.41it/s] 72%|███████▏  | 286091/400000 [00:33<00:14, 7674.22it/s] 72%|███████▏  | 286860/400000 [00:33<00:14, 7590.57it/s] 72%|███████▏  | 287624/400000 [00:33<00:14, 7604.69it/s] 72%|███████▏  | 288389/400000 [00:33<00:14, 7617.80it/s] 72%|███████▏  | 289224/400000 [00:33<00:14, 7821.03it/s] 73%|███████▎  | 290094/400000 [00:33<00:13, 8064.99it/s] 73%|███████▎  | 290904/400000 [00:34<00:13, 8018.37it/s] 73%|███████▎  | 291738/400000 [00:34<00:13, 8111.92it/s] 73%|███████▎  | 292621/400000 [00:34<00:12, 8313.61it/s] 73%|███████▎  | 293455/400000 [00:34<00:12, 8304.85it/s] 74%|███████▎  | 294345/400000 [00:34<00:12, 8470.28it/s] 74%|███████▍  | 295195/400000 [00:34<00:12, 8414.90it/s] 74%|███████▍  | 296039/400000 [00:34<00:12, 8402.66it/s] 74%|███████▍  | 296940/400000 [00:34<00:12, 8572.53it/s] 74%|███████▍  | 297821/400000 [00:34<00:11, 8640.12it/s] 75%|███████▍  | 298687/400000 [00:34<00:11, 8571.54it/s] 75%|███████▍  | 299546/400000 [00:35<00:11, 8445.77it/s] 75%|███████▌  | 300455/400000 [00:35<00:11, 8627.53it/s] 75%|███████▌  | 301343/400000 [00:35<00:11, 8701.63it/s] 76%|███████▌  | 302241/400000 [00:35<00:11, 8782.48it/s] 76%|███████▌  | 303172/400000 [00:35<00:10, 8931.89it/s] 76%|███████▌  | 304067/400000 [00:35<00:10, 8844.69it/s] 76%|███████▌  | 304972/400000 [00:35<00:10, 8903.89it/s] 76%|███████▋  | 305864/400000 [00:35<00:10, 8733.27it/s] 77%|███████▋  | 306739/400000 [00:35<00:10, 8676.14it/s] 77%|███████▋  | 307608/400000 [00:35<00:10, 8466.48it/s] 77%|███████▋  | 308457/400000 [00:36<00:10, 8432.58it/s] 77%|███████▋  | 309359/400000 [00:36<00:10, 8598.84it/s] 78%|███████▊  | 310270/400000 [00:36<00:10, 8744.46it/s] 78%|███████▊  | 311163/400000 [00:36<00:10, 8798.70it/s] 78%|███████▊  | 312108/400000 [00:36<00:09, 8982.69it/s] 78%|███████▊  | 313009/400000 [00:36<00:09, 8775.99it/s] 78%|███████▊  | 313938/400000 [00:36<00:09, 8921.80it/s] 79%|███████▊  | 314844/400000 [00:36<00:09, 8961.25it/s] 79%|███████▉  | 315742/400000 [00:36<00:09, 8877.89it/s] 79%|███████▉  | 316679/400000 [00:37<00:09, 9019.73it/s] 79%|███████▉  | 317583/400000 [00:37<00:09, 8763.90it/s] 80%|███████▉  | 318463/400000 [00:37<00:09, 8725.37it/s] 80%|███████▉  | 319354/400000 [00:37<00:09, 8778.46it/s] 80%|████████  | 320234/400000 [00:37<00:09, 8363.57it/s] 80%|████████  | 321108/400000 [00:37<00:09, 8472.42it/s] 80%|████████  | 321960/400000 [00:37<00:09, 8472.40it/s] 81%|████████  | 322891/400000 [00:37<00:08, 8705.91it/s] 81%|████████  | 323802/400000 [00:37<00:08, 8822.51it/s] 81%|████████  | 324688/400000 [00:37<00:08, 8781.32it/s] 81%|████████▏ | 325569/400000 [00:38<00:08, 8736.25it/s] 82%|████████▏ | 326445/400000 [00:38<00:08, 8562.24it/s] 82%|████████▏ | 327317/400000 [00:38<00:08, 8606.15it/s] 82%|████████▏ | 328210/400000 [00:38<00:08, 8698.04it/s] 82%|████████▏ | 329081/400000 [00:38<00:08, 8669.18it/s] 82%|████████▏ | 329983/400000 [00:38<00:07, 8770.25it/s] 83%|████████▎ | 330861/400000 [00:38<00:07, 8707.25it/s] 83%|████████▎ | 331733/400000 [00:38<00:07, 8582.83it/s] 83%|████████▎ | 332622/400000 [00:38<00:07, 8671.17it/s] 83%|████████▎ | 333490/400000 [00:38<00:07, 8649.53it/s] 84%|████████▎ | 334356/400000 [00:39<00:07, 8641.65it/s] 84%|████████▍ | 335221/400000 [00:39<00:07, 8371.23it/s] 84%|████████▍ | 336149/400000 [00:39<00:07, 8621.95it/s] 84%|████████▍ | 337072/400000 [00:39<00:07, 8794.78it/s] 84%|████████▍ | 337987/400000 [00:39<00:06, 8898.08it/s] 85%|████████▍ | 338902/400000 [00:39<00:06, 8969.36it/s] 85%|████████▍ | 339801/400000 [00:39<00:06, 8912.46it/s] 85%|████████▌ | 340716/400000 [00:39<00:06, 8982.26it/s] 85%|████████▌ | 341616/400000 [00:39<00:06, 8883.07it/s] 86%|████████▌ | 342506/400000 [00:39<00:06, 8788.66it/s] 86%|████████▌ | 343386/400000 [00:40<00:06, 8734.73it/s] 86%|████████▌ | 344261/400000 [00:40<00:06, 8652.89it/s] 86%|████████▋ | 345197/400000 [00:40<00:06, 8852.83it/s] 87%|████████▋ | 346084/400000 [00:40<00:06, 8820.22it/s] 87%|████████▋ | 346968/400000 [00:40<00:06, 8596.05it/s] 87%|████████▋ | 347858/400000 [00:40<00:06, 8684.84it/s] 87%|████████▋ | 348729/400000 [00:40<00:05, 8608.52it/s] 87%|████████▋ | 349592/400000 [00:40<00:05, 8452.58it/s] 88%|████████▊ | 350439/400000 [00:40<00:05, 8323.41it/s] 88%|████████▊ | 351338/400000 [00:41<00:05, 8512.60it/s] 88%|████████▊ | 352224/400000 [00:41<00:05, 8612.41it/s] 88%|████████▊ | 353088/400000 [00:41<00:05, 8554.62it/s] 88%|████████▊ | 353947/400000 [00:41<00:05, 8561.81it/s] 89%|████████▊ | 354834/400000 [00:41<00:05, 8649.88it/s] 89%|████████▉ | 355705/400000 [00:41<00:05, 8666.94it/s] 89%|████████▉ | 356573/400000 [00:41<00:05, 8513.85it/s] 89%|████████▉ | 357440/400000 [00:41<00:04, 8557.46it/s] 90%|████████▉ | 358297/400000 [00:41<00:04, 8545.65it/s] 90%|████████▉ | 359196/400000 [00:41<00:04, 8671.79it/s] 90%|█████████ | 360144/400000 [00:42<00:04, 8897.42it/s] 90%|█████████ | 361058/400000 [00:42<00:04, 8967.76it/s] 90%|█████████ | 361957/400000 [00:42<00:04, 8678.69it/s] 91%|█████████ | 362838/400000 [00:42<00:04, 8715.04it/s] 91%|█████████ | 363712/400000 [00:42<00:04, 8516.50it/s] 91%|█████████ | 364589/400000 [00:42<00:04, 8588.74it/s] 91%|█████████▏| 365515/400000 [00:42<00:03, 8778.26it/s] 92%|█████████▏| 366397/400000 [00:42<00:03, 8789.72it/s] 92%|█████████▏| 367295/400000 [00:42<00:03, 8845.56it/s] 92%|█████████▏| 368197/400000 [00:42<00:03, 8894.96it/s] 92%|█████████▏| 369088/400000 [00:43<00:03, 8639.30it/s] 92%|█████████▏| 369955/400000 [00:43<00:03, 8284.80it/s] 93%|█████████▎| 370789/400000 [00:43<00:03, 7865.50it/s] 93%|█████████▎| 371584/400000 [00:43<00:03, 7794.52it/s] 93%|█████████▎| 372369/400000 [00:43<00:03, 7779.99it/s] 93%|█████████▎| 373185/400000 [00:43<00:03, 7888.42it/s] 94%|█████████▎| 374064/400000 [00:43<00:03, 8136.87it/s] 94%|█████████▎| 374889/400000 [00:43<00:03, 8168.11it/s] 94%|█████████▍| 375733/400000 [00:43<00:02, 8245.53it/s] 94%|█████████▍| 376646/400000 [00:43<00:02, 8491.46it/s] 94%|█████████▍| 377499/400000 [00:44<00:02, 8455.82it/s] 95%|█████████▍| 378347/400000 [00:44<00:02, 8264.99it/s] 95%|█████████▍| 379269/400000 [00:44<00:02, 8529.53it/s] 95%|█████████▌| 380184/400000 [00:44<00:02, 8704.92it/s] 95%|█████████▌| 381093/400000 [00:44<00:02, 8816.02it/s] 96%|█████████▌| 382040/400000 [00:44<00:01, 9000.93it/s] 96%|█████████▌| 382944/400000 [00:44<00:01, 8824.18it/s] 96%|█████████▌| 383830/400000 [00:44<00:01, 8813.23it/s] 96%|█████████▌| 384714/400000 [00:44<00:01, 8759.96it/s] 96%|█████████▋| 385592/400000 [00:45<00:01, 8560.99it/s] 97%|█████████▋| 386508/400000 [00:45<00:01, 8729.99it/s] 97%|█████████▋| 387384/400000 [00:45<00:01, 8733.93it/s] 97%|█████████▋| 388281/400000 [00:45<00:01, 8802.56it/s] 97%|█████████▋| 389163/400000 [00:45<00:01, 8651.66it/s] 98%|█████████▊| 390132/400000 [00:45<00:01, 8938.04it/s] 98%|█████████▊| 391030/400000 [00:45<00:01, 8699.53it/s] 98%|█████████▊| 391904/400000 [00:45<00:00, 8437.35it/s] 98%|█████████▊| 392753/400000 [00:45<00:00, 8411.86it/s] 98%|█████████▊| 393598/400000 [00:45<00:00, 8328.22it/s] 99%|█████████▊| 394434/400000 [00:46<00:00, 8197.54it/s] 99%|█████████▉| 395367/400000 [00:46<00:00, 8506.06it/s] 99%|█████████▉| 396223/400000 [00:46<00:00, 8469.99it/s] 99%|█████████▉| 397097/400000 [00:46<00:00, 8548.69it/s]100%|█████████▉| 398007/400000 [00:46<00:00, 8705.11it/s]100%|█████████▉| 398880/400000 [00:46<00:00, 8361.95it/s]100%|█████████▉| 399721/400000 [00:46<00:00, 8266.18it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8562.82it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f0b60092d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011213240364249342 	 Accuracy: 47
Train Epoch: 1 	 Loss: 0.010974289780875113 	 Accuracy: 68

  model saves at 68% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15741 out of table with 15736 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15741 out of table with 15736 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-13 04:25:44.404887: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 04:25:44.408538: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 04:25:44.408667: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5559e175a600 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 04:25:44.408682: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f0b05b40240> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5900 - accuracy: 0.5050
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7203 - accuracy: 0.4965 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8046 - accuracy: 0.4910
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7740 - accuracy: 0.4930
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5992 - accuracy: 0.5044
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6257 - accuracy: 0.5027
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6294 - accuracy: 0.5024
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5880 - accuracy: 0.5051
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6019 - accuracy: 0.5042
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
11000/25000 [============>.................] - ETA: 4s - loss: 7.6318 - accuracy: 0.5023
12000/25000 [=============>................] - ETA: 4s - loss: 7.6104 - accuracy: 0.5037
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5959 - accuracy: 0.5046
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5823 - accuracy: 0.5055
15000/25000 [=================>............] - ETA: 3s - loss: 7.6186 - accuracy: 0.5031
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6235 - accuracy: 0.5028
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6477 - accuracy: 0.5012
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6521 - accuracy: 0.5009
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6448 - accuracy: 0.5014
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6467 - accuracy: 0.5013
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6564 - accuracy: 0.5007
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6645 - accuracy: 0.5001
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6577 - accuracy: 0.5006
25000/25000 [==============================] - 9s 370us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f0ae7fb31d0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f0acdc1b128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 938ms/step - loss: 1.2734 - crf_viterbi_accuracy: 0.6533 - val_loss: 1.2150 - val_crf_viterbi_accuracy: 0.6800

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

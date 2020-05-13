
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f033df81f60> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 05:13:55.638381
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 05:13:55.643122
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 05:13:55.647878
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 05:13:55.651388
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f0349d4b3c8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 353761.6250
Epoch 2/10

1/1 [==============================] - 0s 97ms/step - loss: 249901.7188
Epoch 3/10

1/1 [==============================] - 0s 89ms/step - loss: 145827.5938
Epoch 4/10

1/1 [==============================] - 0s 85ms/step - loss: 76529.2969
Epoch 5/10

1/1 [==============================] - 0s 87ms/step - loss: 40935.2773
Epoch 6/10

1/1 [==============================] - 0s 95ms/step - loss: 23742.5625
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 15010.9297
Epoch 8/10

1/1 [==============================] - 0s 88ms/step - loss: 10237.4561
Epoch 9/10

1/1 [==============================] - 0s 90ms/step - loss: 7446.5659
Epoch 10/10

1/1 [==============================] - 0s 91ms/step - loss: 5730.9731

  #### Inference Need return ypred, ytrue ######################### 
[[ 9.45304260e-02  7.86568165e+00  8.59567928e+00  7.43558598e+00
   8.99834347e+00  9.89210892e+00  7.23105955e+00  9.43131256e+00
   8.92784786e+00  7.08216476e+00  6.60688448e+00  8.49077320e+00
   8.52362347e+00  8.95988178e+00  9.60707188e+00  9.51582050e+00
   9.07379532e+00  1.07385740e+01  8.12332535e+00  8.21661758e+00
   8.96642017e+00  8.85429955e+00  9.79011154e+00  7.91727400e+00
   7.58216333e+00  9.45407295e+00  1.00489359e+01  8.13682652e+00
   8.04328251e+00  8.93645763e+00  8.06994820e+00  9.79456902e+00
   9.97188854e+00  8.16603088e+00  8.62302208e+00  8.09981632e+00
   8.91547680e+00  1.02585411e+01  7.34607887e+00  9.41759872e+00
   7.90026569e+00  8.94160557e+00  8.10360718e+00  9.03909969e+00
   8.51168060e+00  9.48684692e+00  8.34066963e+00  7.85896254e+00
   8.14478111e+00  9.50773048e+00  1.10178518e+01  8.35550976e+00
   6.91499090e+00  9.92616081e+00  8.16281033e+00  8.42727184e+00
   8.43225384e+00  7.82103682e+00  7.63028622e+00  9.49605846e+00
   5.13226211e-01 -1.08772326e+00 -3.36374193e-01  7.28487432e-01
  -2.36028004e+00  8.27146292e-01 -3.11203152e-01 -9.61947441e-03
  -3.61612439e-01 -3.25336814e-01 -1.34486401e+00 -5.69064140e-01
  -1.44083619e-01  3.16250980e-01 -8.80458415e-01 -6.49673581e-01
   8.08166683e-01  8.93972516e-01 -1.64372063e+00  9.43577766e-01
  -1.26746476e-01 -6.33342385e-01 -4.60678786e-02  3.68339121e-01
   7.83066869e-01 -1.86019540e-02  3.12046200e-01  1.31617403e+00
  -2.76236475e-01  1.57566810e+00 -7.64296532e-01  1.70060605e-01
  -1.18951750e+00  9.52038229e-01 -7.58346975e-01 -6.55342817e-01
   1.20028079e-01  1.38235998e+00 -6.86933994e-02  3.08823228e-01
   8.07936072e-01  2.08765459e+00  7.98219740e-01 -1.57135892e+00
  -2.28028089e-01 -9.02633250e-01  4.81978536e-01 -1.61923802e+00
  -6.06231570e-01 -1.24136877e+00  1.48636878e-01 -1.08742309e+00
  -1.19709587e+00 -7.96500742e-01  6.15645766e-01 -1.03785324e+00
   8.10519695e-01 -4.71065342e-01 -2.71133661e-01 -1.03032565e+00
   7.69925237e-01 -4.98228550e-01 -9.76690292e-01 -7.04143107e-01
   4.46205020e-01 -9.24806952e-01  1.07932496e+00  6.80148840e-01
  -1.21518326e+00 -9.85440254e-01  1.85858345e+00 -7.19981074e-01
   3.40492517e-01  1.34162164e+00 -1.37392223e+00  1.24435805e-01
   4.44133937e-01 -8.21457088e-01 -2.32348251e+00 -4.58699167e-01
  -8.36265564e-01  1.83026552e+00 -1.14177108e+00 -8.94545913e-02
  -1.13795102e-01 -9.43392515e-04 -1.45828831e+00  8.27036738e-01
  -1.70586705e+00 -6.47575617e-01  1.67140436e+00  6.98946834e-01
  -2.10419297e-01 -7.59647787e-02  1.34408593e+00  1.47986531e-01
  -6.86380506e-01 -9.20545399e-01  4.59863305e-01  1.06211579e+00
  -1.34075356e+00 -1.05088699e+00 -5.32880008e-01 -2.51742661e-01
  -1.73756897e-01  9.15633082e-01  2.80884832e-01 -1.47329271e+00
  -6.37872458e-01  1.62620723e-01 -1.44593859e+00 -1.55333877e-01
   4.90472555e-01 -1.91415739e+00 -5.35715640e-01 -7.15963304e-01
   1.07545209e+00  4.34948713e-01  2.19142258e-01  8.15179586e-01
   1.57788634e-01  8.88107300e+00  8.16092873e+00  8.76061440e+00
   9.69111156e+00  1.01865015e+01  9.83573532e+00  9.13088894e+00
   8.50814438e+00  7.93385935e+00  9.95423985e+00  8.33016396e+00
   9.34897900e+00  9.11033916e+00  8.30249691e+00  1.13213034e+01
   8.40931511e+00  7.79929495e+00  8.68008900e+00  1.03613424e+01
   8.97983170e+00  7.04337692e+00  8.42349052e+00  8.70899105e+00
   8.52842712e+00  9.94232464e+00  1.01358385e+01  8.76455307e+00
   9.43923092e+00  9.87103939e+00  9.54304028e+00  7.66228771e+00
   9.44685936e+00  8.36259365e+00  9.65060902e+00  7.26168489e+00
   8.37098694e+00  1.02609816e+01  8.89459896e+00  9.33566856e+00
   9.21405220e+00  9.41438293e+00  9.39206219e+00  1.03843908e+01
   8.37752247e+00  9.45302105e+00  8.25999451e+00  8.59946442e+00
   8.46331882e+00  8.79514980e+00  1.01557617e+01  8.27631760e+00
   8.44959927e+00  9.62454987e+00  8.69480419e+00  8.88176727e+00
   9.61333370e+00  9.46060467e+00  9.47812080e+00  9.50554180e+00
   2.15590572e+00  2.28413391e+00  4.53488469e-01  4.16711330e-01
   1.88031781e+00  1.09206939e+00  1.18821502e+00  7.33493924e-01
   9.83701944e-01  4.43206549e-01  2.09557915e+00  6.60643756e-01
   2.07406807e+00  8.14052641e-01  1.93992376e-01  3.40666914e+00
   6.09979749e-01  3.17672372e-01  6.40832067e-01  1.04211390e+00
   7.51639009e-01  1.84619558e+00  5.40999472e-01  1.07723951e+00
   1.49336886e+00  4.07085955e-01  8.54641020e-01  7.41999149e-01
   1.72687829e+00  7.22929657e-01  9.83191669e-01  9.12289023e-01
   2.12266231e+00  9.36738849e-01  2.10479116e+00  1.48781002e-01
   1.96120203e-01  2.46302176e+00  6.88923657e-01  2.40920019e+00
   1.63679957e-01  1.27332509e+00  3.63778234e-01  2.93999100e+00
   1.07503498e+00  1.98256803e+00  8.22521091e-01  1.09363294e+00
   7.94013202e-01  2.83977699e+00  1.47523832e+00  1.83157206e-01
   9.90501046e-02  2.41129637e+00  9.39179838e-01  1.09063673e+00
   2.50442982e+00  1.19115913e+00  2.10414648e+00  1.86667502e+00
   9.54172015e-01  1.87232804e+00  1.69523644e+00  3.21487546e-01
   1.57583678e+00  1.01764309e+00  2.87210703e+00  1.97999203e+00
   2.02972031e+00  1.50344491e-01  5.96731722e-01  1.91185927e+00
   1.57006335e+00  3.50843549e-01  8.23448777e-01  2.35285330e+00
   1.74715900e+00  4.32399988e-01  7.06093431e-01  9.14215684e-01
   5.16397953e-01  6.92377388e-01  1.66834390e+00  2.44662762e+00
   1.09458566e+00  1.34881806e+00  5.04143476e-01  8.91349792e-01
   2.41131401e+00  4.97261882e-01  1.71317673e+00  1.68783665e-01
   3.37922573e+00  1.90614676e+00  7.83765376e-01  5.97241461e-01
   4.51578081e-01  1.08386397e+00  2.58364129e+00  6.53609514e-01
   9.63426769e-01  2.09964037e+00  1.58158016e+00  9.93876398e-01
   1.12885666e+00  1.67142200e+00  3.05318952e-01  4.87440944e-01
   1.87785447e-01  3.76996160e-01  1.63115978e+00  1.47768247e+00
   7.29462266e-01  5.79651117e-01  1.01395822e+00  4.25574780e-01
   7.84470558e-01  2.79981017e-01  7.88288295e-01  2.47586012e-01
   1.17579918e+01 -3.77991104e+00 -1.48764920e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 05:14:05.232627
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    94.313
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 05:14:05.236301
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8919.45
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 05:14:05.239043
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    94.274
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 05:14:05.241469
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    -797.8
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139651542205776
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139650600837696
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139650600838200
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139650600838704
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139650600839208
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139650600839712

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f0345bdff98> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.450017
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.419636
grad_step = 000002, loss = 0.394893
grad_step = 000003, loss = 0.372431
grad_step = 000004, loss = 0.349015
grad_step = 000005, loss = 0.326351
grad_step = 000006, loss = 0.312296
grad_step = 000007, loss = 0.302779
grad_step = 000008, loss = 0.292995
grad_step = 000009, loss = 0.278861
grad_step = 000010, loss = 0.264555
grad_step = 000011, loss = 0.254547
grad_step = 000012, loss = 0.247082
grad_step = 000013, loss = 0.239112
grad_step = 000014, loss = 0.229542
grad_step = 000015, loss = 0.218698
grad_step = 000016, loss = 0.207730
grad_step = 000017, loss = 0.198071
grad_step = 000018, loss = 0.190181
grad_step = 000019, loss = 0.182635
grad_step = 000020, loss = 0.173869
grad_step = 000021, loss = 0.164431
grad_step = 000022, loss = 0.155878
grad_step = 000023, loss = 0.148484
grad_step = 000024, loss = 0.141464
grad_step = 000025, loss = 0.134171
grad_step = 000026, loss = 0.126628
grad_step = 000027, loss = 0.119346
grad_step = 000028, loss = 0.112733
grad_step = 000029, loss = 0.106554
grad_step = 000030, loss = 0.100226
grad_step = 000031, loss = 0.093803
grad_step = 000032, loss = 0.087898
grad_step = 000033, loss = 0.082610
grad_step = 000034, loss = 0.077501
grad_step = 000035, loss = 0.072362
grad_step = 000036, loss = 0.067355
grad_step = 000037, loss = 0.062703
grad_step = 000038, loss = 0.058467
grad_step = 000039, loss = 0.054423
grad_step = 000040, loss = 0.050408
grad_step = 000041, loss = 0.046622
grad_step = 000042, loss = 0.043194
grad_step = 000043, loss = 0.039993
grad_step = 000044, loss = 0.036910
grad_step = 000045, loss = 0.033999
grad_step = 000046, loss = 0.031346
grad_step = 000047, loss = 0.028873
grad_step = 000048, loss = 0.026510
grad_step = 000049, loss = 0.024335
grad_step = 000050, loss = 0.022374
grad_step = 000051, loss = 0.020569
grad_step = 000052, loss = 0.018855
grad_step = 000053, loss = 0.017265
grad_step = 000054, loss = 0.015861
grad_step = 000055, loss = 0.014567
grad_step = 000056, loss = 0.013334
grad_step = 000057, loss = 0.012233
grad_step = 000058, loss = 0.011266
grad_step = 000059, loss = 0.010378
grad_step = 000060, loss = 0.009538
grad_step = 000061, loss = 0.008779
grad_step = 000062, loss = 0.008116
grad_step = 000063, loss = 0.007508
grad_step = 000064, loss = 0.006945
grad_step = 000065, loss = 0.006449
grad_step = 000066, loss = 0.006004
grad_step = 000067, loss = 0.005590
grad_step = 000068, loss = 0.005211
grad_step = 000069, loss = 0.004879
grad_step = 000070, loss = 0.004582
grad_step = 000071, loss = 0.004307
grad_step = 000072, loss = 0.004063
grad_step = 000073, loss = 0.003845
grad_step = 000074, loss = 0.003646
grad_step = 000075, loss = 0.003469
grad_step = 000076, loss = 0.003312
grad_step = 000077, loss = 0.003170
grad_step = 000078, loss = 0.003043
grad_step = 000079, loss = 0.002933
grad_step = 000080, loss = 0.002836
grad_step = 000081, loss = 0.002745
grad_step = 000082, loss = 0.002669
grad_step = 000083, loss = 0.002602
grad_step = 000084, loss = 0.002543
grad_step = 000085, loss = 0.002491
grad_step = 000086, loss = 0.002446
grad_step = 000087, loss = 0.002406
grad_step = 000088, loss = 0.002371
grad_step = 000089, loss = 0.002342
grad_step = 000090, loss = 0.002315
grad_step = 000091, loss = 0.002292
grad_step = 000092, loss = 0.002271
grad_step = 000093, loss = 0.002252
grad_step = 000094, loss = 0.002235
grad_step = 000095, loss = 0.002219
grad_step = 000096, loss = 0.002204
grad_step = 000097, loss = 0.002190
grad_step = 000098, loss = 0.002177
grad_step = 000099, loss = 0.002165
grad_step = 000100, loss = 0.002152
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002140
grad_step = 000102, loss = 0.002128
grad_step = 000103, loss = 0.002117
grad_step = 000104, loss = 0.002106
grad_step = 000105, loss = 0.002095
grad_step = 000106, loss = 0.002084
grad_step = 000107, loss = 0.002074
grad_step = 000108, loss = 0.002064
grad_step = 000109, loss = 0.002053
grad_step = 000110, loss = 0.002044
grad_step = 000111, loss = 0.002034
grad_step = 000112, loss = 0.002025
grad_step = 000113, loss = 0.002016
grad_step = 000114, loss = 0.002007
grad_step = 000115, loss = 0.001999
grad_step = 000116, loss = 0.001990
grad_step = 000117, loss = 0.001982
grad_step = 000118, loss = 0.001974
grad_step = 000119, loss = 0.001965
grad_step = 000120, loss = 0.001958
grad_step = 000121, loss = 0.001950
grad_step = 000122, loss = 0.001942
grad_step = 000123, loss = 0.001934
grad_step = 000124, loss = 0.001926
grad_step = 000125, loss = 0.001919
grad_step = 000126, loss = 0.001911
grad_step = 000127, loss = 0.001904
grad_step = 000128, loss = 0.001898
grad_step = 000129, loss = 0.001891
grad_step = 000130, loss = 0.001883
grad_step = 000131, loss = 0.001875
grad_step = 000132, loss = 0.001867
grad_step = 000133, loss = 0.001860
grad_step = 000134, loss = 0.001854
grad_step = 000135, loss = 0.001847
grad_step = 000136, loss = 0.001839
grad_step = 000137, loss = 0.001833
grad_step = 000138, loss = 0.001828
grad_step = 000139, loss = 0.001822
grad_step = 000140, loss = 0.001814
grad_step = 000141, loss = 0.001808
grad_step = 000142, loss = 0.001804
grad_step = 000143, loss = 0.001796
grad_step = 000144, loss = 0.001786
grad_step = 000145, loss = 0.001780
grad_step = 000146, loss = 0.001776
grad_step = 000147, loss = 0.001769
grad_step = 000148, loss = 0.001761
grad_step = 000149, loss = 0.001755
grad_step = 000150, loss = 0.001751
grad_step = 000151, loss = 0.001748
grad_step = 000152, loss = 0.001750
grad_step = 000153, loss = 0.001767
grad_step = 000154, loss = 0.001836
grad_step = 000155, loss = 0.001940
grad_step = 000156, loss = 0.001999
grad_step = 000157, loss = 0.001847
grad_step = 000158, loss = 0.001743
grad_step = 000159, loss = 0.001798
grad_step = 000160, loss = 0.001867
grad_step = 000161, loss = 0.001793
grad_step = 000162, loss = 0.001704
grad_step = 000163, loss = 0.001774
grad_step = 000164, loss = 0.001797
grad_step = 000165, loss = 0.001741
grad_step = 000166, loss = 0.001702
grad_step = 000167, loss = 0.001725
grad_step = 000168, loss = 0.001755
grad_step = 000169, loss = 0.001732
grad_step = 000170, loss = 0.001678
grad_step = 000171, loss = 0.001707
grad_step = 000172, loss = 0.001725
grad_step = 000173, loss = 0.001706
grad_step = 000174, loss = 0.001681
grad_step = 000175, loss = 0.001677
grad_step = 000176, loss = 0.001684
grad_step = 000177, loss = 0.001700
grad_step = 000178, loss = 0.001675
grad_step = 000179, loss = 0.001655
grad_step = 000180, loss = 0.001662
grad_step = 000181, loss = 0.001670
grad_step = 000182, loss = 0.001661
grad_step = 000183, loss = 0.001657
grad_step = 000184, loss = 0.001643
grad_step = 000185, loss = 0.001638
grad_step = 000186, loss = 0.001646
grad_step = 000187, loss = 0.001646
grad_step = 000188, loss = 0.001635
grad_step = 000189, loss = 0.001630
grad_step = 000190, loss = 0.001625
grad_step = 000191, loss = 0.001619
grad_step = 000192, loss = 0.001622
grad_step = 000193, loss = 0.001623
grad_step = 000194, loss = 0.001618
grad_step = 000195, loss = 0.001612
grad_step = 000196, loss = 0.001610
grad_step = 000197, loss = 0.001603
grad_step = 000198, loss = 0.001598
grad_step = 000199, loss = 0.001595
grad_step = 000200, loss = 0.001594
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001592
grad_step = 000202, loss = 0.001590
grad_step = 000203, loss = 0.001591
grad_step = 000204, loss = 0.001594
grad_step = 000205, loss = 0.001600
grad_step = 000206, loss = 0.001612
grad_step = 000207, loss = 0.001632
grad_step = 000208, loss = 0.001659
grad_step = 000209, loss = 0.001693
grad_step = 000210, loss = 0.001704
grad_step = 000211, loss = 0.001689
grad_step = 000212, loss = 0.001637
grad_step = 000213, loss = 0.001581
grad_step = 000214, loss = 0.001553
grad_step = 000215, loss = 0.001563
grad_step = 000216, loss = 0.001593
grad_step = 000217, loss = 0.001619
grad_step = 000218, loss = 0.001618
grad_step = 000219, loss = 0.001600
grad_step = 000220, loss = 0.001567
grad_step = 000221, loss = 0.001541
grad_step = 000222, loss = 0.001531
grad_step = 000223, loss = 0.001535
grad_step = 000224, loss = 0.001546
grad_step = 000225, loss = 0.001559
grad_step = 000226, loss = 0.001568
grad_step = 000227, loss = 0.001572
grad_step = 000228, loss = 0.001566
grad_step = 000229, loss = 0.001555
grad_step = 000230, loss = 0.001542
grad_step = 000231, loss = 0.001528
grad_step = 000232, loss = 0.001519
grad_step = 000233, loss = 0.001510
grad_step = 000234, loss = 0.001504
grad_step = 000235, loss = 0.001502
grad_step = 000236, loss = 0.001504
grad_step = 000237, loss = 0.001510
grad_step = 000238, loss = 0.001518
grad_step = 000239, loss = 0.001534
grad_step = 000240, loss = 0.001552
grad_step = 000241, loss = 0.001582
grad_step = 000242, loss = 0.001594
grad_step = 000243, loss = 0.001608
grad_step = 000244, loss = 0.001610
grad_step = 000245, loss = 0.001595
grad_step = 000246, loss = 0.001561
grad_step = 000247, loss = 0.001505
grad_step = 000248, loss = 0.001482
grad_step = 000249, loss = 0.001503
grad_step = 000250, loss = 0.001529
grad_step = 000251, loss = 0.001529
grad_step = 000252, loss = 0.001508
grad_step = 000253, loss = 0.001493
grad_step = 000254, loss = 0.001496
grad_step = 000255, loss = 0.001494
grad_step = 000256, loss = 0.001486
grad_step = 000257, loss = 0.001475
grad_step = 000258, loss = 0.001476
grad_step = 000259, loss = 0.001486
grad_step = 000260, loss = 0.001488
grad_step = 000261, loss = 0.001480
grad_step = 000262, loss = 0.001466
grad_step = 000263, loss = 0.001464
grad_step = 000264, loss = 0.001469
grad_step = 000265, loss = 0.001469
grad_step = 000266, loss = 0.001461
grad_step = 000267, loss = 0.001455
grad_step = 000268, loss = 0.001457
grad_step = 000269, loss = 0.001461
grad_step = 000270, loss = 0.001459
grad_step = 000271, loss = 0.001454
grad_step = 000272, loss = 0.001450
grad_step = 000273, loss = 0.001449
grad_step = 000274, loss = 0.001453
grad_step = 000275, loss = 0.001459
grad_step = 000276, loss = 0.001472
grad_step = 000277, loss = 0.001489
grad_step = 000278, loss = 0.001532
grad_step = 000279, loss = 0.001579
grad_step = 000280, loss = 0.001677
grad_step = 000281, loss = 0.001811
grad_step = 000282, loss = 0.001860
grad_step = 000283, loss = 0.001758
grad_step = 000284, loss = 0.001533
grad_step = 000285, loss = 0.001457
grad_step = 000286, loss = 0.001550
grad_step = 000287, loss = 0.001613
grad_step = 000288, loss = 0.001569
grad_step = 000289, loss = 0.001525
grad_step = 000290, loss = 0.001471
grad_step = 000291, loss = 0.001476
grad_step = 000292, loss = 0.001541
grad_step = 000293, loss = 0.001517
grad_step = 000294, loss = 0.001443
grad_step = 000295, loss = 0.001441
grad_step = 000296, loss = 0.001483
grad_step = 000297, loss = 0.001468
grad_step = 000298, loss = 0.001444
grad_step = 000299, loss = 0.001451
grad_step = 000300, loss = 0.001439
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001436
grad_step = 000302, loss = 0.001454
grad_step = 000303, loss = 0.001440
grad_step = 000304, loss = 0.001417
grad_step = 000305, loss = 0.001429
grad_step = 000306, loss = 0.001437
grad_step = 000307, loss = 0.001423
grad_step = 000308, loss = 0.001419
grad_step = 000309, loss = 0.001426
grad_step = 000310, loss = 0.001418
grad_step = 000311, loss = 0.001409
grad_step = 000312, loss = 0.001414
grad_step = 000313, loss = 0.001419
grad_step = 000314, loss = 0.001413
grad_step = 000315, loss = 0.001405
grad_step = 000316, loss = 0.001406
grad_step = 000317, loss = 0.001408
grad_step = 000318, loss = 0.001404
grad_step = 000319, loss = 0.001400
grad_step = 000320, loss = 0.001402
grad_step = 000321, loss = 0.001403
grad_step = 000322, loss = 0.001398
grad_step = 000323, loss = 0.001394
grad_step = 000324, loss = 0.001394
grad_step = 000325, loss = 0.001395
grad_step = 000326, loss = 0.001393
grad_step = 000327, loss = 0.001390
grad_step = 000328, loss = 0.001389
grad_step = 000329, loss = 0.001389
grad_step = 000330, loss = 0.001390
grad_step = 000331, loss = 0.001390
grad_step = 000332, loss = 0.001389
grad_step = 000333, loss = 0.001391
grad_step = 000334, loss = 0.001396
grad_step = 000335, loss = 0.001405
grad_step = 000336, loss = 0.001421
grad_step = 000337, loss = 0.001447
grad_step = 000338, loss = 0.001493
grad_step = 000339, loss = 0.001539
grad_step = 000340, loss = 0.001591
grad_step = 000341, loss = 0.001586
grad_step = 000342, loss = 0.001543
grad_step = 000343, loss = 0.001478
grad_step = 000344, loss = 0.001406
grad_step = 000345, loss = 0.001376
grad_step = 000346, loss = 0.001408
grad_step = 000347, loss = 0.001452
grad_step = 000348, loss = 0.001456
grad_step = 000349, loss = 0.001419
grad_step = 000350, loss = 0.001386
grad_step = 000351, loss = 0.001377
grad_step = 000352, loss = 0.001381
grad_step = 000353, loss = 0.001391
grad_step = 000354, loss = 0.001406
grad_step = 000355, loss = 0.001408
grad_step = 000356, loss = 0.001391
grad_step = 000357, loss = 0.001366
grad_step = 000358, loss = 0.001360
grad_step = 000359, loss = 0.001367
grad_step = 000360, loss = 0.001371
grad_step = 000361, loss = 0.001372
grad_step = 000362, loss = 0.001373
grad_step = 000363, loss = 0.001372
grad_step = 000364, loss = 0.001363
grad_step = 000365, loss = 0.001352
grad_step = 000366, loss = 0.001349
grad_step = 000367, loss = 0.001352
grad_step = 000368, loss = 0.001354
grad_step = 000369, loss = 0.001353
grad_step = 000370, loss = 0.001353
grad_step = 000371, loss = 0.001354
grad_step = 000372, loss = 0.001354
grad_step = 000373, loss = 0.001351
grad_step = 000374, loss = 0.001346
grad_step = 000375, loss = 0.001343
grad_step = 000376, loss = 0.001341
grad_step = 000377, loss = 0.001339
grad_step = 000378, loss = 0.001336
grad_step = 000379, loss = 0.001334
grad_step = 000380, loss = 0.001332
grad_step = 000381, loss = 0.001331
grad_step = 000382, loss = 0.001331
grad_step = 000383, loss = 0.001329
grad_step = 000384, loss = 0.001328
grad_step = 000385, loss = 0.001326
grad_step = 000386, loss = 0.001325
grad_step = 000387, loss = 0.001325
grad_step = 000388, loss = 0.001324
grad_step = 000389, loss = 0.001324
grad_step = 000390, loss = 0.001324
grad_step = 000391, loss = 0.001327
grad_step = 000392, loss = 0.001335
grad_step = 000393, loss = 0.001354
grad_step = 000394, loss = 0.001398
grad_step = 000395, loss = 0.001494
grad_step = 000396, loss = 0.001669
grad_step = 000397, loss = 0.001945
grad_step = 000398, loss = 0.002137
grad_step = 000399, loss = 0.002006
grad_step = 000400, loss = 0.001671
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001389
grad_step = 000402, loss = 0.001456
grad_step = 000403, loss = 0.001663
grad_step = 000404, loss = 0.001610
grad_step = 000405, loss = 0.001411
grad_step = 000406, loss = 0.001368
grad_step = 000407, loss = 0.001463
grad_step = 000408, loss = 0.001493
grad_step = 000409, loss = 0.001396
grad_step = 000410, loss = 0.001324
grad_step = 000411, loss = 0.001418
grad_step = 000412, loss = 0.001458
grad_step = 000413, loss = 0.001355
grad_step = 000414, loss = 0.001362
grad_step = 000415, loss = 0.001372
grad_step = 000416, loss = 0.001341
grad_step = 000417, loss = 0.001390
grad_step = 000418, loss = 0.001359
grad_step = 000419, loss = 0.001299
grad_step = 000420, loss = 0.001363
grad_step = 000421, loss = 0.001354
grad_step = 000422, loss = 0.001306
grad_step = 000423, loss = 0.001342
grad_step = 000424, loss = 0.001321
grad_step = 000425, loss = 0.001305
grad_step = 000426, loss = 0.001339
grad_step = 000427, loss = 0.001309
grad_step = 000428, loss = 0.001292
grad_step = 000429, loss = 0.001320
grad_step = 000430, loss = 0.001302
grad_step = 000431, loss = 0.001293
grad_step = 000432, loss = 0.001309
grad_step = 000433, loss = 0.001291
grad_step = 000434, loss = 0.001281
grad_step = 000435, loss = 0.001295
grad_step = 000436, loss = 0.001288
grad_step = 000437, loss = 0.001278
grad_step = 000438, loss = 0.001287
grad_step = 000439, loss = 0.001280
grad_step = 000440, loss = 0.001274
grad_step = 000441, loss = 0.001281
grad_step = 000442, loss = 0.001277
grad_step = 000443, loss = 0.001271
grad_step = 000444, loss = 0.001276
grad_step = 000445, loss = 0.001271
grad_step = 000446, loss = 0.001266
grad_step = 000447, loss = 0.001271
grad_step = 000448, loss = 0.001269
grad_step = 000449, loss = 0.001264
grad_step = 000450, loss = 0.001266
grad_step = 000451, loss = 0.001264
grad_step = 000452, loss = 0.001260
grad_step = 000453, loss = 0.001261
grad_step = 000454, loss = 0.001261
grad_step = 000455, loss = 0.001257
grad_step = 000456, loss = 0.001258
grad_step = 000457, loss = 0.001258
grad_step = 000458, loss = 0.001254
grad_step = 000459, loss = 0.001254
grad_step = 000460, loss = 0.001253
grad_step = 000461, loss = 0.001251
grad_step = 000462, loss = 0.001250
grad_step = 000463, loss = 0.001250
grad_step = 000464, loss = 0.001248
grad_step = 000465, loss = 0.001247
grad_step = 000466, loss = 0.001247
grad_step = 000467, loss = 0.001245
grad_step = 000468, loss = 0.001244
grad_step = 000469, loss = 0.001243
grad_step = 000470, loss = 0.001243
grad_step = 000471, loss = 0.001241
grad_step = 000472, loss = 0.001240
grad_step = 000473, loss = 0.001240
grad_step = 000474, loss = 0.001238
grad_step = 000475, loss = 0.001237
grad_step = 000476, loss = 0.001237
grad_step = 000477, loss = 0.001236
grad_step = 000478, loss = 0.001236
grad_step = 000479, loss = 0.001236
grad_step = 000480, loss = 0.001237
grad_step = 000481, loss = 0.001238
grad_step = 000482, loss = 0.001242
grad_step = 000483, loss = 0.001249
grad_step = 000484, loss = 0.001263
grad_step = 000485, loss = 0.001288
grad_step = 000486, loss = 0.001333
grad_step = 000487, loss = 0.001399
grad_step = 000488, loss = 0.001500
grad_step = 000489, loss = 0.001596
grad_step = 000490, loss = 0.001647
grad_step = 000491, loss = 0.001561
grad_step = 000492, loss = 0.001382
grad_step = 000493, loss = 0.001241
grad_step = 000494, loss = 0.001251
grad_step = 000495, loss = 0.001351
grad_step = 000496, loss = 0.001394
grad_step = 000497, loss = 0.001326
grad_step = 000498, loss = 0.001236
grad_step = 000499, loss = 0.001235
grad_step = 000500, loss = 0.001296
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001316
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

  date_run                              2020-05-13 05:14:21.738150
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.265607
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 05:14:21.743719
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.183759
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 05:14:21.749689
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.145499
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 05:14:21.754215
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.79229
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
0   2020-05-13 05:13:55.638381  ...    mean_absolute_error
1   2020-05-13 05:13:55.643122  ...     mean_squared_error
2   2020-05-13 05:13:55.647878  ...  median_absolute_error
3   2020-05-13 05:13:55.651388  ...               r2_score
4   2020-05-13 05:14:05.232627  ...    mean_absolute_error
5   2020-05-13 05:14:05.236301  ...     mean_squared_error
6   2020-05-13 05:14:05.239043  ...  median_absolute_error
7   2020-05-13 05:14:05.241469  ...               r2_score
8   2020-05-13 05:14:21.738150  ...    mean_absolute_error
9   2020-05-13 05:14:21.743719  ...     mean_squared_error
10  2020-05-13 05:14:21.749689  ...  median_absolute_error
11  2020-05-13 05:14:21.754215  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f447404a9b0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▉      | 3858432/9912422 [00:00<00:00, 38071597.92it/s]9920512it [00:00, 35909325.08it/s]                             
0it [00:00, ?it/s]32768it [00:00, 596230.81it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 468169.48it/s]1654784it [00:00, 10972935.74it/s]                         
0it [00:00, ?it/s]8192it [00:00, 185763.46it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f44269fae48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f44260290b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f44269fae48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4426029048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f44237ba4a8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f44260290b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f44269fae48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4426029048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f44237ba4a8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f447404a9b0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7feee49f5208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=c2a1af9766739dcde5a77fa4726b5ebbf4153c901e80a82c12ae72c092b02cb2
  Stored in directory: /tmp/pip-ephem-wheel-cache-2d276193/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7feedad7b080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3006464/17464789 [====>.........................] - ETA: 0s
11821056/17464789 [===================>..........] - ETA: 0s
16801792/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 05:15:45.417657: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 05:15:45.422057: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-05-13 05:15:45.422202: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bf8f038ff0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 05:15:45.422216: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.7280 - accuracy: 0.4960
 2000/25000 [=>............................] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7177 - accuracy: 0.4967
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7356 - accuracy: 0.4955
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6697 - accuracy: 0.4998
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6538 - accuracy: 0.5008
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6031 - accuracy: 0.5041
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6206 - accuracy: 0.5030
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6257 - accuracy: 0.5027
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6498 - accuracy: 0.5011
11000/25000 [============>.................] - ETA: 3s - loss: 7.6764 - accuracy: 0.4994
12000/25000 [=============>................] - ETA: 2s - loss: 7.6845 - accuracy: 0.4988
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6796 - accuracy: 0.4992
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6710 - accuracy: 0.4997
15000/25000 [=================>............] - ETA: 2s - loss: 7.6973 - accuracy: 0.4980
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6944 - accuracy: 0.4982
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6910 - accuracy: 0.4984
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6964 - accuracy: 0.4981
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6868 - accuracy: 0.4987
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6781 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6688 - accuracy: 0.4999
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6729 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6615 - accuracy: 0.5003
25000/25000 [==============================] - 6s 259us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 05:15:57.837530
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 05:15:57.837530  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<44:12:26, 5.42kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<31:10:39, 7.68kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<21:52:32, 10.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<15:18:54, 15.6kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:02<10:41:23, 22.3kB/s].vector_cache/glove.6B.zip:   1%|          | 9.59M/862M [00:02<7:25:57, 31.9kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 15.2M/862M [00:02<5:10:12, 45.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.0M/862M [00:02<3:35:46, 65.0kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.3M/862M [00:02<2:30:21, 92.8kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.4M/862M [00:02<1:44:50, 132kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 33.8M/862M [00:02<1:13:06, 189kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.1M/862M [00:02<51:00, 269kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 42.6M/862M [00:02<35:36, 384kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.7M/862M [00:03<24:54, 546kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:03<17:24, 776kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:03<13:57, 967kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.3M/862M [00:03<1:40:13, 135kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:05<1:11:43, 187kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:05<52:48, 254kB/s]  .vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:05<37:35, 357kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:07<28:24, 470kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<21:18, 627kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:07<15:14, 874kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:09<13:36, 976kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:09<12:15, 1.08MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:09<09:13, 1.44MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:10<06:35, 2.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:11<12:43:36, 17.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:11<8:55:39, 24.7kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:11<6:14:39, 35.2kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:11<4:21:39, 50.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:13<4:14:13, 51.7kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:13<3:00:35, 72.8kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:13<2:06:57, 103kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.1M/862M [00:15<1:30:38, 144kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:15<1:04:45, 202kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.0M/862M [00:15<45:34, 286kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:17<34:52, 373kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:17<27:04, 481kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.2M/862M [00:17<19:35, 664kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:17<13:48, 938kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:19<12:34:39, 17.2kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:19<8:49:20, 24.4kB/s] .vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:19<6:10:07, 34.9kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:21<4:21:23, 49.3kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:21<3:05:35, 69.4kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.4M/862M [00:21<2:10:26, 98.6kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:23<1:33:01, 138kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:23<1:06:23, 193kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.5M/862M [00:23<46:42, 274kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:25<35:35, 358kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:25<25:59, 490kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:25<18:25, 690kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<13:02, 972kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<1:00:13, 210kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<44:43, 283kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:27<31:49, 398kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<22:22, 564kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<22:48, 553kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<17:15, 730kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<12:22, 1.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<11:35, 1.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<10:40, 1.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<08:00, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<05:43, 2.18MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<16:26, 758kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<11:56, 1.04MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<08:40, 1.43MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<09:01, 1.37MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<08:49, 1.40MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<06:48, 1.82MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<06:45, 1.82MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<05:59, 2.06MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<04:30, 2.73MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<06:00, 2.04MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<06:49, 1.79MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:39<05:24, 2.26MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<03:55, 3.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<1:20:47, 151kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<57:47, 211kB/s]  .vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<40:40, 299kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<31:14, 388kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:43<24:20, 498kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<17:38, 687kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<14:14, 846kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:45<11:13, 1.07MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<08:09, 1.47MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:28, 1.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:47<08:21, 1.43MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<06:25, 1.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<04:35, 2.60MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<35:09, 339kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<25:50, 461kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<18:21, 648kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<15:33, 761kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<13:18, 891kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:51<09:54, 1.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<08:48, 1.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:22, 1.60MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<05:24, 2.17MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<06:31, 1.79MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:45, 2.03MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:55<04:19, 2.70MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<05:46, 2.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:24, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:04, 2.29MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<05:24, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<04:58, 2.33MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:46, 3.06MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<05:19, 2.16MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<04:54, 2.35MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:40, 3.13MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<05:15, 2.18MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<04:50, 2.36MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<03:40, 3.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<05:15, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<04:38, 2.45MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<03:28, 3.27MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<02:35, 4.37MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<47:22, 239kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<35:27, 319kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<25:18, 446kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<17:46, 632kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<11:01:12, 17.0kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<7:43:46, 24.2kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<5:24:11, 34.6kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<3:48:48, 48.8kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<2:41:15, 69.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<1:52:53, 98.6kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<1:21:17, 137kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<59:19, 187kB/s]  .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<42:03, 263kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<29:26, 375kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<46:59, 235kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<34:01, 324kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<24:03, 457kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:16<19:18, 568kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<15:50, 691kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<11:40, 938kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<08:16, 1.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<31:45, 343kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:18<23:21, 466kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<16:33, 656kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<14:01, 772kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<12:07, 892kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<09:03, 1.19MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<06:27, 1.67MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<36:47, 292kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<26:53, 400kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<19:01, 564kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<15:43, 679kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<13:17, 804kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<09:51, 1.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<06:59, 1.52MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<29:27, 361kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<21:44, 488kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<15:24, 687kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<13:10, 801kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<11:29, 918kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<08:31, 1.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<06:03, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:30<11:33, 907kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<09:00, 1.16MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<06:31, 1.60MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<04:42, 2.22MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:32<22:54, 454kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<17:06, 608kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<12:10, 852kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<10:53, 949kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:34<09:50, 1.05MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:34<07:26, 1.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:34<05:18, 1.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:35<27:28, 374kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:36<20:19, 505kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<14:25, 710kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<10:12, 999kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<1:07:15, 152kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:38<48:10, 212kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<33:52, 300kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<25:52, 392kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<20:15, 500kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<14:42, 688kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<10:21, 972kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<30:29, 330kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<22:24, 449kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<15:54, 631kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<13:23, 746kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:43<10:26, 957kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<07:30, 1.33MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<07:32, 1.32MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<07:23, 1.34MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:46<05:41, 1.74MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:05, 2.41MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<1:13:15, 135kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<52:18, 188kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:48<36:46, 267kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<27:53, 351kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<20:31, 477kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<14:35, 669kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<12:26, 782kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<10:46, 902kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<07:58, 1.22MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:41, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<08:41, 1.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<07:05, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:11, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<05:51, 1.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<05:04, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<03:45, 2.54MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<04:51, 1.96MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:57<04:23, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:19, 2.86MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<04:22, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<04:54, 1.93MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:53, 2.42MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<04:14, 2.21MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<03:58, 2.36MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<03:01, 3.10MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<04:14, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<03:58, 2.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<03:00, 3.08MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<04:13, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<04:03, 2.27MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<03:06, 2.96MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<04:05, 2.25MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<04:48, 1.90MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<03:51, 2.38MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<02:48, 3.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<30:47, 296kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<22:28, 405kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<15:53, 570kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<13:10, 686kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<10:00, 903kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<07:13, 1.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<07:05, 1.26MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<05:53, 1.52MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<04:21, 2.05MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<05:03, 1.76MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<05:20, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:08, 2.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<03:00, 2.94MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<06:30, 1.36MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<05:27, 1.61MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:01, 2.19MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<04:49, 1.82MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:19<04:17, 2.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<03:11, 2.73MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<04:13, 2.06MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<04:48, 1.81MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<03:48, 2.27MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<02:45, 3.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<1:03:25, 136kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:23<45:16, 190kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<31:49, 270kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<24:08, 354kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<17:46, 481kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<12:35, 676kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<10:45, 788kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<09:21, 906kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<06:59, 1.21MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<04:57, 1.70MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<16:21, 514kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:29<12:18, 683kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<08:48, 951kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:31<08:05, 1.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<06:31, 1.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:44, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<05:12, 1.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<05:23, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<04:12, 1.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:02, 2.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<28:09, 291kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:35<20:32, 399kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<14:33, 562kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<12:02, 676kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:37<10:04, 807kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<07:28, 1.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<05:18, 1.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<1:01:17, 132kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:39<43:42, 185kB/s]  .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<30:42, 262kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<23:14, 344kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:41<17:05, 468kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<12:08, 657kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<10:17, 771kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:43<08:48, 901kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:43<06:33, 1.21MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:39, 1.69MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<44:34, 176kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<31:59, 246kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:45<22:31, 348kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<17:29, 446kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<13:53, 561kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:47<10:04, 773kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<07:05, 1.09MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<15:05, 512kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<11:23, 678kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<08:08, 944kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<07:26, 1.03MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<06:51, 1.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<05:08, 1.48MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:40, 2.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<07:00, 1.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<05:43, 1.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<04:10, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:38, 1.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:51, 1.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:44, 2.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<02:41, 2.77MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<07:22, 1.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<05:56, 1.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<04:19, 1.72MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<04:42, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<04:49, 1.53MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:45, 1.96MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<02:42, 2.71MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<54:06, 135kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<38:37, 189kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<27:07, 269kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<20:33, 352kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<15:51, 457kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<11:26, 632kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:02<08:03, 892kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:04<31:26, 228kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<22:44, 316kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<16:02, 446kB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:06<12:48, 555kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<10:28, 679kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<07:38, 929kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<05:24, 1.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<07:24, 951kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:08<05:54, 1.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<04:17, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<04:36, 1.51MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<04:42, 1.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<03:36, 1.93MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<02:35, 2.67MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<07:38, 905kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:12<06:04, 1.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:23, 1.56MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<04:37, 1.48MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:14<03:57, 1.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<02:55, 2.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<03:35, 1.89MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<03:13, 2.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<02:25, 2.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<03:14, 2.07MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<03:40, 1.82MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<02:53, 2.31MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:04, 3.19MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<10:27, 634kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<07:28, 885kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<05:26, 1.21MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:22<14:01, 468kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<10:29, 625kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<07:29, 872kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<06:43, 966kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<06:05, 1.07MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<04:35, 1.41MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:16, 1.96MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<21:37, 297kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:26<15:47, 407kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<11:10, 572kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<09:15, 687kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:28<07:46, 817kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<05:45, 1.10MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<04:04, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<47:47, 132kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:30<34:04, 184kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<23:53, 262kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<18:03, 345kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:32<13:16, 468kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<09:25, 657kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<07:58, 772kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:34<06:52, 894kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:34<05:07, 1.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:38, 1.67MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<20:51, 292kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<15:14, 399kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<10:45, 563kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<08:51, 679kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<06:50, 879kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<04:54, 1.22MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<04:46, 1.24MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<04:33, 1.30MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:40<03:27, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:27, 2.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<12:54, 456kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<09:38, 609kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<06:52, 851kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<06:06, 950kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<04:53, 1.19MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:32, 1.63MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:46, 1.52MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:48, 1.51MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<02:56, 1.94MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<02:07, 2.68MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<26:33, 214kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<19:10, 296kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<13:30, 418kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<10:40, 525kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:49<08:41, 645kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<06:19, 883kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<04:28, 1.24MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<05:32, 999kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:51<04:28, 1.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:15, 1.69MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<03:30, 1.56MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<03:34, 1.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:46, 1.96MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<01:59, 2.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<45:23, 119kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:55<32:16, 167kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<22:36, 237kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<16:57, 314kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<12:24, 429kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<08:45, 605kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<06:09, 854kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<19:58, 263kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:59<15:05, 348kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 547M/862M [03:59<10:47, 486kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<07:34, 688kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<07:53, 658kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<06:03, 856kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:20, 1.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<04:11, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:03<04:01, 1.27MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:05, 1.65MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:12, 2.29MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<16:42, 302kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<12:13, 413kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<08:38, 581kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<07:09, 697kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:07<06:03, 822kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<04:30, 1.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<03:10, 1.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<13:34, 362kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<10:00, 490kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<07:06, 688kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:11<06:03, 801kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:11<04:39, 1.04MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<03:22, 1.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<03:27, 1.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<03:25, 1.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<02:37, 1.82MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<01:52, 2.51MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<17:17, 273kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<12:34, 374kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<08:51, 528kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<07:14, 641kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:17<05:33, 834kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<03:58, 1.16MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<03:49, 1.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:19<03:38, 1.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:19<02:45, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:58, 2.30MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<04:03, 1.11MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:21<03:18, 1.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:25, 1.85MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:42, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:23<02:21, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<01:44, 2.53MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<02:14, 1.95MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:25<02:01, 2.16MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<01:31, 2.85MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:02, 2.10MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<02:18, 1.86MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:27<01:48, 2.37MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<01:19, 3.22MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:28<02:41, 1.57MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:29<04:31, 934kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<03:10, 1.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<11:18, 368kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<08:20, 498kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:31<05:54, 699kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<05:03, 810kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<03:57, 1.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:50, 1.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:54, 1.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:53, 1.39MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:12, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<01:34, 2.51MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<03:30, 1.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:36<02:52, 1.37MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:05, 1.88MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:20, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:38<02:28, 1.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<01:54, 2.03MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<01:22, 2.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:33, 1.49MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:40<02:11, 1.74MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:36, 2.35MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:59, 1.89MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:47, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:20, 2.77MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<01:46, 2.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<01:37, 2.26MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:13, 3.00MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<01:40, 2.15MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<01:57, 1.85MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<01:33, 2.31MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:07, 3.16MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<11:24, 311kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<08:20, 424kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<05:53, 597kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<04:53, 712kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<04:09, 835kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<03:03, 1.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:10, 1.58MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<02:54, 1.17MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:52<02:19, 1.46MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<01:41, 2.00MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:13, 2.74MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<07:22, 453kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:54<05:29, 607kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:53, 850kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<03:27, 948kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:56<03:06, 1.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:18, 1.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:39, 1.95MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<02:31, 1.27MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:58<02:06, 1.52MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:32, 2.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:47, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:35, 1.97MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:10, 2.65MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:31, 2.02MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:42, 1.79MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<01:20, 2.28MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<00:58, 3.10MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<01:47, 1.67MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:04<01:34, 1.90MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:09, 2.55MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<01:28, 1.99MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<01:20, 2.18MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<00:59, 2.91MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:20, 2.13MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:32, 1.85MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:13, 2.31MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<00:52, 3.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<07:03, 395kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<05:13, 533kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:41, 749kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<03:11, 856kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<02:30, 1.08MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:48, 1.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:52, 1.42MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<01:34, 1.67MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:09, 2.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:24, 1.85MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<01:14, 2.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<00:55, 2.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:13, 2.05MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:18<01:23, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:18<01:06, 2.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<00:47, 3.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<07:54, 310kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<05:46, 423kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:20<04:03, 596kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<03:21, 710kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<02:50, 835kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:22<02:05, 1.13MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:28, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<02:06, 1.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:43, 1.34MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:24<01:15, 1.82MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:22, 1.63MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:25, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:05, 2.03MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<00:46, 2.80MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<09:29, 229kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<06:51, 316kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<04:48, 446kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<03:47, 556kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<02:51, 734kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:02, 1.02MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:52, 1.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<01:43, 1.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:18, 1.55MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:54, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<14:41, 134kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<10:27, 188kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<07:16, 267kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<05:25, 350kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<04:10, 454kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<03:00, 629kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<02:03, 890kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<04:55, 373kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<03:37, 504kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<02:32, 707kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<02:09, 819kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:51, 945kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<01:23, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:57, 1.76MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<09:32, 178kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<06:49, 247kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<04:45, 350kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<03:37, 448kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<02:42, 599kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<01:54, 840kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<01:39, 939kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:45<01:29, 1.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<01:07, 1.38MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:45<00:46, 1.92MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<04:06, 363kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<03:01, 492kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<02:06, 690kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<01:46, 803kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<01:32, 922kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:08, 1.23MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:47, 1.72MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<04:50, 280kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:51<03:30, 383kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<02:26, 540kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<01:57, 654kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<01:37, 785kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<01:11, 1.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:49, 1.48MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:55<10:30, 116kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<07:26, 163kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<05:07, 231kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:57<03:44, 307kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<02:51, 401kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<02:01, 558kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<01:23, 789kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<01:30, 715kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<01:09, 923kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:49, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:47, 1.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:45, 1.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:01<00:34, 1.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:23, 2.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<02:32, 371kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:03<01:51, 503kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<01:17, 705kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:04, 817kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:05<00:55, 935kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:41, 1.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:27, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<02:44, 294kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:07<01:59, 402kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<01:21, 567kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<01:04, 681kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:54, 806kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:09<00:39, 1.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:26, 1.52MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<01:43, 386kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<01:16, 521kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:52, 731kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:42, 840kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:37, 954kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:13<00:27, 1.28MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:18, 1.78MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:25, 1.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:20, 1.53MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<00:14, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:15, 1.77MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:16, 1.67MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:12, 2.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:08, 2.92MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<02:54, 135kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<02:02, 190kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<01:20, 269kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:55, 353kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:39, 479kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:26, 672kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:19, 786kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:14, 1.00MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:09, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:08, 1.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:08, 1.37MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:05, 1.79MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:03, 2.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:09, 771kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:06, 990kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.36MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.38MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:00, 1.81MB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1023/400000 [00:00<00:39, 10221.21it/s]  1%|          | 2030/400000 [00:00<00:39, 10174.54it/s]  1%|          | 3009/400000 [00:00<00:39, 10052.57it/s]  1%|          | 3958/400000 [00:00<00:40, 9873.06it/s]   1%|          | 4957/400000 [00:00<00:39, 9907.76it/s]  1%|▏         | 5963/400000 [00:00<00:39, 9950.04it/s]  2%|▏         | 6979/400000 [00:00<00:39, 10010.95it/s]  2%|▏         | 7917/400000 [00:00<00:39, 9810.41it/s]   2%|▏         | 8883/400000 [00:00<00:40, 9762.02it/s]  2%|▏         | 9819/400000 [00:01<00:40, 9560.54it/s]  3%|▎         | 10764/400000 [00:01<00:40, 9526.30it/s]  3%|▎         | 11698/400000 [00:01<00:41, 9428.04it/s]  3%|▎         | 12628/400000 [00:01<00:41, 9278.23it/s]  3%|▎         | 13565/400000 [00:01<00:41, 9304.16it/s]  4%|▎         | 14514/400000 [00:01<00:41, 9356.41it/s]  4%|▍         | 15466/400000 [00:01<00:40, 9403.33it/s]  4%|▍         | 16432/400000 [00:01<00:40, 9477.57it/s]  4%|▍         | 17378/400000 [00:01<00:40, 9467.70it/s]  5%|▍         | 18419/400000 [00:01<00:39, 9731.41it/s]  5%|▍         | 19394/400000 [00:02<00:39, 9693.58it/s]  5%|▌         | 20452/400000 [00:02<00:38, 9940.92it/s]  5%|▌         | 21474/400000 [00:02<00:37, 10021.25it/s]  6%|▌         | 22478/400000 [00:02<00:38, 9693.71it/s]   6%|▌         | 23504/400000 [00:02<00:38, 9854.86it/s]  6%|▌         | 24530/400000 [00:02<00:37, 9972.97it/s]  6%|▋         | 25530/400000 [00:02<00:37, 9955.51it/s]  7%|▋         | 26533/400000 [00:02<00:37, 9976.52it/s]  7%|▋         | 27532/400000 [00:02<00:37, 9960.88it/s]  7%|▋         | 28578/400000 [00:02<00:36, 10103.53it/s]  7%|▋         | 29617/400000 [00:03<00:36, 10185.95it/s]  8%|▊         | 30637/400000 [00:03<00:36, 10036.65it/s]  8%|▊         | 31642/400000 [00:03<00:37, 9711.82it/s]   8%|▊         | 32617/400000 [00:03<00:38, 9554.10it/s]  8%|▊         | 33666/400000 [00:03<00:37, 9813.98it/s]  9%|▊         | 34652/400000 [00:03<00:37, 9653.85it/s]  9%|▉         | 35637/400000 [00:03<00:37, 9709.19it/s]  9%|▉         | 36611/400000 [00:03<00:37, 9670.26it/s]  9%|▉         | 37580/400000 [00:03<00:38, 9339.82it/s] 10%|▉         | 38549/400000 [00:03<00:38, 9441.67it/s] 10%|▉         | 39569/400000 [00:04<00:37, 9656.94it/s] 10%|█         | 40538/400000 [00:04<00:37, 9611.35it/s] 10%|█         | 41502/400000 [00:04<00:37, 9543.81it/s] 11%|█         | 42463/400000 [00:04<00:37, 9561.58it/s] 11%|█         | 43492/400000 [00:04<00:36, 9767.26it/s] 11%|█         | 44471/400000 [00:04<00:36, 9765.26it/s] 11%|█▏        | 45449/400000 [00:04<00:36, 9670.26it/s] 12%|█▏        | 46418/400000 [00:04<00:37, 9523.59it/s] 12%|█▏        | 47427/400000 [00:04<00:36, 9686.23it/s] 12%|█▏        | 48425/400000 [00:04<00:35, 9772.44it/s] 12%|█▏        | 49438/400000 [00:05<00:35, 9876.24it/s] 13%|█▎        | 50434/400000 [00:05<00:35, 9898.31it/s] 13%|█▎        | 51425/400000 [00:05<00:35, 9887.95it/s] 13%|█▎        | 52415/400000 [00:05<00:35, 9811.92it/s] 13%|█▎        | 53429/400000 [00:05<00:34, 9906.74it/s] 14%|█▎        | 54460/400000 [00:05<00:34, 10022.89it/s] 14%|█▍        | 55464/400000 [00:05<00:34, 9962.60it/s]  14%|█▍        | 56461/400000 [00:05<00:34, 9904.31it/s] 14%|█▍        | 57452/400000 [00:05<00:35, 9747.04it/s] 15%|█▍        | 58428/400000 [00:05<00:35, 9708.34it/s] 15%|█▍        | 59406/400000 [00:06<00:35, 9729.40it/s] 15%|█▌        | 60380/400000 [00:06<00:35, 9593.30it/s] 15%|█▌        | 61345/400000 [00:06<00:35, 9607.87it/s] 16%|█▌        | 62307/400000 [00:06<00:35, 9527.29it/s] 16%|█▌        | 63312/400000 [00:06<00:34, 9677.58it/s] 16%|█▌        | 64338/400000 [00:06<00:34, 9843.69it/s] 16%|█▋        | 65347/400000 [00:06<00:33, 9913.97it/s] 17%|█▋        | 66354/400000 [00:06<00:33, 9958.27it/s] 17%|█▋        | 67351/400000 [00:06<00:33, 9955.28it/s] 17%|█▋        | 68348/400000 [00:07<00:33, 9931.74it/s] 17%|█▋        | 69384/400000 [00:07<00:32, 10055.29it/s] 18%|█▊        | 70391/400000 [00:07<00:33, 9945.14it/s]  18%|█▊        | 71387/400000 [00:07<00:33, 9785.60it/s] 18%|█▊        | 72367/400000 [00:07<00:34, 9484.58it/s] 18%|█▊        | 73319/400000 [00:07<00:34, 9338.33it/s] 19%|█▊        | 74334/400000 [00:07<00:34, 9565.74it/s] 19%|█▉        | 75391/400000 [00:07<00:32, 9845.15it/s] 19%|█▉        | 76380/400000 [00:07<00:33, 9597.44it/s] 19%|█▉        | 77357/400000 [00:07<00:33, 9647.50it/s] 20%|█▉        | 78325/400000 [00:08<00:33, 9538.15it/s] 20%|█▉        | 79343/400000 [00:08<00:32, 9721.32it/s] 20%|██        | 80362/400000 [00:08<00:32, 9855.30it/s] 20%|██        | 81350/400000 [00:08<00:32, 9704.96it/s] 21%|██        | 82339/400000 [00:08<00:32, 9758.98it/s] 21%|██        | 83317/400000 [00:08<00:32, 9744.69it/s] 21%|██        | 84329/400000 [00:08<00:32, 9853.07it/s] 21%|██▏       | 85373/400000 [00:08<00:31, 10021.96it/s] 22%|██▏       | 86377/400000 [00:08<00:31, 9995.74it/s]  22%|██▏       | 87378/400000 [00:08<00:31, 9990.66it/s] 22%|██▏       | 88378/400000 [00:09<00:31, 9904.62it/s] 22%|██▏       | 89385/400000 [00:09<00:31, 9950.78it/s] 23%|██▎       | 90381/400000 [00:09<00:31, 9816.93it/s] 23%|██▎       | 91375/400000 [00:09<00:31, 9851.91it/s] 23%|██▎       | 92409/400000 [00:09<00:30, 9991.85it/s] 23%|██▎       | 93435/400000 [00:09<00:30, 10070.38it/s] 24%|██▎       | 94443/400000 [00:09<00:30, 10073.23it/s] 24%|██▍       | 95451/400000 [00:09<00:30, 10003.44it/s] 24%|██▍       | 96452/400000 [00:09<00:30, 9909.43it/s]  24%|██▍       | 97469/400000 [00:09<00:30, 9983.71it/s] 25%|██▍       | 98468/400000 [00:10<00:30, 9910.21it/s] 25%|██▍       | 99460/400000 [00:10<00:30, 9827.54it/s] 25%|██▌       | 100502/400000 [00:10<00:29, 9997.66it/s] 25%|██▌       | 101503/400000 [00:10<00:30, 9896.76it/s] 26%|██▌       | 102549/400000 [00:10<00:29, 10059.20it/s] 26%|██▌       | 103557/400000 [00:10<00:29, 10064.24it/s] 26%|██▌       | 104593/400000 [00:10<00:29, 10149.78it/s] 26%|██▋       | 105643/400000 [00:10<00:28, 10252.12it/s] 27%|██▋       | 106675/400000 [00:10<00:28, 10269.83it/s] 27%|██▋       | 107703/400000 [00:10<00:28, 10195.49it/s] 27%|██▋       | 108740/400000 [00:11<00:28, 10247.04it/s] 27%|██▋       | 109766/400000 [00:11<00:28, 10236.43it/s] 28%|██▊       | 110790/400000 [00:11<00:28, 10096.19it/s] 28%|██▊       | 111808/400000 [00:11<00:28, 10120.07it/s] 28%|██▊       | 112821/400000 [00:11<00:29, 9844.74it/s]  28%|██▊       | 113808/400000 [00:11<00:29, 9720.29it/s] 29%|██▊       | 114797/400000 [00:11<00:29, 9768.30it/s] 29%|██▉       | 115807/400000 [00:11<00:28, 9865.39it/s] 29%|██▉       | 116811/400000 [00:11<00:28, 9915.11it/s] 29%|██▉       | 117804/400000 [00:11<00:28, 9911.40it/s] 30%|██▉       | 118816/400000 [00:12<00:28, 9972.77it/s] 30%|██▉       | 119814/400000 [00:12<00:28, 9863.96it/s] 30%|███       | 120802/400000 [00:12<00:28, 9799.16it/s] 30%|███       | 121783/400000 [00:12<00:28, 9796.84it/s] 31%|███       | 122773/400000 [00:12<00:28, 9827.26it/s] 31%|███       | 123838/400000 [00:12<00:27, 10059.31it/s] 31%|███       | 124846/400000 [00:12<00:27, 9894.09it/s]  31%|███▏      | 125880/400000 [00:12<00:27, 10023.44it/s] 32%|███▏      | 126928/400000 [00:12<00:26, 10155.45it/s] 32%|███▏      | 127946/400000 [00:13<00:26, 10093.87it/s] 32%|███▏      | 128957/400000 [00:13<00:27, 9824.49it/s]  32%|███▏      | 129942/400000 [00:13<00:28, 9542.73it/s] 33%|███▎      | 130967/400000 [00:13<00:27, 9743.28it/s] 33%|███▎      | 132027/400000 [00:13<00:26, 9984.37it/s] 33%|███▎      | 133034/400000 [00:13<00:26, 10007.80it/s] 34%|███▎      | 134089/400000 [00:13<00:26, 10162.70it/s] 34%|███▍      | 135137/400000 [00:13<00:25, 10253.16it/s] 34%|███▍      | 136165/400000 [00:13<00:25, 10170.60it/s] 34%|███▍      | 137184/400000 [00:13<00:26, 10105.80it/s] 35%|███▍      | 138203/400000 [00:14<00:25, 10128.20it/s] 35%|███▍      | 139247/400000 [00:14<00:25, 10218.75it/s] 35%|███▌      | 140270/400000 [00:14<00:25, 10090.79it/s] 35%|███▌      | 141280/400000 [00:14<00:25, 10014.55it/s] 36%|███▌      | 142332/400000 [00:14<00:25, 10159.03it/s] 36%|███▌      | 143349/400000 [00:14<00:26, 9855.77it/s]  36%|███▌      | 144352/400000 [00:14<00:25, 9905.68it/s] 36%|███▋      | 145390/400000 [00:14<00:25, 10042.70it/s] 37%|███▋      | 146444/400000 [00:14<00:24, 10184.92it/s] 37%|███▋      | 147476/400000 [00:14<00:24, 10225.03it/s] 37%|███▋      | 148500/400000 [00:15<00:24, 10123.88it/s] 37%|███▋      | 149514/400000 [00:15<00:25, 9884.82it/s]  38%|███▊      | 150524/400000 [00:15<00:25, 9948.10it/s] 38%|███▊      | 151534/400000 [00:15<00:24, 9992.86it/s] 38%|███▊      | 152541/400000 [00:15<00:24, 10015.44it/s] 38%|███▊      | 153544/400000 [00:15<00:24, 9914.05it/s]  39%|███▊      | 154549/400000 [00:15<00:24, 9954.30it/s] 39%|███▉      | 155593/400000 [00:15<00:24, 10092.92it/s] 39%|███▉      | 156640/400000 [00:15<00:23, 10200.33it/s] 39%|███▉      | 157661/400000 [00:15<00:24, 10029.18it/s] 40%|███▉      | 158677/400000 [00:16<00:23, 10065.55it/s] 40%|███▉      | 159685/400000 [00:16<00:23, 10049.86it/s] 40%|████      | 160691/400000 [00:16<00:24, 9820.31it/s]  40%|████      | 161719/400000 [00:16<00:23, 9953.65it/s] 41%|████      | 162716/400000 [00:16<00:23, 9939.78it/s] 41%|████      | 163712/400000 [00:16<00:24, 9749.25it/s] 41%|████      | 164711/400000 [00:16<00:23, 9818.12it/s] 41%|████▏     | 165695/400000 [00:16<00:24, 9748.36it/s] 42%|████▏     | 166671/400000 [00:16<00:24, 9676.50it/s] 42%|████▏     | 167640/400000 [00:16<00:24, 9569.44it/s] 42%|████▏     | 168610/400000 [00:17<00:24, 9607.90it/s] 42%|████▏     | 169603/400000 [00:17<00:23, 9700.68it/s] 43%|████▎     | 170584/400000 [00:17<00:23, 9730.96it/s] 43%|████▎     | 171558/400000 [00:17<00:23, 9706.79it/s] 43%|████▎     | 172530/400000 [00:17<00:23, 9678.76it/s] 43%|████▎     | 173499/400000 [00:17<00:23, 9618.57it/s] 44%|████▎     | 174495/400000 [00:17<00:23, 9716.76it/s] 44%|████▍     | 175526/400000 [00:17<00:22, 9886.41it/s] 44%|████▍     | 176580/400000 [00:17<00:22, 10071.68it/s] 44%|████▍     | 177608/400000 [00:18<00:21, 10132.18it/s] 45%|████▍     | 178624/400000 [00:18<00:21, 10139.79it/s] 45%|████▍     | 179639/400000 [00:18<00:21, 10137.13it/s] 45%|████▌     | 180654/400000 [00:18<00:22, 9964.91it/s]  45%|████▌     | 181652/400000 [00:18<00:22, 9903.95it/s] 46%|████▌     | 182644/400000 [00:18<00:22, 9577.93it/s] 46%|████▌     | 183605/400000 [00:18<00:22, 9473.52it/s] 46%|████▌     | 184658/400000 [00:18<00:22, 9767.49it/s] 46%|████▋     | 185713/400000 [00:18<00:21, 9988.16it/s] 47%|████▋     | 186720/400000 [00:18<00:21, 10010.79it/s] 47%|████▋     | 187724/400000 [00:19<00:21, 9972.85it/s]  47%|████▋     | 188727/400000 [00:19<00:21, 9989.54it/s] 47%|████▋     | 189728/400000 [00:19<00:21, 9992.39it/s] 48%|████▊     | 190729/400000 [00:19<00:21, 9948.65it/s] 48%|████▊     | 191796/400000 [00:19<00:20, 10154.26it/s] 48%|████▊     | 192824/400000 [00:19<00:20, 10190.82it/s] 48%|████▊     | 193845/400000 [00:19<00:20, 10090.14it/s] 49%|████▊     | 194862/400000 [00:19<00:20, 10113.72it/s] 49%|████▉     | 195921/400000 [00:19<00:19, 10250.57it/s] 49%|████▉     | 196986/400000 [00:19<00:19, 10364.19it/s] 50%|████▉     | 198024/400000 [00:20<00:19, 10249.95it/s] 50%|████▉     | 199050/400000 [00:20<00:19, 10172.93it/s] 50%|█████     | 200069/400000 [00:20<00:19, 10125.58it/s] 50%|█████     | 201083/400000 [00:20<00:19, 10021.69it/s] 51%|█████     | 202123/400000 [00:20<00:19, 10128.36it/s] 51%|█████     | 203137/400000 [00:20<00:19, 9987.65it/s]  51%|█████     | 204137/400000 [00:20<00:19, 9965.59it/s] 51%|█████▏    | 205213/400000 [00:20<00:19, 10189.68it/s] 52%|█████▏    | 206275/400000 [00:20<00:18, 10313.42it/s] 52%|█████▏    | 207331/400000 [00:20<00:18, 10383.70it/s] 52%|█████▏    | 208371/400000 [00:21<00:18, 10369.61it/s] 52%|█████▏    | 209413/400000 [00:21<00:18, 10382.75it/s] 53%|█████▎    | 210452/400000 [00:21<00:18, 10195.76it/s] 53%|█████▎    | 211499/400000 [00:21<00:18, 10275.27it/s] 53%|█████▎    | 212549/400000 [00:21<00:18, 10338.31it/s] 53%|█████▎    | 213584/400000 [00:21<00:18, 10206.12it/s] 54%|█████▎    | 214606/400000 [00:21<00:18, 10028.57it/s] 54%|█████▍    | 215642/400000 [00:21<00:18, 10122.53it/s] 54%|█████▍    | 216663/400000 [00:21<00:18, 10146.19it/s] 54%|█████▍    | 217679/400000 [00:21<00:18, 9959.41it/s]  55%|█████▍    | 218697/400000 [00:22<00:18, 10023.57it/s] 55%|█████▍    | 219701/400000 [00:22<00:18, 9949.92it/s]  55%|█████▌    | 220704/400000 [00:22<00:17, 9972.86it/s] 55%|█████▌    | 221702/400000 [00:22<00:18, 9807.34it/s] 56%|█████▌    | 222684/400000 [00:22<00:18, 9729.17it/s] 56%|█████▌    | 223714/400000 [00:22<00:17, 9891.02it/s] 56%|█████▌    | 224738/400000 [00:22<00:17, 9992.20it/s] 56%|█████▋    | 225747/400000 [00:22<00:17, 10019.59it/s] 57%|█████▋    | 226790/400000 [00:22<00:17, 10138.92it/s] 57%|█████▋    | 227847/400000 [00:22<00:16, 10262.54it/s] 57%|█████▋    | 228875/400000 [00:23<00:16, 10129.74it/s] 57%|█████▋    | 229890/400000 [00:23<00:17, 9673.53it/s]  58%|█████▊    | 230893/400000 [00:23<00:17, 9775.84it/s] 58%|█████▊    | 231942/400000 [00:23<00:16, 9979.23it/s] 58%|█████▊    | 232944/400000 [00:23<00:16, 9980.64it/s] 58%|█████▊    | 233945/400000 [00:23<00:16, 9942.05it/s] 59%|█████▊    | 234942/400000 [00:23<00:16, 9893.48it/s] 59%|█████▉    | 235933/400000 [00:23<00:16, 9791.32it/s] 59%|█████▉    | 236958/400000 [00:23<00:16, 9924.18it/s] 60%|█████▉    | 238004/400000 [00:24<00:16, 10078.49it/s] 60%|█████▉    | 239031/400000 [00:24<00:15, 10134.89it/s] 60%|██████    | 240046/400000 [00:24<00:15, 10138.94it/s] 60%|██████    | 241061/400000 [00:24<00:15, 10064.32it/s] 61%|██████    | 242069/400000 [00:24<00:15, 10063.16it/s] 61%|██████    | 243104/400000 [00:24<00:15, 10147.09it/s] 61%|██████    | 244120/400000 [00:24<00:15, 10109.60it/s] 61%|██████▏   | 245132/400000 [00:24<00:15, 10053.52it/s] 62%|██████▏   | 246167/400000 [00:24<00:15, 10139.07it/s] 62%|██████▏   | 247187/400000 [00:24<00:15, 10156.95it/s] 62%|██████▏   | 248213/400000 [00:25<00:14, 10184.55it/s] 62%|██████▏   | 249232/400000 [00:25<00:15, 10046.44it/s] 63%|██████▎   | 250284/400000 [00:25<00:14, 10181.55it/s] 63%|██████▎   | 251337/400000 [00:25<00:14, 10282.41it/s] 63%|██████▎   | 252367/400000 [00:25<00:15, 9727.79it/s]  63%|██████▎   | 253347/400000 [00:25<00:15, 9530.00it/s] 64%|██████▎   | 254306/400000 [00:25<00:15, 9430.12it/s] 64%|██████▍   | 255259/400000 [00:25<00:15, 9459.43it/s] 64%|██████▍   | 256318/400000 [00:25<00:14, 9771.58it/s] 64%|██████▍   | 257404/400000 [00:25<00:14, 10072.54it/s] 65%|██████▍   | 258458/400000 [00:26<00:13, 10205.28it/s] 65%|██████▍   | 259483/400000 [00:26<00:13, 10193.57it/s] 65%|██████▌   | 260522/400000 [00:26<00:13, 10248.53it/s] 65%|██████▌   | 261550/400000 [00:26<00:13, 10096.92it/s] 66%|██████▌   | 262615/400000 [00:26<00:13, 10254.74it/s] 66%|██████▌   | 263643/400000 [00:26<00:13, 10050.91it/s] 66%|██████▌   | 264651/400000 [00:26<00:13, 9883.36it/s]  66%|██████▋   | 265642/400000 [00:26<00:13, 9765.43it/s] 67%|██████▋   | 266657/400000 [00:26<00:13, 9877.18it/s] 67%|██████▋   | 267647/400000 [00:26<00:13, 9848.06it/s] 67%|██████▋   | 268655/400000 [00:27<00:13, 9915.12it/s] 67%|██████▋   | 269648/400000 [00:27<00:13, 9857.48it/s] 68%|██████▊   | 270656/400000 [00:27<00:13, 9921.76it/s] 68%|██████▊   | 271649/400000 [00:27<00:13, 9797.56it/s] 68%|██████▊   | 272701/400000 [00:27<00:12, 10002.22it/s] 68%|██████▊   | 273704/400000 [00:27<00:12, 10009.41it/s] 69%|██████▊   | 274707/400000 [00:27<00:12, 9907.72it/s]  69%|██████▉   | 275699/400000 [00:27<00:12, 9895.53it/s] 69%|██████▉   | 276690/400000 [00:27<00:12, 9595.54it/s] 69%|██████▉   | 277712/400000 [00:28<00:12, 9772.82it/s] 70%|██████▉   | 278764/400000 [00:28<00:12, 9985.16it/s] 70%|██████▉   | 279766/400000 [00:28<00:12, 9747.78it/s] 70%|███████   | 280753/400000 [00:28<00:12, 9749.37it/s] 70%|███████   | 281731/400000 [00:28<00:12, 9709.30it/s] 71%|███████   | 282782/400000 [00:28<00:11, 9934.55it/s] 71%|███████   | 283778/400000 [00:28<00:11, 9814.36it/s] 71%|███████   | 284762/400000 [00:28<00:11, 9790.25it/s] 71%|███████▏  | 285813/400000 [00:28<00:11, 9993.13it/s] 72%|███████▏  | 286833/400000 [00:28<00:11, 10050.75it/s] 72%|███████▏  | 287871/400000 [00:29<00:11, 10146.71it/s] 72%|███████▏  | 288901/400000 [00:29<00:10, 10191.15it/s] 72%|███████▏  | 289922/400000 [00:29<00:10, 10111.27it/s] 73%|███████▎  | 290934/400000 [00:29<00:10, 10089.19it/s] 73%|███████▎  | 292000/400000 [00:29<00:10, 10252.59it/s] 73%|███████▎  | 293032/400000 [00:29<00:10, 10272.05it/s] 74%|███████▎  | 294060/400000 [00:29<00:10, 10152.84it/s] 74%|███████▍  | 295077/400000 [00:29<00:10, 9986.18it/s]  74%|███████▍  | 296077/400000 [00:29<00:10, 9970.81it/s] 74%|███████▍  | 297075/400000 [00:29<00:10, 9878.44it/s] 75%|███████▍  | 298099/400000 [00:30<00:10, 9981.52it/s] 75%|███████▍  | 299148/400000 [00:30<00:09, 10126.51it/s] 75%|███████▌  | 300162/400000 [00:30<00:10, 9885.60it/s]  75%|███████▌  | 301153/400000 [00:30<00:10, 9781.91it/s] 76%|███████▌  | 302190/400000 [00:30<00:09, 9950.74it/s] 76%|███████▌  | 303257/400000 [00:30<00:09, 10155.76it/s] 76%|███████▌  | 304309/400000 [00:30<00:09, 10261.16it/s] 76%|███████▋  | 305338/400000 [00:30<00:09, 10210.97it/s] 77%|███████▋  | 306361/400000 [00:30<00:09, 10204.59it/s] 77%|███████▋  | 307399/400000 [00:30<00:09, 10255.54it/s] 77%|███████▋  | 308487/400000 [00:31<00:08, 10433.63it/s] 77%|███████▋  | 309532/400000 [00:31<00:08, 10410.15it/s] 78%|███████▊  | 310574/400000 [00:31<00:08, 10336.06it/s] 78%|███████▊  | 311609/400000 [00:31<00:08, 10085.15it/s] 78%|███████▊  | 312620/400000 [00:31<00:08, 10089.18it/s] 78%|███████▊  | 313631/400000 [00:31<00:08, 9952.24it/s]  79%|███████▊  | 314628/400000 [00:31<00:08, 9815.26it/s] 79%|███████▉  | 315611/400000 [00:31<00:08, 9799.75it/s] 79%|███████▉  | 316675/400000 [00:31<00:08, 10035.70it/s] 79%|███████▉  | 317710/400000 [00:31<00:08, 10126.93it/s] 80%|███████▉  | 318725/400000 [00:32<00:08, 10111.57it/s] 80%|███████▉  | 319753/400000 [00:32<00:07, 10160.18it/s] 80%|████████  | 320770/400000 [00:32<00:07, 10012.06it/s] 80%|████████  | 321809/400000 [00:32<00:07, 10119.96it/s] 81%|████████  | 322854/400000 [00:32<00:07, 10215.53it/s] 81%|████████  | 323877/400000 [00:32<00:07, 10153.49it/s] 81%|████████  | 324894/400000 [00:32<00:07, 10099.57it/s] 81%|████████▏ | 325905/400000 [00:32<00:07, 9982.19it/s]  82%|████████▏ | 326904/400000 [00:32<00:07, 9893.77it/s] 82%|████████▏ | 327895/400000 [00:33<00:07, 9718.49it/s] 82%|████████▏ | 328869/400000 [00:33<00:07, 9598.90it/s] 82%|████████▏ | 329831/400000 [00:33<00:07, 9201.70it/s] 83%|████████▎ | 330828/400000 [00:33<00:07, 9419.18it/s] 83%|████████▎ | 331805/400000 [00:33<00:07, 9519.56it/s] 83%|████████▎ | 332843/400000 [00:33<00:06, 9759.69it/s] 83%|████████▎ | 333823/400000 [00:33<00:06, 9758.19it/s] 84%|████████▎ | 334848/400000 [00:33<00:06, 9899.75it/s] 84%|████████▍ | 335841/400000 [00:33<00:06, 9815.02it/s] 84%|████████▍ | 336849/400000 [00:33<00:06, 9890.17it/s] 84%|████████▍ | 337870/400000 [00:34<00:06, 9982.62it/s] 85%|████████▍ | 338872/400000 [00:34<00:06, 9991.31it/s] 85%|████████▍ | 339872/400000 [00:34<00:06, 9858.72it/s] 85%|████████▌ | 340859/400000 [00:34<00:06, 9733.37it/s] 85%|████████▌ | 341875/400000 [00:34<00:05, 9854.29it/s] 86%|████████▌ | 342906/400000 [00:34<00:05, 9983.33it/s] 86%|████████▌ | 343961/400000 [00:34<00:05, 10144.85it/s] 86%|████████▌ | 344977/400000 [00:34<00:05, 9972.48it/s]  86%|████████▋ | 346000/400000 [00:34<00:05, 10047.48it/s] 87%|████████▋ | 347007/400000 [00:34<00:05, 9925.64it/s]  87%|████████▋ | 348001/400000 [00:35<00:05, 9795.77it/s] 87%|████████▋ | 348982/400000 [00:35<00:05, 9673.92it/s] 87%|████████▋ | 349998/400000 [00:35<00:05, 9812.73it/s] 88%|████████▊ | 350981/400000 [00:35<00:05, 9569.37it/s] 88%|████████▊ | 352021/400000 [00:35<00:04, 9803.49it/s] 88%|████████▊ | 353075/400000 [00:35<00:04, 10012.42it/s] 89%|████████▊ | 354080/400000 [00:35<00:04, 9794.36it/s]  89%|████████▉ | 355117/400000 [00:35<00:04, 9959.35it/s] 89%|████████▉ | 356133/400000 [00:35<00:04, 10018.66it/s] 89%|████████▉ | 357145/400000 [00:35<00:04, 10047.79it/s] 90%|████████▉ | 358158/400000 [00:36<00:04, 10072.20it/s] 90%|████████▉ | 359194/400000 [00:36<00:04, 10155.62it/s] 90%|█████████ | 360211/400000 [00:36<00:03, 10125.39it/s] 90%|█████████ | 361225/400000 [00:36<00:03, 9934.27it/s]  91%|█████████ | 362260/400000 [00:36<00:03, 10055.35it/s] 91%|█████████ | 363348/400000 [00:36<00:03, 10287.98it/s] 91%|█████████ | 364379/400000 [00:36<00:03, 10283.64it/s] 91%|█████████▏| 365409/400000 [00:36<00:03, 10127.08it/s] 92%|█████████▏| 366424/400000 [00:36<00:03, 9802.38it/s]  92%|█████████▏| 367408/400000 [00:36<00:03, 9721.64it/s] 92%|█████████▏| 368383/400000 [00:37<00:03, 9721.66it/s] 92%|█████████▏| 369369/400000 [00:37<00:03, 9761.66it/s] 93%|█████████▎| 370347/400000 [00:37<00:03, 9514.37it/s] 93%|█████████▎| 371301/400000 [00:37<00:03, 9410.87it/s] 93%|█████████▎| 372244/400000 [00:37<00:02, 9411.77it/s] 93%|█████████▎| 373220/400000 [00:37<00:02, 9510.46it/s] 94%|█████████▎| 374283/400000 [00:37<00:02, 9819.96it/s] 94%|█████████▍| 375294/400000 [00:37<00:02, 9904.54it/s] 94%|█████████▍| 376287/400000 [00:37<00:02, 9771.53it/s] 94%|█████████▍| 377267/400000 [00:38<00:02, 9742.33it/s] 95%|█████████▍| 378271/400000 [00:38<00:02, 9828.48it/s] 95%|█████████▍| 379256/400000 [00:38<00:02, 9690.26it/s] 95%|█████████▌| 380227/400000 [00:38<00:02, 9591.16it/s] 95%|█████████▌| 381221/400000 [00:38<00:01, 9690.25it/s] 96%|█████████▌| 382232/400000 [00:38<00:01, 9811.77it/s] 96%|█████████▌| 383219/400000 [00:38<00:01, 9827.97it/s] 96%|█████████▌| 384253/400000 [00:38<00:01, 9974.70it/s] 96%|█████████▋| 385252/400000 [00:38<00:01, 9791.67it/s] 97%|█████████▋| 386233/400000 [00:38<00:01, 9702.40it/s] 97%|█████████▋| 387205/400000 [00:39<00:01, 9684.75it/s] 97%|█████████▋| 388263/400000 [00:39<00:01, 9935.76it/s] 97%|█████████▋| 389289/400000 [00:39<00:01, 10028.66it/s] 98%|█████████▊| 390294/400000 [00:39<00:00, 10000.91it/s] 98%|█████████▊| 391313/400000 [00:39<00:00, 10056.51it/s] 98%|█████████▊| 392320/400000 [00:39<00:00, 10001.26it/s] 98%|█████████▊| 393321/400000 [00:39<00:00, 9970.65it/s]  99%|█████████▊| 394346/400000 [00:39<00:00, 10052.21it/s] 99%|█████████▉| 395352/400000 [00:39<00:00, 9860.70it/s]  99%|█████████▉| 396423/400000 [00:39<00:00, 10099.10it/s] 99%|█████████▉| 397448/400000 [00:40<00:00, 10141.93it/s]100%|█████████▉| 398464/400000 [00:40<00:00, 9899.55it/s] 100%|█████████▉| 399472/400000 [00:40<00:00, 9952.53it/s]100%|█████████▉| 399999/400000 [00:40<00:00, 9923.11it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fc949fe2d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011185477390480314 	 Accuracy: 51
Train Epoch: 1 	 Loss: 0.010888606808257342 	 Accuracy: 68

  model saves at 68% accuracy 

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
2020-05-13 05:24:47.110391: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 05:24:47.114117: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095080000 Hz
2020-05-13 05:24:47.114799: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a5e0a63970 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 05:24:47.114810: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fc9540660b8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 7s - loss: 7.8123 - accuracy: 0.4905 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6398 - accuracy: 0.5017
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6084 - accuracy: 0.5038
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6487 - accuracy: 0.5012
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5965 - accuracy: 0.5046
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6302 - accuracy: 0.5024
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6411 - accuracy: 0.5017
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6467 - accuracy: 0.5013
11000/25000 [============>.................] - ETA: 3s - loss: 7.6374 - accuracy: 0.5019
12000/25000 [=============>................] - ETA: 2s - loss: 7.6423 - accuracy: 0.5016
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6348 - accuracy: 0.5021
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6458 - accuracy: 0.5014
15000/25000 [=================>............] - ETA: 2s - loss: 7.6676 - accuracy: 0.4999
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6772 - accuracy: 0.4993
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6856 - accuracy: 0.4988
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6879 - accuracy: 0.4986
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6852 - accuracy: 0.4988
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6996 - accuracy: 0.4979
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6834 - accuracy: 0.4989
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6959 - accuracy: 0.4981
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6946 - accuracy: 0.4982
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6807 - accuracy: 0.4991
25000/25000 [==============================] - 7s 261us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fc8aae2d710> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fc8ace50e48> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 960ms/step - loss: 1.2966 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.2521 - val_crf_viterbi_accuracy: 0.6533

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

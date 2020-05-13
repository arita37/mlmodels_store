
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f1c4f6bafd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 06:12:53.398997
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 06:12:53.406504
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 06:12:53.409710
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 06:12:53.413019
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f1c5b484470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 351148.9375
Epoch 2/10

1/1 [==============================] - 0s 104ms/step - loss: 218721.3594
Epoch 3/10

1/1 [==============================] - 0s 99ms/step - loss: 122765.4844
Epoch 4/10

1/1 [==============================] - 0s 94ms/step - loss: 63134.6055
Epoch 5/10

1/1 [==============================] - 0s 92ms/step - loss: 33724.8945
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 19999.0410
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 13069.4990
Epoch 8/10

1/1 [==============================] - 0s 95ms/step - loss: 9219.1436
Epoch 9/10

1/1 [==============================] - 0s 94ms/step - loss: 6914.6875
Epoch 10/10

1/1 [==============================] - 0s 93ms/step - loss: 5460.4199

  #### Inference Need return ypred, ytrue ######################### 
[[ 2.73259044e-01  1.06815081e+01  1.08079357e+01  9.03048134e+00
   9.91541576e+00  1.02954245e+01  7.89443207e+00  7.49054193e+00
   9.07242870e+00  8.72194099e+00  1.19922543e+01  8.04625607e+00
   1.11535234e+01  7.98284912e+00  1.18594475e+01  7.74780273e+00
   8.50200558e+00  1.00730953e+01  7.10305738e+00  8.01995087e+00
   9.14319706e+00  6.66474104e+00  9.32682610e+00  9.03772259e+00
   6.82172966e+00  6.54781199e+00  8.15167332e+00  9.78902626e+00
   8.01815987e+00  9.30777931e+00  7.09137011e+00  9.61827469e+00
   1.02995338e+01  9.60208321e+00  7.77484369e+00  9.69347095e+00
   7.25984716e+00  8.18925095e+00  8.51844597e+00  8.39353180e+00
   9.64524841e+00  8.39558506e+00  8.13451672e+00  9.06250095e+00
   9.42697811e+00  8.72238064e+00  1.07282829e+01  8.83946896e+00
   9.42356682e+00  7.50017834e+00  9.97416210e+00  8.11488247e+00
   1.02555943e+01  7.96438122e+00  6.70629025e+00  8.69616318e+00
   9.78854179e+00  1.05625467e+01  1.02132254e+01  8.92546272e+00
  -6.89595222e-01 -8.27795267e-02 -1.20569181e+00 -1.96871758e-01
  -4.98051643e-01 -3.51751387e-01  1.07902622e+00 -7.09053755e-01
  -1.13794231e+00 -3.03911775e-01 -1.01943147e+00 -1.21821272e+00
  -5.77533364e-01  3.24073762e-01 -9.87824917e-01  7.43391097e-01
  -8.45027626e-01  1.02219391e+00  9.92447138e-03 -5.23964047e-01
   1.26965475e+00 -5.18464744e-02  5.06332397e-01 -1.27119970e+00
  -2.66417027e-01  2.64664316e+00  5.39982617e-01 -1.20991302e+00
  -4.49056327e-02 -7.90236175e-01 -8.44587266e-01  1.37135673e+00
   2.52805054e-02  4.99113053e-01  1.61745179e+00  1.08859015e+00
   1.00905406e+00 -1.06706810e+00 -4.78430867e-01  3.68955016e-01
   2.69519836e-02 -1.04073238e+00 -5.13963521e-01 -1.22645482e-01
  -7.54184842e-01 -1.94615185e-01  2.22675204e-02  9.95157182e-01
  -2.90980041e-01 -4.83541280e-01 -6.88083351e-01 -2.22381264e-01
  -1.51456594e+00  3.72897863e-01  1.56956351e+00 -4.15568501e-01
   1.78483665e-01 -1.26961267e+00  1.28356671e+00 -1.14255369e+00
  -1.21612877e-01 -3.43700647e-01 -2.23958045e-01 -5.65034151e-02
   1.11538482e+00  1.21700454e+00 -1.23503256e+00  9.96359527e-01
   1.31124020e+00 -1.14464796e+00 -3.01763475e-01  3.96259934e-01
  -7.39201963e-01 -1.03887320e+00  1.17435443e+00  3.95864308e-01
   1.31986320e+00  1.15533996e+00 -2.40036428e-01  2.60651171e-01
   1.21179521e-01  1.07186854e+00 -7.31121898e-02  1.38239121e+00
   2.22028762e-01 -4.60552573e-01 -5.72732627e-01  1.67099094e+00
   9.40814137e-01 -2.79434800e-01  1.11210048e+00  5.18023968e-04
  -3.56274068e-01 -1.03671658e+00  1.19152832e+00  5.23824394e-01
  -1.81302771e-01 -7.27419257e-01  6.69972718e-01 -1.08518577e+00
  -7.30621293e-02 -8.22037756e-01 -2.40381575e+00  7.61103183e-02
   8.92479181e-01  1.36630905e+00  1.06928182e+00 -4.83597100e-01
  -5.24620056e-01  9.75650132e-01 -1.12206936e+00  1.21719706e+00
   2.81359851e-01  7.60070443e-01 -3.82556379e-01  1.25192690e+00
  -5.96224070e-01  7.89124370e-01  8.31867099e-01 -1.04790568e-01
   7.82580018e-01  8.96252346e+00  9.62480354e+00  1.08632259e+01
   8.60891342e+00  1.01345167e+01  1.06649046e+01  9.07684708e+00
   8.82042789e+00  8.30876827e+00  1.05457973e+01  1.06874886e+01
   8.83930111e+00  1.13711987e+01  8.03805733e+00  9.17757893e+00
   9.10028267e+00  1.03256254e+01  9.44437313e+00  8.64838696e+00
   1.09498825e+01  1.04617910e+01  8.06125355e+00  1.00761137e+01
   7.59655762e+00  8.68370628e+00  6.85546637e+00  8.90652657e+00
   9.10595894e+00  9.65889263e+00  8.96559811e+00  8.77257824e+00
   1.07102518e+01  8.74037075e+00  8.99386787e+00  9.10957146e+00
   9.69520569e+00  9.42493820e+00  9.36892605e+00  9.11148643e+00
   8.65175915e+00  9.75310516e+00  8.61590385e+00  7.83331585e+00
   1.10342369e+01  6.30672789e+00  1.04042282e+01  1.08912458e+01
   8.15577126e+00  9.75400066e+00  8.36493301e+00  1.00601892e+01
   1.05739632e+01  8.98473167e+00  9.32832050e+00  1.00021257e+01
   6.93114281e+00  8.35875416e+00  9.05904484e+00  1.09027920e+01
   2.46063066e+00  2.41934109e+00  1.56340182e+00  5.48660874e-01
   1.33168936e+00  7.91203856e-01  1.70113301e+00  3.79415095e-01
   6.42073572e-01  2.57497907e-01  1.58187103e+00  6.77564144e-01
   4.18614626e-01  6.10526323e-01  3.44441557e+00  1.82436466e-01
   1.06839585e+00  2.71247387e+00  1.77512801e+00  8.74365568e-01
   9.07826841e-01  1.01852953e-01  6.56714499e-01  1.25817823e+00
   2.10270905e+00  4.08841908e-01  1.53266668e-01  7.51145422e-01
   2.05178690e+00  1.85387182e+00  1.53482723e+00  1.78534389e-01
   1.30722296e+00  3.47306967e-01  2.24137366e-01  8.84280384e-01
   3.75868499e-01  7.62474179e-01  6.24655962e-01  2.17094445e+00
   1.87255919e-01  1.70078683e+00  3.99670124e-01  8.95273983e-01
   3.65656018e-01  3.14871120e+00  1.46163487e+00  1.51322103e+00
   3.73417556e-01  6.00438058e-01  1.49501562e+00  4.07613814e-01
   2.75264561e-01  7.83877134e-01  8.63170624e-02  6.51552796e-01
   6.19404614e-01  1.86477733e+00  2.56895494e+00  7.97935247e-01
   6.51315153e-01  8.50285292e-01  2.22768211e+00  6.24368846e-01
   5.13975322e-01  1.21540868e+00  1.58894205e+00  1.11406851e+00
   2.27487981e-01  1.16054678e+00  1.22950780e+00  1.93441057e+00
   2.30766487e+00  1.69289160e+00  1.66361141e+00  1.01002717e+00
   2.35015798e+00  1.23951018e-01  2.38736296e+00  4.32535529e-01
   1.62335122e+00  2.29705393e-01  1.90565228e+00  2.55635262e-01
   1.41366971e+00  2.05070639e+00  3.55459750e-01  2.09430933e-01
   1.20649576e+00  1.11064029e+00  3.58237934e+00  2.90360451e+00
   2.23308110e+00  2.86905646e-01  1.41502559e+00  1.69908464e-01
   7.28592753e-01  3.07895803e+00  4.08316433e-01  1.19675565e+00
   1.49391758e+00  1.20738006e+00  1.34809172e+00  2.27951598e+00
   2.55068660e-01  6.80077374e-01  1.34118795e+00  2.35689592e+00
   2.11272359e-01  8.46683741e-01  1.46495855e+00  1.42273593e+00
   2.04235744e+00  7.67102122e-01  1.11553454e+00  1.69453883e+00
   4.84519243e-01  6.14116490e-01  1.43844724e-01  6.03216290e-02
   1.04829969e+01 -9.76823235e+00 -5.88057756e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 06:13:02.093386
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    92.981
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 06:13:02.097060
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    8663.9
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 06:13:02.100299
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.9974
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 06:13:02.103356
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -774.913
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139759209121384
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139757999366832
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139757999367336
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139757999367840
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139757999368344
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139757999368848

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f1c57303f28> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.540579
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.505362
grad_step = 000002, loss = 0.477921
grad_step = 000003, loss = 0.448860
grad_step = 000004, loss = 0.415956
grad_step = 000005, loss = 0.379207
grad_step = 000006, loss = 0.344910
grad_step = 000007, loss = 0.328200
grad_step = 000008, loss = 0.325408
grad_step = 000009, loss = 0.303269
grad_step = 000010, loss = 0.273149
grad_step = 000011, loss = 0.250810
grad_step = 000012, loss = 0.236726
grad_step = 000013, loss = 0.228008
grad_step = 000014, loss = 0.222772
grad_step = 000015, loss = 0.214807
grad_step = 000016, loss = 0.203630
grad_step = 000017, loss = 0.192217
grad_step = 000018, loss = 0.179946
grad_step = 000019, loss = 0.167621
grad_step = 000020, loss = 0.156323
grad_step = 000021, loss = 0.147521
grad_step = 000022, loss = 0.141146
grad_step = 000023, loss = 0.134635
grad_step = 000024, loss = 0.126768
grad_step = 000025, loss = 0.117698
grad_step = 000026, loss = 0.108972
grad_step = 000027, loss = 0.100877
grad_step = 000028, loss = 0.093315
grad_step = 000029, loss = 0.086283
grad_step = 000030, loss = 0.079965
grad_step = 000031, loss = 0.074568
grad_step = 000032, loss = 0.069740
grad_step = 000033, loss = 0.064876
grad_step = 000034, loss = 0.060011
grad_step = 000035, loss = 0.055517
grad_step = 000036, loss = 0.051507
grad_step = 000037, loss = 0.047852
grad_step = 000038, loss = 0.044417
grad_step = 000039, loss = 0.041170
grad_step = 000040, loss = 0.038135
grad_step = 000041, loss = 0.035322
grad_step = 000042, loss = 0.032708
grad_step = 000043, loss = 0.030227
grad_step = 000044, loss = 0.027853
grad_step = 000045, loss = 0.025644
grad_step = 000046, loss = 0.023634
grad_step = 000047, loss = 0.021785
grad_step = 000048, loss = 0.020062
grad_step = 000049, loss = 0.018501
grad_step = 000050, loss = 0.017131
grad_step = 000051, loss = 0.015888
grad_step = 000052, loss = 0.014591
grad_step = 000053, loss = 0.013404
grad_step = 000054, loss = 0.012358
grad_step = 000055, loss = 0.011417
grad_step = 000056, loss = 0.010523
grad_step = 000057, loss = 0.009649
grad_step = 000058, loss = 0.008862
grad_step = 000059, loss = 0.008218
grad_step = 000060, loss = 0.007657
grad_step = 000061, loss = 0.007103
grad_step = 000062, loss = 0.006554
grad_step = 000063, loss = 0.006060
grad_step = 000064, loss = 0.005655
grad_step = 000065, loss = 0.005303
grad_step = 000066, loss = 0.004964
grad_step = 000067, loss = 0.004652
grad_step = 000068, loss = 0.004368
grad_step = 000069, loss = 0.004111
grad_step = 000070, loss = 0.003889
grad_step = 000071, loss = 0.003697
grad_step = 000072, loss = 0.003526
grad_step = 000073, loss = 0.003362
grad_step = 000074, loss = 0.003210
grad_step = 000075, loss = 0.003086
grad_step = 000076, loss = 0.002978
grad_step = 000077, loss = 0.002874
grad_step = 000078, loss = 0.002779
grad_step = 000079, loss = 0.002701
grad_step = 000080, loss = 0.002639
grad_step = 000081, loss = 0.002580
grad_step = 000082, loss = 0.002518
grad_step = 000083, loss = 0.002463
grad_step = 000084, loss = 0.002417
grad_step = 000085, loss = 0.002382
grad_step = 000086, loss = 0.002351
grad_step = 000087, loss = 0.002323
grad_step = 000088, loss = 0.002298
grad_step = 000089, loss = 0.002282
grad_step = 000090, loss = 0.002268
grad_step = 000091, loss = 0.002253
grad_step = 000092, loss = 0.002238
grad_step = 000093, loss = 0.002226
grad_step = 000094, loss = 0.002217
grad_step = 000095, loss = 0.002210
grad_step = 000096, loss = 0.002204
grad_step = 000097, loss = 0.002196
grad_step = 000098, loss = 0.002190
grad_step = 000099, loss = 0.002183
grad_step = 000100, loss = 0.002179
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002173
grad_step = 000102, loss = 0.002166
grad_step = 000103, loss = 0.002160
grad_step = 000104, loss = 0.002155
grad_step = 000105, loss = 0.002149
grad_step = 000106, loss = 0.002143
grad_step = 000107, loss = 0.002137
grad_step = 000108, loss = 0.002130
grad_step = 000109, loss = 0.002124
grad_step = 000110, loss = 0.002117
grad_step = 000111, loss = 0.002110
grad_step = 000112, loss = 0.002103
grad_step = 000113, loss = 0.002096
grad_step = 000114, loss = 0.002089
grad_step = 000115, loss = 0.002082
grad_step = 000116, loss = 0.002074
grad_step = 000117, loss = 0.002067
grad_step = 000118, loss = 0.002059
grad_step = 000119, loss = 0.002051
grad_step = 000120, loss = 0.002044
grad_step = 000121, loss = 0.002036
grad_step = 000122, loss = 0.002031
grad_step = 000123, loss = 0.002030
grad_step = 000124, loss = 0.002025
grad_step = 000125, loss = 0.002010
grad_step = 000126, loss = 0.002001
grad_step = 000127, loss = 0.002002
grad_step = 000128, loss = 0.001996
grad_step = 000129, loss = 0.001985
grad_step = 000130, loss = 0.001975
grad_step = 000131, loss = 0.001973
grad_step = 000132, loss = 0.001971
grad_step = 000133, loss = 0.001963
grad_step = 000134, loss = 0.001952
grad_step = 000135, loss = 0.001945
grad_step = 000136, loss = 0.001942
grad_step = 000137, loss = 0.001942
grad_step = 000138, loss = 0.001951
grad_step = 000139, loss = 0.001963
grad_step = 000140, loss = 0.001965
grad_step = 000141, loss = 0.001937
grad_step = 000142, loss = 0.001911
grad_step = 000143, loss = 0.001912
grad_step = 000144, loss = 0.001929
grad_step = 000145, loss = 0.001942
grad_step = 000146, loss = 0.001927
grad_step = 000147, loss = 0.001902
grad_step = 000148, loss = 0.001888
grad_step = 000149, loss = 0.001895
grad_step = 000150, loss = 0.001911
grad_step = 000151, loss = 0.001920
grad_step = 000152, loss = 0.001917
grad_step = 000153, loss = 0.001895
grad_step = 000154, loss = 0.001876
grad_step = 000155, loss = 0.001870
grad_step = 000156, loss = 0.001877
grad_step = 000157, loss = 0.001890
grad_step = 000158, loss = 0.001900
grad_step = 000159, loss = 0.001905
grad_step = 000160, loss = 0.001890
grad_step = 000161, loss = 0.001872
grad_step = 000162, loss = 0.001856
grad_step = 000163, loss = 0.001853
grad_step = 000164, loss = 0.001861
grad_step = 000165, loss = 0.001870
grad_step = 000166, loss = 0.001875
grad_step = 000167, loss = 0.001865
grad_step = 000168, loss = 0.001852
grad_step = 000169, loss = 0.001841
grad_step = 000170, loss = 0.001839
grad_step = 000171, loss = 0.001843
grad_step = 000172, loss = 0.001849
grad_step = 000173, loss = 0.001854
grad_step = 000174, loss = 0.001852
grad_step = 000175, loss = 0.001848
grad_step = 000176, loss = 0.001836
grad_step = 000177, loss = 0.001826
grad_step = 000178, loss = 0.001820
grad_step = 000179, loss = 0.001819
grad_step = 000180, loss = 0.001822
grad_step = 000181, loss = 0.001824
grad_step = 000182, loss = 0.001826
grad_step = 000183, loss = 0.001822
grad_step = 000184, loss = 0.001819
grad_step = 000185, loss = 0.001812
grad_step = 000186, loss = 0.001805
grad_step = 000187, loss = 0.001801
grad_step = 000188, loss = 0.001796
grad_step = 000189, loss = 0.001793
grad_step = 000190, loss = 0.001790
grad_step = 000191, loss = 0.001786
grad_step = 000192, loss = 0.001784
grad_step = 000193, loss = 0.001781
grad_step = 000194, loss = 0.001779
grad_step = 000195, loss = 0.001782
grad_step = 000196, loss = 0.001797
grad_step = 000197, loss = 0.001846
grad_step = 000198, loss = 0.001977
grad_step = 000199, loss = 0.001948
grad_step = 000200, loss = 0.001876
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001776
grad_step = 000202, loss = 0.001869
grad_step = 000203, loss = 0.001841
grad_step = 000204, loss = 0.001786
grad_step = 000205, loss = 0.001861
grad_step = 000206, loss = 0.001789
grad_step = 000207, loss = 0.001763
grad_step = 000208, loss = 0.001834
grad_step = 000209, loss = 0.001835
grad_step = 000210, loss = 0.001732
grad_step = 000211, loss = 0.001775
grad_step = 000212, loss = 0.001850
grad_step = 000213, loss = 0.001766
grad_step = 000214, loss = 0.001717
grad_step = 000215, loss = 0.001775
grad_step = 000216, loss = 0.001761
grad_step = 000217, loss = 0.001711
grad_step = 000218, loss = 0.001717
grad_step = 000219, loss = 0.001735
grad_step = 000220, loss = 0.001706
grad_step = 000221, loss = 0.001690
grad_step = 000222, loss = 0.001710
grad_step = 000223, loss = 0.001710
grad_step = 000224, loss = 0.001678
grad_step = 000225, loss = 0.001672
grad_step = 000226, loss = 0.001688
grad_step = 000227, loss = 0.001684
grad_step = 000228, loss = 0.001668
grad_step = 000229, loss = 0.001653
grad_step = 000230, loss = 0.001648
grad_step = 000231, loss = 0.001649
grad_step = 000232, loss = 0.001651
grad_step = 000233, loss = 0.001653
grad_step = 000234, loss = 0.001650
grad_step = 000235, loss = 0.001634
grad_step = 000236, loss = 0.001616
grad_step = 000237, loss = 0.001609
grad_step = 000238, loss = 0.001610
grad_step = 000239, loss = 0.001613
grad_step = 000240, loss = 0.001612
grad_step = 000241, loss = 0.001613
grad_step = 000242, loss = 0.001622
grad_step = 000243, loss = 0.001643
grad_step = 000244, loss = 0.001693
grad_step = 000245, loss = 0.001675
grad_step = 000246, loss = 0.001661
grad_step = 000247, loss = 0.001618
grad_step = 000248, loss = 0.001639
grad_step = 000249, loss = 0.001662
grad_step = 000250, loss = 0.001589
grad_step = 000251, loss = 0.001546
grad_step = 000252, loss = 0.001585
grad_step = 000253, loss = 0.001626
grad_step = 000254, loss = 0.001649
grad_step = 000255, loss = 0.001624
grad_step = 000256, loss = 0.001616
grad_step = 000257, loss = 0.001595
grad_step = 000258, loss = 0.001567
grad_step = 000259, loss = 0.001550
grad_step = 000260, loss = 0.001553
grad_step = 000261, loss = 0.001571
grad_step = 000262, loss = 0.001564
grad_step = 000263, loss = 0.001520
grad_step = 000264, loss = 0.001506
grad_step = 000265, loss = 0.001530
grad_step = 000266, loss = 0.001553
grad_step = 000267, loss = 0.001538
grad_step = 000268, loss = 0.001504
grad_step = 000269, loss = 0.001490
grad_step = 000270, loss = 0.001502
grad_step = 000271, loss = 0.001508
grad_step = 000272, loss = 0.001499
grad_step = 000273, loss = 0.001478
grad_step = 000274, loss = 0.001471
grad_step = 000275, loss = 0.001480
grad_step = 000276, loss = 0.001506
grad_step = 000277, loss = 0.001531
grad_step = 000278, loss = 0.001562
grad_step = 000279, loss = 0.001543
grad_step = 000280, loss = 0.001517
grad_step = 000281, loss = 0.001461
grad_step = 000282, loss = 0.001430
grad_step = 000283, loss = 0.001427
grad_step = 000284, loss = 0.001445
grad_step = 000285, loss = 0.001471
grad_step = 000286, loss = 0.001490
grad_step = 000287, loss = 0.001513
grad_step = 000288, loss = 0.001564
grad_step = 000289, loss = 0.001502
grad_step = 000290, loss = 0.001471
grad_step = 000291, loss = 0.001462
grad_step = 000292, loss = 0.001486
grad_step = 000293, loss = 0.001460
grad_step = 000294, loss = 0.001412
grad_step = 000295, loss = 0.001413
grad_step = 000296, loss = 0.001446
grad_step = 000297, loss = 0.001470
grad_step = 000298, loss = 0.001418
grad_step = 000299, loss = 0.001381
grad_step = 000300, loss = 0.001394
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001423
grad_step = 000302, loss = 0.001434
grad_step = 000303, loss = 0.001392
grad_step = 000304, loss = 0.001367
grad_step = 000305, loss = 0.001387
grad_step = 000306, loss = 0.001406
grad_step = 000307, loss = 0.001412
grad_step = 000308, loss = 0.001373
grad_step = 000309, loss = 0.001354
grad_step = 000310, loss = 0.001384
grad_step = 000311, loss = 0.001395
grad_step = 000312, loss = 0.001398
grad_step = 000313, loss = 0.001379
grad_step = 000314, loss = 0.001404
grad_step = 000315, loss = 0.001560
grad_step = 000316, loss = 0.001638
grad_step = 000317, loss = 0.001706
grad_step = 000318, loss = 0.001382
grad_step = 000319, loss = 0.001420
grad_step = 000320, loss = 0.001653
grad_step = 000321, loss = 0.001422
grad_step = 000322, loss = 0.001463
grad_step = 000323, loss = 0.001503
grad_step = 000324, loss = 0.001431
grad_step = 000325, loss = 0.001493
grad_step = 000326, loss = 0.001338
grad_step = 000327, loss = 0.001458
grad_step = 000328, loss = 0.001428
grad_step = 000329, loss = 0.001391
grad_step = 000330, loss = 0.001403
grad_step = 000331, loss = 0.001426
grad_step = 000332, loss = 0.001482
grad_step = 000333, loss = 0.001336
grad_step = 000334, loss = 0.001342
grad_step = 000335, loss = 0.001410
grad_step = 000336, loss = 0.001365
grad_step = 000337, loss = 0.001320
grad_step = 000338, loss = 0.001312
grad_step = 000339, loss = 0.001341
grad_step = 000340, loss = 0.001310
grad_step = 000341, loss = 0.001264
grad_step = 000342, loss = 0.001305
grad_step = 000343, loss = 0.001296
grad_step = 000344, loss = 0.001266
grad_step = 000345, loss = 0.001261
grad_step = 000346, loss = 0.001260
grad_step = 000347, loss = 0.001276
grad_step = 000348, loss = 0.001272
grad_step = 000349, loss = 0.001240
grad_step = 000350, loss = 0.001229
grad_step = 000351, loss = 0.001237
grad_step = 000352, loss = 0.001235
grad_step = 000353, loss = 0.001237
grad_step = 000354, loss = 0.001226
grad_step = 000355, loss = 0.001206
grad_step = 000356, loss = 0.001201
grad_step = 000357, loss = 0.001203
grad_step = 000358, loss = 0.001200
grad_step = 000359, loss = 0.001201
grad_step = 000360, loss = 0.001193
grad_step = 000361, loss = 0.001180
grad_step = 000362, loss = 0.001171
grad_step = 000363, loss = 0.001166
grad_step = 000364, loss = 0.001159
grad_step = 000365, loss = 0.001157
grad_step = 000366, loss = 0.001158
grad_step = 000367, loss = 0.001154
grad_step = 000368, loss = 0.001154
grad_step = 000369, loss = 0.001153
grad_step = 000370, loss = 0.001157
grad_step = 000371, loss = 0.001155
grad_step = 000372, loss = 0.001166
grad_step = 000373, loss = 0.001161
grad_step = 000374, loss = 0.001166
grad_step = 000375, loss = 0.001141
grad_step = 000376, loss = 0.001124
grad_step = 000377, loss = 0.001100
grad_step = 000378, loss = 0.001083
grad_step = 000379, loss = 0.001078
grad_step = 000380, loss = 0.001082
grad_step = 000381, loss = 0.001095
grad_step = 000382, loss = 0.001114
grad_step = 000383, loss = 0.001174
grad_step = 000384, loss = 0.001218
grad_step = 000385, loss = 0.001345
grad_step = 000386, loss = 0.001249
grad_step = 000387, loss = 0.001190
grad_step = 000388, loss = 0.001058
grad_step = 000389, loss = 0.001039
grad_step = 000390, loss = 0.001109
grad_step = 000391, loss = 0.001109
grad_step = 000392, loss = 0.001058
grad_step = 000393, loss = 0.000997
grad_step = 000394, loss = 0.001019
grad_step = 000395, loss = 0.001073
grad_step = 000396, loss = 0.001058
grad_step = 000397, loss = 0.001037
grad_step = 000398, loss = 0.000987
grad_step = 000399, loss = 0.000955
grad_step = 000400, loss = 0.000959
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000981
grad_step = 000402, loss = 0.001008
grad_step = 000403, loss = 0.001007
grad_step = 000404, loss = 0.001024
grad_step = 000405, loss = 0.000996
grad_step = 000406, loss = 0.000970
grad_step = 000407, loss = 0.000931
grad_step = 000408, loss = 0.000907
grad_step = 000409, loss = 0.000887
grad_step = 000410, loss = 0.000879
grad_step = 000411, loss = 0.000885
grad_step = 000412, loss = 0.000897
grad_step = 000413, loss = 0.000927
grad_step = 000414, loss = 0.000954
grad_step = 000415, loss = 0.001034
grad_step = 000416, loss = 0.001048
grad_step = 000417, loss = 0.001105
grad_step = 000418, loss = 0.000963
grad_step = 000419, loss = 0.000863
grad_step = 000420, loss = 0.000814
grad_step = 000421, loss = 0.000844
grad_step = 000422, loss = 0.000905
grad_step = 000423, loss = 0.000876
grad_step = 000424, loss = 0.000824
grad_step = 000425, loss = 0.000780
grad_step = 000426, loss = 0.000788
grad_step = 000427, loss = 0.000824
grad_step = 000428, loss = 0.000828
grad_step = 000429, loss = 0.000827
grad_step = 000430, loss = 0.000784
grad_step = 000431, loss = 0.000753
grad_step = 000432, loss = 0.000731
grad_step = 000433, loss = 0.000727
grad_step = 000434, loss = 0.000735
grad_step = 000435, loss = 0.000746
grad_step = 000436, loss = 0.000773
grad_step = 000437, loss = 0.000786
grad_step = 000438, loss = 0.000821
grad_step = 000439, loss = 0.000793
grad_step = 000440, loss = 0.000781
grad_step = 000441, loss = 0.000719
grad_step = 000442, loss = 0.000680
grad_step = 000443, loss = 0.000662
grad_step = 000444, loss = 0.000668
grad_step = 000445, loss = 0.000687
grad_step = 000446, loss = 0.000695
grad_step = 000447, loss = 0.000700
grad_step = 000448, loss = 0.000674
grad_step = 000449, loss = 0.000653
grad_step = 000450, loss = 0.000631
grad_step = 000451, loss = 0.000616
grad_step = 000452, loss = 0.000608
grad_step = 000453, loss = 0.000606
grad_step = 000454, loss = 0.000609
grad_step = 000455, loss = 0.000615
grad_step = 000456, loss = 0.000636
grad_step = 000457, loss = 0.000651
grad_step = 000458, loss = 0.000705
grad_step = 000459, loss = 0.000699
grad_step = 000460, loss = 0.000746
grad_step = 000461, loss = 0.000660
grad_step = 000462, loss = 0.000614
grad_step = 000463, loss = 0.000563
grad_step = 000464, loss = 0.000548
grad_step = 000465, loss = 0.000559
grad_step = 000466, loss = 0.000579
grad_step = 000467, loss = 0.000611
grad_step = 000468, loss = 0.000592
grad_step = 000469, loss = 0.000578
grad_step = 000470, loss = 0.000539
grad_step = 000471, loss = 0.000519
grad_step = 000472, loss = 0.000514
grad_step = 000473, loss = 0.000521
grad_step = 000474, loss = 0.000536
grad_step = 000475, loss = 0.000536
grad_step = 000476, loss = 0.000538
grad_step = 000477, loss = 0.000520
grad_step = 000478, loss = 0.000509
grad_step = 000479, loss = 0.000492
grad_step = 000480, loss = 0.000482
grad_step = 000481, loss = 0.000477
grad_step = 000482, loss = 0.000476
grad_step = 000483, loss = 0.000479
grad_step = 000484, loss = 0.000481
grad_step = 000485, loss = 0.000487
grad_step = 000486, loss = 0.000487
grad_step = 000487, loss = 0.000498
grad_step = 000488, loss = 0.000494
grad_step = 000489, loss = 0.000503
grad_step = 000490, loss = 0.000493
grad_step = 000491, loss = 0.000497
grad_step = 000492, loss = 0.000480
grad_step = 000493, loss = 0.000474
grad_step = 000494, loss = 0.000455
grad_step = 000495, loss = 0.000444
grad_step = 000496, loss = 0.000432
grad_step = 000497, loss = 0.000426
grad_step = 000498, loss = 0.000423
grad_step = 000499, loss = 0.000423
grad_step = 000500, loss = 0.000425
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000428
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

  date_run                              2020-05-13 06:13:20.075563
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.232799
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 06:13:20.081109
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.135833
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 06:13:20.088492
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.133163
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 06:13:20.093348
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.06403
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
0   2020-05-13 06:12:53.398997  ...    mean_absolute_error
1   2020-05-13 06:12:53.406504  ...     mean_squared_error
2   2020-05-13 06:12:53.409710  ...  median_absolute_error
3   2020-05-13 06:12:53.413019  ...               r2_score
4   2020-05-13 06:13:02.093386  ...    mean_absolute_error
5   2020-05-13 06:13:02.097060  ...     mean_squared_error
6   2020-05-13 06:13:02.100299  ...  median_absolute_error
7   2020-05-13 06:13:02.103356  ...               r2_score
8   2020-05-13 06:13:20.075563  ...    mean_absolute_error
9   2020-05-13 06:13:20.081109  ...     mean_squared_error
10  2020-05-13 06:13:20.088492  ...  median_absolute_error
11  2020-05-13 06:13:20.093348  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd150d32d30> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  1%|          | 98304/9912422 [00:00<00:12, 788178.87it/s] 11%|█         | 1114112/9912422 [00:00<00:08, 1089454.81it/s] 63%|██████▎   | 6234112/9912422 [00:00<00:02, 1542137.06it/s]9920512it [00:00, 18438898.82it/s]                            
0it [00:00, ?it/s]32768it [00:00, 378539.41it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:01<?, ?it/s]  5%|▌         | 90112/1648877 [00:02<00:01, 889779.84it/s] 35%|███▍      | 573440/1648877 [00:02<00:00, 1177928.89it/s]1654784it [00:02, 745479.21it/s]                             
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 31559.81it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd1036eceb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd102d1d0f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd1036eceb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd102c73128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd1004ae518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd100499780> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd1036eceb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd102c30748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd1004ae518> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fd102aeb588> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f420d601240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=7a5642676dcf190e30811b297b370f60f6de56dab5c2008a8024f006c32307cf
  Stored in directory: /tmp/pip-ephem-wheel-cache-e4q6c7eq/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f420376b080> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1310720/17464789 [=>............................] - ETA: 0s
 4284416/17464789 [======>.......................] - ETA: 0s
 7766016/17464789 [============>.................] - ETA: 0s
11845632/17464789 [===================>..........] - ETA: 0s
16416768/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 06:14:49.431981: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 06:14:49.435919: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-13 06:14:49.436067: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ad9d6bdc40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 06:14:49.436082: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4980 - accuracy: 0.5110
 2000/25000 [=>............................] - ETA: 8s - loss: 7.3830 - accuracy: 0.5185 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6257 - accuracy: 0.5027
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6206 - accuracy: 0.5030
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5961 - accuracy: 0.5046
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6232 - accuracy: 0.5028
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6294 - accuracy: 0.5024
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6820 - accuracy: 0.4990
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6344 - accuracy: 0.5021
11000/25000 [============>.................] - ETA: 3s - loss: 7.6206 - accuracy: 0.5030
12000/25000 [=============>................] - ETA: 3s - loss: 7.6321 - accuracy: 0.5023
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6100 - accuracy: 0.5037
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6425 - accuracy: 0.5016
15000/25000 [=================>............] - ETA: 2s - loss: 7.6288 - accuracy: 0.5025
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6015 - accuracy: 0.5042
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6044 - accuracy: 0.5041
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6326 - accuracy: 0.5022
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6263 - accuracy: 0.5026
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6260 - accuracy: 0.5027
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6294 - accuracy: 0.5024
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6415 - accuracy: 0.5016
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6480 - accuracy: 0.5012
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 7s 274us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 06:15:02.807367
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 06:15:02.807367  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:01<44:26:15, 5.39kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<31:20:16, 7.64kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<21:59:15, 10.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<15:23:35, 15.5kB/s].vector_cache/glove.6B.zip:   0%|          | 3.60M/862M [00:02<10:44:40, 22.2kB/s].vector_cache/glove.6B.zip:   1%|          | 8.04M/862M [00:02<7:29:01, 31.7kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 12.8M/862M [00:02<5:12:40, 45.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.6M/862M [00:02<3:37:29, 64.6kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.3M/862M [00:02<2:31:18, 92.3kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.2M/862M [00:02<1:45:22, 132kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 32.8M/862M [00:02<1:13:34, 188kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:02<51:14, 268kB/s]  .vector_cache/glove.6B.zip:   5%|▍         | 41.4M/862M [00:02<35:52, 381kB/s].vector_cache/glove.6B.zip:   5%|▌         | 45.7M/862M [00:03<25:04, 543kB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.0M/862M [00:03<17:33, 771kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:03<12:34, 1.07MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:05<10:40, 1.26MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:05<11:19, 1.19MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:05<08:50, 1.52MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.7M/862M [00:05<06:24, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:07<10:15, 1.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:07<08:48, 1.52MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:07<06:30, 2.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:09<07:15, 1.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:09<08:02, 1.65MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:09<06:20, 2.10MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.7M/862M [00:09<04:34, 2.89MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:11<19:47, 669kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:11<15:11, 871kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.0M/862M [00:11<10:54, 1.21MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:13<10:41, 1.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:13<10:03, 1.31MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:13<07:36, 1.73MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.0M/862M [00:13<05:32, 2.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:15<08:43, 1.50MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:15<08:54, 1.47MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:15<06:49, 1.92MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:15<04:58, 2.62MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:17<08:17, 1.57MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:17<07:09, 1.82MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.4M/862M [00:17<05:19, 2.44MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:19<06:45, 1.92MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:19<07:21, 1.76MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:19<05:44, 2.25MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.5M/862M [00:19<04:08, 3.12MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:21<39:04, 330kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:21<28:39, 450kB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:21<20:20, 632kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:23<17:12, 745kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:23<14:39, 875kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:23<10:49, 1.18MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.1M/862M [00:23<07:43, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:25<12:51, 993kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:25<10:19, 1.24MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.8M/862M [00:25<07:32, 1.69MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<08:13, 1.54MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<08:28, 1.50MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<06:35, 1.92MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<04:44, 2.66MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<48:50, 258kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<35:28, 355kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<25:06, 501kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<20:26, 614kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:31<16:51, 744kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<12:20, 1.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<08:45, 1.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<18:24, 678kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<14:08, 882kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:33<10:12, 1.22MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:34<10:02, 1.24MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<09:32, 1.30MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<07:18, 1.70MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:05, 1.74MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:13, 1.98MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:37<04:39, 2.64MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:07, 2.00MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<06:53, 1.78MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:22, 2.28MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<03:53, 3.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<13:29, 905kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<10:42, 1.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:47, 1.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<08:17, 1.46MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<07:04, 1.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:15, 2.30MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<06:29, 1.86MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<07:00, 1.72MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<05:32, 2.18MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<04:00, 2.99MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:46<1:28:24, 136kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<1:03:03, 190kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<44:21, 270kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<33:45, 353kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<26:05, 457kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<18:52, 631kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<13:18, 891kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:50<1:33:42, 127kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<1:06:48, 177kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<46:58, 252kB/s]  .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:52<35:31, 332kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<27:16, 432kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<19:40, 599kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<15:36, 751kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<12:09, 964kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<08:48, 1.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<08:50, 1.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<08:42, 1.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<06:42, 1.74MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<04:49, 2.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<1:25:56, 135kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<1:01:20, 189kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<43:09, 268kB/s]  .vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<32:47, 351kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<24:07, 477kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<17:08, 670kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<14:40, 781kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:02<12:35, 909kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<09:20, 1.22MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<06:37, 1.72MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:04<56:10, 203kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<40:27, 281kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<28:32, 398kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:06<22:34, 501kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:06<18:00, 628kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<13:11, 857kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<09:17, 1.21MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:08<1:12:03, 156kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<51:35, 218kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<36:18, 309kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<27:57, 400kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<20:45, 538kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<14:46, 755kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<12:52, 863kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<10:10, 1.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:21, 1.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<07:40, 1.44MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<06:32, 1.69MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<04:51, 2.26MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<05:51, 1.87MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<06:27, 1.70MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<05:05, 2.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<03:42, 2.95MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<28:30, 382kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<21:05, 517kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<15:00, 724kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<12:59, 834kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:20<11:26, 947kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<08:28, 1.28MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<06:08, 1.76MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<07:15, 1.48MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<06:14, 1.72MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<04:38, 2.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<05:41, 1.88MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<06:17, 1.70MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:24<04:55, 2.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<03:32, 3.00MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<15:12, 698kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<11:44, 904kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:26<08:28, 1.25MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<08:22, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<08:10, 1.29MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:28<06:10, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<04:31, 2.32MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<06:10, 1.70MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<05:26, 1.93MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<04:04, 2.57MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<05:15, 1.98MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:31<06:03, 1.72MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:32<04:39, 2.23MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:32<03:34, 2.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:48, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:33<04:28, 2.32MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<03:22, 3.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:44, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:35<04:26, 2.32MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<03:20, 3.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<04:37, 2.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:37<05:24, 1.89MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:19, 2.36MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<03:08, 3.22MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:39<32:40, 311kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:39<23:56, 424kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<16:58, 596kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<14:07, 713kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:41<11:57, 842kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:41<08:53, 1.13MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<06:19, 1.58MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<1:15:51, 132kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:43<54:08, 185kB/s]  .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<38:01, 262kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<28:55, 344kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:45<22:22, 444kB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<16:09, 614kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<11:23, 867kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<39:15, 251kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<28:29, 346kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<20:09, 488kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<16:16, 602kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:49<13:29, 726kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<09:57, 983kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<07:02, 1.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:51<27:24, 355kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:51<20:10, 482kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<14:17, 678kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:53<12:14, 790kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<10:36, 910kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<07:51, 1.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<05:39, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<06:55, 1.39MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:55<05:50, 1.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<04:17, 2.23MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<05:12, 1.83MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:57<05:39, 1.68MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<04:27, 2.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<03:15, 2.90MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<05:49, 1.63MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<05:03, 1.87MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<03:46, 2.50MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<04:48, 1.95MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:01<04:20, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<03:14, 2.89MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<04:27, 2.09MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:03<05:05, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:03<03:59, 2.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<02:53, 3.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<08:01, 1.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:05<06:35, 1.40MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:48, 1.92MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<05:28, 1.68MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<05:47, 1.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<04:28, 2.05MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<03:15, 2.81MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:09<05:59, 1.52MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:09<05:09, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<03:50, 2.37MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:11<04:45, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<05:10, 1.75MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:11<04:05, 2.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<02:57, 3.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<1:15:20, 119kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:13<53:37, 167kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<37:40, 237kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<28:19, 315kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:15<21:44, 410kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<15:40, 568kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<11:00, 803kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:17<29:52, 296kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:17<21:52, 404kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<15:29, 568kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<12:50, 683kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<09:52, 888kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<07:06, 1.23MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<07:00, 1.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:21<06:40, 1.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:21<05:06, 1.70MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:38, 2.37MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:22<17:50, 484kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:23<13:23, 645kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<09:32, 902kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:24<08:37, 993kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:25<07:50, 1.09MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:25<05:56, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:13, 2.01MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:26<13:17, 640kB/s] .vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:27<10:10, 834kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<07:19, 1.16MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:28<07:03, 1.19MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<06:43, 1.25MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<05:06, 1.65MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:40, 2.28MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<21:06, 396kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<15:39, 534kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:31<11:07, 749kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<09:37, 862kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<08:25, 984kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:33<06:18, 1.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<05:43, 1.44MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<04:51, 1.69MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<03:36, 2.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<04:24, 1.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:36<03:57, 2.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:37<02:58, 2.73MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<03:55, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:38<04:24, 1.83MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:30, 2.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<02:32, 3.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<58:54, 136kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<42:02, 191kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<29:32, 270kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<22:25, 354kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:42<17:22, 457kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<12:30, 634kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<08:47, 897kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<13:32, 582kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<10:18, 764kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<07:23, 1.06MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<06:57, 1.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:46<05:39, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:08, 1.87MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<04:42, 1.65MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:48<04:56, 1.57MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<03:51, 2.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<02:46, 2.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<56:47, 135kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:50<40:31, 189kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<28:28, 268kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<21:35, 352kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<16:42, 455kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<12:01, 632kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<08:28, 891kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<09:21, 806kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:54<07:20, 1.03MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<05:19, 1.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<05:25, 1.38MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<05:19, 1.40MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<04:02, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<02:54, 2.54MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:58<06:27, 1.15MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:58<05:16, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<03:52, 1.91MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<04:24, 1.66MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:00<04:34, 1.60MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<03:35, 2.04MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<02:34, 2.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:02<20:01, 363kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:02<14:45, 492kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<10:28, 690kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<08:58, 801kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:04<07:00, 1.03MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<05:04, 1.41MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<05:12, 1.37MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:06<05:09, 1.38MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:06<03:58, 1.79MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<02:50, 2.48MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<52:27, 135kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<37:25, 188kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<26:16, 267kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<19:55, 351kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:10<15:25, 453kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:10<11:09, 625kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<07:50, 883kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<22:06, 313kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<16:11, 427kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<11:28, 601kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<09:34, 716kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:14<07:25, 922kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<05:20, 1.28MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<05:16, 1.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<05:04, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:16<03:51, 1.75MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:45, 2.44MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<12:54, 520kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:18<09:43, 690kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:18<06:55, 964kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:19<06:22, 1.04MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<05:48, 1.14MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:20<04:24, 1.50MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:08, 2.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<55:56, 118kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<39:47, 165kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:22<27:55, 234kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<20:56, 311kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<15:58, 407kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:24<11:27, 566kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<08:02, 802kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<13:07, 490kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<09:51, 653kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<07:02, 910kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<06:23, 998kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<05:07, 1.24MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<03:44, 1.69MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<04:03, 1.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<04:07, 1.53MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:12, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<02:17, 2.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<34:21, 181kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<24:40, 252kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<17:21, 357kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:33<13:31, 456kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<10:06, 610kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<07:12, 852kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<06:24, 950kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<05:43, 1.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<04:16, 1.42MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<03:02, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:37<13:35, 443kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<10:07, 594kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<07:11, 833kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<06:23, 933kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<05:45, 1.04MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:39<04:19, 1.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<03:05, 1.91MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<16:09, 365kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:41<11:55, 494kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<08:26, 694kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:43<07:13, 807kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:43<05:39, 1.03MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:05, 1.42MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<04:11, 1.37MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:45<04:06, 1.40MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:07, 1.83MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:15, 2.53MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<04:29, 1.26MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:47<03:44, 1.52MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:45, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<03:12, 1.75MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<02:49, 1.98MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:06, 2.64MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<02:46, 2.00MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:51<03:07, 1.78MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:51<02:26, 2.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<01:46, 3.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:53<03:49, 1.43MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:53<03:15, 1.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:24, 2.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:55<02:55, 1.85MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<03:11, 1.69MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:55<02:28, 2.18MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<01:47, 3.00MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<04:50, 1.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:57<03:56, 1.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<02:52, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<03:13, 1.63MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:59<02:48, 1.88MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:05, 2.51MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<02:40, 1.95MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:01<02:58, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:01<02:21, 2.21MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<01:42, 3.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<13:25, 383kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:03<09:54, 518kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<07:02, 726kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<06:04, 835kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:05<05:19, 951kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:05<03:56, 1.28MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:48, 1.78MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<04:13, 1.18MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:07<03:28, 1.44MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:07<02:32, 1.95MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<02:54, 1.69MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:09<02:33, 1.92MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:09<01:54, 2.56MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<02:26, 1.99MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:11<02:44, 1.77MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:11<02:08, 2.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:32, 3.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<06:50, 701kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<05:17, 906kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:13<03:48, 1.25MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:43, 1.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<03:37, 1.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:15<02:46, 1.69MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<01:59, 2.35MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<15:22, 303kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<11:14, 414kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<07:56, 582kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<06:34, 697kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<05:36, 818kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<04:07, 1.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:55, 1.55MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:58, 1.14MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:20<03:15, 1.38MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:22, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:40, 1.66MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:22<02:50, 1.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:12, 2.00MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<01:35, 2.76MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:24<15:00, 292kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:24<10:56, 400kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<07:44, 563kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<05:25, 797kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<20:54, 206kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:26<15:32, 277kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:26<11:05, 388kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<07:43, 550kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:28<18:09, 234kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:28<13:08, 323kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<09:13, 457kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<06:28, 647kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:30<20:01, 209kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<14:54, 280kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:30<10:35, 393kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:30<07:25, 557kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<06:47, 606kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<05:11, 791kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<03:43, 1.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<03:28, 1.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:34<03:17, 1.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:30, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:34<01:47, 2.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<13:15, 300kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<09:40, 410kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<06:50, 576kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<05:38, 692kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:38<04:20, 897kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<03:07, 1.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<03:03, 1.25MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:40<02:32, 1.51MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:51, 2.04MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<02:10, 1.73MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:42<02:17, 1.65MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:42<01:47, 2.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:17, 2.88MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:44<03:35, 1.03MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<02:53, 1.27MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<02:06, 1.74MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:46<02:18, 1.57MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<02:22, 1.52MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:46<01:50, 1.97MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:18, 2.72MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<04:21, 818kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:48<03:25, 1.04MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<02:27, 1.44MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<02:31, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:50<02:28, 1.41MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:50<01:54, 1.82MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<01:21, 2.53MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<14:13, 241kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:52<10:17, 332kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<07:14, 469kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<05:46, 580kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<04:23, 762kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<03:08, 1.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<02:56, 1.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:56<02:23, 1.37MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:44, 1.86MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<01:57, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:58<01:42, 1.89MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:58<01:15, 2.54MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<01:36, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:00<01:27, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:00<01:05, 2.86MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<01:28, 2.09MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:02<01:21, 2.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:02<01:01, 2.97MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<01:24, 2.15MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<01:37, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:04<01:17, 2.33MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<00:55, 3.20MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<11:16, 261kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<08:11, 358kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<05:45, 506kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<04:37, 621kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<03:51, 745kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<02:48, 1.02MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:59, 1.42MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<02:41, 1.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<02:10, 1.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:34, 1.77MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:43, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<01:48, 1.52MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<01:23, 1.95MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<00:59, 2.68MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<09:08, 292kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<06:39, 400kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<04:41, 563kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<03:50, 677kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<03:14, 802kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<02:23, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:40, 1.51MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:17<07:28, 339kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<05:29, 460kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<03:52, 646kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<03:14, 761kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<02:46, 885kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<02:03, 1.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<01:26, 1.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:21<13:31, 177kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<09:41, 247kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<06:46, 349kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<05:11, 448kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<04:08, 562kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<02:59, 774kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<02:05, 1.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<02:21, 957kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<01:52, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:21, 1.64MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:26, 1.52MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:27<01:14, 1.76MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<00:54, 2.36MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:07, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:29<01:00, 2.10MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<00:44, 2.81MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<00:58, 2.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<01:07, 1.83MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<00:53, 2.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<00:37, 3.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<14:36, 136kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:33<10:24, 190kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<07:14, 270kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<05:24, 354kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:35<04:11, 456kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<03:00, 632kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<02:05, 893kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:37<02:28, 746kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:55, 960kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:22, 1.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:39<01:21, 1.31MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<01:08, 1.56MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:49, 2.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:57, 1.79MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:41<00:50, 2.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:37, 2.69MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:48, 2.03MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:43<00:54, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:42, 2.29MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:31, 3.09MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:51, 1.83MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:45<00:45, 2.05MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:34, 2.72MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:44, 2.04MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<00:49, 1.80MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<00:39, 2.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:27, 3.11MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<03:46, 381kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:49<02:46, 516kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<01:56, 723kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:38, 832kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:51<01:17, 1.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:55, 1.45MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:55, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:53<00:54, 1.42MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:53<00:41, 1.86MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:29, 2.54MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:44, 1.64MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:38, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:28, 2.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:35, 1.97MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:31, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:23, 2.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:31, 2.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:35, 1.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 798M/862M [05:59<00:27, 2.38MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:19, 3.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:51, 1.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:42, 1.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:30, 1.96MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:33, 1.70MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:35, 1.61MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:27, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<00:19, 2.83MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:31, 1.71MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:27, 1.95MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:19, 2.60MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:24, 2.00MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:27, 1.76MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:21, 2.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:14, 3.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<02:25, 310kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:08<01:45, 423kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<01:12, 596kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:57, 712kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:10<00:48, 836kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:35, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:23, 1.58MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:12<00:32, 1.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<00:26, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:18, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:19, 1.66MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:14<00:17, 1.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:12, 2.52MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:14, 1.96MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:16, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:12, 2.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:08, 3.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<01:03, 384kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:18<00:46, 518kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:31, 726kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:24, 837kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:18, 1.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:12, 1.46MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:11, 1.41MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:22<00:09, 1.65MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:06, 2.23MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:06, 1.84MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:07, 1.69MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:05, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:03, 2.92MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:04, 1.81MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:26<00:03, 2.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:02, 2.74MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:01, 2.05MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:28<00:01, 2.24MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:00, 2.98MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 855/400000 [00:00<00:46, 8547.38it/s]  0%|          | 1716/400000 [00:00<00:46, 8564.85it/s]  1%|          | 2574/400000 [00:00<00:46, 8568.77it/s]  1%|          | 3443/400000 [00:00<00:46, 8604.36it/s]  1%|          | 4310/400000 [00:00<00:45, 8622.06it/s]  1%|▏         | 5173/400000 [00:00<00:45, 8624.05it/s]  2%|▏         | 6032/400000 [00:00<00:45, 8613.26it/s]  2%|▏         | 6893/400000 [00:00<00:45, 8611.73it/s]  2%|▏         | 7758/400000 [00:00<00:45, 8620.98it/s]  2%|▏         | 8622/400000 [00:01<00:45, 8625.58it/s]  2%|▏         | 9491/400000 [00:01<00:45, 8643.84it/s]  3%|▎         | 10344/400000 [00:01<00:45, 8606.64it/s]  3%|▎         | 11208/400000 [00:01<00:45, 8614.11it/s]  3%|▎         | 12068/400000 [00:01<00:45, 8609.46it/s]  3%|▎         | 12928/400000 [00:01<00:44, 8606.01it/s]  3%|▎         | 13785/400000 [00:01<00:45, 8503.76it/s]  4%|▎         | 14651/400000 [00:01<00:45, 8547.99it/s]  4%|▍         | 15505/400000 [00:01<00:44, 8545.14it/s]  4%|▍         | 16369/400000 [00:01<00:44, 8570.53it/s]  4%|▍         | 17226/400000 [00:02<00:44, 8566.63it/s]  5%|▍         | 18095/400000 [00:02<00:44, 8601.35it/s]  5%|▍         | 18967/400000 [00:02<00:44, 8633.82it/s]  5%|▍         | 19831/400000 [00:02<00:44, 8503.43it/s]  5%|▌         | 20682/400000 [00:02<00:45, 8425.69it/s]  5%|▌         | 21553/400000 [00:02<00:44, 8507.94it/s]  6%|▌         | 22418/400000 [00:02<00:44, 8547.09it/s]  6%|▌         | 23281/400000 [00:02<00:43, 8569.17it/s]  6%|▌         | 24139/400000 [00:02<00:44, 8490.48it/s]  6%|▋         | 25006/400000 [00:02<00:43, 8541.25it/s]  6%|▋         | 25877/400000 [00:03<00:43, 8590.60it/s]  7%|▋         | 26744/400000 [00:03<00:43, 8612.18it/s]  7%|▋         | 27615/400000 [00:03<00:43, 8640.66it/s]  7%|▋         | 28480/400000 [00:03<00:43, 8546.55it/s]  7%|▋         | 29352/400000 [00:03<00:43, 8595.06it/s]  8%|▊         | 30212/400000 [00:03<00:43, 8595.20it/s]  8%|▊         | 31084/400000 [00:03<00:42, 8632.12it/s]  8%|▊         | 31948/400000 [00:03<00:43, 8534.42it/s]  8%|▊         | 32802/400000 [00:03<00:43, 8446.17it/s]  8%|▊         | 33680/400000 [00:03<00:42, 8541.57it/s]  9%|▊         | 34555/400000 [00:04<00:42, 8601.35it/s]  9%|▉         | 35433/400000 [00:04<00:42, 8653.20it/s]  9%|▉         | 36307/400000 [00:04<00:41, 8676.94it/s]  9%|▉         | 37176/400000 [00:04<00:41, 8646.45it/s] 10%|▉         | 38053/400000 [00:04<00:41, 8681.18it/s] 10%|▉         | 38928/400000 [00:04<00:41, 8700.07it/s] 10%|▉         | 39803/400000 [00:04<00:41, 8713.63it/s] 10%|█         | 40678/400000 [00:04<00:41, 8723.35it/s] 10%|█         | 41551/400000 [00:04<00:41, 8703.60it/s] 11%|█         | 42422/400000 [00:04<00:41, 8666.51it/s] 11%|█         | 43294/400000 [00:05<00:41, 8681.11it/s] 11%|█         | 44163/400000 [00:05<00:41, 8656.28it/s] 11%|█▏        | 45034/400000 [00:05<00:40, 8671.21it/s] 11%|█▏        | 45902/400000 [00:05<00:40, 8659.63it/s] 12%|█▏        | 46769/400000 [00:05<00:40, 8654.88it/s] 12%|█▏        | 47635/400000 [00:05<00:40, 8656.29it/s] 12%|█▏        | 48513/400000 [00:05<00:40, 8692.39it/s] 12%|█▏        | 49383/400000 [00:05<00:40, 8676.95it/s] 13%|█▎        | 50251/400000 [00:05<00:40, 8589.00it/s] 13%|█▎        | 51111/400000 [00:05<00:41, 8404.84it/s] 13%|█▎        | 51979/400000 [00:06<00:41, 8483.04it/s] 13%|█▎        | 52852/400000 [00:06<00:40, 8554.36it/s] 13%|█▎        | 53727/400000 [00:06<00:40, 8610.21it/s] 14%|█▎        | 54589/400000 [00:06<00:40, 8597.53it/s] 14%|█▍        | 55465/400000 [00:06<00:39, 8642.99it/s] 14%|█▍        | 56331/400000 [00:06<00:39, 8647.08it/s] 14%|█▍        | 57196/400000 [00:06<00:40, 8511.35it/s] 15%|█▍        | 58065/400000 [00:06<00:39, 8562.59it/s] 15%|█▍        | 58922/400000 [00:06<00:40, 8519.84it/s] 15%|█▍        | 59775/400000 [00:06<00:40, 8491.43it/s] 15%|█▌        | 60632/400000 [00:07<00:39, 8514.73it/s] 15%|█▌        | 61490/400000 [00:07<00:39, 8533.07it/s] 16%|█▌        | 62344/400000 [00:07<00:39, 8522.16it/s] 16%|█▌        | 63197/400000 [00:07<00:39, 8434.94it/s] 16%|█▌        | 64071/400000 [00:07<00:39, 8523.80it/s] 16%|█▌        | 64946/400000 [00:07<00:39, 8587.98it/s] 16%|█▋        | 65824/400000 [00:07<00:38, 8643.44it/s] 17%|█▋        | 66692/400000 [00:07<00:38, 8654.09it/s] 17%|█▋        | 67558/400000 [00:07<00:39, 8523.97it/s] 17%|█▋        | 68427/400000 [00:07<00:38, 8572.13it/s] 17%|█▋        | 69304/400000 [00:08<00:38, 8630.16it/s] 18%|█▊        | 70181/400000 [00:08<00:38, 8671.20it/s] 18%|█▊        | 71057/400000 [00:08<00:37, 8695.28it/s] 18%|█▊        | 71927/400000 [00:08<00:37, 8669.34it/s] 18%|█▊        | 72799/400000 [00:08<00:37, 8684.44it/s] 18%|█▊        | 73674/400000 [00:08<00:37, 8702.33it/s] 19%|█▊        | 74545/400000 [00:08<00:37, 8697.07it/s] 19%|█▉        | 75415/400000 [00:08<00:37, 8686.86it/s] 19%|█▉        | 76284/400000 [00:08<00:37, 8664.26it/s] 19%|█▉        | 77159/400000 [00:08<00:37, 8689.25it/s] 20%|█▉        | 78033/400000 [00:09<00:36, 8702.75it/s] 20%|█▉        | 78909/400000 [00:09<00:36, 8718.15it/s] 20%|█▉        | 79781/400000 [00:09<00:36, 8716.64it/s] 20%|██        | 80653/400000 [00:09<00:36, 8687.34it/s] 20%|██        | 81526/400000 [00:09<00:36, 8697.69it/s] 21%|██        | 82400/400000 [00:09<00:36, 8708.88it/s] 21%|██        | 83272/400000 [00:09<00:36, 8711.47it/s] 21%|██        | 84144/400000 [00:09<00:36, 8691.07it/s] 21%|██▏       | 85014/400000 [00:09<00:36, 8672.49it/s] 21%|██▏       | 85891/400000 [00:09<00:36, 8699.11it/s] 22%|██▏       | 86761/400000 [00:10<00:36, 8680.61it/s] 22%|██▏       | 87630/400000 [00:10<00:36, 8658.87it/s] 22%|██▏       | 88505/400000 [00:10<00:35, 8685.76it/s] 22%|██▏       | 89374/400000 [00:10<00:35, 8648.50it/s] 23%|██▎       | 90239/400000 [00:10<00:35, 8609.23it/s] 23%|██▎       | 91113/400000 [00:10<00:35, 8645.28it/s] 23%|██▎       | 91995/400000 [00:10<00:35, 8695.51it/s] 23%|██▎       | 92868/400000 [00:10<00:35, 8705.83it/s] 23%|██▎       | 93739/400000 [00:10<00:35, 8683.49it/s] 24%|██▎       | 94614/400000 [00:10<00:35, 8700.41it/s] 24%|██▍       | 95486/400000 [00:11<00:34, 8703.57it/s] 24%|██▍       | 96357/400000 [00:11<00:35, 8646.46it/s] 24%|██▍       | 97232/400000 [00:11<00:34, 8675.73it/s] 25%|██▍       | 98102/400000 [00:11<00:34, 8682.62it/s] 25%|██▍       | 98971/400000 [00:11<00:34, 8682.93it/s] 25%|██▍       | 99852/400000 [00:11<00:34, 8718.32it/s] 25%|██▌       | 100724/400000 [00:11<00:34, 8717.26it/s] 25%|██▌       | 101599/400000 [00:11<00:34, 8726.13it/s] 26%|██▌       | 102472/400000 [00:11<00:34, 8693.33it/s] 26%|██▌       | 103342/400000 [00:11<00:34, 8610.77it/s] 26%|██▌       | 104214/400000 [00:12<00:34, 8641.13it/s] 26%|██▋       | 105087/400000 [00:12<00:34, 8664.95it/s] 26%|██▋       | 105961/400000 [00:12<00:33, 8685.12it/s] 27%|██▋       | 106830/400000 [00:12<00:33, 8629.64it/s] 27%|██▋       | 107694/400000 [00:12<00:34, 8562.52it/s] 27%|██▋       | 108568/400000 [00:12<00:33, 8614.66it/s] 27%|██▋       | 109441/400000 [00:12<00:33, 8647.33it/s] 28%|██▊       | 110320/400000 [00:12<00:33, 8687.20it/s] 28%|██▊       | 111189/400000 [00:12<00:33, 8664.69it/s] 28%|██▊       | 112056/400000 [00:12<00:33, 8604.56it/s] 28%|██▊       | 112937/400000 [00:13<00:33, 8662.59it/s] 28%|██▊       | 113813/400000 [00:13<00:32, 8690.28it/s] 29%|██▊       | 114691/400000 [00:13<00:32, 8716.49it/s] 29%|██▉       | 115563/400000 [00:13<00:32, 8687.30it/s] 29%|██▉       | 116432/400000 [00:13<00:32, 8648.18it/s] 29%|██▉       | 117311/400000 [00:13<00:32, 8687.68it/s] 30%|██▉       | 118183/400000 [00:13<00:32, 8697.27it/s] 30%|██▉       | 119061/400000 [00:13<00:32, 8719.68it/s] 30%|██▉       | 119934/400000 [00:13<00:32, 8705.22it/s] 30%|███       | 120808/400000 [00:13<00:32, 8714.72it/s] 30%|███       | 121680/400000 [00:14<00:31, 8705.55it/s] 31%|███       | 122552/400000 [00:14<00:31, 8708.12it/s] 31%|███       | 123425/400000 [00:14<00:31, 8714.29it/s] 31%|███       | 124297/400000 [00:14<00:31, 8637.79it/s] 31%|███▏      | 125161/400000 [00:14<00:32, 8577.07it/s] 32%|███▏      | 126022/400000 [00:14<00:31, 8585.57it/s] 32%|███▏      | 126881/400000 [00:14<00:32, 8518.29it/s] 32%|███▏      | 127734/400000 [00:14<00:32, 8508.09it/s] 32%|███▏      | 128591/400000 [00:14<00:31, 8524.75it/s] 32%|███▏      | 129444/400000 [00:15<00:31, 8510.85it/s] 33%|███▎      | 130296/400000 [00:15<00:31, 8510.95it/s] 33%|███▎      | 131172/400000 [00:15<00:31, 8583.85it/s] 33%|███▎      | 132031/400000 [00:15<00:31, 8565.80it/s] 33%|███▎      | 132901/400000 [00:15<00:31, 8604.17it/s] 33%|███▎      | 133764/400000 [00:15<00:30, 8611.33it/s] 34%|███▎      | 134631/400000 [00:15<00:30, 8628.62it/s] 34%|███▍      | 135497/400000 [00:15<00:30, 8637.72it/s] 34%|███▍      | 136375/400000 [00:15<00:30, 8677.45it/s] 34%|███▍      | 137243/400000 [00:15<00:30, 8663.08it/s] 35%|███▍      | 138110/400000 [00:16<00:30, 8650.93it/s] 35%|███▍      | 138981/400000 [00:16<00:30, 8668.03it/s] 35%|███▍      | 139857/400000 [00:16<00:29, 8694.06it/s] 35%|███▌      | 140727/400000 [00:16<00:30, 8551.16it/s] 35%|███▌      | 141597/400000 [00:16<00:30, 8594.60it/s] 36%|███▌      | 142461/400000 [00:16<00:29, 8607.48it/s] 36%|███▌      | 143338/400000 [00:16<00:29, 8655.23it/s] 36%|███▌      | 144207/400000 [00:16<00:29, 8663.33it/s] 36%|███▋      | 145082/400000 [00:16<00:29, 8687.55it/s] 36%|███▋      | 145953/400000 [00:16<00:29, 8692.23it/s] 37%|███▋      | 146823/400000 [00:17<00:29, 8686.00it/s] 37%|███▋      | 147692/400000 [00:17<00:29, 8450.70it/s] 37%|███▋      | 148556/400000 [00:17<00:29, 8505.47it/s] 37%|███▋      | 149418/400000 [00:17<00:29, 8538.02it/s] 38%|███▊      | 150285/400000 [00:17<00:29, 8574.80it/s] 38%|███▊      | 151149/400000 [00:17<00:28, 8592.89it/s] 38%|███▊      | 152020/400000 [00:17<00:28, 8626.24it/s] 38%|███▊      | 152892/400000 [00:17<00:28, 8654.06it/s] 38%|███▊      | 153766/400000 [00:17<00:28, 8677.92it/s] 39%|███▊      | 154642/400000 [00:17<00:28, 8699.69it/s] 39%|███▉      | 155513/400000 [00:18<00:28, 8684.02it/s] 39%|███▉      | 156391/400000 [00:18<00:27, 8710.56it/s] 39%|███▉      | 157263/400000 [00:18<00:27, 8710.51it/s] 40%|███▉      | 158138/400000 [00:18<00:27, 8721.76it/s] 40%|███▉      | 159011/400000 [00:18<00:27, 8649.16it/s] 40%|███▉      | 159884/400000 [00:18<00:27, 8672.94it/s] 40%|████      | 160758/400000 [00:18<00:27, 8692.85it/s] 40%|████      | 161631/400000 [00:18<00:27, 8702.37it/s] 41%|████      | 162506/400000 [00:18<00:27, 8715.16it/s] 41%|████      | 163383/400000 [00:18<00:27, 8731.33it/s] 41%|████      | 164257/400000 [00:19<00:27, 8658.35it/s] 41%|████▏     | 165126/400000 [00:19<00:27, 8666.95it/s] 41%|████▏     | 165993/400000 [00:19<00:27, 8640.63it/s] 42%|████▏     | 166861/400000 [00:19<00:26, 8652.03it/s] 42%|████▏     | 167727/400000 [00:19<00:26, 8625.70it/s] 42%|████▏     | 168590/400000 [00:19<00:26, 8626.24it/s] 42%|████▏     | 169460/400000 [00:19<00:26, 8646.87it/s] 43%|████▎     | 170325/400000 [00:19<00:26, 8626.67it/s] 43%|████▎     | 171199/400000 [00:19<00:26, 8660.13it/s] 43%|████▎     | 172076/400000 [00:19<00:26, 8690.72it/s] 43%|████▎     | 172946/400000 [00:20<00:26, 8679.73it/s] 43%|████▎     | 173815/400000 [00:20<00:26, 8645.33it/s] 44%|████▎     | 174686/400000 [00:20<00:26, 8662.50it/s] 44%|████▍     | 175553/400000 [00:20<00:25, 8664.23it/s] 44%|████▍     | 176425/400000 [00:20<00:25, 8679.27it/s] 44%|████▍     | 177293/400000 [00:20<00:25, 8617.39it/s] 45%|████▍     | 178160/400000 [00:20<00:25, 8630.63it/s] 45%|████▍     | 179024/400000 [00:20<00:25, 8528.95it/s] 45%|████▍     | 179890/400000 [00:20<00:25, 8565.35it/s] 45%|████▌     | 180761/400000 [00:20<00:25, 8608.16it/s] 45%|████▌     | 181630/400000 [00:21<00:25, 8630.38it/s] 46%|████▌     | 182494/400000 [00:21<00:25, 8618.11it/s] 46%|████▌     | 183369/400000 [00:21<00:25, 8654.55it/s] 46%|████▌     | 184241/400000 [00:21<00:24, 8673.29it/s] 46%|████▋     | 185109/400000 [00:21<00:24, 8630.01it/s] 46%|████▋     | 185985/400000 [00:21<00:24, 8667.04it/s] 47%|████▋     | 186852/400000 [00:21<00:24, 8663.03it/s] 47%|████▋     | 187728/400000 [00:21<00:24, 8691.10it/s] 47%|████▋     | 188605/400000 [00:21<00:24, 8713.33it/s] 47%|████▋     | 189482/400000 [00:21<00:24, 8728.19it/s] 48%|████▊     | 190355/400000 [00:22<00:24, 8717.13it/s] 48%|████▊     | 191227/400000 [00:22<00:24, 8683.81it/s] 48%|████▊     | 192096/400000 [00:22<00:23, 8667.87it/s] 48%|████▊     | 192966/400000 [00:22<00:23, 8676.95it/s] 48%|████▊     | 193834/400000 [00:22<00:24, 8548.24it/s] 49%|████▊     | 194707/400000 [00:22<00:23, 8599.38it/s] 49%|████▉     | 195571/400000 [00:22<00:23, 8609.39it/s] 49%|████▉     | 196443/400000 [00:22<00:23, 8641.85it/s] 49%|████▉     | 197319/400000 [00:22<00:23, 8676.94it/s] 50%|████▉     | 198196/400000 [00:22<00:23, 8704.11it/s] 50%|████▉     | 199067/400000 [00:23<00:23, 8702.34it/s] 50%|████▉     | 199938/400000 [00:23<00:23, 8695.14it/s] 50%|█████     | 200808/400000 [00:23<00:22, 8677.51it/s] 50%|█████     | 201686/400000 [00:23<00:22, 8706.09it/s] 51%|█████     | 202557/400000 [00:23<00:23, 8564.07it/s] 51%|█████     | 203432/400000 [00:23<00:22, 8616.24it/s] 51%|█████     | 204295/400000 [00:23<00:22, 8601.16it/s] 51%|█████▏    | 205168/400000 [00:23<00:22, 8637.90it/s] 52%|█████▏    | 206033/400000 [00:23<00:22, 8626.80it/s] 52%|█████▏    | 206905/400000 [00:23<00:22, 8652.77it/s] 52%|█████▏    | 207773/400000 [00:24<00:22, 8659.11it/s] 52%|█████▏    | 208640/400000 [00:24<00:22, 8632.03it/s] 52%|█████▏    | 209513/400000 [00:24<00:21, 8660.34it/s] 53%|█████▎    | 210380/400000 [00:24<00:22, 8528.60it/s] 53%|█████▎    | 211234/400000 [00:24<00:22, 8407.19it/s] 53%|█████▎    | 212108/400000 [00:24<00:22, 8502.21it/s] 53%|█████▎    | 212972/400000 [00:24<00:21, 8540.48it/s] 53%|█████▎    | 213852/400000 [00:24<00:21, 8616.63it/s] 54%|█████▎    | 214727/400000 [00:24<00:21, 8653.30it/s] 54%|█████▍    | 215603/400000 [00:24<00:21, 8684.20it/s] 54%|█████▍    | 216473/400000 [00:25<00:21, 8687.74it/s] 54%|█████▍    | 217343/400000 [00:25<00:21, 8675.03it/s] 55%|█████▍    | 218216/400000 [00:25<00:20, 8691.22it/s] 55%|█████▍    | 219086/400000 [00:25<00:20, 8646.53it/s] 55%|█████▍    | 219951/400000 [00:25<00:20, 8584.40it/s] 55%|█████▌    | 220826/400000 [00:25<00:20, 8633.31it/s] 55%|█████▌    | 221690/400000 [00:25<00:20, 8554.67it/s] 56%|█████▌    | 222546/400000 [00:25<00:20, 8546.97it/s] 56%|█████▌    | 223421/400000 [00:25<00:20, 8605.25it/s] 56%|█████▌    | 224292/400000 [00:25<00:20, 8634.73it/s] 56%|█████▋    | 225172/400000 [00:26<00:20, 8681.08it/s] 57%|█████▋    | 226041/400000 [00:26<00:20, 8661.42it/s] 57%|█████▋    | 226908/400000 [00:26<00:19, 8659.46it/s] 57%|█████▋    | 227784/400000 [00:26<00:19, 8687.94it/s] 57%|█████▋    | 228659/400000 [00:26<00:19, 8706.04it/s] 57%|█████▋    | 229530/400000 [00:26<00:19, 8704.79it/s] 58%|█████▊    | 230401/400000 [00:26<00:19, 8668.22it/s] 58%|█████▊    | 231268/400000 [00:26<00:19, 8626.71it/s] 58%|█████▊    | 232134/400000 [00:26<00:19, 8636.21it/s] 58%|█████▊    | 233009/400000 [00:26<00:19, 8667.13it/s] 58%|█████▊    | 233884/400000 [00:27<00:19, 8690.11it/s] 59%|█████▊    | 234754/400000 [00:27<00:19, 8656.58it/s] 59%|█████▉    | 235622/400000 [00:27<00:18, 8663.39it/s] 59%|█████▉    | 236493/400000 [00:27<00:18, 8676.23it/s] 59%|█████▉    | 237361/400000 [00:27<00:18, 8602.17it/s] 60%|█████▉    | 238223/400000 [00:27<00:18, 8606.89it/s] 60%|█████▉    | 239084/400000 [00:27<00:18, 8598.48it/s] 60%|█████▉    | 239944/400000 [00:27<00:18, 8576.79it/s] 60%|██████    | 240820/400000 [00:27<00:18, 8629.56it/s] 60%|██████    | 241689/400000 [00:27<00:18, 8645.74it/s] 61%|██████    | 242554/400000 [00:28<00:18, 8607.88it/s] 61%|██████    | 243415/400000 [00:28<00:18, 8531.14it/s] 61%|██████    | 244269/400000 [00:28<00:18, 8448.56it/s] 61%|██████▏   | 245132/400000 [00:28<00:18, 8500.65it/s] 61%|██████▏   | 245997/400000 [00:28<00:18, 8543.31it/s] 62%|██████▏   | 246869/400000 [00:28<00:17, 8592.90it/s] 62%|██████▏   | 247729/400000 [00:28<00:17, 8585.02it/s] 62%|██████▏   | 248595/400000 [00:28<00:17, 8604.60it/s] 62%|██████▏   | 249473/400000 [00:28<00:17, 8653.90it/s] 63%|██████▎   | 250346/400000 [00:29<00:17, 8673.65it/s] 63%|██████▎   | 251214/400000 [00:29<00:17, 8641.23it/s] 63%|██████▎   | 252083/400000 [00:29<00:17, 8655.36it/s] 63%|██████▎   | 252951/400000 [00:29<00:16, 8660.23it/s] 63%|██████▎   | 253818/400000 [00:29<00:16, 8650.16it/s] 64%|██████▎   | 254686/400000 [00:29<00:16, 8656.79it/s] 64%|██████▍   | 255555/400000 [00:29<00:16, 8665.12it/s] 64%|██████▍   | 256430/400000 [00:29<00:16, 8688.01it/s] 64%|██████▍   | 257301/400000 [00:29<00:16, 8692.49it/s] 65%|██████▍   | 258181/400000 [00:29<00:16, 8722.56it/s] 65%|██████▍   | 259059/400000 [00:30<00:16, 8737.34it/s] 65%|██████▍   | 259933/400000 [00:30<00:16, 8726.11it/s] 65%|██████▌   | 260806/400000 [00:30<00:15, 8712.59it/s] 65%|██████▌   | 261680/400000 [00:30<00:15, 8717.95it/s] 66%|██████▌   | 262552/400000 [00:30<00:16, 8561.08it/s] 66%|██████▌   | 263415/400000 [00:30<00:15, 8579.62it/s] 66%|██████▌   | 264283/400000 [00:30<00:15, 8609.17it/s] 66%|██████▋   | 265145/400000 [00:30<00:15, 8599.22it/s] 67%|██████▋   | 266011/400000 [00:30<00:15, 8615.66it/s] 67%|██████▋   | 266883/400000 [00:30<00:15, 8644.92it/s] 67%|██████▋   | 267759/400000 [00:31<00:15, 8677.59it/s] 67%|██████▋   | 268639/400000 [00:31<00:15, 8711.33it/s] 67%|██████▋   | 269511/400000 [00:31<00:14, 8704.45it/s] 68%|██████▊   | 270386/400000 [00:31<00:14, 8716.75it/s] 68%|██████▊   | 271258/400000 [00:31<00:14, 8698.02it/s] 68%|██████▊   | 272128/400000 [00:31<00:14, 8670.66it/s] 68%|██████▊   | 272996/400000 [00:31<00:14, 8632.44it/s] 68%|██████▊   | 273864/400000 [00:31<00:14, 8645.32it/s] 69%|██████▊   | 274734/400000 [00:31<00:14, 8660.16it/s] 69%|██████▉   | 275602/400000 [00:31<00:14, 8663.48it/s] 69%|██████▉   | 276477/400000 [00:32<00:14, 8687.96it/s] 69%|██████▉   | 277346/400000 [00:32<00:14, 8681.28it/s] 70%|██████▉   | 278215/400000 [00:32<00:14, 8662.10it/s] 70%|██████▉   | 279082/400000 [00:32<00:13, 8658.50it/s] 70%|██████▉   | 279956/400000 [00:32<00:13, 8681.81it/s] 70%|███████   | 280825/400000 [00:32<00:13, 8671.86it/s] 70%|███████   | 281699/400000 [00:32<00:13, 8689.23it/s] 71%|███████   | 282570/400000 [00:32<00:13, 8694.82it/s] 71%|███████   | 283440/400000 [00:32<00:13, 8548.21it/s] 71%|███████   | 284311/400000 [00:32<00:13, 8595.96it/s] 71%|███████▏  | 285186/400000 [00:33<00:13, 8640.96it/s] 72%|███████▏  | 286061/400000 [00:33<00:13, 8672.35it/s] 72%|███████▏  | 286937/400000 [00:33<00:12, 8697.27it/s] 72%|███████▏  | 287809/400000 [00:33<00:12, 8703.96it/s] 72%|███████▏  | 288680/400000 [00:33<00:12, 8627.01it/s] 72%|███████▏  | 289543/400000 [00:33<00:12, 8583.47it/s] 73%|███████▎  | 290416/400000 [00:33<00:12, 8625.36it/s] 73%|███████▎  | 291279/400000 [00:33<00:12, 8608.93it/s] 73%|███████▎  | 292153/400000 [00:33<00:12, 8646.31it/s] 73%|███████▎  | 293027/400000 [00:33<00:12, 8672.96it/s] 73%|███████▎  | 293899/400000 [00:34<00:12, 8686.34it/s] 74%|███████▎  | 294771/400000 [00:34<00:12, 8694.21it/s] 74%|███████▍  | 295641/400000 [00:34<00:12, 8694.47it/s] 74%|███████▍  | 296511/400000 [00:34<00:11, 8679.50it/s] 74%|███████▍  | 297383/400000 [00:34<00:11, 8689.00it/s] 75%|███████▍  | 298252/400000 [00:34<00:11, 8575.25it/s] 75%|███████▍  | 299127/400000 [00:34<00:11, 8625.65it/s] 75%|███████▍  | 299994/400000 [00:34<00:11, 8637.52it/s] 75%|███████▌  | 300858/400000 [00:34<00:11, 8623.27it/s] 75%|███████▌  | 301726/400000 [00:34<00:11, 8639.96it/s] 76%|███████▌  | 302597/400000 [00:35<00:11, 8660.15it/s] 76%|███████▌  | 303471/400000 [00:35<00:11, 8683.43it/s] 76%|███████▌  | 304342/400000 [00:35<00:11, 8690.18it/s] 76%|███████▋  | 305212/400000 [00:35<00:10, 8682.35it/s] 77%|███████▋  | 306088/400000 [00:35<00:10, 8705.22it/s] 77%|███████▋  | 306959/400000 [00:35<00:10, 8692.47it/s] 77%|███████▋  | 307833/400000 [00:35<00:10, 8704.13it/s] 77%|███████▋  | 308706/400000 [00:35<00:10, 8709.60it/s] 77%|███████▋  | 309577/400000 [00:35<00:10, 8254.82it/s] 78%|███████▊  | 310450/400000 [00:35<00:10, 8391.31it/s] 78%|███████▊  | 311315/400000 [00:36<00:10, 8466.44it/s] 78%|███████▊  | 312192/400000 [00:36<00:10, 8553.39it/s] 78%|███████▊  | 313060/400000 [00:36<00:10, 8590.63it/s] 78%|███████▊  | 313921/400000 [00:36<00:10, 8594.91it/s] 79%|███████▊  | 314782/400000 [00:36<00:09, 8574.87it/s] 79%|███████▉  | 315651/400000 [00:36<00:09, 8608.61it/s] 79%|███████▉  | 316518/400000 [00:36<00:09, 8625.31it/s] 79%|███████▉  | 317389/400000 [00:36<00:09, 8649.42it/s] 80%|███████▉  | 318255/400000 [00:36<00:09, 8611.96it/s] 80%|███████▉  | 319126/400000 [00:36<00:09, 8639.10it/s] 80%|████████  | 320000/400000 [00:37<00:09, 8667.12it/s] 80%|████████  | 320867/400000 [00:37<00:09, 8656.31it/s] 80%|████████  | 321733/400000 [00:37<00:09, 8553.70it/s] 81%|████████  | 322599/400000 [00:37<00:09, 8583.31it/s] 81%|████████  | 323467/400000 [00:37<00:08, 8612.06it/s] 81%|████████  | 324338/400000 [00:37<00:08, 8640.09it/s] 81%|████████▏ | 325207/400000 [00:37<00:08, 8654.43it/s] 82%|████████▏ | 326073/400000 [00:37<00:08, 8645.07it/s] 82%|████████▏ | 326938/400000 [00:37<00:08, 8604.57it/s] 82%|████████▏ | 327807/400000 [00:37<00:08, 8629.87it/s] 82%|████████▏ | 328671/400000 [00:38<00:08, 8629.70it/s] 82%|████████▏ | 329548/400000 [00:38<00:08, 8670.58it/s] 83%|████████▎ | 330421/400000 [00:38<00:08, 8687.90it/s] 83%|████████▎ | 331290/400000 [00:38<00:07, 8680.86it/s] 83%|████████▎ | 332162/400000 [00:38<00:07, 8690.45it/s] 83%|████████▎ | 333037/400000 [00:38<00:07, 8705.41it/s] 83%|████████▎ | 333908/400000 [00:38<00:07, 8703.50it/s] 84%|████████▎ | 334782/400000 [00:38<00:07, 8712.81it/s] 84%|████████▍ | 335654/400000 [00:38<00:07, 8684.59it/s] 84%|████████▍ | 336523/400000 [00:38<00:07, 8671.48it/s] 84%|████████▍ | 337391/400000 [00:39<00:07, 8631.26it/s] 85%|████████▍ | 338258/400000 [00:39<00:07, 8641.12it/s] 85%|████████▍ | 339123/400000 [00:39<00:07, 8615.90it/s] 85%|████████▍ | 339985/400000 [00:39<00:07, 8227.56it/s] 85%|████████▌ | 340845/400000 [00:39<00:07, 8333.38it/s] 85%|████████▌ | 341717/400000 [00:39<00:06, 8442.87it/s] 86%|████████▌ | 342590/400000 [00:39<00:06, 8526.48it/s] 86%|████████▌ | 343448/400000 [00:39<00:06, 8541.12it/s] 86%|████████▌ | 344304/400000 [00:39<00:06, 8537.22it/s] 86%|████████▋ | 345159/400000 [00:39<00:06, 8468.97it/s] 87%|████████▋ | 346026/400000 [00:40<00:06, 8526.42it/s] 87%|████████▋ | 346901/400000 [00:40<00:06, 8591.65it/s] 87%|████████▋ | 347770/400000 [00:40<00:06, 8618.95it/s] 87%|████████▋ | 348640/400000 [00:40<00:05, 8640.12it/s] 87%|████████▋ | 349515/400000 [00:40<00:05, 8672.23it/s] 88%|████████▊ | 350386/400000 [00:40<00:05, 8681.15it/s] 88%|████████▊ | 351257/400000 [00:40<00:05, 8689.22it/s] 88%|████████▊ | 352130/400000 [00:40<00:05, 8700.95it/s] 88%|████████▊ | 353001/400000 [00:40<00:05, 8669.72it/s] 88%|████████▊ | 353871/400000 [00:40<00:05, 8678.01it/s] 89%|████████▊ | 354739/400000 [00:41<00:05, 8591.67it/s] 89%|████████▉ | 355599/400000 [00:41<00:05, 8482.86it/s] 89%|████████▉ | 356458/400000 [00:41<00:05, 8512.96it/s] 89%|████████▉ | 357310/400000 [00:41<00:05, 8505.53it/s] 90%|████████▉ | 358182/400000 [00:41<00:04, 8567.68it/s] 90%|████████▉ | 359064/400000 [00:41<00:04, 8640.71it/s] 90%|████████▉ | 359939/400000 [00:41<00:04, 8670.65it/s] 90%|█████████ | 360815/400000 [00:41<00:04, 8695.35it/s] 90%|█████████ | 361685/400000 [00:41<00:04, 8666.21it/s] 91%|█████████ | 362561/400000 [00:42<00:04, 8693.49it/s] 91%|█████████ | 363431/400000 [00:42<00:04, 8670.82it/s] 91%|█████████ | 364302/400000 [00:42<00:04, 8681.35it/s] 91%|█████████▏| 365171/400000 [00:42<00:04, 8682.55it/s] 92%|█████████▏| 366040/400000 [00:42<00:03, 8572.30it/s] 92%|█████████▏| 366914/400000 [00:42<00:03, 8619.87it/s] 92%|█████████▏| 367784/400000 [00:42<00:03, 8643.30it/s] 92%|█████████▏| 368658/400000 [00:42<00:03, 8671.25it/s] 92%|█████████▏| 369533/400000 [00:42<00:03, 8692.06it/s] 93%|█████████▎| 370403/400000 [00:42<00:03, 8675.22it/s] 93%|█████████▎| 371274/400000 [00:43<00:03, 8684.10it/s] 93%|█████████▎| 372153/400000 [00:43<00:03, 8714.96it/s] 93%|█████████▎| 373027/400000 [00:43<00:03, 8720.82it/s] 93%|█████████▎| 373902/400000 [00:43<00:02, 8728.95it/s] 94%|█████████▎| 374775/400000 [00:43<00:02, 8703.47it/s] 94%|█████████▍| 375649/400000 [00:43<00:02, 8712.67it/s] 94%|█████████▍| 376522/400000 [00:43<00:02, 8717.08it/s] 94%|█████████▍| 377398/400000 [00:43<00:02, 8729.29it/s] 95%|█████████▍| 378273/400000 [00:43<00:02, 8732.60it/s] 95%|█████████▍| 379147/400000 [00:43<00:02, 8697.41it/s] 95%|█████████▌| 380024/400000 [00:44<00:02, 8717.80it/s] 95%|█████████▌| 380900/400000 [00:44<00:02, 8727.65it/s] 95%|█████████▌| 381773/400000 [00:44<00:02, 8720.77it/s] 96%|█████████▌| 382646/400000 [00:44<00:02, 8619.81it/s] 96%|█████████▌| 383509/400000 [00:44<00:01, 8599.03it/s] 96%|█████████▌| 384371/400000 [00:44<00:01, 8604.64it/s] 96%|█████████▋| 385246/400000 [00:44<00:01, 8645.48it/s] 97%|█████████▋| 386111/400000 [00:44<00:01, 8606.67it/s] 97%|█████████▋| 386985/400000 [00:44<00:01, 8644.96it/s] 97%|█████████▋| 387850/400000 [00:44<00:01, 8600.55it/s] 97%|█████████▋| 388711/400000 [00:45<00:01, 8576.29it/s] 97%|█████████▋| 389584/400000 [00:45<00:01, 8620.51it/s] 98%|█████████▊| 390453/400000 [00:45<00:01, 8640.00it/s] 98%|█████████▊| 391318/400000 [00:45<00:01, 8616.80it/s] 98%|█████████▊| 392180/400000 [00:45<00:00, 8595.02it/s] 98%|█████████▊| 393047/400000 [00:45<00:00, 8614.72it/s] 98%|█████████▊| 393918/400000 [00:45<00:00, 8641.76it/s] 99%|█████████▊| 394783/400000 [00:45<00:00, 8608.28it/s] 99%|█████████▉| 395651/400000 [00:45<00:00, 8629.22it/s] 99%|█████████▉| 396516/400000 [00:45<00:00, 8634.74it/s] 99%|█████████▉| 397385/400000 [00:46<00:00, 8650.14it/s]100%|█████████▉| 398253/400000 [00:46<00:00, 8657.86it/s]100%|█████████▉| 399128/400000 [00:46<00:00, 8684.03it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8689.47it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8634.45it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9ff63c9d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011100977190233266 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.010990419515399232 	 Accuracy: 56

  model saves at 56% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 16010 out of table with 15942 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 16010 out of table with 15942 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-13 06:24:09.086296: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 06:24:09.090176: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-13 06:24:09.090865: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562a94c380b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 06:24:09.090880: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fa00044d048> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.2373 - accuracy: 0.5280
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6360 - accuracy: 0.5020 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6871 - accuracy: 0.4987
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6820 - accuracy: 0.4990
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6176 - accuracy: 0.5032
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6078 - accuracy: 0.5038
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6097 - accuracy: 0.5037
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6130 - accuracy: 0.5035
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6598 - accuracy: 0.5004
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6712 - accuracy: 0.4997
11000/25000 [============>.................] - ETA: 3s - loss: 7.6694 - accuracy: 0.4998
12000/25000 [=============>................] - ETA: 3s - loss: 7.6104 - accuracy: 0.5037
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6159 - accuracy: 0.5033
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6184 - accuracy: 0.5031
15000/25000 [=================>............] - ETA: 2s - loss: 7.6370 - accuracy: 0.5019
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6312 - accuracy: 0.5023
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6305 - accuracy: 0.5024
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6487 - accuracy: 0.5012
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6360 - accuracy: 0.5020
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6490 - accuracy: 0.5012
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6768 - accuracy: 0.4993
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6771 - accuracy: 0.4993
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6686 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 7s 286us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f9f62d2ada0> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f9f63f91160> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3353 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.1938 - val_crf_viterbi_accuracy: 0.6800

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

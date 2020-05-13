
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f57d0ebf4e0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 07:14:11.428413
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 07:14:11.432274
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 07:14:11.435466
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 07:14:11.438670
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f57c920f470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356631.2812
Epoch 2/10

1/1 [==============================] - 0s 97ms/step - loss: 265375.5938
Epoch 3/10

1/1 [==============================] - 0s 95ms/step - loss: 162375.2656
Epoch 4/10

1/1 [==============================] - 0s 93ms/step - loss: 91381.9141
Epoch 5/10

1/1 [==============================] - 0s 96ms/step - loss: 48931.1719
Epoch 6/10

1/1 [==============================] - 0s 93ms/step - loss: 27808.3828
Epoch 7/10

1/1 [==============================] - 0s 95ms/step - loss: 17364.7129
Epoch 8/10

1/1 [==============================] - 0s 100ms/step - loss: 11713.4834
Epoch 9/10

1/1 [==============================] - 0s 97ms/step - loss: 8443.2959
Epoch 10/10

1/1 [==============================] - 0s 93ms/step - loss: 6443.6704

  #### Inference Need return ypred, ytrue ######################### 
[[ 1.66466713e+00 -1.34556890e+00  8.58193219e-01  2.58954942e-01
  -4.53903794e-01  1.01403904e+00 -2.69322157e-01 -5.95761597e-01
   1.37893295e+00  1.40039420e+00  3.71354252e-01  9.84656811e-02
  -1.15342367e+00  9.92819965e-02 -5.23719847e-01 -2.47315764e-01
   2.18284935e-01 -5.29757440e-02  8.48022819e-01 -1.58965543e-01
   7.13977754e-01 -3.08574915e-01 -1.43692851e+00 -5.76203585e-01
  -9.80433345e-01 -5.72292507e-01 -2.00214982e-01  7.71526992e-02
   1.43102157e+00  5.95913887e-01  8.77669692e-01  1.52316228e-01
   4.21848238e-01 -7.19169736e-01  4.06790674e-02 -4.26477909e-01
  -1.38944423e+00 -1.41373575e-02 -4.45166141e-01 -7.20654845e-01
   4.97907579e-01 -4.40433711e-01  3.19910049e-01  1.33787286e+00
   8.14595938e-01 -1.91924644e+00 -3.89001459e-01  8.95805657e-02
   9.72507179e-01  6.81689382e-03  5.52394450e-01  8.88934314e-01
   3.44340742e-01  3.00286412e-02  1.06318283e+00 -6.90414727e-01
  -1.38079751e+00 -6.55661523e-01 -5.01366973e-01 -1.13198388e+00
  -3.57340515e-01  8.52452660e+00  7.98207664e+00  6.89520550e+00
   8.45992184e+00  7.49438953e+00  7.70795822e+00  7.22192860e+00
   7.43943882e+00  8.63072586e+00  9.12082291e+00  9.58030891e+00
   8.55622578e+00  7.20701408e+00  8.19658661e+00  5.96305227e+00
   8.24592590e+00  7.73461246e+00  7.05812168e+00  8.37885094e+00
   7.16526127e+00  8.04369640e+00  8.84874249e+00  7.72424078e+00
   9.90418053e+00  7.89597559e+00  8.88620949e+00  7.57826900e+00
   8.46850014e+00  8.07741642e+00  7.33960581e+00  9.08513451e+00
   7.69180584e+00  5.77023172e+00  8.79183960e+00  9.72575188e+00
   8.15298367e+00  8.62607861e+00  8.30662537e+00  7.16983986e+00
   7.77686310e+00  8.96178532e+00  7.87498999e+00  8.43034554e+00
   7.51302290e+00  7.88643646e+00  8.33463383e+00  9.00921631e+00
   9.05758095e+00  9.04358864e+00  8.04360390e+00  8.33333778e+00
   9.11740780e+00  9.41025066e+00  8.18307972e+00  7.89406681e+00
   9.79945278e+00  9.23394394e+00  8.40270042e+00  7.30276537e+00
   8.10485184e-01 -7.63568282e-01  3.41280103e-01 -6.26637518e-01
   2.08612576e-01 -8.39134753e-01 -1.76221478e+00  6.37223661e-01
  -1.40887439e-01 -1.57395840e+00 -4.85864133e-01  1.96971893e+00
   1.14627028e+00  1.45950317e+00  2.68142581e-01  8.03449035e-01
  -4.05808389e-01 -5.85414290e-01  8.26173425e-01 -3.89419198e-01
   8.39876711e-01 -3.35646302e-01  1.12424994e+00 -6.47450566e-01
   1.54588616e+00  6.26290798e-01  1.93340510e-01 -7.53407836e-01
   3.45490098e-01  8.19213331e-01 -1.21505833e+00 -1.23450756e+00
   8.58434558e-01  1.34589231e+00  4.35527772e-01 -3.43715608e-01
  -7.44606495e-01 -1.34492075e+00  3.61509800e-01 -1.43170625e-01
   3.16951990e-01 -1.69279873e-02 -4.15236712e-01  9.88661528e-01
   4.35911417e-01  8.12417507e-01  2.41667557e+00  7.20281124e-01
  -1.42214513e+00  2.41435575e+00 -4.20938671e-01 -1.22139013e+00
   3.75000328e-01  1.49669081e-01  7.08369970e-01  1.41213608e+00
   1.09353453e-01  1.53478503e-01  1.14537013e+00 -6.34190500e-01
   1.98967528e+00  2.69063330e+00  1.67940736e+00  1.78384423e+00
   2.12672853e+00  7.51176536e-01  2.48896694e+00  7.80250490e-01
   1.32193577e+00  2.18520117e+00  7.00282931e-01  2.31473017e+00
   6.27116144e-01  1.98222721e+00  1.76027167e+00  1.26731062e+00
   1.07683754e+00  1.55079365e-01  1.81069732e+00  1.08711004e+00
   1.93712604e+00  1.47570133e+00  1.37986469e+00  2.42332816e+00
   1.56117606e+00  1.53257418e+00  1.14096057e+00  9.78331566e-01
   1.20690370e+00  1.46747065e+00  1.18365431e+00  2.02546310e+00
   5.00312388e-01  2.08255100e+00  1.11601889e+00  4.29244876e-01
   9.28760052e-01  1.09077966e+00  9.80311990e-01  1.14636612e+00
   1.11386955e+00  1.29385686e+00  2.84215450e-01  1.06687582e+00
   6.26033783e-01  1.21036792e+00  1.94032955e+00  1.77875125e+00
   8.62795234e-01  1.23059499e+00  1.57554412e+00  1.95830882e+00
   2.14259386e+00  3.12385559e+00  8.12964559e-01  3.52812588e-01
   1.39092505e+00  3.63633156e-01  6.04934812e-01  1.43209577e+00
   3.68229747e-02  8.41390419e+00  9.26378536e+00  7.88717937e+00
   9.00298405e+00  8.61869144e+00  8.31214905e+00  9.66840076e+00
   7.47180843e+00  9.37838459e+00  8.05796337e+00  8.11851692e+00
   8.71964264e+00  7.84445238e+00  9.09300327e+00  8.06964302e+00
   8.03469467e+00  8.32191086e+00  7.18848848e+00  8.10993958e+00
   8.74185944e+00  9.48396397e+00  9.15913486e+00  9.18339348e+00
   9.03261185e+00  9.66477680e+00  9.05797291e+00  8.03789616e+00
   8.23128986e+00  8.60164928e+00  7.74629736e+00  9.59183121e+00
   7.58054829e+00  9.85963154e+00  9.54121590e+00  7.70183516e+00
   7.28078651e+00  7.51002836e+00  9.12349319e+00  8.56804180e+00
   8.96936893e+00  8.31732750e+00  8.73057842e+00  7.29512644e+00
   9.85037231e+00  8.99720764e+00  6.82258368e+00  8.93526459e+00
   9.70312595e+00  7.67243576e+00  8.81153011e+00  8.83824444e+00
   8.41618729e+00  7.09704590e+00  8.20713711e+00  7.91718960e+00
   9.90258980e+00  7.96649265e+00  8.32985020e+00  9.61501312e+00
   7.38527417e-01  2.46520138e+00  3.81419361e-01  1.01158369e+00
   3.06121409e-01  8.27007532e-01  1.14091587e+00  1.34002566e+00
   4.58511889e-01  5.86267352e-01  3.14899969e+00  8.28120351e-01
   3.21913671e+00  7.26569533e-01  8.01911175e-01  3.43742847e-01
   3.50149035e-01  1.40239167e+00  1.34224629e+00  7.94751763e-01
   3.48364234e-01  7.56688058e-01  4.38113570e-01  2.85955012e-01
   9.89152193e-01  4.09900904e-01  1.06545317e+00  4.31319416e-01
   1.66194248e+00  3.46762061e-01  3.43503475e-01  4.80951905e-01
   6.73806846e-01  1.13217998e+00  1.60296082e+00  1.86078048e+00
   5.67211688e-01  1.59030616e+00  1.79827750e+00  1.82389283e+00
   1.43761802e+00  2.31997681e+00  1.90766764e+00  8.51551831e-01
   1.18044817e+00  4.37291265e-01  1.58838534e+00  6.91153646e-01
   8.50040078e-01  1.29616618e+00  1.14985037e+00  7.70946920e-01
   5.39358735e-01  5.05861819e-01  1.56702173e+00  1.08132541e+00
   1.22454309e+00  1.47198892e+00  1.42464423e+00  2.53878474e-01
  -7.17640305e+00  1.04158144e+01 -4.30559778e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 07:14:20.866976
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    93.852
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 07:14:20.870777
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8830.78
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 07:14:20.874264
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   93.5686
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 07:14:20.877758
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -789.859
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140014455124376
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140013652783736
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140013652784240
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140013652784744
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140013652785248
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140013652785752

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f57c508ef28> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.790853
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.746577
grad_step = 000002, loss = 0.712684
grad_step = 000003, loss = 0.673072
grad_step = 000004, loss = 0.628942
grad_step = 000005, loss = 0.582466
grad_step = 000006, loss = 0.544944
grad_step = 000007, loss = 0.511713
grad_step = 000008, loss = 0.488747
grad_step = 000009, loss = 0.463698
grad_step = 000010, loss = 0.436825
grad_step = 000011, loss = 0.417087
grad_step = 000012, loss = 0.403682
grad_step = 000013, loss = 0.389606
grad_step = 000014, loss = 0.371537
grad_step = 000015, loss = 0.350044
grad_step = 000016, loss = 0.329259
grad_step = 000017, loss = 0.311619
grad_step = 000018, loss = 0.295023
grad_step = 000019, loss = 0.276886
grad_step = 000020, loss = 0.259206
grad_step = 000021, loss = 0.243960
grad_step = 000022, loss = 0.230664
grad_step = 000023, loss = 0.217926
grad_step = 000024, loss = 0.204351
grad_step = 000025, loss = 0.189890
grad_step = 000026, loss = 0.176198
grad_step = 000027, loss = 0.164191
grad_step = 000028, loss = 0.152424
grad_step = 000029, loss = 0.140567
grad_step = 000030, loss = 0.129311
grad_step = 000031, loss = 0.119480
grad_step = 000032, loss = 0.110568
grad_step = 000033, loss = 0.101922
grad_step = 000034, loss = 0.093270
grad_step = 000035, loss = 0.085338
grad_step = 000036, loss = 0.078327
grad_step = 000037, loss = 0.071453
grad_step = 000038, loss = 0.064888
grad_step = 000039, loss = 0.059137
grad_step = 000040, loss = 0.054081
grad_step = 000041, loss = 0.049421
grad_step = 000042, loss = 0.044900
grad_step = 000043, loss = 0.040823
grad_step = 000044, loss = 0.037299
grad_step = 000045, loss = 0.033940
grad_step = 000046, loss = 0.030797
grad_step = 000047, loss = 0.028047
grad_step = 000048, loss = 0.025693
grad_step = 000049, loss = 0.023531
grad_step = 000050, loss = 0.021478
grad_step = 000051, loss = 0.019720
grad_step = 000052, loss = 0.018158
grad_step = 000053, loss = 0.016678
grad_step = 000054, loss = 0.015343
grad_step = 000055, loss = 0.014209
grad_step = 000056, loss = 0.013214
grad_step = 000057, loss = 0.012246
grad_step = 000058, loss = 0.011383
grad_step = 000059, loss = 0.010624
grad_step = 000060, loss = 0.009920
grad_step = 000061, loss = 0.009249
grad_step = 000062, loss = 0.008665
grad_step = 000063, loss = 0.008145
grad_step = 000064, loss = 0.007639
grad_step = 000065, loss = 0.007173
grad_step = 000066, loss = 0.006756
grad_step = 000067, loss = 0.006367
grad_step = 000068, loss = 0.005986
grad_step = 000069, loss = 0.005653
grad_step = 000070, loss = 0.005349
grad_step = 000071, loss = 0.005058
grad_step = 000072, loss = 0.004788
grad_step = 000073, loss = 0.004554
grad_step = 000074, loss = 0.004328
grad_step = 000075, loss = 0.004114
grad_step = 000076, loss = 0.003925
grad_step = 000077, loss = 0.003752
grad_step = 000078, loss = 0.003584
grad_step = 000079, loss = 0.003437
grad_step = 000080, loss = 0.003306
grad_step = 000081, loss = 0.003181
grad_step = 000082, loss = 0.003065
grad_step = 000083, loss = 0.002963
grad_step = 000084, loss = 0.002865
grad_step = 000085, loss = 0.002775
grad_step = 000086, loss = 0.002697
grad_step = 000087, loss = 0.002624
grad_step = 000088, loss = 0.002555
grad_step = 000089, loss = 0.002495
grad_step = 000090, loss = 0.002440
grad_step = 000091, loss = 0.002387
grad_step = 000092, loss = 0.002342
grad_step = 000093, loss = 0.002301
grad_step = 000094, loss = 0.002262
grad_step = 000095, loss = 0.002229
grad_step = 000096, loss = 0.002199
grad_step = 000097, loss = 0.002171
grad_step = 000098, loss = 0.002147
grad_step = 000099, loss = 0.002126
grad_step = 000100, loss = 0.002106
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002090
grad_step = 000102, loss = 0.002075
grad_step = 000103, loss = 0.002062
grad_step = 000104, loss = 0.002050
grad_step = 000105, loss = 0.002040
grad_step = 000106, loss = 0.002031
grad_step = 000107, loss = 0.002023
grad_step = 000108, loss = 0.002015
grad_step = 000109, loss = 0.002008
grad_step = 000110, loss = 0.002002
grad_step = 000111, loss = 0.001996
grad_step = 000112, loss = 0.001990
grad_step = 000113, loss = 0.001985
grad_step = 000114, loss = 0.001980
grad_step = 000115, loss = 0.001974
grad_step = 000116, loss = 0.001969
grad_step = 000117, loss = 0.001963
grad_step = 000118, loss = 0.001958
grad_step = 000119, loss = 0.001952
grad_step = 000120, loss = 0.001948
grad_step = 000121, loss = 0.001945
grad_step = 000122, loss = 0.001941
grad_step = 000123, loss = 0.001936
grad_step = 000124, loss = 0.001928
grad_step = 000125, loss = 0.001920
grad_step = 000126, loss = 0.001912
grad_step = 000127, loss = 0.001905
grad_step = 000128, loss = 0.001900
grad_step = 000129, loss = 0.001902
grad_step = 000130, loss = 0.001935
grad_step = 000131, loss = 0.002084
grad_step = 000132, loss = 0.002286
grad_step = 000133, loss = 0.002381
grad_step = 000134, loss = 0.001917
grad_step = 000135, loss = 0.002104
grad_step = 000136, loss = 0.002453
grad_step = 000137, loss = 0.001911
grad_step = 000138, loss = 0.002264
grad_step = 000139, loss = 0.002205
grad_step = 000140, loss = 0.001935
grad_step = 000141, loss = 0.002293
grad_step = 000142, loss = 0.001971
grad_step = 000143, loss = 0.002032
grad_step = 000144, loss = 0.002108
grad_step = 000145, loss = 0.001883
grad_step = 000146, loss = 0.002066
grad_step = 000147, loss = 0.001927
grad_step = 000148, loss = 0.001925
grad_step = 000149, loss = 0.001997
grad_step = 000150, loss = 0.001863
grad_step = 000151, loss = 0.001959
grad_step = 000152, loss = 0.001910
grad_step = 000153, loss = 0.001863
grad_step = 000154, loss = 0.001934
grad_step = 000155, loss = 0.001851
grad_step = 000156, loss = 0.001843
grad_step = 000157, loss = 0.001891
grad_step = 000158, loss = 0.001834
grad_step = 000159, loss = 0.001790
grad_step = 000160, loss = 0.001861
grad_step = 000161, loss = 0.001901
grad_step = 000162, loss = 0.001803
grad_step = 000163, loss = 0.001760
grad_step = 000164, loss = 0.001798
grad_step = 000165, loss = 0.001860
grad_step = 000166, loss = 0.001841
grad_step = 000167, loss = 0.001806
grad_step = 000168, loss = 0.001738
grad_step = 000169, loss = 0.001730
grad_step = 000170, loss = 0.001792
grad_step = 000171, loss = 0.001975
grad_step = 000172, loss = 0.002160
grad_step = 000173, loss = 0.002242
grad_step = 000174, loss = 0.001846
grad_step = 000175, loss = 0.001707
grad_step = 000176, loss = 0.001898
grad_step = 000177, loss = 0.001923
grad_step = 000178, loss = 0.001752
grad_step = 000179, loss = 0.001713
grad_step = 000180, loss = 0.001822
grad_step = 000181, loss = 0.001806
grad_step = 000182, loss = 0.001688
grad_step = 000183, loss = 0.001731
grad_step = 000184, loss = 0.001795
grad_step = 000185, loss = 0.001708
grad_step = 000186, loss = 0.001659
grad_step = 000187, loss = 0.001704
grad_step = 000188, loss = 0.001759
grad_step = 000189, loss = 0.001741
grad_step = 000190, loss = 0.001682
grad_step = 000191, loss = 0.001638
grad_step = 000192, loss = 0.001638
grad_step = 000193, loss = 0.001674
grad_step = 000194, loss = 0.001733
grad_step = 000195, loss = 0.001766
grad_step = 000196, loss = 0.001749
grad_step = 000197, loss = 0.001695
grad_step = 000198, loss = 0.001626
grad_step = 000199, loss = 0.001610
grad_step = 000200, loss = 0.001651
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001691
grad_step = 000202, loss = 0.001689
grad_step = 000203, loss = 0.001654
grad_step = 000204, loss = 0.001616
grad_step = 000205, loss = 0.001585
grad_step = 000206, loss = 0.001589
grad_step = 000207, loss = 0.001612
grad_step = 000208, loss = 0.001625
grad_step = 000209, loss = 0.001628
grad_step = 000210, loss = 0.001614
grad_step = 000211, loss = 0.001604
grad_step = 000212, loss = 0.001572
grad_step = 000213, loss = 0.001547
grad_step = 000214, loss = 0.001550
grad_step = 000215, loss = 0.001571
grad_step = 000216, loss = 0.001630
grad_step = 000217, loss = 0.001749
grad_step = 000218, loss = 0.001885
grad_step = 000219, loss = 0.001870
grad_step = 000220, loss = 0.001687
grad_step = 000221, loss = 0.001525
grad_step = 000222, loss = 0.001593
grad_step = 000223, loss = 0.001740
grad_step = 000224, loss = 0.001660
grad_step = 000225, loss = 0.001517
grad_step = 000226, loss = 0.001563
grad_step = 000227, loss = 0.001617
grad_step = 000228, loss = 0.001544
grad_step = 000229, loss = 0.001494
grad_step = 000230, loss = 0.001563
grad_step = 000231, loss = 0.001570
grad_step = 000232, loss = 0.001487
grad_step = 000233, loss = 0.001501
grad_step = 000234, loss = 0.001545
grad_step = 000235, loss = 0.001494
grad_step = 000236, loss = 0.001459
grad_step = 000237, loss = 0.001456
grad_step = 000238, loss = 0.001480
grad_step = 000239, loss = 0.001446
grad_step = 000240, loss = 0.001427
grad_step = 000241, loss = 0.001431
grad_step = 000242, loss = 0.001454
grad_step = 000243, loss = 0.001446
grad_step = 000244, loss = 0.001435
grad_step = 000245, loss = 0.001396
grad_step = 000246, loss = 0.001395
grad_step = 000247, loss = 0.001412
grad_step = 000248, loss = 0.001430
grad_step = 000249, loss = 0.001467
grad_step = 000250, loss = 0.001449
grad_step = 000251, loss = 0.001407
grad_step = 000252, loss = 0.001354
grad_step = 000253, loss = 0.001342
grad_step = 000254, loss = 0.001362
grad_step = 000255, loss = 0.001398
grad_step = 000256, loss = 0.001481
grad_step = 000257, loss = 0.001441
grad_step = 000258, loss = 0.001367
grad_step = 000259, loss = 0.001314
grad_step = 000260, loss = 0.001299
grad_step = 000261, loss = 0.001294
grad_step = 000262, loss = 0.001314
grad_step = 000263, loss = 0.001322
grad_step = 000264, loss = 0.001303
grad_step = 000265, loss = 0.001285
grad_step = 000266, loss = 0.001264
grad_step = 000267, loss = 0.001248
grad_step = 000268, loss = 0.001237
grad_step = 000269, loss = 0.001227
grad_step = 000270, loss = 0.001225
grad_step = 000271, loss = 0.001231
grad_step = 000272, loss = 0.001250
grad_step = 000273, loss = 0.001286
grad_step = 000274, loss = 0.001384
grad_step = 000275, loss = 0.001405
grad_step = 000276, loss = 0.001440
grad_step = 000277, loss = 0.001350
grad_step = 000278, loss = 0.001283
grad_step = 000279, loss = 0.001185
grad_step = 000280, loss = 0.001154
grad_step = 000281, loss = 0.001194
grad_step = 000282, loss = 0.001241
grad_step = 000283, loss = 0.001265
grad_step = 000284, loss = 0.001169
grad_step = 000285, loss = 0.001129
grad_step = 000286, loss = 0.001189
grad_step = 000287, loss = 0.001230
grad_step = 000288, loss = 0.001249
grad_step = 000289, loss = 0.001150
grad_step = 000290, loss = 0.001100
grad_step = 000291, loss = 0.001178
grad_step = 000292, loss = 0.001223
grad_step = 000293, loss = 0.001223
grad_step = 000294, loss = 0.001132
grad_step = 000295, loss = 0.001067
grad_step = 000296, loss = 0.001147
grad_step = 000297, loss = 0.001206
grad_step = 000298, loss = 0.001211
grad_step = 000299, loss = 0.001119
grad_step = 000300, loss = 0.001038
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001121
grad_step = 000302, loss = 0.001185
grad_step = 000303, loss = 0.001165
grad_step = 000304, loss = 0.001085
grad_step = 000305, loss = 0.001005
grad_step = 000306, loss = 0.001089
grad_step = 000307, loss = 0.001129
grad_step = 000308, loss = 0.001062
grad_step = 000309, loss = 0.001011
grad_step = 000310, loss = 0.000981
grad_step = 000311, loss = 0.001037
grad_step = 000312, loss = 0.001046
grad_step = 000313, loss = 0.000989
grad_step = 000314, loss = 0.000967
grad_step = 000315, loss = 0.000945
grad_step = 000316, loss = 0.000980
grad_step = 000317, loss = 0.001012
grad_step = 000318, loss = 0.001016
grad_step = 000319, loss = 0.001022
grad_step = 000320, loss = 0.000988
grad_step = 000321, loss = 0.001006
grad_step = 000322, loss = 0.001104
grad_step = 000323, loss = 0.001262
grad_step = 000324, loss = 0.001524
grad_step = 000325, loss = 0.001620
grad_step = 000326, loss = 0.001462
grad_step = 000327, loss = 0.001197
grad_step = 000328, loss = 0.001006
grad_step = 000329, loss = 0.000951
grad_step = 000330, loss = 0.000983
grad_step = 000331, loss = 0.001015
grad_step = 000332, loss = 0.001075
grad_step = 000333, loss = 0.001069
grad_step = 000334, loss = 0.000914
grad_step = 000335, loss = 0.000818
grad_step = 000336, loss = 0.000879
grad_step = 000337, loss = 0.000961
grad_step = 000338, loss = 0.000938
grad_step = 000339, loss = 0.000837
grad_step = 000340, loss = 0.000803
grad_step = 000341, loss = 0.000843
grad_step = 000342, loss = 0.000866
grad_step = 000343, loss = 0.000834
grad_step = 000344, loss = 0.000783
grad_step = 000345, loss = 0.000775
grad_step = 000346, loss = 0.000811
grad_step = 000347, loss = 0.000829
grad_step = 000348, loss = 0.000837
grad_step = 000349, loss = 0.000800
grad_step = 000350, loss = 0.000805
grad_step = 000351, loss = 0.000820
grad_step = 000352, loss = 0.000871
grad_step = 000353, loss = 0.000896
grad_step = 000354, loss = 0.000956
grad_step = 000355, loss = 0.000972
grad_step = 000356, loss = 0.000982
grad_step = 000357, loss = 0.000875
grad_step = 000358, loss = 0.000762
grad_step = 000359, loss = 0.000695
grad_step = 000360, loss = 0.000750
grad_step = 000361, loss = 0.000859
grad_step = 000362, loss = 0.000824
grad_step = 000363, loss = 0.000742
grad_step = 000364, loss = 0.000694
grad_step = 000365, loss = 0.000750
grad_step = 000366, loss = 0.000808
grad_step = 000367, loss = 0.000752
grad_step = 000368, loss = 0.000693
grad_step = 000369, loss = 0.000698
grad_step = 000370, loss = 0.000745
grad_step = 000371, loss = 0.000740
grad_step = 000372, loss = 0.000675
grad_step = 000373, loss = 0.000647
grad_step = 000374, loss = 0.000685
grad_step = 000375, loss = 0.000710
grad_step = 000376, loss = 0.000679
grad_step = 000377, loss = 0.000632
grad_step = 000378, loss = 0.000628
grad_step = 000379, loss = 0.000657
grad_step = 000380, loss = 0.000666
grad_step = 000381, loss = 0.000653
grad_step = 000382, loss = 0.000615
grad_step = 000383, loss = 0.000599
grad_step = 000384, loss = 0.000615
grad_step = 000385, loss = 0.000633
grad_step = 000386, loss = 0.000647
grad_step = 000387, loss = 0.000634
grad_step = 000388, loss = 0.000627
grad_step = 000389, loss = 0.000617
grad_step = 000390, loss = 0.000625
grad_step = 000391, loss = 0.000638
grad_step = 000392, loss = 0.000651
grad_step = 000393, loss = 0.000657
grad_step = 000394, loss = 0.000679
grad_step = 000395, loss = 0.000707
grad_step = 000396, loss = 0.000787
grad_step = 000397, loss = 0.000862
grad_step = 000398, loss = 0.000950
grad_step = 000399, loss = 0.000918
grad_step = 000400, loss = 0.000799
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000676
grad_step = 000402, loss = 0.000649
grad_step = 000403, loss = 0.000672
grad_step = 000404, loss = 0.000629
grad_step = 000405, loss = 0.000569
grad_step = 000406, loss = 0.000561
grad_step = 000407, loss = 0.000623
grad_step = 000408, loss = 0.000671
grad_step = 000409, loss = 0.000625
grad_step = 000410, loss = 0.000555
grad_step = 000411, loss = 0.000526
grad_step = 000412, loss = 0.000552
grad_step = 000413, loss = 0.000585
grad_step = 000414, loss = 0.000574
grad_step = 000415, loss = 0.000541
grad_step = 000416, loss = 0.000508
grad_step = 000417, loss = 0.000505
grad_step = 000418, loss = 0.000528
grad_step = 000419, loss = 0.000556
grad_step = 000420, loss = 0.000588
grad_step = 000421, loss = 0.000588
grad_step = 000422, loss = 0.000608
grad_step = 000423, loss = 0.000607
grad_step = 000424, loss = 0.000630
grad_step = 000425, loss = 0.000635
grad_step = 000426, loss = 0.000642
grad_step = 000427, loss = 0.000637
grad_step = 000428, loss = 0.000607
grad_step = 000429, loss = 0.000558
grad_step = 000430, loss = 0.000503
grad_step = 000431, loss = 0.000470
grad_step = 000432, loss = 0.000468
grad_step = 000433, loss = 0.000485
grad_step = 000434, loss = 0.000503
grad_step = 000435, loss = 0.000510
grad_step = 000436, loss = 0.000509
grad_step = 000437, loss = 0.000507
grad_step = 000438, loss = 0.000506
grad_step = 000439, loss = 0.000525
grad_step = 000440, loss = 0.000544
grad_step = 000441, loss = 0.000595
grad_step = 000442, loss = 0.000629
grad_step = 000443, loss = 0.000683
grad_step = 000444, loss = 0.000683
grad_step = 000445, loss = 0.000678
grad_step = 000446, loss = 0.000645
grad_step = 000447, loss = 0.000624
grad_step = 000448, loss = 0.000599
grad_step = 000449, loss = 0.000562
grad_step = 000450, loss = 0.000509
grad_step = 000451, loss = 0.000451
grad_step = 000452, loss = 0.000434
grad_step = 000453, loss = 0.000461
grad_step = 000454, loss = 0.000502
grad_step = 000455, loss = 0.000519
grad_step = 000456, loss = 0.000501
grad_step = 000457, loss = 0.000471
grad_step = 000458, loss = 0.000451
grad_step = 000459, loss = 0.000449
grad_step = 000460, loss = 0.000471
grad_step = 000461, loss = 0.000477
grad_step = 000462, loss = 0.000484
grad_step = 000463, loss = 0.000468
grad_step = 000464, loss = 0.000456
grad_step = 000465, loss = 0.000441
grad_step = 000466, loss = 0.000446
grad_step = 000467, loss = 0.000457
grad_step = 000468, loss = 0.000474
grad_step = 000469, loss = 0.000488
grad_step = 000470, loss = 0.000507
grad_step = 000471, loss = 0.000516
grad_step = 000472, loss = 0.000539
grad_step = 000473, loss = 0.000548
grad_step = 000474, loss = 0.000567
grad_step = 000475, loss = 0.000559
grad_step = 000476, loss = 0.000542
grad_step = 000477, loss = 0.000491
grad_step = 000478, loss = 0.000450
grad_step = 000479, loss = 0.000426
grad_step = 000480, loss = 0.000422
grad_step = 000481, loss = 0.000421
grad_step = 000482, loss = 0.000409
grad_step = 000483, loss = 0.000401
grad_step = 000484, loss = 0.000401
grad_step = 000485, loss = 0.000414
grad_step = 000486, loss = 0.000427
grad_step = 000487, loss = 0.000430
grad_step = 000488, loss = 0.000422
grad_step = 000489, loss = 0.000407
grad_step = 000490, loss = 0.000395
grad_step = 000491, loss = 0.000391
grad_step = 000492, loss = 0.000396
grad_step = 000493, loss = 0.000410
grad_step = 000494, loss = 0.000424
grad_step = 000495, loss = 0.000456
grad_step = 000496, loss = 0.000486
grad_step = 000497, loss = 0.000549
grad_step = 000498, loss = 0.000620
grad_step = 000499, loss = 0.000718
grad_step = 000500, loss = 0.000795
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000843
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

  date_run                              2020-05-13 07:14:39.106935
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.174957
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 07:14:39.112727
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 0.0749173
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 07:14:39.119417
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.100829
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 07:14:39.124301
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.138394
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
0   2020-05-13 07:14:11.428413  ...    mean_absolute_error
1   2020-05-13 07:14:11.432274  ...     mean_squared_error
2   2020-05-13 07:14:11.435466  ...  median_absolute_error
3   2020-05-13 07:14:11.438670  ...               r2_score
4   2020-05-13 07:14:20.866976  ...    mean_absolute_error
5   2020-05-13 07:14:20.870777  ...     mean_squared_error
6   2020-05-13 07:14:20.874264  ...  median_absolute_error
7   2020-05-13 07:14:20.877758  ...               r2_score
8   2020-05-13 07:14:39.106935  ...    mean_absolute_error
9   2020-05-13 07:14:39.112727  ...     mean_squared_error
10  2020-05-13 07:14:39.119417  ...  median_absolute_error
11  2020-05-13 07:14:39.124301  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f96e67a9c88> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:31, 315957.91it/s]  2%|▏         | 212992/9912422 [00:00<00:23, 408653.23it/s]  9%|▉         | 876544/9912422 [00:00<00:15, 565055.41it/s] 34%|███▍      | 3407872/9912422 [00:00<00:08, 798668.47it/s] 70%|███████   | 6971392/9912422 [00:00<00:02, 1127754.50it/s]9920512it [00:01, 9758539.89it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 147011.61it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 314573.76it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 406628.12it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 562687.53it/s]1654784it [00:00, 2833271.14it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 49228.74it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9699162e10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f96e67b4a58> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9699162e10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f96e67b48d0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9695f25470> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f96e67b4a58> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9699162e10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f96e67b48d0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f9695f25470> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f96987940b8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f7da4bb3208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=10d1ca19d9a0d38c69529308fe32106a47a873b96532d4859e1995bde948649a
  Stored in directory: /tmp/pip-ephem-wheel-cache-sd8z23ju/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f7d3c9ae710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 47s
   57344/17464789 [..............................] - ETA: 40s
  106496/17464789 [..............................] - ETA: 32s
  212992/17464789 [..............................] - ETA: 21s
  417792/17464789 [..............................] - ETA: 13s
  860160/17464789 [>.............................] - ETA: 7s 
 1712128/17464789 [=>............................] - ETA: 4s
 3416064/17464789 [====>.........................] - ETA: 2s
 6414336/17464789 [==========>...................] - ETA: 1s
 9199616/17464789 [==============>...............] - ETA: 0s
12132352/17464789 [===================>..........] - ETA: 0s
15015936/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 07:16:09.241946: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 07:16:09.246007: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095230000 Hz
2020-05-13 07:16:09.246782: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55af5cf92110 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 07:16:09.246801: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.3293 - accuracy: 0.5220
 2000/25000 [=>............................] - ETA: 8s - loss: 7.4290 - accuracy: 0.5155 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.4571 - accuracy: 0.5137
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5095 - accuracy: 0.5102
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5348 - accuracy: 0.5086
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.4980 - accuracy: 0.5110
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5352 - accuracy: 0.5086
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5708 - accuracy: 0.5063
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5951 - accuracy: 0.5047
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6084 - accuracy: 0.5038
11000/25000 [============>.................] - ETA: 3s - loss: 7.6262 - accuracy: 0.5026
12000/25000 [=============>................] - ETA: 3s - loss: 7.6130 - accuracy: 0.5035
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6230 - accuracy: 0.5028
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6064 - accuracy: 0.5039
15000/25000 [=================>............] - ETA: 2s - loss: 7.6308 - accuracy: 0.5023
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6446 - accuracy: 0.5014
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6630 - accuracy: 0.5002
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6709 - accuracy: 0.4997
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6634 - accuracy: 0.5002
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6567 - accuracy: 0.5006
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6776 - accuracy: 0.4993
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6726 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6609 - accuracy: 0.5004
25000/25000 [==============================] - 7s 291us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 07:16:23.196281
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 07:16:23.196281  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<12:00:17, 19.9kB/s].vector_cache/glove.6B.zip:   0%|          | 336k/862M [00:00<8:25:19, 28.4kB/s]  .vector_cache/glove.6B.zip:   0%|          | 3.66M/862M [00:00<5:52:29, 40.6kB/s].vector_cache/glove.6B.zip:   1%|          | 10.2M/862M [00:00<4:04:56, 58.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.1M/862M [00:00<2:49:54, 82.8kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.9M/862M [00:00<1:57:44, 118kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 35.4M/862M [00:01<1:21:38, 169kB/s].vector_cache/glove.6B.zip:   5%|▍         | 43.0M/862M [00:01<56:40, 241kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 48.5M/862M [00:01<39:28, 344kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.2M/862M [00:01<27:42, 488kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:01<19:35, 687kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:03<14:03:50, 15.9kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:03<9:52:05, 22.7kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:05<6:54:24, 32.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:05<4:51:04, 45.9kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.5M/862M [00:05<3:22:57, 65.6kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:07<3:07:17, 71.1kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:07<2:12:04, 101kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 67.4M/862M [00:07<1:32:11, 144kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:09<1:21:59, 161kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:09<58:16, 227kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 71.3M/862M [00:09<40:45, 323kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:11<39:26, 334kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:11<28:49, 457kB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.1M/862M [00:11<20:14, 648kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:13<21:46, 602kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:13<16:07, 812kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:15<13:14, 984kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:15<10:17, 1.27MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:17<09:08, 1.42MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<07:21, 1.76MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:19<07:05, 1.82MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:19<05:51, 2.20MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:21<06:01, 2.13MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:21<05:39, 2.26MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.6M/862M [00:21<04:02, 3.15MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.7M/862M [00:22<1:46:01, 120kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:23<1:15:09, 170kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:23<53:03, 240kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 99.4M/862M [00:24<7:05:11, 29.9kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:24<4:56:26, 42.7kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<3:36:21, 58.4kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<2:32:21, 82.9kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<1:48:02, 116kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<1:16:34, 164kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<55:13, 226kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<39:32, 316kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:30<27:39, 450kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<54:23, 229kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<38:52, 320kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:32<27:12, 455kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<44:02, 281kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:34<31:46, 389kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<23:59, 513kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<17:30, 702kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<14:07, 866kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<10:43, 1.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:38<07:34, 1.61MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<1:43:42, 117kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<1:13:27, 165kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<53:00, 228kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<38:01, 318kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:43<28:17, 425kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<20:43, 580kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<14:56, 802kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<6:41:05, 29.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<4:40:51, 42.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<3:17:27, 60.3kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<2:19:17, 85.0kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<1:38:05, 121kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<1:10:08, 168kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<50:43, 232kB/s]  .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:51<35:24, 331kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<1:06:50, 175kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<47:33, 246kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<34:52, 334kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<25:16, 460kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<19:20, 598kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<14:28, 799kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<11:47, 975kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<09:06, 1.26MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<08:04, 1.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<06:30, 1.76MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<06:15, 1.82MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<05:18, 2.14MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<05:22, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<04:32, 2.48MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:05<03:38, 3.09MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<6:06:45, 30.7kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<4:16:45, 43.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<3:00:31, 61.9kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<2:07:21, 87.2kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<1:29:54, 123kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<1:04:16, 172kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<45:47, 241kB/s]  .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<33:31, 327kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:14<24:13, 452kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<18:32, 588kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<13:47, 790kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<11:14, 963kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<08:42, 1.24MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<07:41, 1.40MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<06:09, 1.75MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:20<04:22, 2.45MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<11:35:51, 15.4kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<8:07:40, 21.9kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<5:40:57, 31.2kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<3:59:15, 44.4kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:24<2:47:20, 63.2kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<7:30:09, 23.5kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<5:14:34, 33.4kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<3:40:57, 47.5kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<2:35:19, 67.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<1:49:24, 95.4kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<1:17:43, 133kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<55:10, 188kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<39:57, 258kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<28:33, 360kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<21:28, 477kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:35<15:47, 648kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<12:31, 811kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<09:25, 1.08MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<08:06, 1.25MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<06:27, 1.56MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<06:00, 1.67MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<04:57, 2.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<04:57, 2.01MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<04:12, 2.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<03:24, 2.91MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<5:00:13, 33.0kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:44<3:29:06, 47.2kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<2:32:25, 64.6kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:46<1:47:51, 91.3kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:46<1:15:07, 130kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<1:10:22, 139kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<50:03, 195kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<36:15, 268kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<26:34, 365kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:50<18:33, 520kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<31:20, 308kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<23:04, 418kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<17:25, 549kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<13:00, 735kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:54<09:07, 1.04MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<3:43:30, 42.5kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<2:37:47, 60.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:56<1:49:55, 85.9kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<1:23:56, 112kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<59:26, 159kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<42:44, 219kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<30:40, 305kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<22:42, 409kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<16:26, 565kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<12:51, 718kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<09:56, 928kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<08:14, 1.11MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<06:26, 1.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<05:51, 1.55MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<04:41, 1.94MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<04:38, 1.94MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:57, 2.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:10<03:06, 2.88MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<5:23:39, 27.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:11<3:45:44, 39.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<2:40:05, 55.6kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<1:53:09, 78.7kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:13<1:18:45, 112kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<1:07:14, 131kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<47:40, 185kB/s]  .vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<34:29, 254kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<24:46, 354kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:17<17:18, 503kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<25:11, 345kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<18:05, 480kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<13:57, 618kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<10:35, 815kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<08:36, 993kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<06:50, 1.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<06:00, 1.42MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<04:45, 1.79MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<04:35, 1.84MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:49, 2.20MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:54, 2.13MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:21, 2.49MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:34, 2.32MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<02:59, 2.77MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:20, 2.45MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:57, 2.77MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:17, 2.47MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:58, 2.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<03:16, 2.47MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<02:40, 3.02MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<03:08, 2.55MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<02:38, 3.04MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:39<02:08, 3.71MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:40<4:53:50, 27.1kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:40<3:24:34, 38.7kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<2:26:05, 54.0kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<1:42:52, 76.6kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<1:12:35, 108kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:44<51:18, 152kB/s]  .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<36:48, 211kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<26:18, 294kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<19:26, 395kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<14:10, 541kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<11:00, 692kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<08:11, 929kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:50<05:45, 1.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<15:23, 490kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<11:19, 665kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<09:00, 831kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:54<06:46, 1.10MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:54<04:46, 1.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:56<40:15, 184kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<28:42, 258kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<21:02, 349kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<15:06, 485kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<11:39, 624kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<08:42, 833kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<07:08, 1.01MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:02<05:20, 1.35MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:50, 1.47MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:04<03:55, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<03:47, 1.86MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:12, 2.20MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:07<03:16, 2.13MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:49, 2.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:08<02:13, 3.12MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<4:15:16, 27.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:09<2:57:49, 38.8kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<2:05:58, 54.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:11<1:28:58, 77.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:11<1:01:47, 110kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<57:25, 119kB/s]  .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<40:57, 166kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<29:22, 229kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<21:08, 318kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<15:39, 426kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<11:24, 584kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<08:54, 741kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<06:48, 969kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:19<04:46, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<14:11, 460kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<10:17, 634kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:23<08:08, 793kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<06:04, 1.06MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<05:13, 1.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:25<04:09, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<03:50, 1.65MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<03:18, 1.91MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:27<02:20, 2.67MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<28:39, 218kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:29<20:29, 305kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<15:08, 409kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<11:07, 556kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<08:36, 710kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<06:34, 930kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<05:26, 1.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<04:14, 1.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:35<02:58, 2.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<57:28, 104kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<40:30, 147kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<28:23, 209kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<3:58:12, 24.9kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:38<2:45:45, 35.6kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<1:57:11, 50.0kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<1:22:24, 71.0kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<57:57, 100kB/s]   .vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<40:46, 142kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<29:11, 196kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<20:42, 276kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<15:13, 371kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<11:10, 506kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:46<07:47, 718kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<16:24, 341kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<12:04, 462kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:48<08:24, 657kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<13:59, 394kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<10:20, 533kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:50<07:12, 756kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<20:23, 267kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<14:54, 365kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:52<10:21, 519kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<45:59, 117kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<32:48, 164kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<23:27, 226kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<17:02, 312kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<12:32, 418kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<09:18, 563kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<07:11, 720kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<05:38, 916kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:00<03:56, 1.29MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<09:41, 527kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<07:15, 703kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:02<05:03, 996kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<25:16, 199kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<18:14, 276kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<13:19, 373kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<09:54, 501kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:06<07:02, 698kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<3:03:45, 26.8kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:07<2:07:38, 38.2kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:09<1:30:24, 53.6kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<1:03:36, 76.1kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<44:40, 107kB/s]   .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<31:48, 150kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:11<22:03, 214kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<18:45, 251kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<13:25, 350kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<09:59, 464kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<07:18, 635kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<05:45, 795kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<04:19, 1.05MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<03:41, 1.22MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<02:53, 1.56MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:40, 1.66MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:08, 2.07MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<02:09, 2.03MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<01:48, 2.40MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<01:54, 2.25MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:38, 2.61MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:46, 2.37MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:31, 2.75MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:41, 2.45MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<01:25, 2.91MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:37, 2.52MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:36, 2.55MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:31<01:08, 3.54MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<12:08, 331kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<08:46, 458kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:34<06:39, 595kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<04:55, 802kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:35<03:33, 1.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<2:25:11, 26.9kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:36<1:40:31, 38.4kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<1:11:22, 53.7kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:38<50:18, 76.1kB/s]  .vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:38<34:51, 109kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<25:49, 146kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<18:26, 204kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<13:13, 280kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<09:36, 384kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<07:08, 508kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<05:19, 680kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:44<03:42, 963kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<06:18, 564kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<04:45, 747kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<03:47, 923kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<03:00, 1.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<02:33, 1.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<02:10, 1.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:59, 1.69MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:47, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:42, 1.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<01:35, 2.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:33, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:56<01:18, 2.46MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:22, 2.29MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:15, 2.48MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:58<00:53, 3.45MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<08:00, 384kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<05:52, 524kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<04:28, 673kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<03:24, 880kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<02:46, 1.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<02:13, 1.32MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<01:56, 1.48MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:33, 1.83MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:06<01:10, 2.39MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<1:46:26, 26.5kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:07<1:13:38, 37.8kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<51:40, 53.3kB/s]  .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<36:18, 75.6kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:09<24:53, 108kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<28:50, 93.0kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<20:20, 132kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:11<13:59, 188kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<11:43, 223kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<08:24, 310kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:13<05:48, 441kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<05:50, 435kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<04:27, 570kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:15<03:07, 804kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<02:51, 869kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<02:17, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:17<01:35, 1.52MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<02:52, 835kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<02:18, 1.04MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:19<01:36, 1.47MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<02:11, 1.07MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<01:43, 1.35MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:21<01:12, 1.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<02:05, 1.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:43, 1.31MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:23<01:11, 1.84MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<11:02, 199kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<07:58, 276kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:25<05:28, 392kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<05:10, 412kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<03:51, 551kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:27<02:39, 781kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<03:21, 616kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<02:34, 799kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<01:57, 1.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:29<01:22, 1.46MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:38, 1.21MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:22, 1.44MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:31<00:56, 2.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<03:14, 590kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<02:23, 798kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:54, 969kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<01:30, 1.23MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<01:16, 1.39MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<01:05, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:59, 1.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:53, 1.90MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:39<00:37, 2.65MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:33, 1.05MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<01:13, 1.33MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<01:03, 1.47MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:52, 1.80MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:48, 1.86MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:42, 2.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:41, 2.08MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:47<00:38, 2.21MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<00:37, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:49<00:34, 2.40MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:34, 2.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:32, 2.41MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<00:32, 2.29MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:31, 2.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:30, 2.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<00:29, 2.33MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:29, 2.24MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:28, 2.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:27, 2.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:26, 2.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:25, 2.24MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:24, 2.31MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:01<00:16, 3.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<03:02, 290kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<02:09, 404kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:32, 530kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<01:07, 713kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:05<00:44, 1.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<01:11, 623kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:54, 815kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:07<00:36, 1.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:44, 916kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:35, 1.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<00:23, 1.59MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:31, 1.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:24, 1.45MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<00:16, 2.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:40, 796kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:31, 1.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:20, 1.37MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<21:47, 22.0kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<14:38, 31.4kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<09:13, 44.5kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<06:26, 63.0kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<03:50, 88.9kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<02:41, 125kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:18<01:32, 178kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<02:40, 102kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<01:52, 143kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<01:01, 199kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:43, 275kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:21, 372kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:15, 499kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:06, 646kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:26<00:04, 832kB/s].vector_cache/glove.6B.zip: 862MB [06:26, 2.23MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 847/400000 [00:00<00:47, 8461.09it/s]  0%|          | 1727/400000 [00:00<00:46, 8558.22it/s]  1%|          | 2645/400000 [00:00<00:45, 8734.66it/s]  1%|          | 3512/400000 [00:00<00:45, 8713.05it/s]  1%|          | 4317/400000 [00:00<00:46, 8501.75it/s]  1%|▏         | 5118/400000 [00:00<00:47, 8347.68it/s]  1%|▏         | 5979/400000 [00:00<00:46, 8423.59it/s]  2%|▏         | 6909/400000 [00:00<00:45, 8667.98it/s]  2%|▏         | 7830/400000 [00:00<00:44, 8822.87it/s]  2%|▏         | 8747/400000 [00:01<00:43, 8921.75it/s]  2%|▏         | 9618/400000 [00:01<00:44, 8798.62it/s]  3%|▎         | 10483/400000 [00:01<00:44, 8670.50it/s]  3%|▎         | 11425/400000 [00:01<00:43, 8881.74it/s]  3%|▎         | 12376/400000 [00:01<00:42, 9058.97it/s]  3%|▎         | 13328/400000 [00:01<00:42, 9192.32it/s]  4%|▎         | 14246/400000 [00:01<00:43, 8943.60it/s]  4%|▍         | 15141/400000 [00:01<00:43, 8823.14it/s]  4%|▍         | 16034/400000 [00:01<00:43, 8853.11it/s]  4%|▍         | 17001/400000 [00:01<00:42, 9081.57it/s]  4%|▍         | 17950/400000 [00:02<00:41, 9198.19it/s]  5%|▍         | 18872/400000 [00:02<00:41, 9193.38it/s]  5%|▍         | 19793/400000 [00:02<00:41, 9077.40it/s]  5%|▌         | 20702/400000 [00:02<00:42, 9025.92it/s]  5%|▌         | 21606/400000 [00:02<00:42, 8888.44it/s]  6%|▌         | 22498/400000 [00:02<00:42, 8895.94it/s]  6%|▌         | 23405/400000 [00:02<00:42, 8944.77it/s]  6%|▌         | 24330/400000 [00:02<00:41, 9030.87it/s]  6%|▋         | 25234/400000 [00:02<00:41, 8943.80it/s]  7%|▋         | 26182/400000 [00:02<00:41, 9097.12it/s]  7%|▋         | 27133/400000 [00:03<00:40, 9214.47it/s]  7%|▋         | 28056/400000 [00:03<00:40, 9142.65it/s]  7%|▋         | 28996/400000 [00:03<00:40, 9215.11it/s]  7%|▋         | 29919/400000 [00:03<00:40, 9043.16it/s]  8%|▊         | 30861/400000 [00:03<00:40, 9150.45it/s]  8%|▊         | 31778/400000 [00:03<00:40, 9063.52it/s]  8%|▊         | 32686/400000 [00:03<00:40, 9035.08it/s]  8%|▊         | 33591/400000 [00:03<00:40, 8987.53it/s]  9%|▊         | 34498/400000 [00:03<00:40, 9010.70it/s]  9%|▉         | 35443/400000 [00:03<00:39, 9137.88it/s]  9%|▉         | 36363/400000 [00:04<00:39, 9155.36it/s]  9%|▉         | 37295/400000 [00:04<00:39, 9202.81it/s] 10%|▉         | 38249/400000 [00:04<00:38, 9300.84it/s] 10%|▉         | 39180/400000 [00:04<00:39, 9151.02it/s] 10%|█         | 40128/400000 [00:04<00:38, 9247.23it/s] 10%|█         | 41087/400000 [00:04<00:38, 9345.35it/s] 11%|█         | 42032/400000 [00:04<00:38, 9374.53it/s] 11%|█         | 42971/400000 [00:04<00:38, 9237.08it/s] 11%|█         | 43905/400000 [00:04<00:38, 9265.13it/s] 11%|█         | 44833/400000 [00:04<00:38, 9251.20it/s] 11%|█▏        | 45759/400000 [00:05<00:38, 9141.02it/s] 12%|█▏        | 46674/400000 [00:05<00:39, 8854.54it/s] 12%|█▏        | 47575/400000 [00:05<00:39, 8899.28it/s] 12%|█▏        | 48467/400000 [00:05<00:39, 8796.63it/s] 12%|█▏        | 49349/400000 [00:05<00:40, 8649.41it/s] 13%|█▎        | 50216/400000 [00:05<00:40, 8630.40it/s] 13%|█▎        | 51157/400000 [00:05<00:39, 8850.32it/s] 13%|█▎        | 52060/400000 [00:05<00:39, 8901.77it/s] 13%|█▎        | 52952/400000 [00:05<00:39, 8884.98it/s] 13%|█▎        | 53842/400000 [00:05<00:39, 8858.39it/s] 14%|█▎        | 54744/400000 [00:06<00:38, 8903.75it/s] 14%|█▍        | 55635/400000 [00:06<00:39, 8773.05it/s] 14%|█▍        | 56514/400000 [00:06<00:39, 8722.56it/s] 14%|█▍        | 57387/400000 [00:06<00:39, 8705.39it/s] 15%|█▍        | 58274/400000 [00:06<00:39, 8753.72it/s] 15%|█▍        | 59185/400000 [00:06<00:38, 8856.17it/s] 15%|█▌        | 60122/400000 [00:06<00:37, 9002.41it/s] 15%|█▌        | 61045/400000 [00:06<00:37, 9066.79it/s] 15%|█▌        | 61983/400000 [00:06<00:36, 9155.79it/s] 16%|█▌        | 62900/400000 [00:07<00:36, 9126.23it/s] 16%|█▌        | 63814/400000 [00:07<00:37, 8913.82it/s] 16%|█▌        | 64707/400000 [00:07<00:38, 8789.33it/s] 16%|█▋        | 65588/400000 [00:07<00:38, 8710.87it/s] 17%|█▋        | 66461/400000 [00:07<00:38, 8639.50it/s] 17%|█▋        | 67326/400000 [00:07<00:38, 8593.98it/s] 17%|█▋        | 68226/400000 [00:07<00:38, 8710.61it/s] 17%|█▋        | 69155/400000 [00:07<00:37, 8875.47it/s] 18%|█▊        | 70052/400000 [00:07<00:37, 8901.44it/s] 18%|█▊        | 70963/400000 [00:07<00:36, 8960.13it/s] 18%|█▊        | 71860/400000 [00:08<00:36, 8940.99it/s] 18%|█▊        | 72786/400000 [00:08<00:36, 9032.51it/s] 18%|█▊        | 73711/400000 [00:08<00:35, 9094.32it/s] 19%|█▊        | 74621/400000 [00:08<00:36, 9029.00it/s] 19%|█▉        | 75535/400000 [00:08<00:35, 9061.65it/s] 19%|█▉        | 76443/400000 [00:08<00:35, 9066.14it/s] 19%|█▉        | 77350/400000 [00:08<00:35, 8988.83it/s] 20%|█▉        | 78250/400000 [00:08<00:36, 8843.54it/s] 20%|█▉        | 79136/400000 [00:08<00:36, 8801.97it/s] 20%|██        | 80017/400000 [00:08<00:36, 8738.15it/s] 20%|██        | 80960/400000 [00:09<00:35, 8933.54it/s] 20%|██        | 81872/400000 [00:09<00:35, 8988.52it/s] 21%|██        | 82772/400000 [00:09<00:35, 8925.21it/s] 21%|██        | 83666/400000 [00:09<00:36, 8725.37it/s] 21%|██        | 84541/400000 [00:09<00:36, 8673.84it/s] 21%|██▏       | 85457/400000 [00:09<00:35, 8813.86it/s] 22%|██▏       | 86366/400000 [00:09<00:35, 8893.83it/s] 22%|██▏       | 87293/400000 [00:09<00:34, 9001.97it/s] 22%|██▏       | 88195/400000 [00:09<00:34, 8996.21it/s] 22%|██▏       | 89150/400000 [00:09<00:33, 9153.80it/s] 23%|██▎       | 90113/400000 [00:10<00:33, 9289.18it/s] 23%|██▎       | 91056/400000 [00:10<00:33, 9327.85it/s] 23%|██▎       | 92009/400000 [00:10<00:32, 9386.98it/s] 23%|██▎       | 92949/400000 [00:10<00:34, 9029.00it/s] 23%|██▎       | 93856/400000 [00:10<00:34, 8850.93it/s] 24%|██▎       | 94745/400000 [00:10<00:34, 8780.22it/s] 24%|██▍       | 95626/400000 [00:10<00:34, 8749.15it/s] 24%|██▍       | 96507/400000 [00:10<00:34, 8764.71it/s] 24%|██▍       | 97408/400000 [00:10<00:34, 8835.06it/s] 25%|██▍       | 98293/400000 [00:10<00:34, 8831.45it/s] 25%|██▍       | 99177/400000 [00:11<00:34, 8827.93it/s] 25%|██▌       | 100061/400000 [00:11<00:33, 8824.45it/s] 25%|██▌       | 100957/400000 [00:11<00:33, 8862.33it/s] 25%|██▌       | 101844/400000 [00:11<00:34, 8599.11it/s] 26%|██▌       | 102706/400000 [00:11<00:34, 8543.18it/s] 26%|██▌       | 103562/400000 [00:11<00:35, 8467.31it/s] 26%|██▌       | 104410/400000 [00:11<00:35, 8418.74it/s] 26%|██▋       | 105253/400000 [00:11<00:35, 8409.53it/s] 27%|██▋       | 106096/400000 [00:11<00:34, 8415.57it/s] 27%|██▋       | 106938/400000 [00:11<00:34, 8390.95it/s] 27%|██▋       | 107784/400000 [00:12<00:34, 8407.64it/s] 27%|██▋       | 108635/400000 [00:12<00:34, 8435.44it/s] 27%|██▋       | 109492/400000 [00:12<00:34, 8475.33it/s] 28%|██▊       | 110340/400000 [00:12<00:34, 8445.51it/s] 28%|██▊       | 111197/400000 [00:12<00:34, 8482.05it/s] 28%|██▊       | 112048/400000 [00:12<00:33, 8489.75it/s] 28%|██▊       | 112898/400000 [00:12<00:33, 8482.10it/s] 28%|██▊       | 113758/400000 [00:12<00:33, 8514.80it/s] 29%|██▊       | 114614/400000 [00:12<00:33, 8525.66it/s] 29%|██▉       | 115469/400000 [00:13<00:33, 8532.93it/s] 29%|██▉       | 116338/400000 [00:13<00:33, 8565.38it/s] 29%|██▉       | 117197/400000 [00:13<00:32, 8571.50it/s] 30%|██▉       | 118099/400000 [00:13<00:32, 8700.70it/s] 30%|██▉       | 118970/400000 [00:13<00:32, 8603.81it/s] 30%|██▉       | 119831/400000 [00:13<00:32, 8556.38it/s] 30%|███       | 120734/400000 [00:13<00:32, 8692.43it/s] 30%|███       | 121631/400000 [00:13<00:31, 8771.66it/s] 31%|███       | 122509/400000 [00:13<00:31, 8738.10it/s] 31%|███       | 123384/400000 [00:13<00:31, 8678.09it/s] 31%|███       | 124259/400000 [00:14<00:31, 8697.16it/s] 31%|███▏      | 125130/400000 [00:14<00:31, 8632.36it/s] 32%|███▏      | 126010/400000 [00:14<00:31, 8680.02it/s] 32%|███▏      | 126919/400000 [00:14<00:31, 8798.70it/s] 32%|███▏      | 127813/400000 [00:14<00:30, 8840.39it/s] 32%|███▏      | 128761/400000 [00:14<00:30, 9021.34it/s] 32%|███▏      | 129700/400000 [00:14<00:29, 9126.71it/s] 33%|███▎      | 130620/400000 [00:14<00:29, 9145.87it/s] 33%|███▎      | 131559/400000 [00:14<00:29, 9215.80it/s] 33%|███▎      | 132482/400000 [00:14<00:29, 9198.31it/s] 33%|███▎      | 133406/400000 [00:15<00:28, 9208.66it/s] 34%|███▎      | 134349/400000 [00:15<00:28, 9273.51it/s] 34%|███▍      | 135277/400000 [00:15<00:28, 9215.53it/s] 34%|███▍      | 136236/400000 [00:15<00:28, 9322.72it/s] 34%|███▍      | 137169/400000 [00:15<00:28, 9256.15it/s] 35%|███▍      | 138096/400000 [00:15<00:28, 9193.74it/s] 35%|███▍      | 139016/400000 [00:15<00:29, 8869.75it/s] 35%|███▍      | 139906/400000 [00:15<00:29, 8733.00it/s] 35%|███▌      | 140782/400000 [00:15<00:30, 8601.39it/s] 35%|███▌      | 141645/400000 [00:15<00:30, 8591.50it/s] 36%|███▌      | 142540/400000 [00:16<00:29, 8695.64it/s] 36%|███▌      | 143420/400000 [00:16<00:29, 8725.94it/s] 36%|███▌      | 144294/400000 [00:16<00:29, 8624.01it/s] 36%|███▋      | 145158/400000 [00:16<00:29, 8584.08it/s] 37%|███▋      | 146018/400000 [00:16<00:29, 8546.94it/s] 37%|███▋      | 146874/400000 [00:16<00:29, 8536.82it/s] 37%|███▋      | 147773/400000 [00:16<00:29, 8666.81it/s] 37%|███▋      | 148647/400000 [00:16<00:28, 8688.20it/s] 37%|███▋      | 149517/400000 [00:16<00:28, 8689.92it/s] 38%|███▊      | 150405/400000 [00:16<00:28, 8745.67it/s] 38%|███▊      | 151304/400000 [00:17<00:28, 8814.80it/s] 38%|███▊      | 152188/400000 [00:17<00:28, 8820.34it/s] 38%|███▊      | 153071/400000 [00:17<00:28, 8725.10it/s] 38%|███▊      | 153944/400000 [00:17<00:28, 8655.71it/s] 39%|███▊      | 154810/400000 [00:17<00:28, 8530.59it/s] 39%|███▉      | 155680/400000 [00:17<00:28, 8579.12it/s] 39%|███▉      | 156572/400000 [00:17<00:28, 8676.24it/s] 39%|███▉      | 157490/400000 [00:17<00:27, 8818.58it/s] 40%|███▉      | 158394/400000 [00:17<00:27, 8883.19it/s] 40%|███▉      | 159284/400000 [00:17<00:27, 8735.94it/s] 40%|████      | 160159/400000 [00:18<00:27, 8716.88it/s] 40%|████      | 161032/400000 [00:18<00:27, 8560.25it/s] 40%|████      | 161890/400000 [00:18<00:27, 8556.53it/s] 41%|████      | 162775/400000 [00:18<00:27, 8640.19it/s] 41%|████      | 163640/400000 [00:18<00:27, 8628.45it/s] 41%|████      | 164504/400000 [00:18<00:27, 8608.01it/s] 41%|████▏     | 165366/400000 [00:18<00:27, 8610.79it/s] 42%|████▏     | 166244/400000 [00:18<00:26, 8658.89it/s] 42%|████▏     | 167171/400000 [00:18<00:26, 8830.98it/s] 42%|████▏     | 168063/400000 [00:18<00:26, 8856.48it/s] 42%|████▏     | 168950/400000 [00:19<00:26, 8825.05it/s] 42%|████▏     | 169850/400000 [00:19<00:25, 8875.19it/s] 43%|████▎     | 170825/400000 [00:19<00:25, 9120.63it/s] 43%|████▎     | 171740/400000 [00:19<00:25, 9100.88it/s] 43%|████▎     | 172652/400000 [00:19<00:24, 9099.65it/s] 43%|████▎     | 173563/400000 [00:19<00:25, 8943.62it/s] 44%|████▎     | 174459/400000 [00:19<00:25, 8800.59it/s] 44%|████▍     | 175362/400000 [00:19<00:25, 8866.61it/s] 44%|████▍     | 176313/400000 [00:19<00:24, 9048.32it/s] 44%|████▍     | 177220/400000 [00:20<00:25, 8839.59it/s] 45%|████▍     | 178107/400000 [00:20<00:25, 8749.77it/s] 45%|████▍     | 178984/400000 [00:20<00:25, 8633.73it/s] 45%|████▍     | 179849/400000 [00:20<00:25, 8594.18it/s] 45%|████▌     | 180710/400000 [00:20<00:25, 8520.32it/s] 45%|████▌     | 181576/400000 [00:20<00:25, 8560.67it/s] 46%|████▌     | 182445/400000 [00:20<00:25, 8598.25it/s] 46%|████▌     | 183327/400000 [00:20<00:25, 8661.74it/s] 46%|████▌     | 184199/400000 [00:20<00:24, 8678.06it/s] 46%|████▋     | 185075/400000 [00:20<00:24, 8702.35it/s] 46%|████▋     | 185946/400000 [00:21<00:24, 8688.66it/s] 47%|████▋     | 186816/400000 [00:21<00:24, 8649.59it/s] 47%|████▋     | 187682/400000 [00:21<00:24, 8575.75it/s] 47%|████▋     | 188541/400000 [00:21<00:24, 8578.09it/s] 47%|████▋     | 189399/400000 [00:21<00:24, 8571.57it/s] 48%|████▊     | 190279/400000 [00:21<00:24, 8637.84it/s] 48%|████▊     | 191145/400000 [00:21<00:24, 8642.88it/s] 48%|████▊     | 192019/400000 [00:21<00:23, 8670.39it/s] 48%|████▊     | 192887/400000 [00:21<00:23, 8668.43it/s] 48%|████▊     | 193755/400000 [00:21<00:23, 8671.89it/s] 49%|████▊     | 194626/400000 [00:22<00:23, 8680.57it/s] 49%|████▉     | 195495/400000 [00:22<00:23, 8649.30it/s] 49%|████▉     | 196360/400000 [00:22<00:24, 8465.13it/s] 49%|████▉     | 197221/400000 [00:22<00:23, 8507.12it/s] 50%|████▉     | 198086/400000 [00:22<00:23, 8547.57it/s] 50%|████▉     | 198942/400000 [00:22<00:23, 8411.61it/s] 50%|████▉     | 199802/400000 [00:22<00:23, 8465.59it/s] 50%|█████     | 200668/400000 [00:22<00:23, 8521.29it/s] 50%|█████     | 201544/400000 [00:22<00:23, 8588.90it/s] 51%|█████     | 202431/400000 [00:22<00:22, 8670.32it/s] 51%|█████     | 203299/400000 [00:23<00:22, 8630.25it/s] 51%|█████     | 204163/400000 [00:23<00:22, 8602.21it/s] 51%|█████▏    | 205024/400000 [00:23<00:22, 8586.52it/s] 51%|█████▏    | 205883/400000 [00:23<00:23, 8417.62it/s] 52%|█████▏    | 206735/400000 [00:23<00:22, 8446.48it/s] 52%|█████▏    | 207581/400000 [00:23<00:22, 8446.48it/s] 52%|█████▏    | 208444/400000 [00:23<00:22, 8500.22it/s] 52%|█████▏    | 209321/400000 [00:23<00:22, 8577.52it/s] 53%|█████▎    | 210198/400000 [00:23<00:21, 8632.67it/s] 53%|█████▎    | 211065/400000 [00:23<00:21, 8642.69it/s] 53%|█████▎    | 211945/400000 [00:24<00:21, 8688.24it/s] 53%|█████▎    | 212815/400000 [00:24<00:21, 8653.01it/s] 53%|█████▎    | 213681/400000 [00:24<00:21, 8638.62it/s] 54%|█████▎    | 214546/400000 [00:24<00:21, 8604.92it/s] 54%|█████▍    | 215407/400000 [00:24<00:21, 8576.15it/s] 54%|█████▍    | 216265/400000 [00:24<00:21, 8567.05it/s] 54%|█████▍    | 217122/400000 [00:24<00:21, 8524.95it/s] 54%|█████▍    | 217976/400000 [00:24<00:21, 8528.07it/s] 55%|█████▍    | 218829/400000 [00:24<00:21, 8523.82it/s] 55%|█████▍    | 219691/400000 [00:24<00:21, 8550.60it/s] 55%|█████▌    | 220550/400000 [00:25<00:20, 8562.36it/s] 55%|█████▌    | 221429/400000 [00:25<00:20, 8626.95it/s] 56%|█████▌    | 222339/400000 [00:25<00:20, 8762.27it/s] 56%|█████▌    | 223292/400000 [00:25<00:19, 8976.84it/s] 56%|█████▌    | 224238/400000 [00:25<00:19, 9113.90it/s] 56%|█████▋    | 225188/400000 [00:25<00:18, 9225.14it/s] 57%|█████▋    | 226113/400000 [00:25<00:18, 9164.89it/s] 57%|█████▋    | 227031/400000 [00:25<00:19, 9049.11it/s] 57%|█████▋    | 227938/400000 [00:25<00:19, 8877.33it/s] 57%|█████▋    | 228828/400000 [00:25<00:19, 8877.51it/s] 57%|█████▋    | 229717/400000 [00:26<00:19, 8809.94it/s] 58%|█████▊    | 230660/400000 [00:26<00:18, 8985.18it/s] 58%|█████▊    | 231593/400000 [00:26<00:18, 9084.60it/s] 58%|█████▊    | 232509/400000 [00:26<00:18, 9104.56it/s] 58%|█████▊    | 233421/400000 [00:26<00:18, 9056.38it/s] 59%|█████▊    | 234328/400000 [00:26<00:18, 8983.25it/s] 59%|█████▉    | 235227/400000 [00:26<00:18, 8765.96it/s] 59%|█████▉    | 236175/400000 [00:26<00:18, 8968.44it/s] 59%|█████▉    | 237136/400000 [00:26<00:17, 9149.94it/s] 60%|█████▉    | 238070/400000 [00:26<00:17, 9204.37it/s] 60%|█████▉    | 238993/400000 [00:27<00:17, 9192.46it/s] 60%|█████▉    | 239917/400000 [00:27<00:17, 9203.50it/s] 60%|██████    | 240868/400000 [00:27<00:17, 9292.20it/s] 60%|██████    | 241826/400000 [00:27<00:16, 9376.70it/s] 61%|██████    | 242765/400000 [00:27<00:16, 9321.25it/s] 61%|██████    | 243698/400000 [00:27<00:17, 9013.02it/s] 61%|██████    | 244602/400000 [00:27<00:17, 8910.61it/s] 61%|██████▏   | 245525/400000 [00:27<00:17, 9002.76it/s] 62%|██████▏   | 246448/400000 [00:27<00:16, 9068.50it/s] 62%|██████▏   | 247357/400000 [00:28<00:16, 8984.95it/s] 62%|██████▏   | 248257/400000 [00:28<00:17, 8841.17it/s] 62%|██████▏   | 249143/400000 [00:28<00:17, 8731.77it/s] 63%|██████▎   | 250018/400000 [00:28<00:17, 8683.61it/s] 63%|██████▎   | 250888/400000 [00:28<00:17, 8590.46it/s] 63%|██████▎   | 251748/400000 [00:28<00:17, 8515.47it/s] 63%|██████▎   | 252686/400000 [00:28<00:16, 8756.49it/s] 63%|██████▎   | 253589/400000 [00:28<00:16, 8834.44it/s] 64%|██████▎   | 254552/400000 [00:28<00:16, 9057.03it/s] 64%|██████▍   | 255498/400000 [00:28<00:15, 9171.68it/s] 64%|██████▍   | 256418/400000 [00:29<00:15, 9039.38it/s] 64%|██████▍   | 257324/400000 [00:29<00:16, 8882.92it/s] 65%|██████▍   | 258215/400000 [00:29<00:16, 8794.41it/s] 65%|██████▍   | 259108/400000 [00:29<00:15, 8832.73it/s] 65%|██████▌   | 260024/400000 [00:29<00:15, 8925.58it/s] 65%|██████▌   | 260949/400000 [00:29<00:15, 9018.66it/s] 65%|██████▌   | 261903/400000 [00:29<00:15, 9167.24it/s] 66%|██████▌   | 262845/400000 [00:29<00:14, 9241.19it/s] 66%|██████▌   | 263776/400000 [00:29<00:14, 9259.49it/s] 66%|██████▌   | 264708/400000 [00:29<00:14, 9276.20it/s] 66%|██████▋   | 265637/400000 [00:30<00:14, 9121.89it/s] 67%|██████▋   | 266577/400000 [00:30<00:14, 9202.36it/s] 67%|██████▋   | 267499/400000 [00:30<00:14, 9202.23it/s] 67%|██████▋   | 268458/400000 [00:30<00:14, 9314.90it/s] 67%|██████▋   | 269402/400000 [00:30<00:13, 9349.90it/s] 68%|██████▊   | 270341/400000 [00:30<00:13, 9361.30it/s] 68%|██████▊   | 271278/400000 [00:30<00:13, 9343.41it/s] 68%|██████▊   | 272213/400000 [00:30<00:13, 9299.20it/s] 68%|██████▊   | 273144/400000 [00:30<00:13, 9226.06it/s] 69%|██████▊   | 274067/400000 [00:30<00:13, 9000.23it/s] 69%|██████▊   | 274969/400000 [00:31<00:14, 8871.97it/s] 69%|██████▉   | 275858/400000 [00:31<00:14, 8842.41it/s] 69%|██████▉   | 276744/400000 [00:31<00:14, 8706.38it/s] 69%|██████▉   | 277626/400000 [00:31<00:14, 8739.05it/s] 70%|██████▉   | 278597/400000 [00:31<00:13, 9007.94it/s] 70%|██████▉   | 279527/400000 [00:31<00:13, 9092.64it/s] 70%|███████   | 280439/400000 [00:31<00:13, 9036.38it/s] 70%|███████   | 281345/400000 [00:31<00:13, 8915.55it/s] 71%|███████   | 282238/400000 [00:31<00:13, 8810.05it/s] 71%|███████   | 283121/400000 [00:31<00:13, 8741.53it/s] 71%|███████   | 283997/400000 [00:32<00:13, 8665.14it/s] 71%|███████   | 284865/400000 [00:32<00:13, 8623.10it/s] 71%|███████▏  | 285728/400000 [00:32<00:13, 8571.33it/s] 72%|███████▏  | 286586/400000 [00:32<00:13, 8573.34it/s] 72%|███████▏  | 287451/400000 [00:32<00:13, 8593.46it/s] 72%|███████▏  | 288322/400000 [00:32<00:12, 8625.77it/s] 72%|███████▏  | 289228/400000 [00:32<00:12, 8751.53it/s] 73%|███████▎  | 290123/400000 [00:32<00:12, 8809.57it/s] 73%|███████▎  | 291009/400000 [00:32<00:12, 8823.47it/s] 73%|███████▎  | 291903/400000 [00:32<00:12, 8857.10it/s] 73%|███████▎  | 292818/400000 [00:33<00:11, 8940.43it/s] 73%|███████▎  | 293731/400000 [00:33<00:11, 8993.84it/s] 74%|███████▎  | 294631/400000 [00:33<00:11, 8865.68it/s] 74%|███████▍  | 295519/400000 [00:33<00:11, 8778.80it/s] 74%|███████▍  | 296409/400000 [00:33<00:11, 8814.19it/s] 74%|███████▍  | 297291/400000 [00:33<00:11, 8796.49it/s] 75%|███████▍  | 298189/400000 [00:33<00:11, 8848.14it/s] 75%|███████▍  | 299130/400000 [00:33<00:11, 9009.41it/s] 75%|███████▌  | 300071/400000 [00:33<00:10, 9125.96it/s] 75%|███████▌  | 301009/400000 [00:34<00:10, 9199.70it/s] 75%|███████▌  | 301958/400000 [00:34<00:10, 9281.81it/s] 76%|███████▌  | 302887/400000 [00:34<00:10, 9253.68it/s] 76%|███████▌  | 303813/400000 [00:34<00:10, 9160.22it/s] 76%|███████▌  | 304765/400000 [00:34<00:10, 9264.55it/s] 76%|███████▋  | 305703/400000 [00:34<00:10, 9296.65it/s] 77%|███████▋  | 306634/400000 [00:34<00:10, 9237.66it/s] 77%|███████▋  | 307565/400000 [00:34<00:09, 9257.72it/s] 77%|███████▋  | 308492/400000 [00:34<00:09, 9260.48it/s] 77%|███████▋  | 309419/400000 [00:34<00:09, 9247.86it/s] 78%|███████▊  | 310346/400000 [00:35<00:09, 9253.10it/s] 78%|███████▊  | 311302/400000 [00:35<00:09, 9341.66it/s] 78%|███████▊  | 312237/400000 [00:35<00:09, 9199.74it/s] 78%|███████▊  | 313158/400000 [00:35<00:09, 9117.12it/s] 79%|███████▊  | 314071/400000 [00:35<00:09, 9095.94it/s] 79%|███████▊  | 314982/400000 [00:35<00:09, 8969.41it/s] 79%|███████▉  | 315880/400000 [00:35<00:09, 8848.68it/s] 79%|███████▉  | 316766/400000 [00:35<00:09, 8757.02it/s] 79%|███████▉  | 317643/400000 [00:35<00:09, 8703.82it/s] 80%|███████▉  | 318515/400000 [00:35<00:09, 8660.78it/s] 80%|███████▉  | 319382/400000 [00:36<00:09, 8629.12it/s] 80%|████████  | 320247/400000 [00:36<00:09, 8632.27it/s] 80%|████████  | 321119/400000 [00:36<00:09, 8657.57it/s] 81%|████████  | 322010/400000 [00:36<00:08, 8730.76it/s] 81%|████████  | 322942/400000 [00:36<00:08, 8897.84it/s] 81%|████████  | 323847/400000 [00:36<00:08, 8940.41it/s] 81%|████████  | 324774/400000 [00:36<00:08, 9035.97it/s] 81%|████████▏ | 325732/400000 [00:36<00:08, 9192.29it/s] 82%|████████▏ | 326653/400000 [00:36<00:08, 8971.52it/s] 82%|████████▏ | 327561/400000 [00:36<00:08, 9002.14it/s] 82%|████████▏ | 328485/400000 [00:37<00:07, 9070.34it/s] 82%|████████▏ | 329394/400000 [00:37<00:07, 8874.24it/s] 83%|████████▎ | 330284/400000 [00:37<00:07, 8878.59it/s] 83%|████████▎ | 331194/400000 [00:37<00:07, 8941.19it/s] 83%|████████▎ | 332144/400000 [00:37<00:07, 9101.35it/s] 83%|████████▎ | 333105/400000 [00:37<00:07, 9245.45it/s] 84%|████████▎ | 334060/400000 [00:37<00:07, 9331.76it/s] 84%|████████▎ | 334995/400000 [00:37<00:07, 9240.25it/s] 84%|████████▍ | 335921/400000 [00:37<00:07, 9109.51it/s] 84%|████████▍ | 336834/400000 [00:37<00:07, 8974.59it/s] 84%|████████▍ | 337768/400000 [00:38<00:06, 9078.84it/s] 85%|████████▍ | 338678/400000 [00:38<00:06, 8893.36it/s] 85%|████████▍ | 339570/400000 [00:38<00:06, 8804.34it/s] 85%|████████▌ | 340457/400000 [00:38<00:06, 8823.51it/s] 85%|████████▌ | 341359/400000 [00:38<00:06, 8881.10it/s] 86%|████████▌ | 342296/400000 [00:38<00:06, 9021.57it/s] 86%|████████▌ | 343220/400000 [00:38<00:06, 9085.96it/s] 86%|████████▌ | 344130/400000 [00:38<00:06, 8878.00it/s] 86%|████████▋ | 345020/400000 [00:38<00:06, 8804.01it/s] 86%|████████▋ | 345944/400000 [00:38<00:06, 8930.30it/s] 87%|████████▋ | 346898/400000 [00:39<00:05, 9102.95it/s] 87%|████████▋ | 347840/400000 [00:39<00:05, 9192.15it/s] 87%|████████▋ | 348761/400000 [00:39<00:05, 9129.16it/s] 87%|████████▋ | 349678/400000 [00:39<00:05, 9139.66it/s] 88%|████████▊ | 350596/400000 [00:39<00:05, 9151.20it/s] 88%|████████▊ | 351512/400000 [00:39<00:05, 8938.36it/s] 88%|████████▊ | 352408/400000 [00:39<00:05, 8794.40it/s] 88%|████████▊ | 353289/400000 [00:39<00:05, 8711.37it/s] 89%|████████▊ | 354162/400000 [00:39<00:05, 8650.26it/s] 89%|████████▉ | 355029/400000 [00:40<00:05, 8628.16it/s] 89%|████████▉ | 355893/400000 [00:40<00:05, 8599.83it/s] 89%|████████▉ | 356756/400000 [00:40<00:05, 8606.36it/s] 89%|████████▉ | 357651/400000 [00:40<00:04, 8706.12it/s] 90%|████████▉ | 358556/400000 [00:40<00:04, 8804.51it/s] 90%|████████▉ | 359455/400000 [00:40<00:04, 8857.53it/s] 90%|█████████ | 360393/400000 [00:40<00:04, 9006.27it/s] 90%|█████████ | 361295/400000 [00:40<00:04, 8973.08it/s] 91%|█████████ | 362194/400000 [00:40<00:04, 8910.09it/s] 91%|█████████ | 363086/400000 [00:40<00:04, 8859.32it/s] 91%|█████████ | 364001/400000 [00:41<00:04, 8942.91it/s] 91%|█████████ | 364914/400000 [00:41<00:03, 8998.02it/s] 91%|█████████▏| 365815/400000 [00:41<00:03, 8959.86it/s] 92%|█████████▏| 366712/400000 [00:41<00:03, 8739.19it/s] 92%|█████████▏| 367595/400000 [00:41<00:03, 8764.78it/s] 92%|█████████▏| 368473/400000 [00:41<00:03, 8738.43it/s] 92%|█████████▏| 369356/400000 [00:41<00:03, 8765.16it/s] 93%|█████████▎| 370234/400000 [00:41<00:03, 8710.98it/s] 93%|█████████▎| 371106/400000 [00:41<00:03, 8622.63it/s] 93%|█████████▎| 371969/400000 [00:41<00:03, 8618.72it/s] 93%|█████████▎| 372832/400000 [00:42<00:03, 8618.80it/s] 93%|█████████▎| 373697/400000 [00:42<00:03, 8626.92it/s] 94%|█████████▎| 374591/400000 [00:42<00:02, 8718.11it/s] 94%|█████████▍| 375477/400000 [00:42<00:02, 8757.35it/s] 94%|█████████▍| 376354/400000 [00:42<00:02, 8727.18it/s] 94%|█████████▍| 377227/400000 [00:42<00:02, 8696.33it/s] 95%|█████████▍| 378110/400000 [00:42<00:02, 8734.91it/s] 95%|█████████▍| 378988/400000 [00:42<00:02, 8746.96it/s] 95%|█████████▍| 379863/400000 [00:42<00:02, 8717.67it/s] 95%|█████████▌| 380735/400000 [00:42<00:02, 8704.62it/s] 95%|█████████▌| 381606/400000 [00:43<00:02, 8639.99it/s] 96%|█████████▌| 382471/400000 [00:43<00:02, 8589.82it/s] 96%|█████████▌| 383331/400000 [00:43<00:01, 8547.31it/s] 96%|█████████▌| 384186/400000 [00:43<00:01, 8516.54it/s] 96%|█████████▋| 385038/400000 [00:43<00:01, 8516.34it/s] 96%|█████████▋| 385890/400000 [00:43<00:01, 8505.03it/s] 97%|█████████▋| 386747/400000 [00:43<00:01, 8522.25it/s] 97%|█████████▋| 387600/400000 [00:43<00:01, 8515.86it/s] 97%|█████████▋| 388452/400000 [00:43<00:01, 8498.12it/s] 97%|█████████▋| 389304/400000 [00:43<00:01, 8502.59it/s] 98%|█████████▊| 390155/400000 [00:44<00:01, 8502.64it/s] 98%|█████████▊| 391006/400000 [00:44<00:01, 8447.88it/s] 98%|█████████▊| 391858/400000 [00:44<00:00, 8469.24it/s] 98%|█████████▊| 392706/400000 [00:44<00:00, 8458.88it/s] 98%|█████████▊| 393570/400000 [00:44<00:00, 8511.03it/s] 99%|█████████▊| 394482/400000 [00:44<00:00, 8683.92it/s] 99%|█████████▉| 395352/400000 [00:44<00:00, 8539.84it/s] 99%|█████████▉| 396208/400000 [00:44<00:00, 8445.55it/s] 99%|█████████▉| 397149/400000 [00:44<00:00, 8712.91it/s]100%|█████████▉| 398102/400000 [00:44<00:00, 8940.72it/s]100%|█████████▉| 399002/400000 [00:45<00:00, 8957.50it/s]100%|█████████▉| 399901/400000 [00:45<00:00, 8845.42it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8852.11it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fb38f0bfd30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010917734690490198 	 Accuracy: 56
Train Epoch: 1 	 Loss: 0.010978669227166319 	 Accuracy: 57

  model saves at 57% accuracy 

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
2020-05-13 07:25:18.893535: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-13 07:25:18.897408: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095230000 Hz
2020-05-13 07:25:18.897558: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5654faf1e010 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 07:25:18.897571: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fb3425fe160> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6206 - accuracy: 0.5030
 2000/25000 [=>............................] - ETA: 8s - loss: 7.3446 - accuracy: 0.5210 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.4571 - accuracy: 0.5137
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5133 - accuracy: 0.5100
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5930 - accuracy: 0.5048
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6002 - accuracy: 0.5043
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5987 - accuracy: 0.5044
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6015 - accuracy: 0.5042
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6084 - accuracy: 0.5038
11000/25000 [============>.................] - ETA: 3s - loss: 7.6206 - accuracy: 0.5030
12000/25000 [=============>................] - ETA: 3s - loss: 7.5989 - accuracy: 0.5044
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6053 - accuracy: 0.5040
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6228 - accuracy: 0.5029
15000/25000 [=================>............] - ETA: 2s - loss: 7.6247 - accuracy: 0.5027
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6264 - accuracy: 0.5026
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6242 - accuracy: 0.5028
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6198 - accuracy: 0.5031
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6456 - accuracy: 0.5014
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6482 - accuracy: 0.5012
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6761 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6766 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6762 - accuracy: 0.4994
25000/25000 [==============================] - 7s 270us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fb2f00bd6d8> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fb334b70e48> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4797 - crf_viterbi_accuracy: 0.1867 - val_loss: 1.4504 - val_crf_viterbi_accuracy: 0.1600

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

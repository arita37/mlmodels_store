
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f67a5830fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 08:13:21.064380
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 08:13:21.069012
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 08:13:21.072811
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 08:13:21.076549
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f67b15fa438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 361257.1562
Epoch 2/10

1/1 [==============================] - 0s 109ms/step - loss: 292090.5000
Epoch 3/10

1/1 [==============================] - 0s 106ms/step - loss: 205167.3750
Epoch 4/10

1/1 [==============================] - 0s 106ms/step - loss: 134925.0312
Epoch 5/10

1/1 [==============================] - 0s 109ms/step - loss: 83434.5625
Epoch 6/10

1/1 [==============================] - 0s 106ms/step - loss: 50452.9766
Epoch 7/10

1/1 [==============================] - 0s 106ms/step - loss: 30983.9746
Epoch 8/10

1/1 [==============================] - 0s 104ms/step - loss: 20097.2402
Epoch 9/10

1/1 [==============================] - 0s 103ms/step - loss: 13816.1699
Epoch 10/10

1/1 [==============================] - 0s 110ms/step - loss: 10069.6123

  #### Inference Need return ypred, ytrue ######################### 
[[ 0.23254901  0.56320727  1.5056695   1.3227506  -1.7882146   0.4865359
   0.10900787  1.120838   -0.8843367  -0.03547588 -0.3038328   0.01119886
  -0.11718059 -0.6855175  -0.22089332 -0.40040052 -1.3112627   0.26433975
  -0.5054625  -1.0563521  -0.28982452 -0.6423204  -1.0014199   1.0211817
   0.06838936  0.14179973  1.7336707   0.20850703 -1.2481627   0.42418772
  -0.58777213  0.9157593   0.11283892 -0.6363374  -1.6386054   0.5661715
  -1.134098    0.78987145 -0.7115917   0.36265197 -0.19937539 -0.46496725
   1.479569   -1.3680775   1.2620481  -0.74013495 -0.60233915 -0.6681988
   0.40254122  0.07983121  1.1346756   1.3014529   1.216764    0.860954
  -0.6694376  -0.17040274  0.28167677  0.5463224  -0.1195634   1.7157094
  -0.0611155   5.455275    6.197651    5.825171    5.9545546   6.5753093
   6.273388    6.8852386   6.585064    5.9566946   5.491623    6.939264
   6.1436024   6.549164    7.5178494   7.095612    6.9598784   6.7456794
   7.3082113   7.167801    7.100337    6.6160445   6.7424145   6.7205544
   7.786494    6.3681774   6.739038    7.191743    6.8348913   5.9810176
   6.745427    7.1940494   5.9007173   5.765512    7.470968    6.434917
   7.0726376   7.6466675   7.376421    6.438779    5.6702595   6.861045
   5.8409867   6.2392573   6.0242167   6.5758653   6.306349    6.7140436
   6.016286    6.595473    6.99797     6.9905357   5.8686953   5.94729
   6.2161646   6.4062133   6.257375    5.581414    6.7480817   6.1404686
  -0.30171907  0.55090094  0.13688524 -1.6497949  -0.82739526  1.0003113
   0.05329347 -0.52266884 -1.1745212   1.8953967   0.5649923   0.03119417
  -0.5842739   0.92233205 -0.7670493  -0.11455256  0.8929143  -0.5790374
  -0.56681484  0.09518047  0.29452527  0.53396326 -1.0177809   0.6571169
  -0.17664903 -0.5456499   1.4263493  -0.94526947 -0.93138725  0.02280056
   1.054497    1.0274861  -0.02261436 -0.3576583  -0.70162714 -0.7513213
  -0.33938754 -1.3736277  -1.0594875   0.16036788  0.31836239  1.1761671
   0.2533732  -0.031252    0.8057295   0.51016855 -1.623853    0.5812259
   0.25544327 -1.1508249   0.7311187  -1.1156948   0.9977356  -0.02759731
  -0.6634973  -0.04701275 -0.01603521 -0.30658796  0.09103078  1.1465179
   1.508408    1.8094639   0.8675867   1.5846579   2.2418501   1.790586
   0.35909718  1.4449438   0.666711    1.6041801   2.0314178   1.2054839
   0.22237879  1.0476162   1.3507948   1.8836683   1.2152824   1.8703729
   0.84305465  0.5949883   1.6814778   0.92106265  2.2839375   0.40121067
   1.0843654   0.36486006  1.4669167   0.6363919   1.1544029   0.6152675
   0.74375904  0.52197397  1.0987432   0.911769    0.38498926  1.4965775
   0.701336    0.77006847  0.66678256  0.5308707   1.5482711   0.69687617
   1.724247    0.87107396  0.34139073  1.4049869   0.1593234   1.0656767
   0.93574697  1.2783453   2.0442426   0.91281176  2.5257382   0.5958454
   2.2650142   1.2473114   1.5275266   1.1206917   0.8109908   1.6601443
   0.03711319  5.886781    7.4643846   6.1929784   6.115287    7.042504
   5.518395    5.8720965   7.360249    6.625376    7.155567    7.0499625
   6.9761343   6.3404098   6.7147303   6.9113827   8.012128    7.705638
   7.536744    6.554674    6.549916    5.536976    7.059766    6.188357
   7.2615232   6.292882    6.1233993   6.316972    7.383782    7.52207
   6.9289594   7.2666154   7.7478957   8.000974    7.3439217   7.6559205
   6.7988095   8.015073    7.806546    6.90435     5.731616    7.3871946
   7.883889    8.067749    6.7292767   5.994323    5.974078    5.3588543
   7.944899    6.854449    6.1421485   7.139077    7.507716    6.4284735
   7.1691074   6.240675    7.5126877   7.2447515   6.79355     6.8944645
   0.46406162  0.72470677  0.62536746  0.78889155  0.7536962   0.5475733
   0.63266385  0.40862787  0.71057403  0.73643404  0.95718163  1.4464458
   1.8160026   2.0134437   1.5052494   0.7445177   0.7962123   1.2533482
   0.36894894  2.1368275   0.6260688   1.9967544   1.0796447   0.5043279
   2.0510602   1.346527    1.2115028   0.6195719   1.6862624   1.1060165
   0.3213908   0.2776122   1.8648423   0.6218706   1.523174    0.504645
   0.9590937   1.0134656   1.4778073   1.4832664   1.5793576   2.3869302
   1.5185194   1.9239862   0.88951844  2.6145325   1.6062045   1.016654
   0.58951366  1.2225559   0.4650631   2.5461278   1.2158284   0.5739614
   0.5097746   1.4921637   1.2753024   0.6252289   2.0211654   1.8334951
  -2.4420047   6.4737716  -4.4525867 ]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 08:13:31.624831
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   96.1724
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 08:13:31.629678
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9266.25
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 08:13:31.634171
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   95.5339
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 08:13:31.637914
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -828.858
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140082776051840
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140080263381680
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140080263382184
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140080263382688
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140080263383192
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140080263383696

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f67ad479f28> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.548228
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.520373
grad_step = 000002, loss = 0.501097
grad_step = 000003, loss = 0.483293
grad_step = 000004, loss = 0.464688
grad_step = 000005, loss = 0.447352
grad_step = 000006, loss = 0.432219
grad_step = 000007, loss = 0.420664
grad_step = 000008, loss = 0.408619
grad_step = 000009, loss = 0.392716
grad_step = 000010, loss = 0.375204
grad_step = 000011, loss = 0.359491
grad_step = 000012, loss = 0.345113
grad_step = 000013, loss = 0.331225
grad_step = 000014, loss = 0.317718
grad_step = 000015, loss = 0.304767
grad_step = 000016, loss = 0.294816
grad_step = 000017, loss = 0.282953
grad_step = 000018, loss = 0.270461
grad_step = 000019, loss = 0.260550
grad_step = 000020, loss = 0.251708
grad_step = 000021, loss = 0.244589
grad_step = 000022, loss = 0.236924
grad_step = 000023, loss = 0.228466
grad_step = 000024, loss = 0.219692
grad_step = 000025, loss = 0.210942
grad_step = 000026, loss = 0.202997
grad_step = 000027, loss = 0.196152
grad_step = 000028, loss = 0.188701
grad_step = 000029, loss = 0.179914
grad_step = 000030, loss = 0.171357
grad_step = 000031, loss = 0.163021
grad_step = 000032, loss = 0.154633
grad_step = 000033, loss = 0.146631
grad_step = 000034, loss = 0.139498
grad_step = 000035, loss = 0.134065
grad_step = 000036, loss = 0.128980
grad_step = 000037, loss = 0.122448
grad_step = 000038, loss = 0.115660
grad_step = 000039, loss = 0.110075
grad_step = 000040, loss = 0.105507
grad_step = 000041, loss = 0.100805
grad_step = 000042, loss = 0.095930
grad_step = 000043, loss = 0.091126
grad_step = 000044, loss = 0.086380
grad_step = 000045, loss = 0.081939
grad_step = 000046, loss = 0.077829
grad_step = 000047, loss = 0.073756
grad_step = 000048, loss = 0.069845
grad_step = 000049, loss = 0.066265
grad_step = 000050, loss = 0.062818
grad_step = 000051, loss = 0.059345
grad_step = 000052, loss = 0.055829
grad_step = 000053, loss = 0.052498
grad_step = 000054, loss = 0.049610
grad_step = 000055, loss = 0.046991
grad_step = 000056, loss = 0.044318
grad_step = 000057, loss = 0.041650
grad_step = 000058, loss = 0.039177
grad_step = 000059, loss = 0.036896
grad_step = 000060, loss = 0.034709
grad_step = 000061, loss = 0.032632
grad_step = 000062, loss = 0.030711
grad_step = 000063, loss = 0.028862
grad_step = 000064, loss = 0.027029
grad_step = 000065, loss = 0.025328
grad_step = 000066, loss = 0.023809
grad_step = 000067, loss = 0.022401
grad_step = 000068, loss = 0.021047
grad_step = 000069, loss = 0.019767
grad_step = 000070, loss = 0.018561
grad_step = 000071, loss = 0.017400
grad_step = 000072, loss = 0.016332
grad_step = 000073, loss = 0.015376
grad_step = 000074, loss = 0.014463
grad_step = 000075, loss = 0.013582
grad_step = 000076, loss = 0.012773
grad_step = 000077, loss = 0.012024
grad_step = 000078, loss = 0.011320
grad_step = 000079, loss = 0.010677
grad_step = 000080, loss = 0.010072
grad_step = 000081, loss = 0.009494
grad_step = 000082, loss = 0.008972
grad_step = 000083, loss = 0.008494
grad_step = 000084, loss = 0.008037
grad_step = 000085, loss = 0.007613
grad_step = 000086, loss = 0.007219
grad_step = 000087, loss = 0.006849
grad_step = 000088, loss = 0.006516
grad_step = 000089, loss = 0.006201
grad_step = 000090, loss = 0.005906
grad_step = 000091, loss = 0.005637
grad_step = 000092, loss = 0.005385
grad_step = 000093, loss = 0.005150
grad_step = 000094, loss = 0.004929
grad_step = 000095, loss = 0.004719
grad_step = 000096, loss = 0.004527
grad_step = 000097, loss = 0.004348
grad_step = 000098, loss = 0.004179
grad_step = 000099, loss = 0.004021
grad_step = 000100, loss = 0.003872
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003737
grad_step = 000102, loss = 0.003610
grad_step = 000103, loss = 0.003486
grad_step = 000104, loss = 0.003367
grad_step = 000105, loss = 0.003255
grad_step = 000106, loss = 0.003153
grad_step = 000107, loss = 0.003060
grad_step = 000108, loss = 0.002975
grad_step = 000109, loss = 0.002897
grad_step = 000110, loss = 0.002823
grad_step = 000111, loss = 0.002749
grad_step = 000112, loss = 0.002675
grad_step = 000113, loss = 0.002606
grad_step = 000114, loss = 0.002546
grad_step = 000115, loss = 0.002492
grad_step = 000116, loss = 0.002446
grad_step = 000117, loss = 0.002406
grad_step = 000118, loss = 0.002377
grad_step = 000119, loss = 0.002370
grad_step = 000120, loss = 0.002379
grad_step = 000121, loss = 0.002374
grad_step = 000122, loss = 0.002305
grad_step = 000123, loss = 0.002212
grad_step = 000124, loss = 0.002163
grad_step = 000125, loss = 0.002171
grad_step = 000126, loss = 0.002189
grad_step = 000127, loss = 0.002168
grad_step = 000128, loss = 0.002112
grad_step = 000129, loss = 0.002069
grad_step = 000130, loss = 0.002064
grad_step = 000131, loss = 0.002080
grad_step = 000132, loss = 0.002082
grad_step = 000133, loss = 0.002059
grad_step = 000134, loss = 0.002024
grad_step = 000135, loss = 0.002004
grad_step = 000136, loss = 0.002004
grad_step = 000137, loss = 0.002014
grad_step = 000138, loss = 0.002020
grad_step = 000139, loss = 0.002014
grad_step = 000140, loss = 0.001997
grad_step = 000141, loss = 0.001978
grad_step = 000142, loss = 0.001965
grad_step = 000143, loss = 0.001960
grad_step = 000144, loss = 0.001961
grad_step = 000145, loss = 0.001967
grad_step = 000146, loss = 0.001975
grad_step = 000147, loss = 0.001984
grad_step = 000148, loss = 0.001994
grad_step = 000149, loss = 0.002005
grad_step = 000150, loss = 0.002013
grad_step = 000151, loss = 0.002014
grad_step = 000152, loss = 0.002002
grad_step = 000153, loss = 0.001980
grad_step = 000154, loss = 0.001953
grad_step = 000155, loss = 0.001931
grad_step = 000156, loss = 0.001920
grad_step = 000157, loss = 0.001920
grad_step = 000158, loss = 0.001928
grad_step = 000159, loss = 0.001940
grad_step = 000160, loss = 0.001951
grad_step = 000161, loss = 0.001960
grad_step = 000162, loss = 0.001964
grad_step = 000163, loss = 0.001962
grad_step = 000164, loss = 0.001952
grad_step = 000165, loss = 0.001936
grad_step = 000166, loss = 0.001917
grad_step = 000167, loss = 0.001901
grad_step = 000168, loss = 0.001890
grad_step = 000169, loss = 0.001885
grad_step = 000170, loss = 0.001884
grad_step = 000171, loss = 0.001887
grad_step = 000172, loss = 0.001892
grad_step = 000173, loss = 0.001899
grad_step = 000174, loss = 0.001910
grad_step = 000175, loss = 0.001923
grad_step = 000176, loss = 0.001938
grad_step = 000177, loss = 0.001951
grad_step = 000178, loss = 0.001960
grad_step = 000179, loss = 0.001954
grad_step = 000180, loss = 0.001931
grad_step = 000181, loss = 0.001896
grad_step = 000182, loss = 0.001864
grad_step = 000183, loss = 0.001847
grad_step = 000184, loss = 0.001850
grad_step = 000185, loss = 0.001864
grad_step = 000186, loss = 0.001878
grad_step = 000187, loss = 0.001885
grad_step = 000188, loss = 0.001879
grad_step = 000189, loss = 0.001862
grad_step = 000190, loss = 0.001844
grad_step = 000191, loss = 0.001828
grad_step = 000192, loss = 0.001821
grad_step = 000193, loss = 0.001821
grad_step = 000194, loss = 0.001826
grad_step = 000195, loss = 0.001832
grad_step = 000196, loss = 0.001838
grad_step = 000197, loss = 0.001841
grad_step = 000198, loss = 0.001844
grad_step = 000199, loss = 0.001844
grad_step = 000200, loss = 0.001835
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001818
grad_step = 000202, loss = 0.001809
grad_step = 000203, loss = 0.001806
grad_step = 000204, loss = 0.001796
grad_step = 000205, loss = 0.001786
grad_step = 000206, loss = 0.001785
grad_step = 000207, loss = 0.001787
grad_step = 000208, loss = 0.001785
grad_step = 000209, loss = 0.001780
grad_step = 000210, loss = 0.001779
grad_step = 000211, loss = 0.001785
grad_step = 000212, loss = 0.001792
grad_step = 000213, loss = 0.001800
grad_step = 000214, loss = 0.001811
grad_step = 000215, loss = 0.001829
grad_step = 000216, loss = 0.001857
grad_step = 000217, loss = 0.001895
grad_step = 000218, loss = 0.001937
grad_step = 000219, loss = 0.001962
grad_step = 000220, loss = 0.001943
grad_step = 000221, loss = 0.001874
grad_step = 000222, loss = 0.001797
grad_step = 000223, loss = 0.001775
grad_step = 000224, loss = 0.001794
grad_step = 000225, loss = 0.001845
grad_step = 000226, loss = 0.001822
grad_step = 000227, loss = 0.001803
grad_step = 000228, loss = 0.001802
grad_step = 000229, loss = 0.001748
grad_step = 000230, loss = 0.001742
grad_step = 000231, loss = 0.001764
grad_step = 000232, loss = 0.001783
grad_step = 000233, loss = 0.001789
grad_step = 000234, loss = 0.001743
grad_step = 000235, loss = 0.001723
grad_step = 000236, loss = 0.001733
grad_step = 000237, loss = 0.001750
grad_step = 000238, loss = 0.001764
grad_step = 000239, loss = 0.001744
grad_step = 000240, loss = 0.001726
grad_step = 000241, loss = 0.001716
grad_step = 000242, loss = 0.001717
grad_step = 000243, loss = 0.001729
grad_step = 000244, loss = 0.001734
grad_step = 000245, loss = 0.001737
grad_step = 000246, loss = 0.001726
grad_step = 000247, loss = 0.001715
grad_step = 000248, loss = 0.001705
grad_step = 000249, loss = 0.001700
grad_step = 000250, loss = 0.001700
grad_step = 000251, loss = 0.001704
grad_step = 000252, loss = 0.001711
grad_step = 000253, loss = 0.001715
grad_step = 000254, loss = 0.001723
grad_step = 000255, loss = 0.001712
grad_step = 000256, loss = 0.001707
grad_step = 000257, loss = 0.001698
grad_step = 000258, loss = 0.001691
grad_step = 000259, loss = 0.001685
grad_step = 000260, loss = 0.001682
grad_step = 000261, loss = 0.001681
grad_step = 000262, loss = 0.001683
grad_step = 000263, loss = 0.001692
grad_step = 000264, loss = 0.001697
grad_step = 000265, loss = 0.001717
grad_step = 000266, loss = 0.001699
grad_step = 000267, loss = 0.001702
grad_step = 000268, loss = 0.001713
grad_step = 000269, loss = 0.001731
grad_step = 000270, loss = 0.001749
grad_step = 000271, loss = 0.001763
grad_step = 000272, loss = 0.001766
grad_step = 000273, loss = 0.001755
grad_step = 000274, loss = 0.001733
grad_step = 000275, loss = 0.001709
grad_step = 000276, loss = 0.001696
grad_step = 000277, loss = 0.001718
grad_step = 000278, loss = 0.001681
grad_step = 000279, loss = 0.001669
grad_step = 000280, loss = 0.001652
grad_step = 000281, loss = 0.001666
grad_step = 000282, loss = 0.001698
grad_step = 000283, loss = 0.001669
grad_step = 000284, loss = 0.001692
grad_step = 000285, loss = 0.001722
grad_step = 000286, loss = 0.001650
grad_step = 000287, loss = 0.001736
grad_step = 000288, loss = 0.001851
grad_step = 000289, loss = 0.001704
grad_step = 000290, loss = 0.001935
grad_step = 000291, loss = 0.001822
grad_step = 000292, loss = 0.001858
grad_step = 000293, loss = 0.001782
grad_step = 000294, loss = 0.001823
grad_step = 000295, loss = 0.001856
grad_step = 000296, loss = 0.001781
grad_step = 000297, loss = 0.001839
grad_step = 000298, loss = 0.001830
grad_step = 000299, loss = 0.001796
grad_step = 000300, loss = 0.001817
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001816
grad_step = 000302, loss = 0.001775
grad_step = 000303, loss = 0.001788
grad_step = 000304, loss = 0.001779
grad_step = 000305, loss = 0.001749
grad_step = 000306, loss = 0.001756
grad_step = 000307, loss = 0.001748
grad_step = 000308, loss = 0.001736
grad_step = 000309, loss = 0.001733
grad_step = 000310, loss = 0.001736
grad_step = 000311, loss = 0.001725
grad_step = 000312, loss = 0.001728
grad_step = 000313, loss = 0.001726
grad_step = 000314, loss = 0.001724
grad_step = 000315, loss = 0.001723
grad_step = 000316, loss = 0.001729
grad_step = 000317, loss = 0.001727
grad_step = 000318, loss = 0.001737
grad_step = 000319, loss = 0.001754
grad_step = 000320, loss = 0.001769
grad_step = 000321, loss = 0.001799
grad_step = 000322, loss = 0.001835
grad_step = 000323, loss = 0.001863
grad_step = 000324, loss = 0.001887
grad_step = 000325, loss = 0.001890
grad_step = 000326, loss = 0.001879
grad_step = 000327, loss = 0.001851
grad_step = 000328, loss = 0.001796
grad_step = 000329, loss = 0.001739
grad_step = 000330, loss = 0.001708
grad_step = 000331, loss = 0.001705
grad_step = 000332, loss = 0.001726
grad_step = 000333, loss = 0.001752
grad_step = 000334, loss = 0.001753
grad_step = 000335, loss = 0.001725
grad_step = 000336, loss = 0.001691
grad_step = 000337, loss = 0.001674
grad_step = 000338, loss = 0.001684
grad_step = 000339, loss = 0.001703
grad_step = 000340, loss = 0.001709
grad_step = 000341, loss = 0.001695
grad_step = 000342, loss = 0.001674
grad_step = 000343, loss = 0.001661
grad_step = 000344, loss = 0.001664
grad_step = 000345, loss = 0.001674
grad_step = 000346, loss = 0.001679
grad_step = 000347, loss = 0.001673
grad_step = 000348, loss = 0.001661
grad_step = 000349, loss = 0.001652
grad_step = 000350, loss = 0.001649
grad_step = 000351, loss = 0.001650
grad_step = 000352, loss = 0.001652
grad_step = 000353, loss = 0.001651
grad_step = 000354, loss = 0.001647
grad_step = 000355, loss = 0.001642
grad_step = 000356, loss = 0.001637
grad_step = 000357, loss = 0.001633
grad_step = 000358, loss = 0.001629
grad_step = 000359, loss = 0.001626
grad_step = 000360, loss = 0.001623
grad_step = 000361, loss = 0.001620
grad_step = 000362, loss = 0.001618
grad_step = 000363, loss = 0.001615
grad_step = 000364, loss = 0.001612
grad_step = 000365, loss = 0.001610
grad_step = 000366, loss = 0.001608
grad_step = 000367, loss = 0.001603
grad_step = 000368, loss = 0.001600
grad_step = 000369, loss = 0.001596
grad_step = 000370, loss = 0.001591
grad_step = 000371, loss = 0.001588
grad_step = 000372, loss = 0.001586
grad_step = 000373, loss = 0.001583
grad_step = 000374, loss = 0.001579
grad_step = 000375, loss = 0.001578
grad_step = 000376, loss = 0.001576
grad_step = 000377, loss = 0.001574
grad_step = 000378, loss = 0.001571
grad_step = 000379, loss = 0.001569
grad_step = 000380, loss = 0.001567
grad_step = 000381, loss = 0.001565
grad_step = 000382, loss = 0.001564
grad_step = 000383, loss = 0.001564
grad_step = 000384, loss = 0.001566
grad_step = 000385, loss = 0.001573
grad_step = 000386, loss = 0.001586
grad_step = 000387, loss = 0.001606
grad_step = 000388, loss = 0.001637
grad_step = 000389, loss = 0.001707
grad_step = 000390, loss = 0.001826
grad_step = 000391, loss = 0.002016
grad_step = 000392, loss = 0.002129
grad_step = 000393, loss = 0.002111
grad_step = 000394, loss = 0.001886
grad_step = 000395, loss = 0.001656
grad_step = 000396, loss = 0.001606
grad_step = 000397, loss = 0.001745
grad_step = 000398, loss = 0.001826
grad_step = 000399, loss = 0.001684
grad_step = 000400, loss = 0.001565
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001634
grad_step = 000402, loss = 0.001707
grad_step = 000403, loss = 0.001654
grad_step = 000404, loss = 0.001573
grad_step = 000405, loss = 0.001561
grad_step = 000406, loss = 0.001617
grad_step = 000407, loss = 0.001644
grad_step = 000408, loss = 0.001589
grad_step = 000409, loss = 0.001539
grad_step = 000410, loss = 0.001562
grad_step = 000411, loss = 0.001594
grad_step = 000412, loss = 0.001574
grad_step = 000413, loss = 0.001543
grad_step = 000414, loss = 0.001548
grad_step = 000415, loss = 0.001563
grad_step = 000416, loss = 0.001557
grad_step = 000417, loss = 0.001541
grad_step = 000418, loss = 0.001536
grad_step = 000419, loss = 0.001541
grad_step = 000420, loss = 0.001547
grad_step = 000421, loss = 0.001541
grad_step = 000422, loss = 0.001530
grad_step = 000423, loss = 0.001528
grad_step = 000424, loss = 0.001533
grad_step = 000425, loss = 0.001534
grad_step = 000426, loss = 0.001528
grad_step = 000427, loss = 0.001524
grad_step = 000428, loss = 0.001525
grad_step = 000429, loss = 0.001525
grad_step = 000430, loss = 0.001522
grad_step = 000431, loss = 0.001521
grad_step = 000432, loss = 0.001521
grad_step = 000433, loss = 0.001518
grad_step = 000434, loss = 0.001515
grad_step = 000435, loss = 0.001515
grad_step = 000436, loss = 0.001517
grad_step = 000437, loss = 0.001516
grad_step = 000438, loss = 0.001512
grad_step = 000439, loss = 0.001510
grad_step = 000440, loss = 0.001510
grad_step = 000441, loss = 0.001510
grad_step = 000442, loss = 0.001509
grad_step = 000443, loss = 0.001507
grad_step = 000444, loss = 0.001507
grad_step = 000445, loss = 0.001506
grad_step = 000446, loss = 0.001504
grad_step = 000447, loss = 0.001503
grad_step = 000448, loss = 0.001502
grad_step = 000449, loss = 0.001502
grad_step = 000450, loss = 0.001501
grad_step = 000451, loss = 0.001500
grad_step = 000452, loss = 0.001500
grad_step = 000453, loss = 0.001499
grad_step = 000454, loss = 0.001499
grad_step = 000455, loss = 0.001498
grad_step = 000456, loss = 0.001497
grad_step = 000457, loss = 0.001498
grad_step = 000458, loss = 0.001499
grad_step = 000459, loss = 0.001503
grad_step = 000460, loss = 0.001511
grad_step = 000461, loss = 0.001526
grad_step = 000462, loss = 0.001557
grad_step = 000463, loss = 0.001611
grad_step = 000464, loss = 0.001694
grad_step = 000465, loss = 0.001793
grad_step = 000466, loss = 0.001839
grad_step = 000467, loss = 0.001783
grad_step = 000468, loss = 0.001666
grad_step = 000469, loss = 0.001581
grad_step = 000470, loss = 0.001566
grad_step = 000471, loss = 0.001575
grad_step = 000472, loss = 0.001582
grad_step = 000473, loss = 0.001554
grad_step = 000474, loss = 0.001536
grad_step = 000475, loss = 0.001551
grad_step = 000476, loss = 0.001571
grad_step = 000477, loss = 0.001557
grad_step = 000478, loss = 0.001503
grad_step = 000479, loss = 0.001486
grad_step = 000480, loss = 0.001514
grad_step = 000481, loss = 0.001531
grad_step = 000482, loss = 0.001523
grad_step = 000483, loss = 0.001499
grad_step = 000484, loss = 0.001496
grad_step = 000485, loss = 0.001503
grad_step = 000486, loss = 0.001492
grad_step = 000487, loss = 0.001480
grad_step = 000488, loss = 0.001484
grad_step = 000489, loss = 0.001496
grad_step = 000490, loss = 0.001496
grad_step = 000491, loss = 0.001482
grad_step = 000492, loss = 0.001474
grad_step = 000493, loss = 0.001476
grad_step = 000494, loss = 0.001476
grad_step = 000495, loss = 0.001473
grad_step = 000496, loss = 0.001473
grad_step = 000497, loss = 0.001473
grad_step = 000498, loss = 0.001473
grad_step = 000499, loss = 0.001472
grad_step = 000500, loss = 0.001469
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001465
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

  date_run                              2020-05-13 08:13:56.101112
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.224166
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 08:13:56.108242
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.124787
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 08:13:56.117041
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.136554
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 08:13:56.122631
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.896188
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
0   2020-05-13 08:13:21.064380  ...    mean_absolute_error
1   2020-05-13 08:13:21.069012  ...     mean_squared_error
2   2020-05-13 08:13:21.072811  ...  median_absolute_error
3   2020-05-13 08:13:21.076549  ...               r2_score
4   2020-05-13 08:13:31.624831  ...    mean_absolute_error
5   2020-05-13 08:13:31.629678  ...     mean_squared_error
6   2020-05-13 08:13:31.634171  ...  median_absolute_error
7   2020-05-13 08:13:31.637914  ...               r2_score
8   2020-05-13 08:13:56.101112  ...    mean_absolute_error
9   2020-05-13 08:13:56.108242  ...     mean_squared_error
10  2020-05-13 08:13:56.117041  ...  median_absolute_error
11  2020-05-13 08:13:56.122631  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fada2087be0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  1%|          | 98304/9912422 [00:00<00:12, 793541.19it/s] 11%|█▏        | 1122304/9912422 [00:00<00:08, 1097077.44it/s] 74%|███████▍  | 7372800/9912422 [00:00<00:01, 1554724.06it/s]9920512it [00:00, 18829426.45it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 207227.93it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  5%|▍         | 81920/1648877 [00:00<00:01, 814290.92it/s] 38%|███▊      | 622592/1648877 [00:00<00:00, 1092153.83it/s]1654784it [00:00, 4137116.83it/s]                            
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 60741.13it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fad54a42eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fad540710f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fad54a42eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fada2092ba8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fad51802518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fada2092ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fad54a42eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fada2092ba8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fad51802518> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fada2092ba8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7ff89aec4240> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=1ca921bad8556c059bdc27ad330ef4aabdcfff0bd5b08dce8a3c35ceedef2d3a
  Stored in directory: /tmp/pip-ephem-wheel-cache-i339fihu/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7ff832cbf6a0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1810432/17464789 [==>...........................] - ETA: 0s
 6471680/17464789 [==========>...................] - ETA: 0s
13500416/17464789 [======================>.......] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 08:15:25.711453: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 08:15:25.716175: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 08:15:25.716443: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b9e5afc0d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 08:15:25.716461: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.6360 - accuracy: 0.5020
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7740 - accuracy: 0.4930
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.8455 - accuracy: 0.4883 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7970 - accuracy: 0.4915
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7801 - accuracy: 0.4926
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7663 - accuracy: 0.4935
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7564 - accuracy: 0.4941
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7395 - accuracy: 0.4952
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7314 - accuracy: 0.4958
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6988 - accuracy: 0.4979
11000/25000 [============>.................] - ETA: 4s - loss: 7.7307 - accuracy: 0.4958
12000/25000 [=============>................] - ETA: 4s - loss: 7.7241 - accuracy: 0.4963
13000/25000 [==============>...............] - ETA: 4s - loss: 7.7020 - accuracy: 0.4977
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6940 - accuracy: 0.4982
15000/25000 [=================>............] - ETA: 3s - loss: 7.7004 - accuracy: 0.4978
16000/25000 [==================>...........] - ETA: 3s - loss: 7.7011 - accuracy: 0.4978
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6991 - accuracy: 0.4979
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7033 - accuracy: 0.4976
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6924 - accuracy: 0.4983
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6781 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6739 - accuracy: 0.4995
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6827 - accuracy: 0.4990
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6700 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
25000/25000 [==============================] - 10s 401us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 08:15:43.579968
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 08:15:43.579968  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:44:00, 11.6kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:44:46, 16.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:22:33, 23.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:16:17, 32.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.61M/862M [00:01<5:04:37, 47.0kB/s].vector_cache/glove.6B.zip:   1%|          | 8.00M/862M [00:01<3:32:14, 67.1kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:01<2:27:58, 95.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:01<1:42:58, 137kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.6M/862M [00:01<1:11:40, 195kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.6M/862M [00:01<50:08, 278kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.7M/862M [00:01<34:57, 396kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.3M/862M [00:02<24:29, 563kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.9M/862M [00:02<17:08, 799kB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:02<12:03, 1.13MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.9M/862M [00:02<08:27, 1.60MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<06:38, 2.03MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:04<06:32, 2.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:05<06:45, 1.98MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:05<05:16, 2.54MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<06:01, 2.22MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:07<05:58, 2.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.2M/862M [00:07<04:37, 2.88MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<05:48, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:09<06:48, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.9M/862M [00:09<05:27, 2.43MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:09<03:58, 3.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:10<1:50:25, 120kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.4M/862M [00:10<1:18:36, 168kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:11<55:15, 239kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<41:38, 316kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<31:49, 413kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:13<22:48, 576kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:13<16:07, 813kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.2M/862M [00:14<16:50, 777kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:14<13:08, 995kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.2M/862M [00:15<09:31, 1.37MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<09:42, 1.34MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<07:53, 1.65MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:16<05:47, 2.24MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:17<04:13, 3.07MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<55:08, 235kB/s] .vector_cache/glove.6B.zip:  10%|▉         | 85.7M/862M [00:18<41:13, 314kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:18<29:25, 439kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:19<20:39, 624kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:20<37:28, 344kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<27:31, 468kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.5M/862M [00:20<19:33, 657kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:22<16:39, 769kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:22<14:22, 891kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.7M/862M [00:22<10:42, 1.19MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:23<07:37, 1.67MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<1:36:27, 132kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<1:08:48, 185kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:24<48:23, 263kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<36:45, 345kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<27:02, 468kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<19:13, 658kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<16:22, 770kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<14:01, 899kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<10:23, 1.21MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<07:23, 1.70MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<17:55, 699kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<13:50, 905kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<09:57, 1.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<09:51, 1.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<09:26, 1.32MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:14, 1.72MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<05:10, 2.39MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<1:46:07, 117kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<1:15:32, 164kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<53:05, 233kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<39:55, 309kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<30:26, 405kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<21:51, 563kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<15:23, 797kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<20:17, 604kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<15:26, 794kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<11:02, 1.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<10:34, 1.15MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:38, 1.41MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<06:20, 1.92MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:18, 1.66MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:20, 1.91MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<04:41, 2.58MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:09, 1.96MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<06:47, 1.78MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<05:15, 2.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:44<03:48, 3.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<11:51, 1.01MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<09:33, 1.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<06:58, 1.71MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<07:40, 1.55MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:35, 1.81MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:54, 2.42MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<06:13, 1.90MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:34, 2.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:09, 2.84MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<05:41, 2.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:12, 2.26MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<03:56, 2.98MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:31, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<05:03, 2.31MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<03:50, 3.05MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:26, 2.14MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:00, 2.33MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:47, 3.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:23, 2.15MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<06:07, 1.89MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:46, 2.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<03:29, 3.30MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<08:34, 1.34MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<07:10, 1.60MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<05:18, 2.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:23, 1.79MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<06:48, 1.68MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:16, 2.17MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:50, 2.96MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:44, 1.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:34, 1.73MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<04:50, 2.34MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:02, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:27, 2.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:05, 2.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:20, 2.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<06:15, 1.80MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<04:59, 2.25MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<03:35, 3.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<14:21, 778kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<11:14, 992kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<08:09, 1.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<08:09, 1.36MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<08:03, 1.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:14, 1.78MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<04:29, 2.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<20:01, 550kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<15:13, 723kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<10:53, 1.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<10:01, 1.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<08:12, 1.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:15<05:59, 1.82MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:35, 1.65MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:56, 1.57MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:27, 1.99MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:56, 2.74MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<19:17, 561kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<14:40, 737kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<10:32, 1.02MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<09:44, 1.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<09:11, 1.17MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<06:56, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:58, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<11:11, 954kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<08:59, 1.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<06:34, 1.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<06:55, 1.53MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<07:12, 1.47MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<05:31, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<04:01, 2.62MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<06:36, 1.59MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:46, 1.83MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<04:18, 2.44MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:19, 1.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:57, 1.76MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<04:38, 2.25MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<03:22, 3.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<07:13, 1.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:10, 1.68MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<04:34, 2.26MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<05:28, 1.89MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:57, 2.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<03:44, 2.75MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:52, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:35, 1.83MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<04:27, 2.29MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<03:15, 3.14MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<17:53, 569kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<13:39, 746kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<09:46, 1.04MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<09:02, 1.12MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<08:34, 1.18MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<06:27, 1.57MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<04:41, 2.15MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<06:19, 1.59MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<05:20, 1.88MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<04:01, 2.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:41<02:57, 3.37MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<13:03, 764kB/s] .vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<11:21, 879kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<08:24, 1.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<05:58, 1.66MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<13:28, 735kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<10:30, 942kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 269M/862M [01:45<07:34, 1.30MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<07:27, 1.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<07:18, 1.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<05:34, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<03:59, 2.45MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<09:32, 1.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<07:44, 1.26MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<05:40, 1.72MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<06:06, 1.59MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 280M/862M [01:51<06:25, 1.51MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<04:57, 1.96MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<03:33, 2.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<10:38, 905kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<08:30, 1.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<06:09, 1.56MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<06:24, 1.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:55<06:36, 1.45MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<05:03, 1.89MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<03:39, 2.59MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<06:33, 1.45MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<05:37, 1.69MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<04:08, 2.28MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:58, 1.89MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<05:30, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:17, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<03:06, 3.01MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<06:41, 1.40MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<05:42, 1.64MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<04:14, 2.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:00, 1.85MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<05:30, 1.68MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<04:16, 2.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<03:06, 2.96MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:52, 1.57MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:07, 1.80MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<03:49, 2.40MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:41, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<05:20, 1.71MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:09, 2.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<03:01, 3.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<06:37, 1.37MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<05:27, 1.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<04:03, 2.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:50, 1.86MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<05:23, 1.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:11, 2.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<03:02, 2.94MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<06:36, 1.35MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<05:35, 1.60MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<04:07, 2.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<04:51, 1.83MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<04:21, 2.03MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<03:16, 2.69MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<04:13, 2.08MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:54, 1.79MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<03:50, 2.29MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:17<02:47, 3.14MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<06:46, 1.29MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:41, 1.53MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:10, 2.08MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:50, 1.79MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<04:18, 2.01MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<03:13, 2.68MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:11, 2.05MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<04:52, 1.76MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<03:48, 2.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<02:47, 3.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:12, 1.64MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:34, 1.86MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:25, 2.48MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:15, 1.98MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<04:47, 1.76MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<03:44, 2.26MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<02:43, 3.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<05:08, 1.63MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:30, 1.86MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:22, 2.48MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:11, 1.98MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:42, 1.77MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:39, 2.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<02:40, 3.09MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<05:17, 1.56MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:35, 1.79MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:25, 2.39MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:12, 1.94MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<04:45, 1.72MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:45, 2.17MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<02:43, 2.98MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<14:06, 574kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<10:42, 756kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<07:41, 1.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<07:12, 1.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<06:45, 1.19MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:08, 1.56MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:39<03:41, 2.16MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<26:31, 300kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<19:24, 410kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<13:45, 577kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<11:21, 695kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<09:42, 813kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<07:09, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<05:05, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<07:31, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:56, 1.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<04:18, 1.81MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:45<03:07, 2.49MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<19:30, 398kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<14:29, 535kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<10:19, 748kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<08:53, 864kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<07:52, 976kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:55, 1.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:49<04:13, 1.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<14:36, 521kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<11:02, 690kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<07:54, 960kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<07:14, 1.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<06:44, 1.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:07, 1.47MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:53<03:39, 2.05MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<13:50, 541kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<10:29, 712kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<07:31, 990kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<06:52, 1.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<06:24, 1.16MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:49, 1.53MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<03:27, 2.13MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<06:19, 1.16MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<05:13, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:49, 1.91MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:17, 1.70MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:37, 1.57MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:34, 2.03MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<02:37, 2.76MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:06, 1.75MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:38, 1.98MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:42, 2.65MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:31, 2.02MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<03:59, 1.78MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:07, 2.27MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:05<02:15, 3.13MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<06:32, 1.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<05:21, 1.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:54, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<04:16, 1.64MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:28, 1.56MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:27, 2.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:09<02:30, 2.77MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<05:02, 1.37MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<04:16, 1.62MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:08, 2.19MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:43, 1.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:21, 2.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:31, 2.70MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:15, 2.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:15<03:01, 2.24MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:16, 2.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:04, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:17<03:35, 1.87MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:49, 2.37MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:17<02:04, 3.21MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:42, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<03:19, 2.00MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:29, 2.65MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:11, 2.06MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<03:25, 1.92MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<02:45, 2.38MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<02:03, 3.18MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<03:11, 2.04MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<02:56, 2.21MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:13, 2.91MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<02:58, 2.16MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<03:30, 1.83MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<02:48, 2.29MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:25<02:01, 3.15MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<11:01, 578kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<08:24, 757kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<06:00, 1.05MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<05:34, 1.13MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<05:17, 1.19MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<03:59, 1.57MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<02:52, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:10, 1.49MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:35, 1.73MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<02:39, 2.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:13, 1.91MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:53, 2.12MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:09, 2.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<02:55, 2.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:21, 1.81MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:38, 2.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<01:53, 3.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<06:29, 927kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<05:11, 1.16MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:45, 1.59MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:56, 1.51MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:57, 1.50MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:02, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<02:10, 2.71MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<08:54, 661kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<06:51, 856kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<04:56, 1.18MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<04:43, 1.23MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<03:56, 1.47MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:54, 1.99MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:17, 1.74MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:31, 1.63MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:46, 2.06MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<01:59, 2.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<08:30, 667kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<06:27, 878kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<04:40, 1.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<03:19, 1.69MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<11:59, 468kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<09:37, 582kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<07:01, 796kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<04:56, 1.12MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<10:23, 533kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:50<07:46, 712kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<05:34, 989kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<05:04, 1.08MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:44, 1.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:33, 1.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<02:31, 2.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<06:59, 771kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<05:30, 980kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<03:57, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:55, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:55, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:01, 1.75MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<02:10, 2.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<09:49, 535kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<07:27, 705kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<05:19, 981kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<04:51, 1.07MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<04:30, 1.15MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<03:23, 1.53MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:00<02:25, 2.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:14, 1.21MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:06, 1.24MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<03:08, 1.62MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:14, 2.26MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<07:26, 679kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<05:45, 876kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<04:08, 1.21MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<03:58, 1.25MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:13, 1.54MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<02:20, 2.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<01:43, 2.86MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<16:35, 296kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<12:41, 387kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<09:05, 538kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:08<06:23, 761kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<06:44, 719kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<05:08, 942kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<03:41, 1.30MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:10<02:37, 1.82MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<10:04, 474kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<08:03, 592kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<05:53, 808kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:12<04:08, 1.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<10:06, 466kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<07:34, 621kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<05:23, 867kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<04:48, 963kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<04:21, 1.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:15, 1.42MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:16<02:18, 1.98MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<10:53, 420kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<08:05, 563kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<05:45, 788kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<05:00, 898kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:20<03:58, 1.13MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<02:52, 1.55MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<03:01, 1.46MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<03:03, 1.45MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:20, 1.89MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:22<01:40, 2.62MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<05:23, 810kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<04:14, 1.03MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<03:03, 1.42MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<03:04, 1.40MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:36, 1.64MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:54, 2.23MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:15, 1.87MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:28, 1.70MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:56, 2.17MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:28<01:23, 2.99MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:42, 1.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<03:02, 1.37MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:13, 1.86MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:28, 1.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<02:36, 1.57MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<02:00, 2.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<01:28, 2.75MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:34<02:12, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<01:59, 2.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:29, 2.68MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:36<01:54, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<02:10, 1.81MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<01:44, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:36<01:15, 3.10MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<06:48, 570kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<05:10, 748kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:41, 1.04MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<03:23, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<03:13, 1.18MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:27, 1.54MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:40<01:45, 2.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<07:05, 527kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:42<05:20, 698kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<03:48, 975kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<03:28, 1.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:44<03:12, 1.14MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<02:24, 1.51MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:43, 2.10MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:46<02:49, 1.28MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 647M/862M [04:46<02:21, 1.52MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<01:43, 2.07MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:48<01:58, 1.79MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:48<02:09, 1.63MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<01:41, 2.07MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<01:13, 2.84MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<06:09, 562kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<04:40, 738kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<03:20, 1.02MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<03:03, 1.11MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<02:30, 1.35MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<01:49, 1.84MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:00, 1.66MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<02:06, 1.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<01:36, 2.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<01:10, 2.80MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<02:10, 1.50MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:56<01:51, 1.75MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:22, 2.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<02:03, 1.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<04:04, 781kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:59<03:35, 884kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:59<02:41, 1.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:53, 1.64MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:01<04:30, 690kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<03:49, 814kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:01<02:50, 1.09MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:00, 1.52MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<02:32, 1.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<02:03, 1.48MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:03<01:35, 1.89MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:08, 2.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<00:51, 3.42MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:06<02:30, 1.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:06<05:43, 515kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:06<06:25, 459kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:06<06:48, 433kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:06<06:34, 448kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<05:30, 534kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:06<04:04, 720kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:06<02:55, 998kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:08<02:37, 1.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:08<02:29, 1.15MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:08<01:55, 1.49MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:08<01:24, 2.02MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:10<01:37, 1.73MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:10<01:44, 1.61MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:10<01:21, 2.05MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:11<00:59, 2.78MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:12<01:33, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:12<01:44, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:12<02:15, 1.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:12<02:12, 1.23MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<01:40, 1.63MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<01:13, 2.22MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<01:31, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<01:32, 1.72MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:14<01:25, 1.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:14<01:10, 2.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:15<00:55, 2.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:15<00:43, 3.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<00:35, 4.36MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:16<01:53, 1.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:16<01:37, 1.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<01:19, 1.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:16<00:59, 2.60MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<00:44, 3.43MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<01:53, 1.34MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:18<01:46, 1.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:18<01:20, 1.87MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:20<01:20, 1.84MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:20<01:14, 1.98MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:20<00:55, 2.64MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:22<01:07, 2.13MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:22<01:17, 1.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:22<01:00, 2.35MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:22<00:45, 3.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:24<01:08, 2.02MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:24<01:03, 2.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:24<00:47, 2.92MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:26<01:02, 2.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:26<01:12, 1.87MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:26<00:57, 2.33MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<00:41, 3.19MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:28<03:49, 571kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:28<02:54, 749kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:28<02:04, 1.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<01:53, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:30<01:46, 1.19MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:30<01:19, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<00:56, 2.19MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:32<01:28, 1.39MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:32<01:15, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:32<00:55, 2.19MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:34<01:04, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:34<01:09, 1.72MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:34<00:53, 2.19MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<00:38, 3.02MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<01:44, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:36<01:25, 1.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:36<01:01, 1.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:38<01:06, 1.66MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:38<01:10, 1.57MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:38<00:54, 2.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:38, 2.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:40<01:11, 1.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:40<00:59, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:40<00:44, 2.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:31, 3.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<02:01, 840kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:42<01:47, 945kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:42<01:19, 1.27MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<00:55, 1.77MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:44<01:29, 1.10MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:44<01:12, 1.34MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:44<00:52, 1.83MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:46<00:57, 1.63MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:46<01:00, 1.56MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:46<00:46, 1.98MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:32, 2.73MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<02:37, 571kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:48<01:59, 750kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:48<01:24, 1.04MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<01:17, 1.11MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:50<01:13, 1.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:50<00:55, 1.53MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:38, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<01:53, 718kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<01:37, 836kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:52<01:11, 1.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:49, 1.58MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<01:37, 790kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<01:16, 1.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:54<00:55, 1.36MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:39, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<01:03, 1.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<00:59, 1.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:56<00:47, 1.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:56<00:34, 2.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:38, 1.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<00:34, 2.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:58<00:25, 2.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<00:31, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<00:28, 2.25MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:00<00:20, 2.99MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:02<00:29, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:02<00:34, 1.75MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:02<00:26, 2.26MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:02<00:19, 2.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:04<00:25, 2.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:04<00:23, 2.43MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:04<00:17, 3.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:12, 4.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<01:00, 866kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:47, 1.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:33, 1.50MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:23, 2.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<04:38, 172kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<03:24, 234kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:08<02:23, 329kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:08<01:38, 464kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<01:16, 573kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:10<00:58, 747kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:40, 1.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:35, 1.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<00:28, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:12<00:20, 1.86MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<00:21, 1.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<00:22, 1.58MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<00:16, 2.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:14<00:12, 2.74MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:16, 1.89MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:14, 2.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:16<00:10, 2.79MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:07, 3.76MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:40, 680kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:30, 891kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:18<00:21, 1.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.69MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:32, 721kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:27, 843kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:19, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:13, 1.58MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:13, 1.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:22<00:11, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:22<00:07, 2.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:07, 1.84MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:24<00:08, 1.65MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:24<00:06, 2.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:24<00:04, 2.88MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:26<00:05, 1.85MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:26<00:05, 2.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:03, 2.68MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:26<00:01, 3.65MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:28<00:08, 776kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:28<00:06, 987kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:28<00:03, 1.37MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:30<00:01, 1.36MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:30<00:01, 1.35MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:00, 1.77MB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 758/400000 [00:00<00:52, 7578.27it/s]  0%|          | 1534/400000 [00:00<00:52, 7629.59it/s]  1%|          | 2296/400000 [00:00<00:52, 7624.60it/s]  1%|          | 3013/400000 [00:00<00:53, 7482.21it/s]  1%|          | 3734/400000 [00:00<00:53, 7397.65it/s]  1%|          | 4469/400000 [00:00<00:53, 7382.11it/s]  1%|▏         | 5235/400000 [00:00<00:52, 7462.50it/s]  2%|▏         | 6008/400000 [00:00<00:52, 7540.21it/s]  2%|▏         | 6754/400000 [00:00<00:52, 7512.46it/s]  2%|▏         | 7506/400000 [00:01<00:52, 7513.09it/s]  2%|▏         | 8237/400000 [00:01<00:53, 7376.05it/s]  2%|▏         | 8988/400000 [00:01<00:52, 7414.84it/s]  2%|▏         | 9743/400000 [00:01<00:52, 7454.29it/s]  3%|▎         | 10482/400000 [00:01<00:52, 7404.86it/s]  3%|▎         | 11241/400000 [00:01<00:52, 7457.59it/s]  3%|▎         | 11987/400000 [00:01<00:52, 7453.98it/s]  3%|▎         | 12755/400000 [00:01<00:51, 7520.29it/s]  3%|▎         | 13528/400000 [00:01<00:50, 7580.19it/s]  4%|▎         | 14286/400000 [00:01<00:51, 7534.05it/s]  4%|▍         | 15058/400000 [00:02<00:50, 7588.23it/s]  4%|▍         | 15817/400000 [00:02<00:50, 7554.12it/s]  4%|▍         | 16597/400000 [00:02<00:50, 7625.77it/s]  4%|▍         | 17360/400000 [00:02<00:50, 7615.37it/s]  5%|▍         | 18138/400000 [00:02<00:49, 7661.60it/s]  5%|▍         | 18905/400000 [00:02<00:50, 7557.93it/s]  5%|▍         | 19664/400000 [00:02<00:50, 7567.50it/s]  5%|▌         | 20473/400000 [00:02<00:49, 7716.64it/s]  5%|▌         | 21246/400000 [00:02<00:49, 7602.98it/s]  6%|▌         | 22008/400000 [00:02<00:50, 7535.86it/s]  6%|▌         | 22770/400000 [00:03<00:49, 7559.13it/s]  6%|▌         | 23527/400000 [00:03<00:49, 7541.92it/s]  6%|▌         | 24282/400000 [00:03<00:49, 7525.10it/s]  6%|▋         | 25041/400000 [00:03<00:49, 7544.07it/s]  6%|▋         | 25796/400000 [00:03<00:49, 7533.89it/s]  7%|▋         | 26550/400000 [00:03<00:49, 7490.25it/s]  7%|▋         | 27300/400000 [00:03<00:50, 7445.34it/s]  7%|▋         | 28076/400000 [00:03<00:49, 7534.62it/s]  7%|▋         | 28873/400000 [00:03<00:48, 7657.88it/s]  7%|▋         | 29640/400000 [00:03<00:48, 7601.45it/s]  8%|▊         | 30442/400000 [00:04<00:47, 7720.38it/s]  8%|▊         | 31218/400000 [00:04<00:47, 7731.01it/s]  8%|▊         | 32017/400000 [00:04<00:47, 7804.89it/s]  8%|▊         | 32818/400000 [00:04<00:46, 7863.38it/s]  8%|▊         | 33605/400000 [00:04<00:47, 7731.00it/s]  9%|▊         | 34379/400000 [00:04<00:48, 7577.06it/s]  9%|▉         | 35138/400000 [00:04<00:48, 7534.74it/s]  9%|▉         | 35941/400000 [00:04<00:47, 7675.07it/s]  9%|▉         | 36723/400000 [00:04<00:47, 7717.93it/s]  9%|▉         | 37496/400000 [00:04<00:47, 7594.99it/s] 10%|▉         | 38260/400000 [00:05<00:47, 7607.20it/s] 10%|▉         | 39022/400000 [00:05<00:48, 7507.71it/s] 10%|▉         | 39796/400000 [00:05<00:47, 7573.98it/s] 10%|█         | 40567/400000 [00:05<00:47, 7613.67it/s] 10%|█         | 41341/400000 [00:05<00:46, 7648.71it/s] 11%|█         | 42107/400000 [00:05<00:47, 7506.68it/s] 11%|█         | 42859/400000 [00:05<00:47, 7500.75it/s] 11%|█         | 43610/400000 [00:05<00:47, 7501.31it/s] 11%|█         | 44361/400000 [00:05<00:47, 7453.64it/s] 11%|█▏        | 45107/400000 [00:05<00:48, 7368.13it/s] 11%|█▏        | 45896/400000 [00:06<00:47, 7516.81it/s] 12%|█▏        | 46649/400000 [00:06<00:47, 7501.18it/s] 12%|█▏        | 47400/400000 [00:06<00:47, 7461.31it/s] 12%|█▏        | 48207/400000 [00:06<00:46, 7633.60it/s] 12%|█▏        | 48972/400000 [00:06<00:46, 7612.18it/s] 12%|█▏        | 49735/400000 [00:06<00:46, 7557.06it/s] 13%|█▎        | 50492/400000 [00:06<00:46, 7515.30it/s] 13%|█▎        | 51249/400000 [00:06<00:46, 7530.97it/s] 13%|█▎        | 52037/400000 [00:06<00:45, 7630.22it/s] 13%|█▎        | 52801/400000 [00:06<00:45, 7596.30it/s] 13%|█▎        | 53571/400000 [00:07<00:45, 7625.07it/s] 14%|█▎        | 54335/400000 [00:07<00:45, 7628.17it/s] 14%|█▍        | 55099/400000 [00:07<00:45, 7597.19it/s] 14%|█▍        | 55871/400000 [00:07<00:45, 7632.08it/s] 14%|█▍        | 56635/400000 [00:07<00:45, 7607.37it/s] 14%|█▍        | 57396/400000 [00:07<00:45, 7538.57it/s] 15%|█▍        | 58151/400000 [00:07<00:46, 7394.13it/s] 15%|█▍        | 58923/400000 [00:07<00:45, 7488.88it/s] 15%|█▍        | 59677/400000 [00:07<00:45, 7503.25it/s] 15%|█▌        | 60448/400000 [00:07<00:44, 7562.78it/s] 15%|█▌        | 61205/400000 [00:08<00:44, 7541.38it/s] 16%|█▌        | 62015/400000 [00:08<00:43, 7699.39it/s] 16%|█▌        | 62813/400000 [00:08<00:43, 7778.65it/s] 16%|█▌        | 63620/400000 [00:08<00:42, 7863.75it/s] 16%|█▌        | 64411/400000 [00:08<00:42, 7877.41it/s] 16%|█▋        | 65203/400000 [00:08<00:42, 7889.19it/s] 16%|█▋        | 65993/400000 [00:08<00:42, 7882.25it/s] 17%|█▋        | 66782/400000 [00:08<00:42, 7840.54it/s] 17%|█▋        | 67567/400000 [00:08<00:42, 7804.63it/s] 17%|█▋        | 68348/400000 [00:08<00:42, 7804.45it/s] 17%|█▋        | 69140/400000 [00:09<00:42, 7837.76it/s] 17%|█▋        | 69924/400000 [00:09<00:42, 7701.20it/s] 18%|█▊        | 70733/400000 [00:09<00:42, 7813.09it/s] 18%|█▊        | 71522/400000 [00:09<00:41, 7834.75it/s] 18%|█▊        | 72307/400000 [00:09<00:41, 7810.79it/s] 18%|█▊        | 73089/400000 [00:09<00:42, 7649.13it/s] 18%|█▊        | 73863/400000 [00:09<00:42, 7676.14it/s] 19%|█▊        | 74632/400000 [00:09<00:42, 7646.35it/s] 19%|█▉        | 75398/400000 [00:09<00:42, 7614.48it/s] 19%|█▉        | 76169/400000 [00:10<00:42, 7642.50it/s] 19%|█▉        | 76934/400000 [00:10<00:42, 7592.05it/s] 19%|█▉        | 77694/400000 [00:10<00:42, 7572.78it/s] 20%|█▉        | 78470/400000 [00:10<00:42, 7625.31it/s] 20%|█▉        | 79241/400000 [00:10<00:41, 7650.44it/s] 20%|██        | 80035/400000 [00:10<00:41, 7731.68it/s] 20%|██        | 80809/400000 [00:10<00:41, 7602.32it/s] 20%|██        | 81570/400000 [00:10<00:41, 7597.74it/s] 21%|██        | 82331/400000 [00:10<00:42, 7521.95it/s] 21%|██        | 83084/400000 [00:10<00:42, 7495.61it/s] 21%|██        | 83870/400000 [00:11<00:41, 7598.03it/s] 21%|██        | 84631/400000 [00:11<00:41, 7600.86it/s] 21%|██▏       | 85403/400000 [00:11<00:41, 7634.09it/s] 22%|██▏       | 86171/400000 [00:11<00:41, 7647.29it/s] 22%|██▏       | 86936/400000 [00:11<00:41, 7566.98it/s] 22%|██▏       | 87694/400000 [00:11<00:41, 7474.51it/s] 22%|██▏       | 88442/400000 [00:11<00:44, 6945.24it/s] 22%|██▏       | 89164/400000 [00:11<00:44, 7023.06it/s] 22%|██▏       | 89926/400000 [00:11<00:43, 7190.78it/s] 23%|██▎       | 90655/400000 [00:11<00:42, 7219.06it/s] 23%|██▎       | 91408/400000 [00:12<00:42, 7306.30it/s] 23%|██▎       | 92166/400000 [00:12<00:41, 7385.33it/s] 23%|██▎       | 92943/400000 [00:12<00:40, 7496.29it/s] 23%|██▎       | 93727/400000 [00:12<00:40, 7596.15it/s] 24%|██▎       | 94489/400000 [00:12<00:40, 7579.36it/s] 24%|██▍       | 95249/400000 [00:12<00:40, 7565.98it/s] 24%|██▍       | 96007/400000 [00:12<00:40, 7529.10it/s] 24%|██▍       | 96798/400000 [00:12<00:39, 7638.19it/s] 24%|██▍       | 97573/400000 [00:12<00:39, 7669.82it/s] 25%|██▍       | 98341/400000 [00:12<00:39, 7555.89it/s] 25%|██▍       | 99099/400000 [00:13<00:39, 7560.73it/s] 25%|██▍       | 99856/400000 [00:13<00:40, 7465.75it/s] 25%|██▌       | 100604/400000 [00:13<00:40, 7374.94it/s] 25%|██▌       | 101343/400000 [00:13<00:40, 7339.84it/s] 26%|██▌       | 102108/400000 [00:13<00:40, 7429.00it/s] 26%|██▌       | 102857/400000 [00:13<00:39, 7446.34it/s] 26%|██▌       | 103603/400000 [00:13<00:39, 7447.19it/s] 26%|██▌       | 104349/400000 [00:13<00:39, 7428.18it/s] 26%|██▋       | 105115/400000 [00:13<00:39, 7494.84it/s] 26%|██▋       | 105873/400000 [00:13<00:39, 7519.78it/s] 27%|██▋       | 106635/400000 [00:14<00:38, 7547.55it/s] 27%|██▋       | 107390/400000 [00:14<00:39, 7439.93it/s] 27%|██▋       | 108135/400000 [00:14<00:39, 7418.05it/s] 27%|██▋       | 108887/400000 [00:14<00:39, 7445.69it/s] 27%|██▋       | 109632/400000 [00:14<00:40, 7247.53it/s] 28%|██▊       | 110368/400000 [00:14<00:39, 7279.32it/s] 28%|██▊       | 111097/400000 [00:14<00:40, 7173.71it/s] 28%|██▊       | 111866/400000 [00:14<00:39, 7318.74it/s] 28%|██▊       | 112618/400000 [00:14<00:38, 7377.10it/s] 28%|██▊       | 113383/400000 [00:15<00:38, 7455.31it/s] 29%|██▊       | 114167/400000 [00:15<00:37, 7566.01it/s] 29%|██▊       | 114928/400000 [00:15<00:37, 7578.20it/s] 29%|██▉       | 115698/400000 [00:15<00:37, 7613.96it/s] 29%|██▉       | 116472/400000 [00:15<00:37, 7650.07it/s] 29%|██▉       | 117243/400000 [00:15<00:36, 7666.39it/s] 30%|██▉       | 118014/400000 [00:15<00:36, 7676.49it/s] 30%|██▉       | 118782/400000 [00:15<00:37, 7586.00it/s] 30%|██▉       | 119541/400000 [00:15<00:37, 7454.11it/s] 30%|███       | 120288/400000 [00:15<00:37, 7412.71it/s] 30%|███       | 121030/400000 [00:16<00:37, 7381.60it/s] 30%|███       | 121808/400000 [00:16<00:37, 7494.86it/s] 31%|███       | 122581/400000 [00:16<00:36, 7560.15it/s] 31%|███       | 123357/400000 [00:16<00:36, 7618.42it/s] 31%|███       | 124137/400000 [00:16<00:35, 7670.95it/s] 31%|███       | 124905/400000 [00:16<00:35, 7666.17it/s] 31%|███▏      | 125672/400000 [00:16<00:35, 7639.40it/s] 32%|███▏      | 126437/400000 [00:16<00:35, 7613.97it/s] 32%|███▏      | 127199/400000 [00:16<00:36, 7560.92it/s] 32%|███▏      | 127956/400000 [00:16<00:36, 7443.94it/s] 32%|███▏      | 128701/400000 [00:17<00:36, 7422.49it/s] 32%|███▏      | 129451/400000 [00:17<00:36, 7443.21it/s] 33%|███▎      | 130202/400000 [00:17<00:36, 7462.30it/s] 33%|███▎      | 130975/400000 [00:17<00:35, 7540.49it/s] 33%|███▎      | 131754/400000 [00:17<00:35, 7611.03it/s] 33%|███▎      | 132516/400000 [00:17<00:35, 7537.28it/s] 33%|███▎      | 133271/400000 [00:17<00:35, 7487.05it/s] 34%|███▎      | 134021/400000 [00:17<00:35, 7472.98it/s] 34%|███▎      | 134769/400000 [00:17<00:35, 7433.64it/s] 34%|███▍      | 135513/400000 [00:17<00:35, 7365.11it/s] 34%|███▍      | 136267/400000 [00:18<00:35, 7413.72it/s] 34%|███▍      | 137015/400000 [00:18<00:35, 7430.14it/s] 34%|███▍      | 137759/400000 [00:18<00:35, 7353.69it/s] 35%|███▍      | 138495/400000 [00:18<00:35, 7290.57it/s] 35%|███▍      | 139249/400000 [00:18<00:35, 7360.96it/s] 35%|███▍      | 139986/400000 [00:18<00:35, 7354.59it/s] 35%|███▌      | 140722/400000 [00:18<00:35, 7298.54it/s] 35%|███▌      | 141473/400000 [00:18<00:35, 7359.93it/s] 36%|███▌      | 142210/400000 [00:18<00:35, 7347.43it/s] 36%|███▌      | 142956/400000 [00:18<00:34, 7380.44it/s] 36%|███▌      | 143707/400000 [00:19<00:34, 7416.64it/s] 36%|███▌      | 144451/400000 [00:19<00:34, 7423.13it/s] 36%|███▋      | 145221/400000 [00:19<00:33, 7501.63it/s] 36%|███▋      | 145972/400000 [00:19<00:33, 7496.60it/s] 37%|███▋      | 146722/400000 [00:19<00:34, 7420.81it/s] 37%|███▋      | 147483/400000 [00:19<00:33, 7473.91it/s] 37%|███▋      | 148231/400000 [00:19<00:33, 7440.09it/s] 37%|███▋      | 149045/400000 [00:19<00:32, 7635.06it/s] 37%|███▋      | 149826/400000 [00:19<00:32, 7684.82it/s] 38%|███▊      | 150596/400000 [00:19<00:32, 7671.62it/s] 38%|███▊      | 151370/400000 [00:20<00:32, 7690.85it/s] 38%|███▊      | 152160/400000 [00:20<00:31, 7751.68it/s] 38%|███▊      | 152941/400000 [00:20<00:31, 7766.74it/s] 38%|███▊      | 153719/400000 [00:20<00:32, 7650.25it/s] 39%|███▊      | 154485/400000 [00:20<00:32, 7625.04it/s] 39%|███▉      | 155248/400000 [00:20<00:32, 7545.35it/s] 39%|███▉      | 156004/400000 [00:20<00:32, 7501.37it/s] 39%|███▉      | 156773/400000 [00:20<00:32, 7553.16it/s] 39%|███▉      | 157529/400000 [00:20<00:32, 7534.81it/s] 40%|███▉      | 158283/400000 [00:20<00:32, 7522.37it/s] 40%|███▉      | 159057/400000 [00:21<00:31, 7585.08it/s] 40%|███▉      | 159816/400000 [00:21<00:32, 7504.17it/s] 40%|████      | 160567/400000 [00:21<00:32, 7424.81it/s] 40%|████      | 161310/400000 [00:21<00:32, 7264.52it/s] 41%|████      | 162048/400000 [00:21<00:32, 7297.63it/s] 41%|████      | 162794/400000 [00:21<00:32, 7344.18it/s] 41%|████      | 163552/400000 [00:21<00:31, 7412.33it/s] 41%|████      | 164296/400000 [00:21<00:31, 7419.16it/s] 41%|████▏     | 165041/400000 [00:21<00:31, 7426.83it/s] 41%|████▏     | 165784/400000 [00:21<00:31, 7377.09it/s] 42%|████▏     | 166527/400000 [00:22<00:31, 7391.37it/s] 42%|████▏     | 167301/400000 [00:22<00:31, 7492.00it/s] 42%|████▏     | 168062/400000 [00:22<00:30, 7524.26it/s] 42%|████▏     | 168815/400000 [00:22<00:31, 7316.04it/s] 42%|████▏     | 169566/400000 [00:22<00:31, 7372.70it/s] 43%|████▎     | 170312/400000 [00:22<00:31, 7398.15it/s] 43%|████▎     | 171085/400000 [00:22<00:30, 7492.61it/s] 43%|████▎     | 171840/400000 [00:22<00:30, 7507.67it/s] 43%|████▎     | 172592/400000 [00:22<00:30, 7437.86it/s] 43%|████▎     | 173337/400000 [00:23<00:30, 7340.89it/s] 44%|████▎     | 174089/400000 [00:23<00:30, 7391.72it/s] 44%|████▎     | 174885/400000 [00:23<00:29, 7551.33it/s] 44%|████▍     | 175659/400000 [00:23<00:29, 7603.75it/s] 44%|████▍     | 176422/400000 [00:23<00:29, 7608.91it/s] 44%|████▍     | 177184/400000 [00:23<00:29, 7599.88it/s] 44%|████▍     | 177945/400000 [00:23<00:29, 7545.81it/s] 45%|████▍     | 178700/400000 [00:23<00:29, 7445.03it/s] 45%|████▍     | 179486/400000 [00:23<00:29, 7563.60it/s] 45%|████▌     | 180244/400000 [00:23<00:29, 7481.83it/s] 45%|████▌     | 181000/400000 [00:24<00:29, 7502.58it/s] 45%|████▌     | 181754/400000 [00:24<00:29, 7513.44it/s] 46%|████▌     | 182512/400000 [00:24<00:28, 7531.76it/s] 46%|████▌     | 183266/400000 [00:24<00:29, 7427.94it/s] 46%|████▌     | 184021/400000 [00:24<00:28, 7463.94it/s] 46%|████▌     | 184812/400000 [00:24<00:28, 7591.89it/s] 46%|████▋     | 185573/400000 [00:24<00:28, 7593.32it/s] 47%|████▋     | 186333/400000 [00:24<00:28, 7540.54it/s] 47%|████▋     | 187088/400000 [00:24<00:28, 7529.54it/s] 47%|████▋     | 187842/400000 [00:24<00:28, 7462.85it/s] 47%|████▋     | 188603/400000 [00:25<00:28, 7503.49it/s] 47%|████▋     | 189358/400000 [00:25<00:28, 7516.57it/s] 48%|████▊     | 190110/400000 [00:25<00:28, 7488.43it/s] 48%|████▊     | 190887/400000 [00:25<00:27, 7569.91it/s] 48%|████▊     | 191645/400000 [00:25<00:27, 7442.76it/s] 48%|████▊     | 192390/400000 [00:25<00:27, 7416.51it/s] 48%|████▊     | 193138/400000 [00:25<00:27, 7435.26it/s] 48%|████▊     | 193899/400000 [00:25<00:27, 7486.74it/s] 49%|████▊     | 194649/400000 [00:25<00:27, 7481.30it/s] 49%|████▉     | 195398/400000 [00:25<00:27, 7424.87it/s] 49%|████▉     | 196141/400000 [00:26<00:27, 7340.68it/s] 49%|████▉     | 196876/400000 [00:26<00:27, 7343.25it/s] 49%|████▉     | 197611/400000 [00:26<00:27, 7301.42it/s] 50%|████▉     | 198342/400000 [00:26<00:27, 7300.10it/s] 50%|████▉     | 199073/400000 [00:26<00:27, 7287.82it/s] 50%|████▉     | 199802/400000 [00:26<00:27, 7199.04it/s] 50%|█████     | 200523/400000 [00:26<00:27, 7161.41it/s] 50%|█████     | 201259/400000 [00:26<00:27, 7218.85it/s] 50%|█████     | 201983/400000 [00:26<00:27, 7219.79it/s] 51%|█████     | 202706/400000 [00:26<00:27, 7173.11it/s] 51%|█████     | 203424/400000 [00:27<00:27, 7077.36it/s] 51%|█████     | 204167/400000 [00:27<00:27, 7177.07it/s] 51%|█████     | 204902/400000 [00:27<00:26, 7227.68it/s] 51%|█████▏    | 205626/400000 [00:27<00:26, 7210.85it/s] 52%|█████▏    | 206348/400000 [00:27<00:26, 7188.65it/s] 52%|█████▏    | 207110/400000 [00:27<00:26, 7311.73it/s] 52%|█████▏    | 207850/400000 [00:27<00:26, 7335.55it/s] 52%|█████▏    | 208594/400000 [00:27<00:25, 7364.07it/s] 52%|█████▏    | 209331/400000 [00:27<00:25, 7356.21it/s] 53%|█████▎    | 210067/400000 [00:27<00:26, 7292.75it/s] 53%|█████▎    | 210799/400000 [00:28<00:25, 7300.15it/s] 53%|█████▎    | 211537/400000 [00:28<00:25, 7321.43it/s] 53%|█████▎    | 212278/400000 [00:28<00:25, 7345.18it/s] 53%|█████▎    | 213041/400000 [00:28<00:25, 7427.30it/s] 53%|█████▎    | 213788/400000 [00:28<00:25, 7437.76it/s] 54%|█████▎    | 214547/400000 [00:28<00:24, 7481.30it/s] 54%|█████▍    | 215296/400000 [00:28<00:24, 7470.28it/s] 54%|█████▍    | 216071/400000 [00:28<00:24, 7550.68it/s] 54%|█████▍    | 216827/400000 [00:28<00:24, 7509.93it/s] 54%|█████▍    | 217579/400000 [00:28<00:24, 7437.55it/s] 55%|█████▍    | 218358/400000 [00:29<00:24, 7538.72it/s] 55%|█████▍    | 219138/400000 [00:29<00:23, 7614.66it/s] 55%|█████▍    | 219901/400000 [00:29<00:23, 7592.89it/s] 55%|█████▌    | 220698/400000 [00:29<00:23, 7700.66it/s] 55%|█████▌    | 221469/400000 [00:29<00:23, 7624.47it/s] 56%|█████▌    | 222258/400000 [00:29<00:23, 7699.80it/s] 56%|█████▌    | 223029/400000 [00:29<00:23, 7638.49it/s] 56%|█████▌    | 223804/400000 [00:29<00:22, 7671.23it/s] 56%|█████▌    | 224572/400000 [00:29<00:23, 7401.57it/s] 56%|█████▋    | 225315/400000 [00:30<00:24, 7268.17it/s] 57%|█████▋    | 226064/400000 [00:30<00:23, 7331.46it/s] 57%|█████▋    | 226807/400000 [00:30<00:23, 7360.22it/s] 57%|█████▋    | 227607/400000 [00:30<00:22, 7540.50it/s] 57%|█████▋    | 228392/400000 [00:30<00:22, 7630.41it/s] 57%|█████▋    | 229179/400000 [00:30<00:22, 7698.56it/s] 57%|█████▋    | 229963/400000 [00:30<00:21, 7736.46it/s] 58%|█████▊    | 230750/400000 [00:30<00:21, 7774.35it/s] 58%|█████▊    | 231529/400000 [00:30<00:22, 7606.36it/s] 58%|█████▊    | 232291/400000 [00:30<00:22, 7537.81it/s] 58%|█████▊    | 233057/400000 [00:31<00:22, 7571.33it/s] 58%|█████▊    | 233877/400000 [00:31<00:21, 7749.22it/s] 59%|█████▊    | 234667/400000 [00:31<00:21, 7792.29it/s] 59%|█████▉    | 235448/400000 [00:31<00:21, 7728.14it/s] 59%|█████▉    | 236238/400000 [00:31<00:21, 7778.32it/s] 59%|█████▉    | 237017/400000 [00:31<00:21, 7738.88it/s] 59%|█████▉    | 237792/400000 [00:31<00:20, 7730.90it/s] 60%|█████▉    | 238593/400000 [00:31<00:20, 7810.85it/s] 60%|█████▉    | 239375/400000 [00:31<00:20, 7735.95it/s] 60%|██████    | 240161/400000 [00:31<00:20, 7771.74it/s] 60%|██████    | 240939/400000 [00:32<00:20, 7656.44it/s] 60%|██████    | 241714/400000 [00:32<00:20, 7683.40it/s] 61%|██████    | 242483/400000 [00:32<00:20, 7664.25it/s] 61%|██████    | 243250/400000 [00:32<00:20, 7664.42it/s] 61%|██████    | 244017/400000 [00:32<00:20, 7576.88it/s] 61%|██████    | 244776/400000 [00:32<00:20, 7503.65it/s] 61%|██████▏   | 245527/400000 [00:32<00:20, 7486.90it/s] 62%|██████▏   | 246277/400000 [00:32<00:20, 7441.56it/s] 62%|██████▏   | 247022/400000 [00:32<00:20, 7407.56it/s] 62%|██████▏   | 247765/400000 [00:32<00:20, 7413.72it/s] 62%|██████▏   | 248507/400000 [00:33<00:20, 7362.26it/s] 62%|██████▏   | 249244/400000 [00:33<00:20, 7328.85it/s] 62%|██████▏   | 249978/400000 [00:33<00:20, 7314.29it/s] 63%|██████▎   | 250726/400000 [00:33<00:20, 7362.36it/s] 63%|██████▎   | 251470/400000 [00:33<00:20, 7382.41it/s] 63%|██████▎   | 252242/400000 [00:33<00:19, 7480.06it/s] 63%|██████▎   | 253002/400000 [00:33<00:19, 7515.27it/s] 63%|██████▎   | 253779/400000 [00:33<00:19, 7588.42it/s] 64%|██████▎   | 254540/400000 [00:33<00:19, 7593.70it/s] 64%|██████▍   | 255300/400000 [00:33<00:19, 7539.41it/s] 64%|██████▍   | 256055/400000 [00:34<00:19, 7505.28it/s] 64%|██████▍   | 256825/400000 [00:34<00:18, 7561.46it/s] 64%|██████▍   | 257582/400000 [00:34<00:18, 7560.76it/s] 65%|██████▍   | 258339/400000 [00:34<00:18, 7560.09it/s] 65%|██████▍   | 259096/400000 [00:34<00:18, 7555.87it/s] 65%|██████▍   | 259852/400000 [00:34<00:18, 7494.84it/s] 65%|██████▌   | 260616/400000 [00:34<00:18, 7535.61it/s] 65%|██████▌   | 261370/400000 [00:34<00:18, 7494.31it/s] 66%|██████▌   | 262138/400000 [00:34<00:18, 7546.79it/s] 66%|██████▌   | 262906/400000 [00:34<00:18, 7584.68it/s] 66%|██████▌   | 263665/400000 [00:35<00:18, 7537.27it/s] 66%|██████▌   | 264419/400000 [00:35<00:18, 7497.06it/s] 66%|██████▋   | 265177/400000 [00:35<00:17, 7520.23it/s] 66%|██████▋   | 265935/400000 [00:35<00:17, 7536.40it/s] 67%|██████▋   | 266694/400000 [00:35<00:17, 7550.63it/s] 67%|██████▋   | 267450/400000 [00:35<00:17, 7521.87it/s] 67%|██████▋   | 268203/400000 [00:35<00:17, 7514.20it/s] 67%|██████▋   | 268956/400000 [00:35<00:17, 7517.32it/s] 67%|██████▋   | 269708/400000 [00:35<00:17, 7451.21it/s] 68%|██████▊   | 270466/400000 [00:35<00:17, 7487.03it/s] 68%|██████▊   | 271228/400000 [00:36<00:17, 7525.19it/s] 68%|██████▊   | 271981/400000 [00:36<00:17, 7524.60it/s] 68%|██████▊   | 272737/400000 [00:36<00:16, 7533.46it/s] 68%|██████▊   | 273491/400000 [00:36<00:16, 7519.58it/s] 69%|██████▊   | 274244/400000 [00:36<00:16, 7485.16it/s] 69%|██████▊   | 274996/400000 [00:36<00:16, 7494.64it/s] 69%|██████▉   | 275746/400000 [00:36<00:16, 7492.47it/s] 69%|██████▉   | 276507/400000 [00:36<00:16, 7525.49it/s] 69%|██████▉   | 277260/400000 [00:36<00:16, 7516.54it/s] 70%|██████▉   | 278047/400000 [00:36<00:16, 7617.51it/s] 70%|██████▉   | 278810/400000 [00:37<00:16, 7489.06it/s] 70%|██████▉   | 279560/400000 [00:37<00:16, 7250.53it/s] 70%|███████   | 280288/400000 [00:37<00:16, 7249.36it/s] 70%|███████   | 281015/400000 [00:37<00:16, 7255.08it/s] 70%|███████   | 281742/400000 [00:37<00:16, 7227.31it/s] 71%|███████   | 282466/400000 [00:37<00:16, 7225.33it/s] 71%|███████   | 283227/400000 [00:37<00:15, 7334.31it/s] 71%|███████   | 284015/400000 [00:37<00:15, 7488.91it/s] 71%|███████   | 284782/400000 [00:37<00:15, 7540.95it/s] 71%|███████▏  | 285558/400000 [00:37<00:15, 7603.98it/s] 72%|███████▏  | 286320/400000 [00:38<00:15, 7524.67it/s] 72%|███████▏  | 287082/400000 [00:38<00:14, 7551.90it/s] 72%|███████▏  | 287838/400000 [00:38<00:14, 7548.43it/s] 72%|███████▏  | 288618/400000 [00:38<00:14, 7619.61it/s] 72%|███████▏  | 289389/400000 [00:38<00:14, 7642.82it/s] 73%|███████▎  | 290154/400000 [00:38<00:14, 7597.00it/s] 73%|███████▎  | 290914/400000 [00:38<00:14, 7546.40it/s] 73%|███████▎  | 291698/400000 [00:38<00:14, 7630.47it/s] 73%|███████▎  | 292462/400000 [00:38<00:14, 7590.42it/s] 73%|███████▎  | 293263/400000 [00:39<00:13, 7709.27it/s] 74%|███████▎  | 294065/400000 [00:39<00:13, 7798.96it/s] 74%|███████▎  | 294846/400000 [00:39<00:13, 7788.21it/s] 74%|███████▍  | 295648/400000 [00:39<00:13, 7854.08it/s] 74%|███████▍  | 296434/400000 [00:39<00:13, 7791.38it/s] 74%|███████▍  | 297214/400000 [00:39<00:13, 7769.31it/s] 74%|███████▍  | 297992/400000 [00:39<00:13, 7741.43it/s] 75%|███████▍  | 298767/400000 [00:39<00:13, 7645.28it/s] 75%|███████▍  | 299532/400000 [00:39<00:13, 7545.23it/s] 75%|███████▌  | 300288/400000 [00:39<00:13, 7472.71it/s] 75%|███████▌  | 301036/400000 [00:40<00:13, 7400.07it/s] 75%|███████▌  | 301777/400000 [00:40<00:13, 7394.10it/s] 76%|███████▌  | 302517/400000 [00:40<00:13, 7163.00it/s] 76%|███████▌  | 303262/400000 [00:40<00:13, 7246.64it/s] 76%|███████▌  | 303989/400000 [00:40<00:13, 7210.99it/s] 76%|███████▌  | 304730/400000 [00:40<00:13, 7268.43it/s] 76%|███████▋  | 305470/400000 [00:40<00:12, 7307.31it/s] 77%|███████▋  | 306207/400000 [00:40<00:12, 7323.83it/s] 77%|███████▋  | 306950/400000 [00:40<00:12, 7352.76it/s] 77%|███████▋  | 307701/400000 [00:40<00:12, 7399.20it/s] 77%|███████▋  | 308444/400000 [00:41<00:12, 7406.70it/s] 77%|███████▋  | 309189/400000 [00:41<00:12, 7417.84it/s] 77%|███████▋  | 309931/400000 [00:41<00:12, 7350.05it/s] 78%|███████▊  | 310667/400000 [00:41<00:12, 7338.24it/s] 78%|███████▊  | 311402/400000 [00:41<00:12, 7315.00it/s] 78%|███████▊  | 312146/400000 [00:41<00:11, 7351.03it/s] 78%|███████▊  | 312897/400000 [00:41<00:11, 7397.24it/s] 78%|███████▊  | 313648/400000 [00:41<00:11, 7429.54it/s] 79%|███████▊  | 314402/400000 [00:41<00:11, 7460.46it/s] 79%|███████▉  | 315149/400000 [00:41<00:11, 7434.08it/s] 79%|███████▉  | 315893/400000 [00:42<00:11, 7422.94it/s] 79%|███████▉  | 316636/400000 [00:42<00:11, 7384.52it/s] 79%|███████▉  | 317375/400000 [00:42<00:11, 7386.12it/s] 80%|███████▉  | 318126/400000 [00:42<00:11, 7421.56it/s] 80%|███████▉  | 318884/400000 [00:42<00:10, 7468.29it/s] 80%|███████▉  | 319640/400000 [00:42<00:10, 7494.45it/s] 80%|████████  | 320396/400000 [00:42<00:10, 7512.50it/s] 80%|████████  | 321149/400000 [00:42<00:10, 7515.75it/s] 80%|████████  | 321907/400000 [00:42<00:10, 7534.83it/s] 81%|████████  | 322665/400000 [00:42<00:10, 7547.46it/s] 81%|████████  | 323423/400000 [00:43<00:10, 7554.73it/s] 81%|████████  | 324184/400000 [00:43<00:10, 7570.02it/s] 81%|████████  | 324942/400000 [00:43<00:09, 7524.83it/s] 81%|████████▏ | 325695/400000 [00:43<00:09, 7515.26it/s] 82%|████████▏ | 326447/400000 [00:43<00:10, 7311.17it/s] 82%|████████▏ | 327189/400000 [00:43<00:09, 7343.17it/s] 82%|████████▏ | 327927/400000 [00:43<00:09, 7352.89it/s] 82%|████████▏ | 328668/400000 [00:43<00:09, 7368.01it/s] 82%|████████▏ | 329419/400000 [00:43<00:09, 7409.35it/s] 83%|████████▎ | 330169/400000 [00:43<00:09, 7434.68it/s] 83%|████████▎ | 330918/400000 [00:44<00:09, 7449.16it/s] 83%|████████▎ | 331671/400000 [00:44<00:09, 7472.18it/s] 83%|████████▎ | 332424/400000 [00:44<00:09, 7487.09it/s] 83%|████████▎ | 333185/400000 [00:44<00:08, 7521.59it/s] 83%|████████▎ | 333946/400000 [00:44<00:08, 7547.05it/s] 84%|████████▎ | 334714/400000 [00:44<00:08, 7585.15it/s] 84%|████████▍ | 335473/400000 [00:44<00:08, 7569.54it/s] 84%|████████▍ | 336231/400000 [00:44<00:08, 7473.41it/s] 84%|████████▍ | 336980/400000 [00:44<00:08, 7476.41it/s] 84%|████████▍ | 337728/400000 [00:44<00:08, 7462.02it/s] 85%|████████▍ | 338475/400000 [00:45<00:08, 7435.26it/s] 85%|████████▍ | 339219/400000 [00:45<00:08, 7177.72it/s] 85%|████████▍ | 339975/400000 [00:45<00:08, 7286.60it/s] 85%|████████▌ | 340719/400000 [00:45<00:08, 7329.42it/s] 85%|████████▌ | 341486/400000 [00:45<00:07, 7425.99it/s] 86%|████████▌ | 342247/400000 [00:45<00:07, 7479.51it/s] 86%|████████▌ | 342996/400000 [00:45<00:07, 7481.49it/s] 86%|████████▌ | 343747/400000 [00:45<00:07, 7489.41it/s] 86%|████████▌ | 344505/400000 [00:45<00:07, 7514.08it/s] 86%|████████▋ | 345259/400000 [00:45<00:07, 7518.97it/s] 87%|████████▋ | 346012/400000 [00:46<00:07, 7402.29it/s] 87%|████████▋ | 346754/400000 [00:46<00:07, 7404.73it/s] 87%|████████▋ | 347495/400000 [00:46<00:07, 7343.69it/s] 87%|████████▋ | 348232/400000 [00:46<00:07, 7350.28it/s] 87%|████████▋ | 348968/400000 [00:46<00:06, 7305.61it/s] 87%|████████▋ | 349699/400000 [00:46<00:06, 7304.22it/s] 88%|████████▊ | 350430/400000 [00:46<00:06, 7305.91it/s] 88%|████████▊ | 351161/400000 [00:46<00:06, 7248.43it/s] 88%|████████▊ | 351887/400000 [00:46<00:06, 7172.13it/s] 88%|████████▊ | 352627/400000 [00:46<00:06, 7238.92it/s] 88%|████████▊ | 353373/400000 [00:47<00:06, 7301.86it/s] 89%|████████▊ | 354113/400000 [00:47<00:06, 7330.92it/s] 89%|████████▊ | 354847/400000 [00:47<00:06, 7325.53it/s] 89%|████████▉ | 355586/400000 [00:47<00:06, 7342.15it/s] 89%|████████▉ | 356332/400000 [00:47<00:05, 7375.02it/s] 89%|████████▉ | 357071/400000 [00:47<00:05, 7379.22it/s] 89%|████████▉ | 357810/400000 [00:47<00:05, 7345.22it/s] 90%|████████▉ | 358545/400000 [00:47<00:05, 7256.32it/s] 90%|████████▉ | 359299/400000 [00:47<00:05, 7338.86it/s] 90%|█████████ | 360052/400000 [00:48<00:05, 7393.33it/s] 90%|█████████ | 360792/400000 [00:48<00:05, 7386.53it/s] 90%|█████████ | 361545/400000 [00:48<00:05, 7426.14it/s] 91%|█████████ | 362288/400000 [00:48<00:05, 7367.43it/s] 91%|█████████ | 363046/400000 [00:48<00:04, 7429.72it/s] 91%|█████████ | 363790/400000 [00:48<00:04, 7341.90it/s] 91%|█████████ | 364546/400000 [00:48<00:04, 7405.19it/s] 91%|█████████▏| 365294/400000 [00:48<00:04, 7426.68it/s] 92%|█████████▏| 366038/400000 [00:48<00:04, 7390.54it/s] 92%|█████████▏| 366778/400000 [00:48<00:04, 7272.22it/s] 92%|█████████▏| 367536/400000 [00:49<00:04, 7361.24it/s] 92%|█████████▏| 368288/400000 [00:49<00:04, 7405.46it/s] 92%|█████████▏| 369048/400000 [00:49<00:04, 7460.24it/s] 92%|█████████▏| 369796/400000 [00:49<00:04, 7464.92it/s] 93%|█████████▎| 370550/400000 [00:49<00:03, 7486.16it/s] 93%|█████████▎| 371301/400000 [00:49<00:03, 7490.87it/s] 93%|█████████▎| 372051/400000 [00:49<00:03, 7488.66it/s] 93%|█████████▎| 372809/400000 [00:49<00:03, 7513.27it/s] 93%|█████████▎| 373561/400000 [00:49<00:03, 7488.67it/s] 94%|█████████▎| 374319/400000 [00:49<00:03, 7514.62it/s] 94%|█████████▍| 375073/400000 [00:50<00:03, 7520.34it/s] 94%|█████████▍| 375826/400000 [00:50<00:03, 7509.90it/s] 94%|█████████▍| 376578/400000 [00:50<00:03, 7291.88it/s] 94%|█████████▍| 377309/400000 [00:50<00:03, 7280.45it/s] 95%|█████████▍| 378039/400000 [00:50<00:03, 6838.89it/s] 95%|█████████▍| 378765/400000 [00:50<00:03, 6958.37it/s] 95%|█████████▍| 379517/400000 [00:50<00:02, 7117.62it/s] 95%|█████████▌| 380256/400000 [00:50<00:02, 7196.20it/s] 95%|█████████▌| 380996/400000 [00:50<00:02, 7253.57it/s] 95%|█████████▌| 381752/400000 [00:50<00:02, 7341.57it/s] 96%|█████████▌| 382512/400000 [00:51<00:02, 7414.73it/s] 96%|█████████▌| 383255/400000 [00:51<00:02, 7416.60it/s] 96%|█████████▌| 384004/400000 [00:51<00:02, 7436.39it/s] 96%|█████████▌| 384750/400000 [00:51<00:02, 7442.98it/s] 96%|█████████▋| 385495/400000 [00:51<00:01, 7388.25it/s] 97%|█████████▋| 386235/400000 [00:51<00:01, 7386.92it/s] 97%|█████████▋| 386975/400000 [00:51<00:01, 7338.08it/s] 97%|█████████▋| 387729/400000 [00:51<00:01, 7395.91it/s] 97%|█████████▋| 388473/400000 [00:51<00:01, 7406.83it/s] 97%|█████████▋| 389227/400000 [00:51<00:01, 7443.42it/s] 97%|█████████▋| 389980/400000 [00:52<00:01, 7466.62it/s] 98%|█████████▊| 390727/400000 [00:52<00:01, 7465.69it/s] 98%|█████████▊| 391474/400000 [00:52<00:01, 7437.97it/s] 98%|█████████▊| 392218/400000 [00:52<00:01, 7409.14it/s] 98%|█████████▊| 392978/400000 [00:52<00:00, 7463.22it/s] 98%|█████████▊| 393730/400000 [00:52<00:00, 7478.99it/s] 99%|█████████▊| 394479/400000 [00:52<00:00, 7477.52it/s] 99%|█████████▉| 395237/400000 [00:52<00:00, 7506.26it/s] 99%|█████████▉| 395988/400000 [00:52<00:00, 7485.65it/s] 99%|█████████▉| 396742/400000 [00:52<00:00, 7500.79it/s] 99%|█████████▉| 397498/400000 [00:53<00:00, 7516.50it/s]100%|█████████▉| 398253/400000 [00:53<00:00, 7524.07it/s]100%|█████████▉| 399007/400000 [00:53<00:00, 7526.70it/s]100%|█████████▉| 399760/400000 [00:53<00:00, 7505.31it/s]100%|█████████▉| 399999/400000 [00:53<00:00, 7490.55it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fece98ebac8> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010979431978452189 	 Accuracy: 56
Train Epoch: 1 	 Loss: 0.01100226090504573 	 Accuracy: 60

  model saves at 60% accuracy 

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
2020-05-13 08:25:01.742532: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 08:25:01.747211: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 08:25:01.747376: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56162423e3e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 08:25:01.747394: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fec8f42dcc0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.3600 - accuracy: 0.5200
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6436 - accuracy: 0.5015
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5797 - accuracy: 0.5057 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5363 - accuracy: 0.5085
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5624 - accuracy: 0.5068
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5516 - accuracy: 0.5075
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5549 - accuracy: 0.5073
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5861 - accuracy: 0.5052
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5542 - accuracy: 0.5073
10000/25000 [===========>..................] - ETA: 5s - loss: 7.5808 - accuracy: 0.5056
11000/25000 [============>.................] - ETA: 4s - loss: 7.5886 - accuracy: 0.5051
12000/25000 [=============>................] - ETA: 4s - loss: 7.5900 - accuracy: 0.5050
13000/25000 [==============>...............] - ETA: 4s - loss: 7.5888 - accuracy: 0.5051
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6020 - accuracy: 0.5042
15000/25000 [=================>............] - ETA: 3s - loss: 7.6268 - accuracy: 0.5026
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6398 - accuracy: 0.5017
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6486 - accuracy: 0.5012
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6424 - accuracy: 0.5016
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6482 - accuracy: 0.5012
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6535 - accuracy: 0.5009
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6631 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6540 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6775 - accuracy: 0.4993
25000/25000 [==============================] - 10s 398us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fec4e4eb358> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fec4a76e710> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4540 - crf_viterbi_accuracy: 0.0133 - val_loss: 1.3611 - val_crf_viterbi_accuracy: 0.6667

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

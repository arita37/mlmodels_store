
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f2b8bc93fd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 09:15:39.619164
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 09:15:39.622967
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 09:15:39.626471
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 09:15:39.629828
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f2b97cab4a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 352964.0625
Epoch 2/10

1/1 [==============================] - 0s 110ms/step - loss: 235372.1094
Epoch 3/10

1/1 [==============================] - 0s 110ms/step - loss: 132917.5469
Epoch 4/10

1/1 [==============================] - 0s 111ms/step - loss: 67811.5781
Epoch 5/10

1/1 [==============================] - 0s 117ms/step - loss: 35893.7852
Epoch 6/10

1/1 [==============================] - 0s 101ms/step - loss: 21056.4590
Epoch 7/10

1/1 [==============================] - 0s 104ms/step - loss: 13683.7266
Epoch 8/10

1/1 [==============================] - 0s 107ms/step - loss: 9658.6914
Epoch 9/10

1/1 [==============================] - 0s 102ms/step - loss: 7250.5654
Epoch 10/10

1/1 [==============================] - 0s 101ms/step - loss: 5736.6836

  #### Inference Need return ypred, ytrue ######################### 
[[-2.07447708e-01 -1.02944720e+00  9.06939983e-01  2.03771448e+00
   7.95312703e-01 -1.01686025e+00 -5.50434351e-01  1.79174826e-01
  -9.60162282e-01  1.72529370e-01  1.32608557e+00  8.80861521e-01
   1.62021017e+00 -2.49975473e-01  7.15011477e-01  2.20862675e+00
  -4.19795185e-01  7.81161606e-01 -9.63496447e-01  1.13948226e+00
   1.63761520e+00  1.24370265e+00  7.42455184e-01 -1.83920413e-01
   4.60114539e-01 -2.62980461e-02  8.63112807e-02 -3.57903659e-01
   1.16843700e-01  5.93590260e-01  1.16755891e+00  8.13133538e-01
   1.35800898e-01  1.15957141e-01  7.47466505e-01 -1.58043528e+00
   1.43660498e+00 -1.95772439e-01 -5.99419117e-01 -2.18549824e+00
  -1.59832454e+00  1.66773105e+00  1.31850839e+00  1.30164099e+00
   5.20844579e-01  1.07835233e+00  2.38912389e-01 -2.64922118e+00
   1.23159647e-01  1.14730299e-01  1.29038429e+00  7.46447444e-01
  -1.38116693e+00  1.59790659e+00 -1.00133789e+00  1.17314506e+00
  -1.06658411e+00 -7.90005028e-02 -6.07204139e-01  2.90050745e-01
  -2.27056098e+00 -5.83255172e-01 -7.15316832e-01  1.70092416e+00
   1.08818150e+00 -1.45753002e+00  9.89175558e-01  2.16529787e-01
   2.24717498e+00  7.59534240e-01 -8.05143714e-02  4.88706231e-02
   8.47519398e-01 -6.72433317e-01 -5.08997679e-01 -2.01140374e-01
  -3.18016052e-01  1.21258390e+00 -1.87233120e-01  4.30876732e-01
   1.22902012e+00  1.45766211e+00  7.34977961e-01  1.26473397e-01
  -1.00855386e+00  1.02084756e-01 -1.05954134e+00 -8.62508774e-01
   4.54766482e-01  8.15908551e-01 -8.93738210e-01 -1.52876258e-01
   1.14610720e+00  1.07890451e+00 -1.28458068e-01  5.86810708e-01
   1.73770118e+00 -2.81459600e-01  7.23088622e-01  2.82747805e-01
  -1.76693857e-01 -2.42514014e-02  4.00533557e-01  1.00487363e+00
   1.00322366e-02 -1.33234143e+00  1.06038320e+00 -3.74958634e-01
   9.16888416e-01 -1.91889834e+00  3.88644040e-01  9.41023529e-02
  -1.79903686e-01 -4.12228703e-01  1.11986351e+00 -3.26541126e-01
  -2.89604157e-01  2.23656559e+00 -2.17443883e-01  8.44499469e-03
  -3.39961529e-01  7.99204254e+00  8.20106602e+00  8.71251488e+00
   7.75189877e+00  9.34284782e+00  6.42652798e+00  6.91121244e+00
   8.84540749e+00  8.45645237e+00  8.20298862e+00  8.76721573e+00
   8.82393074e+00  8.20528889e+00  9.66495800e+00  7.38575602e+00
   9.03708839e+00  8.19956493e+00  8.52154922e+00  7.58546400e+00
   9.21030235e+00  8.36818027e+00  8.50359535e+00  1.01693735e+01
   8.32215214e+00  7.71410799e+00  8.45393085e+00  1.08752918e+01
   9.47332478e+00  7.67943430e+00  8.46044350e+00  8.22907066e+00
   9.15730286e+00  9.18863869e+00  7.13619137e+00  9.74837971e+00
   9.34894943e+00  6.64976406e+00  7.89920235e+00  8.37140942e+00
   7.69468880e+00  7.38459778e+00  9.58363628e+00  1.03916607e+01
   9.26339245e+00  8.52653980e+00  6.87265587e+00  9.34654808e+00
   9.47019672e+00  9.55771446e+00  1.03983068e+01  8.77644444e+00
   7.77099991e+00  9.90176773e+00  8.99059296e+00  7.92537594e+00
   8.68819523e+00  1.06405058e+01  8.60171890e+00  7.71831226e+00
   5.17618299e-01  3.68677616e+00  1.47215652e+00  3.66769552e-01
   1.43063581e+00  6.43508792e-01  4.24853027e-01  2.28585577e+00
   2.08056092e+00  1.46205521e+00  2.08714545e-01  6.55837178e-01
   2.61218977e+00  8.86339068e-01  2.03371716e+00  6.78674221e-01
   5.48706472e-01  9.38856244e-01  2.94007063e+00  3.77416492e-01
   3.24581623e+00  1.02281809e-01  3.52050543e-01  1.38279724e+00
   9.33192313e-01  1.62378049e+00  1.77551568e-01  6.14253879e-01
   8.44117403e-01  6.51651502e-01  7.16134310e-01  2.30014086e+00
   3.35400724e+00  1.18775260e+00  1.17279816e+00  9.95002747e-01
   4.76424932e-01  7.63262987e-01  5.34709334e-01  1.94990838e+00
   2.23791647e+00  2.13136816e+00  8.52208138e-01  2.13586748e-01
   1.09471035e+00  1.94084704e-01  1.45233130e+00  9.19063628e-01
   2.61984205e+00  9.79455233e-01  3.51805329e-01  2.32175827e+00
   1.07096958e+00  3.78776073e-01  2.35812366e-01  3.85625958e-01
   2.19827032e+00  1.73581815e+00  7.95911789e-01  8.97737980e-01
   8.48591685e-01  8.15433800e-01  1.30111456e+00  1.06873035e+00
   2.09654665e+00  3.06854486e-01  6.56716287e-01  8.91728342e-01
   4.90958810e-01  1.26693904e+00  2.02535093e-01  6.56516790e-01
   1.99908185e+00  2.74392223e+00  2.40060568e-01  2.57300735e-01
   6.28807902e-01  4.07263458e-01  7.74333000e-01  6.91940129e-01
   1.20694005e+00  4.74264622e-01  2.21643865e-01  3.20003092e-01
   2.25897491e-01  2.66018331e-01  1.09530818e+00  5.36392748e-01
   2.06681466e+00  3.17477131e+00  8.90804768e-01  1.78583980e-01
   2.27483273e-01  2.26261473e+00  2.03985453e-01  3.30995262e-01
   1.49163651e+00  2.34276474e-01  8.67519736e-01  6.56866074e-01
   1.55711210e+00  8.99586082e-01  1.01635730e+00  1.40067732e+00
   1.27149963e+00  2.31468058e+00  1.86034346e+00  1.78993845e+00
   1.66214287e-01  6.26591623e-01  2.69790292e-01  1.07571995e+00
   8.31902027e-01  1.40562201e+00  6.71511412e-01  4.09914970e-01
   8.75491321e-01  1.60731602e+00  7.01851130e-01  1.83802700e+00
   1.70511067e-01  7.94440556e+00  9.68430328e+00  1.01005402e+01
   8.55898285e+00  8.85980511e+00  9.60110950e+00  9.15346432e+00
   8.03844357e+00  8.25117588e+00  1.08822842e+01  1.06761131e+01
   8.85081863e+00  9.06705189e+00  1.00529289e+01  8.69182205e+00
   8.52755165e+00  8.88057232e+00  8.81758785e+00  9.22402477e+00
   8.97844505e+00  8.94612598e+00  8.74325848e+00  9.56321526e+00
   7.91096306e+00  9.95127296e+00  8.06316853e+00  7.38624287e+00
   8.59083462e+00  7.86433554e+00  9.29012775e+00  9.61677456e+00
   9.88127041e+00  8.83937550e+00  7.38142443e+00  9.75228500e+00
   1.01653538e+01  1.09399815e+01  8.23280907e+00  1.04408045e+01
   8.53787422e+00  7.64349890e+00  1.07024946e+01  9.63877869e+00
   8.10437965e+00  8.20962715e+00  1.00271215e+01  9.98804951e+00
   8.53437424e+00  8.77420521e+00  1.01410618e+01  9.21520233e+00
   8.50870323e+00  9.02943134e+00  6.57744789e+00  7.12907028e+00
   8.88339806e+00  9.69801712e+00  8.25507450e+00  9.23681545e+00
  -8.93467903e+00 -6.22801161e+00  1.28636770e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 09:15:48.548163
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.6196
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 09:15:48.552346
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8605.11
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 09:15:48.555786
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.0626
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 09:15:48.559183
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -769.648
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139824648779368
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139823438926568
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139823438927072
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139823438927576
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139823438928080
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139823438928584

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f2b85618588> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.513038
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.478736
grad_step = 000002, loss = 0.454615
grad_step = 000003, loss = 0.432605
grad_step = 000004, loss = 0.412013
grad_step = 000005, loss = 0.396491
grad_step = 000006, loss = 0.385623
grad_step = 000007, loss = 0.374722
grad_step = 000008, loss = 0.364278
grad_step = 000009, loss = 0.353998
grad_step = 000010, loss = 0.342733
grad_step = 000011, loss = 0.331702
grad_step = 000012, loss = 0.321462
grad_step = 000013, loss = 0.311835
grad_step = 000014, loss = 0.302448
grad_step = 000015, loss = 0.292895
grad_step = 000016, loss = 0.282988
grad_step = 000017, loss = 0.272886
grad_step = 000018, loss = 0.263315
grad_step = 000019, loss = 0.254573
grad_step = 000020, loss = 0.245920
grad_step = 000021, loss = 0.236966
grad_step = 000022, loss = 0.228292
grad_step = 000023, loss = 0.220217
grad_step = 000024, loss = 0.212157
grad_step = 000025, loss = 0.201459
grad_step = 000026, loss = 0.190510
grad_step = 000027, loss = 0.180180
grad_step = 000028, loss = 0.170582
grad_step = 000029, loss = 0.161780
grad_step = 000030, loss = 0.153790
grad_step = 000031, loss = 0.146798
grad_step = 000032, loss = 0.140650
grad_step = 000033, loss = 0.134325
grad_step = 000034, loss = 0.127677
grad_step = 000035, loss = 0.121203
grad_step = 000036, loss = 0.114957
grad_step = 000037, loss = 0.108693
grad_step = 000038, loss = 0.102745
grad_step = 000039, loss = 0.097365
grad_step = 000040, loss = 0.092172
grad_step = 000041, loss = 0.087028
grad_step = 000042, loss = 0.082224
grad_step = 000043, loss = 0.077692
grad_step = 000044, loss = 0.073175
grad_step = 000045, loss = 0.068789
grad_step = 000046, loss = 0.064638
grad_step = 000047, loss = 0.060598
grad_step = 000048, loss = 0.056771
grad_step = 000049, loss = 0.053231
grad_step = 000050, loss = 0.049831
grad_step = 000051, loss = 0.046630
grad_step = 000052, loss = 0.043653
grad_step = 000053, loss = 0.040748
grad_step = 000054, loss = 0.037983
grad_step = 000055, loss = 0.035370
grad_step = 000056, loss = 0.032849
grad_step = 000057, loss = 0.030541
grad_step = 000058, loss = 0.028440
grad_step = 000059, loss = 0.026469
grad_step = 000060, loss = 0.024673
grad_step = 000061, loss = 0.022981
grad_step = 000062, loss = 0.021400
grad_step = 000063, loss = 0.019924
grad_step = 000064, loss = 0.018513
grad_step = 000065, loss = 0.017223
grad_step = 000066, loss = 0.016019
grad_step = 000067, loss = 0.014905
grad_step = 000068, loss = 0.013888
grad_step = 000069, loss = 0.012912
grad_step = 000070, loss = 0.012012
grad_step = 000071, loss = 0.011169
grad_step = 000072, loss = 0.010385
grad_step = 000073, loss = 0.009662
grad_step = 000074, loss = 0.008989
grad_step = 000075, loss = 0.008377
grad_step = 000076, loss = 0.007800
grad_step = 000077, loss = 0.007271
grad_step = 000078, loss = 0.006775
grad_step = 000079, loss = 0.006316
grad_step = 000080, loss = 0.005900
grad_step = 000081, loss = 0.005517
grad_step = 000082, loss = 0.005172
grad_step = 000083, loss = 0.004857
grad_step = 000084, loss = 0.004573
grad_step = 000085, loss = 0.004310
grad_step = 000086, loss = 0.004073
grad_step = 000087, loss = 0.003858
grad_step = 000088, loss = 0.003669
grad_step = 000089, loss = 0.003502
grad_step = 000090, loss = 0.003360
grad_step = 000091, loss = 0.003242
grad_step = 000092, loss = 0.003141
grad_step = 000093, loss = 0.003023
grad_step = 000094, loss = 0.002893
grad_step = 000095, loss = 0.002774
grad_step = 000096, loss = 0.002694
grad_step = 000097, loss = 0.002645
grad_step = 000098, loss = 0.002599
grad_step = 000099, loss = 0.002540
grad_step = 000100, loss = 0.002466
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002400
grad_step = 000102, loss = 0.002358
grad_step = 000103, loss = 0.002335
grad_step = 000104, loss = 0.002320
grad_step = 000105, loss = 0.002299
grad_step = 000106, loss = 0.002270
grad_step = 000107, loss = 0.002233
grad_step = 000108, loss = 0.002200
grad_step = 000109, loss = 0.002173
grad_step = 000110, loss = 0.002157
grad_step = 000111, loss = 0.002149
grad_step = 000112, loss = 0.002146
grad_step = 000113, loss = 0.002150
grad_step = 000114, loss = 0.002163
grad_step = 000115, loss = 0.002190
grad_step = 000116, loss = 0.002210
grad_step = 000117, loss = 0.002217
grad_step = 000118, loss = 0.002168
grad_step = 000119, loss = 0.002102
grad_step = 000120, loss = 0.002057
grad_step = 000121, loss = 0.002059
grad_step = 000122, loss = 0.002092
grad_step = 000123, loss = 0.002119
grad_step = 000124, loss = 0.002121
grad_step = 000125, loss = 0.002084
grad_step = 000126, loss = 0.002041
grad_step = 000127, loss = 0.002017
grad_step = 000128, loss = 0.002020
grad_step = 000129, loss = 0.002041
grad_step = 000130, loss = 0.002061
grad_step = 000131, loss = 0.002069
grad_step = 000132, loss = 0.002055
grad_step = 000133, loss = 0.002030
grad_step = 000134, loss = 0.002002
grad_step = 000135, loss = 0.001984
grad_step = 000136, loss = 0.001978
grad_step = 000137, loss = 0.001983
grad_step = 000138, loss = 0.001993
grad_step = 000139, loss = 0.002007
grad_step = 000140, loss = 0.002024
grad_step = 000141, loss = 0.002038
grad_step = 000142, loss = 0.002049
grad_step = 000143, loss = 0.002042
grad_step = 000144, loss = 0.002020
grad_step = 000145, loss = 0.001983
grad_step = 000146, loss = 0.001952
grad_step = 000147, loss = 0.001936
grad_step = 000148, loss = 0.001937
grad_step = 000149, loss = 0.001949
grad_step = 000150, loss = 0.001967
grad_step = 000151, loss = 0.001988
grad_step = 000152, loss = 0.002003
grad_step = 000153, loss = 0.002014
grad_step = 000154, loss = 0.002006
grad_step = 000155, loss = 0.001987
grad_step = 000156, loss = 0.001952
grad_step = 000157, loss = 0.001920
grad_step = 000158, loss = 0.001900
grad_step = 000159, loss = 0.001897
grad_step = 000160, loss = 0.001907
grad_step = 000161, loss = 0.001926
grad_step = 000162, loss = 0.001950
grad_step = 000163, loss = 0.001978
grad_step = 000164, loss = 0.002010
grad_step = 000165, loss = 0.002027
grad_step = 000166, loss = 0.002021
grad_step = 000167, loss = 0.001980
grad_step = 000168, loss = 0.001922
grad_step = 000169, loss = 0.001877
grad_step = 000170, loss = 0.001867
grad_step = 000171, loss = 0.001887
grad_step = 000172, loss = 0.001919
grad_step = 000173, loss = 0.001945
grad_step = 000174, loss = 0.001952
grad_step = 000175, loss = 0.001936
grad_step = 000176, loss = 0.001904
grad_step = 000177, loss = 0.001870
grad_step = 000178, loss = 0.001850
grad_step = 000179, loss = 0.001847
grad_step = 000180, loss = 0.001857
grad_step = 000181, loss = 0.001873
grad_step = 000182, loss = 0.001889
grad_step = 000183, loss = 0.001901
grad_step = 000184, loss = 0.001904
grad_step = 000185, loss = 0.001899
grad_step = 000186, loss = 0.001886
grad_step = 000187, loss = 0.001867
grad_step = 000188, loss = 0.001848
grad_step = 000189, loss = 0.001833
grad_step = 000190, loss = 0.001825
grad_step = 000191, loss = 0.001823
grad_step = 000192, loss = 0.001826
grad_step = 000193, loss = 0.001833
grad_step = 000194, loss = 0.001846
grad_step = 000195, loss = 0.001867
grad_step = 000196, loss = 0.001904
grad_step = 000197, loss = 0.001959
grad_step = 000198, loss = 0.002030
grad_step = 000199, loss = 0.002089
grad_step = 000200, loss = 0.002074
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001977
grad_step = 000202, loss = 0.001848
grad_step = 000203, loss = 0.001810
grad_step = 000204, loss = 0.001873
grad_step = 000205, loss = 0.001935
grad_step = 000206, loss = 0.001925
grad_step = 000207, loss = 0.001849
grad_step = 000208, loss = 0.001803
grad_step = 000209, loss = 0.001821
grad_step = 000210, loss = 0.001866
grad_step = 000211, loss = 0.001887
grad_step = 000212, loss = 0.001863
grad_step = 000213, loss = 0.001812
grad_step = 000214, loss = 0.001791
grad_step = 000215, loss = 0.001810
grad_step = 000216, loss = 0.001836
grad_step = 000217, loss = 0.001840
grad_step = 000218, loss = 0.001824
grad_step = 000219, loss = 0.001801
grad_step = 000220, loss = 0.001786
grad_step = 000221, loss = 0.001786
grad_step = 000222, loss = 0.001799
grad_step = 000223, loss = 0.001811
grad_step = 000224, loss = 0.001808
grad_step = 000225, loss = 0.001798
grad_step = 000226, loss = 0.001787
grad_step = 000227, loss = 0.001778
grad_step = 000228, loss = 0.001774
grad_step = 000229, loss = 0.001775
grad_step = 000230, loss = 0.001780
grad_step = 000231, loss = 0.001786
grad_step = 000232, loss = 0.001789
grad_step = 000233, loss = 0.001789
grad_step = 000234, loss = 0.001789
grad_step = 000235, loss = 0.001788
grad_step = 000236, loss = 0.001785
grad_step = 000237, loss = 0.001780
grad_step = 000238, loss = 0.001776
grad_step = 000239, loss = 0.001773
grad_step = 000240, loss = 0.001770
grad_step = 000241, loss = 0.001767
grad_step = 000242, loss = 0.001765
grad_step = 000243, loss = 0.001764
grad_step = 000244, loss = 0.001766
grad_step = 000245, loss = 0.001769
grad_step = 000246, loss = 0.001775
grad_step = 000247, loss = 0.001787
grad_step = 000248, loss = 0.001809
grad_step = 000249, loss = 0.001847
grad_step = 000250, loss = 0.001907
grad_step = 000251, loss = 0.001979
grad_step = 000252, loss = 0.002050
grad_step = 000253, loss = 0.002051
grad_step = 000254, loss = 0.001972
grad_step = 000255, loss = 0.001828
grad_step = 000256, loss = 0.001747
grad_step = 000257, loss = 0.001779
grad_step = 000258, loss = 0.001860
grad_step = 000259, loss = 0.001895
grad_step = 000260, loss = 0.001839
grad_step = 000261, loss = 0.001761
grad_step = 000262, loss = 0.001740
grad_step = 000263, loss = 0.001780
grad_step = 000264, loss = 0.001825
grad_step = 000265, loss = 0.001819
grad_step = 000266, loss = 0.001775
grad_step = 000267, loss = 0.001737
grad_step = 000268, loss = 0.001735
grad_step = 000269, loss = 0.001761
grad_step = 000270, loss = 0.001782
grad_step = 000271, loss = 0.001778
grad_step = 000272, loss = 0.001753
grad_step = 000273, loss = 0.001730
grad_step = 000274, loss = 0.001725
grad_step = 000275, loss = 0.001737
grad_step = 000276, loss = 0.001750
grad_step = 000277, loss = 0.001753
grad_step = 000278, loss = 0.001744
grad_step = 000279, loss = 0.001730
grad_step = 000280, loss = 0.001720
grad_step = 000281, loss = 0.001717
grad_step = 000282, loss = 0.001722
grad_step = 000283, loss = 0.001729
grad_step = 000284, loss = 0.001732
grad_step = 000285, loss = 0.001731
grad_step = 000286, loss = 0.001726
grad_step = 000287, loss = 0.001719
grad_step = 000288, loss = 0.001713
grad_step = 000289, loss = 0.001709
grad_step = 000290, loss = 0.001708
grad_step = 000291, loss = 0.001709
grad_step = 000292, loss = 0.001711
grad_step = 000293, loss = 0.001712
grad_step = 000294, loss = 0.001714
grad_step = 000295, loss = 0.001714
grad_step = 000296, loss = 0.001714
grad_step = 000297, loss = 0.001714
grad_step = 000298, loss = 0.001713
grad_step = 000299, loss = 0.001712
grad_step = 000300, loss = 0.001711
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001711
grad_step = 000302, loss = 0.001712
grad_step = 000303, loss = 0.001713
grad_step = 000304, loss = 0.001715
grad_step = 000305, loss = 0.001719
grad_step = 000306, loss = 0.001726
grad_step = 000307, loss = 0.001736
grad_step = 000308, loss = 0.001752
grad_step = 000309, loss = 0.001771
grad_step = 000310, loss = 0.001797
grad_step = 000311, loss = 0.001820
grad_step = 000312, loss = 0.001839
grad_step = 000313, loss = 0.001837
grad_step = 000314, loss = 0.001811
grad_step = 000315, loss = 0.001763
grad_step = 000316, loss = 0.001713
grad_step = 000317, loss = 0.001685
grad_step = 000318, loss = 0.001686
grad_step = 000319, loss = 0.001706
grad_step = 000320, loss = 0.001728
grad_step = 000321, loss = 0.001741
grad_step = 000322, loss = 0.001739
grad_step = 000323, loss = 0.001727
grad_step = 000324, loss = 0.001706
grad_step = 000325, loss = 0.001686
grad_step = 000326, loss = 0.001674
grad_step = 000327, loss = 0.001674
grad_step = 000328, loss = 0.001682
grad_step = 000329, loss = 0.001693
grad_step = 000330, loss = 0.001702
grad_step = 000331, loss = 0.001709
grad_step = 000332, loss = 0.001716
grad_step = 000333, loss = 0.001723
grad_step = 000334, loss = 0.001728
grad_step = 000335, loss = 0.001729
grad_step = 000336, loss = 0.001721
grad_step = 000337, loss = 0.001710
grad_step = 000338, loss = 0.001698
grad_step = 000339, loss = 0.001687
grad_step = 000340, loss = 0.001679
grad_step = 000341, loss = 0.001670
grad_step = 000342, loss = 0.001662
grad_step = 000343, loss = 0.001657
grad_step = 000344, loss = 0.001655
grad_step = 000345, loss = 0.001656
grad_step = 000346, loss = 0.001658
grad_step = 000347, loss = 0.001660
grad_step = 000348, loss = 0.001662
grad_step = 000349, loss = 0.001664
grad_step = 000350, loss = 0.001670
grad_step = 000351, loss = 0.001680
grad_step = 000352, loss = 0.001699
grad_step = 000353, loss = 0.001727
grad_step = 000354, loss = 0.001769
grad_step = 000355, loss = 0.001822
grad_step = 000356, loss = 0.001879
grad_step = 000357, loss = 0.001939
grad_step = 000358, loss = 0.001931
grad_step = 000359, loss = 0.001870
grad_step = 000360, loss = 0.001744
grad_step = 000361, loss = 0.001655
grad_step = 000362, loss = 0.001652
grad_step = 000363, loss = 0.001708
grad_step = 000364, loss = 0.001759
grad_step = 000365, loss = 0.001757
grad_step = 000366, loss = 0.001720
grad_step = 000367, loss = 0.001679
grad_step = 000368, loss = 0.001657
grad_step = 000369, loss = 0.001658
grad_step = 000370, loss = 0.001668
grad_step = 000371, loss = 0.001679
grad_step = 000372, loss = 0.001680
grad_step = 000373, loss = 0.001662
grad_step = 000374, loss = 0.001640
grad_step = 000375, loss = 0.001631
grad_step = 000376, loss = 0.001643
grad_step = 000377, loss = 0.001660
grad_step = 000378, loss = 0.001659
grad_step = 000379, loss = 0.001645
grad_step = 000380, loss = 0.001628
grad_step = 000381, loss = 0.001624
grad_step = 000382, loss = 0.001628
grad_step = 000383, loss = 0.001631
grad_step = 000384, loss = 0.001631
grad_step = 000385, loss = 0.001629
grad_step = 000386, loss = 0.001630
grad_step = 000387, loss = 0.001633
grad_step = 000388, loss = 0.001634
grad_step = 000389, loss = 0.001630
grad_step = 000390, loss = 0.001624
grad_step = 000391, loss = 0.001618
grad_step = 000392, loss = 0.001615
grad_step = 000393, loss = 0.001614
grad_step = 000394, loss = 0.001613
grad_step = 000395, loss = 0.001611
grad_step = 000396, loss = 0.001607
grad_step = 000397, loss = 0.001605
grad_step = 000398, loss = 0.001604
grad_step = 000399, loss = 0.001604
grad_step = 000400, loss = 0.001603
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001602
grad_step = 000402, loss = 0.001601
grad_step = 000403, loss = 0.001599
grad_step = 000404, loss = 0.001598
grad_step = 000405, loss = 0.001597
grad_step = 000406, loss = 0.001596
grad_step = 000407, loss = 0.001596
grad_step = 000408, loss = 0.001595
grad_step = 000409, loss = 0.001595
grad_step = 000410, loss = 0.001595
grad_step = 000411, loss = 0.001596
grad_step = 000412, loss = 0.001601
grad_step = 000413, loss = 0.001613
grad_step = 000414, loss = 0.001641
grad_step = 000415, loss = 0.001705
grad_step = 000416, loss = 0.001847
grad_step = 000417, loss = 0.002105
grad_step = 000418, loss = 0.002544
grad_step = 000419, loss = 0.002764
grad_step = 000420, loss = 0.002594
grad_step = 000421, loss = 0.001876
grad_step = 000422, loss = 0.001617
grad_step = 000423, loss = 0.002000
grad_step = 000424, loss = 0.002160
grad_step = 000425, loss = 0.001771
grad_step = 000426, loss = 0.001636
grad_step = 000427, loss = 0.001924
grad_step = 000428, loss = 0.001955
grad_step = 000429, loss = 0.001654
grad_step = 000430, loss = 0.001687
grad_step = 000431, loss = 0.001876
grad_step = 000432, loss = 0.001714
grad_step = 000433, loss = 0.001613
grad_step = 000434, loss = 0.001744
grad_step = 000435, loss = 0.001735
grad_step = 000436, loss = 0.001602
grad_step = 000437, loss = 0.001648
grad_step = 000438, loss = 0.001714
grad_step = 000439, loss = 0.001615
grad_step = 000440, loss = 0.001601
grad_step = 000441, loss = 0.001662
grad_step = 000442, loss = 0.001652
grad_step = 000443, loss = 0.001589
grad_step = 000444, loss = 0.001599
grad_step = 000445, loss = 0.001646
grad_step = 000446, loss = 0.001602
grad_step = 000447, loss = 0.001577
grad_step = 000448, loss = 0.001600
grad_step = 000449, loss = 0.001606
grad_step = 000450, loss = 0.001579
grad_step = 000451, loss = 0.001569
grad_step = 000452, loss = 0.001589
grad_step = 000453, loss = 0.001582
grad_step = 000454, loss = 0.001564
grad_step = 000455, loss = 0.001567
grad_step = 000456, loss = 0.001574
grad_step = 000457, loss = 0.001568
grad_step = 000458, loss = 0.001556
grad_step = 000459, loss = 0.001559
grad_step = 000460, loss = 0.001565
grad_step = 000461, loss = 0.001557
grad_step = 000462, loss = 0.001551
grad_step = 000463, loss = 0.001553
grad_step = 000464, loss = 0.001555
grad_step = 000465, loss = 0.001552
grad_step = 000466, loss = 0.001545
grad_step = 000467, loss = 0.001547
grad_step = 000468, loss = 0.001548
grad_step = 000469, loss = 0.001545
grad_step = 000470, loss = 0.001541
grad_step = 000471, loss = 0.001540
grad_step = 000472, loss = 0.001542
grad_step = 000473, loss = 0.001541
grad_step = 000474, loss = 0.001537
grad_step = 000475, loss = 0.001535
grad_step = 000476, loss = 0.001535
grad_step = 000477, loss = 0.001535
grad_step = 000478, loss = 0.001533
grad_step = 000479, loss = 0.001531
grad_step = 000480, loss = 0.001530
grad_step = 000481, loss = 0.001530
grad_step = 000482, loss = 0.001529
grad_step = 000483, loss = 0.001527
grad_step = 000484, loss = 0.001525
grad_step = 000485, loss = 0.001524
grad_step = 000486, loss = 0.001524
grad_step = 000487, loss = 0.001523
grad_step = 000488, loss = 0.001522
grad_step = 000489, loss = 0.001520
grad_step = 000490, loss = 0.001519
grad_step = 000491, loss = 0.001518
grad_step = 000492, loss = 0.001517
grad_step = 000493, loss = 0.001516
grad_step = 000494, loss = 0.001515
grad_step = 000495, loss = 0.001514
grad_step = 000496, loss = 0.001513
grad_step = 000497, loss = 0.001512
grad_step = 000498, loss = 0.001511
grad_step = 000499, loss = 0.001510
grad_step = 000500, loss = 0.001508
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001507
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

  date_run                              2020-05-13 09:16:12.393334
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.252224
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 09:16:12.402316
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.167086
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 09:16:12.409681
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.141524
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 09:16:12.415129
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -1.53894
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
0   2020-05-13 09:15:39.619164  ...    mean_absolute_error
1   2020-05-13 09:15:39.622967  ...     mean_squared_error
2   2020-05-13 09:15:39.626471  ...  median_absolute_error
3   2020-05-13 09:15:39.629828  ...               r2_score
4   2020-05-13 09:15:48.548163  ...    mean_absolute_error
5   2020-05-13 09:15:48.552346  ...     mean_squared_error
6   2020-05-13 09:15:48.555786  ...  median_absolute_error
7   2020-05-13 09:15:48.559183  ...               r2_score
8   2020-05-13 09:16:12.393334  ...    mean_absolute_error
9   2020-05-13 09:16:12.402316  ...     mean_squared_error
10  2020-05-13 09:16:12.409681  ...  median_absolute_error
11  2020-05-13 09:16:12.415129  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f744d168cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  1%|          | 90112/9912422 [00:00<00:11, 890528.28it/s]  6%|▌         | 573440/9912422 [00:00<00:07, 1178322.83it/s] 56%|█████▌    | 5554176/9912422 [00:00<00:02, 1665615.38it/s]9920512it [00:00, 17013019.37it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:01, 31799.31it/s]            
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  5%|▍         | 81920/1648877 [00:00<00:01, 813662.30it/s] 34%|███▍      | 565248/1648877 [00:00<00:00, 1083685.18it/s]1654784it [00:00, 4315194.77it/s]                            
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 56721.14it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f73ffb21eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f73ff14f0f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f73ffb21eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f73ff0a8128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f73fc8e3518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f73fc8cdc88> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f73ffb21eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f73ff065748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f73fc8e3518> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f73fef20550> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fecb33f6208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=9f088fee4e58ded2ff26ed6a9cadaee3bca2c3ff7f0624448d1e34312ecb57a1
  Stored in directory: /tmp/pip-ephem-wheel-cache-vux17zbi/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fec4b1f1710> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2588672/17464789 [===>..........................] - ETA: 0s
10813440/17464789 [=================>............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 09:17:39.935103: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 09:17:39.939803: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 09:17:39.939951: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5651ce539120 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 09:17:39.939966: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8353 - accuracy: 0.4890
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6590 - accuracy: 0.5005 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6513 - accuracy: 0.5010
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5976 - accuracy: 0.5045
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6084 - accuracy: 0.5038
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6487 - accuracy: 0.5012
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7236 - accuracy: 0.4963
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7299 - accuracy: 0.4959
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7092 - accuracy: 0.4972
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6881 - accuracy: 0.4986
11000/25000 [============>.................] - ETA: 4s - loss: 7.6875 - accuracy: 0.4986
12000/25000 [=============>................] - ETA: 4s - loss: 7.7024 - accuracy: 0.4977
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7091 - accuracy: 0.4972
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7192 - accuracy: 0.4966
15000/25000 [=================>............] - ETA: 3s - loss: 7.7085 - accuracy: 0.4973
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7270 - accuracy: 0.4961
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7081 - accuracy: 0.4973
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6939 - accuracy: 0.4982
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6981 - accuracy: 0.4979
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7057 - accuracy: 0.4974
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6929 - accuracy: 0.4983
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6694 - accuracy: 0.4998
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6600 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6577 - accuracy: 0.5006
25000/25000 [==============================] - 10s 381us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 09:17:56.509210
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 09:17:56.509210  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:02<68:40:07, 3.49kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:02<48:18:35, 4.96kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:02<33:51:17, 7.07kB/s] .vector_cache/glove.6B.zip:   0%|          | 467k/862M [00:02<23:43:13, 10.1kB/s].vector_cache/glove.6B.zip:   0%|          | 1.56M/862M [00:02<16:35:23, 14.4kB/s].vector_cache/glove.6B.zip:   0%|          | 3.42M/862M [00:02<11:35:30, 20.6kB/s].vector_cache/glove.6B.zip:   1%|          | 7.46M/862M [00:02<8:04:39, 29.4kB/s] .vector_cache/glove.6B.zip:   1%|▏         | 11.4M/862M [00:03<5:37:49, 42.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.9M/862M [00:03<3:55:18, 59.9kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.0M/862M [00:03<2:44:01, 85.6kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.5M/862M [00:03<1:54:17, 122kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 28.1M/862M [00:03<1:19:46, 174kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.7M/862M [00:03<55:37, 249kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 36.3M/862M [00:03<38:53, 354kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.4M/862M [00:03<27:11, 504kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.9M/862M [00:03<19:01, 716kB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.4M/862M [00:03<13:22, 1.01MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:04<09:59, 1.35MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.3M/862M [00:04<07:06, 1.89MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:06<10:32, 1.27MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:06<09:50, 1.36MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:06<07:28, 1.80MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<05:21, 2.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:08<20:26, 654kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:08<15:49, 844kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:08<11:26, 1.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:10<10:49, 1.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:10<08:59, 1.48MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.8M/862M [00:10<06:37, 2.00MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:12<07:41, 1.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:12<07:44, 1.71MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.7M/862M [00:12<06:15, 2.11MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.3M/862M [00:12<04:37, 2.85MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:14<06:47, 1.94MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:14<06:37, 1.98MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.4M/862M [00:14<05:02, 2.60MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.6M/862M [00:14<03:51, 3.40MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:16<06:29, 2.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:16<06:25, 2.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.6M/862M [00:16<04:51, 2.68MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.9M/862M [00:16<03:42, 3.52MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:18<06:49, 1.91MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:18<06:47, 1.91MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.7M/862M [00:18<05:10, 2.51MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<03:45, 3.45MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:20<1:06:44, 194kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:20<48:12, 268kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:20<34:00, 380kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:22<26:27, 487kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:22<19:52, 647kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:22<14:13, 903kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:24<12:52, 994kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.2M/862M [00:24<10:20, 1.24MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:24<07:33, 1.69MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:26<08:13, 1.55MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:26<07:03, 1.80MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:26<05:15, 2.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<06:40, 1.90MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<05:57, 2.13MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:28<04:26, 2.85MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<06:06, 2.06MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:30<05:31, 2.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:30<04:11, 3.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<05:53, 2.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<06:41, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<05:14, 2.39MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<03:56, 3.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<06:09, 2.02MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<05:23, 2.31MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:34<04:21, 2.86MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<03:11, 3.89MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:36<14:33, 852kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<12:45, 972kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:36<09:32, 1.30MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:36<06:51, 1.80MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<09:48, 1.26MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<07:58, 1.55MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:38<06:05, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<04:23, 2.80MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<17:33, 698kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<13:32, 904kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:40<09:46, 1.25MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<09:42, 1.26MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<08:02, 1.52MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:42<05:55, 2.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<07:00, 1.73MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<06:08, 1.97MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:44<04:35, 2.63MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<06:10, 1.95MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<06:46, 1.78MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:46<05:21, 2.24MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:46<04:11, 2.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<05:32, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:48<05:07, 2.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:48<03:59, 3.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:48<02:59, 4.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<08:23, 1.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<09:18, 1.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<08:55, 1.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<07:26, 1.60MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:50<05:36, 2.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:51<04:11, 2.83MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:51<03:17, 3.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<17:05, 693kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:52<13:12, 897kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:52<09:34, 1.23MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<06:50, 1.72MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<19:20, 609kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<14:47, 795kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:54<10:34, 1.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<07:33, 1.55MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<22:58, 509kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<18:38, 628kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:56<13:33, 862kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:56<09:47, 1.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<09:32, 1.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<07:59, 1.46MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:58<06:22, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:58<05:10, 2.25MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:58<04:03, 2.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:59<03:14, 3.57MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:00<06:37, 1.74MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:00<05:55, 1.95MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:00<04:39, 2.48MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:00<03:35, 3.22MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<02:45, 4.18MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<11:51, 969kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<10:38, 1.08MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:02<07:58, 1.44MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:02<05:51, 1.95MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<06:56, 1.65MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:04<06:06, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:04<04:35, 2.49MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<05:43, 1.99MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<05:16, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:06<03:57, 2.86MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<02:55, 3.87MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<44:24, 254kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:08<33:22, 338kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:08<23:54, 472kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:08<16:48, 668kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<20:58, 535kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<15:51, 708kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:10<11:21, 985kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<10:31, 1.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<08:37, 1.29MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:12<06:21, 1.75MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<07:12, 1.54MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<10:31, 1.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<08:40, 1.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:14<06:24, 1.73MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<04:37, 2.38MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<11:19, 973kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<08:29, 1.30MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:16<06:04, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<09:20, 1.17MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<08:23, 1.30MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:18<06:18, 1.73MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:18<04:42, 2.32MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<05:57, 1.82MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<06:24, 1.70MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<05:02, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:20<03:39, 2.96MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<09:47, 1.10MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<08:01, 1.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:22<05:58, 1.81MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:22<04:20, 2.48MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<09:35, 1.12MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<09:06, 1.18MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<06:53, 1.56MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:24<05:04, 2.11MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<06:04, 1.76MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<05:12, 2.05MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<03:54, 2.73MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<05:14, 2.02MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<04:45, 2.23MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:28<03:35, 2.94MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<05:00, 2.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<05:38, 1.87MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:30<04:28, 2.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:30<03:14, 3.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<10:07:14, 17.2kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<7:05:53, 24.6kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:32<4:57:37, 35.1kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<3:30:03, 49.5kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<2:28:00, 70.2kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:34<1:43:36, 100kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:36<1:14:44, 138kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:36<53:19, 194kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:36<37:29, 275kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:38<28:36, 359kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<22:06, 464kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:38<15:54, 644kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:38<11:18, 904kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<10:55, 933kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:40<08:31, 1.20MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:40<06:17, 1.62MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:40<04:31, 2.24MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<49:31, 204kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:42<36:43, 276kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:42<26:11, 386kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:44<19:52, 506kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:44<14:56, 673kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:44<10:42, 937kB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:45<09:46, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:46<07:52, 1.27MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:46<05:45, 1.73MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<06:19, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:48<06:26, 1.54MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:48<04:56, 2.00MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:48<03:34, 2.76MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<07:53, 1.25MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:50<06:33, 1.50MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:50<04:47, 2.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<05:38, 1.73MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<05:57, 1.64MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:52<04:40, 2.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:52<03:20, 2.90MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<58:55, 165kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:53<42:13, 230kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:54<29:41, 326kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:55<22:58, 420kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:55<17:04, 565kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:56<12:09, 790kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<10:44, 891kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:57<09:30, 1.01MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:58<07:08, 1.34MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:58<05:09, 1.85MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<06:49, 1.39MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<06:06, 1.56MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:59<04:42, 2.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:00<03:30, 2.70MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<04:54, 1.92MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<04:48, 1.96MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:01<03:49, 2.46MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:02<02:50, 3.30MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<04:52, 1.92MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<04:19, 2.17MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:03<03:40, 2.55MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:03<02:44, 3.41MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<04:17, 2.17MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:05<05:04, 1.83MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:05<04:27, 2.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:06<03:22, 2.75MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:06<02:27, 3.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<28:00, 329kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<20:46, 444kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:07<14:58, 615kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:08<10:39, 862kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:10<09:17, 982kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:10<14:18, 638kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:10<11:48, 773kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:10<08:41, 1.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:11<06:09, 1.47MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:12<14:04, 644kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:12<11:07, 815kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:12<08:22, 1.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:12<06:06, 1.48MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:12<04:26, 2.03MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:14<08:06, 1.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:14<07:04, 1.27MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<05:17, 1.70MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:14<03:52, 2.31MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:16<05:36, 1.59MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:16<04:44, 1.88MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:16<03:48, 2.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:16<02:46, 3.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:18<06:01, 1.47MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:18<06:05, 1.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:18<04:44, 1.86MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:18<03:31, 2.50MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:20<04:28, 1.96MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:20<04:08, 2.12MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:20<03:09, 2.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:22<04:02, 2.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:22<03:44, 2.32MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<02:50, 3.06MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:24<03:59, 2.17MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:24<03:31, 2.45MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:24<02:52, 2.99MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:24<02:09, 3.99MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:26<04:47, 1.79MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:26<04:23, 1.95MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:26<03:17, 2.60MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:26<02:24, 3.53MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:28<13:14, 643kB/s] .vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:28<11:23, 746kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:28<08:28, 1.00MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<06:04, 1.39MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:30<06:39, 1.27MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:30<05:38, 1.50MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<04:09, 2.02MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:30<03:00, 2.78MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:32<14:18, 585kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:32<10:58, 762kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<07:54, 1.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:32<05:36, 1.48MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:34<1:14:27, 111kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:34<52:53, 157kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:34<37:07, 223kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:34<26:01, 317kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:36<24:33, 335kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:36<18:08, 453kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<12:51, 638kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<09:05, 899kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:38<18:01, 453kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:38<13:32, 602kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<09:40, 840kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:38<06:50, 1.18MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:40<8:04:43, 16.7kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:40<5:40:01, 23.8kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<3:57:33, 33.9kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<2:45:33, 48.5kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:42<9:49:26, 13.6kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:42<6:53:13, 19.4kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<4:48:38, 27.7kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:44<3:22:44, 39.2kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:44<2:23:39, 55.3kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:44<1:40:47, 78.8kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<1:10:35, 112kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:44<49:21, 160kB/s]  .vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:46<47:34, 166kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:46<34:27, 229kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:46<24:19, 323kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:46<17:03, 459kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:48<19:54, 393kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:48<15:03, 519kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<10:47, 722kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:48<07:40, 1.01MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:50<08:55, 868kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:50<07:20, 1.06MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:50<05:23, 1.43MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:50<03:53, 1.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:52<08:01, 956kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:52<07:52, 975kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:52<06:04, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:52<04:20, 1.76MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:54<05:41, 1.34MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:54<05:06, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:54<04:11, 1.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:54<03:10, 2.39MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:54<02:18, 3.26MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:56<08:37, 874kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:56<06:48, 1.11MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:56<05:32, 1.36MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<04:05, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:58<04:19, 1.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:58<04:02, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:58<03:19, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<02:32, 2.92MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:58<01:54, 3.87MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:00<05:28, 1.35MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [03:00<05:09, 1.43MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<04:33, 1.62MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<03:56, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:00<02:58, 2.48MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<03:52, 1.89MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:02<05:43, 1.28MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<04:41, 1.56MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:03<03:33, 2.06MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:03<02:33, 2.84MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:04<12:12, 595kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:04<09:17, 780kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<06:40, 1.08MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:05<04:53, 1.48MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:06<05:40, 1.27MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:06<06:26, 1.12MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:07<05:34, 1.29MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:07<04:15, 1.68MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:07<03:05, 2.31MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<04:29, 1.59MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<03:54, 1.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<03:03, 2.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:09<02:19, 3.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<01:44, 4.07MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<6:31:47, 18.0kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<4:35:41, 25.6kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<3:13:24, 36.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:11<2:15:40, 51.9kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:11<1:34:45, 74.0kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<1:06:13, 105kB/s] .vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<1:07:41, 103kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:12<48:24, 144kB/s]  .vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:13<34:01, 205kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:13<23:47, 291kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:14<21:42, 319kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:14<15:58, 432kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:14<11:34, 596kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<08:09, 841kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:16<08:19, 823kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:16<06:34, 1.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:16<04:46, 1.43MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:18<04:49, 1.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:18<04:46, 1.42MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:18<03:40, 1.84MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:19<02:37, 2.56MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:20<09:21, 717kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:20<07:16, 921kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:20<05:13, 1.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:22<05:09, 1.29MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:22<04:18, 1.54MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:22<03:10, 2.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:24<03:43, 1.76MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:24<03:58, 1.65MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:24<03:05, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:24<02:12, 2.94MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:26<25:22, 256kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:26<18:25, 353kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:26<13:00, 497kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:28<10:33, 609kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<07:57, 808kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:28<05:54, 1.09MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:28<04:13, 1.51MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:30<05:15, 1.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:30<04:20, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:30<03:11, 1.99MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:30<02:20, 2.69MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:32<04:46, 1.32MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:32<04:17, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:32<03:13, 1.94MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:34<03:23, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:34<02:55, 2.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:34<02:13, 2.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:34<01:37, 3.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:36<12:42, 485kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:36<09:31, 646kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:36<06:47, 902kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:38<06:06, 997kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:38<04:53, 1.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:38<03:34, 1.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:38<02:34, 2.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:40<21:16, 283kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:40<15:45, 382kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:40<11:12, 536kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:40<07:55, 754kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:42<07:46, 766kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:42<06:15, 949kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:42<04:34, 1.30MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:42<03:15, 1.81MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:44<10:43, 549kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:44<08:16, 711kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:44<05:56, 985kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:44<04:13, 1.38MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:46<08:54, 653kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:46<07:00, 830kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:46<05:16, 1.10MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:46<03:50, 1.51MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:46<02:45, 2.09MB/s].vector_cache/glove.6B.zip:  60%|██████    | 517M/862M [03:48<18:28, 311kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:48<13:35, 422kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:48<09:37, 594kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:50<07:54, 718kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:50<06:10, 919kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:50<04:26, 1.27MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:52<04:17, 1.31MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:52<04:09, 1.35MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:52<03:11, 1.75MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:52<02:17, 2.43MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:54<05:45, 963kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:54<04:35, 1.20MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:54<03:20, 1.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:54<02:24, 2.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:56<10:48, 506kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:56<08:20, 655kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:56<05:59, 910kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:56<04:20, 1.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:58<04:33, 1.19MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:58<03:58, 1.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<02:56, 1.83MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:58<02:08, 2.50MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:00<04:17, 1.24MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:00<03:39, 1.45MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:00<02:42, 1.96MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<01:57, 2.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:02<06:35, 798kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:02<05:11, 1.01MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:02<03:44, 1.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:04<03:44, 1.38MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:04<03:42, 1.40MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:04<02:48, 1.84MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:04<02:05, 2.47MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:05<02:44, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:06<02:26, 2.10MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:06<01:48, 2.82MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:07<02:26, 2.06MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:08<02:15, 2.23MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:08<01:42, 2.93MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:09<02:17, 2.18MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:10<02:40, 1.87MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:10<02:08, 2.33MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:10<01:35, 3.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<01:12, 4.09MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:11<16:36, 296kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:12<12:27, 394kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:12<08:54, 550kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:12<06:16, 775kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:13<06:15, 775kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:14<05:12, 930kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:14<03:47, 1.27MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:14<02:45, 1.75MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:15<03:24, 1.40MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:15<03:00, 1.59MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<02:13, 2.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<01:37, 2.89MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:17<03:47, 1.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:17<03:18, 1.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<02:27, 1.91MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:18<01:46, 2.62MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:19<04:22, 1.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:19<03:41, 1.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:20<02:44, 1.69MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:21<02:46, 1.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:21<02:32, 1.79MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<01:53, 2.40MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:22<01:24, 3.22MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:23<03:07, 1.44MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:23<02:46, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:24<02:04, 2.16MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:25<02:18, 1.92MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:25<02:05, 2.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:25<01:38, 2.70MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:26<01:12, 3.65MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:27<02:57, 1.47MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:27<02:36, 1.67MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:27<01:56, 2.23MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<01:24, 3.05MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:29<07:31, 570kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:29<05:47, 740kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:29<04:08, 1.03MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:30<02:57, 1.43MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:31<04:31, 934kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:31<04:12, 1.00MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:31<03:10, 1.32MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:32<02:18, 1.82MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:33<02:40, 1.55MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:33<02:18, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:33<01:42, 2.41MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:35<02:09, 1.90MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:35<01:59, 2.06MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:35<01:29, 2.72MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:36<01:06, 3.66MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:37<03:17, 1.22MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:37<02:48, 1.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:37<02:02, 1.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:37<01:29, 2.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:39<03:33, 1.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<02:53, 1.36MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:39<02:15, 1.74MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:39<01:37, 2.39MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:41<02:26, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:41<02:09, 1.79MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:41<01:35, 2.42MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:41<01:10, 3.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:43<03:45, 1.02MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:43<03:01, 1.26MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:43<02:12, 1.72MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<02:24, 1.56MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:45<02:09, 1.73MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:45<01:36, 2.31MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:45<01:10, 3.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<02:50, 1.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:47<02:24, 1.53MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:47<01:45, 2.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:47<01:16, 2.84MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:49<12:13, 295kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:49<08:57, 402kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:49<06:18, 566kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<05:08, 687kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:51<04:25, 798kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:51<03:47, 932kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:51<03:21, 1.05MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:51<02:58, 1.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:52<02:14, 1.56MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:52<01:37, 2.15MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:54<02:50, 1.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:54<02:45, 1.26MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:54<02:05, 1.65MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<01:32, 2.23MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<01:51, 1.83MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<01:38, 2.06MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:56<01:13, 2.73MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:58<01:38, 2.03MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:58<01:29, 2.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:58<01:06, 2.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:00<01:32, 2.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:00<01:44, 1.86MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:00<01:22, 2.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:00<01:01, 3.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<00:48, 3.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:00<00:42, 4.54MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:01<18:27, 173kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:02<13:13, 241kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:02<09:15, 341kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:03<07:09, 435kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:04<05:41, 548kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:04<04:05, 759kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:04<02:56, 1.05MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:04<02:05, 1.46MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:05<04:23, 693kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:05<03:38, 837kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<02:47, 1.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:06<02:01, 1.49MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:07<02:00, 1.49MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:07<01:45, 1.69MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<01:18, 2.25MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:08<00:56, 3.08MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:09<07:15, 401kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:09<05:27, 533kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<03:51, 748kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:10<02:43, 1.05MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<03:29, 814kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:11<03:03, 928kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:12<02:17, 1.24MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:12<01:37, 1.72MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<02:27, 1.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:13<02:00, 1.38MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:14<01:27, 1.88MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<01:38, 1.65MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:15<01:25, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:16<01:02, 2.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:16<00:46, 3.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<04:25, 595kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:17<03:25, 769kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:18<02:27, 1.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<02:12, 1.16MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<02:11, 1.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:19<01:39, 1.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<01:14, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<01:21, 1.84MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<01:12, 2.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:21<00:55, 2.66MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:22<00:41, 3.54MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:22<00:34, 4.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<02:26, 993kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<02:07, 1.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:23<01:34, 1.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:24<01:10, 2.05MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<01:20, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<01:14, 1.90MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:25<00:55, 2.52MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:26<00:40, 3.40MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<01:48, 1.26MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:27<01:31, 1.50MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:27<01:06, 2.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:28<00:48, 2.75MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<01:51, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<01:37, 1.36MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:29<01:12, 1.82MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:30<00:51, 2.50MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:31<01:57, 1.10MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:31<01:39, 1.29MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:31<01:13, 1.73MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:32<00:52, 2.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:33<01:47, 1.16MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:33, 1.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<01:10, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:33<00:52, 2.34MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:34<00:39, 3.06MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:35<01:27, 1.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<01:29, 1.35MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<01:09, 1.72MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:35<00:50, 2.34MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:36<00:37, 3.10MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:37<02:19, 837kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<01:56, 998kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:37<01:25, 1.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:37<01:03, 1.81MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:38<00:46, 2.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<01:25, 1.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<01:24, 1.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:39<01:05, 1.70MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<00:47, 2.31MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<01:01, 1.76MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<00:59, 1.83MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<00:44, 2.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<00:34, 3.03MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:41<00:26, 3.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:43<01:59, 868kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:43<01:37, 1.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:43<01:15, 1.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:43<00:55, 1.83MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:43<00:41, 2.46MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:45<01:02, 1.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:45<00:57, 1.73MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:45<00:43, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:45<00:32, 2.98MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:47<00:49, 1.94MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<01:00, 1.57MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<00:47, 1.99MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:47<00:35, 2.61MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:48<00:25, 3.56MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:49<04:10, 366kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:49<03:20, 456kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:50<02:25, 623kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:50<01:42, 874kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:51<01:31, 951kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:51<01:14, 1.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:52<00:53, 1.61MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:52<00:38, 2.21MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:54<01:38, 849kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:54<02:31, 549kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:54<02:11, 633kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:54<01:40, 823kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:55<01:12, 1.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:55<00:50, 1.57MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:56<01:34, 842kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:56<01:14, 1.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:56<00:54, 1.44MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:56<00:38, 1.99MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:58<01:04, 1.16MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:58<00:53, 1.40MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:58<00:39, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:58<00:28, 2.51MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:59<00:21, 3.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:00<02:24, 491kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:00<01:53, 624kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:00<01:23, 845kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:00<00:58, 1.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:01<00:41, 1.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:02<01:08, 976kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:02<01:01, 1.08MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:02<00:45, 1.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:02<00:32, 1.98MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:04<00:43, 1.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:04<00:37, 1.65MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:04<00:28, 2.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:19, 2.98MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:06<00:47, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:06<00:39, 1.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:06<00:29, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:20, 2.70MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:08<00:53, 1.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:08<00:42, 1.28MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:08<00:31, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:08<00:22, 2.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:10<00:30, 1.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:10<00:26, 1.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:10<00:18, 2.55MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:10<00:14, 3.28MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:12<00:27, 1.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:12<00:24, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:12<00:17, 2.55MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:12<00:12, 3.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:14<00:26, 1.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:14<00:23, 1.74MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:14<00:17, 2.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:14<00:12, 3.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:16<00:36, 1.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:16<00:29, 1.27MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:16<00:20, 1.74MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:18<00:21, 1.56MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:18<00:19, 1.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:18<00:15, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:18<00:11, 2.83MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:20<00:14, 2.06MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:20<00:13, 2.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:20<00:09, 2.88MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<00:06, 3.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:22<00:44, 569kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:22<00:33, 753kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:22<00:24, 1.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:22<00:16, 1.41MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:24<00:17, 1.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:24<00:14, 1.47MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:24<00:11, 1.84MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:24<00:07, 2.49MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:05, 3.37MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:26<01:41, 169kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:26<01:12, 233kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:26<00:48, 330kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:26<00:31, 465kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:26<00:20, 655kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:28<00:32, 405kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:28<00:24, 524kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:28<00:16, 724kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:28<00:09, 1.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:30<00:09, 937kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:30<00:07, 1.10MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:30<00:05, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:30<00:02, 2.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:32<00:03, 1.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:32<00:02, 1.66MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:32<00:01, 2.04MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:32<00:00, 2.79MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:34<00:00, 1.72MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:34<00:00, 2.01MB/s].vector_cache/glove.6B.zip: 862MB [06:34, 2.19MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 849/400000 [00:00<00:47, 8481.68it/s]  0%|          | 1759/400000 [00:00<00:45, 8657.99it/s]  1%|          | 2582/400000 [00:00<00:46, 8522.84it/s]  1%|          | 3455/400000 [00:00<00:46, 8582.06it/s]  1%|          | 4367/400000 [00:00<00:45, 8735.30it/s]  1%|▏         | 5236/400000 [00:00<00:45, 8720.27it/s]  2%|▏         | 6093/400000 [00:00<00:45, 8664.99it/s]  2%|▏         | 6891/400000 [00:00<00:46, 8436.32it/s]  2%|▏         | 7777/400000 [00:00<00:45, 8557.82it/s]  2%|▏         | 8668/400000 [00:01<00:45, 8656.45it/s]  2%|▏         | 9515/400000 [00:01<00:45, 8599.19it/s]  3%|▎         | 10376/400000 [00:01<00:45, 8601.82it/s]  3%|▎         | 11284/400000 [00:01<00:44, 8737.70it/s]  3%|▎         | 12199/400000 [00:01<00:43, 8854.93it/s]  3%|▎         | 13088/400000 [00:01<00:43, 8864.79it/s]  3%|▎         | 13974/400000 [00:01<00:43, 8862.09it/s]  4%|▎         | 14858/400000 [00:01<00:44, 8666.56it/s]  4%|▍         | 15751/400000 [00:01<00:43, 8741.23it/s]  4%|▍         | 16625/400000 [00:01<00:44, 8601.53it/s]  4%|▍         | 17486/400000 [00:02<00:44, 8553.29it/s]  5%|▍         | 18357/400000 [00:02<00:44, 8598.16it/s]  5%|▍         | 19222/400000 [00:02<00:44, 8611.16it/s]  5%|▌         | 20114/400000 [00:02<00:43, 8700.90it/s]  5%|▌         | 20998/400000 [00:02<00:43, 8740.12it/s]  5%|▌         | 21887/400000 [00:02<00:43, 8783.46it/s]  6%|▌         | 22786/400000 [00:02<00:42, 8842.70it/s]  6%|▌         | 23671/400000 [00:02<00:44, 8508.89it/s]  6%|▌         | 24525/400000 [00:02<00:44, 8399.19it/s]  6%|▋         | 25370/400000 [00:02<00:44, 8410.82it/s]  7%|▋         | 26246/400000 [00:03<00:43, 8511.34it/s]  7%|▋         | 27107/400000 [00:03<00:43, 8537.83it/s]  7%|▋         | 27978/400000 [00:03<00:43, 8586.28it/s]  7%|▋         | 28914/400000 [00:03<00:42, 8802.50it/s]  7%|▋         | 29842/400000 [00:03<00:41, 8939.68it/s]  8%|▊         | 30766/400000 [00:03<00:40, 9026.69it/s]  8%|▊         | 31671/400000 [00:03<00:42, 8660.88it/s]  8%|▊         | 32553/400000 [00:03<00:42, 8707.79it/s]  8%|▊         | 33455/400000 [00:03<00:41, 8797.74it/s]  9%|▊         | 34361/400000 [00:03<00:41, 8872.92it/s]  9%|▉         | 35288/400000 [00:04<00:40, 8987.70it/s]  9%|▉         | 36189/400000 [00:04<00:41, 8869.08it/s]  9%|▉         | 37078/400000 [00:04<00:41, 8676.74it/s]  9%|▉         | 37989/400000 [00:04<00:41, 8801.25it/s] 10%|▉         | 38926/400000 [00:04<00:40, 8962.52it/s] 10%|▉         | 39845/400000 [00:04<00:39, 9027.69it/s] 10%|█         | 40750/400000 [00:04<00:40, 8941.64it/s] 10%|█         | 41646/400000 [00:04<00:40, 8881.84it/s] 11%|█         | 42557/400000 [00:04<00:39, 8946.44it/s] 11%|█         | 43475/400000 [00:04<00:39, 9014.19it/s] 11%|█         | 44378/400000 [00:05<00:39, 8985.48it/s] 11%|█▏        | 45278/400000 [00:05<00:40, 8854.17it/s] 12%|█▏        | 46165/400000 [00:05<00:40, 8730.84it/s] 12%|█▏        | 47040/400000 [00:05<00:40, 8664.54it/s] 12%|█▏        | 47915/400000 [00:05<00:40, 8688.18it/s] 12%|█▏        | 48828/400000 [00:05<00:39, 8816.06it/s] 12%|█▏        | 49711/400000 [00:05<00:40, 8732.25it/s] 13%|█▎        | 50618/400000 [00:05<00:39, 8829.22it/s] 13%|█▎        | 51502/400000 [00:05<00:39, 8761.68it/s] 13%|█▎        | 52379/400000 [00:05<00:39, 8734.61it/s] 13%|█▎        | 53288/400000 [00:06<00:39, 8836.45it/s] 14%|█▎        | 54173/400000 [00:06<00:39, 8814.41it/s] 14%|█▍        | 55055/400000 [00:06<00:39, 8799.03it/s] 14%|█▍        | 55936/400000 [00:06<00:39, 8728.57it/s] 14%|█▍        | 56842/400000 [00:06<00:38, 8823.02it/s] 14%|█▍        | 57748/400000 [00:06<00:38, 8891.01it/s] 15%|█▍        | 58638/400000 [00:06<00:38, 8778.14it/s] 15%|█▍        | 59530/400000 [00:06<00:38, 8819.27it/s] 15%|█▌        | 60413/400000 [00:06<00:38, 8789.08it/s] 15%|█▌        | 61293/400000 [00:07<00:39, 8569.36it/s] 16%|█▌        | 62152/400000 [00:07<00:39, 8517.15it/s] 16%|█▌        | 63012/400000 [00:07<00:39, 8540.44it/s] 16%|█▌        | 63890/400000 [00:07<00:39, 8609.31it/s] 16%|█▌        | 64797/400000 [00:07<00:38, 8740.59it/s] 16%|█▋        | 65673/400000 [00:07<00:38, 8635.03it/s] 17%|█▋        | 66575/400000 [00:07<00:38, 8745.89it/s] 17%|█▋        | 67452/400000 [00:07<00:37, 8752.27it/s] 17%|█▋        | 68374/400000 [00:07<00:37, 8885.26it/s] 17%|█▋        | 69272/400000 [00:07<00:37, 8912.05it/s] 18%|█▊        | 70197/400000 [00:08<00:36, 9010.08it/s] 18%|█▊        | 71099/400000 [00:08<00:36, 9012.45it/s] 18%|█▊        | 72001/400000 [00:08<00:37, 8840.90it/s] 18%|█▊        | 72887/400000 [00:08<00:37, 8775.09it/s] 18%|█▊        | 73789/400000 [00:08<00:36, 8844.72it/s] 19%|█▊        | 74675/400000 [00:08<00:36, 8813.32it/s] 19%|█▉        | 75557/400000 [00:08<00:37, 8767.23it/s] 19%|█▉        | 76435/400000 [00:08<00:37, 8713.58it/s] 19%|█▉        | 77307/400000 [00:08<00:37, 8633.19it/s] 20%|█▉        | 78171/400000 [00:08<00:37, 8538.71it/s] 20%|█▉        | 79060/400000 [00:09<00:37, 8639.27it/s] 20%|█▉        | 79981/400000 [00:09<00:36, 8802.23it/s] 20%|██        | 80863/400000 [00:09<00:36, 8648.92it/s] 20%|██        | 81749/400000 [00:09<00:36, 8709.63it/s] 21%|██        | 82632/400000 [00:09<00:36, 8743.13it/s] 21%|██        | 83542/400000 [00:09<00:35, 8844.68it/s] 21%|██        | 84428/400000 [00:09<00:36, 8712.88it/s] 21%|██▏       | 85301/400000 [00:09<00:36, 8667.58it/s] 22%|██▏       | 86192/400000 [00:09<00:35, 8736.57it/s] 22%|██▏       | 87067/400000 [00:09<00:35, 8723.58it/s] 22%|██▏       | 87943/400000 [00:10<00:35, 8734.41it/s] 22%|██▏       | 88830/400000 [00:10<00:35, 8773.51it/s] 22%|██▏       | 89708/400000 [00:10<00:36, 8618.88it/s] 23%|██▎       | 90571/400000 [00:10<00:36, 8584.70it/s] 23%|██▎       | 91431/400000 [00:10<00:35, 8586.63it/s] 23%|██▎       | 92341/400000 [00:10<00:35, 8733.05it/s] 23%|██▎       | 93271/400000 [00:10<00:34, 8894.16it/s] 24%|██▎       | 94162/400000 [00:10<00:34, 8891.87it/s] 24%|██▍       | 95061/400000 [00:10<00:34, 8918.54it/s] 24%|██▍       | 95954/400000 [00:10<00:34, 8918.05it/s] 24%|██▍       | 96847/400000 [00:11<00:34, 8896.63it/s] 24%|██▍       | 97775/400000 [00:11<00:33, 9006.38it/s] 25%|██▍       | 98677/400000 [00:11<00:33, 8864.28it/s] 25%|██▍       | 99565/400000 [00:11<00:34, 8777.96it/s] 25%|██▌       | 100459/400000 [00:11<00:33, 8825.55it/s] 25%|██▌       | 101400/400000 [00:11<00:33, 8991.78it/s] 26%|██▌       | 102328/400000 [00:11<00:32, 9074.58it/s] 26%|██▌       | 103237/400000 [00:11<00:32, 9031.66it/s] 26%|██▌       | 104141/400000 [00:11<00:33, 8889.64it/s] 26%|██▋       | 105032/400000 [00:11<00:33, 8767.24it/s] 26%|██▋       | 105931/400000 [00:12<00:33, 8830.73it/s] 27%|██▋       | 106815/400000 [00:12<00:33, 8770.42it/s] 27%|██▋       | 107693/400000 [00:12<00:34, 8528.42it/s] 27%|██▋       | 108561/400000 [00:12<00:33, 8572.56it/s] 27%|██▋       | 109474/400000 [00:12<00:33, 8730.37it/s] 28%|██▊       | 110408/400000 [00:12<00:32, 8902.38it/s] 28%|██▊       | 111346/400000 [00:12<00:31, 9040.47it/s] 28%|██▊       | 112252/400000 [00:12<00:32, 8978.42it/s] 28%|██▊       | 113152/400000 [00:12<00:32, 8807.02it/s] 29%|██▊       | 114092/400000 [00:13<00:31, 8974.29it/s] 29%|██▉       | 115009/400000 [00:13<00:31, 9032.08it/s] 29%|██▉       | 115918/400000 [00:13<00:31, 9047.99it/s] 29%|██▉       | 116824/400000 [00:13<00:31, 8931.54it/s] 29%|██▉       | 117719/400000 [00:13<00:31, 8907.24it/s] 30%|██▉       | 118611/400000 [00:13<00:31, 8840.00it/s] 30%|██▉       | 119496/400000 [00:13<00:32, 8706.55it/s] 30%|███       | 120374/400000 [00:13<00:32, 8726.10it/s] 30%|███       | 121248/400000 [00:13<00:32, 8592.58it/s] 31%|███       | 122143/400000 [00:13<00:31, 8694.45it/s] 31%|███       | 123042/400000 [00:14<00:31, 8780.59it/s] 31%|███       | 123954/400000 [00:14<00:31, 8877.25it/s] 31%|███       | 124881/400000 [00:14<00:30, 8989.00it/s] 31%|███▏      | 125781/400000 [00:14<00:30, 8888.17it/s] 32%|███▏      | 126671/400000 [00:14<00:30, 8873.87it/s] 32%|███▏      | 127608/400000 [00:14<00:30, 9016.38it/s] 32%|███▏      | 128511/400000 [00:14<00:30, 8979.55it/s] 32%|███▏      | 129436/400000 [00:14<00:29, 9058.67it/s] 33%|███▎      | 130343/400000 [00:14<00:30, 8973.59it/s] 33%|███▎      | 131242/400000 [00:14<00:29, 8966.34it/s] 33%|███▎      | 132166/400000 [00:15<00:29, 9045.62it/s] 33%|███▎      | 133072/400000 [00:15<00:29, 8947.63it/s] 33%|███▎      | 133968/400000 [00:15<00:31, 8553.24it/s] 34%|███▎      | 134828/400000 [00:15<00:31, 8375.20it/s] 34%|███▍      | 135691/400000 [00:15<00:31, 8448.11it/s] 34%|███▍      | 136602/400000 [00:15<00:30, 8635.34it/s] 34%|███▍      | 137512/400000 [00:15<00:29, 8767.73it/s] 35%|███▍      | 138417/400000 [00:15<00:29, 8849.48it/s] 35%|███▍      | 139304/400000 [00:15<00:29, 8754.01it/s] 35%|███▌      | 140185/400000 [00:15<00:29, 8769.55it/s] 35%|███▌      | 141065/400000 [00:16<00:29, 8776.29it/s] 35%|███▌      | 141979/400000 [00:16<00:29, 8881.63it/s] 36%|███▌      | 142881/400000 [00:16<00:28, 8922.42it/s] 36%|███▌      | 143774/400000 [00:16<00:29, 8802.48it/s] 36%|███▌      | 144656/400000 [00:16<00:29, 8780.18it/s] 36%|███▋      | 145565/400000 [00:16<00:28, 8868.71it/s] 37%|███▋      | 146486/400000 [00:16<00:28, 8967.29it/s] 37%|███▋      | 147409/400000 [00:16<00:27, 9044.45it/s] 37%|███▋      | 148315/400000 [00:16<00:29, 8663.80it/s] 37%|███▋      | 149186/400000 [00:16<00:29, 8485.78it/s] 38%|███▊      | 150059/400000 [00:17<00:29, 8555.52it/s] 38%|███▊      | 150952/400000 [00:17<00:28, 8660.60it/s] 38%|███▊      | 151830/400000 [00:17<00:28, 8694.21it/s] 38%|███▊      | 152701/400000 [00:17<00:28, 8684.55it/s] 38%|███▊      | 153579/400000 [00:17<00:28, 8710.89it/s] 39%|███▊      | 154483/400000 [00:17<00:27, 8804.73it/s] 39%|███▉      | 155365/400000 [00:17<00:27, 8808.77it/s] 39%|███▉      | 156247/400000 [00:17<00:27, 8807.53it/s] 39%|███▉      | 157129/400000 [00:17<00:28, 8559.63it/s] 40%|███▉      | 158001/400000 [00:18<00:28, 8606.55it/s] 40%|███▉      | 158863/400000 [00:18<00:28, 8604.63it/s] 40%|███▉      | 159725/400000 [00:18<00:28, 8557.11it/s] 40%|████      | 160626/400000 [00:18<00:27, 8685.36it/s] 40%|████      | 161496/400000 [00:18<00:27, 8601.41it/s] 41%|████      | 162395/400000 [00:18<00:27, 8713.54it/s] 41%|████      | 163268/400000 [00:18<00:27, 8515.37it/s] 41%|████      | 164137/400000 [00:18<00:27, 8564.63it/s] 41%|████      | 164995/400000 [00:18<00:27, 8549.24it/s] 41%|████▏     | 165851/400000 [00:18<00:27, 8377.85it/s] 42%|████▏     | 166707/400000 [00:19<00:27, 8431.39it/s] 42%|████▏     | 167552/400000 [00:19<00:28, 8189.45it/s] 42%|████▏     | 168413/400000 [00:19<00:27, 8309.55it/s] 42%|████▏     | 169265/400000 [00:19<00:27, 8370.13it/s] 43%|████▎     | 170105/400000 [00:19<00:27, 8376.69it/s] 43%|████▎     | 170974/400000 [00:19<00:27, 8466.44it/s] 43%|████▎     | 171868/400000 [00:19<00:26, 8600.87it/s] 43%|████▎     | 172773/400000 [00:19<00:26, 8730.34it/s] 43%|████▎     | 173679/400000 [00:19<00:25, 8825.71it/s] 44%|████▎     | 174563/400000 [00:19<00:26, 8530.86it/s] 44%|████▍     | 175448/400000 [00:20<00:26, 8622.58it/s] 44%|████▍     | 176313/400000 [00:20<00:26, 8547.06it/s] 44%|████▍     | 177170/400000 [00:20<00:26, 8349.36it/s] 45%|████▍     | 178008/400000 [00:20<00:27, 8179.07it/s] 45%|████▍     | 178829/400000 [00:20<00:27, 8136.10it/s] 45%|████▍     | 179679/400000 [00:20<00:26, 8240.18it/s] 45%|████▌     | 180539/400000 [00:20<00:26, 8343.54it/s] 45%|████▌     | 181420/400000 [00:20<00:25, 8477.27it/s] 46%|████▌     | 182309/400000 [00:20<00:25, 8596.64it/s] 46%|████▌     | 183176/400000 [00:20<00:25, 8616.33it/s] 46%|████▌     | 184077/400000 [00:21<00:24, 8728.56it/s] 46%|████▌     | 184979/400000 [00:21<00:24, 8812.26it/s] 46%|████▋     | 185862/400000 [00:21<00:24, 8767.22it/s] 47%|████▋     | 186740/400000 [00:21<00:25, 8496.73it/s] 47%|████▋     | 187592/400000 [00:21<00:25, 8324.28it/s] 47%|████▋     | 188427/400000 [00:21<00:25, 8276.12it/s] 47%|████▋     | 189283/400000 [00:21<00:25, 8358.87it/s] 48%|████▊     | 190181/400000 [00:21<00:24, 8533.60it/s] 48%|████▊     | 191037/400000 [00:21<00:24, 8457.28it/s] 48%|████▊     | 191885/400000 [00:21<00:24, 8349.14it/s] 48%|████▊     | 192782/400000 [00:22<00:24, 8525.13it/s] 48%|████▊     | 193695/400000 [00:22<00:23, 8696.70it/s] 49%|████▊     | 194576/400000 [00:22<00:23, 8727.75it/s] 49%|████▉     | 195459/400000 [00:22<00:23, 8756.18it/s] 49%|████▉     | 196336/400000 [00:22<00:23, 8641.13it/s] 49%|████▉     | 197202/400000 [00:22<00:23, 8619.77it/s] 50%|████▉     | 198065/400000 [00:22<00:23, 8419.02it/s] 50%|████▉     | 198927/400000 [00:22<00:23, 8478.21it/s] 50%|████▉     | 199804/400000 [00:22<00:23, 8561.60it/s] 50%|█████     | 200692/400000 [00:23<00:23, 8653.09it/s] 50%|█████     | 201595/400000 [00:23<00:22, 8762.46it/s] 51%|█████     | 202473/400000 [00:23<00:22, 8674.89it/s] 51%|█████     | 203357/400000 [00:23<00:22, 8722.38it/s] 51%|█████     | 204272/400000 [00:23<00:22, 8844.27it/s] 51%|█████▏    | 205158/400000 [00:23<00:22, 8780.90it/s] 52%|█████▏    | 206037/400000 [00:23<00:22, 8600.51it/s] 52%|█████▏    | 206899/400000 [00:23<00:22, 8496.57it/s] 52%|█████▏    | 207809/400000 [00:23<00:22, 8667.62it/s] 52%|█████▏    | 208705/400000 [00:23<00:21, 8753.09it/s] 52%|█████▏    | 209582/400000 [00:24<00:22, 8644.81it/s] 53%|█████▎    | 210463/400000 [00:24<00:21, 8691.23it/s] 53%|█████▎    | 211334/400000 [00:24<00:21, 8665.73it/s] 53%|█████▎    | 212202/400000 [00:24<00:22, 8529.63it/s] 53%|█████▎    | 213056/400000 [00:24<00:22, 8377.30it/s] 53%|█████▎    | 213896/400000 [00:24<00:22, 8341.96it/s] 54%|█████▎    | 214800/400000 [00:24<00:21, 8538.90it/s] 54%|█████▍    | 215697/400000 [00:24<00:21, 8661.24it/s] 54%|█████▍    | 216612/400000 [00:24<00:20, 8801.78it/s] 54%|█████▍    | 217514/400000 [00:24<00:20, 8863.13it/s] 55%|█████▍    | 218402/400000 [00:25<00:20, 8811.39it/s] 55%|█████▍    | 219314/400000 [00:25<00:20, 8900.92it/s] 55%|█████▌    | 220206/400000 [00:25<00:20, 8789.47it/s] 55%|█████▌    | 221086/400000 [00:25<00:21, 8318.15it/s] 55%|█████▌    | 221929/400000 [00:25<00:21, 8349.80it/s] 56%|█████▌    | 222771/400000 [00:25<00:21, 8369.18it/s] 56%|█████▌    | 223681/400000 [00:25<00:20, 8573.98it/s] 56%|█████▌    | 224542/400000 [00:25<00:20, 8538.11it/s] 56%|█████▋    | 225411/400000 [00:25<00:20, 8580.76it/s] 57%|█████▋    | 226305/400000 [00:25<00:19, 8684.75it/s] 57%|█████▋    | 227175/400000 [00:26<00:20, 8577.23it/s] 57%|█████▋    | 228035/400000 [00:26<00:20, 8538.99it/s] 57%|█████▋    | 228890/400000 [00:26<00:20, 8510.83it/s] 57%|█████▋    | 229777/400000 [00:26<00:19, 8613.91it/s] 58%|█████▊    | 230671/400000 [00:26<00:19, 8706.76it/s] 58%|█████▊    | 231543/400000 [00:26<00:19, 8695.55it/s] 58%|█████▊    | 232466/400000 [00:26<00:18, 8846.77it/s] 58%|█████▊    | 233403/400000 [00:26<00:18, 8994.50it/s] 59%|█████▊    | 234309/400000 [00:26<00:18, 9013.82it/s] 59%|█████▉    | 235212/400000 [00:26<00:18, 9000.23it/s] 59%|█████▉    | 236113/400000 [00:27<00:18, 8855.34it/s] 59%|█████▉    | 237032/400000 [00:27<00:18, 8952.62it/s] 59%|█████▉    | 237929/400000 [00:27<00:18, 8633.08it/s] 60%|█████▉    | 238837/400000 [00:27<00:18, 8760.40it/s] 60%|█████▉    | 239716/400000 [00:27<00:18, 8704.57it/s] 60%|██████    | 240589/400000 [00:27<00:18, 8650.99it/s] 60%|██████    | 241501/400000 [00:27<00:18, 8786.08it/s] 61%|██████    | 242382/400000 [00:27<00:18, 8710.69it/s] 61%|██████    | 243255/400000 [00:27<00:18, 8528.28it/s] 61%|██████    | 244195/400000 [00:28<00:17, 8771.33it/s] 61%|██████▏   | 245076/400000 [00:28<00:17, 8720.15it/s] 61%|██████▏   | 245989/400000 [00:28<00:17, 8837.47it/s] 62%|██████▏   | 246875/400000 [00:28<00:17, 8572.06it/s] 62%|██████▏   | 247771/400000 [00:28<00:17, 8683.35it/s] 62%|██████▏   | 248719/400000 [00:28<00:16, 8905.61it/s] 62%|██████▏   | 249613/400000 [00:28<00:17, 8679.47it/s] 63%|██████▎   | 250498/400000 [00:28<00:17, 8728.46it/s] 63%|██████▎   | 251413/400000 [00:28<00:16, 8849.08it/s] 63%|██████▎   | 252341/400000 [00:28<00:16, 8972.69it/s] 63%|██████▎   | 253245/400000 [00:29<00:16, 8990.13it/s] 64%|██████▎   | 254146/400000 [00:29<00:16, 8723.08it/s] 64%|██████▍   | 255039/400000 [00:29<00:16, 8780.98it/s] 64%|██████▍   | 255920/400000 [00:29<00:16, 8680.25it/s] 64%|██████▍   | 256790/400000 [00:29<00:16, 8559.52it/s] 64%|██████▍   | 257648/400000 [00:29<00:16, 8455.16it/s] 65%|██████▍   | 258495/400000 [00:29<00:17, 8308.14it/s] 65%|██████▍   | 259328/400000 [00:29<00:17, 8255.94it/s] 65%|██████▌   | 260161/400000 [00:29<00:16, 8277.27it/s] 65%|██████▌   | 261043/400000 [00:29<00:16, 8432.82it/s] 65%|██████▌   | 261902/400000 [00:30<00:16, 8477.95it/s] 66%|██████▌   | 262751/400000 [00:30<00:16, 8454.24it/s] 66%|██████▌   | 263598/400000 [00:30<00:16, 8343.19it/s] 66%|██████▌   | 264434/400000 [00:30<00:16, 8313.37it/s] 66%|██████▋   | 265267/400000 [00:30<00:16, 8316.32it/s] 67%|██████▋   | 266120/400000 [00:30<00:15, 8378.61it/s] 67%|██████▋   | 266959/400000 [00:30<00:16, 8312.92it/s] 67%|██████▋   | 267791/400000 [00:30<00:15, 8300.79it/s] 67%|██████▋   | 268648/400000 [00:30<00:15, 8377.63it/s] 67%|██████▋   | 269512/400000 [00:30<00:15, 8453.18it/s] 68%|██████▊   | 270410/400000 [00:31<00:15, 8603.42it/s] 68%|██████▊   | 271276/400000 [00:31<00:14, 8617.91it/s] 68%|██████▊   | 272139/400000 [00:31<00:14, 8580.45it/s] 68%|██████▊   | 272998/400000 [00:31<00:14, 8552.07it/s] 68%|██████▊   | 273882/400000 [00:31<00:14, 8633.67it/s] 69%|██████▊   | 274795/400000 [00:31<00:14, 8776.74it/s] 69%|██████▉   | 275676/400000 [00:31<00:14, 8786.52it/s] 69%|██████▉   | 276556/400000 [00:31<00:14, 8672.55it/s] 69%|██████▉   | 277457/400000 [00:31<00:13, 8770.13it/s] 70%|██████▉   | 278335/400000 [00:32<00:14, 8465.83it/s] 70%|██████▉   | 279185/400000 [00:32<00:14, 8459.89it/s] 70%|███████   | 280033/400000 [00:32<00:14, 8438.15it/s] 70%|███████   | 280914/400000 [00:32<00:13, 8545.99it/s] 70%|███████   | 281809/400000 [00:32<00:13, 8659.68it/s] 71%|███████   | 282710/400000 [00:32<00:13, 8761.56it/s] 71%|███████   | 283650/400000 [00:32<00:13, 8943.45it/s] 71%|███████   | 284547/400000 [00:32<00:13, 8761.47it/s] 71%|███████▏  | 285426/400000 [00:32<00:13, 8564.94it/s] 72%|███████▏  | 286285/400000 [00:32<00:13, 8420.99it/s] 72%|███████▏  | 287159/400000 [00:33<00:13, 8514.14it/s] 72%|███████▏  | 288038/400000 [00:33<00:13, 8593.77it/s] 72%|███████▏  | 288899/400000 [00:33<00:13, 8222.36it/s] 72%|███████▏  | 289780/400000 [00:33<00:13, 8388.94it/s] 73%|███████▎  | 290624/400000 [00:33<00:13, 8402.72it/s] 73%|███████▎  | 291532/400000 [00:33<00:12, 8593.02it/s] 73%|███████▎  | 292395/400000 [00:33<00:12, 8461.45it/s] 73%|███████▎  | 293244/400000 [00:33<00:13, 8089.25it/s] 74%|███████▎  | 294104/400000 [00:33<00:12, 8234.64it/s] 74%|███████▎  | 294963/400000 [00:33<00:12, 8336.19it/s] 74%|███████▍  | 295844/400000 [00:34<00:12, 8472.01it/s] 74%|███████▍  | 296711/400000 [00:34<00:12, 8527.65it/s] 74%|███████▍  | 297567/400000 [00:34<00:11, 8536.91it/s] 75%|███████▍  | 298480/400000 [00:34<00:11, 8704.05it/s] 75%|███████▍  | 299353/400000 [00:34<00:11, 8693.16it/s] 75%|███████▌  | 300250/400000 [00:34<00:11, 8773.30it/s] 75%|███████▌  | 301129/400000 [00:34<00:11, 8463.25it/s] 75%|███████▌  | 301979/400000 [00:34<00:11, 8463.53it/s] 76%|███████▌  | 302828/400000 [00:34<00:11, 8393.75it/s] 76%|███████▌  | 303675/400000 [00:34<00:11, 8416.35it/s] 76%|███████▌  | 304568/400000 [00:35<00:11, 8563.75it/s] 76%|███████▋  | 305426/400000 [00:35<00:11, 8567.14it/s] 77%|███████▋  | 306286/400000 [00:35<00:10, 8573.23it/s] 77%|███████▋  | 307145/400000 [00:35<00:11, 8356.54it/s] 77%|███████▋  | 307983/400000 [00:35<00:11, 8238.27it/s] 77%|███████▋  | 308848/400000 [00:35<00:10, 8357.43it/s] 77%|███████▋  | 309705/400000 [00:35<00:10, 8419.88it/s] 78%|███████▊  | 310574/400000 [00:35<00:10, 8499.00it/s] 78%|███████▊  | 311425/400000 [00:35<00:10, 8369.84it/s] 78%|███████▊  | 312370/400000 [00:36<00:10, 8665.45it/s] 78%|███████▊  | 313281/400000 [00:36<00:09, 8790.56it/s] 79%|███████▊  | 314163/400000 [00:36<00:09, 8586.50it/s] 79%|███████▉  | 315025/400000 [00:36<00:10, 8490.57it/s] 79%|███████▉  | 315877/400000 [00:36<00:09, 8421.96it/s] 79%|███████▉  | 316728/400000 [00:36<00:09, 8445.75it/s] 79%|███████▉  | 317611/400000 [00:36<00:09, 8555.52it/s] 80%|███████▉  | 318468/400000 [00:36<00:09, 8452.37it/s] 80%|███████▉  | 319329/400000 [00:36<00:09, 8498.46it/s] 80%|████████  | 320204/400000 [00:36<00:09, 8571.90it/s] 80%|████████  | 321062/400000 [00:37<00:09, 8563.30it/s] 80%|████████  | 321919/400000 [00:37<00:09, 8448.96it/s] 81%|████████  | 322765/400000 [00:37<00:09, 8390.52it/s] 81%|████████  | 323690/400000 [00:37<00:08, 8629.78it/s] 81%|████████  | 324670/400000 [00:37<00:08, 8947.92it/s] 81%|████████▏ | 325641/400000 [00:37<00:08, 9163.60it/s] 82%|████████▏ | 326562/400000 [00:37<00:08, 9113.04it/s] 82%|████████▏ | 327477/400000 [00:37<00:08, 8917.56it/s] 82%|████████▏ | 328383/400000 [00:37<00:07, 8959.28it/s] 82%|████████▏ | 329369/400000 [00:37<00:07, 9211.00it/s] 83%|████████▎ | 330364/400000 [00:38<00:07, 9418.00it/s] 83%|████████▎ | 331310/400000 [00:38<00:07, 9376.39it/s] 83%|████████▎ | 332251/400000 [00:38<00:07, 9065.38it/s] 83%|████████▎ | 333184/400000 [00:38<00:07, 9141.96it/s] 84%|████████▎ | 334120/400000 [00:38<00:07, 9203.71it/s] 84%|████████▍ | 335043/400000 [00:38<00:07, 8942.09it/s] 84%|████████▍ | 335957/400000 [00:38<00:07, 8998.99it/s] 84%|████████▍ | 336860/400000 [00:38<00:07, 8821.91it/s] 84%|████████▍ | 337756/400000 [00:38<00:07, 8862.50it/s] 85%|████████▍ | 338743/400000 [00:38<00:06, 9139.84it/s] 85%|████████▍ | 339718/400000 [00:39<00:06, 9311.13it/s] 85%|████████▌ | 340658/400000 [00:39<00:06, 9335.71it/s] 85%|████████▌ | 341594/400000 [00:39<00:06, 8993.81it/s] 86%|████████▌ | 342498/400000 [00:39<00:06, 8897.09it/s] 86%|████████▌ | 343391/400000 [00:39<00:06, 8781.03it/s] 86%|████████▌ | 344318/400000 [00:39<00:06, 8921.02it/s] 86%|████████▋ | 345234/400000 [00:39<00:06, 8989.45it/s] 87%|████████▋ | 346135/400000 [00:39<00:06, 8649.90it/s] 87%|████████▋ | 347115/400000 [00:39<00:05, 8963.01it/s] 87%|████████▋ | 348017/400000 [00:40<00:05, 8895.01it/s] 87%|████████▋ | 348985/400000 [00:40<00:05, 9114.72it/s] 87%|████████▋ | 349901/400000 [00:40<00:05, 8936.30it/s] 88%|████████▊ | 350799/400000 [00:40<00:05, 8767.51it/s] 88%|████████▊ | 351680/400000 [00:40<00:05, 8657.36it/s] 88%|████████▊ | 352640/400000 [00:40<00:05, 8919.00it/s] 88%|████████▊ | 353630/400000 [00:40<00:05, 9191.51it/s] 89%|████████▊ | 354591/400000 [00:40<00:04, 9311.52it/s] 89%|████████▉ | 355526/400000 [00:40<00:04, 9191.72it/s] 89%|████████▉ | 356470/400000 [00:40<00:04, 9262.15it/s] 89%|████████▉ | 357436/400000 [00:41<00:04, 9375.99it/s] 90%|████████▉ | 358397/400000 [00:41<00:04, 9443.24it/s] 90%|████████▉ | 359384/400000 [00:41<00:04, 9567.31it/s] 90%|█████████ | 360343/400000 [00:41<00:04, 9247.58it/s] 90%|█████████ | 361272/400000 [00:41<00:04, 9175.23it/s] 91%|█████████ | 362192/400000 [00:41<00:04, 9049.63it/s] 91%|█████████ | 363107/400000 [00:41<00:04, 9077.33it/s] 91%|█████████ | 364017/400000 [00:41<00:03, 9076.52it/s] 91%|█████████ | 364926/400000 [00:41<00:03, 8804.88it/s] 91%|█████████▏| 365884/400000 [00:41<00:03, 9021.13it/s] 92%|█████████▏| 366815/400000 [00:42<00:03, 9104.68it/s] 92%|█████████▏| 367763/400000 [00:42<00:03, 9212.69it/s] 92%|█████████▏| 368751/400000 [00:42<00:03, 9401.46it/s] 92%|█████████▏| 369694/400000 [00:42<00:03, 9081.18it/s] 93%|█████████▎| 370607/400000 [00:42<00:03, 8983.01it/s] 93%|█████████▎| 371509/400000 [00:42<00:03, 8979.24it/s] 93%|█████████▎| 372487/400000 [00:42<00:02, 9204.71it/s] 93%|█████████▎| 373411/400000 [00:42<00:02, 9158.78it/s] 94%|█████████▎| 374329/400000 [00:42<00:02, 9147.75it/s] 94%|█████████▍| 375246/400000 [00:42<00:02, 9103.80it/s] 94%|█████████▍| 376178/400000 [00:43<00:02, 9166.17it/s] 94%|█████████▍| 377106/400000 [00:43<00:02, 9200.01it/s] 95%|█████████▍| 378027/400000 [00:43<00:02, 8929.21it/s] 95%|█████████▍| 378945/400000 [00:43<00:02, 9001.23it/s] 95%|█████████▍| 379925/400000 [00:43<00:02, 9223.80it/s] 95%|█████████▌| 380850/400000 [00:43<00:02, 9069.07it/s] 95%|█████████▌| 381760/400000 [00:43<00:02, 9014.01it/s] 96%|█████████▌| 382664/400000 [00:43<00:02, 8527.33it/s] 96%|█████████▌| 383570/400000 [00:43<00:01, 8678.89it/s] 96%|█████████▌| 384498/400000 [00:44<00:01, 8848.59it/s] 96%|█████████▋| 385388/400000 [00:44<00:01, 8766.79it/s] 97%|█████████▋| 386269/400000 [00:44<00:01, 8679.32it/s] 97%|█████████▋| 387140/400000 [00:44<00:01, 8334.00it/s] 97%|█████████▋| 387979/400000 [00:44<00:01, 8341.25it/s] 97%|█████████▋| 388910/400000 [00:44<00:01, 8609.90it/s] 97%|█████████▋| 389793/400000 [00:44<00:01, 8673.53it/s] 98%|█████████▊| 390664/400000 [00:44<00:01, 8652.14it/s] 98%|█████████▊| 391532/400000 [00:44<00:00, 8489.78it/s] 98%|█████████▊| 392416/400000 [00:44<00:00, 8590.57it/s] 98%|█████████▊| 393319/400000 [00:45<00:00, 8715.89it/s] 99%|█████████▊| 394235/400000 [00:45<00:00, 8843.09it/s] 99%|█████████▉| 395121/400000 [00:45<00:00, 8821.27it/s] 99%|█████████▉| 396005/400000 [00:45<00:00, 8464.60it/s] 99%|█████████▉| 396856/400000 [00:45<00:00, 8158.89it/s] 99%|█████████▉| 397678/400000 [00:45<00:00, 8101.86it/s]100%|█████████▉| 398531/400000 [00:45<00:00, 8224.99it/s]100%|█████████▉| 399424/400000 [00:45<00:00, 8422.29it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8720.57it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f5a7e04bcc0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01074267235266122 	 Accuracy: 57
Train Epoch: 1 	 Loss: 0.011118469629000661 	 Accuracy: 53

  model saves at 53% accuracy 

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
2020-05-13 09:26:57.508316: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 09:26:57.511764: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 09:26:57.512482: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556d9c405960 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 09:26:57.512500: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f5a89bc7f98> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6206 - accuracy: 0.5030
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6973 - accuracy: 0.4980 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6564 - accuracy: 0.5007
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5210 - accuracy: 0.5095
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.4796 - accuracy: 0.5122
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5516 - accuracy: 0.5075
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5943 - accuracy: 0.5047
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6340 - accuracy: 0.5021
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6377 - accuracy: 0.5019
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6375 - accuracy: 0.5019
11000/25000 [============>.................] - ETA: 4s - loss: 7.6861 - accuracy: 0.4987
12000/25000 [=============>................] - ETA: 4s - loss: 7.6871 - accuracy: 0.4987
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6808 - accuracy: 0.4991
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6885 - accuracy: 0.4986
15000/25000 [=================>............] - ETA: 3s - loss: 7.6728 - accuracy: 0.4996
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6733 - accuracy: 0.4996
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6711 - accuracy: 0.4997
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6777 - accuracy: 0.4993
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6642 - accuracy: 0.5002
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6620 - accuracy: 0.5003
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6381 - accuracy: 0.5019
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6597 - accuracy: 0.5005
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6533 - accuracy: 0.5009
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6513 - accuracy: 0.5010
25000/25000 [==============================] - 9s 373us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f59e1ec5668> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f5a23b97400> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 936ms/step - loss: 1.2041 - crf_viterbi_accuracy: 0.3333 - val_loss: 1.1570 - val_crf_viterbi_accuracy: 0.2800

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


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f218704ef60> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 10:12:42.713725
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 10:12:42.718077
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 10:12:42.721782
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 10:12:42.725068
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f2192e18438> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 355255.7812
Epoch 2/10

1/1 [==============================] - 0s 122ms/step - loss: 251822.4062
Epoch 3/10

1/1 [==============================] - 0s 99ms/step - loss: 149348.6406
Epoch 4/10

1/1 [==============================] - 0s 105ms/step - loss: 74033.4766
Epoch 5/10

1/1 [==============================] - 0s 101ms/step - loss: 38018.3164
Epoch 6/10

1/1 [==============================] - 0s 106ms/step - loss: 21728.9180
Epoch 7/10

1/1 [==============================] - 0s 121ms/step - loss: 13747.0596
Epoch 8/10

1/1 [==============================] - 0s 101ms/step - loss: 9365.7559
Epoch 9/10

1/1 [==============================] - 0s 114ms/step - loss: 6788.2100
Epoch 10/10

1/1 [==============================] - 0s 102ms/step - loss: 5179.1416

  #### Inference Need return ypred, ytrue ######################### 
[[ 5.05298018e-01  9.93497276e+00  1.07705870e+01  1.13981876e+01
   1.10969868e+01  1.11188011e+01  9.31511211e+00  9.74743652e+00
   8.89193821e+00  1.15845900e+01  8.75329113e+00  1.05891113e+01
   7.88857079e+00  9.86190510e+00  1.00312128e+01  1.24891968e+01
   1.12480288e+01  9.10372066e+00  9.81602955e+00  7.81904268e+00
   7.34073925e+00  1.07562208e+01  1.11075287e+01  9.55082035e+00
   1.02845726e+01  8.01147556e+00  9.63366318e+00  9.57629395e+00
   1.12930431e+01  8.68293571e+00  9.20820236e+00  8.96977901e+00
   1.09721184e+01  1.02791090e+01  9.81431293e+00  8.06630325e+00
   1.00389729e+01  8.67725563e+00  9.26760006e+00  8.60407448e+00
   9.20152092e+00  8.41107941e+00  9.16114235e+00  1.06539497e+01
   1.18466244e+01  8.76936817e+00  6.15356493e+00  9.00297737e+00
   1.02880249e+01  9.19498253e+00  1.04497452e+01  9.10553169e+00
   9.52567101e+00  9.52587605e+00  7.77481890e+00  1.10125303e+01
   9.23026562e+00  1.04788857e+01  1.02196932e+01  8.65727043e+00
  -1.01769948e+00 -9.62181538e-02 -2.50497788e-01 -5.38472891e-01
  -1.50767338e+00  7.31437981e-01  7.04732656e-01  5.92505932e-03
   1.31691408e+00  3.88633877e-01 -2.13628769e-01  8.51131737e-01
   1.19496441e+00  2.04390860e+00  7.93192089e-02 -2.44857931e+00
   1.53215587e-01 -3.32959890e-01  2.91019464e+00  4.88236755e-01
  -1.32807505e+00 -2.51824093e+00  3.58156919e-01 -2.20769715e+00
  -1.30078745e+00  2.58820391e+00 -2.03913152e-01  9.13815439e-01
  -2.52168965e+00  1.22777092e+00  7.95335710e-01  1.93769431e+00
  -8.14612925e-01 -2.11677313e+00  6.92099154e-01  1.86911607e+00
  -6.57923579e-01 -3.44724655e-01  7.29053795e-01  3.10288072e-02
   2.01696068e-01 -9.88694787e-01 -1.08578694e+00  1.49971977e-01
   1.63051367e-01  4.05909657e-01  1.63331139e+00  2.57710719e+00
   6.58629656e-01 -1.44313502e+00  2.56877005e-01  7.07293212e-01
  -1.64343476e+00 -7.81094909e-01  8.54075134e-01  1.13583827e+00
   6.71391606e-01 -2.64262366e+00 -3.00546467e-01 -1.58444762e+00
   7.71860361e-01 -1.09038901e+00 -3.00871074e-01  1.14936912e+00
  -1.09163785e+00  7.22185493e-01  6.21662080e-01  1.68400824e+00
   9.76930380e-01 -1.39511538e+00 -1.15441728e+00  1.16289544e+00
   7.66706049e-01  9.24264908e-01  1.18936658e+00 -2.31606174e+00
  -4.11510676e-01  7.07174063e-01 -4.20017481e-01  6.54809237e-01
  -5.31001747e-01  1.05971622e+00  2.92751074e-01  1.30518341e+00
  -1.89364642e-01 -6.77875519e-01 -9.41482782e-02  1.27619052e+00
  -1.07837629e+00 -5.71417987e-01 -1.97236478e-01 -4.37033296e-01
  -1.51308560e+00 -2.98404217e-01 -3.79549861e-01  9.49761868e-01
   2.23954868e+00  5.65042496e-01  8.41501534e-01  1.36096311e+00
  -8.27791095e-01 -1.76527071e+00  5.61632633e-01 -1.09822369e+00
  -5.92217624e-01 -9.97389555e-02  3.85253787e-01 -3.56666386e-01
  -1.66593480e+00  4.91631657e-01  2.14414978e+00  1.04623675e-01
  -2.98987359e-01 -9.84981656e-03 -2.44503450e+00  4.25309062e-01
  -7.21592188e-01 -4.22907174e-01  4.74121094e-01  2.31308413e+00
   4.81204987e-01  8.55686474e+00  1.01441784e+01  9.13903141e+00
   8.64887714e+00  8.65773773e+00  9.46260834e+00  8.90674210e+00
   1.06023798e+01  1.06391163e+01  1.06013517e+01  9.00302029e+00
   9.72549534e+00  8.40041924e+00  1.01665049e+01  1.04293518e+01
   8.07978439e+00  1.03623781e+01  1.01986256e+01  9.67071819e+00
   9.90165234e+00  9.63571548e+00  8.29755974e+00  9.27119923e+00
   9.34011078e+00  1.00375795e+01  9.49497223e+00  9.71144295e+00
   8.47877216e+00  9.47844601e+00  8.59535313e+00  1.08304873e+01
   9.57209682e+00  9.27368546e+00  1.12751856e+01  8.26946735e+00
   1.06668501e+01  1.11664400e+01  1.07348709e+01  9.01179695e+00
   1.04935055e+01  1.01848574e+01  8.37026215e+00  1.02855024e+01
   9.60578918e+00  1.09233170e+01  9.75415897e+00  9.44019890e+00
   9.83027840e+00  7.71709442e+00  8.80432034e+00  9.94555092e+00
   1.02101336e+01  1.06838999e+01  8.04531956e+00  8.41587448e+00
   9.28612423e+00  1.02733679e+01  1.02320623e+01  9.28122425e+00
   1.55919909e-01  9.82509851e-02  1.80517137e+00  3.04717875e+00
   7.91901350e-02  3.74690175e-01  1.06533527e+00  9.12155330e-01
   1.77794635e-01  1.17440057e+00  1.23089409e+00  4.57062960e-01
   5.71897447e-01  6.22060239e-01  6.61396742e-01  4.31123912e-01
   2.08265471e+00  1.51651382e+00  9.23042893e-01  7.43996501e-01
   2.58436108e+00  1.40512371e+00  6.61863804e-01  4.01898086e-01
   4.61300850e-01  2.39733219e+00  2.65924263e+00  1.36544025e+00
   1.93732905e+00  3.39226246e-01  1.78811610e-01  1.31642354e+00
   3.15000534e-01  2.13653922e-01  3.22488737e+00  1.80387092e+00
   5.56367099e-01  8.55684280e-02  2.65216589e-01  1.80838859e+00
   4.26677167e-01  7.81189978e-01  7.15863943e-01  1.31468081e+00
   2.58047342e+00  1.90874398e+00  1.12483180e+00  1.74502492e-01
   5.01618862e-01  1.75693357e+00  9.99948025e-01  2.02824354e+00
   1.16144931e+00  3.33182871e-01  4.04317796e-01  1.59293818e+00
   1.67831516e+00  1.57226229e+00  3.89495671e-01  1.30140662e-01
   1.43158579e+00  3.03150177e+00  1.56300414e+00  9.87752438e-01
   7.15661705e-01  7.61768043e-01  3.02923012e+00  3.44962597e-01
   1.95733249e-01  2.28567123e+00  1.09860492e+00  3.25053096e-01
   1.19466269e+00  1.23785651e+00  2.57083368e+00  2.92393565e-01
   2.43945420e-01  2.23938417e+00  8.29966068e-01  9.92879152e-01
   3.64954591e-01  1.22609782e+00  7.05320477e-01  3.44180107e-01
   1.03624225e+00  9.80494320e-01  1.49630475e+00  1.24041915e-01
   1.26652944e+00  2.77463770e+00  2.39109945e+00  2.41629720e-01
   3.10498595e-01  7.34627426e-01  4.61955905e-01  9.97891605e-01
   7.06161380e-01  2.45366907e+00  2.78690279e-01  1.74324930e+00
   2.56717682e+00  2.00810766e+00  1.32649457e+00  2.80807078e-01
   2.10031080e+00  9.21103358e-02  2.83746052e+00  6.79778099e-01
   2.80168796e+00  1.12323523e-01  1.73795033e+00  1.55519772e+00
   2.68730640e-01  1.23256922e-01  1.15355456e+00  1.95292556e+00
   1.02939415e+00  5.32601893e-01  2.65598059e-01  5.38377762e-01
   1.10453539e+01 -9.52302265e+00 -5.29394913e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 10:12:51.681847
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   92.1911
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 10:12:51.686002
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8522.65
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 10:12:51.689172
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   91.7508
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 10:12:51.692279
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -762.264
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139781616829776
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139780541030976
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139780541031480
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139780541031984
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139780541032488
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139780541032992

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f2186d0eb38> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.673304
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.643548
grad_step = 000002, loss = 0.622719
grad_step = 000003, loss = 0.599871
grad_step = 000004, loss = 0.574173
grad_step = 000005, loss = 0.547785
grad_step = 000006, loss = 0.522700
grad_step = 000007, loss = 0.502258
grad_step = 000008, loss = 0.493850
grad_step = 000009, loss = 0.491339
grad_step = 000010, loss = 0.476237
grad_step = 000011, loss = 0.458234
grad_step = 000012, loss = 0.444510
grad_step = 000013, loss = 0.435174
grad_step = 000014, loss = 0.427292
grad_step = 000015, loss = 0.418617
grad_step = 000016, loss = 0.408317
grad_step = 000017, loss = 0.396464
grad_step = 000018, loss = 0.383758
grad_step = 000019, loss = 0.371462
grad_step = 000020, loss = 0.360795
grad_step = 000021, loss = 0.352101
grad_step = 000022, loss = 0.343937
grad_step = 000023, loss = 0.334427
grad_step = 000024, loss = 0.323603
grad_step = 000025, loss = 0.312825
grad_step = 000026, loss = 0.303208
grad_step = 000027, loss = 0.294780
grad_step = 000028, loss = 0.286707
grad_step = 000029, loss = 0.278248
grad_step = 000030, loss = 0.269286
grad_step = 000031, loss = 0.260213
grad_step = 000032, loss = 0.251547
grad_step = 000033, loss = 0.243648
grad_step = 000034, loss = 0.236320
grad_step = 000035, loss = 0.228947
grad_step = 000036, loss = 0.221213
grad_step = 000037, loss = 0.213445
grad_step = 000038, loss = 0.206203
grad_step = 000039, loss = 0.199501
grad_step = 000040, loss = 0.192960
grad_step = 000041, loss = 0.186395
grad_step = 000042, loss = 0.179769
grad_step = 000043, loss = 0.173311
grad_step = 000044, loss = 0.167257
grad_step = 000045, loss = 0.161495
grad_step = 000046, loss = 0.155806
grad_step = 000047, loss = 0.150082
grad_step = 000048, loss = 0.144483
grad_step = 000049, loss = 0.139154
grad_step = 000050, loss = 0.134096
grad_step = 000051, loss = 0.129126
grad_step = 000052, loss = 0.124193
grad_step = 000053, loss = 0.119370
grad_step = 000054, loss = 0.114741
grad_step = 000055, loss = 0.110335
grad_step = 000056, loss = 0.106027
grad_step = 000057, loss = 0.101781
grad_step = 000058, loss = 0.097640
grad_step = 000059, loss = 0.093666
grad_step = 000060, loss = 0.089854
grad_step = 000061, loss = 0.086137
grad_step = 000062, loss = 0.082508
grad_step = 000063, loss = 0.078985
grad_step = 000064, loss = 0.075602
grad_step = 000065, loss = 0.072342
grad_step = 000066, loss = 0.069175
grad_step = 000067, loss = 0.066101
grad_step = 000068, loss = 0.063128
grad_step = 000069, loss = 0.060276
grad_step = 000070, loss = 0.057525
grad_step = 000071, loss = 0.054864
grad_step = 000072, loss = 0.052291
grad_step = 000073, loss = 0.049821
grad_step = 000074, loss = 0.047450
grad_step = 000075, loss = 0.045169
grad_step = 000076, loss = 0.042962
grad_step = 000077, loss = 0.040846
grad_step = 000078, loss = 0.038826
grad_step = 000079, loss = 0.036887
grad_step = 000080, loss = 0.035018
grad_step = 000081, loss = 0.033229
grad_step = 000082, loss = 0.031528
grad_step = 000083, loss = 0.029898
grad_step = 000084, loss = 0.028332
grad_step = 000085, loss = 0.026841
grad_step = 000086, loss = 0.025426
grad_step = 000087, loss = 0.024077
grad_step = 000088, loss = 0.022799
grad_step = 000089, loss = 0.021616
grad_step = 000090, loss = 0.020504
grad_step = 000091, loss = 0.019425
grad_step = 000092, loss = 0.018300
grad_step = 000093, loss = 0.017256
grad_step = 000094, loss = 0.016354
grad_step = 000095, loss = 0.015474
grad_step = 000096, loss = 0.014543
grad_step = 000097, loss = 0.013681
grad_step = 000098, loss = 0.012971
grad_step = 000099, loss = 0.012300
grad_step = 000100, loss = 0.011569
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.010871
grad_step = 000102, loss = 0.010288
grad_step = 000103, loss = 0.009761
grad_step = 000104, loss = 0.009204
grad_step = 000105, loss = 0.008650
grad_step = 000106, loss = 0.008175
grad_step = 000107, loss = 0.007762
grad_step = 000108, loss = 0.007357
grad_step = 000109, loss = 0.006936
grad_step = 000110, loss = 0.006563
grad_step = 000111, loss = 0.006274
grad_step = 000112, loss = 0.006020
grad_step = 000113, loss = 0.005723
grad_step = 000114, loss = 0.005379
grad_step = 000115, loss = 0.005076
grad_step = 000116, loss = 0.004805
grad_step = 000117, loss = 0.004516
grad_step = 000118, loss = 0.004287
grad_step = 000119, loss = 0.004151
grad_step = 000120, loss = 0.004015
grad_step = 000121, loss = 0.003814
grad_step = 000122, loss = 0.003607
grad_step = 000123, loss = 0.003462
grad_step = 000124, loss = 0.003328
grad_step = 000125, loss = 0.003174
grad_step = 000126, loss = 0.003050
grad_step = 000127, loss = 0.002981
grad_step = 000128, loss = 0.002919
grad_step = 000129, loss = 0.002824
grad_step = 000130, loss = 0.002738
grad_step = 000131, loss = 0.002684
grad_step = 000132, loss = 0.002626
grad_step = 000133, loss = 0.002540
grad_step = 000134, loss = 0.002460
grad_step = 000135, loss = 0.002408
grad_step = 000136, loss = 0.002364
grad_step = 000137, loss = 0.002306
grad_step = 000138, loss = 0.002252
grad_step = 000139, loss = 0.002220
grad_step = 000140, loss = 0.002196
grad_step = 000141, loss = 0.002166
grad_step = 000142, loss = 0.002133
grad_step = 000143, loss = 0.002110
grad_step = 000144, loss = 0.002097
grad_step = 000145, loss = 0.002085
grad_step = 000146, loss = 0.002073
grad_step = 000147, loss = 0.002079
grad_step = 000148, loss = 0.002143
grad_step = 000149, loss = 0.002341
grad_step = 000150, loss = 0.002794
grad_step = 000151, loss = 0.003114
grad_step = 000152, loss = 0.002764
grad_step = 000153, loss = 0.002122
grad_step = 000154, loss = 0.002318
grad_step = 000155, loss = 0.002650
grad_step = 000156, loss = 0.002324
grad_step = 000157, loss = 0.002087
grad_step = 000158, loss = 0.002471
grad_step = 000159, loss = 0.002290
grad_step = 000160, loss = 0.002020
grad_step = 000161, loss = 0.002350
grad_step = 000162, loss = 0.002178
grad_step = 000163, loss = 0.001985
grad_step = 000164, loss = 0.002276
grad_step = 000165, loss = 0.002020
grad_step = 000166, loss = 0.002035
grad_step = 000167, loss = 0.002160
grad_step = 000168, loss = 0.001965
grad_step = 000169, loss = 0.002055
grad_step = 000170, loss = 0.002040
grad_step = 000171, loss = 0.001973
grad_step = 000172, loss = 0.002024
grad_step = 000173, loss = 0.001970
grad_step = 000174, loss = 0.002006
grad_step = 000175, loss = 0.001964
grad_step = 000176, loss = 0.001967
grad_step = 000177, loss = 0.001996
grad_step = 000178, loss = 0.001923
grad_step = 000179, loss = 0.001974
grad_step = 000180, loss = 0.001964
grad_step = 000181, loss = 0.001914
grad_step = 000182, loss = 0.001964
grad_step = 000183, loss = 0.001934
grad_step = 000184, loss = 0.001921
grad_step = 000185, loss = 0.001937
grad_step = 000186, loss = 0.001918
grad_step = 000187, loss = 0.001926
grad_step = 000188, loss = 0.001910
grad_step = 000189, loss = 0.001909
grad_step = 000190, loss = 0.001921
grad_step = 000191, loss = 0.001893
grad_step = 000192, loss = 0.001900
grad_step = 000193, loss = 0.001909
grad_step = 000194, loss = 0.001889
grad_step = 000195, loss = 0.001890
grad_step = 000196, loss = 0.001891
grad_step = 000197, loss = 0.001884
grad_step = 000198, loss = 0.001886
grad_step = 000199, loss = 0.001876
grad_step = 000200, loss = 0.001876
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001880
grad_step = 000202, loss = 0.001870
grad_step = 000203, loss = 0.001866
grad_step = 000204, loss = 0.001869
grad_step = 000205, loss = 0.001866
grad_step = 000206, loss = 0.001861
grad_step = 000207, loss = 0.001861
grad_step = 000208, loss = 0.001857
grad_step = 000209, loss = 0.001855
grad_step = 000210, loss = 0.001855
grad_step = 000211, loss = 0.001852
grad_step = 000212, loss = 0.001847
grad_step = 000213, loss = 0.001847
grad_step = 000214, loss = 0.001845
grad_step = 000215, loss = 0.001842
grad_step = 000216, loss = 0.001841
grad_step = 000217, loss = 0.001840
grad_step = 000218, loss = 0.001836
grad_step = 000219, loss = 0.001834
grad_step = 000220, loss = 0.001833
grad_step = 000221, loss = 0.001830
grad_step = 000222, loss = 0.001827
grad_step = 000223, loss = 0.001826
grad_step = 000224, loss = 0.001824
grad_step = 000225, loss = 0.001822
grad_step = 000226, loss = 0.001821
grad_step = 000227, loss = 0.001821
grad_step = 000228, loss = 0.001825
grad_step = 000229, loss = 0.001836
grad_step = 000230, loss = 0.001867
grad_step = 000231, loss = 0.001935
grad_step = 000232, loss = 0.002069
grad_step = 000233, loss = 0.002243
grad_step = 000234, loss = 0.002361
grad_step = 000235, loss = 0.002181
grad_step = 000236, loss = 0.001887
grad_step = 000237, loss = 0.001829
grad_step = 000238, loss = 0.002009
grad_step = 000239, loss = 0.002058
grad_step = 000240, loss = 0.001882
grad_step = 000241, loss = 0.001809
grad_step = 000242, loss = 0.001930
grad_step = 000243, loss = 0.001972
grad_step = 000244, loss = 0.001857
grad_step = 000245, loss = 0.001799
grad_step = 000246, loss = 0.001885
grad_step = 000247, loss = 0.001915
grad_step = 000248, loss = 0.001831
grad_step = 000249, loss = 0.001793
grad_step = 000250, loss = 0.001843
grad_step = 000251, loss = 0.001869
grad_step = 000252, loss = 0.001817
grad_step = 000253, loss = 0.001787
grad_step = 000254, loss = 0.001813
grad_step = 000255, loss = 0.001838
grad_step = 000256, loss = 0.001809
grad_step = 000257, loss = 0.001780
grad_step = 000258, loss = 0.001789
grad_step = 000259, loss = 0.001809
grad_step = 000260, loss = 0.001804
grad_step = 000261, loss = 0.001779
grad_step = 000262, loss = 0.001771
grad_step = 000263, loss = 0.001782
grad_step = 000264, loss = 0.001790
grad_step = 000265, loss = 0.001783
grad_step = 000266, loss = 0.001768
grad_step = 000267, loss = 0.001764
grad_step = 000268, loss = 0.001769
grad_step = 000269, loss = 0.001774
grad_step = 000270, loss = 0.001770
grad_step = 000271, loss = 0.001762
grad_step = 000272, loss = 0.001756
grad_step = 000273, loss = 0.001755
grad_step = 000274, loss = 0.001759
grad_step = 000275, loss = 0.001761
grad_step = 000276, loss = 0.001757
grad_step = 000277, loss = 0.001752
grad_step = 000278, loss = 0.001746
grad_step = 000279, loss = 0.001745
grad_step = 000280, loss = 0.001746
grad_step = 000281, loss = 0.001747
grad_step = 000282, loss = 0.001747
grad_step = 000283, loss = 0.001746
grad_step = 000284, loss = 0.001743
grad_step = 000285, loss = 0.001739
grad_step = 000286, loss = 0.001736
grad_step = 000287, loss = 0.001734
grad_step = 000288, loss = 0.001732
grad_step = 000289, loss = 0.001730
grad_step = 000290, loss = 0.001728
grad_step = 000291, loss = 0.001727
grad_step = 000292, loss = 0.001725
grad_step = 000293, loss = 0.001724
grad_step = 000294, loss = 0.001723
grad_step = 000295, loss = 0.001721
grad_step = 000296, loss = 0.001720
grad_step = 000297, loss = 0.001720
grad_step = 000298, loss = 0.001721
grad_step = 000299, loss = 0.001726
grad_step = 000300, loss = 0.001742
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001782
grad_step = 000302, loss = 0.001859
grad_step = 000303, loss = 0.002010
grad_step = 000304, loss = 0.002157
grad_step = 000305, loss = 0.002282
grad_step = 000306, loss = 0.002182
grad_step = 000307, loss = 0.001929
grad_step = 000308, loss = 0.001869
grad_step = 000309, loss = 0.001945
grad_step = 000310, loss = 0.001927
grad_step = 000311, loss = 0.001835
grad_step = 000312, loss = 0.001894
grad_step = 000313, loss = 0.001902
grad_step = 000314, loss = 0.001769
grad_step = 000315, loss = 0.001780
grad_step = 000316, loss = 0.001863
grad_step = 000317, loss = 0.001823
grad_step = 000318, loss = 0.001712
grad_step = 000319, loss = 0.001774
grad_step = 000320, loss = 0.001845
grad_step = 000321, loss = 0.001741
grad_step = 000322, loss = 0.001712
grad_step = 000323, loss = 0.001772
grad_step = 000324, loss = 0.001770
grad_step = 000325, loss = 0.001708
grad_step = 000326, loss = 0.001716
grad_step = 000327, loss = 0.001756
grad_step = 000328, loss = 0.001714
grad_step = 000329, loss = 0.001696
grad_step = 000330, loss = 0.001725
grad_step = 000331, loss = 0.001723
grad_step = 000332, loss = 0.001691
grad_step = 000333, loss = 0.001690
grad_step = 000334, loss = 0.001713
grad_step = 000335, loss = 0.001699
grad_step = 000336, loss = 0.001682
grad_step = 000337, loss = 0.001691
grad_step = 000338, loss = 0.001694
grad_step = 000339, loss = 0.001682
grad_step = 000340, loss = 0.001676
grad_step = 000341, loss = 0.001684
grad_step = 000342, loss = 0.001685
grad_step = 000343, loss = 0.001673
grad_step = 000344, loss = 0.001670
grad_step = 000345, loss = 0.001675
grad_step = 000346, loss = 0.001673
grad_step = 000347, loss = 0.001667
grad_step = 000348, loss = 0.001665
grad_step = 000349, loss = 0.001668
grad_step = 000350, loss = 0.001668
grad_step = 000351, loss = 0.001662
grad_step = 000352, loss = 0.001660
grad_step = 000353, loss = 0.001661
grad_step = 000354, loss = 0.001660
grad_step = 000355, loss = 0.001656
grad_step = 000356, loss = 0.001654
grad_step = 000357, loss = 0.001654
grad_step = 000358, loss = 0.001654
grad_step = 000359, loss = 0.001652
grad_step = 000360, loss = 0.001649
grad_step = 000361, loss = 0.001648
grad_step = 000362, loss = 0.001648
grad_step = 000363, loss = 0.001648
grad_step = 000364, loss = 0.001647
grad_step = 000365, loss = 0.001646
grad_step = 000366, loss = 0.001648
grad_step = 000367, loss = 0.001651
grad_step = 000368, loss = 0.001656
grad_step = 000369, loss = 0.001666
grad_step = 000370, loss = 0.001686
grad_step = 000371, loss = 0.001721
grad_step = 000372, loss = 0.001778
grad_step = 000373, loss = 0.001857
grad_step = 000374, loss = 0.001928
grad_step = 000375, loss = 0.001953
grad_step = 000376, loss = 0.001885
grad_step = 000377, loss = 0.001752
grad_step = 000378, loss = 0.001649
grad_step = 000379, loss = 0.001644
grad_step = 000380, loss = 0.001710
grad_step = 000381, loss = 0.001767
grad_step = 000382, loss = 0.001762
grad_step = 000383, loss = 0.001696
grad_step = 000384, loss = 0.001639
grad_step = 000385, loss = 0.001634
grad_step = 000386, loss = 0.001672
grad_step = 000387, loss = 0.001706
grad_step = 000388, loss = 0.001702
grad_step = 000389, loss = 0.001669
grad_step = 000390, loss = 0.001631
grad_step = 000391, loss = 0.001621
grad_step = 000392, loss = 0.001639
grad_step = 000393, loss = 0.001659
grad_step = 000394, loss = 0.001663
grad_step = 000395, loss = 0.001645
grad_step = 000396, loss = 0.001622
grad_step = 000397, loss = 0.001612
grad_step = 000398, loss = 0.001619
grad_step = 000399, loss = 0.001631
grad_step = 000400, loss = 0.001639
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001636
grad_step = 000402, loss = 0.001625
grad_step = 000403, loss = 0.001613
grad_step = 000404, loss = 0.001606
grad_step = 000405, loss = 0.001605
grad_step = 000406, loss = 0.001609
grad_step = 000407, loss = 0.001614
grad_step = 000408, loss = 0.001617
grad_step = 000409, loss = 0.001616
grad_step = 000410, loss = 0.001612
grad_step = 000411, loss = 0.001607
grad_step = 000412, loss = 0.001601
grad_step = 000413, loss = 0.001596
grad_step = 000414, loss = 0.001594
grad_step = 000415, loss = 0.001593
grad_step = 000416, loss = 0.001594
grad_step = 000417, loss = 0.001595
grad_step = 000418, loss = 0.001597
grad_step = 000419, loss = 0.001599
grad_step = 000420, loss = 0.001602
grad_step = 000421, loss = 0.001605
grad_step = 000422, loss = 0.001609
grad_step = 000423, loss = 0.001616
grad_step = 000424, loss = 0.001624
grad_step = 000425, loss = 0.001637
grad_step = 000426, loss = 0.001653
grad_step = 000427, loss = 0.001674
grad_step = 000428, loss = 0.001698
grad_step = 000429, loss = 0.001718
grad_step = 000430, loss = 0.001728
grad_step = 000431, loss = 0.001710
grad_step = 000432, loss = 0.001670
grad_step = 000433, loss = 0.001618
grad_step = 000434, loss = 0.001583
grad_step = 000435, loss = 0.001576
grad_step = 000436, loss = 0.001592
grad_step = 000437, loss = 0.001618
grad_step = 000438, loss = 0.001639
grad_step = 000439, loss = 0.001648
grad_step = 000440, loss = 0.001642
grad_step = 000441, loss = 0.001624
grad_step = 000442, loss = 0.001601
grad_step = 000443, loss = 0.001580
grad_step = 000444, loss = 0.001569
grad_step = 000445, loss = 0.001567
grad_step = 000446, loss = 0.001570
grad_step = 000447, loss = 0.001578
grad_step = 000448, loss = 0.001585
grad_step = 000449, loss = 0.001591
grad_step = 000450, loss = 0.001595
grad_step = 000451, loss = 0.001596
grad_step = 000452, loss = 0.001594
grad_step = 000453, loss = 0.001588
grad_step = 000454, loss = 0.001580
grad_step = 000455, loss = 0.001573
grad_step = 000456, loss = 0.001565
grad_step = 000457, loss = 0.001559
grad_step = 000458, loss = 0.001555
grad_step = 000459, loss = 0.001551
grad_step = 000460, loss = 0.001549
grad_step = 000461, loss = 0.001547
grad_step = 000462, loss = 0.001546
grad_step = 000463, loss = 0.001546
grad_step = 000464, loss = 0.001545
grad_step = 000465, loss = 0.001545
grad_step = 000466, loss = 0.001546
grad_step = 000467, loss = 0.001546
grad_step = 000468, loss = 0.001548
grad_step = 000469, loss = 0.001552
grad_step = 000470, loss = 0.001559
grad_step = 000471, loss = 0.001573
grad_step = 000472, loss = 0.001597
grad_step = 000473, loss = 0.001640
grad_step = 000474, loss = 0.001712
grad_step = 000475, loss = 0.001822
grad_step = 000476, loss = 0.001934
grad_step = 000477, loss = 0.002008
grad_step = 000478, loss = 0.001950
grad_step = 000479, loss = 0.001773
grad_step = 000480, loss = 0.001622
grad_step = 000481, loss = 0.001614
grad_step = 000482, loss = 0.001704
grad_step = 000483, loss = 0.001740
grad_step = 000484, loss = 0.001687
grad_step = 000485, loss = 0.001560
grad_step = 000486, loss = 0.001552
grad_step = 000487, loss = 0.001644
grad_step = 000488, loss = 0.001645
grad_step = 000489, loss = 0.001579
grad_step = 000490, loss = 0.001538
grad_step = 000491, loss = 0.001551
grad_step = 000492, loss = 0.001591
grad_step = 000493, loss = 0.001600
grad_step = 000494, loss = 0.001552
grad_step = 000495, loss = 0.001519
grad_step = 000496, loss = 0.001536
grad_step = 000497, loss = 0.001559
grad_step = 000498, loss = 0.001561
grad_step = 000499, loss = 0.001546
grad_step = 000500, loss = 0.001523
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001513
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

  date_run                              2020-05-13 10:13:15.623190
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.221469
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 10:13:15.630692
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.123105
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 10:13:15.637853
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.130712
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 10:13:15.642823
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.870623
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
0   2020-05-13 10:12:42.713725  ...    mean_absolute_error
1   2020-05-13 10:12:42.718077  ...     mean_squared_error
2   2020-05-13 10:12:42.721782  ...  median_absolute_error
3   2020-05-13 10:12:42.725068  ...               r2_score
4   2020-05-13 10:12:51.681847  ...    mean_absolute_error
5   2020-05-13 10:12:51.686002  ...     mean_squared_error
6   2020-05-13 10:12:51.689172  ...  median_absolute_error
7   2020-05-13 10:12:51.692279  ...               r2_score
8   2020-05-13 10:13:15.623190  ...    mean_absolute_error
9   2020-05-13 10:13:15.630692  ...     mean_squared_error
10  2020-05-13 10:13:15.637853  ...  median_absolute_error
11  2020-05-13 10:13:15.642823  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6fc9e0ccf8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  1%|          | 81920/9912422 [00:00<00:12, 818465.11it/s]  6%|▌         | 606208/9912422 [00:00<00:08, 1095857.95it/s] 56%|█████▋    | 5586944/9912422 [00:00<00:02, 1550454.45it/s]9920512it [00:00, 19026465.42it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 208675.58it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  5%|▍         | 81920/1648877 [00:00<00:01, 813616.06it/s] 39%|███▉      | 638976/1648877 [00:00<00:00, 1093663.16it/s]1654784it [00:00, 4343819.13it/s]                            
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 69143.19it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6f7c7c5eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6f7bdf30f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6f7c7c5eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6f7bd4c128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6f79587518> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6f79573c88> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6f7c7c5eb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6f7bd09748> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6f79587518> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f6f7bbc4550> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f80c2037208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=4343d65ec7f2a3debdf31885efb12030f0353aba4f08a7518f4772847d99045b
  Stored in directory: /tmp/pip-ephem-wheel-cache-c5w2vm_r/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f8059e2f860> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3907584/17464789 [=====>........................] - ETA: 0s
11264000/17464789 [==================>...........] - ETA: 0s
14114816/17464789 [=======================>......] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 10:14:42.087615: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 10:14:42.091833: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 10:14:42.091976: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55932a9303c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 10:14:42.091992: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 8.1726 - accuracy: 0.4670
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6896 - accuracy: 0.4985
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6768 - accuracy: 0.4993 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5670 - accuracy: 0.5065
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5562 - accuracy: 0.5072
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5567 - accuracy: 0.5072
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5987 - accuracy: 0.5044
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6187 - accuracy: 0.5031
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6274 - accuracy: 0.5026
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6528 - accuracy: 0.5009
11000/25000 [============>.................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
12000/25000 [=============>................] - ETA: 4s - loss: 7.6641 - accuracy: 0.5002
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6560 - accuracy: 0.5007
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
15000/25000 [=================>............] - ETA: 3s - loss: 7.6462 - accuracy: 0.5013
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6360 - accuracy: 0.5020
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6314 - accuracy: 0.5023
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6274 - accuracy: 0.5026
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6335 - accuracy: 0.5022
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6360 - accuracy: 0.5020
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6287 - accuracy: 0.5025
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6360 - accuracy: 0.5020
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6553 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6615 - accuracy: 0.5003
25000/25000 [==============================] - 10s 387us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 10:14:58.867707
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 10:14:58.867707  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<27:50:55, 8.60kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:01<19:45:39, 12.1kB/s].vector_cache/glove.6B.zip:   0%|          | 123k/862M [00:01<13:56:14, 17.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 238k/862M [00:01<9:49:08, 24.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 754k/862M [00:01<6:52:59, 34.8kB/s].vector_cache/glove.6B.zip:   0%|          | 1.96M/862M [00:01<4:49:03, 49.6kB/s].vector_cache/glove.6B.zip:   1%|          | 5.78M/862M [00:01<3:21:32, 70.8kB/s].vector_cache/glove.6B.zip:   1%|          | 10.1M/862M [00:01<2:20:28, 101kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 14.0M/862M [00:01<1:37:58, 144kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.8M/862M [00:01<1:08:23, 206kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.8M/862M [00:02<47:52, 293kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 27.3M/862M [00:02<33:22, 417kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.1M/862M [00:02<23:21, 593kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.5M/862M [00:02<16:21, 842kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.7M/862M [00:02<11:29, 1.19MB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.4M/862M [00:02<08:07, 1.68MB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.8M/862M [00:02<05:44, 2.36MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.6M/862M [00:02<04:16, 3.16MB/s].vector_cache/glove.6B.zip:   6%|▌         | 53.8M/862M [00:03<03:09, 4.26MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:05<07:25, 1.81MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:05<12:35, 1.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:05<10:38, 1.26MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:05<07:54, 1.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:07<07:54, 1.69MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:07<07:02, 1.90MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:07<05:16, 2.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:09<06:32, 2.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:09<07:20, 1.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<05:42, 2.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:09<04:14, 3.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:11<07:14, 1.83MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:11<06:11, 2.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:11<05:04, 2.61MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.4M/862M [00:11<03:42, 3.55MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:14<14:26, 911kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:14<20:55, 629kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.4M/862M [00:14<17:53, 736kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:14<13:11, 997kB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.5M/862M [00:14<09:29, 1.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:16<09:48, 1.33MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.7M/862M [00:16<08:39, 1.51MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:16<06:31, 2.00MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.4M/862M [00:16<04:46, 2.73MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:18<09:17, 1.40MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:18<10:05, 1.29MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:18<07:57, 1.64MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.8M/862M [00:18<05:48, 2.24MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:20<07:29, 1.73MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.0M/862M [00:20<07:04, 1.83MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:20<05:20, 2.42MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.9M/862M [00:20<03:56, 3.28MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:22<09:44, 1.32MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:22<08:34, 1.50MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:22<06:23, 2.01MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.5M/862M [00:22<04:38, 2.77MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:24<17:05, 750kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.3M/862M [00:24<13:40, 937kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.4M/862M [00:24<09:57, 1.28MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<07:05, 1.80MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:26<1:40:14, 127kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.4M/862M [00:26<1:11:53, 177kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:26<50:39, 251kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:26<35:36, 357kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:29<36:15, 350kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:30<37:54, 334kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:30<29:30, 430kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<21:19, 594kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:30<15:04, 838kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:31<16:05, 784kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:31<11:33, 1.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:31<08:27, 1.49MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:31<06:11, 2.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<16:28, 761kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<13:31, 927kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:33<09:54, 1.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:33<08:00, 1.56MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<08:53, 1.40MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<07:34, 1.64MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<06:11, 2.01MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:35<04:33, 2.73MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<06:29, 1.91MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<06:12, 2.00MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:37<05:12, 2.38MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:37<03:52, 3.19MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<06:00, 2.05MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:39<05:57, 2.07MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:39<04:48, 2.56MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:39<03:35, 3.42MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:41<06:02, 2.03MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:41<05:38, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:41<04:45, 2.58MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:41<03:35, 3.41MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:43<06:02, 2.02MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:43<07:24, 1.65MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:43<06:00, 2.03MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:43<04:24, 2.76MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:43<03:21, 3.62MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:45<22:40, 535kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<17:34, 690kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<13:46, 880kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:45<10:45, 1.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:45<08:09, 1.48MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:45<06:00, 2.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<07:44, 1.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<06:34, 1.83MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:47<05:34, 2.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:47<04:22, 2.75MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:47<03:22, 3.56MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:47<02:39, 4.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<21:52, 548kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<18:05, 663kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<14:20, 835kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<11:30, 1.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:49<08:38, 1.38MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:49<06:23, 1.87MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:49<04:47, 2.49MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<09:30, 1.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:51<07:53, 1.51MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:51<05:51, 2.03MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:51<04:13, 2.80MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<29:27, 402kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<25:58, 456kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<22:31, 526kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<19:56, 594kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:53<18:31, 640kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:53<16:01, 739kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:53<12:32, 944kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:53<09:20, 1.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:53<06:41, 1.76MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:55<09:51, 1.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:55<07:42, 1.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:55<05:45, 2.04MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:55<04:18, 2.73MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<07:30, 1.56MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:57<07:48, 1.50MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:57<06:32, 1.79MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:57<05:04, 2.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:57<03:51, 3.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:57<03:01, 3.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<09:09, 1.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:59<07:41, 1.51MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:59<05:41, 2.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<06:42, 1.72MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<10:15, 1.13MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<11:13, 1.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:01<11:54, 971kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:01<11:23, 1.02MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:01<09:07, 1.27MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:01<06:44, 1.71MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:03<06:34, 1.75MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:03<05:22, 2.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:03<04:03, 2.82MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:03<03:05, 3.71MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<07:33, 1.51MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:05<06:59, 1.63MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:05<05:57, 1.92MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:05<04:31, 2.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<03:20, 3.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:07<10:29, 1.08MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:07<09:05, 1.25MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:07<06:58, 1.63MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:07<05:07, 2.21MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<06:24, 1.76MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<06:01, 1.87MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<04:43, 2.39MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:09<03:30, 3.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<05:57, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<06:00, 1.87MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:11<04:45, 2.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:11<03:32, 3.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<05:41, 1.96MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<05:31, 2.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:13<04:20, 2.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:13<03:18, 3.37MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:13<02:35, 4.29MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<08:59, 1.23MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<08:50, 1.25MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:15<06:57, 1.59MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:15<05:23, 2.05MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:15<04:12, 2.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:15<03:11, 3.45MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<07:53, 1.40MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:17<07:21, 1.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:17<05:35, 1.97MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:17<04:02, 2.71MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<12:06, 904kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:19<09:57, 1.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:19<07:19, 1.49MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<05:16, 2.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:21<13:22, 814kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<10:37, 1.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:21<07:45, 1.40MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:21<05:37, 1.93MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:23<08:00, 1.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<07:35, 1.42MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<06:25, 1.68MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<05:07, 2.10MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:23<03:52, 2.78MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<02:52, 3.74MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<38:10, 281kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:25<28:14, 380kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:25<20:06, 533kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<16:02, 666kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<12:31, 852kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<09:30, 1.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:27<06:49, 1.56MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<07:53, 1.34MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:29<07:00, 1.51MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:29<05:15, 2.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:29<03:47, 2.78MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<21:26, 492kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<16:10, 651kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<11:54, 884kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:31<08:27, 1.24MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<10:44, 974kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<10:14, 1.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:33<07:45, 1.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:33<05:42, 1.83MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<06:24, 1.62MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<07:23, 1.41MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:35<05:45, 1.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<04:20, 2.39MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<05:19, 1.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<06:33, 1.57MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<05:16, 1.95MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:37<03:53, 2.65MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<05:20, 1.92MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<05:07, 2.00MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:39<03:53, 2.63MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<02:51, 3.57MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<13:26, 758kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<10:46, 946kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:41<07:52, 1.29MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<07:26, 1.36MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<06:36, 1.53MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:43<04:54, 2.06MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:43<03:33, 2.83MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<14:01, 717kB/s] .vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<11:08, 901kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:45<08:04, 1.24MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:45<05:49, 1.72MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:47<08:34, 1.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:47<07:07, 1.40MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:47<05:16, 1.89MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<05:46, 1.72MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:49<06:27, 1.54MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:49<04:59, 1.98MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:49<03:41, 2.68MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:51<05:11, 1.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:51<05:49, 1.69MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:51<04:30, 2.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:51<03:26, 2.84MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<04:30, 2.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<05:14, 1.86MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:53<04:08, 2.36MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<03:00, 3.24MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<08:18, 1.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<07:47, 1.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:55<05:54, 1.64MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:55<04:17, 2.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<06:12, 1.55MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<06:19, 1.52MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<04:50, 1.99MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:57<03:37, 2.65MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<04:55, 1.94MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<05:30, 1.74MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:59<04:23, 2.18MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:59<03:22, 2.82MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<04:24, 2.16MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<04:56, 1.92MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:01<03:55, 2.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:01<02:53, 3.26MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<05:26, 1.74MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<06:02, 1.56MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<04:47, 1.97MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:03<03:44, 2.52MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:03<02:54, 3.22MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<02:18, 4.05MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<06:47, 1.38MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<06:53, 1.36MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:05<05:18, 1.76MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:05<04:00, 2.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<03:01, 3.09MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<06:33, 1.42MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<08:49, 1.05MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:07<07:13, 1.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:07<05:18, 1.75MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<05:36, 1.64MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<07:09, 1.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<09:24, 981kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<10:06, 912kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<10:01, 919kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<08:02, 1.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:09<05:56, 1.55MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<04:17, 2.14MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<07:46, 1.18MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:11<06:27, 1.42MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:11<04:56, 1.85MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:11<03:44, 2.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:11<02:51, 3.19MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<06:00, 1.51MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<06:20, 1.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:13<05:05, 1.78MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:13<03:47, 2.39MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:15<04:31, 2.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<04:45, 1.89MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<03:50, 2.34MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:15<02:51, 3.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<04:26, 2.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:17<05:07, 1.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:17<04:09, 2.15MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:17<03:09, 2.82MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:19<03:58, 2.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:19<04:20, 2.04MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:19<03:21, 2.64MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:19<02:33, 3.46MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:21<04:15, 2.06MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:21<04:58, 1.77MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:21<03:59, 2.21MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:21<02:54, 3.01MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:23<05:41, 1.53MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:23<07:02, 1.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:23<05:35, 1.56MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:23<04:15, 2.04MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:23<03:18, 2.64MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:23<02:29, 3.48MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:25<07:27, 1.16MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:25<07:17, 1.19MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:25<05:30, 1.57MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:25<04:07, 2.10MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:27<04:41, 1.83MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:27<05:13, 1.64MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:27<04:08, 2.07MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:27<03:00, 2.85MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:29<06:05, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:29<05:43, 1.49MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:29<04:20, 1.96MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:29<03:13, 2.63MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:31<04:34, 1.85MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:31<05:06, 1.66MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:31<04:07, 2.05MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:31<03:06, 2.71MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:33<03:57, 2.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:33<04:13, 1.98MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:33<03:16, 2.56MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:33<02:28, 3.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:35<04:05, 2.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:35<04:44, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:35<03:42, 2.24MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:35<02:46, 2.99MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:37<04:04, 2.03MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:37<04:15, 1.94MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:37<03:17, 2.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:37<02:27, 3.33MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:39<05:09, 1.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:39<11:38, 703kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:39<10:04, 812kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:39<07:32, 1.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:40<05:31, 1.48MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:40<03:57, 2.05MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:41<19:57, 407kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:41<14:52, 545kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:41<10:39, 760kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:41<07:31, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:43<14:36, 551kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:43<11:54, 676kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:43<08:41, 924kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:43<06:09, 1.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:45<08:23, 951kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:45<07:13, 1.10MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:45<05:25, 1.47MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:45<03:56, 2.01MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:47<05:01, 1.57MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:47<05:13, 1.51MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:47<03:58, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:47<03:00, 2.61MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:49<03:52, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:49<04:24, 1.78MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:49<03:26, 2.28MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:49<02:35, 3.00MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:51<03:41, 2.10MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:51<04:21, 1.79MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:51<03:28, 2.23MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:51<02:33, 3.03MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:53<04:23, 1.75MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:53<04:39, 1.65MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:53<03:36, 2.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:53<02:45, 2.78MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:55<03:36, 2.12MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:55<04:09, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:55<03:15, 2.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:55<02:28, 3.07MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:57<03:32, 2.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:57<04:07, 1.84MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:57<03:14, 2.33MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:57<02:27, 3.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:59<03:29, 2.15MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:59<04:04, 1.84MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:59<03:12, 2.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:59<02:25, 3.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:01<03:26, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:01<04:05, 1.81MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:01<03:17, 2.25MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:01<02:27, 3.00MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:01<01:51, 3.95MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:03<09:58, 737kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:03<08:36, 854kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:03<06:26, 1.14MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:03<04:39, 1.57MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:05<05:08, 1.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:05<04:49, 1.51MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:05<03:41, 1.97MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:05<02:42, 2.68MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:07<04:30, 1.60MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:07<04:47, 1.51MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:07<03:41, 1.95MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:07<02:45, 2.60MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:09<03:36, 1.98MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:09<04:07, 1.73MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:09<03:16, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:09<02:25, 2.92MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:11<03:37, 1.95MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:11<04:07, 1.72MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:11<03:13, 2.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:11<02:27, 2.87MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:13<03:14, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:13<03:37, 1.93MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:13<03:19, 2.10MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:13<02:32, 2.75MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:13<01:56, 3.59MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:15<03:51, 1.80MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:15<04:15, 1.63MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:15<03:27, 2.01MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:15<02:34, 2.68MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:15<01:54, 3.59MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:17<07:01, 979kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:17<06:26, 1.07MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:17<04:56, 1.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<03:37, 1.88MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:19<03:58, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:19<04:18, 1.58MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:19<03:20, 2.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:19<02:28, 2.74MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:21<03:31, 1.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:21<03:55, 1.72MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:21<03:05, 2.18MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:21<02:20, 2.86MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:23<03:04, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:23<03:13, 2.07MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:23<02:30, 2.65MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<01:50, 3.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:25<06:30, 1.01MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:25<05:55, 1.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:25<04:26, 1.48MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<03:15, 2.01MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:27<03:56, 1.65MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:27<04:06, 1.59MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:27<03:11, 2.04MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:27<02:20, 2.76MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:29<03:30, 1.84MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:29<03:47, 1.70MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:29<02:55, 2.20MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:29<02:10, 2.96MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:31<03:25, 1.86MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:31<03:43, 1.72MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:31<02:54, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:31<02:07, 2.97MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:33<03:37, 1.74MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:33<03:27, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:33<02:36, 2.42MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:33<01:54, 3.28MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:35<04:55, 1.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:35<04:37, 1.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:35<03:32, 1.76MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:36<03:28, 1.78MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:37<03:39, 1.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:37<02:50, 2.17MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:38<02:59, 2.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:39<03:15, 1.88MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:39<02:34, 2.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:39<01:53, 3.20MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:40<03:34, 1.69MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:41<03:37, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:41<02:47, 2.17MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:41<02:04, 2.89MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:42<03:06, 1.92MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:43<03:20, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:43<02:34, 2.31MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:43<01:55, 3.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:44<03:01, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:45<03:14, 1.82MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:45<02:32, 2.31MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<01:50, 3.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:46<05:06, 1.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:46<04:41, 1.24MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:47<03:31, 1.65MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:47<02:32, 2.27MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:48<04:01, 1.44MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:48<03:54, 1.48MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:49<03:00, 1.91MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:49<02:08, 2.66MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:50<25:16, 226kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:50<18:48, 303kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:51<13:22, 425kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:51<09:24, 601kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:52<08:34, 657kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:52<07:05, 795kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:53<05:13, 1.08MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:54<04:32, 1.23MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:54<04:14, 1.31MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:54<03:13, 1.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:56<03:08, 1.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:56<03:17, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:56<02:33, 2.14MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<01:51, 2.93MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:58<04:00, 1.36MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:58<03:52, 1.40MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:58<02:58, 1.82MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:58<02:07, 2.53MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:00<05:16, 1.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:00<05:55, 903kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:00<04:34, 1.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:01<03:18, 1.61MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:02<03:26, 1.54MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:02<03:27, 1.53MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:02<02:38, 2.00MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:03<01:53, 2.77MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:04<08:59, 580kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:04<07:17, 716kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:04<05:18, 980kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:05<03:44, 1.38MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:06<08:08, 632kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:06<06:40, 770kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:06<04:54, 1.04MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:08<04:14, 1.20MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:08<03:56, 1.29MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:08<02:59, 1.69MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:10<02:54, 1.73MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:10<03:00, 1.66MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:10<02:20, 2.14MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:12<02:26, 2.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:12<02:38, 1.87MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:12<02:03, 2.40MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<01:28, 3.30MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:14<05:05, 957kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:14<04:29, 1.08MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:14<03:22, 1.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:16<03:07, 1.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:16<03:08, 1.53MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:16<02:23, 2.00MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:16<01:45, 2.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:18<02:45, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:18<04:15, 1.11MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:18<03:37, 1.31MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:18<02:45, 1.71MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:18<02:00, 2.34MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:20<02:43, 1.71MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:20<02:50, 1.64MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:20<02:12, 2.11MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:22<02:17, 2.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:22<02:28, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:22<01:55, 2.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:22<01:30, 3.01MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<01:07, 4.04MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:24<05:44, 790kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:24<04:44, 953kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:24<03:47, 1.19MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:24<02:43, 1.65MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:26<02:57, 1.51MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:26<02:56, 1.51MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:26<02:16, 1.96MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:26<01:36, 2.72MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:28<1:26:31, 50.8kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:28<1:01:23, 71.5kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:28<43:01, 102kB/s]   .vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:28<30:00, 145kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:30<22:20, 193kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:30<16:26, 263kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:30<11:39, 369kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:30<08:14, 521kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:32<06:46, 628kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:32<05:27, 778kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:32<04:05, 1.04MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:32<02:55, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:34<03:04, 1.36MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:34<03:01, 1.39MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:34<02:18, 1.80MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:34<01:40, 2.47MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:36<02:49, 1.46MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:36<03:33, 1.15MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:36<02:50, 1.45MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:36<02:05, 1.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:36<01:30, 2.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:38<04:56, 818kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:38<04:13, 957kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:38<03:08, 1.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:38<02:13, 1.80MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:40<06:56, 572kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:40<05:37, 707kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:40<04:06, 963kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:40<02:53, 1.35MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:42<05:52, 665kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:42<04:50, 808kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:42<03:45, 1.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:42<02:41, 1.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:44<02:43, 1.41MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:44<02:24, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:44<01:48, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:46<01:58, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:46<02:04, 1.82MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:46<01:35, 2.36MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:46<01:09, 3.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:48<02:33, 1.45MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:48<02:28, 1.50MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:48<01:53, 1.95MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:50<01:55, 1.89MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:50<01:59, 1.83MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:50<01:31, 2.36MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<01:06, 3.22MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:52<02:40, 1.33MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:52<02:31, 1.41MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:52<01:55, 1.84MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:54<01:55, 1.82MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:54<01:58, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:54<01:32, 2.27MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:54<01:05, 3.13MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:56<06:53, 497kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:56<06:02, 567kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:56<04:31, 755kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<03:13, 1.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:58<03:06, 1.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:58<02:46, 1.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:58<02:05, 1.60MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:58<01:28, 2.23MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:00<08:26, 390kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:00<06:30, 506kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:00<04:40, 699kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:02<03:46, 854kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:02<03:12, 1.01MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:02<02:22, 1.35MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:03<02:10, 1.45MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:04<02:05, 1.51MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:04<01:34, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:04<01:09, 2.68MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:05<01:41, 1.82MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:06<01:44, 1.77MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:06<01:21, 2.27MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:07<01:26, 2.09MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:08<01:33, 1.93MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:08<01:13, 2.45MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<00:52, 3.37MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:09<04:42, 626kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:09<03:48, 774kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:10<02:45, 1.06MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<01:57, 1.48MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:11<03:08, 918kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:11<02:43, 1.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:12<02:01, 1.41MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:12<01:25, 1.98MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:13<10:07, 278kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:13<07:35, 370kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:14<05:23, 518kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:14<03:47, 729kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:15<03:29, 785kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:15<02:56, 932kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:16<02:10, 1.26MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:16<01:31, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:17<03:35, 743kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:17<02:54, 915kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:17<02:11, 1.22MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:18<01:34, 1.68MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:19<01:45, 1.47MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:19<01:42, 1.53MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:19<01:18, 1.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:20<00:55, 2.74MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:21<02:45, 917kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:21<02:23, 1.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:21<01:45, 1.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:22<01:17, 1.94MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:23<01:32, 1.60MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:23<01:31, 1.61MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:23<01:10, 2.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:24<00:50, 2.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:25<02:02, 1.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:25<02:17, 1.05MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:26<01:49, 1.31MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:26<01:18, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:26<00:57, 2.45MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:27<02:14, 1.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:27<02:00, 1.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:27<01:29, 1.54MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:28<01:05, 2.10MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:29<01:20, 1.68MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:29<01:21, 1.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:29<01:02, 2.15MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:30<00:45, 2.90MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:31<01:18, 1.68MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:31<01:18, 1.67MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:31<01:00, 2.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:32<00:44, 2.91MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:33<01:19, 1.61MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:33<01:18, 1.62MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:33<01:01, 2.05MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:33<00:45, 2.75MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:35<01:06, 1.86MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:35<01:29, 1.37MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:35<01:13, 1.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:36<00:54, 2.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:36<00:40, 2.98MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:37<01:21, 1.46MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:37<01:19, 1.49MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:37<01:01, 1.93MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:38<00:43, 2.67MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:40<02:12, 865kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:40<02:32, 748kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:40<02:04, 916kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:40<01:33, 1.22MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:40<01:06, 1.68MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:42<01:12, 1.53MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:42<01:12, 1.51MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:42<00:59, 1.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:42<00:44, 2.43MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:44<00:50, 2.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:44<00:53, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:44<00:40, 2.60MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:44<00:30, 3.42MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:45<00:51, 1.97MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:46<00:52, 1.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:46<00:44, 2.29MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:46<00:34, 2.94MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:46<00:24, 3.98MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:47<02:08, 762kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:48<01:51, 880kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:48<01:25, 1.14MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:48<01:02, 1.55MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:49<01:00, 1.56MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:50<00:54, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:50<00:46, 2.00MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:50<00:34, 2.64MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:51<00:41, 2.16MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:52<00:40, 2.21MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:52<00:30, 2.86MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:52<00:22, 3.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:53<00:53, 1.59MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:54<01:06, 1.29MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:54<00:53, 1.58MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:54<00:40, 2.06MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:54<00:29, 2.82MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:55<01:12, 1.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:56<01:04, 1.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:56<00:48, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:56<00:34, 2.27MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:57<00:45, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:57<00:41, 1.85MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:58<00:31, 2.40MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:58<00:22, 3.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:59<00:55, 1.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:59<01:03, 1.16MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:00<00:51, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:00<00:38, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:00<00:26, 2.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:01<01:48, 634kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:01<01:27, 786kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:02<01:04, 1.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:02<00:45, 1.46MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:03<00:46, 1.39MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:03<00:40, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:04<00:29, 2.12MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:04<00:21, 2.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:05<00:50, 1.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:05<00:51, 1.18MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:05<00:39, 1.51MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:06<00:29, 2.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:07<00:30, 1.86MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:07<00:30, 1.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:08<00:23, 2.36MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:08<00:16, 3.26MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:09<04:06, 212kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:09<03:00, 289kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:09<02:06, 405kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:11<01:31, 526kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:11<01:12, 665kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:11<00:51, 920kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:12<00:34, 1.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:13<01:02, 706kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:13<00:49, 886kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:13<00:38, 1.14MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:13<00:26, 1.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:15<00:28, 1.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:15<00:26, 1.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:15<00:19, 1.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:15<00:13, 2.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:17<00:23, 1.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:17<00:21, 1.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:17<00:15, 2.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:17<00:11, 2.94MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:19<00:18, 1.73MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:19<00:23, 1.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:20<00:18, 1.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:20<00:13, 2.23MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:21<00:13, 2.02MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:21<00:13, 1.96MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:22<00:10, 2.51MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:24<00:11, 2.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:24<00:24, 961kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:24<00:19, 1.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:24<00:15, 1.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:24<00:11, 1.95MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:26<00:10, 1.78MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:26<00:10, 1.77MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:26<00:07, 2.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:26<00:04, 3.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:28<00:49, 303kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:28<00:36, 408kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:28<00:26, 553kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:28<00:17, 766kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:28<00:11, 1.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:29<00:12, 883kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:30<00:10, 1.04MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:30<00:07, 1.38MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:30<00:04, 1.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:31<00:04, 1.58MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:32<00:03, 1.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:32<00:02, 2.32MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:32<00:01, 3.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:33<00:02, 1.26MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:34<00:02, 1.13MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:34<00:01, 1.38MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:34<00:00, 1.83MB/s].vector_cache/glove.6B.zip: 862MB [06:34, 2.19MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 854/400000 [00:00<00:46, 8524.30it/s]  0%|          | 1654/400000 [00:00<00:47, 8357.88it/s]  1%|          | 2553/400000 [00:00<00:46, 8536.82it/s]  1%|          | 3399/400000 [00:00<00:46, 8510.52it/s]  1%|          | 4193/400000 [00:00<00:47, 8329.58it/s]  1%|▏         | 5047/400000 [00:00<00:47, 8390.34it/s]  1%|▏         | 5958/400000 [00:00<00:45, 8589.96it/s]  2%|▏         | 6830/400000 [00:00<00:45, 8628.31it/s]  2%|▏         | 7734/400000 [00:00<00:44, 8747.59it/s]  2%|▏         | 8576/400000 [00:01<00:45, 8623.49it/s]  2%|▏         | 9416/400000 [00:01<00:47, 8257.85it/s]  3%|▎         | 10238/400000 [00:01<00:47, 8245.77it/s]  3%|▎         | 11071/400000 [00:01<00:47, 8269.24it/s]  3%|▎         | 11973/400000 [00:01<00:45, 8480.30it/s]  3%|▎         | 12851/400000 [00:01<00:45, 8567.64it/s]  3%|▎         | 13706/400000 [00:01<00:45, 8547.34it/s]  4%|▎         | 14606/400000 [00:01<00:44, 8676.96it/s]  4%|▍         | 15474/400000 [00:01<00:45, 8464.68it/s]  4%|▍         | 16374/400000 [00:01<00:44, 8618.39it/s]  4%|▍         | 17284/400000 [00:02<00:43, 8754.77it/s]  5%|▍         | 18161/400000 [00:02<00:44, 8528.70it/s]  5%|▍         | 19031/400000 [00:02<00:44, 8577.76it/s]  5%|▍         | 19961/400000 [00:02<00:43, 8780.15it/s]  5%|▌         | 20911/400000 [00:02<00:42, 8984.26it/s]  5%|▌         | 21829/400000 [00:02<00:41, 9041.29it/s]  6%|▌         | 22736/400000 [00:02<00:42, 8825.18it/s]  6%|▌         | 23622/400000 [00:02<00:43, 8671.57it/s]  6%|▌         | 24565/400000 [00:02<00:42, 8883.72it/s]  6%|▋         | 25457/400000 [00:02<00:42, 8827.08it/s]  7%|▋         | 26342/400000 [00:03<00:42, 8826.93it/s]  7%|▋         | 27227/400000 [00:03<00:42, 8708.30it/s]  7%|▋         | 28137/400000 [00:03<00:42, 8821.88it/s]  7%|▋         | 29021/400000 [00:03<00:42, 8787.92it/s]  7%|▋         | 29901/400000 [00:03<00:42, 8628.74it/s]  8%|▊         | 30766/400000 [00:03<00:43, 8554.66it/s]  8%|▊         | 31623/400000 [00:03<00:43, 8399.84it/s]  8%|▊         | 32465/400000 [00:03<00:44, 8340.10it/s]  8%|▊         | 33368/400000 [00:03<00:42, 8535.50it/s]  9%|▊         | 34279/400000 [00:03<00:42, 8699.43it/s]  9%|▉         | 35164/400000 [00:04<00:41, 8742.10it/s]  9%|▉         | 36040/400000 [00:04<00:42, 8612.29it/s]  9%|▉         | 36903/400000 [00:04<00:42, 8603.33it/s]  9%|▉         | 37802/400000 [00:04<00:41, 8713.89it/s] 10%|▉         | 38675/400000 [00:04<00:41, 8705.36it/s] 10%|▉         | 39547/400000 [00:04<00:41, 8676.21it/s] 10%|█         | 40416/400000 [00:04<00:41, 8631.16it/s] 10%|█         | 41287/400000 [00:04<00:41, 8652.58it/s] 11%|█         | 42153/400000 [00:04<00:41, 8536.21it/s] 11%|█         | 43008/400000 [00:04<00:41, 8531.02it/s] 11%|█         | 43865/400000 [00:05<00:41, 8537.70it/s] 11%|█         | 44720/400000 [00:05<00:42, 8405.24it/s] 11%|█▏        | 45563/400000 [00:05<00:42, 8411.54it/s] 12%|█▏        | 46405/400000 [00:05<00:43, 8201.84it/s] 12%|█▏        | 47268/400000 [00:05<00:42, 8324.65it/s] 12%|█▏        | 48138/400000 [00:05<00:41, 8432.74it/s] 12%|█▏        | 48983/400000 [00:05<00:42, 8226.89it/s] 12%|█▏        | 49833/400000 [00:05<00:42, 8306.39it/s] 13%|█▎        | 50666/400000 [00:05<00:42, 8210.51it/s] 13%|█▎        | 51491/400000 [00:06<00:42, 8220.31it/s] 13%|█▎        | 52355/400000 [00:06<00:41, 8340.83it/s] 13%|█▎        | 53202/400000 [00:06<00:41, 8379.08it/s] 14%|█▎        | 54041/400000 [00:06<00:41, 8356.03it/s] 14%|█▎        | 54915/400000 [00:06<00:40, 8466.79it/s] 14%|█▍        | 55793/400000 [00:06<00:40, 8555.04it/s] 14%|█▍        | 56650/400000 [00:06<00:40, 8394.74it/s] 14%|█▍        | 57491/400000 [00:06<00:41, 8301.35it/s] 15%|█▍        | 58375/400000 [00:06<00:40, 8454.88it/s] 15%|█▍        | 59294/400000 [00:06<00:39, 8661.25it/s] 15%|█▌        | 60163/400000 [00:07<00:39, 8549.29it/s] 15%|█▌        | 61020/400000 [00:07<00:40, 8465.01it/s] 15%|█▌        | 61869/400000 [00:07<00:40, 8396.85it/s] 16%|█▌        | 62710/400000 [00:07<00:40, 8348.95it/s] 16%|█▌        | 63585/400000 [00:07<00:39, 8464.35it/s] 16%|█▌        | 64439/400000 [00:07<00:39, 8486.74it/s] 16%|█▋        | 65289/400000 [00:07<00:39, 8434.73it/s] 17%|█▋        | 66134/400000 [00:07<00:40, 8277.73it/s] 17%|█▋        | 66990/400000 [00:07<00:39, 8360.38it/s] 17%|█▋        | 67827/400000 [00:07<00:39, 8331.63it/s] 17%|█▋        | 68661/400000 [00:08<00:39, 8316.66it/s] 17%|█▋        | 69539/400000 [00:08<00:39, 8448.10it/s] 18%|█▊        | 70385/400000 [00:08<00:39, 8340.18it/s] 18%|█▊        | 71220/400000 [00:08<00:39, 8328.69it/s] 18%|█▊        | 72105/400000 [00:08<00:38, 8477.26it/s] 18%|█▊        | 72991/400000 [00:08<00:38, 8587.54it/s] 18%|█▊        | 73851/400000 [00:08<00:38, 8571.14it/s] 19%|█▊        | 74709/400000 [00:08<00:38, 8450.74it/s] 19%|█▉        | 75581/400000 [00:08<00:38, 8528.94it/s] 19%|█▉        | 76435/400000 [00:08<00:38, 8307.95it/s] 19%|█▉        | 77315/400000 [00:09<00:38, 8449.17it/s] 20%|█▉        | 78193/400000 [00:09<00:37, 8544.19it/s] 20%|█▉        | 79049/400000 [00:09<00:37, 8465.72it/s] 20%|█▉        | 79917/400000 [00:09<00:37, 8527.83it/s] 20%|██        | 80828/400000 [00:09<00:36, 8692.34it/s] 20%|██        | 81699/400000 [00:09<00:37, 8592.45it/s] 21%|██        | 82594/400000 [00:09<00:36, 8696.28it/s] 21%|██        | 83465/400000 [00:09<00:37, 8504.78it/s] 21%|██        | 84349/400000 [00:09<00:36, 8602.52it/s] 21%|██▏       | 85244/400000 [00:09<00:36, 8701.90it/s] 22%|██▏       | 86116/400000 [00:10<00:36, 8671.26it/s] 22%|██▏       | 86985/400000 [00:10<00:36, 8670.52it/s] 22%|██▏       | 87853/400000 [00:10<00:36, 8482.54it/s] 22%|██▏       | 88703/400000 [00:10<00:36, 8417.50it/s] 22%|██▏       | 89607/400000 [00:10<00:36, 8593.19it/s] 23%|██▎       | 90468/400000 [00:10<00:36, 8567.43it/s] 23%|██▎       | 91326/400000 [00:10<00:36, 8566.37it/s] 23%|██▎       | 92184/400000 [00:10<00:36, 8331.36it/s] 23%|██▎       | 93020/400000 [00:10<00:36, 8336.76it/s] 23%|██▎       | 93856/400000 [00:11<00:38, 8054.32it/s] 24%|██▎       | 94714/400000 [00:11<00:37, 8203.52it/s] 24%|██▍       | 95604/400000 [00:11<00:36, 8397.56it/s] 24%|██▍       | 96523/400000 [00:11<00:35, 8618.39it/s] 24%|██▍       | 97440/400000 [00:11<00:34, 8776.49it/s] 25%|██▍       | 98325/400000 [00:11<00:34, 8796.91it/s] 25%|██▍       | 99207/400000 [00:11<00:34, 8745.48it/s] 25%|██▌       | 100084/400000 [00:11<00:34, 8667.19it/s] 25%|██▌       | 100953/400000 [00:11<00:34, 8663.95it/s] 25%|██▌       | 101821/400000 [00:11<00:34, 8654.18it/s] 26%|██▌       | 102688/400000 [00:12<00:34, 8656.41it/s] 26%|██▌       | 103571/400000 [00:12<00:34, 8705.94it/s] 26%|██▌       | 104442/400000 [00:12<00:34, 8566.97it/s] 26%|██▋       | 105357/400000 [00:12<00:33, 8732.76it/s] 27%|██▋       | 106320/400000 [00:12<00:32, 8981.58it/s] 27%|██▋       | 107227/400000 [00:12<00:32, 9006.38it/s] 27%|██▋       | 108130/400000 [00:12<00:33, 8704.39it/s] 27%|██▋       | 109004/400000 [00:12<00:34, 8518.94it/s] 27%|██▋       | 109867/400000 [00:12<00:33, 8548.61it/s] 28%|██▊       | 110773/400000 [00:12<00:33, 8694.79it/s] 28%|██▊       | 111645/400000 [00:13<00:33, 8546.86it/s] 28%|██▊       | 112563/400000 [00:13<00:32, 8727.22it/s] 28%|██▊       | 113444/400000 [00:13<00:32, 8745.81it/s] 29%|██▊       | 114321/400000 [00:13<00:33, 8626.90it/s] 29%|██▉       | 115189/400000 [00:13<00:32, 8642.70it/s] 29%|██▉       | 116055/400000 [00:13<00:33, 8600.40it/s] 29%|██▉       | 116916/400000 [00:13<00:32, 8584.87it/s] 29%|██▉       | 117776/400000 [00:13<00:33, 8514.02it/s] 30%|██▉       | 118628/400000 [00:13<00:33, 8487.18it/s] 30%|██▉       | 119484/400000 [00:13<00:32, 8507.37it/s] 30%|███       | 120336/400000 [00:14<00:33, 8443.53it/s] 30%|███       | 121181/400000 [00:14<00:33, 8366.77it/s] 31%|███       | 122077/400000 [00:14<00:32, 8535.29it/s] 31%|███       | 122983/400000 [00:14<00:31, 8685.45it/s] 31%|███       | 123853/400000 [00:14<00:32, 8619.72it/s] 31%|███       | 124717/400000 [00:14<00:32, 8563.20it/s] 31%|███▏      | 125575/400000 [00:14<00:32, 8515.69it/s] 32%|███▏      | 126437/400000 [00:14<00:32, 8544.38it/s] 32%|███▏      | 127319/400000 [00:14<00:31, 8624.22it/s] 32%|███▏      | 128182/400000 [00:14<00:31, 8620.39it/s] 32%|███▏      | 129045/400000 [00:15<00:31, 8563.69it/s] 32%|███▏      | 129902/400000 [00:15<00:31, 8519.88it/s] 33%|███▎      | 130755/400000 [00:15<00:31, 8424.75it/s] 33%|███▎      | 131630/400000 [00:15<00:31, 8517.77it/s] 33%|███▎      | 132558/400000 [00:15<00:30, 8732.09it/s] 33%|███▎      | 133549/400000 [00:15<00:29, 9053.26it/s] 34%|███▎      | 134465/400000 [00:15<00:29, 9083.73it/s] 34%|███▍      | 135377/400000 [00:15<00:29, 8966.41it/s] 34%|███▍      | 136277/400000 [00:15<00:30, 8776.92it/s] 34%|███▍      | 137158/400000 [00:16<00:30, 8658.01it/s] 35%|███▍      | 138026/400000 [00:16<00:31, 8442.94it/s] 35%|███▍      | 138873/400000 [00:16<00:31, 8234.50it/s] 35%|███▍      | 139700/400000 [00:16<00:32, 8079.34it/s] 35%|███▌      | 140581/400000 [00:16<00:31, 8285.48it/s] 35%|███▌      | 141490/400000 [00:16<00:30, 8507.96it/s] 36%|███▌      | 142412/400000 [00:16<00:29, 8709.19it/s] 36%|███▌      | 143287/400000 [00:16<00:29, 8689.99it/s] 36%|███▌      | 144159/400000 [00:16<00:30, 8488.88it/s] 36%|███▋      | 145040/400000 [00:16<00:29, 8581.68it/s] 36%|███▋      | 145901/400000 [00:17<00:29, 8528.95it/s] 37%|███▋      | 146756/400000 [00:17<00:30, 8407.55it/s] 37%|███▋      | 147680/400000 [00:17<00:29, 8639.87it/s] 37%|███▋      | 148547/400000 [00:17<00:29, 8396.42it/s] 37%|███▋      | 149391/400000 [00:17<00:30, 8277.45it/s] 38%|███▊      | 150222/400000 [00:17<00:30, 8223.40it/s] 38%|███▊      | 151081/400000 [00:17<00:29, 8329.84it/s] 38%|███▊      | 151941/400000 [00:17<00:29, 8406.66it/s] 38%|███▊      | 152784/400000 [00:17<00:29, 8360.78it/s] 38%|███▊      | 153622/400000 [00:17<00:29, 8309.52it/s] 39%|███▊      | 154454/400000 [00:18<00:29, 8257.07it/s] 39%|███▉      | 155319/400000 [00:18<00:29, 8369.51it/s] 39%|███▉      | 156157/400000 [00:18<00:29, 8327.21it/s] 39%|███▉      | 156991/400000 [00:18<00:29, 8287.06it/s] 39%|███▉      | 157832/400000 [00:18<00:29, 8320.78it/s] 40%|███▉      | 158665/400000 [00:18<00:29, 8308.63it/s] 40%|███▉      | 159497/400000 [00:18<00:29, 8229.82it/s] 40%|████      | 160346/400000 [00:18<00:28, 8305.70it/s] 40%|████      | 161177/400000 [00:18<00:28, 8237.02it/s] 41%|████      | 162089/400000 [00:18<00:28, 8482.61it/s] 41%|████      | 162940/400000 [00:19<00:27, 8485.03it/s] 41%|████      | 163852/400000 [00:19<00:27, 8665.41it/s] 41%|████      | 164749/400000 [00:19<00:26, 8752.87it/s] 41%|████▏     | 165626/400000 [00:19<00:27, 8581.99it/s] 42%|████▏     | 166487/400000 [00:19<00:27, 8437.60it/s] 42%|████▏     | 167428/400000 [00:19<00:26, 8705.45it/s] 42%|████▏     | 168306/400000 [00:19<00:26, 8727.18it/s] 42%|████▏     | 169182/400000 [00:19<00:26, 8651.65it/s] 43%|████▎     | 170050/400000 [00:19<00:26, 8542.73it/s] 43%|████▎     | 171006/400000 [00:20<00:25, 8823.64it/s] 43%|████▎     | 171892/400000 [00:20<00:26, 8718.71it/s] 43%|████▎     | 172767/400000 [00:20<00:26, 8461.20it/s] 43%|████▎     | 173637/400000 [00:20<00:26, 8530.79it/s] 44%|████▎     | 174493/400000 [00:20<00:27, 8337.57it/s] 44%|████▍     | 175330/400000 [00:20<00:27, 8280.85it/s] 44%|████▍     | 176182/400000 [00:20<00:26, 8350.75it/s] 44%|████▍     | 177019/400000 [00:20<00:26, 8301.71it/s] 44%|████▍     | 177945/400000 [00:20<00:25, 8567.08it/s] 45%|████▍     | 178805/400000 [00:20<00:26, 8456.32it/s] 45%|████▍     | 179654/400000 [00:21<00:26, 8429.67it/s] 45%|████▌     | 180550/400000 [00:21<00:25, 8580.82it/s] 45%|████▌     | 181410/400000 [00:21<00:25, 8583.28it/s] 46%|████▌     | 182270/400000 [00:21<00:25, 8545.52it/s] 46%|████▌     | 183133/400000 [00:21<00:25, 8568.91it/s] 46%|████▌     | 184010/400000 [00:21<00:25, 8627.67it/s] 46%|████▌     | 184927/400000 [00:21<00:24, 8782.09it/s] 46%|████▋     | 185807/400000 [00:21<00:24, 8787.13it/s] 47%|████▋     | 186710/400000 [00:21<00:24, 8856.51it/s] 47%|████▋     | 187597/400000 [00:21<00:25, 8440.47it/s] 47%|████▋     | 188446/400000 [00:22<00:25, 8288.92it/s] 47%|████▋     | 189318/400000 [00:22<00:25, 8412.86it/s] 48%|████▊     | 190251/400000 [00:22<00:24, 8665.37it/s] 48%|████▊     | 191122/400000 [00:22<00:24, 8584.69it/s] 48%|████▊     | 192025/400000 [00:22<00:23, 8713.30it/s] 48%|████▊     | 192901/400000 [00:22<00:23, 8724.71it/s] 48%|████▊     | 193776/400000 [00:22<00:23, 8620.28it/s] 49%|████▊     | 194752/400000 [00:22<00:22, 8931.99it/s] 49%|████▉     | 195650/400000 [00:22<00:23, 8743.80it/s] 49%|████▉     | 196529/400000 [00:22<00:23, 8622.66it/s] 49%|████▉     | 197395/400000 [00:23<00:23, 8460.10it/s] 50%|████▉     | 198244/400000 [00:23<00:24, 8378.61it/s] 50%|████▉     | 199164/400000 [00:23<00:23, 8609.15it/s] 50%|█████     | 200055/400000 [00:23<00:22, 8695.54it/s] 50%|█████     | 200927/400000 [00:23<00:23, 8562.50it/s] 50%|█████     | 201812/400000 [00:23<00:22, 8645.70it/s] 51%|█████     | 202679/400000 [00:23<00:22, 8623.74it/s] 51%|█████     | 203578/400000 [00:23<00:22, 8729.63it/s] 51%|█████     | 204489/400000 [00:23<00:22, 8838.67it/s] 51%|█████▏    | 205374/400000 [00:24<00:22, 8778.44it/s] 52%|█████▏    | 206253/400000 [00:24<00:22, 8751.33it/s] 52%|█████▏    | 207151/400000 [00:24<00:21, 8816.95it/s] 52%|█████▏    | 208035/400000 [00:24<00:21, 8822.50it/s] 52%|█████▏    | 208930/400000 [00:24<00:21, 8857.91it/s] 52%|█████▏    | 209817/400000 [00:24<00:22, 8593.63it/s] 53%|█████▎    | 210734/400000 [00:24<00:21, 8758.03it/s] 53%|█████▎    | 211635/400000 [00:24<00:21, 8829.90it/s] 53%|█████▎    | 212520/400000 [00:24<00:21, 8700.41it/s] 53%|█████▎    | 213433/400000 [00:24<00:21, 8824.71it/s] 54%|█████▎    | 214317/400000 [00:25<00:21, 8654.43it/s] 54%|█████▍    | 215190/400000 [00:25<00:21, 8675.50it/s] 54%|█████▍    | 216059/400000 [00:25<00:21, 8673.95it/s] 54%|█████▍    | 216928/400000 [00:25<00:21, 8653.33it/s] 54%|█████▍    | 217804/400000 [00:25<00:20, 8684.02it/s] 55%|█████▍    | 218673/400000 [00:25<00:21, 8492.50it/s] 55%|█████▍    | 219541/400000 [00:25<00:21, 8546.37it/s] 55%|█████▌    | 220455/400000 [00:25<00:20, 8714.66it/s] 55%|█████▌    | 221346/400000 [00:25<00:20, 8772.10it/s] 56%|█████▌    | 222225/400000 [00:25<00:20, 8724.02it/s] 56%|█████▌    | 223099/400000 [00:26<00:20, 8424.83it/s] 56%|█████▌    | 223945/400000 [00:26<00:21, 8326.68it/s] 56%|█████▌    | 224780/400000 [00:26<00:21, 8187.31it/s] 56%|█████▋    | 225693/400000 [00:26<00:20, 8447.34it/s] 57%|█████▋    | 226561/400000 [00:26<00:20, 8513.50it/s] 57%|█████▋    | 227415/400000 [00:26<00:20, 8463.34it/s] 57%|█████▋    | 228317/400000 [00:26<00:19, 8621.62it/s] 57%|█████▋    | 229182/400000 [00:26<00:20, 8501.53it/s] 58%|█████▊    | 230034/400000 [00:26<00:20, 8454.78it/s] 58%|█████▊    | 230920/400000 [00:26<00:19, 8570.54it/s] 58%|█████▊    | 231779/400000 [00:27<00:19, 8447.99it/s] 58%|█████▊    | 232670/400000 [00:27<00:19, 8578.97it/s] 58%|█████▊    | 233578/400000 [00:27<00:19, 8722.48it/s] 59%|█████▊    | 234452/400000 [00:27<00:19, 8589.85it/s] 59%|█████▉    | 235314/400000 [00:27<00:19, 8597.23it/s] 59%|█████▉    | 236175/400000 [00:27<00:19, 8412.79it/s] 59%|█████▉    | 237018/400000 [00:27<00:19, 8210.19it/s] 59%|█████▉    | 237842/400000 [00:27<00:19, 8192.18it/s] 60%|█████▉    | 238688/400000 [00:27<00:19, 8269.60it/s] 60%|█████▉    | 239551/400000 [00:28<00:19, 8372.62it/s] 60%|██████    | 240390/400000 [00:28<00:19, 8277.31it/s] 60%|██████    | 241219/400000 [00:28<00:19, 8255.35it/s] 61%|██████    | 242088/400000 [00:28<00:18, 8378.17it/s] 61%|██████    | 242935/400000 [00:28<00:18, 8403.11it/s] 61%|██████    | 243789/400000 [00:28<00:18, 8439.07it/s] 61%|██████    | 244634/400000 [00:28<00:18, 8301.29it/s] 61%|██████▏   | 245495/400000 [00:28<00:18, 8390.36it/s] 62%|██████▏   | 246364/400000 [00:28<00:18, 8477.86it/s] 62%|██████▏   | 247221/400000 [00:28<00:17, 8503.95it/s] 62%|██████▏   | 248154/400000 [00:29<00:17, 8732.92it/s] 62%|██████▏   | 249030/400000 [00:29<00:17, 8723.11it/s] 62%|██████▏   | 249927/400000 [00:29<00:17, 8795.30it/s] 63%|██████▎   | 250840/400000 [00:29<00:16, 8890.82it/s] 63%|██████▎   | 251731/400000 [00:29<00:16, 8848.39it/s] 63%|██████▎   | 252743/400000 [00:29<00:16, 9194.37it/s] 63%|██████▎   | 253667/400000 [00:29<00:16, 9105.75it/s] 64%|██████▎   | 254590/400000 [00:29<00:15, 9140.93it/s] 64%|██████▍   | 255507/400000 [00:29<00:15, 9098.63it/s] 64%|██████▍   | 256419/400000 [00:29<00:15, 9026.03it/s] 64%|██████▍   | 257343/400000 [00:30<00:15, 9086.92it/s] 65%|██████▍   | 258253/400000 [00:30<00:15, 9034.06it/s] 65%|██████▍   | 259158/400000 [00:30<00:15, 8893.22it/s] 65%|██████▌   | 260056/400000 [00:30<00:15, 8918.52it/s] 65%|██████▌   | 260967/400000 [00:30<00:15, 8972.85it/s] 65%|██████▌   | 261914/400000 [00:30<00:15, 9116.18it/s] 66%|██████▌   | 262827/400000 [00:30<00:15, 8999.92it/s] 66%|██████▌   | 263796/400000 [00:30<00:14, 9195.61it/s] 66%|██████▌   | 264718/400000 [00:30<00:14, 9162.00it/s] 66%|██████▋   | 265682/400000 [00:30<00:14, 9297.23it/s] 67%|██████▋   | 266614/400000 [00:31<00:15, 8853.03it/s] 67%|██████▋   | 267505/400000 [00:31<00:15, 8677.20it/s] 67%|██████▋   | 268411/400000 [00:31<00:14, 8788.22it/s] 67%|██████▋   | 269347/400000 [00:31<00:14, 8951.97it/s] 68%|██████▊   | 270303/400000 [00:31<00:14, 9124.21it/s] 68%|██████▊   | 271219/400000 [00:31<00:15, 8574.91it/s] 68%|██████▊   | 272107/400000 [00:31<00:14, 8662.97it/s] 68%|██████▊   | 273033/400000 [00:31<00:14, 8826.48it/s] 68%|██████▊   | 273921/400000 [00:31<00:14, 8704.92it/s] 69%|██████▊   | 274796/400000 [00:32<00:14, 8568.68it/s] 69%|██████▉   | 275659/400000 [00:32<00:14, 8585.58it/s] 69%|██████▉   | 276520/400000 [00:32<00:14, 8396.64it/s] 69%|██████▉   | 277424/400000 [00:32<00:14, 8577.97it/s] 70%|██████▉   | 278309/400000 [00:32<00:14, 8655.51it/s] 70%|██████▉   | 279284/400000 [00:32<00:13, 8956.56it/s] 70%|███████   | 280184/400000 [00:32<00:13, 8957.21it/s] 70%|███████   | 281083/400000 [00:32<00:13, 8632.22it/s] 71%|███████   | 282005/400000 [00:32<00:13, 8799.98it/s] 71%|███████   | 282889/400000 [00:32<00:13, 8681.55it/s] 71%|███████   | 283795/400000 [00:33<00:13, 8788.94it/s] 71%|███████   | 284677/400000 [00:33<00:13, 8715.15it/s] 71%|███████▏  | 285551/400000 [00:33<00:13, 8667.76it/s] 72%|███████▏  | 286420/400000 [00:33<00:13, 8538.02it/s] 72%|███████▏  | 287298/400000 [00:33<00:13, 8608.65it/s] 72%|███████▏  | 288160/400000 [00:33<00:13, 8557.07it/s] 72%|███████▏  | 289053/400000 [00:33<00:12, 8664.44it/s] 73%|███████▎  | 290016/400000 [00:33<00:12, 8930.54it/s] 73%|███████▎  | 290931/400000 [00:33<00:12, 8993.09it/s] 73%|███████▎  | 291863/400000 [00:33<00:11, 9086.88it/s] 73%|███████▎  | 292774/400000 [00:34<00:12, 8809.81it/s] 73%|███████▎  | 293658/400000 [00:34<00:12, 8495.66it/s] 74%|███████▎  | 294570/400000 [00:34<00:12, 8672.19it/s] 74%|███████▍  | 295451/400000 [00:34<00:11, 8713.02it/s] 74%|███████▍  | 296326/400000 [00:34<00:11, 8663.84it/s] 74%|███████▍  | 297253/400000 [00:34<00:11, 8836.61it/s] 75%|███████▍  | 298139/400000 [00:34<00:11, 8702.22it/s] 75%|███████▍  | 299038/400000 [00:34<00:11, 8786.54it/s] 75%|███████▍  | 299942/400000 [00:34<00:11, 8860.92it/s] 75%|███████▌  | 300895/400000 [00:34<00:10, 9050.31it/s] 75%|███████▌  | 301802/400000 [00:35<00:11, 8897.26it/s] 76%|███████▌  | 302694/400000 [00:35<00:11, 8722.84it/s] 76%|███████▌  | 303569/400000 [00:35<00:11, 8626.91it/s] 76%|███████▌  | 304434/400000 [00:35<00:11, 8586.62it/s] 76%|███████▋  | 305294/400000 [00:35<00:11, 8559.51it/s] 77%|███████▋  | 306185/400000 [00:35<00:10, 8660.29it/s] 77%|███████▋  | 307052/400000 [00:35<00:10, 8542.91it/s] 77%|███████▋  | 307934/400000 [00:35<00:10, 8621.97it/s] 77%|███████▋  | 308798/400000 [00:35<00:10, 8436.66it/s] 77%|███████▋  | 309644/400000 [00:36<00:10, 8419.11it/s] 78%|███████▊  | 310568/400000 [00:36<00:10, 8648.38it/s] 78%|███████▊  | 311436/400000 [00:36<00:10, 8627.84it/s] 78%|███████▊  | 312318/400000 [00:36<00:10, 8684.20it/s] 78%|███████▊  | 313188/400000 [00:36<00:10, 8584.27it/s] 79%|███████▊  | 314048/400000 [00:36<00:10, 8472.59it/s] 79%|███████▊  | 314908/400000 [00:36<00:10, 8507.84it/s] 79%|███████▉  | 315760/400000 [00:36<00:10, 8345.09it/s] 79%|███████▉  | 316613/400000 [00:36<00:09, 8397.78it/s] 79%|███████▉  | 317490/400000 [00:36<00:09, 8503.75it/s] 80%|███████▉  | 318421/400000 [00:37<00:09, 8729.60it/s] 80%|███████▉  | 319297/400000 [00:37<00:09, 8507.04it/s] 80%|████████  | 320151/400000 [00:37<00:09, 8224.89it/s] 80%|████████  | 320978/400000 [00:37<00:09, 8195.86it/s] 80%|████████  | 321848/400000 [00:37<00:09, 8339.12it/s] 81%|████████  | 322719/400000 [00:37<00:09, 8445.92it/s] 81%|████████  | 323566/400000 [00:37<00:09, 8323.29it/s] 81%|████████  | 324401/400000 [00:37<00:09, 8134.81it/s] 81%|████████▏ | 325217/400000 [00:37<00:09, 8036.26it/s] 82%|████████▏ | 326023/400000 [00:37<00:09, 7880.13it/s] 82%|████████▏ | 326858/400000 [00:38<00:09, 8013.71it/s] 82%|████████▏ | 327711/400000 [00:38<00:08, 8158.74it/s] 82%|████████▏ | 328572/400000 [00:38<00:08, 8287.86it/s] 82%|████████▏ | 329403/400000 [00:38<00:08, 8207.83it/s] 83%|████████▎ | 330226/400000 [00:38<00:08, 8149.48it/s] 83%|████████▎ | 331043/400000 [00:38<00:08, 7779.71it/s] 83%|████████▎ | 331842/400000 [00:38<00:08, 7839.11it/s] 83%|████████▎ | 332630/400000 [00:38<00:08, 7782.80it/s] 83%|████████▎ | 333411/400000 [00:38<00:08, 7743.25it/s] 84%|████████▎ | 334256/400000 [00:38<00:08, 7941.63it/s] 84%|████████▍ | 335139/400000 [00:39<00:07, 8188.16it/s] 84%|████████▍ | 335998/400000 [00:39<00:07, 8302.52it/s] 84%|████████▍ | 336832/400000 [00:39<00:07, 8309.65it/s] 84%|████████▍ | 337700/400000 [00:39<00:07, 8415.16it/s] 85%|████████▍ | 338634/400000 [00:39<00:07, 8672.17it/s] 85%|████████▍ | 339505/400000 [00:39<00:06, 8657.82it/s] 85%|████████▌ | 340386/400000 [00:39<00:06, 8701.84it/s] 85%|████████▌ | 341258/400000 [00:39<00:06, 8672.37it/s] 86%|████████▌ | 342187/400000 [00:39<00:06, 8847.60it/s] 86%|████████▌ | 343074/400000 [00:40<00:06, 8712.73it/s] 86%|████████▌ | 343947/400000 [00:40<00:06, 8511.55it/s] 86%|████████▌ | 344801/400000 [00:40<00:06, 8261.28it/s] 86%|████████▋ | 345631/400000 [00:40<00:06, 8078.07it/s] 87%|████████▋ | 346456/400000 [00:40<00:06, 8128.45it/s] 87%|████████▋ | 347388/400000 [00:40<00:06, 8451.14it/s] 87%|████████▋ | 348339/400000 [00:40<00:05, 8742.80it/s] 87%|████████▋ | 349236/400000 [00:40<00:05, 8807.69it/s] 88%|████████▊ | 350122/400000 [00:40<00:05, 8550.51it/s] 88%|████████▊ | 350982/400000 [00:40<00:05, 8524.45it/s] 88%|████████▊ | 351866/400000 [00:41<00:05, 8614.64it/s] 88%|████████▊ | 352802/400000 [00:41<00:05, 8824.44it/s] 88%|████████▊ | 353688/400000 [00:41<00:05, 8744.13it/s] 89%|████████▊ | 354567/400000 [00:41<00:05, 8757.45it/s] 89%|████████▉ | 355445/400000 [00:41<00:05, 8723.96it/s] 89%|████████▉ | 356319/400000 [00:41<00:05, 8573.67it/s] 89%|████████▉ | 357209/400000 [00:41<00:04, 8668.08it/s] 90%|████████▉ | 358078/400000 [00:41<00:04, 8579.22it/s] 90%|████████▉ | 358937/400000 [00:41<00:04, 8266.10it/s] 90%|████████▉ | 359801/400000 [00:41<00:04, 8373.90it/s] 90%|█████████ | 360714/400000 [00:42<00:04, 8585.25it/s] 90%|█████████ | 361625/400000 [00:42<00:04, 8735.84it/s] 91%|█████████ | 362502/400000 [00:42<00:04, 8465.85it/s] 91%|█████████ | 363373/400000 [00:42<00:04, 8536.73it/s] 91%|█████████ | 364230/400000 [00:42<00:04, 8542.29it/s] 91%|█████████▏| 365122/400000 [00:42<00:04, 8648.32it/s] 91%|█████████▏| 365989/400000 [00:42<00:03, 8602.71it/s] 92%|█████████▏| 366851/400000 [00:42<00:03, 8348.41it/s] 92%|█████████▏| 367705/400000 [00:42<00:03, 8402.78it/s] 92%|█████████▏| 368548/400000 [00:42<00:03, 8311.23it/s] 92%|█████████▏| 369381/400000 [00:43<00:03, 8219.55it/s] 93%|█████████▎| 370255/400000 [00:43<00:03, 8366.59it/s] 93%|█████████▎| 371094/400000 [00:43<00:03, 8296.43it/s] 93%|█████████▎| 371986/400000 [00:43<00:03, 8471.71it/s] 93%|█████████▎| 372856/400000 [00:43<00:03, 8537.98it/s] 93%|█████████▎| 373712/400000 [00:43<00:03, 8421.44it/s] 94%|█████████▎| 374566/400000 [00:43<00:03, 8454.96it/s] 94%|█████████▍| 375413/400000 [00:43<00:02, 8357.44it/s] 94%|█████████▍| 376303/400000 [00:43<00:02, 8512.42it/s] 94%|█████████▍| 377156/400000 [00:44<00:02, 8460.96it/s] 95%|█████████▍| 378036/400000 [00:44<00:02, 8559.76it/s] 95%|█████████▍| 378952/400000 [00:44<00:02, 8731.19it/s] 95%|█████████▍| 379849/400000 [00:44<00:02, 8799.23it/s] 95%|█████████▌| 380731/400000 [00:44<00:02, 8712.48it/s] 95%|█████████▌| 381604/400000 [00:44<00:02, 8634.05it/s] 96%|█████████▌| 382469/400000 [00:44<00:02, 8517.12it/s] 96%|█████████▌| 383403/400000 [00:44<00:01, 8746.55it/s] 96%|█████████▌| 384319/400000 [00:44<00:01, 8865.56it/s] 96%|█████████▋| 385219/400000 [00:44<00:01, 8905.09it/s] 97%|█████████▋| 386111/400000 [00:45<00:01, 8747.27it/s] 97%|█████████▋| 387013/400000 [00:45<00:01, 8824.15it/s] 97%|█████████▋| 387951/400000 [00:45<00:01, 8982.21it/s] 97%|█████████▋| 388851/400000 [00:45<00:01, 8948.23it/s] 97%|█████████▋| 389747/400000 [00:45<00:01, 8564.49it/s] 98%|█████████▊| 390608/400000 [00:45<00:01, 8255.66it/s] 98%|█████████▊| 391439/400000 [00:45<00:01, 8267.23it/s] 98%|█████████▊| 392327/400000 [00:45<00:00, 8440.55it/s] 98%|█████████▊| 393175/400000 [00:45<00:00, 8420.34it/s] 99%|█████████▊| 394127/400000 [00:45<00:00, 8720.46it/s] 99%|█████████▉| 395032/400000 [00:46<00:00, 8816.38it/s] 99%|█████████▉| 395935/400000 [00:46<00:00, 8878.15it/s] 99%|█████████▉| 396841/400000 [00:46<00:00, 8931.21it/s] 99%|█████████▉| 397736/400000 [00:46<00:00, 8808.66it/s]100%|█████████▉| 398631/400000 [00:46<00:00, 8847.91it/s]100%|█████████▉| 399517/400000 [00:46<00:00, 8707.17it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8577.15it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f13080d1d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01140016428220255 	 Accuracy: 52
Train Epoch: 1 	 Loss: 0.01099351915627419 	 Accuracy: 68

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
2020-05-13 10:24:10.121292: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 10:24:10.126088: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 10:24:10.126244: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55bf527e3a10 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 10:24:10.126260: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f12b41009b0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6666 - accuracy: 0.5000
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4060 - accuracy: 0.5170 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.3600 - accuracy: 0.5200
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.4366 - accuracy: 0.5150
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6053 - accuracy: 0.5040
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5746 - accuracy: 0.5060
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5790 - accuracy: 0.5057
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6034 - accuracy: 0.5041
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6104 - accuracy: 0.5037
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6344 - accuracy: 0.5021
11000/25000 [============>.................] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
12000/25000 [=============>................] - ETA: 4s - loss: 7.6308 - accuracy: 0.5023
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6242 - accuracy: 0.5028
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6108 - accuracy: 0.5036
15000/25000 [=================>............] - ETA: 3s - loss: 7.6084 - accuracy: 0.5038
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6120 - accuracy: 0.5036
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6008 - accuracy: 0.5043
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6172 - accuracy: 0.5032
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6206 - accuracy: 0.5030
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6237 - accuracy: 0.5028
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6170 - accuracy: 0.5032
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6346 - accuracy: 0.5021
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6453 - accuracy: 0.5014
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 10s 388us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f1268762710> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f125f1f4ef0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.8470 - crf_viterbi_accuracy: 0.0267 - val_loss: 1.7071 - val_crf_viterbi_accuracy: 0.0000e+00

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

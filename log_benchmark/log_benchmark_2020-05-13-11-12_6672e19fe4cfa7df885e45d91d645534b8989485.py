
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f676512cfd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 11:12:38.105158
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 11:12:38.110023
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 11:12:38.113803
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 11:12:38.119232
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f6770ef6470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 354381.9062
Epoch 2/10

1/1 [==============================] - 0s 125ms/step - loss: 271464.7188
Epoch 3/10

1/1 [==============================] - 0s 107ms/step - loss: 175721.4219
Epoch 4/10

1/1 [==============================] - 0s 108ms/step - loss: 101797.6797
Epoch 5/10

1/1 [==============================] - 0s 104ms/step - loss: 56765.7812
Epoch 6/10

1/1 [==============================] - 0s 111ms/step - loss: 31793.8574
Epoch 7/10

1/1 [==============================] - 0s 106ms/step - loss: 18528.6309
Epoch 8/10

1/1 [==============================] - 0s 111ms/step - loss: 12179.9141
Epoch 9/10

1/1 [==============================] - 0s 112ms/step - loss: 8625.2686
Epoch 10/10

1/1 [==============================] - 0s 109ms/step - loss: 6456.0933

  #### Inference Need return ypred, ytrue ######################### 
[[ 3.97794634e-01  5.84368050e-01 -4.90358740e-01  1.20427907e-01
   6.67524397e-01 -4.22264010e-01  8.73061478e-01 -1.04055941e+00
  -9.21295047e-01  8.11889708e-01  5.60915947e-01  1.42765582e-01
   4.17639792e-01 -3.18625033e-01 -6.71345294e-01 -5.70303202e-03
  -2.03457093e+00  9.37465310e-01 -1.29169917e+00 -5.42002320e-01
   6.75593972e-01  9.04909253e-01  3.73208523e-02  1.55282930e-01
  -1.62500948e-01  4.30169046e-01 -5.92172623e-01  6.67292029e-02
  -8.57085705e-01  1.63689780e+00  3.76045704e-02  1.74328756e+00
  -9.17054415e-01  2.67703295e-01 -9.40202832e-01  1.02663314e+00
   1.54070497e+00 -4.36850131e-01 -9.87172425e-01 -1.34783661e+00
   1.23748446e+00 -1.14420140e+00 -6.81858063e-02  9.18802619e-01
  -1.53903234e+00 -4.80676115e-01 -9.18665648e-01  3.22723567e-01
   1.07152259e+00  7.03658044e-01 -1.77930474e-01  6.79172814e-01
  -2.01151490e-01 -1.31276593e-01 -1.93673790e-01  1.95793182e-01
   2.48028278e-01 -8.17111254e-01  9.91077304e-01  4.45730120e-01
  -8.60508978e-01  1.40928519e+00  6.94965959e-01 -1.02513993e+00
   9.65744853e-01  1.82823205e+00  2.61602938e-01  4.60190445e-01
  -1.13787103e+00  1.36607993e+00 -5.42067230e-01  9.51145291e-02
  -1.69653296e-01  1.14346415e-01 -2.82725096e-02  1.55454367e-01
  -9.85050261e-01  1.07375836e+00 -7.94058084e-01 -8.09456110e-02
   1.05901682e+00 -1.29483247e+00 -3.77309948e-01 -2.01719612e-01
   1.41465902e-01 -1.19579601e+00  1.56242847e-02  4.81813878e-01
  -3.41257513e-01  8.88685763e-01  1.70043707e+00  9.97930884e-01
   4.26351368e-01 -1.07356966e+00 -7.16601551e-01  7.77661204e-01
   3.88456881e-01  1.52467799e+00 -2.16315538e-01 -1.51077151e+00
  -1.66921222e+00  1.40973425e+00  6.08477056e-01 -1.15102649e+00
   3.27744603e-01 -2.93392420e-01 -5.32945395e-02  1.14363635e+00
   2.13766903e-01 -1.54626369e-01 -3.24466199e-01 -2.36475766e-01
  -2.25373411e+00 -1.00252461e+00 -1.75104499e-01 -4.16891485e-01
  -9.77103710e-02 -3.00181329e-01  6.13972008e-01 -4.59850848e-01
   1.98479041e-01  8.94238281e+00  9.31787682e+00  9.48941231e+00
   7.86215401e+00  8.87595844e+00  7.72503757e+00  8.63433647e+00
   7.93887615e+00  8.57886028e+00  8.33823776e+00  6.84713650e+00
   8.83065987e+00  8.18068600e+00  8.61110210e+00  8.24332619e+00
   9.04137135e+00  8.78021526e+00  7.31417322e+00  8.12568474e+00
   7.77786207e+00  8.76153755e+00  9.23400307e+00  7.39811182e+00
   8.22181606e+00  5.72506189e+00  9.17232227e+00  9.18104172e+00
   1.04327936e+01  9.19362450e+00  7.06129789e+00  6.57965326e+00
   8.35006905e+00  6.19999790e+00  6.93443632e+00  8.30098152e+00
   7.53352976e+00  8.69004059e+00  8.95720005e+00  7.39246130e+00
   9.19686890e+00  8.42935371e+00  6.39285374e+00  6.94412231e+00
   8.71490669e+00  7.93006706e+00  6.91733170e+00  9.00519466e+00
   7.59801292e+00  8.69496536e+00  8.89244747e+00  8.47898674e+00
   8.44888210e+00  7.65121078e+00  7.38524294e+00  8.33271027e+00
   8.36005878e+00  6.42508888e+00  8.51484680e+00  7.07313824e+00
   1.62788534e+00  9.31950033e-01  3.42150807e-01  8.69645000e-01
   3.53630424e-01  1.81732404e+00  1.56113291e+00  2.40285492e+00
   1.81901836e+00  2.85352230e-01  5.70862591e-01  5.26865542e-01
   1.04916716e+00  8.53569627e-01  1.73755276e+00  1.42886496e+00
   1.19808686e+00  7.07602382e-01  1.20225191e+00  1.85782671e+00
   4.52325284e-01  1.47208071e+00  2.10873389e+00  1.26266408e+00
   7.27975726e-01  5.93509257e-01  3.16152632e-01  7.13410854e-01
   1.81038678e-01  2.53983855e-01  2.75323331e-01  2.62945795e+00
   1.12046254e+00  1.96239007e+00  1.21958423e+00  2.58841395e-01
   3.69473994e-01  5.94283700e-01  2.98296738e+00  1.11759269e+00
   1.96736324e+00  1.46263301e-01  4.08934832e-01  1.17679822e+00
   8.62380862e-02  2.92099297e-01  6.75203383e-01  1.21822584e+00
   2.92694306e+00  6.35883808e-01  3.46350574e+00  3.38476324e+00
   1.31488216e+00  1.03183103e+00  1.06930685e+00  8.59562635e-01
   5.91172099e-01  8.43407154e-01  4.79494691e-01  1.25605166e+00
   6.85135126e-01  5.84775448e-01  1.23956800e-01  9.06230867e-01
   1.86356390e+00  7.74800897e-01  6.67489529e-01  4.94890034e-01
   1.82944655e+00  3.21659744e-01  1.06664681e+00  9.89122331e-01
   1.20450616e+00  1.85816908e+00  1.01579738e+00  1.98936045e+00
   8.43194485e-01  1.85034156e+00  2.41025269e-01  7.48497844e-01
   6.55794501e-01  1.01224339e+00  2.29526877e-01  6.94038689e-01
   9.43658888e-01  7.80619860e-01  1.32853127e+00  1.86893475e+00
   3.30853105e-01  2.62719727e+00  2.13972998e+00  2.34854341e-01
   1.14197564e+00  1.57236123e+00  1.25623679e+00  2.76169777e+00
   3.02632189e+00  1.18838096e+00  2.04426241e+00  2.37448263e+00
   2.56979895e+00  3.02469254e-01  3.54674816e-01  6.40758514e-01
   2.30883956e-01  2.40710354e+00  2.74039626e-01  9.28028584e-01
   5.98254979e-01  1.58660376e+00  1.45680451e+00  1.66726410e+00
   1.38935542e+00  1.49796426e-01  3.00531578e+00  2.93998003e+00
   1.78174913e+00  2.65659213e-01  1.75317740e+00  2.04250193e+00
   1.35383606e-01  8.95349598e+00  9.31241608e+00  7.87972832e+00
   8.14382362e+00  7.61313820e+00  8.17382431e+00  9.25417423e+00
   1.05645924e+01  8.24064732e+00  8.55552292e+00  8.24073505e+00
   9.29316521e+00  9.35349464e+00  8.29057121e+00  7.44338465e+00
   9.29764748e+00  8.95005417e+00  8.52876759e+00  8.89418507e+00
   8.71300316e+00  7.92077684e+00  7.11268330e+00  7.72265625e+00
   9.65556335e+00  8.50209808e+00  8.01175976e+00  9.60056877e+00
   8.72910786e+00  9.17298317e+00  9.07080269e+00  9.48514462e+00
   8.85172176e+00  8.42952251e+00  8.50024986e+00  9.31682110e+00
   9.74869156e+00  8.16850662e+00  8.79492664e+00  8.61824322e+00
   8.10315704e+00  7.68481064e+00  9.13842869e+00  8.51538372e+00
   8.07565975e+00  8.59041405e+00  9.15178204e+00  8.40793991e+00
   8.11351299e+00  8.57742596e+00  7.43806601e+00  8.56662941e+00
   8.95588207e+00  7.29832554e+00  7.39900970e+00  9.22377396e+00
   7.83844805e+00  1.02484989e+01  9.68290138e+00  9.35141182e+00
  -7.05388260e+00 -7.46972656e+00  9.93496418e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 11:12:48.169272
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.6694
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 11:12:48.173471
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8987.59
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 11:12:48.176706
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   94.4422
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 11:12:48.180361
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -803.902
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140081694942824
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140080484999912
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @140080485000416
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140080485000920
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140080485001424
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @140080485001928

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f6764df7470> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.592013
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.568256
grad_step = 000002, loss = 0.547875
grad_step = 000003, loss = 0.529522
grad_step = 000004, loss = 0.512097
grad_step = 000005, loss = 0.496208
grad_step = 000006, loss = 0.481929
grad_step = 000007, loss = 0.470042
grad_step = 000008, loss = 0.459740
grad_step = 000009, loss = 0.448235
grad_step = 000010, loss = 0.436859
grad_step = 000011, loss = 0.426352
grad_step = 000012, loss = 0.416024
grad_step = 000013, loss = 0.405286
grad_step = 000014, loss = 0.393834
grad_step = 000015, loss = 0.382458
grad_step = 000016, loss = 0.370074
grad_step = 000017, loss = 0.357525
grad_step = 000018, loss = 0.344188
grad_step = 000019, loss = 0.330644
grad_step = 000020, loss = 0.317446
grad_step = 000021, loss = 0.304801
grad_step = 000022, loss = 0.293554
grad_step = 000023, loss = 0.283636
grad_step = 000024, loss = 0.274079
grad_step = 000025, loss = 0.264545
grad_step = 000026, loss = 0.254717
grad_step = 000027, loss = 0.244501
grad_step = 000028, loss = 0.234451
grad_step = 000029, loss = 0.224547
grad_step = 000030, loss = 0.215043
grad_step = 000031, loss = 0.205974
grad_step = 000032, loss = 0.197217
grad_step = 000033, loss = 0.188730
grad_step = 000034, loss = 0.180169
grad_step = 000035, loss = 0.171908
grad_step = 000036, loss = 0.163859
grad_step = 000037, loss = 0.156030
grad_step = 000038, loss = 0.148473
grad_step = 000039, loss = 0.141196
grad_step = 000040, loss = 0.134243
grad_step = 000041, loss = 0.127539
grad_step = 000042, loss = 0.121123
grad_step = 000043, loss = 0.114861
grad_step = 000044, loss = 0.108764
grad_step = 000045, loss = 0.102914
grad_step = 000046, loss = 0.097310
grad_step = 000047, loss = 0.091990
grad_step = 000048, loss = 0.086921
grad_step = 000049, loss = 0.082085
grad_step = 000050, loss = 0.077416
grad_step = 000051, loss = 0.072959
grad_step = 000052, loss = 0.068681
grad_step = 000053, loss = 0.064610
grad_step = 000054, loss = 0.060786
grad_step = 000055, loss = 0.057161
grad_step = 000056, loss = 0.053754
grad_step = 000057, loss = 0.050504
grad_step = 000058, loss = 0.047400
grad_step = 000059, loss = 0.044485
grad_step = 000060, loss = 0.041742
grad_step = 000061, loss = 0.039179
grad_step = 000062, loss = 0.036736
grad_step = 000063, loss = 0.034394
grad_step = 000064, loss = 0.032162
grad_step = 000065, loss = 0.030101
grad_step = 000066, loss = 0.028201
grad_step = 000067, loss = 0.026401
grad_step = 000068, loss = 0.024672
grad_step = 000069, loss = 0.023026
grad_step = 000070, loss = 0.021496
grad_step = 000071, loss = 0.020091
grad_step = 000072, loss = 0.018779
grad_step = 000073, loss = 0.017537
grad_step = 000074, loss = 0.016335
grad_step = 000075, loss = 0.015203
grad_step = 000076, loss = 0.014169
grad_step = 000077, loss = 0.013221
grad_step = 000078, loss = 0.012340
grad_step = 000079, loss = 0.011501
grad_step = 000080, loss = 0.010710
grad_step = 000081, loss = 0.009975
grad_step = 000082, loss = 0.009297
grad_step = 000083, loss = 0.008669
grad_step = 000084, loss = 0.008081
grad_step = 000085, loss = 0.007531
grad_step = 000086, loss = 0.007023
grad_step = 000087, loss = 0.006559
grad_step = 000088, loss = 0.006136
grad_step = 000089, loss = 0.005754
grad_step = 000090, loss = 0.005412
grad_step = 000091, loss = 0.005114
grad_step = 000092, loss = 0.004865
grad_step = 000093, loss = 0.004665
grad_step = 000094, loss = 0.004497
grad_step = 000095, loss = 0.004312
grad_step = 000096, loss = 0.004039
grad_step = 000097, loss = 0.003722
grad_step = 000098, loss = 0.003480
grad_step = 000099, loss = 0.003403
grad_step = 000100, loss = 0.003337
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003131
grad_step = 000102, loss = 0.002918
grad_step = 000103, loss = 0.002860
grad_step = 000104, loss = 0.002855
grad_step = 000105, loss = 0.002734
grad_step = 000106, loss = 0.002569
grad_step = 000107, loss = 0.002520
grad_step = 000108, loss = 0.002533
grad_step = 000109, loss = 0.002465
grad_step = 000110, loss = 0.002358
grad_step = 000111, loss = 0.002317
grad_step = 000112, loss = 0.002317
grad_step = 000113, loss = 0.002281
grad_step = 000114, loss = 0.002222
grad_step = 000115, loss = 0.002201
grad_step = 000116, loss = 0.002198
grad_step = 000117, loss = 0.002166
grad_step = 000118, loss = 0.002125
grad_step = 000119, loss = 0.002112
grad_step = 000120, loss = 0.002120
grad_step = 000121, loss = 0.002112
grad_step = 000122, loss = 0.002082
grad_step = 000123, loss = 0.002060
grad_step = 000124, loss = 0.002057
grad_step = 000125, loss = 0.002058
grad_step = 000126, loss = 0.002048
grad_step = 000127, loss = 0.002033
grad_step = 000128, loss = 0.002027
grad_step = 000129, loss = 0.002030
grad_step = 000130, loss = 0.002030
grad_step = 000131, loss = 0.002024
grad_step = 000132, loss = 0.002014
grad_step = 000133, loss = 0.002008
grad_step = 000134, loss = 0.002008
grad_step = 000135, loss = 0.002011
grad_step = 000136, loss = 0.002013
grad_step = 000137, loss = 0.002014
grad_step = 000138, loss = 0.002017
grad_step = 000139, loss = 0.002027
grad_step = 000140, loss = 0.002040
grad_step = 000141, loss = 0.002058
grad_step = 000142, loss = 0.002059
grad_step = 000143, loss = 0.002047
grad_step = 000144, loss = 0.002011
grad_step = 000145, loss = 0.001973
grad_step = 000146, loss = 0.001949
grad_step = 000147, loss = 0.001945
grad_step = 000148, loss = 0.001957
grad_step = 000149, loss = 0.001972
grad_step = 000150, loss = 0.001983
grad_step = 000151, loss = 0.001980
grad_step = 000152, loss = 0.001968
grad_step = 000153, loss = 0.001949
grad_step = 000154, loss = 0.001934
grad_step = 000155, loss = 0.001928
grad_step = 000156, loss = 0.001931
grad_step = 000157, loss = 0.001940
grad_step = 000158, loss = 0.001953
grad_step = 000159, loss = 0.001967
grad_step = 000160, loss = 0.001977
grad_step = 000161, loss = 0.001983
grad_step = 000162, loss = 0.001985
grad_step = 000163, loss = 0.001980
grad_step = 000164, loss = 0.001971
grad_step = 000165, loss = 0.001958
grad_step = 000166, loss = 0.001948
grad_step = 000167, loss = 0.001945
grad_step = 000168, loss = 0.001952
grad_step = 000169, loss = 0.001963
grad_step = 000170, loss = 0.001975
grad_step = 000171, loss = 0.001963
grad_step = 000172, loss = 0.001938
grad_step = 000173, loss = 0.001899
grad_step = 000174, loss = 0.001873
grad_step = 000175, loss = 0.001868
grad_step = 000176, loss = 0.001881
grad_step = 000177, loss = 0.001902
grad_step = 000178, loss = 0.001916
grad_step = 000179, loss = 0.001921
grad_step = 000180, loss = 0.001912
grad_step = 000181, loss = 0.001899
grad_step = 000182, loss = 0.001890
grad_step = 000183, loss = 0.001897
grad_step = 000184, loss = 0.001921
grad_step = 000185, loss = 0.001962
grad_step = 000186, loss = 0.002003
grad_step = 000187, loss = 0.002019
grad_step = 000188, loss = 0.001986
grad_step = 000189, loss = 0.001924
grad_step = 000190, loss = 0.001861
grad_step = 000191, loss = 0.001840
grad_step = 000192, loss = 0.001864
grad_step = 000193, loss = 0.001904
grad_step = 000194, loss = 0.001919
grad_step = 000195, loss = 0.001893
grad_step = 000196, loss = 0.001850
grad_step = 000197, loss = 0.001825
grad_step = 000198, loss = 0.001828
grad_step = 000199, loss = 0.001847
grad_step = 000200, loss = 0.001862
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001861
grad_step = 000202, loss = 0.001843
grad_step = 000203, loss = 0.001820
grad_step = 000204, loss = 0.001809
grad_step = 000205, loss = 0.001813
grad_step = 000206, loss = 0.001826
grad_step = 000207, loss = 0.001839
grad_step = 000208, loss = 0.001847
grad_step = 000209, loss = 0.001861
grad_step = 000210, loss = 0.001887
grad_step = 000211, loss = 0.001955
grad_step = 000212, loss = 0.002075
grad_step = 000213, loss = 0.002246
grad_step = 000214, loss = 0.002361
grad_step = 000215, loss = 0.002250
grad_step = 000216, loss = 0.001994
grad_step = 000217, loss = 0.001845
grad_step = 000218, loss = 0.001908
grad_step = 000219, loss = 0.002022
grad_step = 000220, loss = 0.002013
grad_step = 000221, loss = 0.001900
grad_step = 000222, loss = 0.001831
grad_step = 000223, loss = 0.001890
grad_step = 000224, loss = 0.001932
grad_step = 000225, loss = 0.001874
grad_step = 000226, loss = 0.001825
grad_step = 000227, loss = 0.001836
grad_step = 000228, loss = 0.001852
grad_step = 000229, loss = 0.001840
grad_step = 000230, loss = 0.001823
grad_step = 000231, loss = 0.001807
grad_step = 000232, loss = 0.001810
grad_step = 000233, loss = 0.001816
grad_step = 000234, loss = 0.001798
grad_step = 000235, loss = 0.001786
grad_step = 000236, loss = 0.001794
grad_step = 000237, loss = 0.001794
grad_step = 000238, loss = 0.001781
grad_step = 000239, loss = 0.001773
grad_step = 000240, loss = 0.001774
grad_step = 000241, loss = 0.001775
grad_step = 000242, loss = 0.001772
grad_step = 000243, loss = 0.001764
grad_step = 000244, loss = 0.001756
grad_step = 000245, loss = 0.001758
grad_step = 000246, loss = 0.001762
grad_step = 000247, loss = 0.001758
grad_step = 000248, loss = 0.001749
grad_step = 000249, loss = 0.001744
grad_step = 000250, loss = 0.001745
grad_step = 000251, loss = 0.001747
grad_step = 000252, loss = 0.001746
grad_step = 000253, loss = 0.001742
grad_step = 000254, loss = 0.001735
grad_step = 000255, loss = 0.001731
grad_step = 000256, loss = 0.001732
grad_step = 000257, loss = 0.001734
grad_step = 000258, loss = 0.001733
grad_step = 000259, loss = 0.001729
grad_step = 000260, loss = 0.001725
grad_step = 000261, loss = 0.001721
grad_step = 000262, loss = 0.001719
grad_step = 000263, loss = 0.001719
grad_step = 000264, loss = 0.001719
grad_step = 000265, loss = 0.001719
grad_step = 000266, loss = 0.001716
grad_step = 000267, loss = 0.001713
grad_step = 000268, loss = 0.001710
grad_step = 000269, loss = 0.001708
grad_step = 000270, loss = 0.001705
grad_step = 000271, loss = 0.001704
grad_step = 000272, loss = 0.001703
grad_step = 000273, loss = 0.001703
grad_step = 000274, loss = 0.001704
grad_step = 000275, loss = 0.001705
grad_step = 000276, loss = 0.001709
grad_step = 000277, loss = 0.001718
grad_step = 000278, loss = 0.001736
grad_step = 000279, loss = 0.001780
grad_step = 000280, loss = 0.001877
grad_step = 000281, loss = 0.002056
grad_step = 000282, loss = 0.002313
grad_step = 000283, loss = 0.002484
grad_step = 000284, loss = 0.002345
grad_step = 000285, loss = 0.001955
grad_step = 000286, loss = 0.001706
grad_step = 000287, loss = 0.001844
grad_step = 000288, loss = 0.002085
grad_step = 000289, loss = 0.002009
grad_step = 000290, loss = 0.001743
grad_step = 000291, loss = 0.001709
grad_step = 000292, loss = 0.001889
grad_step = 000293, loss = 0.001909
grad_step = 000294, loss = 0.001744
grad_step = 000295, loss = 0.001714
grad_step = 000296, loss = 0.001808
grad_step = 000297, loss = 0.001805
grad_step = 000298, loss = 0.001720
grad_step = 000299, loss = 0.001703
grad_step = 000300, loss = 0.001760
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001755
grad_step = 000302, loss = 0.001675
grad_step = 000303, loss = 0.001675
grad_step = 000304, loss = 0.001732
grad_step = 000305, loss = 0.001714
grad_step = 000306, loss = 0.001662
grad_step = 000307, loss = 0.001670
grad_step = 000308, loss = 0.001695
grad_step = 000309, loss = 0.001684
grad_step = 000310, loss = 0.001657
grad_step = 000311, loss = 0.001657
grad_step = 000312, loss = 0.001674
grad_step = 000313, loss = 0.001669
grad_step = 000314, loss = 0.001642
grad_step = 000315, loss = 0.001641
grad_step = 000316, loss = 0.001657
grad_step = 000317, loss = 0.001655
grad_step = 000318, loss = 0.001640
grad_step = 000319, loss = 0.001634
grad_step = 000320, loss = 0.001637
grad_step = 000321, loss = 0.001639
grad_step = 000322, loss = 0.001634
grad_step = 000323, loss = 0.001625
grad_step = 000324, loss = 0.001624
grad_step = 000325, loss = 0.001629
grad_step = 000326, loss = 0.001628
grad_step = 000327, loss = 0.001621
grad_step = 000328, loss = 0.001615
grad_step = 000329, loss = 0.001613
grad_step = 000330, loss = 0.001614
grad_step = 000331, loss = 0.001615
grad_step = 000332, loss = 0.001612
grad_step = 000333, loss = 0.001607
grad_step = 000334, loss = 0.001603
grad_step = 000335, loss = 0.001602
grad_step = 000336, loss = 0.001601
grad_step = 000337, loss = 0.001601
grad_step = 000338, loss = 0.001600
grad_step = 000339, loss = 0.001598
grad_step = 000340, loss = 0.001596
grad_step = 000341, loss = 0.001596
grad_step = 000342, loss = 0.001599
grad_step = 000343, loss = 0.001611
grad_step = 000344, loss = 0.001642
grad_step = 000345, loss = 0.001697
grad_step = 000346, loss = 0.001811
grad_step = 000347, loss = 0.001917
grad_step = 000348, loss = 0.002038
grad_step = 000349, loss = 0.001946
grad_step = 000350, loss = 0.001770
grad_step = 000351, loss = 0.001604
grad_step = 000352, loss = 0.001632
grad_step = 000353, loss = 0.001758
grad_step = 000354, loss = 0.001772
grad_step = 000355, loss = 0.001663
grad_step = 000356, loss = 0.001580
grad_step = 000357, loss = 0.001628
grad_step = 000358, loss = 0.001718
grad_step = 000359, loss = 0.001696
grad_step = 000360, loss = 0.001601
grad_step = 000361, loss = 0.001566
grad_step = 000362, loss = 0.001608
grad_step = 000363, loss = 0.001649
grad_step = 000364, loss = 0.001616
grad_step = 000365, loss = 0.001569
grad_step = 000366, loss = 0.001566
grad_step = 000367, loss = 0.001596
grad_step = 000368, loss = 0.001603
grad_step = 000369, loss = 0.001574
grad_step = 000370, loss = 0.001547
grad_step = 000371, loss = 0.001554
grad_step = 000372, loss = 0.001574
grad_step = 000373, loss = 0.001576
grad_step = 000374, loss = 0.001555
grad_step = 000375, loss = 0.001538
grad_step = 000376, loss = 0.001540
grad_step = 000377, loss = 0.001550
grad_step = 000378, loss = 0.001552
grad_step = 000379, loss = 0.001540
grad_step = 000380, loss = 0.001527
grad_step = 000381, loss = 0.001526
grad_step = 000382, loss = 0.001530
grad_step = 000383, loss = 0.001533
grad_step = 000384, loss = 0.001528
grad_step = 000385, loss = 0.001520
grad_step = 000386, loss = 0.001516
grad_step = 000387, loss = 0.001518
grad_step = 000388, loss = 0.001523
grad_step = 000389, loss = 0.001526
grad_step = 000390, loss = 0.001531
grad_step = 000391, loss = 0.001547
grad_step = 000392, loss = 0.001579
grad_step = 000393, loss = 0.001642
grad_step = 000394, loss = 0.001704
grad_step = 000395, loss = 0.001774
grad_step = 000396, loss = 0.001772
grad_step = 000397, loss = 0.001703
grad_step = 000398, loss = 0.001592
grad_step = 000399, loss = 0.001518
grad_step = 000400, loss = 0.001517
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001564
grad_step = 000402, loss = 0.001601
grad_step = 000403, loss = 0.001588
grad_step = 000404, loss = 0.001540
grad_step = 000405, loss = 0.001494
grad_step = 000406, loss = 0.001486
grad_step = 000407, loss = 0.001514
grad_step = 000408, loss = 0.001546
grad_step = 000409, loss = 0.001550
grad_step = 000410, loss = 0.001521
grad_step = 000411, loss = 0.001482
grad_step = 000412, loss = 0.001461
grad_step = 000413, loss = 0.001466
grad_step = 000414, loss = 0.001484
grad_step = 000415, loss = 0.001496
grad_step = 000416, loss = 0.001490
grad_step = 000417, loss = 0.001474
grad_step = 000418, loss = 0.001459
grad_step = 000419, loss = 0.001452
grad_step = 000420, loss = 0.001451
grad_step = 000421, loss = 0.001449
grad_step = 000422, loss = 0.001446
grad_step = 000423, loss = 0.001442
grad_step = 000424, loss = 0.001441
grad_step = 000425, loss = 0.001444
grad_step = 000426, loss = 0.001449
grad_step = 000427, loss = 0.001454
grad_step = 000428, loss = 0.001456
grad_step = 000429, loss = 0.001455
grad_step = 000430, loss = 0.001454
grad_step = 000431, loss = 0.001456
grad_step = 000432, loss = 0.001462
grad_step = 000433, loss = 0.001466
grad_step = 000434, loss = 0.001468
grad_step = 000435, loss = 0.001463
grad_step = 000436, loss = 0.001452
grad_step = 000437, loss = 0.001436
grad_step = 000438, loss = 0.001418
grad_step = 000439, loss = 0.001403
grad_step = 000440, loss = 0.001391
grad_step = 000441, loss = 0.001384
grad_step = 000442, loss = 0.001380
grad_step = 000443, loss = 0.001378
grad_step = 000444, loss = 0.001379
grad_step = 000445, loss = 0.001382
grad_step = 000446, loss = 0.001393
grad_step = 000447, loss = 0.001424
grad_step = 000448, loss = 0.001490
grad_step = 000449, loss = 0.001623
grad_step = 000450, loss = 0.001756
grad_step = 000451, loss = 0.001887
grad_step = 000452, loss = 0.001763
grad_step = 000453, loss = 0.001530
grad_step = 000454, loss = 0.001376
grad_step = 000455, loss = 0.001429
grad_step = 000456, loss = 0.001562
grad_step = 000457, loss = 0.001563
grad_step = 000458, loss = 0.001438
grad_step = 000459, loss = 0.001350
grad_step = 000460, loss = 0.001392
grad_step = 000461, loss = 0.001475
grad_step = 000462, loss = 0.001471
grad_step = 000463, loss = 0.001387
grad_step = 000464, loss = 0.001332
grad_step = 000465, loss = 0.001363
grad_step = 000466, loss = 0.001408
grad_step = 000467, loss = 0.001400
grad_step = 000468, loss = 0.001355
grad_step = 000469, loss = 0.001325
grad_step = 000470, loss = 0.001335
grad_step = 000471, loss = 0.001364
grad_step = 000472, loss = 0.001372
grad_step = 000473, loss = 0.001350
grad_step = 000474, loss = 0.001322
grad_step = 000475, loss = 0.001311
grad_step = 000476, loss = 0.001317
grad_step = 000477, loss = 0.001331
grad_step = 000478, loss = 0.001335
grad_step = 000479, loss = 0.001323
grad_step = 000480, loss = 0.001307
grad_step = 000481, loss = 0.001299
grad_step = 000482, loss = 0.001300
grad_step = 000483, loss = 0.001304
grad_step = 000484, loss = 0.001309
grad_step = 000485, loss = 0.001308
grad_step = 000486, loss = 0.001300
grad_step = 000487, loss = 0.001291
grad_step = 000488, loss = 0.001286
grad_step = 000489, loss = 0.001286
grad_step = 000490, loss = 0.001287
grad_step = 000491, loss = 0.001289
grad_step = 000492, loss = 0.001290
grad_step = 000493, loss = 0.001288
grad_step = 000494, loss = 0.001285
grad_step = 000495, loss = 0.001282
grad_step = 000496, loss = 0.001281
grad_step = 000497, loss = 0.001288
grad_step = 000498, loss = 0.001308
grad_step = 000499, loss = 0.001349
grad_step = 000500, loss = 0.001420
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001481
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

  date_run                              2020-05-13 11:13:11.939943
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.285637
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 11:13:11.946351
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.211317
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 11:13:11.954351
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.153495
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 11:13:11.960045
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -2.21104
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
0   2020-05-13 11:12:38.105158  ...    mean_absolute_error
1   2020-05-13 11:12:38.110023  ...     mean_squared_error
2   2020-05-13 11:12:38.113803  ...  median_absolute_error
3   2020-05-13 11:12:38.119232  ...               r2_score
4   2020-05-13 11:12:48.169272  ...    mean_absolute_error
5   2020-05-13 11:12:48.173471  ...     mean_squared_error
6   2020-05-13 11:12:48.176706  ...  median_absolute_error
7   2020-05-13 11:12:48.180361  ...               r2_score
8   2020-05-13 11:13:11.939943  ...    mean_absolute_error
9   2020-05-13 11:13:11.946351  ...     mean_squared_error
10  2020-05-13 11:13:11.954351  ...  median_absolute_error
11  2020-05-13 11:13:11.960045  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff6c2d94cf8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 31%|███▏      | 3104768/9912422 [00:00<00:00, 31042306.83it/s]9920512it [00:00, 33927450.54it/s]                             
0it [00:00, ?it/s]32768it [00:00, 615975.66it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 462539.39it/s]1654784it [00:00, 10944023.94it/s]                         
0it [00:00, ?it/s]8192it [00:00, 198433.42it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff67574deb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff674d7b0f0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff67574deb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff674cd4128> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff67250fa58> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff6724fa780> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff67574deb8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff674c91780> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff67250fa58> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7ff6c2d57f28> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f03f98fb208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=3dd330322936e80ce0e8164dc38b54075e8aaa9bccfc9ca0d7832d6d2668e2e5
  Stored in directory: /tmp/pip-ephem-wheel-cache-xqq2v0bg/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f03916f57b8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 4915200/17464789 [=======>......................] - ETA: 0s
10756096/17464789 [=================>............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 11:14:40.398791: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 11:14:40.403274: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 11:14:40.403448: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559004f92030 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 11:14:40.403464: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.3600 - accuracy: 0.5200
 2000/25000 [=>............................] - ETA: 10s - loss: 7.5210 - accuracy: 0.5095
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.4928 - accuracy: 0.5113 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5210 - accuracy: 0.5095
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5470 - accuracy: 0.5078
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6257 - accuracy: 0.5027
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6294 - accuracy: 0.5024
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6379 - accuracy: 0.5019
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6172 - accuracy: 0.5032
10000/25000 [===========>..................] - ETA: 5s - loss: 7.5654 - accuracy: 0.5066
11000/25000 [============>.................] - ETA: 4s - loss: 7.5816 - accuracy: 0.5055
12000/25000 [=============>................] - ETA: 4s - loss: 7.6040 - accuracy: 0.5041
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6171 - accuracy: 0.5032
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6239 - accuracy: 0.5028
15000/25000 [=================>............] - ETA: 3s - loss: 7.6135 - accuracy: 0.5035
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6245 - accuracy: 0.5027
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6414 - accuracy: 0.5016
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6368 - accuracy: 0.5019
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6473 - accuracy: 0.5013
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6586 - accuracy: 0.5005
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6687 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6626 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6730 - accuracy: 0.4996
25000/25000 [==============================] - 10s 394us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 11:14:57.895745
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 11:14:57.895745  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:37:05, 9.73kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:28:19, 13.7kB/s].vector_cache/glove.6B.zip:   0%|          | 106k/862M [00:01<12:29:23, 19.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:01<8:46:02, 27.3kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.73M/862M [00:01<6:08:08, 39.0kB/s].vector_cache/glove.6B.zip:   1%|          | 6.48M/862M [00:01<4:16:21, 55.6kB/s].vector_cache/glove.6B.zip:   1%|          | 9.39M/862M [00:01<2:58:59, 79.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.7M/862M [00:01<2:04:46, 113kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 18.2M/862M [00:01<1:26:58, 162kB/s].vector_cache/glove.6B.zip:   3%|▎         | 22.5M/862M [00:01<1:00:40, 231kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.7M/862M [00:02<42:21, 329kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.0M/862M [00:02<29:36, 468kB/s].vector_cache/glove.6B.zip:   4%|▍         | 34.9M/862M [00:02<20:43, 665kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.4M/862M [00:02<14:31, 944kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.9M/862M [00:02<10:14, 1.33MB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.8M/862M [00:04<15:03, 906kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 44.3M/862M [00:04<11:29, 1.19MB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.0M/862M [00:04<08:16, 1.64MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.5M/862M [00:04<05:52, 2.30MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:04<04:50, 2.79MB/s].vector_cache/glove.6B.zip:   6%|▋         | 54.9M/862M [00:05<03:31, 3.83MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:06<07:59, 1.68MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:07<08:18, 1.61MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.2M/862M [00:07<06:24, 2.09MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.8M/862M [00:07<04:44, 2.82MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:09<07:52, 1.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:09<08:14, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:09<06:20, 2.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.7M/862M [00:09<04:44, 2.81MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:11<08:00, 1.66MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:11<11:32, 1.15MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:11<09:29, 1.40MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.5M/862M [00:11<06:56, 1.91MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:14<09:05, 1.46MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:14<17:19, 763kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:14<14:57, 884kB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.6M/862M [00:14<11:11, 1.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:14<07:56, 1.66MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:16<23:30, 560kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:16<17:36, 747kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:16<12:51, 1.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.7M/862M [00:16<09:11, 1.43MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.9M/862M [00:18<11:56, 1.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:18<09:49, 1.33MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:18<07:45, 1.69MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:18<05:35, 2.33MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.0M/862M [00:20<23:11, 561kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:20<17:25, 747kB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.0M/862M [00:20<12:49, 1.01MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.7M/862M [00:20<09:11, 1.41MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:22<10:46, 1.20MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:22<09:24, 1.38MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:22<07:25, 1.74MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.0M/862M [00:22<05:24, 2.39MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:24<08:24, 1.53MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:24<07:15, 1.77MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:24<05:24, 2.37MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:26<06:40, 1.92MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.9M/862M [00:26<06:02, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.3M/862M [00:26<04:30, 2.83MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:28<06:02, 2.11MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:28<06:27, 1.97MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.5M/862M [00:28<05:11, 2.45MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:28<04:04, 3.12MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<05:44, 2.21MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<05:20, 2.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:30<04:01, 3.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:30<03:18, 3.81MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<06:31, 1.93MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:32<06:02, 2.08MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:32<04:33, 2.76MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:33<05:50, 2.15MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<05:29, 2.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:34<04:10, 3.00MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<06:49, 1.82MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<15:23, 810kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<13:23, 931kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:37<10:01, 1.24MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:37<07:07, 1.74MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:38<29:33, 420kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:38<21:46, 569kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:38<15:32, 796kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:40<13:41, 900kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<11:44, 1.05MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<08:48, 1.40MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:40<06:29, 1.89MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:42<07:17, 1.68MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<06:23, 1.92MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:42<04:47, 2.56MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<06:10, 1.98MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<05:35, 2.18MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<04:10, 2.91MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<05:43, 2.12MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<06:27, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:46<05:04, 2.38MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:46<03:42, 3.26MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<08:53, 1.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<07:18, 1.65MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:48<05:21, 2.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<03:55, 3.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<34:05, 352kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<25:06, 477kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:50<17:51, 670kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:53<16:34, 719kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:53<59:24, 201kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:53<46:12, 258kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:53<35:23, 337kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:53<25:31, 467kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:53<18:09, 655kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:56<16:02, 739kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:56<16:30, 718kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:56<12:49, 924kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:56<09:16, 1.27MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:58<09:02, 1.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:58<07:33, 1.56MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:58<05:34, 2.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [01:00<07:35, 1.54MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:00<15:15, 768kB/s] .vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:00<13:15, 883kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [01:01<09:53, 1.18MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:01<07:03, 1.65MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:02<11:10, 1.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:02<08:51, 1.31MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:02<06:40, 1.74MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:02<04:47, 2.42MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:04<14:36, 792kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:04<11:14, 1.03MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:04<08:28, 1.36MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:04<06:05, 1.89MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:07<12:54, 891kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:07<20:33, 560kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:08<17:05, 673kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:08<12:34, 914kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:08<08:54, 1.28MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:09<12:24, 922kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:09<09:52, 1.16MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:09<07:44, 1.48MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:10<05:43, 1.99MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:10<04:15, 2.68MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:10<03:22, 3.37MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:11<36:45, 309kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:11<29:20, 387kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:11<25:14, 450kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:12<20:40, 549kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:12<16:15, 699kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:12<11:50, 959kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:12<08:24, 1.35MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:14<18:17, 618kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:14<17:25, 649kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:14<13:24, 842kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:14<09:38, 1.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:16<09:08, 1.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:16<07:36, 1.48MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:16<05:34, 2.01MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:18<06:27, 1.73MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:18<05:30, 2.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:18<04:16, 2.61MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:18<03:06, 3.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:20<23:04, 481kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:20<17:18, 641kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:20<12:21, 895kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:22<11:11, 985kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:22<10:05, 1.09MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:22<07:33, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:22<05:25, 2.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:24<08:51, 1.24MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:24<08:12, 1.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:24<06:17, 1.74MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:24<04:30, 2.42MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:26<12:01, 906kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:26<10:40, 1.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:26<08:01, 1.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:26<05:48, 1.87MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:28<07:23, 1.46MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:28<06:55, 1.56MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:28<05:55, 1.82MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:28<04:27, 2.43MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:30<05:10, 2.08MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:30<04:31, 2.38MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:30<03:49, 2.80MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:30<02:51, 3.75MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:34<09:08, 1.17MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:34<16:55, 631kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:34<14:21, 744kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:34<10:39, 1.00MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:34<07:35, 1.40MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:34<05:38, 1.88MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:34<04:15, 2.48MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:35<10:31, 1.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:35<07:29, 1.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:37<10:59, 956kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:38<16:19, 644kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:38<13:21, 786kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:38<09:54, 1.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:38<07:04, 1.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:39<09:07, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:40<07:28, 1.40MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:40<05:29, 1.90MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:41<06:14, 1.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:41<05:25, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:42<04:03, 2.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:43<05:14, 1.96MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:43<05:46, 1.78MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:44<04:29, 2.29MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:44<03:17, 3.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:45<06:37, 1.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:45<05:41, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:46<04:13, 2.42MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:47<05:19, 1.91MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:47<04:47, 2.12MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:48<03:36, 2.81MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:50<05:39, 1.78MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:50<12:24, 813kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:50<10:52, 928kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:50<08:11, 1.23MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:50<05:56, 1.69MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:50<04:31, 2.22MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:52<07:38, 1.31MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:52<06:22, 1.57MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:52<04:59, 2.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:52<03:37, 2.75MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:54<06:39, 1.50MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:54<08:29, 1.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:54<09:24, 1.06MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:54<10:11, 977kB/s] .vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:54<11:34, 860kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:54<11:16, 882kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:54<09:14, 1.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:54<06:48, 1.46MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:55<04:56, 2.00MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:56<06:37, 1.49MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:56<05:08, 1.92MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:56<03:59, 2.47MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:56<03:12, 3.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:58<04:36, 2.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:58<04:28, 2.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:58<03:26, 2.85MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [02:00<04:20, 2.25MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [02:00<06:02, 1.62MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [02:00<05:20, 1.82MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [02:00<04:08, 2.35MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [02:00<03:01, 3.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [02:02<06:49, 1.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [02:02<05:48, 1.67MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [02:02<04:17, 2.25MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:04<05:12, 1.84MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [02:04<05:40, 1.70MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [02:04<04:27, 2.15MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:04<03:13, 2.95MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [02:06<1:10:21, 136kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:06<50:04, 191kB/s]  .vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:06<35:10, 271kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:06<24:39, 385kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:08<41:04, 231kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:08<30:23, 312kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:08<21:45, 435kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:08<15:19, 616kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 297M/862M [02:09<14:34, 646kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:10<11:10, 841kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:10<08:03, 1.16MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:11<07:47, 1.20MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:12<06:25, 1.45MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:12<04:44, 1.97MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:14<05:45, 1.61MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:14<05:55, 1.57MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:14<04:34, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:14<03:21, 2.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:16<05:09, 1.78MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:16<04:35, 2.00MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:16<03:26, 2.66MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:18<04:31, 2.02MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:18<05:02, 1.81MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:18<04:00, 2.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:18<02:54, 3.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:20<1:15:55, 119kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:20<54:02, 168kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:20<37:57, 238kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:21<28:33, 315kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:22<21:53, 411kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:22<15:46, 570kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:22<11:05, 805kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:23<1:11:37, 125kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:24<51:01, 175kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:24<35:50, 248kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:25<27:02, 328kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:26<20:49, 425kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:26<14:56, 592kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:26<10:34, 834kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:27<10:27, 840kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:27<08:06, 1.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:28<05:53, 1.49MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:29<06:07, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:29<05:11, 1.68MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:30<03:48, 2.28MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:31<04:42, 1.84MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:31<05:07, 1.69MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:32<04:02, 2.14MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:32<02:55, 2.94MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:33<1:03:17, 136kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:33<45:10, 190kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:34<31:44, 270kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:35<24:06, 353kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:35<17:36, 483kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:35<12:29, 680kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:36<08:49, 958kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:37<22:38, 373kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:37<17:38, 479kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:37<12:42, 664kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:38<08:58, 935kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:39<09:43, 862kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:39<07:41, 1.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:39<05:32, 1.51MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:41<05:45, 1.45MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:41<05:47, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:41<04:26, 1.87MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:42<03:16, 2.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:43<04:42, 1.75MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:43<04:02, 2.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:43<03:06, 2.64MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:43<02:18, 3.55MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:45<05:24, 1.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:45<05:35, 1.46MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:45<04:20, 1.88MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:45<03:14, 2.51MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:47<04:06, 1.97MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:47<03:44, 2.16MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:47<02:49, 2.86MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:47<02:05, 3.86MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:49<09:59, 804kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:49<07:41, 1.04MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:49<05:53, 1.36MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:49<04:12, 1.90MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:51<07:38, 1.04MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:51<06:13, 1.28MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:51<04:34, 1.74MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:51<03:18, 2.39MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:53<07:20, 1.08MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:53<06:03, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:53<04:26, 1.77MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:53<03:11, 2.46MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:55<41:13, 190kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:55<29:42, 263kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:55<20:53, 373kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:55<14:41, 529kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:57<22:05, 351kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:57<16:16, 477kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:57<11:32, 670kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:59<09:57, 772kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:59<08:22, 919kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:59<06:09, 1.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:59<04:32, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [03:01<04:54, 1.55MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [03:01<04:13, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [03:01<03:07, 2.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:03<03:56, 1.91MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [03:03<04:20, 1.74MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [03:03<03:23, 2.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:03<02:29, 3.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:05<04:55, 1.52MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [03:05<04:58, 1.50MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:06<03:52, 1.93MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:06<02:47, 2.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:07<07:25, 1.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [03:07<05:57, 1.24MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:07<04:19, 1.71MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:09<04:42, 1.56MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:09<04:04, 1.81MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:09<03:00, 2.43MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:11<03:46, 1.93MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:11<03:24, 2.13MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:11<02:34, 2.82MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:13<03:35, 2.01MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:13<03:59, 1.81MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:14<03:10, 2.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:14<02:19, 3.09MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:15<04:21, 1.64MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:15<03:47, 1.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:15<02:49, 2.51MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:17<03:37, 1.96MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:17<03:15, 2.17MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:17<02:26, 2.89MB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:19<03:19, 2.11MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:19<03:03, 2.29MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:19<02:18, 3.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:21<03:11, 2.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:21<02:58, 2.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:21<02:13, 3.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:21<01:38, 4.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:23<26:34, 259kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:23<19:18, 356kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:23<13:38, 501kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:25<11:04, 614kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:25<08:27, 803kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:25<06:02, 1.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:27<05:57, 1.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:27<05:37, 1.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:27<04:16, 1.57MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:28<03:02, 2.19MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:29<17:35, 379kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:29<12:59, 512kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:29<09:12, 720kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:31<07:56, 830kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:31<06:58, 944kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:31<05:10, 1.27MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:31<03:44, 1.75MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:33<04:23, 1.48MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:33<03:45, 1.73MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:33<02:47, 2.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:35<03:26, 1.88MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:35<03:04, 2.10MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:35<02:19, 2.77MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:37<03:05, 2.06MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:37<02:49, 2.26MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:37<02:05, 3.03MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:37<01:33, 4.08MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:39<26:07, 242kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:39<18:55, 334kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:39<13:21, 471kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:41<10:45, 582kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:41<08:48, 709kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:41<06:28, 963kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:41<04:33, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:43<18:15, 339kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:43<13:19, 464kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:43<09:25, 654kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:43<06:38, 921kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:45<16:10, 378kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:45<12:33, 487kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:45<09:03, 674kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:45<06:22, 951kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:47<08:44, 691kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:47<07:22, 820kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:47<05:27, 1.10MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:47<03:51, 1.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:49<16:37, 360kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:49<12:23, 482kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:49<09:03, 659kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:49<06:31, 912kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:49<04:38, 1.27MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:51<07:01, 842kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:51<05:52, 1.00MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:51<04:18, 1.37MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:51<03:05, 1.90MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:53<05:07, 1.14MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:53<04:10, 1.40MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:53<03:02, 1.91MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:55<03:28, 1.66MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:55<03:36, 1.60MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:55<02:46, 2.07MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:55<02:01, 2.83MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:57<03:28, 1.64MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:57<04:01, 1.41MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:57<03:06, 1.83MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:57<02:13, 2.53MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:59<15:14, 370kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:59<11:15, 500kB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:59<07:57, 704kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [04:01<06:50, 814kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [04:01<05:58, 932kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [04:01<04:59, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [04:01<03:37, 1.53MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [04:03<03:44, 1.47MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:03<03:11, 1.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:03<02:26, 2.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:03<01:48, 3.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:05<03:17, 1.65MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [04:05<03:02, 1.78MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:05<02:17, 2.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:05<01:39, 3.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:07<11:22, 472kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:07<09:03, 592kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:07<06:35, 810kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:07<04:38, 1.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:09<41:01, 129kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:09<29:14, 181kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:09<20:30, 257kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:11<15:28, 338kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:11<11:16, 463kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:11<07:58, 652kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:11<05:38, 917kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:13<08:37, 597kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:13<06:33, 785kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:13<04:42, 1.09MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:15<04:27, 1.14MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:15<03:38, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:15<02:40, 1.89MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:17<03:01, 1.66MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:17<02:38, 1.90MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:17<01:56, 2.56MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:19<02:43, 1.82MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:19<04:07, 1.20MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:19<03:24, 1.45MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:19<02:29, 1.97MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:21<02:47, 1.74MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:21<02:28, 1.97MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:21<01:50, 2.63MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:23<02:23, 2.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:23<02:05, 2.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:23<01:36, 2.99MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:23<01:10, 4.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:25<12:15, 387kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:25<09:35, 494kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:25<06:56, 681kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:25<04:52, 960kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:27<36:44, 127kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:27<26:05, 179kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:27<18:16, 254kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:27<12:45, 361kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:29<17:56, 257kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:29<13:32, 340kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:29<09:39, 475kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:29<06:49, 669kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:31<05:57, 762kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:31<04:33, 995kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:31<03:20, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:31<02:22, 1.89MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:33<12:29, 358kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:33<09:41, 461kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:33<06:59, 636kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:33<04:53, 901kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:35<10:44, 410kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:35<07:58, 551kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:35<05:39, 773kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:36<04:55, 880kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:37<04:22, 990kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:37<03:16, 1.32MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:37<02:19, 1.84MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:38<31:59, 133kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:39<22:48, 187kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:39<15:58, 265kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:40<12:03, 348kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:41<08:52, 472kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:41<06:16, 663kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:42<05:18, 777kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:43<04:07, 997kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 616M/862M [04:43<02:58, 1.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:44<03:01, 1.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:44<02:32, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:45<01:52, 2.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:46<02:13, 1.79MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:46<02:00, 1.99MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:47<01:30, 2.63MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:48<01:53, 2.07MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:48<01:45, 2.24MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:48<01:19, 2.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:49<00:58, 3.97MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:50<04:54, 785kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:50<04:13, 910kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:51<03:08, 1.22MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:51<02:12, 1.71MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:52<1:51:46, 33.9kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:52<1:18:29, 48.1kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:52<54:40, 68.6kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:54<38:46, 95.8kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:54<27:28, 135kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:54<19:10, 192kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:56<14:09, 258kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:56<10:34, 345kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:56<07:39, 475kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:56<05:25, 667kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:57<03:48, 941kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:58<13:21, 268kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:58<09:48, 364kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:58<06:55, 513kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:58<04:49, 727kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [05:00<15:01, 233kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [05:00<10:52, 322kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [05:00<07:38, 455kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:02<06:05, 565kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [05:02<04:58, 691kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [05:02<03:37, 945kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [05:02<02:34, 1.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:04<02:54, 1.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:04<02:23, 1.41MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [05:04<01:44, 1.91MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:06<01:58, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:06<01:40, 1.97MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:06<01:14, 2.63MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:06<00:54, 3.58MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:08<13:36, 238kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [05:08<09:50, 328kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:08<06:54, 464kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:10<05:31, 573kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:10<04:10, 755kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:10<02:58, 1.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:12<02:47, 1.11MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:12<02:16, 1.36MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:12<01:39, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:14<01:50, 1.64MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:14<01:36, 1.88MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:14<01:11, 2.51MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:16<01:30, 1.96MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:16<01:19, 2.24MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:16<00:58, 3.01MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:16<00:43, 4.04MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:18<07:46, 372kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:18<06:03, 477kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:18<04:20, 661kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:18<03:03, 929kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:20<03:02, 929kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:20<02:22, 1.19MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:20<01:48, 1.55MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:20<01:16, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:22<03:44, 738kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:22<03:50, 715kB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:22<02:59, 920kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:22<02:07, 1.28MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:24<02:05, 1.28MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:24<01:41, 1.59MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:24<01:20, 1.99MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:24<01:00, 2.64MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:24<00:47, 3.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:25<00:38, 4.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:26<05:04, 516kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:26<04:09, 629kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:26<03:14, 805kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:27<02:21, 1.10MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:27<01:39, 1.54MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:28<02:40, 952kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:28<02:23, 1.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:29<01:57, 1.30MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:29<01:26, 1.75MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:29<01:01, 2.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:32<06:19, 391kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:32<06:57, 356kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:32<05:42, 434kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:32<04:18, 575kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:32<03:05, 796kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:32<02:10, 1.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:34<02:25, 997kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:34<02:03, 1.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:34<01:39, 1.45MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:34<01:15, 1.90MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:34<00:56, 2.54MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:34<00:41, 3.38MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:36<05:20, 439kB/s] .vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:36<04:12, 555kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:36<03:17, 710kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:36<02:31, 922kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:36<01:49, 1.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:36<01:18, 1.75MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:38<01:54, 1.19MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:38<01:46, 1.27MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:38<01:30, 1.50MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:38<01:10, 1.92MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:38<00:51, 2.60MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:40<01:13, 1.80MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:40<01:17, 1.71MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:40<01:05, 2.01MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:40<00:52, 2.49MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:40<00:40, 3.25MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:40<00:29, 4.30MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:42<05:46, 370kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:42<04:42, 453kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:42<03:24, 622kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:42<02:28, 856kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:44<02:02, 1.01MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:44<01:41, 1.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:44<01:17, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:44<00:55, 2.19MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:46<01:15, 1.59MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:46<01:05, 1.83MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:46<00:47, 2.48MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:46<00:34, 3.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:48<10:48, 178kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:48<07:52, 244kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:48<05:34, 343kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:48<03:53, 485kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:50<03:10, 586kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:50<02:24, 768kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:50<01:42, 1.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:50<01:11, 1.50MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:52<07:28, 240kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:52<05:21, 333kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:52<03:50, 462kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:52<02:40, 652kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:54<02:26, 706kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:54<01:51, 925kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:54<01:52, 916kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:54<01:21, 1.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:54<01:01, 1.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:54<00:48, 2.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:54<00:37, 2.63MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:54<00:31, 3.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:56<03:28, 476kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:56<03:11, 519kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:56<02:25, 680kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:56<01:45, 930kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:56<01:16, 1.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:56<00:57, 1.68MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:56<00:43, 2.18MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:58<02:05, 755kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:58<02:12, 715kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:58<01:52, 841kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:58<01:23, 1.13MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:58<01:02, 1.49MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:58<00:47, 1.97MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:58<00:36, 2.52MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:58<00:28, 3.20MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [06:00<03:34, 424kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [06:01<04:02, 375kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [06:01<03:08, 482kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [06:01<02:22, 633kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [06:01<01:42, 871kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [06:01<01:13, 1.20MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [06:01<00:55, 1.58MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [06:03<01:13, 1.19MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [06:03<01:27, 992kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [06:03<01:09, 1.24MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [06:03<00:51, 1.66MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [06:03<00:38, 2.18MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [06:03<00:29, 2.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [06:05<03:10, 435kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [06:06<03:40, 375kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [06:06<03:01, 454kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [06:06<02:20, 587kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [06:06<01:44, 785kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [06:06<01:16, 1.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [06:06<00:55, 1.45MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:06<00:41, 1.91MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [06:06<00:31, 2.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:08<05:00, 261kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:08<04:51, 269kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:09<03:50, 340kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:09<02:56, 442kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:09<02:26, 533kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:09<02:01, 640kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:09<01:51, 701kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:09<01:47, 727kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [06:09<01:42, 761kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:09<01:37, 794kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:09<01:35, 815kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:09<01:34, 817kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:10<01:30, 850kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:10<01:27, 883kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:10<01:23, 922kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:10<01:20, 952kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:10<01:13, 1.05MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:10<00:55, 1.37MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [06:10<00:41, 1.81MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [06:10<00:31, 2.36MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:11<00:50, 1.48MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [06:11<00:41, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:11<00:34, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [06:11<00:32, 2.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:11<00:31, 2.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:11<00:27, 2.62MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:11<00:26, 2.76MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:12<00:24, 3.00MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:12<00:19, 3.63MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:12<00:15, 4.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:13<02:20, 499kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [06:13<01:51, 629kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:13<01:21, 857kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:13<00:58, 1.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:13<00:42, 1.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:13<00:31, 2.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:15<00:59, 1.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:15<01:05, 1.01MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [06:15<00:54, 1.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:15<00:40, 1.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:15<00:31, 2.06MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:15<00:24, 2.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:15<00:18, 3.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:16<00:15, 4.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:17<11:49, 87.4kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:17<08:38, 119kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:17<06:25, 160kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [06:17<04:48, 214kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:17<03:41, 278kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:17<02:49, 362kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:17<02:11, 466kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:17<01:50, 555kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:18<01:35, 641kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:18<01:18, 775kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:18<00:57, 1.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:18<00:42, 1.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:18<00:31, 1.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:20<01:01, 938kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:20<01:41, 567kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:20<01:25, 676kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:20<01:02, 915kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:21<00:45, 1.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:21<00:32, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:21<00:24, 2.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:21<00:20, 2.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:22<01:56, 463kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:22<01:29, 595kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:22<01:04, 816kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:22<00:45, 1.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:23<00:32, 1.53MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:24<01:02, 794kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:24<01:20, 613kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:24<01:10, 702kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:24<01:01, 803kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:24<00:52, 933kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:25<00:49, 981kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:25<00:47, 1.03MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:25<00:45, 1.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:25<00:36, 1.33MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:25<00:26, 1.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:25<00:19, 2.35MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:25<00:15, 2.89MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:25<00:16, 2.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:25<00:16, 2.72MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:27<09:21, 80.9kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:27<07:00, 108kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:27<05:00, 150kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:27<03:27, 213kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:27<02:21, 302kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:28<01:38, 425kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:29<01:32, 446kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:29<01:20, 512kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:29<01:07, 612kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:29<00:56, 723kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:29<00:49, 830kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:29<00:45, 899kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:30<00:39, 1.03MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:30<00:35, 1.15MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:30<00:33, 1.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:30<00:30, 1.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:30<00:28, 1.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:30<00:23, 1.71MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:30<00:18, 2.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:30<00:15, 2.47MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:30<00:16, 2.40MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:30<00:16, 2.36MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:31<00:15, 2.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:31<00:15, 2.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:31<00:15, 2.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:31<00:16, 2.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:31<00:15, 2.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:31<00:14, 2.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:31<00:13, 2.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:31<00:12, 2.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:31<00:11, 3.17MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:32<00:09, 3.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:32<00:08, 4.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:33<00:15, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:33<00:18, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:33<00:14, 2.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:33<00:10, 3.00MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:33<00:07, 3.91MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:35<01:16, 378kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:35<01:03, 456kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:35<00:51, 554kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:35<00:37, 747kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:35<00:26, 1.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:35<00:17, 1.44MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:37<00:33, 732kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:38<00:48, 512kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:38<00:39, 624kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:38<00:28, 845kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:38<00:19, 1.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:40<00:19, 1.06MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:40<00:34, 603kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:40<00:27, 737kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:41<00:21, 927kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:41<00:15, 1.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:41<00:10, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:41<00:07, 2.33MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:43<00:43, 376kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:43<00:47, 350kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:43<00:35, 454kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:44<00:25, 623kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:44<00:17, 867kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:44<00:10, 1.21MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:45<00:14, 849kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:46<00:13, 912kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:46<00:09, 1.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:46<00:05, 1.65MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:48<00:07, 1.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:48<00:14, 580kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:49<00:12, 634kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:49<00:11, 702kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:49<00:10, 763kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:49<00:08, 913kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:49<00:06, 1.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:49<00:04, 1.51MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:49<00:03, 1.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:49<00:02, 2.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:49<00:02, 2.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:49<00:01, 3.18MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:50<00:01, 3.70MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:50<00:04, 888kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:50<00:03, 1.14MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:50<00:02, 1.48MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:51<00:01, 1.86MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:51<00:00, 2.32MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:51<00:00, 2.84MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:51<00:00, 3.56MB/s].vector_cache/glove.6B.zip: 862MB [06:51, 2.10MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 733/400000 [00:00<00:54, 7328.73it/s]  0%|          | 1451/400000 [00:00<00:54, 7282.64it/s]  1%|          | 2206/400000 [00:00<00:54, 7360.31it/s]  1%|          | 2969/400000 [00:00<00:53, 7437.24it/s]  1%|          | 3727/400000 [00:00<00:52, 7477.16it/s]  1%|          | 4510/400000 [00:00<00:52, 7578.78it/s]  1%|▏         | 5279/400000 [00:00<00:51, 7611.14it/s]  2%|▏         | 6016/400000 [00:00<00:52, 7536.79it/s]  2%|▏         | 6766/400000 [00:00<00:52, 7523.97it/s]  2%|▏         | 7538/400000 [00:01<00:51, 7580.28it/s]  2%|▏         | 8278/400000 [00:01<00:52, 7523.58it/s]  2%|▏         | 9017/400000 [00:01<00:52, 7432.69it/s]  2%|▏         | 9757/400000 [00:01<00:52, 7420.89it/s]  3%|▎         | 10493/400000 [00:01<00:52, 7369.48it/s]  3%|▎         | 11226/400000 [00:01<00:53, 7330.24it/s]  3%|▎         | 11978/400000 [00:01<00:52, 7383.81it/s]  3%|▎         | 12715/400000 [00:01<00:54, 7150.72it/s]  3%|▎         | 13434/400000 [00:01<00:53, 7159.94it/s]  4%|▎         | 14208/400000 [00:01<00:52, 7322.71it/s]  4%|▎         | 14982/400000 [00:02<00:51, 7442.00it/s]  4%|▍         | 15775/400000 [00:02<00:50, 7580.62it/s]  4%|▍         | 16564/400000 [00:02<00:50, 7668.54it/s]  4%|▍         | 17333/400000 [00:02<00:49, 7658.63it/s]  5%|▍         | 18129/400000 [00:02<00:49, 7745.25it/s]  5%|▍         | 18905/400000 [00:02<00:50, 7588.73it/s]  5%|▍         | 19666/400000 [00:02<00:50, 7558.97it/s]  5%|▌         | 20423/400000 [00:02<00:50, 7559.13it/s]  5%|▌         | 21183/400000 [00:02<00:50, 7569.22it/s]  5%|▌         | 21955/400000 [00:02<00:49, 7611.99it/s]  6%|▌         | 22737/400000 [00:03<00:49, 7670.90it/s]  6%|▌         | 23522/400000 [00:03<00:48, 7722.30it/s]  6%|▌         | 24308/400000 [00:03<00:48, 7763.09it/s]  6%|▋         | 25095/400000 [00:03<00:48, 7793.05it/s]  6%|▋         | 25875/400000 [00:03<00:49, 7528.17it/s]  7%|▋         | 26634/400000 [00:03<00:49, 7544.76it/s]  7%|▋         | 27393/400000 [00:03<00:49, 7555.74it/s]  7%|▋         | 28172/400000 [00:03<00:48, 7621.65it/s]  7%|▋         | 28936/400000 [00:03<00:49, 7565.00it/s]  7%|▋         | 29694/400000 [00:03<00:49, 7555.27it/s]  8%|▊         | 30499/400000 [00:04<00:48, 7696.48it/s]  8%|▊         | 31284/400000 [00:04<00:47, 7741.18it/s]  8%|▊         | 32059/400000 [00:04<00:48, 7584.73it/s]  8%|▊         | 32819/400000 [00:04<00:48, 7549.03it/s]  8%|▊         | 33617/400000 [00:04<00:47, 7672.30it/s]  9%|▊         | 34413/400000 [00:04<00:47, 7755.52it/s]  9%|▉         | 35190/400000 [00:04<00:47, 7676.34it/s]  9%|▉         | 35959/400000 [00:04<00:47, 7640.09it/s]  9%|▉         | 36742/400000 [00:04<00:47, 7692.67it/s]  9%|▉         | 37512/400000 [00:04<00:47, 7600.67it/s] 10%|▉         | 38296/400000 [00:05<00:47, 7668.63it/s] 10%|▉         | 39088/400000 [00:05<00:46, 7741.66it/s] 10%|▉         | 39867/400000 [00:05<00:46, 7754.70it/s] 10%|█         | 40643/400000 [00:05<00:47, 7603.70it/s] 10%|█         | 41405/400000 [00:05<00:47, 7604.41it/s] 11%|█         | 42204/400000 [00:05<00:46, 7715.70it/s] 11%|█         | 42977/400000 [00:05<00:46, 7710.28it/s] 11%|█         | 43799/400000 [00:05<00:45, 7854.81it/s] 11%|█         | 44586/400000 [00:05<00:46, 7698.74it/s] 11%|█▏        | 45358/400000 [00:05<00:46, 7627.73it/s] 12%|█▏        | 46163/400000 [00:06<00:45, 7748.23it/s] 12%|█▏        | 46940/400000 [00:06<00:46, 7668.51it/s] 12%|█▏        | 47745/400000 [00:06<00:45, 7775.94it/s] 12%|█▏        | 48535/400000 [00:06<00:45, 7810.28it/s] 12%|█▏        | 49317/400000 [00:06<00:44, 7810.43it/s] 13%|█▎        | 50099/400000 [00:06<00:44, 7809.81it/s] 13%|█▎        | 50881/400000 [00:06<00:46, 7554.50it/s] 13%|█▎        | 51655/400000 [00:06<00:45, 7607.88it/s] 13%|█▎        | 52433/400000 [00:06<00:45, 7657.91it/s] 13%|█▎        | 53200/400000 [00:06<00:45, 7603.73it/s] 13%|█▎        | 53987/400000 [00:07<00:45, 7681.65it/s] 14%|█▎        | 54765/400000 [00:07<00:44, 7710.55it/s] 14%|█▍        | 55537/400000 [00:07<00:45, 7579.07it/s] 14%|█▍        | 56324/400000 [00:07<00:44, 7661.04it/s] 14%|█▍        | 57091/400000 [00:07<00:45, 7462.78it/s] 14%|█▍        | 57840/400000 [00:07<00:46, 7376.18it/s] 15%|█▍        | 58591/400000 [00:07<00:46, 7414.80it/s] 15%|█▍        | 59383/400000 [00:07<00:45, 7556.83it/s] 15%|█▌        | 60141/400000 [00:07<00:45, 7511.12it/s] 15%|█▌        | 60918/400000 [00:08<00:44, 7586.39it/s] 15%|█▌        | 61723/400000 [00:08<00:43, 7717.89it/s] 16%|█▌        | 62505/400000 [00:08<00:43, 7746.90it/s] 16%|█▌        | 63296/400000 [00:08<00:43, 7794.44it/s] 16%|█▌        | 64095/400000 [00:08<00:42, 7850.35it/s] 16%|█▌        | 64881/400000 [00:08<00:42, 7820.03it/s] 16%|█▋        | 65666/400000 [00:08<00:42, 7828.34it/s] 17%|█▋        | 66450/400000 [00:08<00:42, 7831.56it/s] 17%|█▋        | 67249/400000 [00:08<00:42, 7877.46it/s] 17%|█▋        | 68037/400000 [00:08<00:42, 7765.13it/s] 17%|█▋        | 68815/400000 [00:09<00:43, 7689.33it/s] 17%|█▋        | 69616/400000 [00:09<00:42, 7782.62it/s] 18%|█▊        | 70395/400000 [00:09<00:42, 7696.40it/s] 18%|█▊        | 71179/400000 [00:09<00:42, 7736.29it/s] 18%|█▊        | 71954/400000 [00:09<00:42, 7718.02it/s] 18%|█▊        | 72731/400000 [00:09<00:42, 7733.25it/s] 18%|█▊        | 73527/400000 [00:09<00:41, 7798.07it/s] 19%|█▊        | 74318/400000 [00:09<00:41, 7828.94it/s] 19%|█▉        | 75110/400000 [00:09<00:41, 7854.57it/s] 19%|█▉        | 75896/400000 [00:09<00:41, 7801.40it/s] 19%|█▉        | 76677/400000 [00:10<00:42, 7660.34it/s] 19%|█▉        | 77451/400000 [00:10<00:41, 7683.69it/s] 20%|█▉        | 78230/400000 [00:10<00:41, 7713.67it/s] 20%|█▉        | 79034/400000 [00:10<00:41, 7807.96it/s] 20%|█▉        | 79816/400000 [00:10<00:41, 7775.36it/s] 20%|██        | 80594/400000 [00:10<00:41, 7719.34it/s] 20%|██        | 81373/400000 [00:10<00:41, 7738.85it/s] 21%|██        | 82183/400000 [00:10<00:40, 7841.59it/s] 21%|██        | 82968/400000 [00:10<00:40, 7774.43it/s] 21%|██        | 83746/400000 [00:10<00:41, 7579.17it/s] 21%|██        | 84509/400000 [00:11<00:41, 7593.98it/s] 21%|██▏       | 85297/400000 [00:11<00:41, 7675.21it/s] 22%|██▏       | 86085/400000 [00:11<00:40, 7733.62it/s] 22%|██▏       | 86860/400000 [00:11<00:40, 7692.67it/s] 22%|██▏       | 87630/400000 [00:11<00:40, 7688.07it/s] 22%|██▏       | 88400/400000 [00:11<00:40, 7634.77it/s] 22%|██▏       | 89164/400000 [00:11<00:40, 7583.21it/s] 22%|██▏       | 89923/400000 [00:11<00:42, 7298.17it/s] 23%|██▎       | 90656/400000 [00:11<00:42, 7255.18it/s] 23%|██▎       | 91413/400000 [00:11<00:42, 7345.99it/s] 23%|██▎       | 92152/400000 [00:12<00:41, 7356.38it/s] 23%|██▎       | 92920/400000 [00:12<00:41, 7448.52it/s] 23%|██▎       | 93686/400000 [00:12<00:40, 7508.33it/s] 24%|██▎       | 94455/400000 [00:12<00:40, 7558.28it/s] 24%|██▍       | 95212/400000 [00:12<00:40, 7495.78it/s] 24%|██▍       | 95963/400000 [00:12<00:40, 7452.66it/s] 24%|██▍       | 96739/400000 [00:12<00:40, 7541.47it/s] 24%|██▍       | 97526/400000 [00:12<00:39, 7634.85it/s] 25%|██▍       | 98302/400000 [00:12<00:39, 7669.18it/s] 25%|██▍       | 99073/400000 [00:12<00:39, 7680.18it/s] 25%|██▍       | 99842/400000 [00:13<00:40, 7463.86it/s] 25%|██▌       | 100599/400000 [00:13<00:39, 7494.54it/s] 25%|██▌       | 101358/400000 [00:13<00:39, 7522.34it/s] 26%|██▌       | 102135/400000 [00:13<00:39, 7594.75it/s] 26%|██▌       | 102896/400000 [00:13<00:39, 7462.34it/s] 26%|██▌       | 103644/400000 [00:13<00:40, 7356.87it/s] 26%|██▌       | 104381/400000 [00:13<00:40, 7342.66it/s] 26%|██▋       | 105151/400000 [00:13<00:39, 7446.02it/s] 26%|██▋       | 105912/400000 [00:13<00:39, 7491.89it/s] 27%|██▋       | 106672/400000 [00:14<00:39, 7515.28it/s] 27%|██▋       | 107434/400000 [00:14<00:38, 7546.25it/s] 27%|██▋       | 108199/400000 [00:14<00:38, 7574.21it/s] 27%|██▋       | 109004/400000 [00:14<00:37, 7710.13it/s] 27%|██▋       | 109785/400000 [00:14<00:37, 7738.92it/s] 28%|██▊       | 110560/400000 [00:14<00:38, 7487.68it/s] 28%|██▊       | 111311/400000 [00:14<00:38, 7415.40it/s] 28%|██▊       | 112076/400000 [00:14<00:38, 7481.71it/s] 28%|██▊       | 112848/400000 [00:14<00:38, 7551.02it/s] 28%|██▊       | 113625/400000 [00:14<00:37, 7612.19it/s] 29%|██▊       | 114393/400000 [00:15<00:37, 7630.22it/s] 29%|██▉       | 115158/400000 [00:15<00:37, 7633.87it/s] 29%|██▉       | 115951/400000 [00:15<00:36, 7718.17it/s] 29%|██▉       | 116724/400000 [00:15<00:37, 7643.72it/s] 29%|██▉       | 117530/400000 [00:15<00:36, 7762.76it/s] 30%|██▉       | 118311/400000 [00:15<00:36, 7776.15it/s] 30%|██▉       | 119090/400000 [00:15<00:36, 7728.91it/s] 30%|██▉       | 119915/400000 [00:15<00:35, 7876.83it/s] 30%|███       | 120704/400000 [00:15<00:35, 7778.62it/s] 30%|███       | 121497/400000 [00:15<00:35, 7821.29it/s] 31%|███       | 122284/400000 [00:16<00:35, 7835.15it/s] 31%|███       | 123069/400000 [00:16<00:36, 7559.41it/s] 31%|███       | 123895/400000 [00:16<00:35, 7755.98it/s] 31%|███       | 124694/400000 [00:16<00:35, 7822.59it/s] 31%|███▏      | 125482/400000 [00:16<00:35, 7839.06it/s] 32%|███▏      | 126268/400000 [00:16<00:35, 7787.49it/s] 32%|███▏      | 127048/400000 [00:16<00:35, 7725.83it/s] 32%|███▏      | 127850/400000 [00:16<00:34, 7810.10it/s] 32%|███▏      | 128632/400000 [00:16<00:34, 7808.73it/s] 32%|███▏      | 129414/400000 [00:16<00:34, 7762.74it/s] 33%|███▎      | 130191/400000 [00:17<00:34, 7743.31it/s] 33%|███▎      | 130966/400000 [00:17<00:35, 7575.40it/s] 33%|███▎      | 131756/400000 [00:17<00:34, 7669.09it/s] 33%|███▎      | 132546/400000 [00:17<00:34, 7734.75it/s] 33%|███▎      | 133327/400000 [00:17<00:34, 7756.21it/s] 34%|███▎      | 134104/400000 [00:17<00:34, 7729.92it/s] 34%|███▎      | 134878/400000 [00:17<00:34, 7667.71it/s] 34%|███▍      | 135668/400000 [00:17<00:34, 7733.56it/s] 34%|███▍      | 136453/400000 [00:17<00:33, 7765.94it/s] 34%|███▍      | 137230/400000 [00:17<00:33, 7761.17it/s] 35%|███▍      | 138035/400000 [00:18<00:33, 7842.98it/s] 35%|███▍      | 138820/400000 [00:18<00:33, 7713.88it/s] 35%|███▍      | 139605/400000 [00:18<00:33, 7752.60it/s] 35%|███▌      | 140397/400000 [00:18<00:33, 7800.36it/s] 35%|███▌      | 141188/400000 [00:18<00:33, 7830.49it/s] 35%|███▌      | 141993/400000 [00:18<00:32, 7892.66it/s] 36%|███▌      | 142783/400000 [00:18<00:32, 7797.80it/s] 36%|███▌      | 143566/400000 [00:18<00:32, 7805.93it/s] 36%|███▌      | 144347/400000 [00:18<00:32, 7781.22it/s] 36%|███▋      | 145126/400000 [00:18<00:32, 7780.64it/s] 36%|███▋      | 145911/400000 [00:19<00:32, 7801.29it/s] 37%|███▋      | 146692/400000 [00:19<00:32, 7730.19it/s] 37%|███▋      | 147466/400000 [00:19<00:32, 7711.87it/s] 37%|███▋      | 148246/400000 [00:19<00:32, 7736.97it/s] 37%|███▋      | 149020/400000 [00:19<00:32, 7643.11it/s] 37%|███▋      | 149799/400000 [00:19<00:32, 7684.41it/s] 38%|███▊      | 150568/400000 [00:19<00:32, 7659.12it/s] 38%|███▊      | 151335/400000 [00:19<00:32, 7634.09it/s] 38%|███▊      | 152109/400000 [00:19<00:32, 7662.63it/s] 38%|███▊      | 152876/400000 [00:19<00:32, 7632.79it/s] 38%|███▊      | 153644/400000 [00:20<00:32, 7643.34it/s] 39%|███▊      | 154409/400000 [00:20<00:32, 7616.10it/s] 39%|███▉      | 155189/400000 [00:20<00:31, 7668.77it/s] 39%|███▉      | 155977/400000 [00:20<00:31, 7728.66it/s] 39%|███▉      | 156763/400000 [00:20<00:31, 7767.58it/s] 39%|███▉      | 157547/400000 [00:20<00:31, 7787.33it/s] 40%|███▉      | 158326/400000 [00:20<00:31, 7635.97it/s] 40%|███▉      | 159091/400000 [00:20<00:31, 7630.28it/s] 40%|███▉      | 159855/400000 [00:20<00:31, 7608.51it/s] 40%|████      | 160627/400000 [00:21<00:31, 7640.25it/s] 40%|████      | 161399/400000 [00:21<00:31, 7662.20it/s] 41%|████      | 162166/400000 [00:21<00:31, 7663.57it/s] 41%|████      | 162948/400000 [00:21<00:30, 7708.24it/s] 41%|████      | 163722/400000 [00:21<00:30, 7714.91it/s] 41%|████      | 164514/400000 [00:21<00:30, 7774.78it/s] 41%|████▏     | 165295/400000 [00:21<00:30, 7784.94it/s] 42%|████▏     | 166074/400000 [00:21<00:30, 7657.07it/s] 42%|████▏     | 166885/400000 [00:21<00:29, 7785.60it/s] 42%|████▏     | 167665/400000 [00:21<00:29, 7786.96it/s] 42%|████▏     | 168449/400000 [00:22<00:29, 7800.56it/s] 42%|████▏     | 169230/400000 [00:22<00:29, 7799.57it/s] 43%|████▎     | 170011/400000 [00:22<00:30, 7641.85it/s] 43%|████▎     | 170786/400000 [00:22<00:29, 7673.92it/s] 43%|████▎     | 171568/400000 [00:22<00:29, 7715.79it/s] 43%|████▎     | 172350/400000 [00:22<00:29, 7745.30it/s] 43%|████▎     | 173125/400000 [00:22<00:29, 7604.16it/s] 43%|████▎     | 173890/400000 [00:22<00:29, 7616.01it/s] 44%|████▎     | 174703/400000 [00:22<00:29, 7763.11it/s] 44%|████▍     | 175481/400000 [00:22<00:29, 7708.63it/s] 44%|████▍     | 176280/400000 [00:23<00:28, 7789.68it/s] 44%|████▍     | 177073/400000 [00:23<00:28, 7828.67it/s] 44%|████▍     | 177857/400000 [00:23<00:28, 7740.35it/s] 45%|████▍     | 178638/400000 [00:23<00:28, 7758.84it/s] 45%|████▍     | 179426/400000 [00:23<00:28, 7793.68it/s] 45%|████▌     | 180206/400000 [00:23<00:28, 7762.17it/s] 45%|████▌     | 180983/400000 [00:23<00:28, 7639.72it/s] 45%|████▌     | 181748/400000 [00:23<00:28, 7634.07it/s] 46%|████▌     | 182551/400000 [00:23<00:28, 7748.48it/s] 46%|████▌     | 183343/400000 [00:23<00:27, 7797.45it/s] 46%|████▌     | 184138/400000 [00:24<00:27, 7840.59it/s] 46%|████▌     | 184936/400000 [00:24<00:27, 7880.08it/s] 46%|████▋     | 185725/400000 [00:24<00:27, 7778.19it/s] 47%|████▋     | 186535/400000 [00:24<00:27, 7869.45it/s] 47%|████▋     | 187323/400000 [00:24<00:27, 7858.79it/s] 47%|████▋     | 188110/400000 [00:24<00:26, 7856.35it/s] 47%|████▋     | 188896/400000 [00:24<00:27, 7704.78it/s] 47%|████▋     | 189668/400000 [00:24<00:27, 7619.27it/s] 48%|████▊     | 190447/400000 [00:24<00:27, 7669.01it/s] 48%|████▊     | 191223/400000 [00:24<00:27, 7694.55it/s] 48%|████▊     | 191999/400000 [00:25<00:26, 7712.25it/s] 48%|████▊     | 192800/400000 [00:25<00:26, 7797.35it/s] 48%|████▊     | 193581/400000 [00:25<00:26, 7699.20it/s] 49%|████▊     | 194352/400000 [00:25<00:26, 7670.22it/s] 49%|████▉     | 195120/400000 [00:25<00:26, 7599.51it/s] 49%|████▉     | 195881/400000 [00:25<00:26, 7581.18it/s] 49%|████▉     | 196647/400000 [00:25<00:26, 7604.08it/s] 49%|████▉     | 197408/400000 [00:25<00:27, 7490.14it/s] 50%|████▉     | 198158/400000 [00:25<00:26, 7493.05it/s] 50%|████▉     | 198908/400000 [00:25<00:26, 7488.45it/s] 50%|████▉     | 199670/400000 [00:26<00:26, 7526.61it/s] 50%|█████     | 200423/400000 [00:26<00:27, 7287.69it/s] 50%|█████     | 201175/400000 [00:26<00:27, 7354.83it/s] 50%|█████     | 201912/400000 [00:26<00:27, 7333.51it/s] 51%|█████     | 202708/400000 [00:26<00:26, 7507.70it/s] 51%|█████     | 203493/400000 [00:26<00:25, 7604.60it/s] 51%|█████     | 204256/400000 [00:26<00:25, 7611.82it/s] 51%|█████▏    | 205023/400000 [00:26<00:25, 7627.82it/s] 51%|█████▏    | 205787/400000 [00:26<00:26, 7450.48it/s] 52%|█████▏    | 206560/400000 [00:26<00:25, 7531.57it/s] 52%|█████▏    | 207343/400000 [00:27<00:25, 7616.54it/s] 52%|█████▏    | 208127/400000 [00:27<00:24, 7681.11it/s] 52%|█████▏    | 208900/400000 [00:27<00:24, 7695.17it/s] 52%|█████▏    | 209686/400000 [00:27<00:24, 7743.20it/s] 53%|█████▎    | 210461/400000 [00:27<00:24, 7692.46it/s] 53%|█████▎    | 211234/400000 [00:27<00:24, 7700.94it/s] 53%|█████▎    | 212005/400000 [00:27<00:24, 7667.65it/s] 53%|█████▎    | 212773/400000 [00:27<00:24, 7644.90it/s] 53%|█████▎    | 213549/400000 [00:27<00:24, 7675.97it/s] 54%|█████▎    | 214323/400000 [00:27<00:24, 7694.36it/s] 54%|█████▍    | 215116/400000 [00:28<00:23, 7762.81it/s] 54%|█████▍    | 215893/400000 [00:28<00:24, 7655.28it/s] 54%|█████▍    | 216660/400000 [00:28<00:23, 7643.56it/s] 54%|█████▍    | 217454/400000 [00:28<00:23, 7728.04it/s] 55%|█████▍    | 218239/400000 [00:28<00:23, 7761.92it/s] 55%|█████▍    | 219016/400000 [00:28<00:23, 7727.63it/s] 55%|█████▍    | 219790/400000 [00:28<00:23, 7713.84it/s] 55%|█████▌    | 220562/400000 [00:28<00:23, 7705.74it/s] 55%|█████▌    | 221358/400000 [00:28<00:22, 7779.00it/s] 56%|█████▌    | 222137/400000 [00:29<00:22, 7741.17it/s] 56%|█████▌    | 222912/400000 [00:29<00:22, 7728.30it/s] 56%|█████▌    | 223685/400000 [00:29<00:22, 7715.61it/s] 56%|█████▌    | 224457/400000 [00:29<00:22, 7664.37it/s] 56%|█████▋    | 225234/400000 [00:29<00:22, 7695.53it/s] 57%|█████▋    | 226010/400000 [00:29<00:22, 7713.93it/s] 57%|█████▋    | 226782/400000 [00:29<00:22, 7680.09it/s] 57%|█████▋    | 227558/400000 [00:29<00:22, 7701.43it/s] 57%|█████▋    | 228329/400000 [00:29<00:22, 7630.92it/s] 57%|█████▋    | 229109/400000 [00:29<00:22, 7680.26it/s] 57%|█████▋    | 229886/400000 [00:30<00:22, 7705.98it/s] 58%|█████▊    | 230658/400000 [00:30<00:21, 7709.26it/s] 58%|█████▊    | 231447/400000 [00:30<00:21, 7761.50it/s] 58%|█████▊    | 232224/400000 [00:30<00:21, 7704.81it/s] 58%|█████▊    | 233026/400000 [00:30<00:21, 7794.50it/s] 58%|█████▊    | 233810/400000 [00:30<00:21, 7806.11it/s] 59%|█████▊    | 234593/400000 [00:30<00:21, 7812.21it/s] 59%|█████▉    | 235411/400000 [00:30<00:20, 7918.30it/s] 59%|█████▉    | 236204/400000 [00:30<00:21, 7691.65it/s] 59%|█████▉    | 236985/400000 [00:30<00:21, 7726.53it/s] 59%|█████▉    | 237759/400000 [00:31<00:21, 7723.61it/s] 60%|█████▉    | 238537/400000 [00:31<00:20, 7737.94it/s] 60%|█████▉    | 239325/400000 [00:31<00:20, 7776.66it/s] 60%|██████    | 240104/400000 [00:31<00:20, 7614.98it/s] 60%|██████    | 240867/400000 [00:31<00:20, 7613.70it/s] 60%|██████    | 241639/400000 [00:31<00:20, 7644.92it/s] 61%|██████    | 242405/400000 [00:31<00:21, 7494.01it/s] 61%|██████    | 243193/400000 [00:31<00:20, 7604.60it/s] 61%|██████    | 243967/400000 [00:31<00:20, 7642.11it/s] 61%|██████    | 244750/400000 [00:31<00:20, 7697.27it/s] 61%|██████▏   | 245550/400000 [00:32<00:19, 7785.09it/s] 62%|██████▏   | 246338/400000 [00:32<00:19, 7810.63it/s] 62%|██████▏   | 247142/400000 [00:32<00:19, 7875.37it/s] 62%|██████▏   | 247931/400000 [00:32<00:19, 7826.55it/s] 62%|██████▏   | 248715/400000 [00:32<00:19, 7677.60it/s] 62%|██████▏   | 249484/400000 [00:32<00:19, 7546.88it/s] 63%|██████▎   | 250240/400000 [00:32<00:19, 7500.05it/s] 63%|██████▎   | 251018/400000 [00:32<00:19, 7580.10it/s] 63%|██████▎   | 251780/400000 [00:32<00:19, 7591.61it/s] 63%|██████▎   | 252567/400000 [00:32<00:19, 7670.64it/s] 63%|██████▎   | 253348/400000 [00:33<00:19, 7710.97it/s] 64%|██████▎   | 254139/400000 [00:33<00:18, 7768.96it/s] 64%|██████▎   | 254924/400000 [00:33<00:18, 7791.08it/s] 64%|██████▍   | 255704/400000 [00:33<00:18, 7719.27it/s] 64%|██████▍   | 256477/400000 [00:33<00:18, 7691.39it/s] 64%|██████▍   | 257263/400000 [00:33<00:18, 7739.95it/s] 65%|██████▍   | 258038/400000 [00:33<00:18, 7725.22it/s] 65%|██████▍   | 258856/400000 [00:33<00:17, 7853.44it/s] 65%|██████▍   | 259642/400000 [00:33<00:17, 7825.29it/s] 65%|██████▌   | 260450/400000 [00:33<00:17, 7897.46it/s] 65%|██████▌   | 261266/400000 [00:34<00:17, 7972.35it/s] 66%|██████▌   | 262064/400000 [00:34<00:17, 7814.52it/s] 66%|██████▌   | 262849/400000 [00:34<00:17, 7820.13it/s] 66%|██████▌   | 263632/400000 [00:34<00:17, 7776.49it/s] 66%|██████▌   | 264434/400000 [00:34<00:17, 7847.21it/s] 66%|██████▋   | 265228/400000 [00:34<00:17, 7872.67it/s] 67%|██████▋   | 266016/400000 [00:34<00:17, 7794.04it/s] 67%|██████▋   | 266826/400000 [00:34<00:16, 7882.44it/s] 67%|██████▋   | 267615/400000 [00:34<00:17, 7784.47it/s] 67%|██████▋   | 268414/400000 [00:34<00:16, 7844.68it/s] 67%|██████▋   | 269220/400000 [00:35<00:16, 7906.46it/s] 68%|██████▊   | 270012/400000 [00:35<00:16, 7840.94it/s] 68%|██████▊   | 270815/400000 [00:35<00:16, 7895.01it/s] 68%|██████▊   | 271605/400000 [00:35<00:16, 7808.51it/s] 68%|██████▊   | 272387/400000 [00:35<00:16, 7720.91it/s] 68%|██████▊   | 273184/400000 [00:35<00:16, 7792.41it/s] 68%|██████▊   | 273964/400000 [00:35<00:16, 7735.05it/s] 69%|██████▊   | 274738/400000 [00:35<00:16, 7470.59it/s] 69%|██████▉   | 275488/400000 [00:35<00:16, 7338.96it/s] 69%|██████▉   | 276225/400000 [00:36<00:16, 7328.09it/s] 69%|██████▉   | 276986/400000 [00:36<00:16, 7408.20it/s] 69%|██████▉   | 277755/400000 [00:36<00:16, 7487.61it/s] 70%|██████▉   | 278521/400000 [00:36<00:16, 7536.07it/s] 70%|██████▉   | 279290/400000 [00:36<00:15, 7579.57it/s] 70%|███████   | 280082/400000 [00:36<00:15, 7676.11it/s] 70%|███████   | 280864/400000 [00:36<00:15, 7716.82it/s] 70%|███████   | 281638/400000 [00:36<00:15, 7721.05it/s] 71%|███████   | 282411/400000 [00:36<00:15, 7591.12it/s] 71%|███████   | 283174/400000 [00:36<00:15, 7601.16it/s] 71%|███████   | 283952/400000 [00:37<00:15, 7653.17it/s] 71%|███████   | 284744/400000 [00:37<00:14, 7731.29it/s] 71%|███████▏  | 285525/400000 [00:37<00:14, 7752.38it/s] 72%|███████▏  | 286303/400000 [00:37<00:14, 7758.51it/s] 72%|███████▏  | 287080/400000 [00:37<00:14, 7757.12it/s] 72%|███████▏  | 287856/400000 [00:37<00:14, 7648.34it/s] 72%|███████▏  | 288622/400000 [00:37<00:14, 7544.96it/s] 72%|███████▏  | 289378/400000 [00:37<00:14, 7502.73it/s] 73%|███████▎  | 290129/400000 [00:37<00:14, 7479.81it/s] 73%|███████▎  | 290900/400000 [00:37<00:14, 7543.80it/s] 73%|███████▎  | 291695/400000 [00:38<00:14, 7660.84it/s] 73%|███████▎  | 292505/400000 [00:38<00:13, 7786.52it/s] 73%|███████▎  | 293285/400000 [00:38<00:13, 7779.00it/s] 74%|███████▎  | 294105/400000 [00:38<00:13, 7899.78it/s] 74%|███████▎  | 294896/400000 [00:38<00:13, 7808.17it/s] 74%|███████▍  | 295696/400000 [00:38<00:13, 7863.45it/s] 74%|███████▍  | 296495/400000 [00:38<00:13, 7900.41it/s] 74%|███████▍  | 297303/400000 [00:38<00:12, 7952.20it/s] 75%|███████▍  | 298109/400000 [00:38<00:12, 7983.30it/s] 75%|███████▍  | 298908/400000 [00:38<00:13, 7758.03it/s] 75%|███████▍  | 299687/400000 [00:39<00:12, 7764.54it/s] 75%|███████▌  | 300465/400000 [00:39<00:12, 7752.48it/s] 75%|███████▌  | 301242/400000 [00:39<00:12, 7697.13it/s] 76%|███████▌  | 302015/400000 [00:39<00:12, 7704.45it/s] 76%|███████▌  | 302786/400000 [00:39<00:12, 7702.44it/s] 76%|███████▌  | 303557/400000 [00:39<00:12, 7603.54it/s] 76%|███████▌  | 304328/400000 [00:39<00:12, 7635.10it/s] 76%|███████▋  | 305110/400000 [00:39<00:12, 7688.46it/s] 76%|███████▋  | 305891/400000 [00:39<00:12, 7723.21it/s] 77%|███████▋  | 306664/400000 [00:39<00:12, 7655.62it/s] 77%|███████▋  | 307434/400000 [00:40<00:12, 7666.39it/s] 77%|███████▋  | 308230/400000 [00:40<00:11, 7751.08it/s] 77%|███████▋  | 309024/400000 [00:40<00:11, 7805.21it/s] 77%|███████▋  | 309829/400000 [00:40<00:11, 7874.77it/s] 78%|███████▊  | 310617/400000 [00:40<00:11, 7739.25it/s] 78%|███████▊  | 311412/400000 [00:40<00:11, 7801.27it/s] 78%|███████▊  | 312193/400000 [00:40<00:11, 7738.74it/s] 78%|███████▊  | 312997/400000 [00:40<00:11, 7824.02it/s] 78%|███████▊  | 313788/400000 [00:40<00:10, 7848.10it/s] 79%|███████▊  | 314574/400000 [00:40<00:11, 7677.37it/s] 79%|███████▉  | 315349/400000 [00:41<00:10, 7697.36it/s] 79%|███████▉  | 316146/400000 [00:41<00:10, 7776.94it/s] 79%|███████▉  | 316951/400000 [00:41<00:10, 7854.18it/s] 79%|███████▉  | 317756/400000 [00:41<00:10, 7911.67it/s] 80%|███████▉  | 318548/400000 [00:41<00:10, 7836.38it/s] 80%|███████▉  | 319337/400000 [00:41<00:10, 7852.30it/s] 80%|████████  | 320125/400000 [00:41<00:10, 7859.23it/s] 80%|████████  | 320952/400000 [00:41<00:09, 7977.93it/s] 80%|████████  | 321751/400000 [00:41<00:09, 7972.85it/s] 81%|████████  | 322549/400000 [00:41<00:09, 7805.25it/s] 81%|████████  | 323364/400000 [00:42<00:09, 7903.55it/s] 81%|████████  | 324165/400000 [00:42<00:09, 7934.23it/s] 81%|████████  | 324973/400000 [00:42<00:09, 7975.03it/s] 81%|████████▏ | 325776/400000 [00:42<00:09, 7991.34it/s] 82%|████████▏ | 326576/400000 [00:42<00:09, 7908.77it/s] 82%|████████▏ | 327368/400000 [00:42<00:09, 7792.60it/s] 82%|████████▏ | 328153/400000 [00:42<00:09, 7807.30it/s] 82%|████████▏ | 328965/400000 [00:42<00:08, 7897.33it/s] 82%|████████▏ | 329756/400000 [00:42<00:08, 7812.68it/s] 83%|████████▎ | 330538/400000 [00:43<00:09, 7702.78it/s] 83%|████████▎ | 331310/400000 [00:43<00:08, 7654.71it/s] 83%|████████▎ | 332099/400000 [00:43<00:08, 7721.07it/s] 83%|████████▎ | 332872/400000 [00:43<00:09, 7451.19it/s] 83%|████████▎ | 333679/400000 [00:43<00:08, 7626.48it/s] 84%|████████▎ | 334445/400000 [00:43<00:08, 7568.01it/s] 84%|████████▍ | 335204/400000 [00:43<00:08, 7409.25it/s] 84%|████████▍ | 335971/400000 [00:43<00:08, 7482.98it/s] 84%|████████▍ | 336781/400000 [00:43<00:08, 7656.71it/s] 84%|████████▍ | 337590/400000 [00:43<00:08, 7779.81it/s] 85%|████████▍ | 338370/400000 [00:44<00:07, 7754.45it/s] 85%|████████▍ | 339160/400000 [00:44<00:07, 7796.44it/s] 85%|████████▍ | 339941/400000 [00:44<00:07, 7633.44it/s] 85%|████████▌ | 340718/400000 [00:44<00:07, 7673.84it/s] 85%|████████▌ | 341528/400000 [00:44<00:07, 7794.62it/s] 86%|████████▌ | 342309/400000 [00:44<00:07, 7781.51it/s] 86%|████████▌ | 343098/400000 [00:44<00:07, 7812.93it/s] 86%|████████▌ | 343889/400000 [00:44<00:07, 7839.46it/s] 86%|████████▌ | 344674/400000 [00:44<00:07, 7813.69it/s] 86%|████████▋ | 345475/400000 [00:44<00:06, 7869.97it/s] 87%|████████▋ | 346263/400000 [00:45<00:06, 7824.13it/s] 87%|████████▋ | 347057/400000 [00:45<00:06, 7857.92it/s] 87%|████████▋ | 347861/400000 [00:45<00:06, 7910.27it/s] 87%|████████▋ | 348663/400000 [00:45<00:06, 7940.97it/s] 87%|████████▋ | 349459/400000 [00:45<00:06, 7946.47it/s] 88%|████████▊ | 350254/400000 [00:45<00:06, 7802.78it/s] 88%|████████▊ | 351035/400000 [00:45<00:06, 7742.54it/s] 88%|████████▊ | 351823/400000 [00:45<00:06, 7781.34it/s] 88%|████████▊ | 352602/400000 [00:45<00:06, 7721.63it/s] 88%|████████▊ | 353393/400000 [00:45<00:05, 7777.09it/s] 89%|████████▊ | 354183/400000 [00:46<00:05, 7813.48it/s] 89%|████████▊ | 354996/400000 [00:46<00:05, 7905.58it/s] 89%|████████▉ | 355788/400000 [00:46<00:05, 7761.01it/s] 89%|████████▉ | 356617/400000 [00:46<00:05, 7910.49it/s] 89%|████████▉ | 357436/400000 [00:46<00:05, 7990.34it/s] 90%|████████▉ | 358237/400000 [00:46<00:05, 7925.12it/s] 90%|████████▉ | 359031/400000 [00:46<00:05, 7834.68it/s] 90%|████████▉ | 359816/400000 [00:46<00:05, 7827.79it/s] 90%|█████████ | 360611/400000 [00:46<00:05, 7863.61it/s] 90%|█████████ | 361398/400000 [00:46<00:04, 7782.64it/s] 91%|█████████ | 362177/400000 [00:47<00:04, 7763.58it/s] 91%|█████████ | 362983/400000 [00:47<00:04, 7848.65it/s] 91%|█████████ | 363769/400000 [00:47<00:04, 7827.41it/s] 91%|█████████ | 364553/400000 [00:47<00:04, 7819.47it/s] 91%|█████████▏| 365336/400000 [00:47<00:04, 7801.12it/s] 92%|█████████▏| 366117/400000 [00:47<00:04, 7617.94it/s] 92%|█████████▏| 366880/400000 [00:47<00:04, 7591.60it/s] 92%|█████████▏| 367656/400000 [00:47<00:04, 7639.57it/s] 92%|█████████▏| 368448/400000 [00:47<00:04, 7719.66it/s] 92%|█████████▏| 369244/400000 [00:47<00:03, 7788.14it/s] 93%|█████████▎| 370024/400000 [00:48<00:03, 7566.19it/s] 93%|█████████▎| 370783/400000 [00:48<00:03, 7520.08it/s] 93%|█████████▎| 371588/400000 [00:48<00:03, 7669.70it/s] 93%|█████████▎| 372379/400000 [00:48<00:03, 7738.00it/s] 93%|█████████▎| 373183/400000 [00:48<00:03, 7824.18it/s] 93%|█████████▎| 373967/400000 [00:48<00:03, 7786.13it/s] 94%|█████████▎| 374755/400000 [00:48<00:03, 7812.55it/s] 94%|█████████▍| 375544/400000 [00:48<00:03, 7834.66it/s] 94%|█████████▍| 376328/400000 [00:48<00:03, 7820.44it/s] 94%|█████████▍| 377123/400000 [00:49<00:02, 7857.87it/s] 94%|█████████▍| 377910/400000 [00:49<00:02, 7770.23it/s] 95%|█████████▍| 378688/400000 [00:49<00:02, 7736.61it/s] 95%|█████████▍| 379462/400000 [00:49<00:02, 7569.71it/s] 95%|█████████▌| 380249/400000 [00:49<00:02, 7655.79it/s] 95%|█████████▌| 381016/400000 [00:49<00:02, 7653.07it/s] 95%|█████████▌| 381782/400000 [00:49<00:02, 7578.13it/s] 96%|█████████▌| 382541/400000 [00:49<00:02, 7513.11it/s] 96%|█████████▌| 383310/400000 [00:49<00:02, 7562.77it/s] 96%|█████████▌| 384067/400000 [00:49<00:02, 7465.32it/s] 96%|█████████▌| 384873/400000 [00:50<00:01, 7631.85it/s] 96%|█████████▋| 385638/400000 [00:50<00:01, 7612.65it/s] 97%|█████████▋| 386439/400000 [00:50<00:01, 7726.86it/s] 97%|█████████▋| 387215/400000 [00:50<00:01, 7736.03it/s] 97%|█████████▋| 388001/400000 [00:50<00:01, 7770.33it/s] 97%|█████████▋| 388798/400000 [00:50<00:01, 7829.13it/s] 97%|█████████▋| 389582/400000 [00:50<00:01, 7722.49it/s] 98%|█████████▊| 390370/400000 [00:50<00:01, 7768.41it/s] 98%|█████████▊| 391151/400000 [00:50<00:01, 7779.54it/s] 98%|█████████▊| 391930/400000 [00:50<00:01, 7751.60it/s] 98%|█████████▊| 392716/400000 [00:51<00:00, 7782.54it/s] 98%|█████████▊| 393495/400000 [00:51<00:00, 7687.40it/s] 99%|█████████▊| 394298/400000 [00:51<00:00, 7784.67it/s] 99%|█████████▉| 395085/400000 [00:51<00:00, 7807.20it/s] 99%|█████████▉| 395867/400000 [00:51<00:00, 7808.27it/s] 99%|█████████▉| 396649/400000 [00:51<00:00, 7801.63it/s] 99%|█████████▉| 397430/400000 [00:51<00:00, 7763.48it/s]100%|█████████▉| 398211/400000 [00:51<00:00, 7777.32it/s]100%|█████████▉| 398991/400000 [00:51<00:00, 7780.81it/s]100%|█████████▉| 399772/400000 [00:51<00:00, 7786.56it/s]100%|█████████▉| 399999/400000 [00:51<00:00, 7696.16it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7fae5a0a3cc0> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.010862101045289264 	 Accuracy: 56
Train Epoch: 1 	 Loss: 0.010946493284359425 	 Accuracy: 62

  model saves at 62% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15684 out of table with 15681 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15684 out of table with 15681 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-13 11:24:31.374321: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 11:24:31.378549: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 11:24:31.379308: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55895e5ec690 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 11:24:31.379324: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7fae0d5e1128> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.3293 - accuracy: 0.5220
 2000/25000 [=>............................] - ETA: 10s - loss: 7.4596 - accuracy: 0.5135
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5133 - accuracy: 0.5100 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5095 - accuracy: 0.5102
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5654 - accuracy: 0.5066
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5618 - accuracy: 0.5068
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6381 - accuracy: 0.5019
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6436 - accuracy: 0.5015
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6223 - accuracy: 0.5029
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6084 - accuracy: 0.5038
11000/25000 [============>.................] - ETA: 4s - loss: 7.6053 - accuracy: 0.5040
12000/25000 [=============>................] - ETA: 4s - loss: 7.6155 - accuracy: 0.5033
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6395 - accuracy: 0.5018
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6272 - accuracy: 0.5026
15000/25000 [=================>............] - ETA: 3s - loss: 7.6625 - accuracy: 0.5003
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6513 - accuracy: 0.5010
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6369 - accuracy: 0.5019
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6462 - accuracy: 0.5013
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6352 - accuracy: 0.5021
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6429 - accuracy: 0.5016
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6520 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6443 - accuracy: 0.5015
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6493 - accuracy: 0.5011
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6494 - accuracy: 0.5011
25000/25000 [==============================] - 10s 397us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7fadc6ace470> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7fadc7cb00f0> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 2.0315 - crf_viterbi_accuracy: 0.2133 - val_loss: 1.9701 - val_crf_viterbi_accuracy: 0.2533

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

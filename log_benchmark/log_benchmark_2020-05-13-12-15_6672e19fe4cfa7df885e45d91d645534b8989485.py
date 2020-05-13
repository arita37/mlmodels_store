
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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f057dcabfd0> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 12:15:39.232503
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 12:15:39.236799
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 12:15:39.240701
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 12:15:39.244836
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f0589a754a8> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 356542.3438
Epoch 2/10

1/1 [==============================] - 0s 110ms/step - loss: 320249.0938
Epoch 3/10

1/1 [==============================] - 0s 104ms/step - loss: 261736.5938
Epoch 4/10

1/1 [==============================] - 0s 117ms/step - loss: 203955.9688
Epoch 5/10

1/1 [==============================] - 0s 101ms/step - loss: 146615.4688
Epoch 6/10

1/1 [==============================] - 0s 102ms/step - loss: 101040.2344
Epoch 7/10

1/1 [==============================] - 0s 103ms/step - loss: 67727.7969
Epoch 8/10

1/1 [==============================] - 0s 103ms/step - loss: 45597.3008
Epoch 9/10

1/1 [==============================] - 0s 105ms/step - loss: 30911.4688
Epoch 10/10

1/1 [==============================] - 0s 104ms/step - loss: 21364.3438

  #### Inference Need return ypred, ytrue ######################### 
[[ 2.59348333e-01  8.51649582e-01 -1.08687997e-01 -4.31854695e-01
  -5.88318944e-01 -2.12229788e-02  3.32361430e-01  6.42566532e-02
   1.65994376e-01 -3.89320254e-01  1.35139287e+00  2.98903763e-01
   9.73449424e-02  1.28542989e-01 -5.94889283e-01 -1.16093397e-01
   5.15034676e-01 -3.03323090e-01  1.25055939e-01 -9.15275693e-01
  -4.42917943e-01 -8.50667536e-01 -7.09768653e-01 -5.05042911e-01
  -5.01553714e-03 -7.08239079e-02 -9.51521471e-02  2.55147249e-01
   5.00429571e-01 -4.95564699e-01 -9.96176898e-03  3.28923404e-01
   7.09863901e-02 -5.45595646e-01 -5.91014326e-01  6.95128560e-01
   3.58195901e-02  3.21347833e-01 -3.51309299e-01  2.71356434e-01
  -4.72445309e-01  4.17162836e-01  1.77036166e-01 -1.42432034e-01
  -8.15228641e-01  1.03944674e-01  2.29925811e-01 -4.87924635e-01
   6.04749501e-01  1.42686713e+00  3.63234371e-01  9.43323612e-01
  -3.38480175e-02  5.66161036e-01  8.72392356e-02  9.13417339e-02
   6.38878167e-01  1.79351211e-01  9.64218676e-02 -7.51443982e-01
   5.26003614e-02  2.95046759e+00  3.39023566e+00  3.70844841e+00
   4.26760054e+00  3.46804285e+00  3.92707205e+00  4.42249680e+00
   3.14628386e+00  4.31727362e+00  3.65886116e+00  4.31299210e+00
   3.88116550e+00  4.51554489e+00  3.81259179e+00  3.52027702e+00
   4.13327312e+00  4.62453938e+00  4.32323551e+00  4.00384808e+00
   4.32457018e+00  3.34418607e+00  4.91459274e+00  4.24840832e+00
   3.59314346e+00  4.31579304e+00  4.57083654e+00  3.73441219e+00
   3.40146017e+00  4.67701578e+00  5.29220343e+00  3.79489851e+00
   3.71801901e+00  4.14056206e+00  4.15335083e+00  3.26990294e+00
   4.71355391e+00  4.46320963e+00  2.76208925e+00  3.92733335e+00
   4.06077909e+00  4.28671169e+00  5.12085485e+00  3.98178840e+00
   4.46813059e+00  4.76814270e+00  4.67485046e+00  3.85491371e+00
   3.62494063e+00  3.74024534e+00  4.33995390e+00  3.61279082e+00
   4.40090656e+00  3.78075409e+00  4.72234774e+00  4.27275276e+00
   4.30447960e+00  3.83840513e+00  4.32690859e+00  3.91178703e+00
  -7.67399907e-01 -1.27334297e-02  3.39488804e-01  4.46347862e-01
   3.50186735e-01  2.57825583e-01 -6.83047354e-01 -4.88080859e-01
   5.20170569e-01  3.05808723e-01 -3.66203457e-01  2.83352345e-01
  -9.09772515e-01 -2.28088692e-01  5.85966289e-01  3.21613193e-01
  -6.57547653e-01  5.48049808e-02  6.78134441e-01 -3.81022066e-01
   6.05471671e-01  7.26560473e-01 -9.80795324e-02  4.95798290e-01
  -2.85726190e-01 -9.03370619e-01  8.08011472e-01  2.77675629e-01
   7.49051571e-02  6.25351489e-01  4.64980155e-01  8.31149220e-01
  -1.27470702e-01 -8.86587024e-01  4.24020231e-01 -3.76202404e-01
  -1.53567702e-01 -5.92845380e-01  7.07415760e-01 -1.34368360e-01
   5.44375002e-01  8.81357074e-01  2.98546404e-01 -7.72862613e-01
  -4.76035774e-01  4.31336462e-01  6.84311032e-01  4.49738652e-01
  -3.21463585e-01 -6.53164908e-02 -6.26436770e-01 -9.98785079e-01
   8.15271378e-01  3.93755734e-02  3.11724871e-01 -8.16135228e-01
   2.64609694e-01  4.59844351e-01  3.22509706e-01 -5.05396664e-01
   1.06216526e+00  4.45981622e-01  6.06715798e-01  1.38506973e+00
   1.76428020e+00  1.09459937e+00  1.30770659e+00  1.45398521e+00
   3.33553970e-01  3.63435507e-01  1.30113161e+00  5.88699341e-01
   1.22625554e+00  6.78863108e-01  4.85347152e-01  1.56575429e+00
   7.21132100e-01  1.70655298e+00  1.16969156e+00  1.29662943e+00
   1.38277411e+00  4.58173871e-01  1.83038259e+00  1.90669072e+00
   5.66142201e-01  1.13368237e+00  5.48734903e-01  1.68252373e+00
   6.49623871e-01  1.24251294e+00  4.68095422e-01  4.01823163e-01
   8.16309035e-01  2.04761171e+00  4.46229339e-01  1.78072155e+00
   5.38322091e-01  1.49215853e+00  1.05067527e+00  1.11330724e+00
   1.72105885e+00  1.92291319e+00  1.37881088e+00  7.73832738e-01
   1.76693892e+00  3.55717540e-01  1.39999676e+00  9.41964269e-01
   8.28273416e-01  1.16041005e+00  5.86320877e-01  1.16797602e+00
   6.01995468e-01  1.85675263e+00  1.07462275e+00  4.10637498e-01
   1.29628038e+00  1.40383840e+00  9.70616162e-01  9.44491506e-01
   2.57164836e-02  4.87430668e+00  5.58294678e+00  4.75583315e+00
   4.83544683e+00  4.96015644e+00  5.26108980e+00  5.72374344e+00
   4.44109297e+00  5.34265089e+00  4.59457922e+00  4.91851711e+00
   4.58303881e+00  4.58070946e+00  4.22360134e+00  4.30493927e+00
   5.42732000e+00  4.57783031e+00  5.29026413e+00  4.29558372e+00
   4.78931475e+00  4.82813835e+00  4.66618299e+00  4.87111664e+00
   4.79488230e+00  4.99661112e+00  4.11103678e+00  4.59773827e+00
   4.26470661e+00  4.85201502e+00  4.67398453e+00  5.24578238e+00
   4.73115301e+00  5.17210722e+00  4.35804796e+00  4.81966877e+00
   4.45288467e+00  4.99600410e+00  4.90827608e+00  4.32676172e+00
   4.70001507e+00  4.65759087e+00  4.50002527e+00  4.83669901e+00
   5.27953148e+00  4.46958637e+00  5.26034069e+00  5.38629723e+00
   5.50096178e+00  4.85068178e+00  4.01871824e+00  4.90584087e+00
   3.97819138e+00  5.19220781e+00  3.98152161e+00  4.23911047e+00
   5.28647089e+00  4.92173910e+00  5.20431805e+00  4.02574825e+00
   2.06096363e+00  1.23084104e+00  1.36748362e+00  4.38807786e-01
   2.18162823e+00  1.34534812e+00  7.37177372e-01  4.52692032e-01
   9.18884277e-01  1.06620538e+00  8.37673068e-01  8.61713409e-01
   5.15281141e-01  1.12942100e+00  1.15797198e+00  7.56350219e-01
   1.59704280e+00  1.00047076e+00  1.10240126e+00  4.63389874e-01
   3.08620155e-01  7.57857978e-01  1.03688025e+00  1.25411463e+00
   6.40509605e-01  5.22751749e-01  4.63922322e-01  1.47124410e+00
   7.47134924e-01  1.68558073e+00  1.01694643e+00  1.11550701e+00
   1.22250450e+00  5.95926046e-01  5.78143358e-01  1.43863082e+00
   4.60435390e-01  1.52785110e+00  1.83452189e+00  3.23895574e-01
   1.00345922e+00  1.44442523e+00  6.28243387e-01  1.75146854e+00
   8.80933404e-01  6.96900129e-01  1.78963876e+00  7.39345908e-01
   1.19119143e+00  1.14145696e+00  1.13392591e+00  4.35448229e-01
   1.03730965e+00  1.06842291e+00  8.30313206e-01  1.51493227e+00
   3.78657162e-01  8.64052117e-01  1.14267325e+00  1.01783168e+00
  -5.20546007e+00  4.01542330e+00 -1.55279708e+00]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 12:15:48.771973
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.5248
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 12:15:48.776444
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   9526.14
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 12:15:48.780194
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   97.7037
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 12:15:48.783818
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -852.133
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139661202826856
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139660261319344
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139660261319848
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139660261320352
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139660261320856
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139660261321360

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f057dcab240> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.696789
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.665589
grad_step = 000002, loss = 0.644310
grad_step = 000003, loss = 0.623245
grad_step = 000004, loss = 0.601101
grad_step = 000005, loss = 0.579863
grad_step = 000006, loss = 0.560856
grad_step = 000007, loss = 0.544845
grad_step = 000008, loss = 0.526259
grad_step = 000009, loss = 0.502621
grad_step = 000010, loss = 0.477705
grad_step = 000011, loss = 0.455251
grad_step = 000012, loss = 0.435545
grad_step = 000013, loss = 0.417218
grad_step = 000014, loss = 0.400163
grad_step = 000015, loss = 0.381982
grad_step = 000016, loss = 0.364452
grad_step = 000017, loss = 0.347455
grad_step = 000018, loss = 0.329139
grad_step = 000019, loss = 0.311670
grad_step = 000020, loss = 0.297433
grad_step = 000021, loss = 0.285391
grad_step = 000022, loss = 0.273740
grad_step = 000023, loss = 0.262176
grad_step = 000024, loss = 0.251393
grad_step = 000025, loss = 0.241325
grad_step = 000026, loss = 0.230644
grad_step = 000027, loss = 0.219491
grad_step = 000028, loss = 0.209047
grad_step = 000029, loss = 0.199233
grad_step = 000030, loss = 0.189492
grad_step = 000031, loss = 0.180032
grad_step = 000032, loss = 0.171380
grad_step = 000033, loss = 0.163201
grad_step = 000034, loss = 0.155063
grad_step = 000035, loss = 0.147335
grad_step = 000036, loss = 0.140147
grad_step = 000037, loss = 0.133137
grad_step = 000038, loss = 0.126116
grad_step = 000039, loss = 0.119390
grad_step = 000040, loss = 0.113052
grad_step = 000041, loss = 0.106807
grad_step = 000042, loss = 0.100705
grad_step = 000043, loss = 0.094962
grad_step = 000044, loss = 0.089534
grad_step = 000045, loss = 0.084344
grad_step = 000046, loss = 0.079416
grad_step = 000047, loss = 0.074799
grad_step = 000048, loss = 0.070386
grad_step = 000049, loss = 0.066104
grad_step = 000050, loss = 0.062083
grad_step = 000051, loss = 0.058288
grad_step = 000052, loss = 0.054611
grad_step = 000053, loss = 0.051122
grad_step = 000054, loss = 0.047883
grad_step = 000055, loss = 0.044786
grad_step = 000056, loss = 0.041844
grad_step = 000057, loss = 0.039125
grad_step = 000058, loss = 0.036531
grad_step = 000059, loss = 0.034072
grad_step = 000060, loss = 0.031800
grad_step = 000061, loss = 0.029654
grad_step = 000062, loss = 0.027617
grad_step = 000063, loss = 0.025716
grad_step = 000064, loss = 0.023919
grad_step = 000065, loss = 0.022227
grad_step = 000066, loss = 0.020644
grad_step = 000067, loss = 0.019162
grad_step = 000068, loss = 0.017769
grad_step = 000069, loss = 0.016474
grad_step = 000070, loss = 0.015274
grad_step = 000071, loss = 0.014144
grad_step = 000072, loss = 0.013100
grad_step = 000073, loss = 0.012127
grad_step = 000074, loss = 0.011222
grad_step = 000075, loss = 0.010389
grad_step = 000076, loss = 0.009608
grad_step = 000077, loss = 0.008870
grad_step = 000078, loss = 0.008190
grad_step = 000079, loss = 0.007567
grad_step = 000080, loss = 0.007002
grad_step = 000081, loss = 0.006492
grad_step = 000082, loss = 0.006023
grad_step = 000083, loss = 0.005590
grad_step = 000084, loss = 0.005191
grad_step = 000085, loss = 0.004822
grad_step = 000086, loss = 0.004492
grad_step = 000087, loss = 0.004199
grad_step = 000088, loss = 0.003938
grad_step = 000089, loss = 0.003702
grad_step = 000090, loss = 0.003491
grad_step = 000091, loss = 0.003297
grad_step = 000092, loss = 0.003127
grad_step = 000093, loss = 0.002975
grad_step = 000094, loss = 0.002842
grad_step = 000095, loss = 0.002728
grad_step = 000096, loss = 0.002626
grad_step = 000097, loss = 0.002537
grad_step = 000098, loss = 0.002457
grad_step = 000099, loss = 0.002386
grad_step = 000100, loss = 0.002322
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002265
grad_step = 000102, loss = 0.002215
grad_step = 000103, loss = 0.002171
grad_step = 000104, loss = 0.002133
grad_step = 000105, loss = 0.002100
grad_step = 000106, loss = 0.002072
grad_step = 000107, loss = 0.002048
grad_step = 000108, loss = 0.002028
grad_step = 000109, loss = 0.002012
grad_step = 000110, loss = 0.002000
grad_step = 000111, loss = 0.001996
grad_step = 000112, loss = 0.002001
grad_step = 000113, loss = 0.002029
grad_step = 000114, loss = 0.002072
grad_step = 000115, loss = 0.002135
grad_step = 000116, loss = 0.002131
grad_step = 000117, loss = 0.002063
grad_step = 000118, loss = 0.001957
grad_step = 000119, loss = 0.001924
grad_step = 000120, loss = 0.001972
grad_step = 000121, loss = 0.002021
grad_step = 000122, loss = 0.002011
grad_step = 000123, loss = 0.001943
grad_step = 000124, loss = 0.001902
grad_step = 000125, loss = 0.001919
grad_step = 000126, loss = 0.001957
grad_step = 000127, loss = 0.001966
grad_step = 000128, loss = 0.001930
grad_step = 000129, loss = 0.001891
grad_step = 000130, loss = 0.001885
grad_step = 000131, loss = 0.001905
grad_step = 000132, loss = 0.001922
grad_step = 000133, loss = 0.001914
grad_step = 000134, loss = 0.001889
grad_step = 000135, loss = 0.001868
grad_step = 000136, loss = 0.001865
grad_step = 000137, loss = 0.001874
grad_step = 000138, loss = 0.001884
grad_step = 000139, loss = 0.001884
grad_step = 000140, loss = 0.001872
grad_step = 000141, loss = 0.001857
grad_step = 000142, loss = 0.001845
grad_step = 000143, loss = 0.001841
grad_step = 000144, loss = 0.001843
grad_step = 000145, loss = 0.001848
grad_step = 000146, loss = 0.001851
grad_step = 000147, loss = 0.001851
grad_step = 000148, loss = 0.001848
grad_step = 000149, loss = 0.001842
grad_step = 000150, loss = 0.001835
grad_step = 000151, loss = 0.001828
grad_step = 000152, loss = 0.001821
grad_step = 000153, loss = 0.001816
grad_step = 000154, loss = 0.001811
grad_step = 000155, loss = 0.001807
grad_step = 000156, loss = 0.001803
grad_step = 000157, loss = 0.001800
grad_step = 000158, loss = 0.001797
grad_step = 000159, loss = 0.001794
grad_step = 000160, loss = 0.001791
grad_step = 000161, loss = 0.001788
grad_step = 000162, loss = 0.001785
grad_step = 000163, loss = 0.001782
grad_step = 000164, loss = 0.001780
grad_step = 000165, loss = 0.001779
grad_step = 000166, loss = 0.001783
grad_step = 000167, loss = 0.001797
grad_step = 000168, loss = 0.001842
grad_step = 000169, loss = 0.001958
grad_step = 000170, loss = 0.002257
grad_step = 000171, loss = 0.002717
grad_step = 000172, loss = 0.003123
grad_step = 000173, loss = 0.002476
grad_step = 000174, loss = 0.001777
grad_step = 000175, loss = 0.002077
grad_step = 000176, loss = 0.002492
grad_step = 000177, loss = 0.002110
grad_step = 000178, loss = 0.001767
grad_step = 000179, loss = 0.002173
grad_step = 000180, loss = 0.002195
grad_step = 000181, loss = 0.001767
grad_step = 000182, loss = 0.001952
grad_step = 000183, loss = 0.002111
grad_step = 000184, loss = 0.001780
grad_step = 000185, loss = 0.001859
grad_step = 000186, loss = 0.002013
grad_step = 000187, loss = 0.001788
grad_step = 000188, loss = 0.001803
grad_step = 000189, loss = 0.001924
grad_step = 000190, loss = 0.001779
grad_step = 000191, loss = 0.001776
grad_step = 000192, loss = 0.001855
grad_step = 000193, loss = 0.001774
grad_step = 000194, loss = 0.001760
grad_step = 000195, loss = 0.001805
grad_step = 000196, loss = 0.001765
grad_step = 000197, loss = 0.001747
grad_step = 000198, loss = 0.001763
grad_step = 000199, loss = 0.001758
grad_step = 000200, loss = 0.001737
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001734
grad_step = 000202, loss = 0.001754
grad_step = 000203, loss = 0.001728
grad_step = 000204, loss = 0.001712
grad_step = 000205, loss = 0.001748
grad_step = 000206, loss = 0.001718
grad_step = 000207, loss = 0.001699
grad_step = 000208, loss = 0.001737
grad_step = 000209, loss = 0.001709
grad_step = 000210, loss = 0.001693
grad_step = 000211, loss = 0.001719
grad_step = 000212, loss = 0.001700
grad_step = 000213, loss = 0.001691
grad_step = 000214, loss = 0.001698
grad_step = 000215, loss = 0.001690
grad_step = 000216, loss = 0.001690
grad_step = 000217, loss = 0.001682
grad_step = 000218, loss = 0.001680
grad_step = 000219, loss = 0.001686
grad_step = 000220, loss = 0.001674
grad_step = 000221, loss = 0.001670
grad_step = 000222, loss = 0.001678
grad_step = 000223, loss = 0.001669
grad_step = 000224, loss = 0.001664
grad_step = 000225, loss = 0.001668
grad_step = 000226, loss = 0.001664
grad_step = 000227, loss = 0.001661
grad_step = 000228, loss = 0.001660
grad_step = 000229, loss = 0.001656
grad_step = 000230, loss = 0.001656
grad_step = 000231, loss = 0.001656
grad_step = 000232, loss = 0.001650
grad_step = 000233, loss = 0.001648
grad_step = 000234, loss = 0.001650
grad_step = 000235, loss = 0.001646
grad_step = 000236, loss = 0.001643
grad_step = 000237, loss = 0.001642
grad_step = 000238, loss = 0.001640
grad_step = 000239, loss = 0.001638
grad_step = 000240, loss = 0.001637
grad_step = 000241, loss = 0.001634
grad_step = 000242, loss = 0.001632
grad_step = 000243, loss = 0.001631
grad_step = 000244, loss = 0.001630
grad_step = 000245, loss = 0.001628
grad_step = 000246, loss = 0.001626
grad_step = 000247, loss = 0.001624
grad_step = 000248, loss = 0.001623
grad_step = 000249, loss = 0.001621
grad_step = 000250, loss = 0.001619
grad_step = 000251, loss = 0.001618
grad_step = 000252, loss = 0.001616
grad_step = 000253, loss = 0.001614
grad_step = 000254, loss = 0.001612
grad_step = 000255, loss = 0.001611
grad_step = 000256, loss = 0.001609
grad_step = 000257, loss = 0.001607
grad_step = 000258, loss = 0.001606
grad_step = 000259, loss = 0.001607
grad_step = 000260, loss = 0.001610
grad_step = 000261, loss = 0.001618
grad_step = 000262, loss = 0.001634
grad_step = 000263, loss = 0.001663
grad_step = 000264, loss = 0.001705
grad_step = 000265, loss = 0.001766
grad_step = 000266, loss = 0.001821
grad_step = 000267, loss = 0.001863
grad_step = 000268, loss = 0.001848
grad_step = 000269, loss = 0.001783
grad_step = 000270, loss = 0.001682
grad_step = 000271, loss = 0.001606
grad_step = 000272, loss = 0.001587
grad_step = 000273, loss = 0.001619
grad_step = 000274, loss = 0.001671
grad_step = 000275, loss = 0.001708
grad_step = 000276, loss = 0.001716
grad_step = 000277, loss = 0.001682
grad_step = 000278, loss = 0.001635
grad_step = 000279, loss = 0.001595
grad_step = 000280, loss = 0.001577
grad_step = 000281, loss = 0.001581
grad_step = 000282, loss = 0.001600
grad_step = 000283, loss = 0.001623
grad_step = 000284, loss = 0.001637
grad_step = 000285, loss = 0.001640
grad_step = 000286, loss = 0.001631
grad_step = 000287, loss = 0.001617
grad_step = 000288, loss = 0.001597
grad_step = 000289, loss = 0.001579
grad_step = 000290, loss = 0.001568
grad_step = 000291, loss = 0.001563
grad_step = 000292, loss = 0.001562
grad_step = 000293, loss = 0.001566
grad_step = 000294, loss = 0.001573
grad_step = 000295, loss = 0.001582
grad_step = 000296, loss = 0.001597
grad_step = 000297, loss = 0.001620
grad_step = 000298, loss = 0.001655
grad_step = 000299, loss = 0.001696
grad_step = 000300, loss = 0.001751
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001800
grad_step = 000302, loss = 0.001842
grad_step = 000303, loss = 0.001833
grad_step = 000304, loss = 0.001779
grad_step = 000305, loss = 0.001679
grad_step = 000306, loss = 0.001593
grad_step = 000307, loss = 0.001548
grad_step = 000308, loss = 0.001559
grad_step = 000309, loss = 0.001605
grad_step = 000310, loss = 0.001652
grad_step = 000311, loss = 0.001682
grad_step = 000312, loss = 0.001673
grad_step = 000313, loss = 0.001639
grad_step = 000314, loss = 0.001590
grad_step = 000315, loss = 0.001552
grad_step = 000316, loss = 0.001536
grad_step = 000317, loss = 0.001542
grad_step = 000318, loss = 0.001562
grad_step = 000319, loss = 0.001585
grad_step = 000320, loss = 0.001603
grad_step = 000321, loss = 0.001608
grad_step = 000322, loss = 0.001601
grad_step = 000323, loss = 0.001585
grad_step = 000324, loss = 0.001565
grad_step = 000325, loss = 0.001546
grad_step = 000326, loss = 0.001532
grad_step = 000327, loss = 0.001525
grad_step = 000328, loss = 0.001524
grad_step = 000329, loss = 0.001528
grad_step = 000330, loss = 0.001535
grad_step = 000331, loss = 0.001544
grad_step = 000332, loss = 0.001556
grad_step = 000333, loss = 0.001573
grad_step = 000334, loss = 0.001594
grad_step = 000335, loss = 0.001626
grad_step = 000336, loss = 0.001666
grad_step = 000337, loss = 0.001722
grad_step = 000338, loss = 0.001784
grad_step = 000339, loss = 0.001862
grad_step = 000340, loss = 0.001890
grad_step = 000341, loss = 0.001863
grad_step = 000342, loss = 0.001748
grad_step = 000343, loss = 0.001615
grad_step = 000344, loss = 0.001524
grad_step = 000345, loss = 0.001517
grad_step = 000346, loss = 0.001574
grad_step = 000347, loss = 0.001640
grad_step = 000348, loss = 0.001673
grad_step = 000349, loss = 0.001645
grad_step = 000350, loss = 0.001584
grad_step = 000351, loss = 0.001526
grad_step = 000352, loss = 0.001503
grad_step = 000353, loss = 0.001520
grad_step = 000354, loss = 0.001554
grad_step = 000355, loss = 0.001581
grad_step = 000356, loss = 0.001583
grad_step = 000357, loss = 0.001563
grad_step = 000358, loss = 0.001531
grad_step = 000359, loss = 0.001506
grad_step = 000360, loss = 0.001497
grad_step = 000361, loss = 0.001503
grad_step = 000362, loss = 0.001519
grad_step = 000363, loss = 0.001533
grad_step = 000364, loss = 0.001539
grad_step = 000365, loss = 0.001539
grad_step = 000366, loss = 0.001536
grad_step = 000367, loss = 0.001528
grad_step = 000368, loss = 0.001518
grad_step = 000369, loss = 0.001508
grad_step = 000370, loss = 0.001498
grad_step = 000371, loss = 0.001491
grad_step = 000372, loss = 0.001487
grad_step = 000373, loss = 0.001485
grad_step = 000374, loss = 0.001485
grad_step = 000375, loss = 0.001485
grad_step = 000376, loss = 0.001487
grad_step = 000377, loss = 0.001495
grad_step = 000378, loss = 0.001511
grad_step = 000379, loss = 0.001541
grad_step = 000380, loss = 0.001592
grad_step = 000381, loss = 0.001682
grad_step = 000382, loss = 0.001803
grad_step = 000383, loss = 0.001961
grad_step = 000384, loss = 0.002072
grad_step = 000385, loss = 0.002088
grad_step = 000386, loss = 0.001898
grad_step = 000387, loss = 0.001647
grad_step = 000388, loss = 0.001486
grad_step = 000389, loss = 0.001514
grad_step = 000390, loss = 0.001654
grad_step = 000391, loss = 0.001741
grad_step = 000392, loss = 0.001692
grad_step = 000393, loss = 0.001553
grad_step = 000394, loss = 0.001472
grad_step = 000395, loss = 0.001508
grad_step = 000396, loss = 0.001590
grad_step = 000397, loss = 0.001620
grad_step = 000398, loss = 0.001562
grad_step = 000399, loss = 0.001488
grad_step = 000400, loss = 0.001469
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001507
grad_step = 000402, loss = 0.001549
grad_step = 000403, loss = 0.001545
grad_step = 000404, loss = 0.001504
grad_step = 000405, loss = 0.001468
grad_step = 000406, loss = 0.001466
grad_step = 000407, loss = 0.001489
grad_step = 000408, loss = 0.001509
grad_step = 000409, loss = 0.001506
grad_step = 000410, loss = 0.001482
grad_step = 000411, loss = 0.001461
grad_step = 000412, loss = 0.001457
grad_step = 000413, loss = 0.001468
grad_step = 000414, loss = 0.001481
grad_step = 000415, loss = 0.001483
grad_step = 000416, loss = 0.001474
grad_step = 000417, loss = 0.001460
grad_step = 000418, loss = 0.001452
grad_step = 000419, loss = 0.001451
grad_step = 000420, loss = 0.001456
grad_step = 000421, loss = 0.001462
grad_step = 000422, loss = 0.001464
grad_step = 000423, loss = 0.001462
grad_step = 000424, loss = 0.001457
grad_step = 000425, loss = 0.001451
grad_step = 000426, loss = 0.001446
grad_step = 000427, loss = 0.001443
grad_step = 000428, loss = 0.001443
grad_step = 000429, loss = 0.001444
grad_step = 000430, loss = 0.001446
grad_step = 000431, loss = 0.001448
grad_step = 000432, loss = 0.001449
grad_step = 000433, loss = 0.001450
grad_step = 000434, loss = 0.001452
grad_step = 000435, loss = 0.001452
grad_step = 000436, loss = 0.001454
grad_step = 000437, loss = 0.001455
grad_step = 000438, loss = 0.001458
grad_step = 000439, loss = 0.001462
grad_step = 000440, loss = 0.001468
grad_step = 000441, loss = 0.001476
grad_step = 000442, loss = 0.001490
grad_step = 000443, loss = 0.001509
grad_step = 000444, loss = 0.001540
grad_step = 000445, loss = 0.001580
grad_step = 000446, loss = 0.001639
grad_step = 000447, loss = 0.001702
grad_step = 000448, loss = 0.001778
grad_step = 000449, loss = 0.001816
grad_step = 000450, loss = 0.001818
grad_step = 000451, loss = 0.001732
grad_step = 000452, loss = 0.001607
grad_step = 000453, loss = 0.001485
grad_step = 000454, loss = 0.001425
grad_step = 000455, loss = 0.001442
grad_step = 000456, loss = 0.001504
grad_step = 000457, loss = 0.001562
grad_step = 000458, loss = 0.001571
grad_step = 000459, loss = 0.001535
grad_step = 000460, loss = 0.001473
grad_step = 000461, loss = 0.001428
grad_step = 000462, loss = 0.001418
grad_step = 000463, loss = 0.001441
grad_step = 000464, loss = 0.001473
grad_step = 000465, loss = 0.001491
grad_step = 000466, loss = 0.001488
grad_step = 000467, loss = 0.001463
grad_step = 000468, loss = 0.001435
grad_step = 000469, loss = 0.001415
grad_step = 000470, loss = 0.001412
grad_step = 000471, loss = 0.001421
grad_step = 000472, loss = 0.001435
grad_step = 000473, loss = 0.001445
grad_step = 000474, loss = 0.001445
grad_step = 000475, loss = 0.001438
grad_step = 000476, loss = 0.001426
grad_step = 000477, loss = 0.001414
grad_step = 000478, loss = 0.001406
grad_step = 000479, loss = 0.001403
grad_step = 000480, loss = 0.001404
grad_step = 000481, loss = 0.001408
grad_step = 000482, loss = 0.001413
grad_step = 000483, loss = 0.001418
grad_step = 000484, loss = 0.001423
grad_step = 000485, loss = 0.001427
grad_step = 000486, loss = 0.001430
grad_step = 000487, loss = 0.001431
grad_step = 000488, loss = 0.001433
grad_step = 000489, loss = 0.001434
grad_step = 000490, loss = 0.001434
grad_step = 000491, loss = 0.001435
grad_step = 000492, loss = 0.001436
grad_step = 000493, loss = 0.001437
grad_step = 000494, loss = 0.001440
grad_step = 000495, loss = 0.001444
grad_step = 000496, loss = 0.001451
grad_step = 000497, loss = 0.001459
grad_step = 000498, loss = 0.001473
grad_step = 000499, loss = 0.001490
grad_step = 000500, loss = 0.001514
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001542
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

  date_run                              2020-05-13 12:16:13.892120
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.218118
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 12:16:13.898969
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                    0.1153
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 12:16:13.907896
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.138648
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 12:16:13.914092
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.752031
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
0   2020-05-13 12:15:39.232503  ...    mean_absolute_error
1   2020-05-13 12:15:39.236799  ...     mean_squared_error
2   2020-05-13 12:15:39.240701  ...  median_absolute_error
3   2020-05-13 12:15:39.244836  ...               r2_score
4   2020-05-13 12:15:48.771973  ...    mean_absolute_error
5   2020-05-13 12:15:48.776444  ...     mean_squared_error
6   2020-05-13 12:15:48.780194  ...  median_absolute_error
7   2020-05-13 12:15:48.783818  ...               r2_score
8   2020-05-13 12:16:13.892120  ...    mean_absolute_error
9   2020-05-13 12:16:13.898969  ...     mean_squared_error
10  2020-05-13 12:16:13.907896  ...  median_absolute_error
11  2020-05-13 12:16:13.914092  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fba62ec1fd0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 32%|███▏      | 3211264/9912422 [00:00<00:00, 32102950.11it/s]9920512it [00:00, 33191750.54it/s]                             
0it [00:00, ?it/s]32768it [00:00, 551478.63it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 157630.83it/s]1654784it [00:00, 11229762.73it/s]                         
0it [00:00, ?it/s]8192it [00:00, 210538.90it/s]dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fba1586fe48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fba126bc0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fba1586fe48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fba126bc048> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fba126314a8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fba126bc0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fba1586fe48> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fba126bc048> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fba126314a8> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7fba62ec1fd0> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f348ea8f208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=70b7471f61a1826e8de83e7bf191953840ebf53a12e3316b2b99251d8e04166f
  Stored in directory: /tmp/pip-ephem-wheel-cache-1bc4aad3/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f342688a748> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 1810432/17464789 [==>...........................] - ETA: 0s
 5906432/17464789 [=========>....................] - ETA: 0s
10289152/17464789 [================>.............] - ETA: 0s
14557184/17464789 [========================>.....] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 12:17:42.653425: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 12:17:42.659790: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-13 12:17:42.660019: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560efa568df0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 12:17:42.660052: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 14s - loss: 7.6053 - accuracy: 0.5040
 2000/25000 [=>............................] - ETA: 10s - loss: 7.4750 - accuracy: 0.5125
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.4724 - accuracy: 0.5127 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.5823 - accuracy: 0.5055
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5562 - accuracy: 0.5072
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5618 - accuracy: 0.5068
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5878 - accuracy: 0.5051
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5574 - accuracy: 0.5071
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5848 - accuracy: 0.5053
10000/25000 [===========>..................] - ETA: 5s - loss: 7.5777 - accuracy: 0.5058
11000/25000 [============>.................] - ETA: 4s - loss: 7.5732 - accuracy: 0.5061
12000/25000 [=============>................] - ETA: 4s - loss: 7.5861 - accuracy: 0.5052
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6006 - accuracy: 0.5043
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5976 - accuracy: 0.5045
15000/25000 [=================>............] - ETA: 3s - loss: 7.6227 - accuracy: 0.5029
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6177 - accuracy: 0.5032
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6251 - accuracy: 0.5027
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6274 - accuracy: 0.5026
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6198 - accuracy: 0.5031
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6344 - accuracy: 0.5021
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6433 - accuracy: 0.5015
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6499 - accuracy: 0.5011
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 10s 398us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 12:18:00.361049
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 12:18:00.361049  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:03<90:38:41, 2.64kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:03<63:40:48, 3.76kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:03<44:37:22, 5.37kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:03<31:13:31, 7.66kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:03<21:47:30, 10.9kB/s].vector_cache/glove.6B.zip:   1%|          | 8.61M/862M [00:03<15:10:02, 15.6kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.1M/862M [00:03<10:34:32, 22.3kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.4M/862M [00:03<7:21:29, 31.9kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.6M/862M [00:04<5:08:00, 45.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.6M/862M [00:04<3:34:24, 65.0kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.2M/862M [00:04<2:29:33, 92.8kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.8M/862M [00:04<1:44:12, 132kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 37.9M/862M [00:04<1:12:41, 189kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.5M/862M [00:04<50:41, 270kB/s]  .vector_cache/glove.6B.zip:   5%|▌         | 46.4M/862M [00:04<35:25, 384kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.1M/862M [00:04<24:44, 546kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.1M/862M [00:05<19:04, 708kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:07<15:12, 883kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:07<12:27, 1.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.7M/862M [00:07<09:10, 1.46MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:09<08:56, 1.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:09<07:59, 1.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.0M/862M [00:09<05:57, 2.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:09<04:19, 3.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:11<6:39:58, 33.2kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.8M/862M [00:11<4:43:12, 46.9kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:11<3:18:45, 66.8kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.1M/862M [00:11<2:19:04, 95.3kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:13<1:41:01, 131kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:13<1:12:02, 183kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:13<50:40, 260kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:15<38:26, 342kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:15<29:27, 446kB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.9M/862M [00:15<21:10, 620kB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.9M/862M [00:15<14:58, 875kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:17<16:07, 812kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.4M/862M [00:17<12:37, 1.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.9M/862M [00:17<09:09, 1.43MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:19<09:25, 1.38MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:19<09:16, 1.40MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.1M/862M [00:19<07:03, 1.84MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.1M/862M [00:19<05:07, 2.53MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:20<09:00, 1.44MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:21<07:37, 1.70MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.2M/862M [00:21<05:39, 2.28MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.3M/862M [00:22<06:58, 1.85MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:23<06:11, 2.08MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.3M/862M [00:23<04:36, 2.79MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:24<06:15, 2.05MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:25<05:40, 2.25MB/s].vector_cache/glove.6B.zip:  11%|█         | 95.4M/862M [00:25<04:17, 2.98MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:26<06:01, 2.12MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.0M/862M [00:26<05:29, 2.32MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:27<04:09, 3.05MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<05:55, 2.14MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:28<06:38, 1.91MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:29<05:14, 2.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:29<03:49, 3.30MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<08:54, 1.42MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:30<07:32, 1.67MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:31<05:35, 2.25MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<06:50, 1.83MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:32<07:27, 1.68MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:32<05:47, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:33<04:10, 2.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:34<13:35, 917kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<15:43, 793kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:36<12:16, 1.01MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 119M/862M [00:36<09:52, 1.25MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:36<07:13, 1.71MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:38<07:52, 1.57MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:38<06:42, 1.84MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:38<05:00, 2.46MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:40<06:27, 1.90MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<07:00, 1.75MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:40<05:31, 2.22MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<05:49, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:42<05:19, 2.29MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:42<04:02, 3.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<05:39, 2.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:44<06:25, 1.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:44<05:06, 2.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<05:30, 2.19MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:46<05:06, 2.36MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:46<03:52, 3.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<05:29, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:48<05:04, 2.36MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:48<03:50, 3.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<05:30, 2.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:50<06:17, 1.89MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:50<04:55, 2.42MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:50<03:35, 3.30MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<08:42, 1.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:52<07:18, 1.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:52<05:24, 2.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<06:31, 1.80MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:54<06:59, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:54<05:24, 2.18MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:54<03:56, 2.98MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:56<08:30, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:56<07:11, 1.63MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:56<05:16, 2.21MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:58<06:25, 1.81MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:58<05:41, 2.04MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:58<04:13, 2.75MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:00<05:43, 2.02MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:00<05:09, 2.24MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:00<03:53, 2.96MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<05:26, 2.11MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<04:59, 2.31MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:02<03:46, 3.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<05:20, 2.14MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:04<04:54, 2.33MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:04<03:42, 3.07MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<05:17, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<06:00, 1.89MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<04:46, 2.38MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<05:09, 2.19MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<04:46, 2.36MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:08<03:37, 3.11MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<05:30, 2.04MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:10<06:16, 1.79MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<04:53, 2.30MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:10<03:34, 3.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<06:59, 1.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:12<06:01, 1.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:12<04:29, 2.48MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:14<05:44, 1.93MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<06:15, 1.77MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<04:56, 2.24MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<03:32, 3.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:16<10:47:05, 17.0kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:16<7:33:48, 24.3kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<5:17:12, 34.7kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:18<3:43:52, 49.0kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<2:37:43, 69.4kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:18<1:50:25, 99.0kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<1:19:37, 137kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<57:55, 188kB/s]  .vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<41:03, 265kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<30:21, 356kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<22:20, 484kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:22<15:50, 681kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:23<13:35, 792kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<10:34, 1.02MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:24<07:39, 1.40MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:25<07:52, 1.36MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<06:25, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:26<04:41, 2.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:26<03:25, 3.10MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:27<42:32, 250kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<31:54, 333kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<22:50, 464kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:29<17:37, 598kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<13:24, 787kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:30<09:35, 1.10MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:31<09:09, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:31<07:28, 1.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:32<05:26, 1.92MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<06:17, 1.66MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:33<05:27, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:34<04:04, 2.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<05:18, 1.95MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:35<04:45, 2.17MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:35<03:32, 2.91MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<04:53, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:37<05:31, 1.86MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:37<04:17, 2.39MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:38<03:09, 3.24MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<06:24, 1.59MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:39<05:20, 1.91MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:39<03:56, 2.58MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:40<02:55, 3.47MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<48:31, 209kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:41<36:01, 281kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:41<25:41, 394kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<19:31, 515kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<14:41, 685kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:43<10:30, 955kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:45<09:40, 1.03MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:45<07:46, 1.28MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:45<05:41, 1.75MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:47<06:19, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:47<05:16, 1.88MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:47<03:53, 2.54MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:47<02:51, 3.46MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:49<41:45, 236kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<31:13, 316kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:49<22:19, 441kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<17:08, 572kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:51<12:58, 754kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:51<09:16, 1.05MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<08:45, 1.11MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:53<08:06, 1.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:53<06:09, 1.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:53<04:24, 2.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:55<9:17:40, 17.3kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:55<6:31:06, 24.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:55<4:33:16, 35.2kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:57<3:12:48, 49.7kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:57<2:15:51, 70.5kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:57<1:35:03, 101kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:59<1:08:33, 139kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:59<49:51, 191kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:59<35:19, 269kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:59<24:41, 383kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:01<1:18:32, 120kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:01<55:55, 169kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:01<39:14, 240kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:03<29:33, 317kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:03<21:39, 433kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:03<15:19, 610kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:05<12:52, 723kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:05<09:57, 935kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:05<07:10, 1.29MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:07<07:12, 1.28MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<06:54, 1.34MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:07<05:14, 1.76MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:07<03:45, 2.45MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:09<16:52, 544kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:09<12:44, 719kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:09<09:08, 1.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:11<08:30, 1.07MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:11<07:47, 1.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:11<05:52, 1.55MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:11<04:11, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:13<13:43, 659kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:13<10:30, 859kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:13<07:34, 1.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:14<07:23, 1.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:15<06:59, 1.28MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:15<05:20, 1.68MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:16<05:09, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:17<04:30, 1.97MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:17<03:22, 2.63MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:18<04:25, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:19<04:57, 1.78MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:19<03:52, 2.28MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:19<02:47, 3.14MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:20<12:18, 712kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:21<09:31, 920kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:21<06:52, 1.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:22<06:49, 1.27MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:22<06:32, 1.33MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:23<04:57, 1.75MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:23<03:34, 2.41MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:24<06:23, 1.35MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:24<05:20, 1.61MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:25<03:57, 2.18MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:26<04:46, 1.80MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:26<04:12, 2.04MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:27<03:08, 2.71MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<04:12, 2.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:28<04:39, 1.82MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:28<03:41, 2.30MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:29<02:39, 3.16MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<8:08:36, 17.2kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:30<5:42:38, 24.6kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:30<3:59:17, 35.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:32<2:48:44, 49.5kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:32<1:59:45, 69.7kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:32<1:24:07, 99.1kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:34<59:52, 138kB/s]   .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:34<42:43, 194kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:34<30:01, 275kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:36<22:51, 360kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:36<17:39, 465kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:36<12:41, 646kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:36<08:58, 910kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:38<09:33, 853kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:38<07:30, 1.08MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:38<05:26, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:40<05:41, 1.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<05:37, 1.44MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:40<04:17, 1.88MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:40<03:05, 2.60MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:42<07:22, 1.09MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:42<05:58, 1.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:42<04:22, 1.82MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:44<04:55, 1.62MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:44<05:02, 1.57MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:44<03:56, 2.01MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<02:50, 2.77MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:46<58:18, 135kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:46<41:35, 189kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:46<29:13, 268kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:48<22:10, 352kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:48<17:06, 456kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:48<12:20, 631kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:50<09:50, 786kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:50<07:40, 1.01MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:50<05:30, 1.40MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:52<05:38, 1.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:52<05:30, 1.39MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:52<04:11, 1.83MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:52<02:59, 2.54MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:54<11:39, 652kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:54<08:56, 850kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:54<06:23, 1.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:56<06:13, 1.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<05:52, 1.28MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<04:26, 1.69MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:56<03:10, 2.35MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:58<12:28, 598kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:58<09:28, 786kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:58<06:48, 1.09MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<06:28, 1.14MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<06:01, 1.23MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<04:35, 1.61MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:00<03:17, 2.23MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<1:02:12, 118kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:02<44:14, 165kB/s]  .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:02<31:03, 235kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:04<23:19, 311kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:04<17:47, 408kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:04<12:44, 568kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:04<08:57, 804kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:05<11:16, 638kB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:06<08:37, 833kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:06<06:10, 1.16MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:07<05:58, 1.19MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<05:37, 1.26MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:08<04:17, 1.65MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:08<03:03, 2.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:09<53:37, 131kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:10<38:13, 184kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:10<26:50, 262kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<20:19, 343kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:11<15:38, 446kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:12<11:17, 617kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:12<07:55, 872kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:13<1:02:04, 111kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:13<44:08, 157kB/s]  .vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:14<30:56, 222kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:15<23:07, 296kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:15<17:31, 390kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:16<12:32, 544kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:16<08:49, 770kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<10:05, 672kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:17<07:45, 873kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:18<05:35, 1.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<05:26, 1.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:19<05:11, 1.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:19<03:54, 1.71MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:20<02:49, 2.36MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<05:09, 1.29MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:21<04:18, 1.54MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:21<03:08, 2.10MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<03:44, 1.76MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:23<03:56, 1.66MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:23<03:05, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<02:13, 2.93MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:25<09:28, 687kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:25<07:17, 892kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:25<05:14, 1.23MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:27<05:09, 1.25MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:27<04:09, 1.54MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:27<03:06, 2.07MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:27<02:14, 2.84MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:29<48:57, 130kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:29<35:32, 179kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:29<25:06, 253kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:29<17:32, 360kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:31<17:29, 360kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:31<12:45, 493kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:31<09:02, 693kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:31<06:22, 978kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:33<29:34, 211kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:33<21:18, 292kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:33<14:59, 413kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:35<11:53, 518kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:35<09:33, 644kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:35<06:57, 884kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:35<04:54, 1.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:37<07:58, 763kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:37<06:12, 980kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:37<04:29, 1.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:39<04:32, 1.33MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:39<03:47, 1.59MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:39<02:46, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:41<03:20, 1.78MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:41<03:29, 1.71MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:41<02:44, 2.17MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:43<02:51, 2.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:43<02:36, 2.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:43<01:58, 2.97MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:45<02:43, 2.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:45<03:05, 1.88MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:45<02:24, 2.42MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:45<01:45, 3.27MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:47<03:22, 1.70MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:47<02:56, 1.95MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:47<02:11, 2.60MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:49<02:51, 1.98MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:49<03:09, 1.80MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:49<02:29, 2.28MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:49<01:47, 3.14MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:51<41:47, 134kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:51<29:47, 188kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:51<20:54, 267kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:53<15:50, 350kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:53<11:37, 476kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:53<08:13, 670kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<07:01, 779kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<05:27, 1.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:55<03:56, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:56<04:01, 1.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:57<03:55, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:57<03:00, 1.79MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:58<02:57, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:59<02:37, 2.04MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:59<01:57, 2.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:00<02:36, 2.02MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:01<02:21, 2.23MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:01<01:46, 2.95MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:02<02:28, 2.11MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:03<02:47, 1.87MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:03<02:12, 2.34MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:04<02:21, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:04<02:10, 2.36MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:05<01:37, 3.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:06<02:19, 2.18MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:06<02:03, 2.45MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:07<01:32, 3.27MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:07<01:08, 4.38MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<19:45, 253kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:08<14:18, 348kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:09<10:05, 491kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<08:10, 602kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:10<06:12, 791kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:10<04:27, 1.10MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:12<04:14, 1.14MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:12<03:58, 1.22MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:12<03:01, 1.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:13<02:08, 2.23MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:14<07:11, 665kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:14<05:55, 807kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:15<04:21, 1.09MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:15<03:04, 1.54MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:16<13:38, 345kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:16<10:03, 468kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:17<07:08, 655kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:18<05:57, 779kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:18<05:09, 901kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:19<03:48, 1.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:19<02:45, 1.67MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<03:03, 1.50MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:20<02:36, 1.75MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:21<01:54, 2.38MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<02:23, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:22<02:35, 1.73MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:22<02:00, 2.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:23<01:26, 3.08MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<04:58, 892kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:24<03:56, 1.12MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:24<02:51, 1.54MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:26<03:00, 1.45MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:26<02:32, 1.71MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:26<01:52, 2.32MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<02:19, 1.85MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:28<02:30, 1.72MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:28<01:57, 2.18MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:30<02:02, 2.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:30<01:51, 2.27MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:30<01:24, 3.00MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<01:56, 2.14MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<02:12, 1.89MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:32<01:43, 2.40MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:32<01:15, 3.28MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<02:55, 1.40MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:34<02:28, 1.65MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:34<01:49, 2.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:36<02:12, 1.83MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:36<02:21, 1.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:36<01:49, 2.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:36<01:19, 3.00MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:38<02:29, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:38<02:08, 1.84MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:38<01:35, 2.46MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:40<02:01, 1.93MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:40<02:12, 1.76MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:40<01:42, 2.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:40<01:15, 3.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:42<02:05, 1.82MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:42<01:51, 2.06MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:42<01:22, 2.76MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<01:50, 2.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<02:03, 1.83MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:44<01:37, 2.30MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:44<01:09, 3.17MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:46<3:32:19, 17.3kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:46<2:28:45, 24.7kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:46<1:43:28, 35.3kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<1:12:34, 49.8kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:48<51:04, 70.6kB/s]  .vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:48<35:35, 101kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:50<25:30, 139kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:50<18:33, 191kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:50<13:05, 270kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:50<09:08, 383kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:52<07:36, 457kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:52<05:40, 611kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:52<04:01, 857kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:54<03:35, 950kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:54<03:11, 1.06MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:54<02:22, 1.42MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:54<01:41, 1.98MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:56<02:58, 1.12MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:56<02:24, 1.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:56<01:45, 1.88MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:58<01:59, 1.64MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:58<02:03, 1.59MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:58<01:34, 2.07MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:58<01:07, 2.85MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:59<03:22, 948kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:00<02:41, 1.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:00<01:56, 1.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:01<02:04, 1.51MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:02<02:05, 1.49MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:02<01:35, 1.95MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:02<01:09, 2.67MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:03<01:53, 1.62MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:04<01:38, 1.87MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:04<01:12, 2.50MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:05<01:32, 1.94MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:06<01:24, 2.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:06<01:02, 2.83MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:07<01:25, 2.07MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:07<01:34, 1.86MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:08<01:13, 2.38MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:08<00:53, 3.24MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<01:57, 1.46MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:09<01:39, 1.72MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:10<01:12, 2.33MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<01:29, 1.86MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:11<01:17, 2.16MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:11<00:57, 2.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:12<00:41, 3.90MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<11:25, 239kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:13<08:32, 319kB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:13<06:05, 445kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:15<04:36, 576kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:15<03:26, 770kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:15<02:26, 1.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:16<01:43, 1.49MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<09:19, 277kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:17<06:46, 381kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:17<04:45, 537kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:19<03:52, 649kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:19<03:13, 780kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:19<02:21, 1.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:19<01:39, 1.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:21<02:16, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:21<01:49, 1.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:21<01:19, 1.83MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:23<01:28, 1.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:23<01:16, 1.87MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:23<00:56, 2.50MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:25<01:11, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:25<01:04, 2.16MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:25<00:47, 2.89MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:27<01:04, 2.08MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:27<01:12, 1.85MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:27<00:57, 2.33MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:29<01:00, 2.16MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:29<00:55, 2.34MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:29<00:41, 3.07MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:31<00:58, 2.17MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:31<01:06, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:31<00:51, 2.42MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:31<00:36, 3.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:33<02:03, 991kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:33<01:38, 1.24MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:33<01:10, 1.70MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:35<01:16, 1.55MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:35<01:17, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:35<00:59, 1.96MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<00:41, 2.73MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:37<05:32, 343kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:37<04:01, 470kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:37<02:50, 659kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:58, 931kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<06:39, 275kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:39<05:01, 364kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:39<03:34, 507kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:39<02:27, 718kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:41<14:28, 122kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:41<10:16, 171kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:41<07:07, 243kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:43<05:16, 321kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:43<04:01, 419kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:43<02:53, 581kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<01:58, 822kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:45<1:34:32, 17.2kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:45<1:06:05, 24.5kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:45<45:33, 35.0kB/s]  .vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:47<31:31, 49.4kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:47<22:20, 69.6kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:47<15:34, 98.9kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:47<10:37, 141kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:49<08:37, 173kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:49<06:08, 242kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:49<04:15, 342kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:51<03:14, 438kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:51<02:33, 555kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:51<01:50, 762kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:52<01:27, 925kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:53<01:09, 1.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:53<00:49, 1.61MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:54<00:51, 1.49MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:55<00:43, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:55<00:31, 2.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:56<00:38, 1.87MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:57<00:34, 2.10MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:57<00:25, 2.79MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:58<00:33, 2.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:58<00:37, 1.83MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:59<00:28, 2.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:59<00:20, 3.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<01:10, 921kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:00<00:55, 1.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:01<00:39, 1.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:02<00:40, 1.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:02<00:40, 1.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:03<00:30, 1.94MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:03<00:21, 2.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:04<00:33, 1.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:04<00:29, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:05<00:21, 2.56MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:26, 1.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:06<00:23, 2.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:06<00:17, 2.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:23, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:08<00:25, 1.86MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:08<00:19, 2.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:09<00:13, 3.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:10<00:39, 1.13MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:10<00:31, 1.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:10<00:22, 1.88MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:24, 1.65MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:12<00:24, 1.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:12<00:19, 2.04MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:14<00:18, 1.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:14<00:16, 2.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:14<00:11, 2.91MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:16<00:15, 2.10MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:17, 1.85MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:16<00:13, 2.35MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:16<00:08, 3.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<01:19, 346kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:18<00:57, 471kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:18<00:38, 662kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:30, 773kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:20<00:25, 903kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:20<00:18, 1.22MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:20<00:11, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:18, 1.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:22<00:15, 1.27MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:22<00:10, 1.73MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:09, 1.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:24<00:07, 1.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:24<00:05, 2.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:24<00:03, 3.45MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:26<00:58, 190kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:26<00:40, 265kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:26<00:24, 375kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:14, 475kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:28<00:11, 597kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:28<00:07, 821kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:28<00:03, 1.15MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:03, 934kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:30<00:02, 1.17MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:30<00:00, 1.62MB/s].vector_cache/glove.6B.zip: 862MB [06:30, 2.21MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 715/400000 [00:00<00:55, 7144.59it/s]  0%|          | 1451/400000 [00:00<00:55, 7207.47it/s]  1%|          | 2195/400000 [00:00<00:54, 7274.53it/s]  1%|          | 2947/400000 [00:00<00:54, 7346.43it/s]  1%|          | 3683/400000 [00:00<00:53, 7349.57it/s]  1%|          | 4427/400000 [00:00<00:53, 7374.66it/s]  1%|▏         | 5181/400000 [00:00<00:53, 7422.95it/s]  1%|▏         | 5956/400000 [00:00<00:52, 7515.97it/s]  2%|▏         | 6729/400000 [00:00<00:51, 7578.53it/s]  2%|▏         | 7472/400000 [00:01<00:52, 7531.41it/s]  2%|▏         | 8206/400000 [00:01<00:52, 7469.34it/s]  2%|▏         | 8947/400000 [00:01<00:52, 7448.99it/s]  2%|▏         | 9684/400000 [00:01<00:52, 7423.29it/s]  3%|▎         | 10450/400000 [00:01<00:52, 7490.39it/s]  3%|▎         | 11242/400000 [00:01<00:51, 7611.70it/s]  3%|▎         | 12058/400000 [00:01<00:49, 7766.31it/s]  3%|▎         | 12834/400000 [00:01<00:50, 7687.92it/s]  3%|▎         | 13603/400000 [00:01<00:50, 7621.35it/s]  4%|▎         | 14365/400000 [00:01<00:50, 7592.50it/s]  4%|▍         | 15125/400000 [00:02<00:50, 7589.17it/s]  4%|▍         | 15892/400000 [00:02<00:50, 7613.18it/s]  4%|▍         | 16654/400000 [00:02<00:50, 7532.24it/s]  4%|▍         | 17408/400000 [00:02<00:51, 7446.91it/s]  5%|▍         | 18154/400000 [00:02<00:51, 7392.22it/s]  5%|▍         | 18909/400000 [00:02<00:51, 7438.49it/s]  5%|▍         | 19656/400000 [00:02<00:51, 7446.23it/s]  5%|▌         | 20401/400000 [00:02<00:51, 7439.68it/s]  5%|▌         | 21159/400000 [00:02<00:50, 7479.32it/s]  5%|▌         | 21916/400000 [00:02<00:50, 7505.62it/s]  6%|▌         | 22669/400000 [00:03<00:50, 7510.43it/s]  6%|▌         | 23421/400000 [00:03<00:50, 7462.24it/s]  6%|▌         | 24168/400000 [00:03<00:51, 7318.81it/s]  6%|▌         | 24915/400000 [00:03<00:50, 7358.37it/s]  6%|▋         | 25652/400000 [00:03<00:50, 7354.33it/s]  7%|▋         | 26411/400000 [00:03<00:50, 7420.62it/s]  7%|▋         | 27175/400000 [00:03<00:49, 7482.04it/s]  7%|▋         | 27938/400000 [00:03<00:49, 7524.20it/s]  7%|▋         | 28695/400000 [00:03<00:49, 7536.46it/s]  7%|▋         | 29453/400000 [00:03<00:49, 7548.88it/s]  8%|▊         | 30209/400000 [00:04<00:49, 7406.62it/s]  8%|▊         | 30963/400000 [00:04<00:49, 7442.63it/s]  8%|▊         | 31708/400000 [00:04<00:49, 7373.76it/s]  8%|▊         | 32446/400000 [00:04<00:50, 7343.50it/s]  8%|▊         | 33199/400000 [00:04<00:49, 7397.43it/s]  8%|▊         | 33964/400000 [00:04<00:48, 7471.34it/s]  9%|▊         | 34712/400000 [00:04<00:51, 7077.40it/s]  9%|▉         | 35476/400000 [00:04<00:50, 7235.68it/s]  9%|▉         | 36248/400000 [00:04<00:49, 7372.51it/s]  9%|▉         | 36989/400000 [00:04<00:49, 7315.88it/s]  9%|▉         | 37724/400000 [00:05<00:50, 7231.33it/s] 10%|▉         | 38452/400000 [00:05<00:49, 7245.50it/s] 10%|▉         | 39181/400000 [00:05<00:49, 7258.40it/s] 10%|▉         | 39908/400000 [00:05<00:49, 7236.12it/s] 10%|█         | 40633/400000 [00:05<00:49, 7238.35it/s] 10%|█         | 41376/400000 [00:05<00:49, 7294.66it/s] 11%|█         | 42106/400000 [00:05<00:50, 7098.89it/s] 11%|█         | 42845/400000 [00:05<00:49, 7182.26it/s] 11%|█         | 43583/400000 [00:05<00:49, 7238.48it/s] 11%|█         | 44308/400000 [00:05<00:50, 7060.60it/s] 11%|█▏        | 45016/400000 [00:06<00:50, 7065.43it/s] 11%|█▏        | 45724/400000 [00:06<00:51, 6870.44it/s] 12%|█▏        | 46414/400000 [00:06<00:51, 6863.51it/s] 12%|█▏        | 47147/400000 [00:06<00:50, 6996.03it/s] 12%|█▏        | 47859/400000 [00:06<00:50, 7030.19it/s] 12%|█▏        | 48582/400000 [00:06<00:49, 7087.51it/s] 12%|█▏        | 49302/400000 [00:06<00:49, 7120.33it/s] 13%|█▎        | 50035/400000 [00:06<00:48, 7180.24it/s] 13%|█▎        | 50788/400000 [00:06<00:47, 7280.04it/s] 13%|█▎        | 51517/400000 [00:06<00:48, 7243.43it/s] 13%|█▎        | 52258/400000 [00:07<00:47, 7290.97it/s] 13%|█▎        | 52993/400000 [00:07<00:47, 7306.35it/s] 13%|█▎        | 53745/400000 [00:07<00:47, 7366.51it/s] 14%|█▎        | 54498/400000 [00:07<00:46, 7413.10it/s] 14%|█▍        | 55244/400000 [00:07<00:46, 7426.30it/s] 14%|█▍        | 55987/400000 [00:07<00:48, 7150.40it/s] 14%|█▍        | 56736/400000 [00:07<00:47, 7248.33it/s] 14%|█▍        | 57482/400000 [00:07<00:46, 7309.97it/s] 15%|█▍        | 58215/400000 [00:07<00:46, 7309.76it/s] 15%|█▍        | 58948/400000 [00:08<00:46, 7310.71it/s] 15%|█▍        | 59693/400000 [00:08<00:46, 7351.59it/s] 15%|█▌        | 60429/400000 [00:08<00:46, 7336.62it/s] 15%|█▌        | 61164/400000 [00:08<00:46, 7254.88it/s] 15%|█▌        | 61890/400000 [00:08<00:47, 7090.79it/s] 16%|█▌        | 62601/400000 [00:08<00:47, 7056.33it/s] 16%|█▌        | 63332/400000 [00:08<00:47, 7128.47it/s] 16%|█▌        | 64046/400000 [00:08<00:47, 7110.47it/s] 16%|█▌        | 64779/400000 [00:08<00:46, 7173.18it/s] 16%|█▋        | 65515/400000 [00:08<00:46, 7227.38it/s] 17%|█▋        | 66239/400000 [00:09<00:46, 7143.50it/s] 17%|█▋        | 66988/400000 [00:09<00:45, 7240.58it/s] 17%|█▋        | 67713/400000 [00:09<00:46, 7177.72it/s] 17%|█▋        | 68432/400000 [00:09<00:47, 7007.59it/s] 17%|█▋        | 69135/400000 [00:09<00:47, 6956.63it/s] 17%|█▋        | 69850/400000 [00:09<00:47, 6924.39it/s] 18%|█▊        | 70554/400000 [00:09<00:47, 6957.32it/s] 18%|█▊        | 71295/400000 [00:09<00:46, 7084.71it/s] 18%|█▊        | 72060/400000 [00:09<00:45, 7242.45it/s] 18%|█▊        | 72789/400000 [00:09<00:45, 7253.40it/s] 18%|█▊        | 73516/400000 [00:10<00:45, 7216.23it/s] 19%|█▊        | 74265/400000 [00:10<00:44, 7294.56it/s] 19%|█▊        | 74996/400000 [00:10<00:44, 7282.37it/s] 19%|█▉        | 75747/400000 [00:10<00:44, 7347.78it/s] 19%|█▉        | 76486/400000 [00:10<00:43, 7358.11it/s] 19%|█▉        | 77236/400000 [00:10<00:43, 7399.51it/s] 19%|█▉        | 77977/400000 [00:10<00:43, 7399.61it/s] 20%|█▉        | 78733/400000 [00:10<00:43, 7444.45it/s] 20%|█▉        | 79484/400000 [00:10<00:42, 7462.89it/s] 20%|██        | 80231/400000 [00:10<00:43, 7417.33it/s] 20%|██        | 80973/400000 [00:11<00:43, 7315.17it/s] 20%|██        | 81708/400000 [00:11<00:43, 7324.25it/s] 21%|██        | 82441/400000 [00:11<00:44, 7211.76it/s] 21%|██        | 83163/400000 [00:11<00:45, 6989.42it/s] 21%|██        | 83890/400000 [00:11<00:44, 7071.17it/s] 21%|██        | 84629/400000 [00:11<00:44, 7162.53it/s] 21%|██▏       | 85364/400000 [00:11<00:43, 7216.18it/s] 22%|██▏       | 86087/400000 [00:11<00:43, 7163.19it/s] 22%|██▏       | 86805/400000 [00:11<00:43, 7154.29it/s] 22%|██▏       | 87551/400000 [00:11<00:43, 7241.70it/s] 22%|██▏       | 88276/400000 [00:12<00:43, 7196.65it/s] 22%|██▏       | 88997/400000 [00:12<00:43, 7161.74it/s] 22%|██▏       | 89714/400000 [00:12<00:43, 7145.12it/s] 23%|██▎       | 90463/400000 [00:12<00:42, 7245.04it/s] 23%|██▎       | 91189/400000 [00:12<00:43, 7084.92it/s] 23%|██▎       | 91899/400000 [00:12<00:44, 6960.23it/s] 23%|██▎       | 92649/400000 [00:12<00:43, 7112.77it/s] 23%|██▎       | 93370/400000 [00:12<00:42, 7139.50it/s] 24%|██▎       | 94086/400000 [00:12<00:43, 7069.75it/s] 24%|██▎       | 94794/400000 [00:13<00:43, 6983.53it/s] 24%|██▍       | 95530/400000 [00:13<00:42, 7090.17it/s] 24%|██▍       | 96274/400000 [00:13<00:42, 7189.65it/s] 24%|██▍       | 96995/400000 [00:13<00:42, 7185.87it/s] 24%|██▍       | 97715/400000 [00:13<00:42, 7160.86it/s] 25%|██▍       | 98432/400000 [00:13<00:42, 7139.20it/s] 25%|██▍       | 99160/400000 [00:13<00:41, 7179.87it/s] 25%|██▍       | 99906/400000 [00:13<00:41, 7259.96it/s] 25%|██▌       | 100645/400000 [00:13<00:41, 7295.56it/s] 25%|██▌       | 101396/400000 [00:13<00:40, 7358.43it/s] 26%|██▌       | 102133/400000 [00:14<00:40, 7343.90it/s] 26%|██▌       | 102868/400000 [00:14<00:40, 7330.11it/s] 26%|██▌       | 103602/400000 [00:14<00:40, 7297.41it/s] 26%|██▌       | 104343/400000 [00:14<00:40, 7328.66it/s] 26%|██▋       | 105077/400000 [00:14<00:40, 7276.67it/s] 26%|██▋       | 105805/400000 [00:14<00:40, 7191.63it/s] 27%|██▋       | 106525/400000 [00:14<00:41, 7149.31it/s] 27%|██▋       | 107290/400000 [00:14<00:40, 7290.41it/s] 27%|██▋       | 108028/400000 [00:14<00:39, 7316.09it/s] 27%|██▋       | 108764/400000 [00:14<00:39, 7328.82it/s] 27%|██▋       | 109498/400000 [00:15<00:40, 7149.54it/s] 28%|██▊       | 110215/400000 [00:15<00:40, 7151.39it/s] 28%|██▊       | 110945/400000 [00:15<00:40, 7194.72it/s] 28%|██▊       | 111671/400000 [00:15<00:39, 7212.40it/s] 28%|██▊       | 112393/400000 [00:15<00:40, 7168.07it/s] 28%|██▊       | 113127/400000 [00:15<00:39, 7217.31it/s] 28%|██▊       | 113850/400000 [00:15<00:39, 7206.86it/s] 29%|██▊       | 114597/400000 [00:15<00:39, 7282.86it/s] 29%|██▉       | 115353/400000 [00:15<00:38, 7362.48it/s] 29%|██▉       | 116090/400000 [00:15<00:38, 7311.01it/s] 29%|██▉       | 116826/400000 [00:16<00:38, 7322.60it/s] 29%|██▉       | 117559/400000 [00:16<00:38, 7300.96it/s] 30%|██▉       | 118306/400000 [00:16<00:38, 7350.61it/s] 30%|██▉       | 119046/400000 [00:16<00:38, 7364.39it/s] 30%|██▉       | 119810/400000 [00:16<00:37, 7444.35it/s] 30%|███       | 120555/400000 [00:16<00:37, 7369.46it/s] 30%|███       | 121293/400000 [00:16<00:38, 7296.32it/s] 31%|███       | 122026/400000 [00:16<00:38, 7304.15it/s] 31%|███       | 122770/400000 [00:16<00:37, 7341.78it/s] 31%|███       | 123532/400000 [00:16<00:37, 7421.84it/s] 31%|███       | 124275/400000 [00:17<00:37, 7423.75it/s] 31%|███▏      | 125018/400000 [00:17<00:37, 7395.21it/s] 31%|███▏      | 125772/400000 [00:17<00:36, 7435.55it/s] 32%|███▏      | 126520/400000 [00:17<00:36, 7448.56it/s] 32%|███▏      | 127266/400000 [00:17<00:36, 7423.50it/s] 32%|███▏      | 128009/400000 [00:17<00:36, 7383.82it/s] 32%|███▏      | 128748/400000 [00:17<00:36, 7384.42it/s] 32%|███▏      | 129493/400000 [00:17<00:36, 7402.70it/s] 33%|███▎      | 130245/400000 [00:17<00:36, 7435.06it/s] 33%|███▎      | 130990/400000 [00:17<00:36, 7436.73it/s] 33%|███▎      | 131734/400000 [00:18<00:36, 7382.63it/s] 33%|███▎      | 132473/400000 [00:18<00:36, 7374.91it/s] 33%|███▎      | 133217/400000 [00:18<00:36, 7393.16it/s] 33%|███▎      | 133957/400000 [00:18<00:36, 7376.79it/s] 34%|███▎      | 134712/400000 [00:18<00:35, 7427.86it/s] 34%|███▍      | 135470/400000 [00:18<00:35, 7472.19it/s] 34%|███▍      | 136218/400000 [00:18<00:35, 7330.34it/s] 34%|███▍      | 136973/400000 [00:18<00:35, 7392.25it/s] 34%|███▍      | 137725/400000 [00:18<00:35, 7429.95it/s] 35%|███▍      | 138493/400000 [00:18<00:34, 7501.79it/s] 35%|███▍      | 139244/400000 [00:19<00:34, 7453.55it/s] 35%|███▍      | 139990/400000 [00:19<00:35, 7411.99it/s] 35%|███▌      | 140732/400000 [00:19<00:35, 7392.20it/s] 35%|███▌      | 141475/400000 [00:19<00:34, 7400.60it/s] 36%|███▌      | 142226/400000 [00:19<00:34, 7432.82it/s] 36%|███▌      | 142970/400000 [00:19<00:34, 7434.17it/s] 36%|███▌      | 143714/400000 [00:19<00:35, 7181.71it/s] 36%|███▌      | 144470/400000 [00:19<00:35, 7289.24it/s] 36%|███▋      | 145231/400000 [00:19<00:34, 7380.89it/s] 36%|███▋      | 145978/400000 [00:19<00:34, 7407.22it/s] 37%|███▋      | 146720/400000 [00:20<00:34, 7324.26it/s] 37%|███▋      | 147454/400000 [00:20<00:34, 7289.00it/s] 37%|███▋      | 148204/400000 [00:20<00:34, 7349.79it/s] 37%|███▋      | 148971/400000 [00:20<00:33, 7440.71it/s] 37%|███▋      | 149716/400000 [00:20<00:33, 7434.25it/s] 38%|███▊      | 150461/400000 [00:20<00:33, 7436.34it/s] 38%|███▊      | 151205/400000 [00:20<00:33, 7408.76it/s] 38%|███▊      | 151955/400000 [00:20<00:33, 7434.90it/s] 38%|███▊      | 152700/400000 [00:20<00:33, 7437.68it/s] 38%|███▊      | 153444/400000 [00:20<00:33, 7386.08it/s] 39%|███▊      | 154189/400000 [00:21<00:33, 7403.79it/s] 39%|███▊      | 154930/400000 [00:21<00:33, 7290.64it/s] 39%|███▉      | 155660/400000 [00:21<00:33, 7204.50it/s] 39%|███▉      | 156382/400000 [00:21<00:34, 7155.53it/s] 39%|███▉      | 157114/400000 [00:21<00:33, 7204.09it/s] 39%|███▉      | 157844/400000 [00:21<00:33, 7230.21it/s] 40%|███▉      | 158588/400000 [00:21<00:33, 7289.73it/s] 40%|███▉      | 159332/400000 [00:21<00:32, 7332.50it/s] 40%|████      | 160066/400000 [00:21<00:32, 7330.68it/s] 40%|████      | 160831/400000 [00:22<00:32, 7423.33it/s] 40%|████      | 161574/400000 [00:22<00:32, 7342.18it/s] 41%|████      | 162309/400000 [00:22<00:32, 7306.62it/s] 41%|████      | 163041/400000 [00:22<00:32, 7272.17it/s] 41%|████      | 163772/400000 [00:22<00:32, 7280.86it/s] 41%|████      | 164501/400000 [00:22<00:32, 7254.59it/s] 41%|████▏     | 165227/400000 [00:22<00:32, 7224.52it/s] 41%|████▏     | 165950/400000 [00:22<00:32, 7122.01it/s] 42%|████▏     | 166670/400000 [00:22<00:32, 7142.96it/s] 42%|████▏     | 167385/400000 [00:22<00:32, 7120.67it/s] 42%|████▏     | 168099/400000 [00:23<00:32, 7125.28it/s] 42%|████▏     | 168812/400000 [00:23<00:33, 6980.65it/s] 42%|████▏     | 169524/400000 [00:23<00:32, 7020.95it/s] 43%|████▎     | 170256/400000 [00:23<00:32, 7107.90it/s] 43%|████▎     | 171024/400000 [00:23<00:31, 7269.28it/s] 43%|████▎     | 171789/400000 [00:23<00:30, 7377.60it/s] 43%|████▎     | 172529/400000 [00:23<00:30, 7363.24it/s] 43%|████▎     | 173267/400000 [00:23<00:30, 7354.04it/s] 44%|████▎     | 174004/400000 [00:23<00:30, 7352.57it/s] 44%|████▎     | 174743/400000 [00:23<00:30, 7363.07it/s] 44%|████▍     | 175524/400000 [00:24<00:29, 7490.45it/s] 44%|████▍     | 176286/400000 [00:24<00:29, 7528.07it/s] 44%|████▍     | 177040/400000 [00:24<00:29, 7502.55it/s] 44%|████▍     | 177791/400000 [00:24<00:29, 7438.00it/s] 45%|████▍     | 178536/400000 [00:24<00:29, 7429.13it/s] 45%|████▍     | 179302/400000 [00:24<00:29, 7493.75it/s] 45%|████▌     | 180052/400000 [00:24<00:29, 7410.71it/s] 45%|████▌     | 180794/400000 [00:24<00:30, 7294.06it/s] 45%|████▌     | 181537/400000 [00:24<00:29, 7331.44it/s] 46%|████▌     | 182291/400000 [00:24<00:29, 7392.20it/s] 46%|████▌     | 183044/400000 [00:25<00:29, 7432.67it/s] 46%|████▌     | 183789/400000 [00:25<00:29, 7436.22it/s] 46%|████▌     | 184560/400000 [00:25<00:28, 7514.07it/s] 46%|████▋     | 185318/400000 [00:25<00:28, 7531.68it/s] 47%|████▋     | 186105/400000 [00:25<00:28, 7629.90it/s] 47%|████▋     | 186879/400000 [00:25<00:27, 7660.79it/s] 47%|████▋     | 187646/400000 [00:25<00:28, 7521.51it/s] 47%|████▋     | 188399/400000 [00:25<00:28, 7468.25it/s] 47%|████▋     | 189172/400000 [00:25<00:27, 7544.73it/s] 47%|████▋     | 189928/400000 [00:25<00:28, 7495.83it/s] 48%|████▊     | 190679/400000 [00:26<00:27, 7497.85it/s] 48%|████▊     | 191430/400000 [00:26<00:27, 7498.17it/s] 48%|████▊     | 192181/400000 [00:26<00:27, 7442.50it/s] 48%|████▊     | 192926/400000 [00:26<00:28, 7360.97it/s] 48%|████▊     | 193663/400000 [00:26<00:28, 7334.26it/s] 49%|████▊     | 194397/400000 [00:26<00:28, 7265.46it/s] 49%|████▉     | 195124/400000 [00:26<00:28, 7262.51it/s] 49%|████▉     | 195874/400000 [00:26<00:27, 7327.91it/s] 49%|████▉     | 196617/400000 [00:26<00:27, 7358.13it/s] 49%|████▉     | 197371/400000 [00:26<00:27, 7410.95it/s] 50%|████▉     | 198113/400000 [00:27<00:27, 7403.83it/s] 50%|████▉     | 198854/400000 [00:27<00:28, 6975.94it/s] 50%|████▉     | 199557/400000 [00:27<00:29, 6840.64it/s] 50%|█████     | 200291/400000 [00:27<00:28, 6981.26it/s] 50%|█████     | 201076/400000 [00:27<00:27, 7220.13it/s] 50%|█████     | 201833/400000 [00:27<00:27, 7318.45it/s] 51%|█████     | 202579/400000 [00:27<00:26, 7356.56it/s] 51%|█████     | 203318/400000 [00:27<00:26, 7349.03it/s] 51%|█████     | 204074/400000 [00:27<00:26, 7410.28it/s] 51%|█████     | 204817/400000 [00:27<00:26, 7363.29it/s] 51%|█████▏    | 205573/400000 [00:28<00:26, 7419.10it/s] 52%|█████▏    | 206321/400000 [00:28<00:26, 7436.60it/s] 52%|█████▏    | 207066/400000 [00:28<00:26, 7378.34it/s] 52%|█████▏    | 207818/400000 [00:28<00:25, 7417.44it/s] 52%|█████▏    | 208561/400000 [00:28<00:25, 7405.06it/s] 52%|█████▏    | 209302/400000 [00:28<00:25, 7385.30it/s] 53%|█████▎    | 210041/400000 [00:28<00:25, 7348.87it/s] 53%|█████▎    | 210784/400000 [00:28<00:25, 7372.41it/s] 53%|█████▎    | 211537/400000 [00:28<00:25, 7416.84it/s] 53%|█████▎    | 212307/400000 [00:29<00:25, 7498.04it/s] 53%|█████▎    | 213061/400000 [00:29<00:24, 7508.14it/s] 53%|█████▎    | 213813/400000 [00:29<00:25, 7440.41it/s] 54%|█████▎    | 214558/400000 [00:29<00:25, 7386.88it/s] 54%|█████▍    | 215297/400000 [00:29<00:25, 7299.41it/s] 54%|█████▍    | 216038/400000 [00:29<00:25, 7332.02it/s] 54%|█████▍    | 216821/400000 [00:29<00:24, 7474.58it/s] 54%|█████▍    | 217570/400000 [00:29<00:24, 7438.04it/s] 55%|█████▍    | 218319/400000 [00:29<00:24, 7451.33it/s] 55%|█████▍    | 219083/400000 [00:29<00:24, 7506.71it/s] 55%|█████▍    | 219835/400000 [00:30<00:24, 7482.26it/s] 55%|█████▌    | 220595/400000 [00:30<00:23, 7515.05it/s] 55%|█████▌    | 221353/400000 [00:30<00:23, 7531.89it/s] 56%|█████▌    | 222107/400000 [00:30<00:23, 7462.84it/s] 56%|█████▌    | 222858/400000 [00:30<00:23, 7475.88it/s] 56%|█████▌    | 223646/400000 [00:30<00:23, 7590.85it/s] 56%|█████▌    | 224406/400000 [00:30<00:23, 7496.10it/s] 56%|█████▋    | 225157/400000 [00:30<00:23, 7442.98it/s] 56%|█████▋    | 225905/400000 [00:30<00:23, 7451.61it/s] 57%|█████▋    | 226676/400000 [00:30<00:23, 7525.42it/s] 57%|█████▋    | 227442/400000 [00:31<00:22, 7563.13it/s] 57%|█████▋    | 228206/400000 [00:31<00:22, 7583.03it/s] 57%|█████▋    | 228965/400000 [00:31<00:22, 7552.32it/s] 57%|█████▋    | 229721/400000 [00:31<00:22, 7462.79it/s] 58%|█████▊    | 230468/400000 [00:31<00:23, 7295.77it/s] 58%|█████▊    | 231270/400000 [00:31<00:22, 7497.04it/s] 58%|█████▊    | 232037/400000 [00:31<00:22, 7547.79it/s] 58%|█████▊    | 232803/400000 [00:31<00:22, 7578.82it/s] 58%|█████▊    | 233563/400000 [00:31<00:22, 7465.10it/s] 59%|█████▊    | 234319/400000 [00:31<00:22, 7491.87it/s] 59%|█████▉    | 235070/400000 [00:32<00:22, 7436.53it/s] 59%|█████▉    | 235824/400000 [00:32<00:21, 7464.64it/s] 59%|█████▉    | 236571/400000 [00:32<00:21, 7433.79it/s] 59%|█████▉    | 237315/400000 [00:32<00:22, 7264.06it/s] 60%|█████▉    | 238046/400000 [00:32<00:22, 7277.42it/s] 60%|█████▉    | 238789/400000 [00:32<00:22, 7320.50it/s] 60%|█████▉    | 239532/400000 [00:32<00:21, 7353.01it/s] 60%|██████    | 240270/400000 [00:32<00:21, 7358.31it/s] 60%|██████    | 241012/400000 [00:32<00:21, 7376.47it/s] 60%|██████    | 241791/400000 [00:32<00:21, 7495.16it/s] 61%|██████    | 242553/400000 [00:33<00:20, 7531.51it/s] 61%|██████    | 243307/400000 [00:33<00:21, 7445.78it/s] 61%|██████    | 244061/400000 [00:33<00:20, 7471.99it/s] 61%|██████    | 244809/400000 [00:33<00:21, 7377.08it/s] 61%|██████▏   | 245559/400000 [00:33<00:20, 7412.04it/s] 62%|██████▏   | 246340/400000 [00:33<00:20, 7526.61it/s] 62%|██████▏   | 247097/400000 [00:33<00:20, 7537.06it/s] 62%|██████▏   | 247853/400000 [00:33<00:20, 7541.19it/s] 62%|██████▏   | 248608/400000 [00:33<00:20, 7490.66it/s] 62%|██████▏   | 249358/400000 [00:33<00:20, 7485.53it/s] 63%|██████▎   | 250111/400000 [00:34<00:19, 7497.65it/s] 63%|██████▎   | 250879/400000 [00:34<00:19, 7548.81it/s] 63%|██████▎   | 251637/400000 [00:34<00:19, 7555.21it/s] 63%|██████▎   | 252393/400000 [00:34<00:19, 7413.42it/s] 63%|██████▎   | 253136/400000 [00:34<00:19, 7394.78it/s] 63%|██████▎   | 253879/400000 [00:34<00:19, 7404.59it/s] 64%|██████▎   | 254630/400000 [00:34<00:19, 7435.01it/s] 64%|██████▍   | 255374/400000 [00:34<00:19, 7382.44it/s] 64%|██████▍   | 256113/400000 [00:34<00:19, 7347.09it/s] 64%|██████▍   | 256849/400000 [00:34<00:19, 7348.75it/s] 64%|██████▍   | 257588/400000 [00:35<00:19, 7359.41it/s] 65%|██████▍   | 258333/400000 [00:35<00:19, 7385.64it/s] 65%|██████▍   | 259094/400000 [00:35<00:18, 7447.26it/s] 65%|██████▍   | 259839/400000 [00:35<00:18, 7396.38it/s] 65%|██████▌   | 260588/400000 [00:35<00:18, 7424.17it/s] 65%|██████▌   | 261344/400000 [00:35<00:18, 7463.29it/s] 66%|██████▌   | 262120/400000 [00:35<00:18, 7549.64it/s] 66%|██████▌   | 262876/400000 [00:35<00:18, 7543.03it/s] 66%|██████▌   | 263631/400000 [00:35<00:18, 7319.99it/s] 66%|██████▌   | 264379/400000 [00:35<00:18, 7366.10it/s] 66%|██████▋   | 265129/400000 [00:36<00:18, 7404.13it/s] 66%|██████▋   | 265871/400000 [00:36<00:18, 7342.80it/s] 67%|██████▋   | 266607/400000 [00:36<00:18, 7174.74it/s] 67%|██████▋   | 267329/400000 [00:36<00:18, 7186.63it/s] 67%|██████▋   | 268075/400000 [00:36<00:18, 7266.03it/s] 67%|██████▋   | 268856/400000 [00:36<00:17, 7420.86it/s] 67%|██████▋   | 269606/400000 [00:36<00:17, 7442.61it/s] 68%|██████▊   | 270355/400000 [00:36<00:17, 7456.16it/s] 68%|██████▊   | 271102/400000 [00:36<00:17, 7188.11it/s] 68%|██████▊   | 271824/400000 [00:37<00:17, 7146.11it/s] 68%|██████▊   | 272562/400000 [00:37<00:17, 7212.54it/s] 68%|██████▊   | 273293/400000 [00:37<00:17, 7239.06it/s] 69%|██████▊   | 274024/400000 [00:37<00:17, 7258.69it/s] 69%|██████▊   | 274751/400000 [00:37<00:17, 7194.03it/s] 69%|██████▉   | 275472/400000 [00:37<00:17, 7194.18it/s] 69%|██████▉   | 276216/400000 [00:37<00:17, 7264.66it/s] 69%|██████▉   | 276946/400000 [00:37<00:16, 7272.93it/s] 69%|██████▉   | 277711/400000 [00:37<00:16, 7380.95it/s] 70%|██████▉   | 278450/400000 [00:37<00:16, 7345.90it/s] 70%|██████▉   | 279208/400000 [00:38<00:16, 7412.05it/s] 70%|██████▉   | 279959/400000 [00:38<00:16, 7439.66it/s] 70%|███████   | 280704/400000 [00:38<00:16, 7367.98it/s] 70%|███████   | 281442/400000 [00:38<00:16, 7363.51it/s] 71%|███████   | 282179/400000 [00:38<00:16, 7323.35it/s] 71%|███████   | 282912/400000 [00:38<00:15, 7324.46it/s] 71%|███████   | 283664/400000 [00:38<00:15, 7378.88it/s] 71%|███████   | 284426/400000 [00:38<00:15, 7449.51it/s] 71%|███████▏  | 285175/400000 [00:38<00:15, 7458.01it/s] 71%|███████▏  | 285922/400000 [00:38<00:15, 7423.22it/s] 72%|███████▏  | 286674/400000 [00:39<00:15, 7448.80it/s] 72%|███████▏  | 287428/400000 [00:39<00:15, 7473.62it/s] 72%|███████▏  | 288176/400000 [00:39<00:15, 7341.19it/s] 72%|███████▏  | 288911/400000 [00:39<00:15, 7274.69it/s] 72%|███████▏  | 289640/400000 [00:39<00:15, 7272.52it/s] 73%|███████▎  | 290397/400000 [00:39<00:14, 7358.14it/s] 73%|███████▎  | 291134/400000 [00:39<00:14, 7274.40it/s] 73%|███████▎  | 291868/400000 [00:39<00:14, 7292.63it/s] 73%|███████▎  | 292607/400000 [00:39<00:14, 7321.07it/s] 73%|███████▎  | 293340/400000 [00:39<00:14, 7220.37it/s] 74%|███████▎  | 294091/400000 [00:40<00:14, 7302.64it/s] 74%|███████▎  | 294822/400000 [00:40<00:14, 7274.19it/s] 74%|███████▍  | 295551/400000 [00:40<00:14, 7277.33it/s] 74%|███████▍  | 296283/400000 [00:40<00:14, 7288.42it/s] 74%|███████▍  | 297013/400000 [00:40<00:14, 7237.37it/s] 74%|███████▍  | 297742/400000 [00:40<00:14, 7251.80it/s] 75%|███████▍  | 298507/400000 [00:40<00:13, 7365.02it/s] 75%|███████▍  | 299245/400000 [00:40<00:13, 7366.47it/s] 75%|███████▍  | 299995/400000 [00:40<00:13, 7403.70it/s] 75%|███████▌  | 300736/400000 [00:40<00:13, 7350.24it/s] 75%|███████▌  | 301493/400000 [00:41<00:13, 7414.40it/s] 76%|███████▌  | 302235/400000 [00:41<00:13, 7370.47it/s] 76%|███████▌  | 302979/400000 [00:41<00:13, 7389.88it/s] 76%|███████▌  | 303719/400000 [00:41<00:13, 7346.71it/s] 76%|███████▌  | 304454/400000 [00:41<00:13, 7157.08it/s] 76%|███████▋  | 305214/400000 [00:41<00:13, 7283.88it/s] 76%|███████▋  | 305975/400000 [00:41<00:12, 7375.88it/s] 77%|███████▋  | 306734/400000 [00:41<00:12, 7438.53it/s] 77%|███████▋  | 307479/400000 [00:41<00:12, 7419.32it/s] 77%|███████▋  | 308222/400000 [00:41<00:12, 7139.55it/s] 77%|███████▋  | 308957/400000 [00:42<00:12, 7201.37it/s] 77%|███████▋  | 309708/400000 [00:42<00:12, 7289.88it/s] 78%|███████▊  | 310454/400000 [00:42<00:12, 7338.87it/s] 78%|███████▊  | 311209/400000 [00:42<00:12, 7397.45it/s] 78%|███████▊  | 311950/400000 [00:42<00:11, 7364.42it/s] 78%|███████▊  | 312688/400000 [00:42<00:12, 7151.84it/s] 78%|███████▊  | 313429/400000 [00:42<00:11, 7225.85it/s] 79%|███████▊  | 314195/400000 [00:42<00:11, 7349.07it/s] 79%|███████▊  | 314959/400000 [00:42<00:11, 7432.94it/s] 79%|███████▉  | 315704/400000 [00:43<00:11, 7352.99it/s] 79%|███████▉  | 316473/400000 [00:43<00:11, 7447.18it/s] 79%|███████▉  | 317219/400000 [00:43<00:11, 7299.48it/s] 79%|███████▉  | 317954/400000 [00:43<00:11, 7314.01it/s] 80%|███████▉  | 318697/400000 [00:43<00:11, 7345.88it/s] 80%|███████▉  | 319433/400000 [00:43<00:11, 7299.13it/s] 80%|████████  | 320179/400000 [00:43<00:10, 7344.03it/s] 80%|████████  | 320945/400000 [00:43<00:10, 7435.39it/s] 80%|████████  | 321722/400000 [00:43<00:10, 7531.30it/s] 81%|████████  | 322498/400000 [00:43<00:10, 7594.88it/s] 81%|████████  | 323259/400000 [00:44<00:10, 7510.44it/s] 81%|████████  | 324033/400000 [00:44<00:10, 7577.42it/s] 81%|████████  | 324799/400000 [00:44<00:09, 7600.00it/s] 81%|████████▏ | 325560/400000 [00:44<00:09, 7564.18it/s] 82%|████████▏ | 326317/400000 [00:44<00:09, 7530.57it/s] 82%|████████▏ | 327071/400000 [00:44<00:09, 7458.69it/s] 82%|████████▏ | 327818/400000 [00:44<00:09, 7458.17it/s] 82%|████████▏ | 328582/400000 [00:44<00:09, 7511.16it/s] 82%|████████▏ | 329346/400000 [00:44<00:09, 7545.82it/s] 83%|████████▎ | 330115/400000 [00:44<00:09, 7586.85it/s] 83%|████████▎ | 330874/400000 [00:45<00:09, 7417.84it/s] 83%|████████▎ | 331636/400000 [00:45<00:09, 7475.82it/s] 83%|████████▎ | 332394/400000 [00:45<00:09, 7506.76it/s] 83%|████████▎ | 333148/400000 [00:45<00:08, 7514.90it/s] 83%|████████▎ | 333921/400000 [00:45<00:08, 7575.65it/s] 84%|████████▎ | 334679/400000 [00:45<00:08, 7485.33it/s] 84%|████████▍ | 335429/400000 [00:45<00:08, 7346.56it/s] 84%|████████▍ | 336165/400000 [00:45<00:08, 7349.85it/s] 84%|████████▍ | 336929/400000 [00:45<00:08, 7432.86it/s] 84%|████████▍ | 337717/400000 [00:45<00:08, 7561.02it/s] 85%|████████▍ | 338475/400000 [00:46<00:08, 7454.46it/s] 85%|████████▍ | 339242/400000 [00:46<00:08, 7515.84it/s] 85%|████████▍ | 339995/400000 [00:46<00:08, 7416.66it/s] 85%|████████▌ | 340738/400000 [00:46<00:07, 7420.42it/s] 85%|████████▌ | 341509/400000 [00:46<00:07, 7502.87it/s] 86%|████████▌ | 342260/400000 [00:46<00:07, 7369.93it/s] 86%|████████▌ | 343018/400000 [00:46<00:07, 7429.82it/s] 86%|████████▌ | 343768/400000 [00:46<00:07, 7450.51it/s] 86%|████████▌ | 344514/400000 [00:46<00:07, 7438.75it/s] 86%|████████▋ | 345274/400000 [00:46<00:07, 7484.54it/s] 87%|████████▋ | 346023/400000 [00:47<00:07, 7406.59it/s] 87%|████████▋ | 346801/400000 [00:47<00:07, 7512.50it/s] 87%|████████▋ | 347563/400000 [00:47<00:06, 7541.28it/s] 87%|████████▋ | 348318/400000 [00:47<00:06, 7522.40it/s] 87%|████████▋ | 349074/400000 [00:47<00:06, 7533.25it/s] 87%|████████▋ | 349828/400000 [00:47<00:06, 7438.45it/s] 88%|████████▊ | 350581/400000 [00:47<00:06, 7465.53it/s] 88%|████████▊ | 351328/400000 [00:47<00:06, 7312.04it/s] 88%|████████▊ | 352061/400000 [00:47<00:06, 7290.50it/s] 88%|████████▊ | 352847/400000 [00:47<00:06, 7450.34it/s] 88%|████████▊ | 353594/400000 [00:48<00:06, 7317.91it/s] 89%|████████▊ | 354355/400000 [00:48<00:06, 7402.59it/s] 89%|████████▉ | 355104/400000 [00:48<00:06, 7426.34it/s] 89%|████████▉ | 355848/400000 [00:48<00:05, 7391.06it/s] 89%|████████▉ | 356596/400000 [00:48<00:05, 7417.19it/s] 89%|████████▉ | 357339/400000 [00:48<00:05, 7333.35it/s] 90%|████████▉ | 358091/400000 [00:48<00:05, 7387.44it/s] 90%|████████▉ | 358841/400000 [00:48<00:05, 7418.87it/s] 90%|████████▉ | 359605/400000 [00:48<00:05, 7482.12it/s] 90%|█████████ | 360398/400000 [00:48<00:05, 7609.24it/s] 90%|█████████ | 361160/400000 [00:49<00:05, 7530.40it/s] 90%|█████████ | 361948/400000 [00:49<00:04, 7631.21it/s] 91%|█████████ | 362712/400000 [00:49<00:04, 7619.16it/s] 91%|█████████ | 363475/400000 [00:49<00:04, 7555.07it/s] 91%|█████████ | 364233/400000 [00:49<00:04, 7561.23it/s] 91%|█████████ | 364990/400000 [00:49<00:04, 7473.69it/s] 91%|█████████▏| 365738/400000 [00:49<00:04, 7165.03it/s] 92%|█████████▏| 366458/400000 [00:49<00:04, 7111.51it/s] 92%|█████████▏| 367232/400000 [00:49<00:04, 7286.92it/s] 92%|█████████▏| 367997/400000 [00:50<00:04, 7391.31it/s] 92%|█████████▏| 368739/400000 [00:50<00:04, 7349.00it/s] 92%|█████████▏| 369507/400000 [00:50<00:04, 7445.12it/s] 93%|█████████▎| 370279/400000 [00:50<00:03, 7524.85it/s] 93%|█████████▎| 371033/400000 [00:50<00:03, 7497.54it/s] 93%|█████████▎| 371791/400000 [00:50<00:03, 7521.09it/s] 93%|█████████▎| 372544/400000 [00:50<00:03, 7257.13it/s] 93%|█████████▎| 373296/400000 [00:50<00:03, 7333.75it/s] 94%|█████████▎| 374063/400000 [00:50<00:03, 7431.11it/s] 94%|█████████▎| 374808/400000 [00:50<00:03, 7395.90it/s] 94%|█████████▍| 375549/400000 [00:51<00:03, 7361.46it/s] 94%|█████████▍| 376287/400000 [00:51<00:03, 7310.73it/s] 94%|█████████▍| 377050/400000 [00:51<00:03, 7402.60it/s] 94%|█████████▍| 377792/400000 [00:51<00:02, 7402.99it/s] 95%|█████████▍| 378550/400000 [00:51<00:02, 7454.80it/s] 95%|█████████▍| 379303/400000 [00:51<00:02, 7475.10it/s] 95%|█████████▌| 380051/400000 [00:51<00:02, 7367.56it/s] 95%|█████████▌| 380789/400000 [00:51<00:02, 7354.45it/s] 95%|█████████▌| 381525/400000 [00:51<00:02, 7342.72it/s] 96%|█████████▌| 382286/400000 [00:51<00:02, 7420.05it/s] 96%|█████████▌| 383046/400000 [00:52<00:02, 7470.93it/s] 96%|█████████▌| 383794/400000 [00:52<00:02, 7369.96it/s] 96%|█████████▌| 384544/400000 [00:52<00:02, 7406.70it/s] 96%|█████████▋| 385292/400000 [00:52<00:01, 7427.76it/s] 97%|█████████▋| 386045/400000 [00:52<00:01, 7457.86it/s] 97%|█████████▋| 386806/400000 [00:52<00:01, 7501.93it/s] 97%|█████████▋| 387557/400000 [00:52<00:01, 7463.80it/s] 97%|█████████▋| 388304/400000 [00:52<00:01, 7246.34it/s] 97%|█████████▋| 389067/400000 [00:52<00:01, 7356.48it/s] 97%|█████████▋| 389826/400000 [00:52<00:01, 7423.17it/s] 98%|█████████▊| 390586/400000 [00:53<00:01, 7472.20it/s] 98%|█████████▊| 391335/400000 [00:53<00:01, 7149.31it/s] 98%|█████████▊| 392092/400000 [00:53<00:01, 7269.27it/s] 98%|█████████▊| 392834/400000 [00:53<00:00, 7311.83it/s] 98%|█████████▊| 393578/400000 [00:53<00:00, 7349.10it/s] 99%|█████████▊| 394322/400000 [00:53<00:00, 7374.47it/s] 99%|█████████▉| 395061/400000 [00:53<00:00, 7363.05it/s] 99%|█████████▉| 395842/400000 [00:53<00:00, 7490.07it/s] 99%|█████████▉| 396610/400000 [00:53<00:00, 7544.81it/s] 99%|█████████▉| 397366/400000 [00:53<00:00, 7544.42it/s]100%|█████████▉| 398122/400000 [00:54<00:00, 7506.84it/s]100%|█████████▉| 398874/400000 [00:54<00:00, 7442.77it/s]100%|█████████▉| 399626/400000 [00:54<00:00, 7465.31it/s]100%|█████████▉| 399999/400000 [00:54<00:00, 7360.50it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f8663019d30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.01126330570772141 	 Accuracy: 53
Train Epoch: 1 	 Loss: 0.011332359401677365 	 Accuracy: 51

  model saves at 51% accuracy 

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
2020-05-13 12:27:21.700620: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 12:27:21.705382: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-13 12:27:21.705545: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5637dbd41e50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 12:27:21.705561: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f866eb96fd0> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.3906 - accuracy: 0.5180
 2000/25000 [=>............................] - ETA: 10s - loss: 7.5900 - accuracy: 0.5050
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6513 - accuracy: 0.5010 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6015 - accuracy: 0.5042
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5838 - accuracy: 0.5054
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5976 - accuracy: 0.5045
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6710 - accuracy: 0.4997
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6628 - accuracy: 0.5002
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6615 - accuracy: 0.5003
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7080 - accuracy: 0.4973
11000/25000 [============>.................] - ETA: 4s - loss: 7.7294 - accuracy: 0.4959
12000/25000 [=============>................] - ETA: 4s - loss: 7.7292 - accuracy: 0.4959
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7480 - accuracy: 0.4947
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7291 - accuracy: 0.4959
15000/25000 [=================>............] - ETA: 3s - loss: 7.6912 - accuracy: 0.4984
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6733 - accuracy: 0.4996
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6630 - accuracy: 0.5002
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6862 - accuracy: 0.4987
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6602 - accuracy: 0.5004
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6567 - accuracy: 0.5006
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6485 - accuracy: 0.5012
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6520 - accuracy: 0.5010
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 10s 399us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f85c7f51710> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f85be1bc128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.3102 - crf_viterbi_accuracy: 0.6800 - val_loss: 1.2754 - val_crf_viterbi_accuracy: 0.6533

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


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
>>>model:  <mlmodels.model_gluon.fb_prophet.Model object at 0x7f08e07dff60> <class 'mlmodels.model_gluon.fb_prophet.Model'>

  #### Inference Need return ypred, ytrue ######################### 

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 13:12:24.685399
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.3339
metric_name                                  mean_absolute_error
Name: 0, dtype: object 

  date_run                              2020-05-13 13:12:24.690871
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   215.367
metric_name                                   mean_squared_error
Name: 1, dtype: object 

  date_run                              2020-05-13 13:12:24.695110
model_uri                              model_gluon/fb_prophet.py
json           [{'model_uri': 'model_gluon/fb_prophet.py'}, {...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   14.4309
metric_name                                median_absolute_error
Name: 2, dtype: object 

  date_run                              2020-05-13 13:12:24.699659
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
>>>model:  <mlmodels.model_keras.armdn.Model object at 0x7f08ec5a9470> <class 'mlmodels.model_keras.armdn.Model'>

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

1/1 [==============================] - 2s 2s/step - loss: 347139.5938
Epoch 2/10

1/1 [==============================] - 0s 115ms/step - loss: 178506.8438
Epoch 3/10

1/1 [==============================] - 0s 121ms/step - loss: 79254.8203
Epoch 4/10

1/1 [==============================] - 0s 102ms/step - loss: 37411.4297
Epoch 5/10

1/1 [==============================] - 0s 104ms/step - loss: 19975.7129
Epoch 6/10

1/1 [==============================] - 0s 104ms/step - loss: 12043.4912
Epoch 7/10

1/1 [==============================] - 0s 100ms/step - loss: 8043.7734
Epoch 8/10

1/1 [==============================] - 0s 104ms/step - loss: 5823.1118
Epoch 9/10

1/1 [==============================] - 0s 101ms/step - loss: 4491.2295
Epoch 10/10

1/1 [==============================] - 0s 105ms/step - loss: 3652.1938

  #### Inference Need return ypred, ytrue ######################### 
[[-7.04439223e-01  1.08347244e+01  1.01681290e+01  1.25518322e+01
   1.29815578e+01  1.00631599e+01  1.52884321e+01  1.01187840e+01
   1.22900858e+01  1.29938478e+01  1.34749136e+01  1.18573980e+01
   1.18043938e+01  1.26224422e+01  1.16580954e+01  9.49803066e+00
   1.22211819e+01  1.41337252e+01  1.34789629e+01  1.38814459e+01
   1.25171242e+01  1.26717644e+01  9.35169792e+00  1.19103022e+01
   9.41865349e+00  1.09425335e+01  1.25578022e+01  1.15682964e+01
   1.24891272e+01  1.09593592e+01  1.21512957e+01  1.12075787e+01
   1.29696941e+01  1.15180082e+01  1.06682119e+01  1.25636206e+01
   1.05413036e+01  1.34857845e+01  1.38509474e+01  1.39925261e+01
   8.24639797e+00  9.99988174e+00  1.26371479e+01  1.23162355e+01
   1.11693983e+01  1.24628572e+01  9.61915207e+00  1.15260582e+01
   8.53029156e+00  1.14673872e+01  1.07305145e+01  1.16148777e+01
   1.19149122e+01  1.09150133e+01  1.41345320e+01  1.43281460e+01
   1.19352951e+01  1.10856333e+01  1.19773445e+01  9.64939213e+00
  -1.31130207e+00 -2.01292324e+00  7.55736470e-01 -9.63567793e-01
   5.63676655e-01  1.48811257e+00 -1.73930633e+00  1.53774023e-02
   5.27779520e-01 -1.39403987e+00 -1.36367238e+00 -8.06637764e-01
  -1.41017631e-01 -1.82685304e+00  9.29221570e-01  1.51476532e-01
  -8.41161132e-01 -1.87482047e+00 -3.49537432e-01 -2.75754404e+00
  -2.12924302e-01  2.99617052e-01 -1.13654661e+00 -5.95854521e-01
  -1.72431135e+00 -2.41835070e+00 -1.83373117e+00  4.27719265e-01
   8.23479116e-01 -6.55782342e-01 -5.68160057e-01  2.23800611e+00
  -4.09820616e-01  8.20852757e-01 -4.32958186e-01  2.45472765e+00
   7.31654286e-01  1.13489354e+00  4.24902767e-01  1.47101378e+00
  -3.91505182e-01 -9.84608531e-01 -9.37344134e-01 -5.68739414e-01
  -1.77456021e-01 -1.16753531e+00 -1.10488605e+00 -1.07911003e+00
  -6.95148468e-01 -1.50341856e+00  2.09352279e+00  1.22273111e+00
  -1.16356611e-02  1.06967914e+00  1.10740542e+00  7.71300793e-02
   9.63795483e-02  1.93325138e+00  7.21617162e-01  1.72236693e+00
   1.17562020e+00 -9.95633900e-01  1.95035410e+00 -2.91583180e+00
   1.71356869e+00 -2.16710567e+00 -9.64979649e-01  2.96949792e+00
   1.68516979e-01 -1.16902530e+00 -7.97833204e-01  1.96598041e+00
   1.88152820e-01 -8.81439447e-02  1.19315755e+00  1.89677596e+00
  -2.04172254e+00  9.03977334e-01 -1.94789755e+00  8.74969661e-01
  -9.00255740e-01 -4.83315140e-01 -2.17939925e+00 -5.02744436e-01
   5.16762257e-01 -3.64916056e-01 -6.82512164e-01  4.27013695e-01
  -7.86397099e-01  1.68412685e+00  5.72663099e-02 -4.05085415e-01
   2.79514551e-01 -7.02881336e-01 -1.50088382e+00 -9.29831505e-01
  -1.84054911e+00  1.13395095e+00  2.37148428e+00 -1.50850189e+00
   2.15629864e+00  7.44811714e-01 -1.37079224e-01  6.84927106e-01
   9.51539278e-02  1.15263760e+00  1.06323838e+00 -1.20212114e+00
  -2.49091625e+00  8.10143709e-01  9.84313607e-01  2.32361555e-02
   6.98381722e-01  2.33981609e+00 -2.07719207e-01  1.45167184e+00
  -4.94766831e-02  7.41922796e-01  7.13822305e-01 -2.24772787e+00
   1.08373451e+00  1.09177294e+01  1.39462862e+01  1.19612074e+01
   1.29699030e+01  1.09004908e+01  1.29464703e+01  1.09600706e+01
   1.47810144e+01  9.66049290e+00  1.16069994e+01  1.18194695e+01
   1.09801683e+01  1.15497932e+01  1.09003201e+01  8.90622139e+00
   1.27112570e+01  1.31098919e+01  1.33905277e+01  1.33250818e+01
   1.15176792e+01  1.24638309e+01  1.13236809e+01  1.12564201e+01
   9.52345943e+00  1.11095657e+01  1.41346378e+01  1.10212889e+01
   1.11789370e+01  1.34142647e+01  1.23614807e+01  1.18897619e+01
   1.12936993e+01  1.17763109e+01  1.08836870e+01  1.16002569e+01
   1.19516220e+01  1.24962425e+01  1.37811804e+01  1.26986313e+01
   9.46070004e+00  1.15635166e+01  1.18370695e+01  1.22402449e+01
   1.25087948e+01  8.88081741e+00  1.09170713e+01  1.20622549e+01
   1.05907621e+01  1.14953356e+01  8.81236362e+00  1.30501413e+01
   9.08056068e+00  1.28760138e+01  9.69389629e+00  1.17144308e+01
   8.86544609e+00  1.11605902e+01  9.88684082e+00  1.15186749e+01
   8.03943098e-01  5.08428514e-01  2.80310583e+00  5.67750931e-02
   1.43909729e+00  3.05893469e+00  1.29938650e+00  2.87986565e+00
   4.44479704e-01  1.94570899e-01  2.85538077e-01  3.85464478e+00
   1.38439882e+00  3.24643016e-01  1.15957212e+00  2.42714119e+00
   4.37391186e+00  3.58132076e+00  1.70142293e+00  8.10880661e-02
   3.12030554e+00  4.56976891e-02  9.79424596e-01  3.23641658e-01
   5.12100816e-01  1.73648584e+00  2.28330708e+00  2.77568722e+00
   8.78177643e-01  4.68703508e+00  3.59139681e-01  1.93201280e+00
   2.15166283e+00  2.02326012e+00  4.15093005e-01  1.11789250e+00
   6.13316298e-01  5.23750067e-01  9.31526840e-01  2.02325821e+00
   9.59046662e-01  1.90660655e-01  7.58690715e-01  2.98102093e+00
   1.96706057e-01  2.57816505e+00  2.93967056e+00  3.83760870e-01
   8.86690259e-01  3.21731615e+00  3.06450367e+00  1.78554654e-01
   1.82087350e+00  2.87912512e+00  1.12261307e+00  1.14149427e+00
   1.40163708e+00  2.05485106e+00  2.28480875e-01  8.20759177e-01
   1.84473133e+00  1.70994878e+00  1.33624458e+00  1.61696982e+00
   6.94519162e-01  1.44518185e+00  2.18581343e+00  3.02958608e-01
   2.99129987e+00  1.72885776e-01  8.06395411e-02  7.43673623e-01
   1.73785293e+00  1.14649057e+00  2.33146858e+00  5.14085770e-01
   2.33957005e+00  6.45546913e-02  1.95142198e+00  1.10425937e+00
   1.27397776e-01  2.21946263e+00  2.66389942e+00  7.09991217e-01
   6.16843820e-01  5.23097098e-01  1.98033476e+00  2.39451694e+00
   2.54255295e-01  2.85308504e+00  5.18293858e-01  1.03198349e-01
   2.15654206e+00  9.87812221e-01  1.58364582e+00  1.14577723e+00
   4.30314302e-01  1.23310590e+00  3.73266995e-01  3.55972648e-02
   2.92493105e-01  2.43251801e+00  5.66683590e-01  7.28342295e-01
   2.69616747e+00  7.09678113e-01  4.65334058e-02  1.53259599e+00
   1.43367827e-01  3.70853543e-02  2.75158548e+00  2.25630522e-01
   3.18260670e+00  7.49910712e-01  5.81173003e-01  2.56815791e-01
   4.36589837e-01  3.97470534e-01  1.47595155e+00  2.92534542e+00
   9.01181889e+00 -1.91093826e+01 -1.34565487e+01]]

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 13:12:34.328696
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   90.3976
metric_name                                  mean_absolute_error
Name: 4, dtype: object 

  date_run                              2020-05-13 13:12:34.333099
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   8196.93
metric_name                                   mean_squared_error
Name: 5, dtype: object 

  date_run                              2020-05-13 13:12:34.336773
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   90.0515
metric_name                                median_absolute_error
Name: 6, dtype: object 

  date_run                              2020-05-13 13:12:34.340562
model_uri                                   model_keras.armdn.py
json           [{'model_uri': 'model_keras.armdn.py', 'lstm_h...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  -733.093
metric_name                                             r2_score
Name: 7, dtype: object 

  


### Running {'hypermodel_pars': {}, 'data_pars': {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': 'dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': 'dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'}, 'model_pars': {'model_uri': 'model_tch.nbeats.py', 'forecast_length': 60, 'backcast_length': 100, 'stack_types': ['NBeatsNet.GENERIC_BLOCK', 'NBeatsNet.GENERIC_BLOCK'], 'device': 'cpu', 'nb_blocks_per_stack': 3, 'thetas_dims': [7, 8], 'share_weights_in_stack': 0, 'hidden_layer_units': 256}, 'compute_pars': {'batch_size': 100, 'disable_plot': 1, 'norm_constant': 1.0, 'result_path': 'ztest/model_tch/nbeats/n_beats_{}test.png', 'model_path': 'ztest/mycheckpoint'}, 'out_pars': {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'forecast_length': 60, 'backcast_length': 100, 'train_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_train.csv', 'test_data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/stock/qqq_us_test.csv', 'col_Xinput': ['Close'], 'col_ytarget': 'Close'} {'out_path': 'mlmodels/ztest/model_tch/nbeats/', 'model_checkpoint': 'ztest/model_tch/nbeats/model_checkpoint/'} 

  #### Setup Model   ############################################## 
| N-Beats
| --  Stack Nbeatsnet.Generic_Block (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139675743643072
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139673231172104
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=100, forecast_length=60, share_thetas=False) at @139673231172608
| --  Stack Nbeatsnet.Generic_Block (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139673231173112
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139673231173616
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=100, forecast_length=60, share_thetas=False) at @139673231174120

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.nbeats.Model object at 0x7f08cc1bae10> <class 'mlmodels.model_tch.nbeats.Model'>
[[0.40504701]
 [0.40695405]
 [0.39710839]
 ...
 [0.93587014]
 [0.95086039]
 [0.95547277]]
--- fiting ---
grad_step = 000000, loss = 0.544910
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.500833
grad_step = 000002, loss = 0.458759
grad_step = 000003, loss = 0.412114
grad_step = 000004, loss = 0.361747
grad_step = 000005, loss = 0.318916
grad_step = 000006, loss = 0.298247
grad_step = 000007, loss = 0.288399
grad_step = 000008, loss = 0.264807
grad_step = 000009, loss = 0.234911
grad_step = 000010, loss = 0.211663
grad_step = 000011, loss = 0.196757
grad_step = 000012, loss = 0.184480
grad_step = 000013, loss = 0.174006
grad_step = 000014, loss = 0.167714
grad_step = 000015, loss = 0.161061
grad_step = 000016, loss = 0.152485
grad_step = 000017, loss = 0.141473
grad_step = 000018, loss = 0.130897
grad_step = 000019, loss = 0.124019
grad_step = 000020, loss = 0.119021
grad_step = 000021, loss = 0.113266
grad_step = 000022, loss = 0.106217
grad_step = 000023, loss = 0.098677
grad_step = 000024, loss = 0.091628
grad_step = 000025, loss = 0.085637
grad_step = 000026, loss = 0.080713
grad_step = 000027, loss = 0.076641
grad_step = 000028, loss = 0.072395
grad_step = 000029, loss = 0.067238
grad_step = 000030, loss = 0.062030
grad_step = 000031, loss = 0.057802
grad_step = 000032, loss = 0.054457
grad_step = 000033, loss = 0.051327
grad_step = 000034, loss = 0.048029
grad_step = 000035, loss = 0.044618
grad_step = 000036, loss = 0.041321
grad_step = 000037, loss = 0.038352
grad_step = 000038, loss = 0.035801
grad_step = 000039, loss = 0.033568
grad_step = 000040, loss = 0.031391
grad_step = 000041, loss = 0.029092
grad_step = 000042, loss = 0.026818
grad_step = 000043, loss = 0.024843
grad_step = 000044, loss = 0.023210
grad_step = 000045, loss = 0.021715
grad_step = 000046, loss = 0.020181
grad_step = 000047, loss = 0.018623
grad_step = 000048, loss = 0.017170
grad_step = 000049, loss = 0.015908
grad_step = 000050, loss = 0.014815
grad_step = 000051, loss = 0.013783
grad_step = 000052, loss = 0.012718
grad_step = 000053, loss = 0.011672
grad_step = 000054, loss = 0.010768
grad_step = 000055, loss = 0.010025
grad_step = 000056, loss = 0.009348
grad_step = 000057, loss = 0.008663
grad_step = 000058, loss = 0.007991
grad_step = 000059, loss = 0.007387
grad_step = 000060, loss = 0.006880
grad_step = 000061, loss = 0.006440
grad_step = 000062, loss = 0.006008
grad_step = 000063, loss = 0.005571
grad_step = 000064, loss = 0.005178
grad_step = 000065, loss = 0.004863
grad_step = 000066, loss = 0.004595
grad_step = 000067, loss = 0.004332
grad_step = 000068, loss = 0.004075
grad_step = 000069, loss = 0.003847
grad_step = 000070, loss = 0.003661
grad_step = 000071, loss = 0.003504
grad_step = 000072, loss = 0.003349
grad_step = 000073, loss = 0.003195
grad_step = 000074, loss = 0.003065
grad_step = 000075, loss = 0.002963
grad_step = 000076, loss = 0.002871
grad_step = 000077, loss = 0.002780
grad_step = 000078, loss = 0.002695
grad_step = 000079, loss = 0.002626
grad_step = 000080, loss = 0.002571
grad_step = 000081, loss = 0.002521
grad_step = 000082, loss = 0.002469
grad_step = 000083, loss = 0.002423
grad_step = 000084, loss = 0.002389
grad_step = 000085, loss = 0.002360
grad_step = 000086, loss = 0.002331
grad_step = 000087, loss = 0.002302
grad_step = 000088, loss = 0.002279
grad_step = 000089, loss = 0.002262
grad_step = 000090, loss = 0.002246
grad_step = 000091, loss = 0.002229
grad_step = 000092, loss = 0.002214
grad_step = 000093, loss = 0.002203
grad_step = 000094, loss = 0.002194
grad_step = 000095, loss = 0.002184
grad_step = 000096, loss = 0.002173
grad_step = 000097, loss = 0.002165
grad_step = 000098, loss = 0.002158
grad_step = 000099, loss = 0.002152
grad_step = 000100, loss = 0.002144
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002137
grad_step = 000102, loss = 0.002131
grad_step = 000103, loss = 0.002125
grad_step = 000104, loss = 0.002119
grad_step = 000105, loss = 0.002113
grad_step = 000106, loss = 0.002108
grad_step = 000107, loss = 0.002103
grad_step = 000108, loss = 0.002097
grad_step = 000109, loss = 0.002091
grad_step = 000110, loss = 0.002086
grad_step = 000111, loss = 0.002080
grad_step = 000112, loss = 0.002074
grad_step = 000113, loss = 0.002068
grad_step = 000114, loss = 0.002062
grad_step = 000115, loss = 0.002056
grad_step = 000116, loss = 0.002050
grad_step = 000117, loss = 0.002043
grad_step = 000118, loss = 0.002037
grad_step = 000119, loss = 0.002030
grad_step = 000120, loss = 0.002023
grad_step = 000121, loss = 0.002016
grad_step = 000122, loss = 0.002008
grad_step = 000123, loss = 0.002001
grad_step = 000124, loss = 0.001995
grad_step = 000125, loss = 0.001994
grad_step = 000126, loss = 0.002004
grad_step = 000127, loss = 0.002001
grad_step = 000128, loss = 0.001970
grad_step = 000129, loss = 0.001970
grad_step = 000130, loss = 0.001980
grad_step = 000131, loss = 0.001958
grad_step = 000132, loss = 0.001946
grad_step = 000133, loss = 0.001957
grad_step = 000134, loss = 0.001948
grad_step = 000135, loss = 0.001928
grad_step = 000136, loss = 0.001930
grad_step = 000137, loss = 0.001936
grad_step = 000138, loss = 0.001922
grad_step = 000139, loss = 0.001907
grad_step = 000140, loss = 0.001906
grad_step = 000141, loss = 0.001911
grad_step = 000142, loss = 0.001909
grad_step = 000143, loss = 0.001896
grad_step = 000144, loss = 0.001883
grad_step = 000145, loss = 0.001878
grad_step = 000146, loss = 0.001879
grad_step = 000147, loss = 0.001884
grad_step = 000148, loss = 0.001888
grad_step = 000149, loss = 0.001893
grad_step = 000150, loss = 0.001882
grad_step = 000151, loss = 0.001866
grad_step = 000152, loss = 0.001850
grad_step = 000153, loss = 0.001843
grad_step = 000154, loss = 0.001846
grad_step = 000155, loss = 0.001853
grad_step = 000156, loss = 0.001864
grad_step = 000157, loss = 0.001862
grad_step = 000158, loss = 0.001856
grad_step = 000159, loss = 0.001831
grad_step = 000160, loss = 0.001816
grad_step = 000161, loss = 0.001815
grad_step = 000162, loss = 0.001823
grad_step = 000163, loss = 0.001833
grad_step = 000164, loss = 0.001822
grad_step = 000165, loss = 0.001811
grad_step = 000166, loss = 0.001793
grad_step = 000167, loss = 0.001788
grad_step = 000168, loss = 0.001790
grad_step = 000169, loss = 0.001792
grad_step = 000170, loss = 0.001797
grad_step = 000171, loss = 0.001788
grad_step = 000172, loss = 0.001782
grad_step = 000173, loss = 0.001769
grad_step = 000174, loss = 0.001759
grad_step = 000175, loss = 0.001753
grad_step = 000176, loss = 0.001751
grad_step = 000177, loss = 0.001752
grad_step = 000178, loss = 0.001751
grad_step = 000179, loss = 0.001757
grad_step = 000180, loss = 0.001757
grad_step = 000181, loss = 0.001769
grad_step = 000182, loss = 0.001759
grad_step = 000183, loss = 0.001751
grad_step = 000184, loss = 0.001724
grad_step = 000185, loss = 0.001709
grad_step = 000186, loss = 0.001713
grad_step = 000187, loss = 0.001720
grad_step = 000188, loss = 0.001720
grad_step = 000189, loss = 0.001703
grad_step = 000190, loss = 0.001688
grad_step = 000191, loss = 0.001680
grad_step = 000192, loss = 0.001682
grad_step = 000193, loss = 0.001693
grad_step = 000194, loss = 0.001699
grad_step = 000195, loss = 0.001716
grad_step = 000196, loss = 0.001696
grad_step = 000197, loss = 0.001677
grad_step = 000198, loss = 0.001650
grad_step = 000199, loss = 0.001644
grad_step = 000200, loss = 0.001654
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001654
grad_step = 000202, loss = 0.001646
grad_step = 000203, loss = 0.001626
grad_step = 000204, loss = 0.001616
grad_step = 000205, loss = 0.001618
grad_step = 000206, loss = 0.001622
grad_step = 000207, loss = 0.001628
grad_step = 000208, loss = 0.001618
grad_step = 000209, loss = 0.001609
grad_step = 000210, loss = 0.001594
grad_step = 000211, loss = 0.001582
grad_step = 000212, loss = 0.001575
grad_step = 000213, loss = 0.001571
grad_step = 000214, loss = 0.001571
grad_step = 000215, loss = 0.001575
grad_step = 000216, loss = 0.001590
grad_step = 000217, loss = 0.001610
grad_step = 000218, loss = 0.001624
grad_step = 000219, loss = 0.001602
grad_step = 000220, loss = 0.001582
grad_step = 000221, loss = 0.001547
grad_step = 000222, loss = 0.001535
grad_step = 000223, loss = 0.001553
grad_step = 000224, loss = 0.001560
grad_step = 000225, loss = 0.001554
grad_step = 000226, loss = 0.001542
grad_step = 000227, loss = 0.001542
grad_step = 000228, loss = 0.001527
grad_step = 000229, loss = 0.001511
grad_step = 000230, loss = 0.001513
grad_step = 000231, loss = 0.001523
grad_step = 000232, loss = 0.001524
grad_step = 000233, loss = 0.001522
grad_step = 000234, loss = 0.001536
grad_step = 000235, loss = 0.001532
grad_step = 000236, loss = 0.001525
grad_step = 000237, loss = 0.001517
grad_step = 000238, loss = 0.001518
grad_step = 000239, loss = 0.001503
grad_step = 000240, loss = 0.001489
grad_step = 000241, loss = 0.001487
grad_step = 000242, loss = 0.001485
grad_step = 000243, loss = 0.001478
grad_step = 000244, loss = 0.001476
grad_step = 000245, loss = 0.001481
grad_step = 000246, loss = 0.001483
grad_step = 000247, loss = 0.001488
grad_step = 000248, loss = 0.001508
grad_step = 000249, loss = 0.001560
grad_step = 000250, loss = 0.001586
grad_step = 000251, loss = 0.001610
grad_step = 000252, loss = 0.001545
grad_step = 000253, loss = 0.001491
grad_step = 000254, loss = 0.001459
grad_step = 000255, loss = 0.001497
grad_step = 000256, loss = 0.001544
grad_step = 000257, loss = 0.001497
grad_step = 000258, loss = 0.001457
grad_step = 000259, loss = 0.001469
grad_step = 000260, loss = 0.001494
grad_step = 000261, loss = 0.001502
grad_step = 000262, loss = 0.001456
grad_step = 000263, loss = 0.001442
grad_step = 000264, loss = 0.001465
grad_step = 000265, loss = 0.001474
grad_step = 000266, loss = 0.001467
grad_step = 000267, loss = 0.001444
grad_step = 000268, loss = 0.001431
grad_step = 000269, loss = 0.001434
grad_step = 000270, loss = 0.001445
grad_step = 000271, loss = 0.001451
grad_step = 000272, loss = 0.001445
grad_step = 000273, loss = 0.001435
grad_step = 000274, loss = 0.001423
grad_step = 000275, loss = 0.001418
grad_step = 000276, loss = 0.001419
grad_step = 000277, loss = 0.001422
grad_step = 000278, loss = 0.001426
grad_step = 000279, loss = 0.001428
grad_step = 000280, loss = 0.001429
grad_step = 000281, loss = 0.001425
grad_step = 000282, loss = 0.001421
grad_step = 000283, loss = 0.001413
grad_step = 000284, loss = 0.001407
grad_step = 000285, loss = 0.001403
grad_step = 000286, loss = 0.001399
grad_step = 000287, loss = 0.001397
grad_step = 000288, loss = 0.001396
grad_step = 000289, loss = 0.001396
grad_step = 000290, loss = 0.001396
grad_step = 000291, loss = 0.001399
grad_step = 000292, loss = 0.001404
grad_step = 000293, loss = 0.001413
grad_step = 000294, loss = 0.001425
grad_step = 000295, loss = 0.001447
grad_step = 000296, loss = 0.001453
grad_step = 000297, loss = 0.001464
grad_step = 000298, loss = 0.001438
grad_step = 000299, loss = 0.001413
grad_step = 000300, loss = 0.001384
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.001375
grad_step = 000302, loss = 0.001383
grad_step = 000303, loss = 0.001393
grad_step = 000304, loss = 0.001402
grad_step = 000305, loss = 0.001391
grad_step = 000306, loss = 0.001379
grad_step = 000307, loss = 0.001368
grad_step = 000308, loss = 0.001364
grad_step = 000309, loss = 0.001365
grad_step = 000310, loss = 0.001369
grad_step = 000311, loss = 0.001376
grad_step = 000312, loss = 0.001381
grad_step = 000313, loss = 0.001389
grad_step = 000314, loss = 0.001391
grad_step = 000315, loss = 0.001395
grad_step = 000316, loss = 0.001387
grad_step = 000317, loss = 0.001380
grad_step = 000318, loss = 0.001370
grad_step = 000319, loss = 0.001361
grad_step = 000320, loss = 0.001353
grad_step = 000321, loss = 0.001347
grad_step = 000322, loss = 0.001343
grad_step = 000323, loss = 0.001341
grad_step = 000324, loss = 0.001339
grad_step = 000325, loss = 0.001339
grad_step = 000326, loss = 0.001339
grad_step = 000327, loss = 0.001341
grad_step = 000328, loss = 0.001345
grad_step = 000329, loss = 0.001351
grad_step = 000330, loss = 0.001364
grad_step = 000331, loss = 0.001375
grad_step = 000332, loss = 0.001396
grad_step = 000333, loss = 0.001399
grad_step = 000334, loss = 0.001407
grad_step = 000335, loss = 0.001375
grad_step = 000336, loss = 0.001347
grad_step = 000337, loss = 0.001324
grad_step = 000338, loss = 0.001323
grad_step = 000339, loss = 0.001333
grad_step = 000340, loss = 0.001343
grad_step = 000341, loss = 0.001350
grad_step = 000342, loss = 0.001335
grad_step = 000343, loss = 0.001322
grad_step = 000344, loss = 0.001313
grad_step = 000345, loss = 0.001310
grad_step = 000346, loss = 0.001313
grad_step = 000347, loss = 0.001319
grad_step = 000348, loss = 0.001327
grad_step = 000349, loss = 0.001333
grad_step = 000350, loss = 0.001343
grad_step = 000351, loss = 0.001346
grad_step = 000352, loss = 0.001353
grad_step = 000353, loss = 0.001346
grad_step = 000354, loss = 0.001341
grad_step = 000355, loss = 0.001322
grad_step = 000356, loss = 0.001308
grad_step = 000357, loss = 0.001296
grad_step = 000358, loss = 0.001292
grad_step = 000359, loss = 0.001291
grad_step = 000360, loss = 0.001293
grad_step = 000361, loss = 0.001297
grad_step = 000362, loss = 0.001299
grad_step = 000363, loss = 0.001303
grad_step = 000364, loss = 0.001305
grad_step = 000365, loss = 0.001308
grad_step = 000366, loss = 0.001308
grad_step = 000367, loss = 0.001309
grad_step = 000368, loss = 0.001304
grad_step = 000369, loss = 0.001300
grad_step = 000370, loss = 0.001292
grad_step = 000371, loss = 0.001286
grad_step = 000372, loss = 0.001279
grad_step = 000373, loss = 0.001275
grad_step = 000374, loss = 0.001272
grad_step = 000375, loss = 0.001269
grad_step = 000376, loss = 0.001267
grad_step = 000377, loss = 0.001266
grad_step = 000378, loss = 0.001264
grad_step = 000379, loss = 0.001263
grad_step = 000380, loss = 0.001263
grad_step = 000381, loss = 0.001263
grad_step = 000382, loss = 0.001266
grad_step = 000383, loss = 0.001274
grad_step = 000384, loss = 0.001296
grad_step = 000385, loss = 0.001329
grad_step = 000386, loss = 0.001403
grad_step = 000387, loss = 0.001461
grad_step = 000388, loss = 0.001547
grad_step = 000389, loss = 0.001443
grad_step = 000390, loss = 0.001340
grad_step = 000391, loss = 0.001264
grad_step = 000392, loss = 0.001297
grad_step = 000393, loss = 0.001378
grad_step = 000394, loss = 0.001344
grad_step = 000395, loss = 0.001280
grad_step = 000396, loss = 0.001252
grad_step = 000397, loss = 0.001295
grad_step = 000398, loss = 0.001330
grad_step = 000399, loss = 0.001284
grad_step = 000400, loss = 0.001245
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.001258
grad_step = 000402, loss = 0.001279
grad_step = 000403, loss = 0.001280
grad_step = 000404, loss = 0.001264
grad_step = 000405, loss = 0.001255
grad_step = 000406, loss = 0.001246
grad_step = 000407, loss = 0.001241
grad_step = 000408, loss = 0.001249
grad_step = 000409, loss = 0.001261
grad_step = 000410, loss = 0.001258
grad_step = 000411, loss = 0.001239
grad_step = 000412, loss = 0.001228
grad_step = 000413, loss = 0.001233
grad_step = 000414, loss = 0.001237
grad_step = 000415, loss = 0.001234
grad_step = 000416, loss = 0.001231
grad_step = 000417, loss = 0.001232
grad_step = 000418, loss = 0.001229
grad_step = 000419, loss = 0.001221
grad_step = 000420, loss = 0.001217
grad_step = 000421, loss = 0.001219
grad_step = 000422, loss = 0.001223
grad_step = 000423, loss = 0.001222
grad_step = 000424, loss = 0.001221
grad_step = 000425, loss = 0.001222
grad_step = 000426, loss = 0.001225
grad_step = 000427, loss = 0.001224
grad_step = 000428, loss = 0.001222
grad_step = 000429, loss = 0.001220
grad_step = 000430, loss = 0.001221
grad_step = 000431, loss = 0.001221
grad_step = 000432, loss = 0.001221
grad_step = 000433, loss = 0.001219
grad_step = 000434, loss = 0.001221
grad_step = 000435, loss = 0.001223
grad_step = 000436, loss = 0.001228
grad_step = 000437, loss = 0.001228
grad_step = 000438, loss = 0.001232
grad_step = 000439, loss = 0.001232
grad_step = 000440, loss = 0.001238
grad_step = 000441, loss = 0.001237
grad_step = 000442, loss = 0.001239
grad_step = 000443, loss = 0.001232
grad_step = 000444, loss = 0.001229
grad_step = 000445, loss = 0.001219
grad_step = 000446, loss = 0.001211
grad_step = 000447, loss = 0.001201
grad_step = 000448, loss = 0.001192
grad_step = 000449, loss = 0.001187
grad_step = 000450, loss = 0.001185
grad_step = 000451, loss = 0.001185
grad_step = 000452, loss = 0.001188
grad_step = 000453, loss = 0.001193
grad_step = 000454, loss = 0.001200
grad_step = 000455, loss = 0.001215
grad_step = 000456, loss = 0.001231
grad_step = 000457, loss = 0.001264
grad_step = 000458, loss = 0.001295
grad_step = 000459, loss = 0.001349
grad_step = 000460, loss = 0.001353
grad_step = 000461, loss = 0.001352
grad_step = 000462, loss = 0.001279
grad_step = 000463, loss = 0.001214
grad_step = 000464, loss = 0.001175
grad_step = 000465, loss = 0.001187
grad_step = 000466, loss = 0.001228
grad_step = 000467, loss = 0.001243
grad_step = 000468, loss = 0.001235
grad_step = 000469, loss = 0.001197
grad_step = 000470, loss = 0.001171
grad_step = 000471, loss = 0.001170
grad_step = 000472, loss = 0.001187
grad_step = 000473, loss = 0.001206
grad_step = 000474, loss = 0.001209
grad_step = 000475, loss = 0.001203
grad_step = 000476, loss = 0.001182
grad_step = 000477, loss = 0.001166
grad_step = 000478, loss = 0.001158
grad_step = 000479, loss = 0.001161
grad_step = 000480, loss = 0.001169
grad_step = 000481, loss = 0.001176
grad_step = 000482, loss = 0.001179
grad_step = 000483, loss = 0.001174
grad_step = 000484, loss = 0.001166
grad_step = 000485, loss = 0.001159
grad_step = 000486, loss = 0.001155
grad_step = 000487, loss = 0.001154
grad_step = 000488, loss = 0.001154
grad_step = 000489, loss = 0.001155
grad_step = 000490, loss = 0.001154
grad_step = 000491, loss = 0.001153
grad_step = 000492, loss = 0.001153
grad_step = 000493, loss = 0.001154
grad_step = 000494, loss = 0.001155
grad_step = 000495, loss = 0.001155
grad_step = 000496, loss = 0.001154
grad_step = 000497, loss = 0.001152
grad_step = 000498, loss = 0.001148
grad_step = 000499, loss = 0.001144
grad_step = 000500, loss = 0.001142
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.001141
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

  date_run                              2020-05-13 13:12:58.976668
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                  0.228625
metric_name                                  mean_absolute_error
Name: 8, dtype: object 

  date_run                              2020-05-13 13:12:58.982670
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.12526
metric_name                                   mean_squared_error
Name: 9, dtype: object 

  date_run                              2020-05-13 13:12:58.989472
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                   0.13684
metric_name                                median_absolute_error
Name: 10, dtype: object 

  date_run                              2020-05-13 13:12:58.994850
model_uri                                    model_tch.nbeats.py
json           [{'forecast_length': 60, 'backcast_length': 10...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                 -0.903371
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
0   2020-05-13 13:12:24.685399  ...    mean_absolute_error
1   2020-05-13 13:12:24.690871  ...     mean_squared_error
2   2020-05-13 13:12:24.695110  ...  median_absolute_error
3   2020-05-13 13:12:24.699659  ...               r2_score
4   2020-05-13 13:12:34.328696  ...    mean_absolute_error
5   2020-05-13 13:12:34.333099  ...     mean_squared_error
6   2020-05-13 13:12:34.336773  ...  median_absolute_error
7   2020-05-13 13:12:34.340562  ...               r2_score
8   2020-05-13 13:12:58.976668  ...    mean_absolute_error
9   2020-05-13 13:12:58.982670  ...     mean_squared_error
10  2020-05-13 13:12:58.989472  ...  median_absolute_error
11  2020-05-13 13:12:58.994850  ...               r2_score

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4088b219b0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:32, 307721.65it/s]  2%|▏         | 212992/9912422 [00:00<00:24, 397437.35it/s]  9%|▉         | 876544/9912422 [00:00<00:16, 550379.30it/s] 36%|███▌      | 3522560/9912422 [00:00<00:08, 777324.94it/s] 68%|██████▊   | 6717440/9912422 [00:00<00:02, 1096502.78it/s]9920512it [00:00, 10273874.35it/s]                            
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 148047.44it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 309297.34it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 396504.38it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 550132.27it/s]1654784it [00:00, 2761138.03it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 51177.70it/s]            dataset :  <class 'torchvision.datasets.mnist.MNIST'>
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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f403b4cfe10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet34', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet34/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet34/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f403831d0b8> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x0_5', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x0_5/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x0_5/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f403b4cfe10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'shufflenet_v2_x1_0', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/shufflenet_v2_x1_0/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/shufflenet_v2_x1_0/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet101', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet101/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet101/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet101/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f403aa58080> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4038292470> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'wide_resnet101_2', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/wide_resnet101_2/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/wide_resnet101_2/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext101_32x8d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f403827cbe0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext101_32x8d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext101_32x8d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext101_32x8d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnext50_32x4d/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f403b4cfe10> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnext50_32x4d', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnext50_32x4d/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnext50_32x4d/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet50/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f403aa166a0> <class 'mlmodels.model_tch.torchhub.Model'>

  #### If transformer URI is Provided 

  #### Loading dataloader URI 
dataset :  <class 'torchvision.datasets.mnist.MNIST'>

  {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet50', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'train': True}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet50/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet50/'}} default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'> 

  


### Running {'hypermodel_pars': {'learning_rate': {'type': 'log_uniform', 'init': 0.01, 'range': [0.001, 0.1]}}, 'model_pars': {'model_uri': 'model_tch.torchhub.py', 'repo_uri': 'pytorch/vision', 'model': 'resnet152', 'num_classes': 10, 'pretrained': 0, '_comment': '0: False, 1: True', 'num_layers': 5, 'size': 6, 'size_layer': 128, 'output_size': 6, 'timestep': 4, 'epoch': 2}, 'data_pars': {'dataset': 'torchvision.datasets:MNIST', 'data_path': 'dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'}, 'compute_pars': {'distributed': 'mpi', 'max_batch_sample': 10, 'epochs': 5, 'learning_rate': 0.001}, 'out_pars': {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': 'ztest/model_tch/torchhub/resnet152/'}} ############################################ 

  #### Model URI and Config JSON 

  data_pars out_pars {'dataset': 'torchvision.datasets:MNIST', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/', 'train_batch_size': 100, 'test_batch_size': 10, 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'} {'checkpointdir': 'ztest/model_tch/torchhub/resnet152/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/resnet152/'} 

  #### Setup Model   ############################################## 

  #### Fit  ####################################################### 
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4038292470> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.torchhub.Model object at 0x7f4088ad9e80> <class 'mlmodels.model_tch.torchhub.Model'>

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
>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f9481a52208> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=a53255c7289890f2cd5e417e4584a792b1dc749419891ce0ea6dc7707e123d15
  Stored in directory: /tmp/pip-ephem-wheel-cache-x9q9ayue/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f941984e6d8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 45s
   57344/17464789 [..............................] - ETA: 39s
   90112/17464789 [..............................] - ETA: 37s
  163840/17464789 [..............................] - ETA: 27s
  319488/17464789 [..............................] - ETA: 17s
  540672/17464789 [..............................] - ETA: 12s
  999424/17464789 [>.............................] - ETA: 7s 
 1900544/17464789 [==>...........................] - ETA: 4s
 3686400/17464789 [=====>........................] - ETA: 2s
 6504448/17464789 [==========>...................] - ETA: 1s
 9273344/17464789 [==============>...............] - ETA: 0s
12238848/17464789 [====================>.........] - ETA: 0s
15204352/17464789 [=========================>....] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-13 13:14:29.905681: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 13:14:29.909882: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 13:14:29.910020: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56208e3ed3c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 13:14:29.910038: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5900 - accuracy: 0.5050
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8276 - accuracy: 0.4895 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7586 - accuracy: 0.4940
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7310 - accuracy: 0.4958
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7050 - accuracy: 0.4975
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6820 - accuracy: 0.4990
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6781 - accuracy: 0.4992
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7348 - accuracy: 0.4956
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7080 - accuracy: 0.4973
11000/25000 [============>.................] - ETA: 4s - loss: 7.7043 - accuracy: 0.4975
12000/25000 [=============>................] - ETA: 4s - loss: 7.7024 - accuracy: 0.4977
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6831 - accuracy: 0.4989
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7006 - accuracy: 0.4978
15000/25000 [=================>............] - ETA: 3s - loss: 7.7075 - accuracy: 0.4973
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7145 - accuracy: 0.4969
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7189 - accuracy: 0.4966
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7263 - accuracy: 0.4961
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7159 - accuracy: 0.4968
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7157 - accuracy: 0.4968
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7009 - accuracy: 0.4978
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6917 - accuracy: 0.4984
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6833 - accuracy: 0.4989
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6852 - accuracy: 0.4988
25000/25000 [==============================] - 10s 389us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Inference Need return ypred, ytrue ######################### 
Loading data...

  ### Calculate Metrics    ######################################## 

  date_run                              2020-05-13 13:14:46.942036
model_uri                                 model_keras.textcnn.py
json           [{'model_uri': 'model_keras.textcnn.py', 'maxl...
dataset_uri                   /HOBBIES_1_001_CA_1_validation.csv
metric                                                       0.5
metric_name                                       accuracy_score
Name: 0, dtype: object 

  benchmark file saved at /home/runner/work/mlmodels/mlmodels/mlmodels/example/benchmark/text_classification/ 

                       date_run               model_uri  ... metric     metric_name
0  2020-05-13 13:14:46.942036  model_keras.textcnn.py  ...    0.5  accuracy_score

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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<12:03:00, 19.9kB/s].vector_cache/glove.6B.zip:   0%|          | 451k/862M [00:00<8:27:01, 28.3kB/s]  .vector_cache/glove.6B.zip:   1%|          | 6.33M/862M [00:00<5:52:33, 40.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 13.6M/862M [00:00<4:04:45, 57.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.1M/862M [00:00<2:50:05, 82.5kB/s].vector_cache/glove.6B.zip:   3%|▎         | 25.8M/862M [00:00<1:58:19, 118kB/s] .vector_cache/glove.6B.zip:   4%|▍         | 33.2M/862M [00:01<1:22:08, 168kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.8M/862M [00:01<57:02, 240kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 48.3M/862M [00:01<39:36, 342kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:01<27:59, 482kB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:03<21:25, 627kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:03<16:29, 814kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:05<13:26, 995kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:05<10:15, 1.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:07<09:11, 1.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:07<07:28, 1.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:09<07:11, 1.84MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.8M/862M [00:09<06:00, 2.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:11<06:10, 2.13MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:11<04:56, 2.66MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:11<03:50, 3.41MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:13<11:11:14, 19.5kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:13<7:49:18, 27.9kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:15<5:30:02, 39.5kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:15<3:52:00, 56.1kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:17<2:43:34, 79.2kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:17<1:55:00, 113kB/s] .vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:19<1:22:16, 157kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:19<58:08, 221kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:21<42:37, 301kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:21<30:25, 421kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:23<23:14, 549kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:23<16:55, 753kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:25<13:47, 920kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<10:14, 1.24MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:27<09:12, 1.37MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:27<07:03, 1.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<06:57, 1.80MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:29<05:27, 2.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<05:51, 2.13MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:31<04:41, 2.66MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:32<05:18, 2.34MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<04:21, 2.84MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<04:58, 2.48MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<04:08, 2.98MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:36<04:48, 2.55MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<04:06, 2.99MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<04:44, 2.57MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:38<03:52, 3.14MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<04:39, 2.61MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:40<03:55, 3.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<04:37, 2.61MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<03:56, 3.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<04:36, 2.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<03:44, 3.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<04:32, 2.62MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<03:47, 3.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:48<04:33, 2.60MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<03:49, 3.10MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<04:30, 2.61MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:50<03:46, 3.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:52<04:30, 2.60MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<03:44, 3.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:30, 2.59MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:54<03:53, 2.99MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<04:29, 2.58MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:56<03:44, 3.08MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<04:28, 2.58MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<03:45, 3.06MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<04:24, 2.60MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<03:42, 3.08MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<04:21, 2.61MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<03:33, 3.19MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<04:18, 2.62MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<03:32, 3.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<04:18, 2.61MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<03:33, 3.16MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<04:18, 2.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:08<03:30, 3.18MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:10<04:14, 2.62MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<03:27, 3.20MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<04:11, 2.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<03:27, 3.18MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:12<02:45, 3.98MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<9:15:54, 19.7kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<6:28:37, 28.2kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<4:32:59, 39.9kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<3:11:28, 56.9kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<2:15:01, 80.2kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<1:34:56, 114kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<1:07:52, 159kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<47:57, 224kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<35:09, 304kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<25:03, 426kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<19:09, 554kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<14:11, 748kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<11:27, 920kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<08:32, 1.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<07:39, 1.37MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<05:50, 1.79MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 237M/862M [01:30<05:46, 1.80MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<04:32, 2.29MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<04:51, 2.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:32<03:53, 2.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 246M/862M [01:33<04:23, 2.34MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<03:37, 2.83MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:35<04:09, 2.46MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:23, 3.01MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:37<03:59, 2.54MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<03:17, 3.07MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:39<03:55, 2.56MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:39<03:16, 3.08MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<03:51, 2.60MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:41<03:09, 3.16MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:43<03:50, 2.59MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:43<03:08, 3.15MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<03:48, 2.59MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<03:31, 2.80MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<03:54, 2.51MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<03:11, 3.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:49<03:46, 2.57MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:49<03:05, 3.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<03:44, 2.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:51<03:04, 3.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<03:42, 2.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:53<03:03, 3.13MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:55<03:41, 2.58MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<03:01, 3.15MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<03:37, 2.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<02:58, 3.17MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<03:36, 2.60MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<03:02, 3.07MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<03:34, 2.60MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:01<03:07, 2.98MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<03:34, 2.58MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<02:53, 3.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<03:30, 2.61MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<02:53, 3.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<03:30, 2.60MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:07<02:52, 3.17MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:29, 2.59MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<02:51, 3.16MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<03:25, 2.61MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<02:50, 3.14MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:25, 2.60MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<02:53, 3.08MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:13<02:16, 3.87MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<7:26:08, 19.8kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:15<5:11:40, 28.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<3:38:48, 40.0kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<2:33:19, 57.1kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:19<1:48:05, 80.4kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<1:15:56, 114kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<54:15, 159kB/s]  .vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<38:19, 225kB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<28:03, 305kB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<19:57, 428kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<15:17, 555kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<11:04, 765kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<09:05, 926kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<06:44, 1.25MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<06:03, 1.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<04:35, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:35, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:40, 2.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:51, 2.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:05, 2.66MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:34<03:28, 2.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:50, 2.86MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:36<03:17, 2.46MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<02:46, 2.91MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<03:10, 2.51MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<02:36, 3.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:40<03:06, 2.55MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:36, 3.04MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<03:04, 2.57MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:42<02:31, 3.12MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<03:01, 2.57MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:44<02:29, 3.13MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:59, 2.58MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:46<02:26, 3.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<02:56, 2.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:48<02:19, 3.29MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:54, 2.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:27, 3.08MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<02:52, 2.61MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:52<02:24, 3.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<02:50, 2.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<02:30, 2.97MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:51, 2.58MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<02:36, 2.82MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:58<02:54, 2.52MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<02:29, 2.93MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<02:50, 2.55MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:00<02:22, 3.05MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<02:46, 2.59MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<02:29, 2.88MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:47, 2.54MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<02:33, 2.77MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:49, 2.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<02:34, 2.74MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:08<02:49, 2.47MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:08<02:28, 2.82MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<02:45, 2.50MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:10<02:27, 2.81MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:43, 2.51MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<02:16, 2.99MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<02:38, 2.56MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:14<02:25, 2.79MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:14<01:52, 3.57MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<5:37:06, 19.9kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:16<3:54:49, 28.4kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<2:45:40, 40.0kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<1:55:34, 56.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:20<1:21:05, 80.8kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<57:19, 113kB/s]   .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<40:23, 161kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<29:03, 221kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<20:36, 311kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<15:20, 414kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<10:58, 578kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<08:39, 726kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<06:20, 990kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<05:25, 1.15MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<04:04, 1.52MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<03:50, 1.60MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<02:56, 2.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<03:03, 1.98MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:25, 2.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<02:40, 2.25MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:10, 2.76MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:37<02:26, 2.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<01:56, 3.05MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:39<02:21, 2.49MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<01:53, 3.10MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<02:17, 2.53MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:04, 2.79MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:43<02:17, 2.49MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:03, 2.78MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:45<02:16, 2.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:53, 3.00MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<02:12, 2.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:47<01:48, 3.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 530M/862M [03:49<02:09, 2.56MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:49<01:42, 3.24MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<02:06, 2.58MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<01:41, 3.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:53<02:05, 2.57MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:53<01:39, 3.23MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<02:02, 2.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:55<01:48, 2.93MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<02:02, 2.56MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:57<01:41, 3.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [03:59<01:59, 2.60MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<01:45, 2.94MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<01:59, 2.57MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:01<01:38, 3.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<01:56, 2.60MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<01:35, 3.17MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<01:54, 2.61MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<01:33, 3.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<01:51, 2.63MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:07<01:32, 3.18MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<01:50, 2.63MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:09<01:31, 3.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<01:48, 2.63MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:11<01:28, 3.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:47, 2.63MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<01:34, 2.99MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:15<01:47, 2.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<01:31, 3.02MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:45, 2.59MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:17<01:34, 2.90MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<01:45, 2.55MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<01:34, 2.85MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<01:44, 2.53MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:21<01:31, 2.91MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:21<01:10, 3.72MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<3:53:23, 18.7kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<2:42:45, 26.7kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<1:53:26, 37.8kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<1:19:31, 53.8kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<55:31, 76.0kB/s]  .vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<39:10, 108kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<27:37, 150kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<19:27, 213kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<14:06, 290kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<10:08, 402kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:31<07:01, 571kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<28:35, 140kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<20:12, 198kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:33<13:59, 282kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<16:20, 241kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<11:41, 337kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<08:38, 449kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<06:14, 619kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<04:54, 777kB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<03:36, 1.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<03:04, 1.21MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<02:26, 1.53MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:13, 1.64MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:43<01:49, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<01:48, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:26, 2.49MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:33, 2.28MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 651M/862M [04:47<01:16, 2.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<01:25, 2.44MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:08, 3.00MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:50<01:20, 2.53MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:07, 3.01MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:52<01:17, 2.57MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:04, 3.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:15, 2.60MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:54<01:04, 3.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:56<01:13, 2.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:56<00:59, 3.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:58<01:11, 2.62MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:58<01:01, 3.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:00<01:10, 2.60MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:00<00:56, 3.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:02<01:08, 2.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:02<00:56, 3.14MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<01:06, 2.62MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:04<00:59, 2.95MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<01:06, 2.58MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:06<00:55, 3.07MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<01:04, 2.60MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:08<00:51, 3.20MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<01:01, 2.62MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<00:53, 3.02MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:00, 2.59MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:12<00:54, 2.91MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<01:00, 2.56MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:14<00:49, 3.12MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<00:57, 2.59MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:16<00:52, 2.84MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<00:57, 2.53MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:18<00:47, 3.07MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<00:55, 2.56MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<00:50, 2.81MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:54, 2.51MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:22<00:44, 3.09MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<00:51, 2.57MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<00:42, 3.09MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<00:49, 2.60MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:40, 3.14MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<00:47, 2.62MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<00:38, 3.23MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:45, 2.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:37, 3.18MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:30<00:29, 4.02MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<1:45:25, 18.5kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<1:13:17, 26.4kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<50:11, 37.4kB/s]  .vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<35:01, 53.3kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<24:02, 75.2kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:36<16:49, 107kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<11:39, 149kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<08:16, 209kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<05:50, 286kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:40<04:08, 401kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<03:03, 525kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<02:11, 725kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:43, 887kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<01:16, 1.20MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:05, 1.34MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:49, 1.75MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:47, 1.78MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:36, 2.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:37, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:29, 2.69MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:31, 2.36MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:24, 2.99MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<00:28, 2.49MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:22, 3.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:26, 2.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:21, 3.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:24, 2.60MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:19, 3.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [05:59<00:22, 2.62MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:18, 3.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:01<00:20, 2.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:16, 3.21MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:03<00:19, 2.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:16, 3.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<00:17, 2.62MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:05<00:14, 3.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:07<00:16, 2.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:07<00:13, 3.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:09<00:14, 2.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:09<00:12, 3.09MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:11<00:13, 2.61MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:11<00:10, 3.20MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:13<00:11, 2.62MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:13<00:09, 3.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:09, 2.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:15<00:07, 3.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:17<00:08, 2.60MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:17<00:06, 3.16MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:06, 2.59MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:05, 3.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:21<00:05, 2.59MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:21<00:03, 3.19MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:21<00:02, 4.34MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:07, 1.17MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:05, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:03, 1.55MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:25<00:02, 1.84MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:26<00:00, 2.56MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:01, 835kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 1.12MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 811/400000 [00:00<00:49, 8101.99it/s]  0%|          | 1603/400000 [00:00<00:49, 8043.00it/s]  1%|          | 2443/400000 [00:00<00:48, 8145.29it/s]  1%|          | 3250/400000 [00:00<00:48, 8120.95it/s]  1%|          | 4077/400000 [00:00<00:48, 8162.68it/s]  1%|          | 4930/400000 [00:00<00:47, 8267.75it/s]  1%|▏         | 5736/400000 [00:00<00:48, 8202.26it/s]  2%|▏         | 6546/400000 [00:00<00:48, 8170.93it/s]  2%|▏         | 7350/400000 [00:00<00:48, 8126.82it/s]  2%|▏         | 8153/400000 [00:01<00:48, 8096.61it/s]  2%|▏         | 8995/400000 [00:01<00:47, 8190.19it/s]  2%|▏         | 9799/400000 [00:01<00:48, 8125.92it/s]  3%|▎         | 10605/400000 [00:01<00:48, 8105.92it/s]  3%|▎         | 11409/400000 [00:01<00:48, 8005.21it/s]  3%|▎         | 12237/400000 [00:01<00:47, 8083.83it/s]  3%|▎         | 13064/400000 [00:01<00:47, 8138.50it/s]  3%|▎         | 13876/400000 [00:01<00:47, 8125.15it/s]  4%|▎         | 14715/400000 [00:01<00:46, 8200.42it/s]  4%|▍         | 15558/400000 [00:01<00:46, 8265.30it/s]  4%|▍         | 16385/400000 [00:02<00:46, 8192.59it/s]  4%|▍         | 17239/400000 [00:02<00:46, 8291.31it/s]  5%|▍         | 18069/400000 [00:02<00:46, 8244.77it/s]  5%|▍         | 18921/400000 [00:02<00:45, 8324.77it/s]  5%|▍         | 19780/400000 [00:02<00:45, 8400.10it/s]  5%|▌         | 20621/400000 [00:02<00:45, 8328.40it/s]  5%|▌         | 21455/400000 [00:02<00:46, 8115.93it/s]  6%|▌         | 22269/400000 [00:02<00:46, 8081.18it/s]  6%|▌         | 23095/400000 [00:02<00:46, 8131.97it/s]  6%|▌         | 23910/400000 [00:02<00:46, 8021.62it/s]  6%|▌         | 24714/400000 [00:03<00:46, 8013.62it/s]  6%|▋         | 25530/400000 [00:03<00:46, 8056.17it/s]  7%|▋         | 26387/400000 [00:03<00:45, 8202.35it/s]  7%|▋         | 27237/400000 [00:03<00:44, 8287.38it/s]  7%|▋         | 28067/400000 [00:03<00:44, 8286.68it/s]  7%|▋         | 28897/400000 [00:03<00:45, 8156.16it/s]  7%|▋         | 29714/400000 [00:03<00:46, 8030.94it/s]  8%|▊         | 30545/400000 [00:03<00:45, 8109.52it/s]  8%|▊         | 31357/400000 [00:03<00:46, 7955.76it/s]  8%|▊         | 32154/400000 [00:03<00:47, 7778.59it/s]  8%|▊         | 32934/400000 [00:04<00:47, 7766.91it/s]  8%|▊         | 33758/400000 [00:04<00:46, 7901.10it/s]  9%|▊         | 34550/400000 [00:04<00:46, 7896.38it/s]  9%|▉         | 35346/400000 [00:04<00:46, 7913.94it/s]  9%|▉         | 36166/400000 [00:04<00:45, 7997.23it/s]  9%|▉         | 36980/400000 [00:04<00:45, 8037.67it/s]  9%|▉         | 37813/400000 [00:04<00:44, 8121.49it/s] 10%|▉         | 38679/400000 [00:04<00:43, 8274.77it/s] 10%|▉         | 39521/400000 [00:04<00:43, 8316.62it/s] 10%|█         | 40354/400000 [00:04<00:43, 8301.04it/s] 10%|█         | 41185/400000 [00:05<00:44, 8109.57it/s] 11%|█         | 42015/400000 [00:05<00:43, 8163.44it/s] 11%|█         | 42833/400000 [00:05<00:43, 8167.45it/s] 11%|█         | 43651/400000 [00:05<00:43, 8122.09it/s] 11%|█         | 44486/400000 [00:05<00:43, 8187.61it/s] 11%|█▏        | 45306/400000 [00:05<00:43, 8144.70it/s] 12%|█▏        | 46151/400000 [00:05<00:42, 8233.63it/s] 12%|█▏        | 46987/400000 [00:05<00:42, 8269.35it/s] 12%|█▏        | 47821/400000 [00:05<00:42, 8289.13it/s] 12%|█▏        | 48651/400000 [00:05<00:42, 8267.56it/s] 12%|█▏        | 49478/400000 [00:06<00:43, 8083.81it/s] 13%|█▎        | 50305/400000 [00:06<00:42, 8138.17it/s] 13%|█▎        | 51159/400000 [00:06<00:42, 8253.84it/s] 13%|█▎        | 51990/400000 [00:06<00:42, 8266.67it/s] 13%|█▎        | 52818/400000 [00:06<00:42, 8213.34it/s] 13%|█▎        | 53640/400000 [00:06<00:42, 8056.02it/s] 14%|█▎        | 54448/400000 [00:06<00:42, 8061.47it/s] 14%|█▍        | 55282/400000 [00:06<00:42, 8140.01it/s] 14%|█▍        | 56097/400000 [00:06<00:42, 8090.67it/s] 14%|█▍        | 56907/400000 [00:06<00:42, 8054.95it/s] 14%|█▍        | 57713/400000 [00:07<00:42, 7990.73it/s] 15%|█▍        | 58517/400000 [00:07<00:42, 8003.93it/s] 15%|█▍        | 59383/400000 [00:07<00:41, 8187.66it/s] 15%|█▌        | 60244/400000 [00:07<00:40, 8309.67it/s] 15%|█▌        | 61105/400000 [00:07<00:40, 8395.10it/s] 15%|█▌        | 61946/400000 [00:07<00:40, 8279.41it/s] 16%|█▌        | 62776/400000 [00:07<00:40, 8268.77it/s] 16%|█▌        | 63646/400000 [00:07<00:40, 8392.85it/s] 16%|█▌        | 64487/400000 [00:07<00:40, 8348.91it/s] 16%|█▋        | 65323/400000 [00:08<00:40, 8229.19it/s] 17%|█▋        | 66147/400000 [00:08<00:42, 7940.99it/s] 17%|█▋        | 66944/400000 [00:08<00:42, 7898.68it/s] 17%|█▋        | 67743/400000 [00:08<00:41, 7924.45it/s] 17%|█▋        | 68537/400000 [00:08<00:41, 7914.09it/s] 17%|█▋        | 69330/400000 [00:08<00:42, 7740.08it/s] 18%|█▊        | 70106/400000 [00:08<00:43, 7544.84it/s] 18%|█▊        | 70863/400000 [00:08<00:44, 7463.79it/s] 18%|█▊        | 71612/400000 [00:08<00:43, 7469.55it/s] 18%|█▊        | 72365/400000 [00:08<00:43, 7487.07it/s] 18%|█▊        | 73115/400000 [00:09<00:43, 7469.87it/s] 18%|█▊        | 73863/400000 [00:09<00:44, 7322.48it/s] 19%|█▊        | 74675/400000 [00:09<00:43, 7544.21it/s] 19%|█▉        | 75462/400000 [00:09<00:42, 7637.29it/s] 19%|█▉        | 76247/400000 [00:09<00:42, 7699.00it/s] 19%|█▉        | 77052/400000 [00:09<00:41, 7799.56it/s] 19%|█▉        | 77834/400000 [00:09<00:41, 7769.80it/s] 20%|█▉        | 78645/400000 [00:09<00:40, 7867.50it/s] 20%|█▉        | 79433/400000 [00:09<00:41, 7804.07it/s] 20%|██        | 80251/400000 [00:09<00:40, 7912.87it/s] 20%|██        | 81055/400000 [00:10<00:40, 7949.48it/s] 20%|██        | 81851/400000 [00:10<00:40, 7880.39it/s] 21%|██        | 82649/400000 [00:10<00:40, 7908.06it/s] 21%|██        | 83470/400000 [00:10<00:39, 7994.77it/s] 21%|██        | 84271/400000 [00:10<00:39, 7992.85it/s] 21%|██▏       | 85071/400000 [00:10<00:39, 7954.13it/s] 21%|██▏       | 85867/400000 [00:10<00:39, 7905.71it/s] 22%|██▏       | 86694/400000 [00:10<00:39, 8010.33it/s] 22%|██▏       | 87496/400000 [00:10<00:39, 7912.20it/s] 22%|██▏       | 88288/400000 [00:10<00:39, 7812.88it/s] 22%|██▏       | 89071/400000 [00:11<00:40, 7763.61it/s] 22%|██▏       | 89848/400000 [00:11<00:40, 7726.96it/s] 23%|██▎       | 90666/400000 [00:11<00:39, 7856.29it/s] 23%|██▎       | 91453/400000 [00:11<00:39, 7780.25it/s] 23%|██▎       | 92232/400000 [00:11<00:39, 7778.56it/s] 23%|██▎       | 93021/400000 [00:11<00:39, 7807.65it/s] 23%|██▎       | 93803/400000 [00:11<00:39, 7707.54it/s] 24%|██▎       | 94575/400000 [00:11<00:40, 7594.04it/s] 24%|██▍       | 95386/400000 [00:11<00:39, 7739.81it/s] 24%|██▍       | 96167/400000 [00:11<00:39, 7758.79it/s] 24%|██▍       | 96946/400000 [00:12<00:39, 7766.86it/s] 24%|██▍       | 97724/400000 [00:12<00:39, 7673.60it/s] 25%|██▍       | 98493/400000 [00:12<00:39, 7649.78it/s] 25%|██▍       | 99291/400000 [00:12<00:38, 7745.09it/s] 25%|██▌       | 100089/400000 [00:12<00:38, 7811.93it/s] 25%|██▌       | 100871/400000 [00:12<00:38, 7797.90it/s] 25%|██▌       | 101652/400000 [00:12<00:39, 7547.56it/s] 26%|██▌       | 102409/400000 [00:12<00:40, 7378.60it/s] 26%|██▌       | 103150/400000 [00:12<00:41, 7225.37it/s] 26%|██▌       | 103921/400000 [00:13<00:40, 7361.39it/s] 26%|██▌       | 104687/400000 [00:13<00:39, 7447.47it/s] 26%|██▋       | 105434/400000 [00:13<00:39, 7454.12it/s] 27%|██▋       | 106235/400000 [00:13<00:38, 7610.64it/s] 27%|██▋       | 107030/400000 [00:13<00:38, 7706.21it/s] 27%|██▋       | 107812/400000 [00:13<00:37, 7736.93it/s] 27%|██▋       | 108618/400000 [00:13<00:37, 7829.50it/s] 27%|██▋       | 109407/400000 [00:13<00:37, 7847.07it/s] 28%|██▊       | 110199/400000 [00:13<00:36, 7868.36it/s] 28%|██▊       | 111027/400000 [00:13<00:36, 7984.93it/s] 28%|██▊       | 111827/400000 [00:14<00:36, 7969.30it/s] 28%|██▊       | 112625/400000 [00:14<00:36, 7862.12it/s] 28%|██▊       | 113412/400000 [00:14<00:37, 7735.08it/s] 29%|██▊       | 114209/400000 [00:14<00:36, 7802.49it/s] 29%|██▊       | 114994/400000 [00:14<00:36, 7815.06it/s] 29%|██▉       | 115780/400000 [00:14<00:36, 7825.25it/s] 29%|██▉       | 116563/400000 [00:14<00:36, 7779.61it/s] 29%|██▉       | 117349/400000 [00:14<00:36, 7801.48it/s] 30%|██▉       | 118166/400000 [00:14<00:35, 7906.79it/s] 30%|██▉       | 119001/400000 [00:14<00:34, 8034.64it/s] 30%|██▉       | 119835/400000 [00:15<00:34, 8122.10it/s] 30%|███       | 120649/400000 [00:15<00:34, 8101.10it/s] 30%|███       | 121460/400000 [00:15<00:34, 8005.04it/s] 31%|███       | 122288/400000 [00:15<00:34, 8084.33it/s] 31%|███       | 123098/400000 [00:15<00:34, 7986.72it/s] 31%|███       | 123898/400000 [00:15<00:34, 7930.90it/s] 31%|███       | 124696/400000 [00:15<00:34, 7945.22it/s] 31%|███▏      | 125491/400000 [00:15<00:34, 7863.00it/s] 32%|███▏      | 126314/400000 [00:15<00:34, 7968.59it/s] 32%|███▏      | 127112/400000 [00:15<00:34, 7814.82it/s] 32%|███▏      | 127924/400000 [00:16<00:34, 7901.46it/s] 32%|███▏      | 128716/400000 [00:16<00:35, 7689.41it/s] 32%|███▏      | 129493/400000 [00:16<00:35, 7713.15it/s] 33%|███▎      | 130301/400000 [00:16<00:34, 7817.46it/s] 33%|███▎      | 131110/400000 [00:16<00:34, 7896.06it/s] 33%|███▎      | 131912/400000 [00:16<00:33, 7930.18it/s] 33%|███▎      | 132711/400000 [00:16<00:33, 7947.97it/s] 33%|███▎      | 133507/400000 [00:16<00:33, 7951.55it/s] 34%|███▎      | 134321/400000 [00:16<00:33, 8006.39it/s] 34%|███▍      | 135123/400000 [00:16<00:33, 7981.68it/s] 34%|███▍      | 135922/400000 [00:17<00:33, 7921.62it/s] 34%|███▍      | 136715/400000 [00:17<00:33, 7807.36it/s] 34%|███▍      | 137512/400000 [00:17<00:33, 7852.63it/s] 35%|███▍      | 138344/400000 [00:17<00:32, 7986.99it/s] 35%|███▍      | 139172/400000 [00:17<00:32, 8072.59it/s] 35%|███▍      | 139989/400000 [00:17<00:32, 8100.21it/s] 35%|███▌      | 140804/400000 [00:17<00:31, 8111.30it/s] 35%|███▌      | 141616/400000 [00:17<00:32, 8017.02it/s] 36%|███▌      | 142440/400000 [00:17<00:31, 8080.81it/s] 36%|███▌      | 143258/400000 [00:17<00:31, 8107.97it/s] 36%|███▌      | 144070/400000 [00:18<00:31, 8065.32it/s] 36%|███▌      | 144877/400000 [00:18<00:32, 7809.35it/s] 36%|███▋      | 145660/400000 [00:18<00:32, 7721.63it/s] 37%|███▋      | 146458/400000 [00:18<00:32, 7796.31it/s] 37%|███▋      | 147239/400000 [00:18<00:32, 7721.83it/s] 37%|███▋      | 148020/400000 [00:18<00:32, 7746.70it/s] 37%|███▋      | 148823/400000 [00:18<00:32, 7826.66it/s] 37%|███▋      | 149617/400000 [00:18<00:31, 7859.74it/s] 38%|███▊      | 150427/400000 [00:18<00:31, 7927.38it/s] 38%|███▊      | 151221/400000 [00:19<00:31, 7897.28it/s] 38%|███▊      | 152013/400000 [00:19<00:31, 7902.89it/s] 38%|███▊      | 152804/400000 [00:19<00:31, 7902.89it/s] 38%|███▊      | 153595/400000 [00:19<00:31, 7831.31it/s] 39%|███▊      | 154379/400000 [00:19<00:31, 7776.58it/s] 39%|███▉      | 155174/400000 [00:19<00:31, 7825.80it/s] 39%|███▉      | 155957/400000 [00:19<00:31, 7809.74it/s] 39%|███▉      | 156739/400000 [00:19<00:31, 7801.58it/s] 39%|███▉      | 157540/400000 [00:19<00:30, 7862.14it/s] 40%|███▉      | 158354/400000 [00:19<00:30, 7942.74it/s] 40%|███▉      | 159149/400000 [00:20<00:30, 7913.42it/s] 40%|███▉      | 159945/400000 [00:20<00:30, 7925.49it/s] 40%|████      | 160762/400000 [00:20<00:29, 7997.20it/s] 40%|████      | 161563/400000 [00:20<00:30, 7920.57it/s] 41%|████      | 162356/400000 [00:20<00:30, 7893.39it/s] 41%|████      | 163168/400000 [00:20<00:29, 7958.68it/s] 41%|████      | 163973/400000 [00:20<00:29, 7984.81it/s] 41%|████      | 164776/400000 [00:20<00:29, 7996.48it/s] 41%|████▏     | 165576/400000 [00:20<00:29, 7959.35it/s] 42%|████▏     | 166380/400000 [00:20<00:29, 7982.49it/s] 42%|████▏     | 167184/400000 [00:21<00:29, 7997.47it/s] 42%|████▏     | 167984/400000 [00:21<00:29, 7804.88it/s] 42%|████▏     | 168781/400000 [00:21<00:29, 7850.66it/s] 42%|████▏     | 169567/400000 [00:21<00:29, 7757.11it/s] 43%|████▎     | 170370/400000 [00:21<00:29, 7834.96it/s] 43%|████▎     | 171174/400000 [00:21<00:28, 7892.81it/s] 43%|████▎     | 171964/400000 [00:21<00:28, 7867.39it/s] 43%|████▎     | 172752/400000 [00:21<00:28, 7844.49it/s] 43%|████▎     | 173537/400000 [00:21<00:29, 7802.99it/s] 44%|████▎     | 174340/400000 [00:21<00:28, 7867.17it/s] 44%|████▍     | 175138/400000 [00:22<00:28, 7898.89it/s] 44%|████▍     | 175944/400000 [00:22<00:28, 7945.86it/s] 44%|████▍     | 176739/400000 [00:22<00:28, 7925.35it/s] 44%|████▍     | 177532/400000 [00:22<00:28, 7870.37it/s] 45%|████▍     | 178320/400000 [00:22<00:28, 7865.61it/s] 45%|████▍     | 179107/400000 [00:22<00:28, 7797.26it/s] 45%|████▍     | 179888/400000 [00:22<00:28, 7797.83it/s] 45%|████▌     | 180668/400000 [00:22<00:28, 7752.77it/s] 45%|████▌     | 181444/400000 [00:22<00:28, 7710.76it/s] 46%|████▌     | 182237/400000 [00:22<00:28, 7773.64it/s] 46%|████▌     | 183015/400000 [00:23<00:28, 7585.61it/s] 46%|████▌     | 183775/400000 [00:23<00:28, 7548.28it/s] 46%|████▌     | 184537/400000 [00:23<00:28, 7568.06it/s] 46%|████▋     | 185304/400000 [00:23<00:28, 7597.91it/s] 47%|████▋     | 186065/400000 [00:23<00:28, 7475.33it/s] 47%|████▋     | 186842/400000 [00:23<00:28, 7560.57it/s] 47%|████▋     | 187634/400000 [00:23<00:27, 7663.04it/s] 47%|████▋     | 188437/400000 [00:23<00:27, 7766.71it/s] 47%|████▋     | 189218/400000 [00:23<00:27, 7778.68it/s] 48%|████▊     | 190013/400000 [00:23<00:26, 7827.93it/s] 48%|████▊     | 190817/400000 [00:24<00:26, 7888.58it/s] 48%|████▊     | 191632/400000 [00:24<00:26, 7964.58it/s] 48%|████▊     | 192449/400000 [00:24<00:25, 8022.99it/s] 48%|████▊     | 193252/400000 [00:24<00:25, 7994.78it/s] 49%|████▊     | 194052/400000 [00:24<00:26, 7891.76it/s] 49%|████▊     | 194858/400000 [00:24<00:25, 7938.74it/s] 49%|████▉     | 195689/400000 [00:24<00:25, 8042.83it/s] 49%|████▉     | 196510/400000 [00:24<00:25, 8090.70it/s] 49%|████▉     | 197320/400000 [00:24<00:25, 8059.85it/s] 50%|████▉     | 198127/400000 [00:24<00:25, 8025.73it/s] 50%|████▉     | 198930/400000 [00:25<00:25, 7995.87it/s] 50%|████▉     | 199730/400000 [00:25<00:25, 7952.24it/s] 50%|█████     | 200526/400000 [00:25<00:25, 7920.78it/s] 50%|█████     | 201319/400000 [00:25<00:25, 7789.25it/s] 51%|█████     | 202116/400000 [00:25<00:25, 7839.45it/s] 51%|█████     | 202918/400000 [00:25<00:24, 7891.44it/s] 51%|█████     | 203708/400000 [00:25<00:24, 7867.63it/s] 51%|█████     | 204533/400000 [00:25<00:24, 7976.84it/s] 51%|█████▏    | 205332/400000 [00:25<00:24, 7974.25it/s] 52%|█████▏    | 206130/400000 [00:25<00:24, 7878.91it/s] 52%|█████▏    | 206975/400000 [00:26<00:24, 8041.60it/s] 52%|█████▏    | 207781/400000 [00:26<00:23, 8017.53it/s] 52%|█████▏    | 208602/400000 [00:26<00:23, 8071.85it/s] 52%|█████▏    | 209410/400000 [00:26<00:23, 8026.14it/s] 53%|█████▎    | 210236/400000 [00:26<00:23, 8092.94it/s] 53%|█████▎    | 211049/400000 [00:26<00:23, 8103.29it/s] 53%|█████▎    | 211882/400000 [00:26<00:23, 8168.07it/s] 53%|█████▎    | 212736/400000 [00:26<00:22, 8275.32it/s] 53%|█████▎    | 213565/400000 [00:26<00:22, 8260.72it/s] 54%|█████▎    | 214399/400000 [00:26<00:22, 8281.90it/s] 54%|█████▍    | 215246/400000 [00:27<00:22, 8332.83it/s] 54%|█████▍    | 216082/400000 [00:27<00:22, 8338.37it/s] 54%|█████▍    | 216917/400000 [00:27<00:22, 8174.38it/s] 54%|█████▍    | 217736/400000 [00:27<00:22, 8074.27it/s] 55%|█████▍    | 218562/400000 [00:27<00:22, 8127.10it/s] 55%|█████▍    | 219401/400000 [00:27<00:22, 8203.87it/s] 55%|█████▌    | 220223/400000 [00:27<00:22, 8165.23it/s] 55%|█████▌    | 221041/400000 [00:27<00:22, 8126.13it/s] 55%|█████▌    | 221855/400000 [00:27<00:22, 7995.46it/s] 56%|█████▌    | 222671/400000 [00:28<00:22, 8041.60it/s] 56%|█████▌    | 223476/400000 [00:28<00:22, 7924.20it/s] 56%|█████▌    | 224270/400000 [00:28<00:22, 7847.74it/s] 56%|█████▋    | 225067/400000 [00:28<00:22, 7881.12it/s] 56%|█████▋    | 225856/400000 [00:28<00:22, 7880.62it/s] 57%|█████▋    | 226668/400000 [00:28<00:21, 7949.61it/s] 57%|█████▋    | 227475/400000 [00:28<00:21, 7983.40it/s] 57%|█████▋    | 228299/400000 [00:28<00:21, 8056.92it/s] 57%|█████▋    | 229130/400000 [00:28<00:21, 8129.59it/s] 57%|█████▋    | 229959/400000 [00:28<00:20, 8175.54it/s] 58%|█████▊    | 230819/400000 [00:29<00:20, 8298.15it/s] 58%|█████▊    | 231666/400000 [00:29<00:20, 8347.18it/s] 58%|█████▊    | 232502/400000 [00:29<00:20, 8337.93it/s] 58%|█████▊    | 233337/400000 [00:29<00:19, 8337.19it/s] 59%|█████▊    | 234171/400000 [00:29<00:20, 8205.91it/s] 59%|█████▊    | 234993/400000 [00:29<00:20, 8126.78it/s] 59%|█████▉    | 235810/400000 [00:29<00:20, 8137.52it/s] 59%|█████▉    | 236650/400000 [00:29<00:19, 8212.94it/s] 59%|█████▉    | 237472/400000 [00:29<00:19, 8145.01it/s] 60%|█████▉    | 238291/400000 [00:29<00:19, 8158.00it/s] 60%|█████▉    | 239108/400000 [00:30<00:19, 8088.97it/s] 60%|█████▉    | 239938/400000 [00:30<00:19, 8148.74it/s] 60%|██████    | 240754/400000 [00:30<00:19, 8124.37it/s] 60%|██████    | 241572/400000 [00:30<00:19, 8138.80it/s] 61%|██████    | 242390/400000 [00:30<00:19, 8150.03it/s] 61%|██████    | 243206/400000 [00:30<00:19, 8102.55it/s] 61%|██████    | 244017/400000 [00:30<00:19, 8025.01it/s] 61%|██████    | 244820/400000 [00:30<00:20, 7750.89it/s] 61%|██████▏   | 245598/400000 [00:30<00:20, 7713.33it/s] 62%|██████▏   | 246375/400000 [00:30<00:19, 7729.08it/s] 62%|██████▏   | 247183/400000 [00:31<00:19, 7830.55it/s] 62%|██████▏   | 247968/400000 [00:31<00:19, 7779.68it/s] 62%|██████▏   | 248747/400000 [00:31<00:19, 7694.72it/s] 62%|██████▏   | 249518/400000 [00:31<00:19, 7673.61it/s] 63%|██████▎   | 250290/400000 [00:31<00:19, 7687.13it/s] 63%|██████▎   | 251094/400000 [00:31<00:19, 7786.41it/s] 63%|██████▎   | 251907/400000 [00:31<00:18, 7884.85it/s] 63%|██████▎   | 252726/400000 [00:31<00:18, 7972.40it/s] 63%|██████▎   | 253551/400000 [00:31<00:18, 8052.45it/s] 64%|██████▎   | 254358/400000 [00:31<00:18, 8056.45it/s] 64%|██████▍   | 255165/400000 [00:32<00:18, 7928.21it/s] 64%|██████▍   | 255959/400000 [00:32<00:18, 7719.28it/s] 64%|██████▍   | 256733/400000 [00:32<00:18, 7686.90it/s] 64%|██████▍   | 257503/400000 [00:32<00:18, 7650.05it/s] 65%|██████▍   | 258269/400000 [00:32<00:18, 7527.06it/s] 65%|██████▍   | 259023/400000 [00:32<00:18, 7431.50it/s] 65%|██████▍   | 259768/400000 [00:32<00:19, 7370.14it/s] 65%|██████▌   | 260506/400000 [00:32<00:18, 7355.70it/s] 65%|██████▌   | 261243/400000 [00:32<00:18, 7351.08it/s] 66%|██████▌   | 262009/400000 [00:33<00:18, 7439.31it/s] 66%|██████▌   | 262823/400000 [00:33<00:17, 7635.33it/s] 66%|██████▌   | 263632/400000 [00:33<00:17, 7763.19it/s] 66%|██████▌   | 264444/400000 [00:33<00:17, 7866.25it/s] 66%|██████▋   | 265275/400000 [00:33<00:16, 7993.31it/s] 67%|██████▋   | 266126/400000 [00:33<00:16, 8140.26it/s] 67%|██████▋   | 266942/400000 [00:33<00:16, 8058.96it/s] 67%|██████▋   | 267750/400000 [00:33<00:16, 7939.12it/s] 67%|██████▋   | 268587/400000 [00:33<00:16, 8061.53it/s] 67%|██████▋   | 269395/400000 [00:33<00:16, 8039.82it/s] 68%|██████▊   | 270228/400000 [00:34<00:15, 8123.71it/s] 68%|██████▊   | 271042/400000 [00:34<00:16, 7909.44it/s] 68%|██████▊   | 271874/400000 [00:34<00:15, 8027.15it/s] 68%|██████▊   | 272718/400000 [00:34<00:15, 8143.27it/s] 68%|██████▊   | 273552/400000 [00:34<00:15, 8200.90it/s] 69%|██████▊   | 274374/400000 [00:34<00:15, 8176.18it/s] 69%|██████▉   | 275193/400000 [00:34<00:15, 8139.31it/s] 69%|██████▉   | 276027/400000 [00:34<00:15, 8198.29it/s] 69%|██████▉   | 276893/400000 [00:34<00:14, 8329.38it/s] 69%|██████▉   | 277761/400000 [00:34<00:14, 8430.35it/s] 70%|██████▉   | 278632/400000 [00:35<00:14, 8510.24it/s] 70%|██████▉   | 279484/400000 [00:35<00:14, 8390.94it/s] 70%|███████   | 280325/400000 [00:35<00:14, 8384.45it/s] 70%|███████   | 281165/400000 [00:35<00:14, 8289.66it/s] 70%|███████   | 281995/400000 [00:35<00:14, 8283.41it/s] 71%|███████   | 282832/400000 [00:35<00:14, 8306.80it/s] 71%|███████   | 283664/400000 [00:35<00:14, 8082.31it/s] 71%|███████   | 284474/400000 [00:35<00:14, 7875.55it/s] 71%|███████▏  | 285264/400000 [00:35<00:15, 7610.26it/s] 72%|███████▏  | 286029/400000 [00:35<00:15, 7578.71it/s] 72%|███████▏  | 286790/400000 [00:36<00:15, 7507.57it/s] 72%|███████▏  | 287543/400000 [00:36<00:15, 7414.58it/s] 72%|███████▏  | 288287/400000 [00:36<00:15, 7306.94it/s] 72%|███████▏  | 289020/400000 [00:36<00:15, 7236.01it/s] 72%|███████▏  | 289745/400000 [00:36<00:15, 7237.88it/s] 73%|███████▎  | 290470/400000 [00:36<00:15, 7206.30it/s] 73%|███████▎  | 291192/400000 [00:36<00:15, 7166.53it/s] 73%|███████▎  | 291911/400000 [00:36<00:15, 7171.12it/s] 73%|███████▎  | 292661/400000 [00:36<00:14, 7266.53it/s] 73%|███████▎  | 293452/400000 [00:36<00:14, 7446.84it/s] 74%|███████▎  | 294254/400000 [00:37<00:13, 7604.18it/s] 74%|███████▍  | 295067/400000 [00:37<00:13, 7754.39it/s] 74%|███████▍  | 295933/400000 [00:37<00:12, 8005.32it/s] 74%|███████▍  | 296737/400000 [00:37<00:13, 7938.86it/s] 74%|███████▍  | 297534/400000 [00:37<00:12, 7913.40it/s] 75%|███████▍  | 298358/400000 [00:37<00:12, 8006.52it/s] 75%|███████▍  | 299161/400000 [00:37<00:12, 7882.17it/s] 75%|███████▍  | 299956/400000 [00:37<00:12, 7901.84it/s] 75%|███████▌  | 300748/400000 [00:37<00:12, 7837.25it/s] 75%|███████▌  | 301533/400000 [00:38<00:12, 7819.65it/s] 76%|███████▌  | 302316/400000 [00:38<00:12, 7705.95it/s] 76%|███████▌  | 303088/400000 [00:38<00:12, 7663.69it/s] 76%|███████▌  | 303873/400000 [00:38<00:12, 7718.21it/s] 76%|███████▌  | 304646/400000 [00:38<00:12, 7716.59it/s] 76%|███████▋  | 305457/400000 [00:38<00:12, 7829.26it/s] 77%|███████▋  | 306272/400000 [00:38<00:11, 7922.41it/s] 77%|███████▋  | 307111/400000 [00:38<00:11, 8054.82it/s] 77%|███████▋  | 307968/400000 [00:38<00:11, 8202.73it/s] 77%|███████▋  | 308809/400000 [00:38<00:11, 8263.28it/s] 77%|███████▋  | 309644/400000 [00:39<00:10, 8288.09it/s] 78%|███████▊  | 310474/400000 [00:39<00:10, 8200.19it/s] 78%|███████▊  | 311295/400000 [00:39<00:11, 7862.09it/s] 78%|███████▊  | 312085/400000 [00:39<00:11, 7848.78it/s] 78%|███████▊  | 312873/400000 [00:39<00:11, 7771.06it/s] 78%|███████▊  | 313652/400000 [00:39<00:11, 7629.41it/s] 79%|███████▊  | 314429/400000 [00:39<00:11, 7668.35it/s] 79%|███████▉  | 315224/400000 [00:39<00:10, 7749.79it/s] 79%|███████▉  | 316095/400000 [00:39<00:10, 8012.42it/s] 79%|███████▉  | 316948/400000 [00:39<00:10, 8157.68it/s] 79%|███████▉  | 317791/400000 [00:40<00:09, 8235.18it/s] 80%|███████▉  | 318626/400000 [00:40<00:09, 8269.22it/s] 80%|███████▉  | 319455/400000 [00:40<00:09, 8160.58it/s] 80%|████████  | 320321/400000 [00:40<00:09, 8303.46it/s] 80%|████████  | 321165/400000 [00:40<00:09, 8342.69it/s] 81%|████████  | 322001/400000 [00:40<00:09, 8345.55it/s] 81%|████████  | 322837/400000 [00:40<00:09, 8243.80it/s] 81%|████████  | 323663/400000 [00:40<00:09, 8147.18it/s] 81%|████████  | 324479/400000 [00:40<00:09, 8133.28it/s] 81%|████████▏ | 325293/400000 [00:40<00:09, 8082.79it/s] 82%|████████▏ | 326102/400000 [00:41<00:09, 8042.70it/s] 82%|████████▏ | 326907/400000 [00:41<00:09, 7979.65it/s] 82%|████████▏ | 327706/400000 [00:41<00:09, 7946.22it/s] 82%|████████▏ | 328501/400000 [00:41<00:09, 7932.20it/s] 82%|████████▏ | 329346/400000 [00:41<00:08, 8080.26it/s] 83%|████████▎ | 330172/400000 [00:41<00:08, 8132.41it/s] 83%|████████▎ | 330986/400000 [00:41<00:08, 8127.27it/s] 83%|████████▎ | 331815/400000 [00:41<00:08, 8173.70it/s] 83%|████████▎ | 332653/400000 [00:41<00:08, 8233.06it/s] 83%|████████▎ | 333498/400000 [00:41<00:08, 8296.08it/s] 84%|████████▎ | 334328/400000 [00:42<00:07, 8272.01it/s] 84%|████████▍ | 335156/400000 [00:42<00:07, 8127.30it/s] 84%|████████▍ | 335970/400000 [00:42<00:08, 7734.48it/s] 84%|████████▍ | 336810/400000 [00:42<00:07, 7919.98it/s] 84%|████████▍ | 337649/400000 [00:42<00:07, 8053.98it/s] 85%|████████▍ | 338502/400000 [00:42<00:07, 8190.41it/s] 85%|████████▍ | 339363/400000 [00:42<00:07, 8309.52it/s] 85%|████████▌ | 340197/400000 [00:42<00:07, 8301.84it/s] 85%|████████▌ | 341042/400000 [00:42<00:07, 8343.46it/s] 85%|████████▌ | 341878/400000 [00:42<00:07, 8174.97it/s] 86%|████████▌ | 342698/400000 [00:43<00:07, 8084.13it/s] 86%|████████▌ | 343508/400000 [00:43<00:07, 7883.88it/s] 86%|████████▌ | 344337/400000 [00:43<00:06, 8000.89it/s] 86%|████████▋ | 345207/400000 [00:43<00:06, 8195.94it/s] 87%|████████▋ | 346030/400000 [00:43<00:06, 8152.00it/s] 87%|████████▋ | 346863/400000 [00:43<00:06, 8191.47it/s] 87%|████████▋ | 347684/400000 [00:43<00:06, 8111.91it/s] 87%|████████▋ | 348507/400000 [00:43<00:06, 8146.33it/s] 87%|████████▋ | 349361/400000 [00:43<00:06, 8257.43it/s] 88%|████████▊ | 350213/400000 [00:44<00:05, 8332.57it/s] 88%|████████▊ | 351076/400000 [00:44<00:05, 8418.59it/s] 88%|████████▊ | 351919/400000 [00:44<00:05, 8311.00it/s] 88%|████████▊ | 352756/400000 [00:44<00:05, 8326.80it/s] 88%|████████▊ | 353614/400000 [00:44<00:05, 8400.40it/s] 89%|████████▊ | 354455/400000 [00:44<00:05, 8312.49it/s] 89%|████████▉ | 355287/400000 [00:44<00:05, 8184.44it/s] 89%|████████▉ | 356107/400000 [00:44<00:05, 7917.43it/s] 89%|████████▉ | 356940/400000 [00:44<00:05, 8034.08it/s] 89%|████████▉ | 357746/400000 [00:44<00:05, 7903.13it/s] 90%|████████▉ | 358539/400000 [00:45<00:05, 7746.32it/s] 90%|████████▉ | 359316/400000 [00:45<00:05, 7666.55it/s] 90%|█████████ | 360144/400000 [00:45<00:05, 7840.34it/s] 90%|█████████ | 360947/400000 [00:45<00:04, 7895.31it/s] 90%|█████████ | 361756/400000 [00:45<00:04, 7952.47it/s] 91%|█████████ | 362598/400000 [00:45<00:04, 8084.83it/s] 91%|█████████ | 363454/400000 [00:45<00:04, 8219.24it/s] 91%|█████████ | 364300/400000 [00:45<00:04, 8289.02it/s] 91%|█████████▏| 365148/400000 [00:45<00:04, 8344.45it/s] 92%|█████████▏| 366016/400000 [00:45<00:04, 8441.34it/s] 92%|█████████▏| 366894/400000 [00:46<00:03, 8538.61it/s] 92%|█████████▏| 367749/400000 [00:46<00:03, 8284.35it/s] 92%|█████████▏| 368580/400000 [00:46<00:03, 7963.61it/s] 92%|█████████▏| 369381/400000 [00:46<00:03, 7918.85it/s] 93%|█████████▎| 370194/400000 [00:46<00:03, 7978.93it/s] 93%|█████████▎| 371008/400000 [00:46<00:03, 8025.37it/s] 93%|█████████▎| 371813/400000 [00:46<00:03, 7911.88it/s] 93%|█████████▎| 372606/400000 [00:46<00:03, 7878.29it/s] 93%|█████████▎| 373422/400000 [00:46<00:03, 7958.52it/s] 94%|█████████▎| 374241/400000 [00:46<00:03, 8026.33it/s] 94%|█████████▍| 375045/400000 [00:47<00:03, 8009.72it/s] 94%|█████████▍| 375847/400000 [00:47<00:03, 7957.95it/s] 94%|█████████▍| 376676/400000 [00:47<00:02, 8053.82it/s] 94%|█████████▍| 377495/400000 [00:47<00:02, 8092.68it/s] 95%|█████████▍| 378358/400000 [00:47<00:02, 8244.15it/s] 95%|█████████▍| 379184/400000 [00:47<00:02, 8139.28it/s] 95%|█████████▍| 379999/400000 [00:47<00:02, 8090.60it/s] 95%|█████████▌| 380809/400000 [00:47<00:02, 8049.27it/s] 95%|█████████▌| 381644/400000 [00:47<00:02, 8135.33it/s] 96%|█████████▌| 382459/400000 [00:48<00:02, 7959.48it/s] 96%|█████████▌| 383257/400000 [00:48<00:02, 7639.17it/s] 96%|█████████▌| 384025/400000 [00:48<00:02, 7547.51it/s] 96%|█████████▌| 384832/400000 [00:48<00:01, 7695.30it/s] 96%|█████████▋| 385630/400000 [00:48<00:01, 7777.18it/s] 97%|█████████▋| 386487/400000 [00:48<00:01, 7997.77it/s] 97%|█████████▋| 387312/400000 [00:48<00:01, 8069.96it/s] 97%|█████████▋| 388131/400000 [00:48<00:01, 8104.54it/s] 97%|█████████▋| 388944/400000 [00:48<00:01, 8086.82it/s] 97%|█████████▋| 389787/400000 [00:48<00:01, 8186.26it/s] 98%|█████████▊| 390633/400000 [00:49<00:01, 8266.19it/s] 98%|█████████▊| 391471/400000 [00:49<00:01, 8298.16it/s] 98%|█████████▊| 392302/400000 [00:49<00:00, 8251.47it/s] 98%|█████████▊| 393130/400000 [00:49<00:00, 8257.59it/s] 98%|█████████▊| 393976/400000 [00:49<00:00, 8316.74it/s] 99%|█████████▊| 394847/400000 [00:49<00:00, 8428.65it/s] 99%|█████████▉| 395717/400000 [00:49<00:00, 8506.76it/s] 99%|█████████▉| 396569/400000 [00:49<00:00, 8340.81it/s] 99%|█████████▉| 397405/400000 [00:49<00:00, 8265.14it/s]100%|█████████▉| 398267/400000 [00:49<00:00, 8367.26it/s]100%|█████████▉| 399148/400000 [00:50<00:00, 8494.77it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 8495.06it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7977.17it/s]>>>model:  <mlmodels.model_tch.textcnn.Model object at 0x7f62a16cad30> <class 'mlmodels.model_tch.textcnn.Model'>
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...
Train Epoch: 1 	 Loss: 0.011296079735217006 	 Accuracy: 54
Train Epoch: 1 	 Loss: 0.01141397789569204 	 Accuracy: 50

  model saves at 50% accuracy 

  #### Inference Need return ypred, ytrue ######################### 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 1, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}
Spliting original file to train/valid set...
Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  {'hypermodel_pars': {}, 'data_pars': {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64, 'train': False}, 'model_pars': {'model_uri': 'model_tch.textcnn.py', 'dim_channel': 100, 'kernel_height': [3, 4, 5], 'dropout_rate': 0.5, 'num_class': 2}, 'compute_pars': {'learning_rate': 0.001, 'epochs': 1, 'checkpointdir': './output/text_cnn_tch/checkpoint/'}, 'out_pars': {'path': './output/text_cnn_tch/model.h5', 'checkpointdir': './output/text_cnn_tch/checkpoint/'}} index out of range: Tried to access index 15962 out of table with 15953 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237 

  


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
RuntimeError: index out of range: Tried to access index 15962 out of table with 15953 rows. at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:237
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
2020-05-13 13:23:54.263862: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 13:23:54.268517: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-13 13:23:54.268727: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b876ece910 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-13 13:23:54.268743: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

>>>model:  <mlmodels.model_keras.textcnn.Model object at 0x7f62ab74e0b8> <class 'mlmodels.model_keras.textcnn.Model'>
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.0993 - accuracy: 0.5370
 2000/25000 [=>............................] - ETA: 10s - loss: 7.4443 - accuracy: 0.5145
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5695 - accuracy: 0.5063 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6130 - accuracy: 0.5035
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6145 - accuracy: 0.5034
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6232 - accuracy: 0.5028
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6228 - accuracy: 0.5029
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6187 - accuracy: 0.5031
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6291 - accuracy: 0.5024
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6068 - accuracy: 0.5039
11000/25000 [============>.................] - ETA: 4s - loss: 7.6192 - accuracy: 0.5031
12000/25000 [=============>................] - ETA: 4s - loss: 7.6385 - accuracy: 0.5018
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6584 - accuracy: 0.5005
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6688 - accuracy: 0.4999
15000/25000 [=================>............] - ETA: 3s - loss: 7.6482 - accuracy: 0.5012
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6465 - accuracy: 0.5013
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6504 - accuracy: 0.5011
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6334 - accuracy: 0.5022
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6360 - accuracy: 0.5020
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6536 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6418 - accuracy: 0.5016
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6471 - accuracy: 0.5013
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6553 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 10s 400us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
>>>model:  <mlmodels.model_tch.transformer_sentence.Model object at 0x7f620e0f8b38> <class 'mlmodels.model_tch.transformer_sentence.Model'>

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
>>>model:  <mlmodels.model_keras.namentity_crm_bilstm.Model object at 0x7f620f2db128> <class 'mlmodels.model_keras.namentity_crm_bilstm.Model'>
Train on 1 samples, validate on 1 samples
Epoch 1/1

1/1 [==============================] - 1s 1s/step - loss: 1.4302 - crf_viterbi_accuracy: 0.0800 - val_loss: 1.3151 - val_crf_viterbi_accuracy: 0.0933

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


  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/a463a24ea257f46bfcbd4006f805952aace8f2b1', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'a463a24ea257f46bfcbd4006f805952aace8f2b1', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/a463a24ea257f46bfcbd4006f805952aace8f2b1

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/a463a24ea257f46bfcbd4006f805952aace8f2b1

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/a463a24ea257f46bfcbd4006f805952aace8f2b1

 ************************************************************************************************************************

  ############Check model ################################ 

  ['model_gluon.fb_prophet', 'model_gluon.gluonts_model', 'model_gluon.gluon_automl', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.charcnn_zhang', 'model_keras.armdn', 'model_keras.textvae', 'model_keras.Autokeras', 'model_keras.textcnn', 'model_keras.01_deepctr', 'model_keras.nbeats', 'model_keras.keras_gan', 'model_keras.02_cnn', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm', 'model_tf.1_lstm', 'model_tf.temporal_fusion_google', 'model_tch.03_nbeats_dataloader', 'model_tch.transformer_sentence', 'model_tch.transformer_classifier', 'model_tch.pplm', 'model_tch.matchzoo_models', 'model_tch.torchhub', 'model_tch.textcnn', 'model_tch.pytorch_vae', 'model_tch.nbeats', 'model_tch.mlp', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn'] 

  Used ['model_gluon.fb_prophet', 'model_gluon.gluonts_model', 'model_gluon.gluon_automl', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.charcnn_zhang', 'model_keras.armdn', 'model_keras.textvae', 'model_keras.Autokeras', 'model_keras.textcnn', 'model_keras.01_deepctr', 'model_keras.nbeats', 'model_keras.keras_gan', 'model_keras.02_cnn', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm', 'model_tf.1_lstm', 'model_tf.temporal_fusion_google', 'model_tch.03_nbeats_dataloader', 'model_tch.transformer_sentence', 'model_tch.transformer_classifier', 'model_tch.pplm', 'model_tch.matchzoo_models', 'model_tch.torchhub', 'model_tch.textcnn', 'model_tch.pytorch_vae', 'model_tch.nbeats', 'model_tch.mlp', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn'] 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//fb_prophet.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//fb_prophet.py", line 160, in <module>
    test(data_path = "model_fb/fbprophet.json", choice="json" )
TypeError: test() got an unexpected keyword argument 'choice'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
Already up to date.
[master 855a6ea] ml_store
 2 files changed, 65 insertions(+), 10236 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   1cbd902..855a6ea  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//gluonts_model.py 
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU

  #### Loading params   ############################################## 

  model_gluon.gluonts_model 
{'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 
INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:numexpr.utils:NumExpr defaulting to 2 threads.
INFO:root:Number of parameters in DeepARTrainingNetwork: 26844
100%|██████████| 10/10 [00:02<00:00,  3.69it/s, avg_epoch_loss=5.23]
INFO:root:Epoch[0] Elapsed time 2.713 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.229596
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.229595947265625 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fcef5fa9a58>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fcef5fa9a58>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['type.txt', 'input_transform.json', 'glutonts_model_pars.pkl', 'prediction_net-0000.params', 'parameters.json', 'prediction_net-network.json', 'version.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]WARNING:root:multiple 5 does not divide base seasonality 1.Falling back to seasonality 1
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 103.45it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 930.1177571614584,
    "abs_error": 342.6952209472656,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.2706813381864537,
    "sMAPE": 0.4851059280171759,
    "MSIS": 90.8272454391405,
    "QuantileLoss[0.5]": 342.6952209472656,
    "Coverage[0.5]": 1.0,
    "RMSE": 30.497832007561758,
    "NRMSE": 0.6420596212118265,
    "ND": 0.6012196858723958,
    "wQuantileLoss[0.5]": 0.6012196858723958,
    "mean_wQuantileLoss": 0.6012196858723958,
    "MAE_Coverage": 0.5
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model 
{'model_name': 'deepfactor', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepFactorTrainingNetwork: 12466
100%|██████████| 10/10 [00:01<00:00,  7.82it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.281 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fcecafc1cc0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fcecafc1cc0>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['type.txt', 'input_transform.json', 'glutonts_model_pars.pkl', 'prediction_net-0000.params', 'parameters.json', 'prediction_net-network.json', 'version.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 147.56it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 2262.8567708333335,
    "abs_error": 552.1011962890625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 3.6581948231980244,
    "sMAPE": 1.8700508550675963,
    "MSIS": 146.3277993985751,
    "QuantileLoss[0.5]": 552.1012096405029,
    "Coverage[0.5]": 0.0,
    "RMSE": 47.5694941200065,
    "NRMSE": 1.0014630341054,
    "ND": 0.9685985899808114,
    "wQuantileLoss[0.5]": 0.9685986134043911,
    "mean_wQuantileLoss": 0.9685986134043911,
    "MAE_Coverage": 0.5
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model 
{'model_name': 'transformer', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in TransformerTrainingNetwork: 33911
100%|██████████| 10/10 [00:01<00:00,  5.07it/s, avg_epoch_loss=5.24]
INFO:root:Epoch[0] Elapsed time 1.975 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.240646
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.240645503997802 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fcec874c240>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fcec874c240>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['type.txt', 'input_transform.json', 'glutonts_model_pars.pkl', 'prediction_net-0000.params', 'parameters.json', 'prediction_net-network.json', 'version.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 157.12it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 395.6704508463542,
    "abs_error": 213.33938598632812,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.4135760665709896,
    "sMAPE": 0.3355350853911448,
    "MSIS": 56.543047515830175,
    "QuantileLoss[0.5]": 213.33940505981445,
    "Coverage[0.5]": 0.8333333333333334,
    "RMSE": 19.891466784688205,
    "NRMSE": 0.4187677217829096,
    "ND": 0.37427962453741775,
    "wQuantileLoss[0.5]": 0.3742796579996745,
    "mean_wQuantileLoss": 0.3742796579996745,
    "MAE_Coverage": 0.33333333333333337
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model 
{'model_name': 'wavenet', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:gluonts.model.wavenet._estimator:Using dilation depth 10 and receptive field length 1024
INFO:root:using training windows of length = 12
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in WaveNet: 97636
 30%|███       | 3/10 [00:13<00:31,  4.47s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:25<00:17,  4.30s/it, avg_epoch_loss=6.91] 90%|█████████ | 9/10 [00:36<00:04,  4.17s/it, avg_epoch_loss=6.89]100%|██████████| 10/10 [00:40<00:00,  4.05s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 40.546 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.870230
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.870230102539063 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fcecaec5f98>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fcecaec5f98>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['type.txt', 'input_transform.json', 'glutonts_model_pars.pkl', 'prediction_net-0000.params', 'parameters.json', 'prediction_net-network.json', 'version.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU
INFO:gluonts.model.wavenet._estimator:Using dilation depth 10 and receptive field length 1024

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 120.23it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 53472.526041666664,
    "abs_error": 2729.78466796875,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 18.08741623443291,
    "sMAPE": 1.413887981433052,
    "MSIS": 723.496675259933,
    "QuantileLoss[0.5]": 2729.7847442626953,
    "Coverage[0.5]": 1.0,
    "RMSE": 231.24127235782686,
    "NRMSE": 4.868237312796355,
    "ND": 4.789095908717106,
    "wQuantileLoss[0.5]": 4.789096042566132,
    "mean_wQuantileLoss": 4.789096042566132,
    "MAE_Coverage": 0.5
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model 
{'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in SimpleFeedForwardTrainingNetwork: 20323
100%|██████████| 10/10 [00:00<00:00, 57.45it/s, avg_epoch_loss=5.21]
INFO:root:Epoch[0] Elapsed time 0.175 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.214407
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.2144066333770756 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fcec84bcf28>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fcec84bcf28>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['type.txt', 'input_transform.json', 'glutonts_model_pars.pkl', 'prediction_net-0000.params', 'parameters.json', 'prediction_net-network.json', 'version.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 156.28it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 324.5646565755208,
    "abs_error": 179.00177001953125,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.1860567461728304,
    "sMAPE": 0.31131379828927674,
    "MSIS": 47.44226984691322,
    "QuantileLoss[0.5]": 179.00177001953125,
    "Coverage[0.5]": 0.5833333333333334,
    "RMSE": 18.015678077039478,
    "NRMSE": 0.3792774332008311,
    "ND": 0.3140381930167215,
    "wQuantileLoss[0.5]": 0.3140381930167215,
    "mean_wQuantileLoss": 0.3140381930167215,
    "MAE_Coverage": 0.08333333333333337
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model 
{'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in GaussianProcessTrainingNetwork: 14
100%|██████████| 10/10 [00:01<00:00,  7.66it/s, avg_epoch_loss=123]
INFO:root:Epoch[0] Elapsed time 1.307 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=122.866774
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 122.86677375545166 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fcec84cd208>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fcec84cd208>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['type.txt', 'input_transform.json', 'glutonts_model_pars.pkl', 'prediction_net-0000.params', 'parameters.json', 'prediction_net-network.json', 'version.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 162.66it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 2213.317910911945,
    "abs_error": 549.0071371123157,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 3.6376937423470475,
    "sMAPE": 1.8441387313887985,
    "MSIS": 145.5077496938819,
    "QuantileLoss[0.5]": 549.0071371123157,
    "Coverage[0.5]": 0.0,
    "RMSE": 47.04591279709583,
    "NRMSE": 0.9904402694125438,
    "ND": 0.9631704159865188,
    "wQuantileLoss[0.5]": 0.9631704159865188,
    "mean_wQuantileLoss": 0.9631704159865188,
    "MAE_Coverage": 0.5
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model 
{'model_name': 'deepstate', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepStateTrainingNetwork: 28054
 10%|█         | 1/10 [02:00<18:01, 120.14s/it, avg_epoch_loss=0.703] 20%|██        | 2/10 [04:49<17:58, 134.78s/it, avg_epoch_loss=0.686] 30%|███       | 3/10 [07:53<17:27, 149.68s/it, avg_epoch_loss=0.669] 40%|████      | 4/10 [11:18<16:37, 166.22s/it, avg_epoch_loss=0.652] 50%|█████     | 5/10 [14:59<15:13, 182.64s/it, avg_epoch_loss=0.634] 60%|██████    | 6/10 [18:04<12:13, 183.46s/it, avg_epoch_loss=0.617] 70%|███████   | 7/10 [21:20<09:21, 187.08s/it, avg_epoch_loss=0.599] 80%|████████  | 8/10 [24:58<06:32, 196.33s/it, avg_epoch_loss=0.581] 90%|█████████ | 9/10 [28:25<03:19, 199.64s/it, avg_epoch_loss=0.563]100%|██████████| 10/10 [31:58<00:00, 203.55s/it, avg_epoch_loss=0.546]100%|██████████| 10/10 [31:58<00:00, 191.82s/it, avg_epoch_loss=0.546]
INFO:root:Epoch[0] Elapsed time 1918.181 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.546187
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.5461866021156311 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fcec8488128>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fcec8488128>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['type.txt', 'input_transform.json', 'glutonts_model_pars.pkl', 'prediction_net-0000.params', 'parameters.json', 'prediction_net-network.json', 'version.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 15.73it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 144.20114135742188,
    "abs_error": 105.91581726074219,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 0.7017928905105525,
    "sMAPE": 0.1842843903726714,
    "MSIS": 28.071716429253865,
    "QuantileLoss[0.5]": 105.91581344604492,
    "Coverage[0.5]": 0.3333333333333333,
    "RMSE": 12.00837796529664,
    "NRMSE": 0.2528079571641398,
    "ND": 0.18581722326445999,
    "wQuantileLoss[0.5]": 0.18581721657200864,
    "mean_wQuantileLoss": 0.18581721657200864,
    "MAE_Coverage": 0.16666666666666669
}

  #### Plot   ####################################################### 


   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
Already up to date.
[master 4beef2d] ml_store
 1 file changed, 502 insertions(+)
To github.com:arita37/mlmodels_store.git
   855a6ea..4beef2d  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//gluon_automl.py 

  #### Loading params   ############################################## 

  #### Model params   ################################################ 

  #### Loading dataset   ############################################# 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/optimizer/optimizer.py:167: UserWarning: WARNING: New optimizer gluonnlp.optimizer.lamb.LAMB is overriding existing optimizer mxnet.optimizer.optimizer.LAMB
  Optimizer.opt_registry[name].__name__))
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv | Columns = 15 / 15 | Rows = 39073 -> 39073

  #### Model init, fit   ############################################# 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv | Columns = 15 / 15 | Rows = 39073 -> 39073
Warning: `hyperparameter_tune=True` is currently experimental and may cause the process to hang. Setting `auto_stack=True` instead is recommended to achieve maximum quality models.
Beginning AutoGluon training ... Time limit = 120s
AutoGluon will save models to dataset/
Train Data Rows:    39073
Train Data Columns: 15
Preprocessing data ...
Here are the first 10 unique label values in your data:  [' Tech-support' ' Transport-moving' ' Other-service' ' ?'
 ' Handlers-cleaners' ' Sales' ' Craft-repair' ' Adm-clerical'
 ' Exec-managerial' ' Prof-specialty']
AutoGluon infers your prediction problem is: multiclass  (because dtype of label-column == object)
If this is wrong, please specify `problem_type` argument in fit() instead (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])

Feature Generator processed 39073 data points with 14 features
Original Features:
	int features: 6
	object features: 8
Generated Features:
	int features: 0
All Features:
	int features: 6
	object features: 8
	Data preprocessing and feature engineering runtime = 0.25s ...
AutoGluon will gauge predictive performance using evaluation metric: accuracy
To change this, specify the eval_metric argument of fit()
AutoGluon will early stop models using evaluation metric: accuracy
Saving dataset/learner.pkl
Beginning hyperparameter tuning for Gradient Boosting Model...
Hyperparameter search space for Gradient Boosting Model: 
num_leaves:   Int: lower=26, upper=66
learning_rate:   Real: lower=0.005, upper=0.2
feature_fraction:   Real: lower=0.75, upper=1.0
min_data_in_leaf:   Int: lower=2, upper=30
Starting Experiments
Num of Finished Tasks is 0
Num of Pending Tasks is 5
  0%|          | 0/5 [00:00<?, ?it/s]Saving dataset/models/LightGBMClassifier/trial_0_model.pkl
Finished Task with config: {'feature_fraction': 1.0, 'learning_rate': 0.1, 'min_data_in_leaf': 20, 'num_leaves': 36} and reward: 0.3908
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00learning_rateq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K$u.' and reward: 0.3908
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00learning_rateq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K$u.' and reward: 0.3908
 40%|████      | 2/5 [00:20<00:31, 10.50s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.8597602438038345, 'learning_rate': 0.1258227684069875, 'min_data_in_leaf': 28, 'num_leaves': 36} and reward: 0.3936
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\x83'\xea1;\x08X\r\x00\x00\x00learning_rateq\x02G?\xc0\x1a\xf5\xe1\xb39\x89X\x10\x00\x00\x00min_data_in_leafq\x03K\x1cX\n\x00\x00\x00num_leavesq\x04K$u." and reward: 0.3936
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\x83'\xea1;\x08X\r\x00\x00\x00learning_rateq\x02G?\xc0\x1a\xf5\xe1\xb39\x89X\x10\x00\x00\x00min_data_in_leafq\x03K\x1cX\n\x00\x00\x00num_leavesq\x04K$u." and reward: 0.3936
 60%|██████    | 3/5 [00:41<00:26, 13.49s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9790831923744072, 'learning_rate': 0.01687757445055765, 'min_data_in_leaf': 16, 'num_leaves': 56} and reward: 0.3912
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xefT\xa6Fi\xf6\xa0X\r\x00\x00\x00learning_rateq\x02G?\x91HZ\xd94,OX\x10\x00\x00\x00min_data_in_leafq\x03K\x10X\n\x00\x00\x00num_leavesq\x04K8u.' and reward: 0.3912
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xefT\xa6Fi\xf6\xa0X\r\x00\x00\x00learning_rateq\x02G?\x91HZ\xd94,OX\x10\x00\x00\x00min_data_in_leafq\x03K\x10X\n\x00\x00\x00num_leavesq\x04K8u.' and reward: 0.3912
 80%|████████  | 4/5 [01:11<00:18, 18.41s/it] 80%|████████  | 4/5 [01:11<00:17, 17.84s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.8189225029572743, 'learning_rate': 0.07583329921835996, 'min_data_in_leaf': 6, 'num_leaves': 37} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea4\x9c\xf7\x05\x1e]X\r\x00\x00\x00learning_rateq\x02G?\xb3i\xcf\xa4\x174\x14X\x10\x00\x00\x00min_data_in_leafq\x03K\x06X\n\x00\x00\x00num_leavesq\x04K%u.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea4\x9c\xf7\x05\x1e]X\r\x00\x00\x00learning_rateq\x02G?\xb3i\xcf\xa4\x174\x14X\x10\x00\x00\x00min_data_in_leafq\x03K\x06X\n\x00\x00\x00num_leavesq\x04K%u.' and reward: 0.3894
Time for Gradient Boosting hyperparameter optimization: 93.7950451374054
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.8597602438038345, 'learning_rate': 0.1258227684069875, 'min_data_in_leaf': 28, 'num_leaves': 36}
Saving dataset/models/trainer.pkl
Beginning hyperparameter tuning for Neural Network...
Hyperparameter search space for Neural Network: 
network_type:   Categorical['widedeep', 'feedforward']
layers:   Categorical[[100], [1000], [200, 100], [300, 200, 100]]
activation:   Categorical['relu', 'softrelu', 'tanh']
embedding_size_factor:   Real: lower=0.5, upper=1.5
use_batchnorm:   Categorical[True, False]
dropout_prob:   Real: lower=0.0, upper=0.5
learning_rate:   Real: lower=0.0001, upper=0.01
weight_decay:   Real: lower=1e-12, upper=0.1
AutoGluon Neural Network infers features are of the following types:
{
    "continuous": [
        "age",
        "education-num",
        "hours-per-week"
    ],
    "skewed": [
        "fnlwgt",
        "capital-gain",
        "capital-loss"
    ],
    "onehot": [
        "sex",
        "class"
    ],
    "embed": [
        "workclass",
        "education",
        "marital-status",
        "relationship",
        "race",
        "native-country"
    ],
    "language": []
}


Saving dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Starting Experiments
Num of Finished Tasks is 0
Num of Pending Tasks is 5
  0%|          | 0/5 [00:00<?, ?it/s]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
 40%|████      | 2/5 [00:53<01:20, 26.75s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.1616333837631133, 'embedding_size_factor': 0.7420189431256679, 'layers.choice': 3, 'learning_rate': 0.000298615804090219, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 4.8233148945231356e-05} and reward: 0.3636
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xc4\xb0g\x18\x9a)kX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe7\xbe\x9e\x82\xb7\x97^X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?3\x91\xf1\x1c\xd5\xbalX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\tI\xbe`\xe0S%u.' and reward: 0.3636
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xc4\xb0g\x18\x9a)kX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe7\xbe\x9e\x82\xb7\x97^X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?3\x91\xf1\x1c\xd5\xbalX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\tI\xbe`\xe0S%u.' and reward: 0.3636
 60%|██████    | 3/5 [01:48<01:10, 35.26s/it] 60%|██████    | 3/5 [01:48<01:12, 36.21s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.2884814489746941, 'embedding_size_factor': 1.1578694329587473, 'layers.choice': 1, 'learning_rate': 0.0008918790013845011, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.5140694276002881e-09} and reward: 0.3848
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd2vz\xe56V\x14X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2\x86\xa2\x199\x88\xbeX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?M9\x9f\x92K\xb6yX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x1a\x02\xf2\xa0\xa3\x14\xebu.' and reward: 0.3848
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd2vz\xe56V\x14X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2\x86\xa2\x199\x88\xbeX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?M9\x9f\x92K\xb6yX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\x1a\x02\xf2\xa0\xa3\x14\xebu.' and reward: 0.3848
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 179.50518774986267
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -157.65s of remaining time.
Ensemble size: 82
Ensemble weights: 
[0.17073171 0.15853659 0.03658537 0.13414634 0.2804878  0.15853659
 0.06097561]
	0.4008	 = Validation accuracy score
	1.64s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 279.35s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7fa8e85f1ba8>

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
Already up to date.
[master 53bb3f3] ml_store
 1 file changed, 304 insertions(+)
To github.com:arita37/mlmodels_store.git
   4beef2d..53bb3f3  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//namentity_crm_bilstm_dataloader.py 

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.namentity_crm_bilstm_dataloader' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm_dataloader.py'> 

  #### Loading params   ############################################## 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//namentity_crm_bilstm_dataloader.py", line 306, in <module>
    test_module(model_uri=MODEL_URI, param_pars=param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 257, in test_module
    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm_dataloader.py", line 197, in get_params
    cf = json.load(open(data_path, mode="r"))
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader.json'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Already up to date.
[master 5a8cda4] ml_store
 1 file changed, 47 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   53bb3f3..5a8cda4  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py 
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset

  #### Loading params   ############################################## 

  #### Loading daaset   ############################################# 
Loading data...
Data loaded from /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ag_news_csv/train.csv
Data loaded from /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ag_news_csv/test.csv

  #### Model init, fit   ############################################# 
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

2020-05-22 16:57:56.001702: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-22 16:57:56.015430: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-22 16:57:56.015647: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56022beb5360 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 16:57:56.015663: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

CharCNNZhang model built: 
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
sent_input (InputLayer)      (None, 1014)              0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 1014, 128)         8960      
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 1008, 256)         229632    
_________________________________________________________________
thresholded_re_lu_1 (Thresho (None, 1008, 256)         0         
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 336, 256)          0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 330, 256)          459008    
_________________________________________________________________
thresholded_re_lu_2 (Thresho (None, 330, 256)          0         
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 110, 256)          0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 108, 256)          196864    
_________________________________________________________________
thresholded_re_lu_3 (Thresho (None, 108, 256)          0         
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 106, 256)          196864    
_________________________________________________________________
thresholded_re_lu_4 (Thresho (None, 106, 256)          0         
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 104, 256)          196864    
_________________________________________________________________
thresholded_re_lu_5 (Thresho (None, 104, 256)          0         
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 102, 256)          196864    
_________________________________________________________________
thresholded_re_lu_6 (Thresho (None, 102, 256)          0         
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 34, 256)           0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8704)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              8913920   
_________________________________________________________________
thresholded_re_lu_7 (Thresho (None, 1024)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
thresholded_re_lu_8 (Thresho (None, 1024)              0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 4100      
=================================================================
Total params: 11,452,676
Trainable params: 11,452,676
Non-trainable params: 0
_________________________________________________________________
Loading data...
Data loaded from /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ag_news_csv/train.csv
Data loaded from /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ag_news_csv/test.csv
Train on 354 samples, validate on 236 samples
Epoch 1/1

128/354 [=========>....................] - ETA: 6s - loss: 1.3901
256/354 [====================>.........] - ETA: 2s - loss: 1.3033
354/354 [==============================] - 11s 31ms/step - loss: 1.1803 - val_loss: 1.8767

  #### Predict   ##################################################### 
Data loaded from /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ag_news_csv/test.csv

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/callbacks/callbacks.py:846: RuntimeWarning: Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: val_loss,loss
  (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
{'path': 'ztest/ml_keras/charcnn_zhang/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
{'path': 'ztest/ml_keras/charcnn_zhang/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py", line 284, in <module>
    test(pars_choice="json", data_path= f"{root_path}/model_keras/charcnn_zhang.json")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py", line 268, in test
    model2 = load(out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py", line 118, in load
    model = load_keras(load_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 602, in load_keras
    model.model = load_model(path_file)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/keras/saving/save.py", line 146, in load_model
    loader_impl.parse_saved_model(filepath)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/saved_model/loader_impl.py", line 83, in parse_saved_model
    constants.SAVED_MODEL_FILENAME_PB))
OSError: SavedModel file does not exist at: ztest/ml_keras/charcnn_zhang//model.h5/{saved_model.pbtxt|saved_model.pb}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Already up to date.
[master 1d420a4] ml_store
 1 file changed, 151 insertions(+)
To github.com:arita37/mlmodels_store.git
   5a8cda4..1d420a4  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//armdn.py 

  #### Loading params   ############################################## 

  #### Model init   ################################################## 
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
LSTM_1 (LSTM)                (None, 12, 300)           362400    
_________________________________________________________________
LSTM_2 (LSTM)                (None, 12, 200)           400800    
_________________________________________________________________
LSTM_3 (LSTM)                (None, 12, 24)            21600     
_________________________________________________________________
LSTM_4 (LSTM)                (None, 12)                1776      
_________________________________________________________________
dense_1 (Dense)              (None, 10)                130       
_________________________________________________________________
mdn_1 (MDN)                  (None, 75)                825       
=================================================================
Total params: 787,531
Trainable params: 787,531
Non-trainable params: 0
_________________________________________________________________

  ### Model Fit ###################################################### 

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

13/13 [==============================] - 2s 166ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 6ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 8/10

13/13 [==============================] - 0s 6ms/step - loss: nan
Epoch 9/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 10/10

13/13 [==============================] - 0s 5ms/step - loss: nan

  fitted metrics {'loss': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]} 

  #### Predict   ##################################################### 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mdn/__init__.py:209: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.

[[nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
  nan nan nan]]
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//armdn.py", line 380, in <module>
    test(pars_choice="json", data_path= "model_keras/armdn.json")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//armdn.py", line 354, in test
    y_pred, y_test = predict(model=model, model_pars=model_pars, data_pars=data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//armdn.py", line 170, in predict
    model.model_pars["n_mixes"], temp=1.0)
  File "<__array_function__ internals>", line 6, in apply_along_axis
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/shape_base.py", line 379, in apply_along_axis
    res = asanyarray(func1d(inarr_view[ind0], *args, **kwargs))
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mdn/__init__.py", line 237, in sample_from_output
    cov_matrix = np.identity(output_dim) * sig_vector
ValueError: operands could not be broadcast together with shapes (12,12) (0,) 

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Already up to date.
[master 3d2b41b] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   1d420a4..3d2b41b  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textvae.py 

  #### Loading params   ############################################## 

  #### Path params   ################################################### 

  #### Model params   ################################################# 

  #### Loading dataset   ############################################# 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textvae.py", line 356, in <module>
    test(pars_choice="test01")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textvae.py", line 327, in test
    xtuple = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textvae.py", line 269, in get_dataset
    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
    file = builtins.open(filename, mode, buffering)
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Already up to date.
[master 387c71a] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   3d2b41b..387c71a  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//Autokeras.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//Autokeras.py", line 12, in <module>
    import autokeras as ak
ModuleNotFoundError: No module named 'autokeras'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Already up to date.
[master f47e06f] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   387c71a..f47e06f  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2154496/17464789 [==>...........................] - ETA: 0s
 9625600/17464789 [===============>..............] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...

  #### Model init, fit   ############################################# 
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-22 16:59:14.515140: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-22 16:59:14.519769: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-22 16:59:14.519950: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557bbbd7bec0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 16:59:14.519985: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

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
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.1913 - accuracy: 0.5310
 2000/25000 [=>............................] - ETA: 9s - loss: 7.4136 - accuracy: 0.5165 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5031 - accuracy: 0.5107
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.4788 - accuracy: 0.5123
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5010 - accuracy: 0.5108
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.5107 - accuracy: 0.5102
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5264 - accuracy: 0.5091
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6015 - accuracy: 0.5042
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6240 - accuracy: 0.5028
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6252 - accuracy: 0.5027
11000/25000 [============>.................] - ETA: 3s - loss: 7.6262 - accuracy: 0.5026
12000/25000 [=============>................] - ETA: 3s - loss: 7.5938 - accuracy: 0.5048
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6124 - accuracy: 0.5035
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6097 - accuracy: 0.5037
15000/25000 [=================>............] - ETA: 2s - loss: 7.6176 - accuracy: 0.5032
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6120 - accuracy: 0.5036
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6188 - accuracy: 0.5031
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6189 - accuracy: 0.5031
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6327 - accuracy: 0.5022
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6452 - accuracy: 0.5014
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6528 - accuracy: 0.5009
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6847 - accuracy: 0.4988
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6713 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 8s 310us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### save the trained model  ####################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/textcnn/model.h5'}

  #### Predict   ##################################################### 
Loading data...

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/keras/initializers.py:119: calling RandomUniform.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/textcnn/model.h5'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/textcnn/model.h5'}
(<mlmodels.util.Model_empty object at 0x7ff68d489908>, None)

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.textcnn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            (None, 40)           0                                            
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 40, 50)       250         input_2[0][0]                    
__________________________________________________________________________________________________
conv1d_4 (Conv1D)               (None, 38, 128)      19328       embedding_2[0][0]                
__________________________________________________________________________________________________
conv1d_5 (Conv1D)               (None, 37, 128)      25728       embedding_2[0][0]                
__________________________________________________________________________________________________
conv1d_6 (Conv1D)               (None, 36, 128)      32128       embedding_2[0][0]                
__________________________________________________________________________________________________
global_max_pooling1d_4 (GlobalM (None, 128)          0           conv1d_4[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_5 (GlobalM (None, 128)          0           conv1d_5[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_6 (GlobalM (None, 128)          0           conv1d_6[0][0]                   
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 384)          0           global_max_pooling1d_4[0][0]     
                                                                 global_max_pooling1d_5[0][0]     
                                                                 global_max_pooling1d_6[0][0]     
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            385         concatenate_2[0][0]              
==================================================================================================
Total params: 77,819
Trainable params: 77,819
Non-trainable params: 0
__________________________________________________________________________________________________

  <mlmodels.model_keras.textcnn.Model object at 0x7ff69af33390> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.9120 - accuracy: 0.4840
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6973 - accuracy: 0.4980 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7740 - accuracy: 0.4930
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7096 - accuracy: 0.4972
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6896 - accuracy: 0.4985
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7170 - accuracy: 0.4967
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7893 - accuracy: 0.4920
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7467 - accuracy: 0.4948
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7264 - accuracy: 0.4961
11000/25000 [============>.................] - ETA: 3s - loss: 7.7210 - accuracy: 0.4965
12000/25000 [=============>................] - ETA: 3s - loss: 7.7254 - accuracy: 0.4962
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7339 - accuracy: 0.4956
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7203 - accuracy: 0.4965
15000/25000 [=================>............] - ETA: 2s - loss: 7.7024 - accuracy: 0.4977
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6982 - accuracy: 0.4979
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7117 - accuracy: 0.4971
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7033 - accuracy: 0.4976
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6884 - accuracy: 0.4986
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6820 - accuracy: 0.4990
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6907 - accuracy: 0.4984
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6659 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6526 - accuracy: 0.5009
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6577 - accuracy: 0.5006
25000/25000 [==============================] - 7s 298us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Predict   #################################################### 
Loading data...
(array([[1.],
       [1.],
       [1.],
       ...,
       [1.],
       [1.],
       [1.]], dtype=float32), None)

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

  ############ Model preparation   ################################## 

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.textcnn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 
Model: "model_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_3 (InputLayer)            (None, 40)           0                                            
__________________________________________________________________________________________________
embedding_3 (Embedding)         (None, 40, 50)       250         input_3[0][0]                    
__________________________________________________________________________________________________
conv1d_7 (Conv1D)               (None, 38, 128)      19328       embedding_3[0][0]                
__________________________________________________________________________________________________
conv1d_8 (Conv1D)               (None, 37, 128)      25728       embedding_3[0][0]                
__________________________________________________________________________________________________
conv1d_9 (Conv1D)               (None, 36, 128)      32128       embedding_3[0][0]                
__________________________________________________________________________________________________
global_max_pooling1d_7 (GlobalM (None, 128)          0           conv1d_7[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_8 (GlobalM (None, 128)          0           conv1d_8[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_9 (GlobalM (None, 128)          0           conv1d_9[0][0]                   
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 384)          0           global_max_pooling1d_7[0][0]     
                                                                 global_max_pooling1d_8[0][0]     
                                                                 global_max_pooling1d_9[0][0]     
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 1)            385         concatenate_3[0][0]              
==================================================================================================
Total params: 77,819
Trainable params: 77,819
Non-trainable params: 0
__________________________________________________________________________________________________

  ############ Model fit   ########################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.7740 - accuracy: 0.4930
 2000/25000 [=>............................] - ETA: 8s - loss: 7.6130 - accuracy: 0.5035 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6717 - accuracy: 0.4997
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6590 - accuracy: 0.5005
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6237 - accuracy: 0.5028
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6104 - accuracy: 0.5037
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5681 - accuracy: 0.5064
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6245 - accuracy: 0.5027
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6649 - accuracy: 0.5001
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
11000/25000 [============>.................] - ETA: 3s - loss: 7.6262 - accuracy: 0.5026
12000/25000 [=============>................] - ETA: 3s - loss: 7.6270 - accuracy: 0.5026
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6124 - accuracy: 0.5035
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6184 - accuracy: 0.5031
15000/25000 [=================>............] - ETA: 2s - loss: 7.6390 - accuracy: 0.5018
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6446 - accuracy: 0.5014
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6477 - accuracy: 0.5012
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6445 - accuracy: 0.5014
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6384 - accuracy: 0.5018
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6413 - accuracy: 0.5016
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6608 - accuracy: 0.5004
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6736 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 7s 288us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
fit success None

  ############ Prediction############################################ 
Loading data...
(array([[1.],
       [1.],
       [1.],
       ...,
       [1.],
       [1.],
       [1.]], dtype=float32), None)

  ############ Save/ Load ############################################ 

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Already up to date.
[master eeb5063] ml_store
 1 file changed, 317 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   f47e06f..eeb5063  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//01_deepctr.py 

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'AFM', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'AFM', 'sparse_feature_num': 3, 'dense_feature_num': 0} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_AFM.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/keras/initializers.py:143: calling RandomNormal.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/layers/sequence.py:159: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/layers/utils.py:199: calling softmax (from tensorflow.python.ops.nn_ops) with dim is deprecated and will be removed in a future version.
Instructions for updating:
dim is deprecated, use axis instead
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/layers/utils.py:163: calling reduce_sum_v1 (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/layers/utils.py:193: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Deprecated in favor of operator or tf.math.divide.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/layers/utils.py:180: calling reduce_max_v1 (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_4 (Seque (None, 1, 1)         0           weighted_sequence_layer_1[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_5 (Seque (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_6 (Seque (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_7 (Seque (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
no_mask (NoMask)                (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_4[0][0]   
                                                                 sequence_pooling_layer_5[0][0]   
                                                                 sequence_pooling_layer_6[0][0]   
                                                                 sequence_pooling_layer_7[0][0]   
__________________________________________________________________________________________________
weighted_sequence_layer (Weight (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-22 17:00:27.500152: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-22 17:00:27.505208: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095190000 Hz
2020-05-22 17:00:27.505374: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56239adddef0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 17:00:27.505388: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_1 (Seque (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_2 (Seque (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_3 (Seque (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
linear (Linear)                 (None, 1, 1)         0           concatenate[0][0]                
__________________________________________________________________________________________________
afm_layer (AFMLayer)            (None, 1)            52          sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sequence_pooling_layer[0][0]     
                                                                 sequence_pooling_layer_1[0][0]   
                                                                 sequence_pooling_layer_2[0][0]   
                                                                 sequence_pooling_layer_3[0][0]   
__________________________________________________________________________________________________
no_mask_1 (NoMask)              (None, 1, 1)         0           linear[0][0]                     
__________________________________________________________________________________________________
add (Add)                       (None, 1)            0           afm_layer[0][0]                  
__________________________________________________________________________________________________
add_1 (Add)                     (None, 1, 1)         0           no_mask_1[0][0]                  
                                                                 add[0][0]                        
__________________________________________________________________________________________________
prediction_layer (PredictionLay (None, 1)            1           add_1[0][0]                      
==================================================================================================
Total params: 218
Trainable params: 218
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2490 - binary_crossentropy: 0.6902 - val_loss: 0.2517 - val_binary_crossentropy: 0.7230

  #### metrics   #################################################### 
{'MSE': 0.25004737756372947}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_4 (Seque (None, 1, 1)         0           weighted_sequence_layer_1[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_5 (Seque (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_6 (Seque (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_7 (Seque (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
no_mask (NoMask)                (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_4[0][0]   
                                                                 sequence_pooling_layer_5[0][0]   
                                                                 sequence_pooling_layer_6[0][0]   
                                                                 sequence_pooling_layer_7[0][0]   
__________________________________________________________________________________________________
weighted_sequence_layer (Weight (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_1 (Seque (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_2 (Seque (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_3 (Seque (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
linear (Linear)                 (None, 1, 1)         0           concatenate[0][0]                
__________________________________________________________________________________________________
afm_layer (AFMLayer)            (None, 1)            52          sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sequence_pooling_layer[0][0]     
                                                                 sequence_pooling_layer_1[0][0]   
                                                                 sequence_pooling_layer_2[0][0]   
                                                                 sequence_pooling_layer_3[0][0]   
__________________________________________________________________________________________________
no_mask_1 (NoMask)              (None, 1, 1)         0           linear[0][0]                     
__________________________________________________________________________________________________
add (Add)                       (None, 1)            0           afm_layer[0][0]                  
__________________________________________________________________________________________________
add_1 (Add)                     (None, 1, 1)         0           no_mask_1[0][0]                  
                                                                 add[0][0]                        
__________________________________________________________________________________________________
prediction_layer (PredictionLay (None, 1)            1           add_1[0][0]                      
==================================================================================================
Total params: 218
Trainable params: 218
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'AutoInt', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'AutoInt', 'sparse_feature_num': 1, 'dense_feature_num': 1} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_AutoInt.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/layers/interaction.py:565: The name tf.keras.initializers.TruncatedNormal is deprecated. Please use tf.compat.v1.keras.initializers.TruncatedNormal instead.

WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/keras/initializers.py:94: calling TruncatedNormal.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_12 (Sequ (None, 1, 4)         0           weighted_sequence_layer_3[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_13 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_14 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_15 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
weighted_sequence_layer_4 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_16 (Sequ (None, 1, 1)         0           weighted_sequence_layer_4[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_17 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_18 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_19 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 5, 4)         0           no_mask_5[0][0]                  
                                                                 no_mask_5[1][0]                  
                                                                 no_mask_5[2][0]                  
                                                                 no_mask_5[3][0]                  
                                                                 no_mask_5[4][0]                  
__________________________________________________________________________________________________
no_mask_2 (NoMask)              (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_16[0][0]  
                                                                 sequence_pooling_layer_17[0][0]  
                                                                 sequence_pooling_layer_18[0][0]  
                                                                 sequence_pooling_layer_19[0][0]  
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
interacting_layer (InteractingL (None, 5, 16)        256         concatenate_2[0][0]              
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 1, 5)         0           no_mask_2[0][0]                  
                                                                 no_mask_2[1][0]                  
                                                                 no_mask_2[2][0]                  
                                                                 no_mask_2[3][0]                  
                                                                 no_mask_2[4][0]                  
__________________________________________________________________________________________________
no_mask_3 (NoMask)              (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
flatten (Flatten)               (None, 80)           0           interacting_layer[0][0]          
__________________________________________________________________________________________________
linear_1 (Linear)               (None, 1)            1           concatenate_1[0][0]              
                                                                 no_mask_3[0][0]                  
__________________________________________________________________________________________________
dense (Dense)                   (None, 1)            80          flatten[0][0]                    
__________________________________________________________________________________________________
no_mask_4 (NoMask)              (None, 1)            0           linear_1[0][0]                   
__________________________________________________________________________________________________
add_4 (Add)                     (None, 1)            0           dense[0][0]                      
                                                                 no_mask_4[0][0]                  
__________________________________________________________________________________________________
prediction_layer_1 (PredictionL (None, 1)            1           add_4[0][0]                      
==================================================================================================
Total params: 453
Trainable params: 453
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2966 - binary_crossentropy: 1.7072500/500 [==============================] - 1s 2ms/sample - loss: 0.3069 - binary_crossentropy: 1.9329 - val_loss: 0.2798 - val_binary_crossentropy: 1.4020

  #### metrics   #################################################### 
{'MSE': 0.2928824455354793}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/init_ops.py:97: calling GlorotUniform.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/init_ops.py:97: calling Zeros.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_12 (Sequ (None, 1, 4)         0           weighted_sequence_layer_3[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_13 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_14 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_15 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
weighted_sequence_layer_4 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_16 (Sequ (None, 1, 1)         0           weighted_sequence_layer_4[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_17 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_18 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_19 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 5, 4)         0           no_mask_5[0][0]                  
                                                                 no_mask_5[1][0]                  
                                                                 no_mask_5[2][0]                  
                                                                 no_mask_5[3][0]                  
                                                                 no_mask_5[4][0]                  
__________________________________________________________________________________________________
no_mask_2 (NoMask)              (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_16[0][0]  
                                                                 sequence_pooling_layer_17[0][0]  
                                                                 sequence_pooling_layer_18[0][0]  
                                                                 sequence_pooling_layer_19[0][0]  
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
interacting_layer (InteractingL (None, 5, 16)        256         concatenate_2[0][0]              
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 1, 5)         0           no_mask_2[0][0]                  
                                                                 no_mask_2[1][0]                  
                                                                 no_mask_2[2][0]                  
                                                                 no_mask_2[3][0]                  
                                                                 no_mask_2[4][0]                  
__________________________________________________________________________________________________
no_mask_3 (NoMask)              (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
flatten (Flatten)               (None, 80)           0           interacting_layer[0][0]          
__________________________________________________________________________________________________
linear_1 (Linear)               (None, 1)            1           concatenate_1[0][0]              
                                                                 no_mask_3[0][0]                  
__________________________________________________________________________________________________
dense (Dense)                   (None, 1)            80          flatten[0][0]                    
__________________________________________________________________________________________________
no_mask_4 (NoMask)              (None, 1)            0           linear_1[0][0]                   
__________________________________________________________________________________________________
add_4 (Add)                     (None, 1)            0           dense[0][0]                      
                                                                 no_mask_4[0][0]                  
__________________________________________________________________________________________________
prediction_layer_1 (PredictionL (None, 1)            1           add_4[0][0]                      
==================================================================================================
Total params: 453
Trainable params: 453
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'CCPM', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'CCPM', 'sparse_feature_num': 3, 'dense_feature_num': 0} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_CCPM.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_24 (Sequ (None, 1, 4)         0           weighted_sequence_layer_6[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_25 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_26 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_27 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_11 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sequence_pooling_layer_24[0][0]  
                                                                 sequence_pooling_layer_25[0][0]  
                                                                 sequence_pooling_layer_26[0][0]  
                                                                 sequence_pooling_layer_27[0][0]  
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 7, 4)         0           no_mask_11[0][0]                 
                                                                 no_mask_11[1][0]                 
                                                                 no_mask_11[2][0]                 
                                                                 no_mask_11[3][0]                 
                                                                 no_mask_11[4][0]                 
                                                                 no_mask_11[5][0]                 
                                                                 no_mask_11[6][0]                 
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 7, 4, 1)      0           concatenate_6[0][0]              
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 7, 4, 2)      8           lambda_2[0][0]                   
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
k_max_pooling (KMaxPooling)     (None, 3, 4, 2)      0           conv2d[0][0]                     
__________________________________________________________________________________________________
weighted_sequence_layer_7 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_28 (Sequ (None, 1, 1)         0           weighted_sequence_layer_7[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_29 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_30 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_31 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
k_max_pooling_1 (KMaxPooling)   (None, 3, 4, 1)      0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
no_mask_9 (NoMask)              (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_28[0][0]  
                                                                 sequence_pooling_layer_29[0][0]  
                                                                 sequence_pooling_layer_30[0][0]  
                                                                 sequence_pooling_layer_31[0][0]  
__________________________________________________________________________________________________
flatten_3 (Flatten)             (None, 12)           0           k_max_pooling_1[0][0]            
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 1, 7)         0           no_mask_9[0][0]                  
                                                                 no_mask_9[1][0]                  
                                                                 no_mask_9[2][0]                  
                                                                 no_mask_9[3][0]                  
                                                                 no_mask_9[4][0]                  
                                                                 no_mask_9[5][0]                  
                                                                 no_mask_9[6][0]                  
__________________________________________________________________________________________________
dnn (DNN)                       (None, 32)           416         flatten_3[0][0]                  
__________________________________________________________________________________________________
linear_2 (Linear)               (None, 1, 1)         0           concatenate_5[0][0]              
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            32          dnn[0][0]                        
__________________________________________________________________________________________________
no_mask_10 (NoMask)             (None, 1, 1)         0           linear_2[0][0]                   
__________________________________________________________________________________________________
add_7 (Add)                     (None, 1, 1)         0           dense_1[0][0]                    
                                                                 no_mask_10[0][0]                 
__________________________________________________________________________________________________
prediction_layer_2 (PredictionL (None, 1)            1           add_7[0][0]                      
==================================================================================================
Total params: 592
Trainable params: 592
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.5500 - binary_crossentropy: 8.4837500/500 [==============================] - 1s 3ms/sample - loss: 0.4940 - binary_crossentropy: 7.6199 - val_loss: 0.5000 - val_binary_crossentropy: 7.7125

  #### metrics   #################################################### 
{'MSE': 0.497}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_24 (Sequ (None, 1, 4)         0           weighted_sequence_layer_6[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_25 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_26 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_27 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_11 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sequence_pooling_layer_24[0][0]  
                                                                 sequence_pooling_layer_25[0][0]  
                                                                 sequence_pooling_layer_26[0][0]  
                                                                 sequence_pooling_layer_27[0][0]  
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 7, 4)         0           no_mask_11[0][0]                 
                                                                 no_mask_11[1][0]                 
                                                                 no_mask_11[2][0]                 
                                                                 no_mask_11[3][0]                 
                                                                 no_mask_11[4][0]                 
                                                                 no_mask_11[5][0]                 
                                                                 no_mask_11[6][0]                 
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 7, 4, 1)      0           concatenate_6[0][0]              
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 7, 4, 2)      8           lambda_2[0][0]                   
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
k_max_pooling (KMaxPooling)     (None, 3, 4, 2)      0           conv2d[0][0]                     
__________________________________________________________________________________________________
weighted_sequence_layer_7 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_28 (Sequ (None, 1, 1)         0           weighted_sequence_layer_7[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_29 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_30 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_31 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
k_max_pooling_1 (KMaxPooling)   (None, 3, 4, 1)      0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
no_mask_9 (NoMask)              (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_28[0][0]  
                                                                 sequence_pooling_layer_29[0][0]  
                                                                 sequence_pooling_layer_30[0][0]  
                                                                 sequence_pooling_layer_31[0][0]  
__________________________________________________________________________________________________
flatten_3 (Flatten)             (None, 12)           0           k_max_pooling_1[0][0]            
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 1, 7)         0           no_mask_9[0][0]                  
                                                                 no_mask_9[1][0]                  
                                                                 no_mask_9[2][0]                  
                                                                 no_mask_9[3][0]                  
                                                                 no_mask_9[4][0]                  
                                                                 no_mask_9[5][0]                  
                                                                 no_mask_9[6][0]                  
__________________________________________________________________________________________________
dnn (DNN)                       (None, 32)           416         flatten_3[0][0]                  
__________________________________________________________________________________________________
linear_2 (Linear)               (None, 1, 1)         0           concatenate_5[0][0]              
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            32          dnn[0][0]                        
__________________________________________________________________________________________________
no_mask_10 (NoMask)             (None, 1, 1)         0           linear_2[0][0]                   
__________________________________________________________________________________________________
add_7 (Add)                     (None, 1, 1)         0           dense_1[0][0]                    
                                                                 no_mask_10[0][0]                 
__________________________________________________________________________________________________
prediction_layer_2 (PredictionL (None, 1)            1           add_7[0][0]                      
==================================================================================================
Total params: 592
Trainable params: 592
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'DCN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'DCN', 'sparse_feature_num': 3, 'dense_feature_num': 3} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_DCN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_9 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_36 (Sequ (None, 1, 4)         0           weighted_sequence_layer_9[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_37 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_38 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_39 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_2 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_15 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sequence_pooling_layer_36[0][0]  
                                                                 sequence_pooling_layer_37[0][0]  
                                                                 sequence_pooling_layer_38[0][0]  
                                                                 sequence_pooling_layer_39[0][0]  
__________________________________________________________________________________________________
no_mask_16 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
concatenate_9 (Concatenate)     (None, 1, 28)        0           no_mask_15[0][0]                 
                                                                 no_mask_15[1][0]                 
                                                                 no_mask_15[2][0]                 
                                                                 no_mask_15[3][0]                 
                                                                 no_mask_15[4][0]                 
                                                                 no_mask_15[5][0]                 
                                                                 no_mask_15[6][0]                 
__________________________________________________________________________________________________
concatenate_10 (Concatenate)    (None, 3)            0           no_mask_16[0][0]                 
                                                                 no_mask_16[1][0]                 
                                                                 no_mask_16[2][0]                 
__________________________________________________________________________________________________
weighted_sequence_layer_10 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_40 (Sequ (None, 1, 1)         0           weighted_sequence_layer_10[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_41 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_42 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_43 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
no_mask_17 (NoMask)             multiple             0           flatten_4[0][0]                  
                                                                 flatten_5[0][0]                  
__________________________________________________________________________________________________
no_mask_12 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_40[0][0]  
                                                                 sequence_pooling_layer_41[0][0]  
                                                                 sequence_pooling_layer_42[0][0]  
                                                                 sequence_pooling_layer_43[0][0]  
__________________________________________________________________________________________________
no_mask_13 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
__________________________________________________________________________________________________
concatenate_11 (Concatenate)    (None, 31)           0           no_mask_17[0][0]                 
                                                                 no_mask_17[1][0]                 
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 1, 7)         0           no_mask_12[0][0]                 
                                                                 no_mask_12[1][0]                 
                                                                 no_mask_12[2][0]                 
                                                                 no_mask_12[3][0]                 
                                                                 no_mask_12[4][0]                 
                                                                 no_mask_12[5][0]                 
                                                                 no_mask_12[6][0]                 
__________________________________________________________________________________________________
concatenate_8 (Concatenate)     (None, 3)            0           no_mask_13[0][0]                 
                                                                 no_mask_13[1][0]                 
                                                                 no_mask_13[2][0]                 
__________________________________________________________________________________________________
dnn_1 (DNN)                     (None, 8)            256         concatenate_11[0][0]             
__________________________________________________________________________________________________
linear_3 (Linear)               (None, 1)            3           concatenate_7[0][0]              
                                                                 concatenate_8[0][0]              
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            8           dnn_1[0][0]                      
__________________________________________________________________________________________________
no_mask_14 (NoMask)             (None, 1)            0           linear_3[0][0]                   
__________________________________________________________________________________________________
add_10 (Add)                    (None, 1)            0           dense_2[0][0]                    
                                                                 no_mask_14[0][0]                 
__________________________________________________________________________________________________
prediction_layer_3 (PredictionL (None, 1)            1           add_10[0][0]                     
==================================================================================================
Total params: 393
Trainable params: 393
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.5100 - binary_crossentropy: 7.8667500/500 [==============================] - 2s 3ms/sample - loss: 0.4600 - binary_crossentropy: 7.0955 - val_loss: 0.5240 - val_binary_crossentropy: 8.0827

  #### metrics   #################################################### 
{'MSE': 0.492}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_9 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_36 (Sequ (None, 1, 4)         0           weighted_sequence_layer_9[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_37 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_38 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_39 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_2 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_15 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sequence_pooling_layer_36[0][0]  
                                                                 sequence_pooling_layer_37[0][0]  
                                                                 sequence_pooling_layer_38[0][0]  
                                                                 sequence_pooling_layer_39[0][0]  
__________________________________________________________________________________________________
no_mask_16 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
concatenate_9 (Concatenate)     (None, 1, 28)        0           no_mask_15[0][0]                 
                                                                 no_mask_15[1][0]                 
                                                                 no_mask_15[2][0]                 
                                                                 no_mask_15[3][0]                 
                                                                 no_mask_15[4][0]                 
                                                                 no_mask_15[5][0]                 
                                                                 no_mask_15[6][0]                 
__________________________________________________________________________________________________
concatenate_10 (Concatenate)    (None, 3)            0           no_mask_16[0][0]                 
                                                                 no_mask_16[1][0]                 
                                                                 no_mask_16[2][0]                 
__________________________________________________________________________________________________
weighted_sequence_layer_10 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_40 (Sequ (None, 1, 1)         0           weighted_sequence_layer_10[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_41 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_42 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_43 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
no_mask_17 (NoMask)             multiple             0           flatten_4[0][0]                  
                                                                 flatten_5[0][0]                  
__________________________________________________________________________________________________
no_mask_12 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_40[0][0]  
                                                                 sequence_pooling_layer_41[0][0]  
                                                                 sequence_pooling_layer_42[0][0]  
                                                                 sequence_pooling_layer_43[0][0]  
__________________________________________________________________________________________________
no_mask_13 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
__________________________________________________________________________________________________
concatenate_11 (Concatenate)    (None, 31)           0           no_mask_17[0][0]                 
                                                                 no_mask_17[1][0]                 
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 1, 7)         0           no_mask_12[0][0]                 
                                                                 no_mask_12[1][0]                 
                                                                 no_mask_12[2][0]                 
                                                                 no_mask_12[3][0]                 
                                                                 no_mask_12[4][0]                 
                                                                 no_mask_12[5][0]                 
                                                                 no_mask_12[6][0]                 
__________________________________________________________________________________________________
concatenate_8 (Concatenate)     (None, 3)            0           no_mask_13[0][0]                 
                                                                 no_mask_13[1][0]                 
                                                                 no_mask_13[2][0]                 
__________________________________________________________________________________________________
dnn_1 (DNN)                     (None, 8)            256         concatenate_11[0][0]             
__________________________________________________________________________________________________
linear_3 (Linear)               (None, 1)            3           concatenate_7[0][0]              
                                                                 concatenate_8[0][0]              
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            8           dnn_1[0][0]                      
__________________________________________________________________________________________________
no_mask_14 (NoMask)             (None, 1)            0           linear_3[0][0]                   
__________________________________________________________________________________________________
add_10 (Add)                    (None, 1)            0           dense_2[0][0]                    
                                                                 no_mask_14[0][0]                 
__________________________________________________________________________________________________
prediction_layer_3 (PredictionL (None, 1)            1           add_10[0][0]                     
==================================================================================================
Total params: 393
Trainable params: 393
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'DeepFM', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'DeepFM', 'sparse_feature_num': 1, 'dense_feature_num': 1} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_DeepFM.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_4"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_48 (Sequ (None, 1, 4)         0           weighted_sequence_layer_12[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_49 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_50 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_51 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_22 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_48[0][0]  
                                                                 sequence_pooling_layer_49[0][0]  
                                                                 sequence_pooling_layer_50[0][0]  
                                                                 sequence_pooling_layer_51[0][0]  
__________________________________________________________________________________________________
weighted_sequence_layer_13 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_52 (Sequ (None, 1, 1)         0           weighted_sequence_layer_13[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_53 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_54 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_55 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_6 (Flatten)             (None, 20)           0           concatenate_14[0][0]             
__________________________________________________________________________________________________
flatten_7 (Flatten)             (None, 1)            0           no_mask_23[0][0]                 
__________________________________________________________________________________________________
no_mask_18 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_52[0][0]  
                                                                 sequence_pooling_layer_53[0][0]  
                                                                 sequence_pooling_layer_54[0][0]  
                                                                 sequence_pooling_layer_55[0][0]  
__________________________________________________________________________________________________
no_mask_21 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_48[0][0]  
                                                                 sequence_pooling_layer_49[0][0]  
                                                                 sequence_pooling_layer_50[0][0]  
                                                                 sequence_pooling_layer_51[0][0]  
__________________________________________________________________________________________________
no_mask_24 (NoMask)             multiple             0           flatten_6[0][0]                  
                                                                 flatten_7[0][0]                  
__________________________________________________________________________________________________
concatenate_12 (Concatenate)    (None, 1, 5)         0           no_mask_18[0][0]                 
                                                                 no_mask_18[1][0]                 
                                                                 no_mask_18[2][0]                 
                                                                 no_mask_18[3][0]                 
                                                                 no_mask_18[4][0]                 
__________________________________________________________________________________________________
no_mask_19 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
concatenate_13 (Concatenate)    (None, 5, 4)         0           no_mask_21[0][0]                 
                                                                 no_mask_21[1][0]                 
                                                                 no_mask_21[2][0]                 
                                                                 no_mask_21[3][0]                 
                                                                 no_mask_21[4][0]                 
__________________________________________________________________________________________________
concatenate_15 (Concatenate)    (None, 21)           0           no_mask_24[0][0]                 
                                                                 no_mask_24[1][0]                 
__________________________________________________________________________________________________
linear_4 (Linear)               (None, 1)            1           concatenate_12[0][0]             
                                                                 no_mask_19[0][0]                 
__________________________________________________________________________________________________
fm (FM)                         (None, 1)            0           concatenate_13[0][0]             
__________________________________________________________________________________________________
dnn_2 (DNN)                     (None, 2)            44          concatenate_15[0][0]             
__________________________________________________________________________________________________
no_mask_20 (NoMask)             (None, 1)            0           linear_4[0][0]                   
__________________________________________________________________________________________________
add_13 (Add)                    (None, 1)            0           fm[0][0]                         
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 1)            2           dnn_2[0][0]                      
__________________________________________________________________________________________________
add_14 (Add)                    (None, 1)            0           no_mask_20[0][0]                 
                                                                 add_13[0][0]                     
                                                                 dense_3[0][0]                    
__________________________________________________________________________________________________
prediction_layer_4 (PredictionL (None, 1)            1           add_14[0][0]                     
==================================================================================================
Total params: 173
Trainable params: 173
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 3s - loss: 0.2837 - binary_crossentropy: 0.7641500/500 [==============================] - 2s 4ms/sample - loss: 0.2666 - binary_crossentropy: 0.7287 - val_loss: 0.2663 - val_binary_crossentropy: 0.7280

  #### metrics   #################################################### 
{'MSE': 0.2652834355272852}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_4"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_48 (Sequ (None, 1, 4)         0           weighted_sequence_layer_12[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_49 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_50 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_51 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_22 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_48[0][0]  
                                                                 sequence_pooling_layer_49[0][0]  
                                                                 sequence_pooling_layer_50[0][0]  
                                                                 sequence_pooling_layer_51[0][0]  
__________________________________________________________________________________________________
weighted_sequence_layer_13 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_52 (Sequ (None, 1, 1)         0           weighted_sequence_layer_13[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_53 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_54 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_55 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_6 (Flatten)             (None, 20)           0           concatenate_14[0][0]             
__________________________________________________________________________________________________
flatten_7 (Flatten)             (None, 1)            0           no_mask_23[0][0]                 
__________________________________________________________________________________________________
no_mask_18 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_52[0][0]  
                                                                 sequence_pooling_layer_53[0][0]  
                                                                 sequence_pooling_layer_54[0][0]  
                                                                 sequence_pooling_layer_55[0][0]  
__________________________________________________________________________________________________
no_mask_21 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_48[0][0]  
                                                                 sequence_pooling_layer_49[0][0]  
                                                                 sequence_pooling_layer_50[0][0]  
                                                                 sequence_pooling_layer_51[0][0]  
__________________________________________________________________________________________________
no_mask_24 (NoMask)             multiple             0           flatten_6[0][0]                  
                                                                 flatten_7[0][0]                  
__________________________________________________________________________________________________
concatenate_12 (Concatenate)    (None, 1, 5)         0           no_mask_18[0][0]                 
                                                                 no_mask_18[1][0]                 
                                                                 no_mask_18[2][0]                 
                                                                 no_mask_18[3][0]                 
                                                                 no_mask_18[4][0]                 
__________________________________________________________________________________________________
no_mask_19 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
concatenate_13 (Concatenate)    (None, 5, 4)         0           no_mask_21[0][0]                 
                                                                 no_mask_21[1][0]                 
                                                                 no_mask_21[2][0]                 
                                                                 no_mask_21[3][0]                 
                                                                 no_mask_21[4][0]                 
__________________________________________________________________________________________________
concatenate_15 (Concatenate)    (None, 21)           0           no_mask_24[0][0]                 
                                                                 no_mask_24[1][0]                 
__________________________________________________________________________________________________
linear_4 (Linear)               (None, 1)            1           concatenate_12[0][0]             
                                                                 no_mask_19[0][0]                 
__________________________________________________________________________________________________
fm (FM)                         (None, 1)            0           concatenate_13[0][0]             
__________________________________________________________________________________________________
dnn_2 (DNN)                     (None, 2)            44          concatenate_15[0][0]             
__________________________________________________________________________________________________
no_mask_20 (NoMask)             (None, 1)            0           linear_4[0][0]                   
__________________________________________________________________________________________________
add_13 (Add)                    (None, 1)            0           fm[0][0]                         
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 1)            2           dnn_2[0][0]                      
__________________________________________________________________________________________________
add_14 (Add)                    (None, 1)            0           no_mask_20[0][0]                 
                                                                 add_13[0][0]                     
                                                                 dense_3[0][0]                    
__________________________________________________________________________________________________
prediction_layer_4 (PredictionL (None, 1)            1           add_14[0][0]                     
==================================================================================================
Total params: 173
Trainable params: 173
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'DIEN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'DIEN'} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_DIEN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/layers/sequence.py:724: GRUCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
Instructions for updating:
This class is equivalent as tf.keras.layers.GRUCell, and will be replaced by that in Tensorflow 2.0.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.AUTO_REUSE is deprecated. Please use tf.compat.v1.AUTO_REUSE instead.

WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/contrib/rnn.py:798: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.cast` instead.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/rnn_cell_impl.py:559: Layer.add_variable (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.add_weight` method instead.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/rnn_cell_impl.py:565: calling Constant.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/models/dien.py:282: The name tf.keras.backend.get_session is deprecated. Please use tf.compat.v1.keras.backend.get_session instead.

WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/models/dien.py:282: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

Model: "model_5"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
item (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
item_gender (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
hist_item (InputLayer)          [(None, 4)]          0                                            
__________________________________________________________________________________________________
hist_item_gender (InputLayer)   [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_hist_item (Embed multiple             32          item[0][0]                       
                                                                 hist_item[0][0]                  
                                                                 item[0][0]                       
__________________________________________________________________________________________________
sparse_seq_emb_hist_item_gender multiple             12          item_gender[0][0]                
                                                                 hist_item_gender[0][0]           
                                                                 item_gender[0][0]                
__________________________________________________________________________________________________
no_mask_25 (NoMask)             multiple             0           sparse_seq_emb_hist_item[1][0]   
                                                                 sparse_seq_emb_hist_item_gender[1
__________________________________________________________________________________________________
user (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
gender (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
concatenate_16 (Concatenate)    (None, 4, 12)        0           no_mask_25[0][0]                 
                                                                 no_mask_25[1][0]                 
__________________________________________________________________________________________________
seq_length (InputLayer)         [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_emb_user (Embedding)     (None, 1, 1)         3           user[0][0]                       
__________________________________________________________________________________________________
sparse_emb_gender (Embedding)   (None, 1, 1)         2           gender[0][0]                     
__________________________________________________________________________________________________
no_mask_27 (NoMask)             multiple             0           sparse_seq_emb_hist_item[0][0]   
                                                                 sparse_seq_emb_hist_item_gender[0
__________________________________________________________________________________________________
gru1 (DynamicGRU)               (None, 4, 12)        900         concatenate_16[0][0]             
                                                                 seq_length[0][0]                 
__________________________________________________________________________________________________
no_mask_26 (NoMask)             multiple             0           sparse_emb_user[0][0]            
                                                                 sparse_emb_gender[0][0]          
                                                                 sparse_seq_emb_hist_item[2][0]   
                                                                 sparse_seq_emb_hist_item_gender[2
__________________________________________________________________________________________________
concatenate_18 (Concatenate)    (None, 1, 12)        0           no_mask_27[0][0]                 
                                                                 no_mask_27[1][0]                 
__________________________________________________________________________________________________
gru2 (DynamicGRU)               (None, 4, 12)        900         gru1[0][0]                       
                                                                 seq_length[0][0]                 
__________________________________________________________________________________________________
concatenate_17 (Concatenate)    (None, 1, 14)        0           no_mask_26[0][0]                 
                                                                 no_mask_26[1][0]                 
                                                                 no_mask_26[2][0]                 
                                                                 no_mask_26[3][0]                 
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 12)        4433        concatenate_18[0][0]             
                                                                 gru2[0][0]                       
                                                                 seq_length[0][0]                 
__________________________________________________________________________________________________
concatenate_19 (Concatenate)    (None, 1, 26)        0           concatenate_17[0][0]             
                                                                 attention_sequence_pooling_layer[
__________________________________________________________________________________________________
flatten_8 (Flatten)             (None, 26)           0           concatenate_19[0][0]             
__________________________________________________________________________________________________
score (InputLayer)              [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_28 (NoMask)             (None, 26)           0           flatten_8[0][0]                  
__________________________________________________________________________________________________
no_mask_29 (NoMask)             (None, 1)            0           score[0][0]                      
__________________________________________________________________________________________________
flatten_9 (Flatten)             (None, 26)           0           no_mask_28[0][0]                 
__________________________________________________________________________________________________
flatten_10 (Flatten)            (None, 1)            0           no_mask_29[0][0]                 
__________________________________________________________________________________________________
no_mask_30 (NoMask)             multiple             0           flatten_9[0][0]                  
                                                                 flatten_10[0][0]                 
__________________________________________________________________________________________________
concatenate_20 (Concatenate)    (None, 27)           0           no_mask_30[0][0]                 
                                                                 no_mask_30[1][0]                 
__________________________________________________________________________________________________
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-22 17:02:00.996329: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:02:00.998353: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:02:01.004279: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 17:02:01.015141: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 17:02:01.017100: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 17:02:01.018754: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:02:01.020263: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 1)            4           dnn_4[0][0]                      
__________________________________________________________________________________________________
prediction_layer_5 (PredictionL (None, 1)            1           dense_4[0][0]                    
==================================================================================================
Total params: 6,439
Trainable params: 6,279
Non-trainable params: 160
__________________________________________________________________________________________________
Train on 1 samples, validate on 2 samples
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2517 - val_binary_crossentropy: 0.6966
2020-05-22 17:02:02.550481: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:02:02.552306: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:02:02.556928: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 17:02:02.566169: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 17:02:02.567690: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 17:02:02.569112: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:02:02.570406: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25219899047297467}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_5"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
item (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
item_gender (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
hist_item (InputLayer)          [(None, 4)]          0                                            
__________________________________________________________________________________________________
hist_item_gender (InputLayer)   [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_hist_item (Embed multiple             32          item[0][0]                       
                                                                 hist_item[0][0]                  
                                                                 item[0][0]                       
__________________________________________________________________________________________________
sparse_seq_emb_hist_item_gender multiple             12          item_gender[0][0]                
                                                                 hist_item_gender[0][0]           
                                                                 item_gender[0][0]                
__________________________________________________________________________________________________
no_mask_25 (NoMask)             multiple             0           sparse_seq_emb_hist_item[1][0]   
                                                                 sparse_seq_emb_hist_item_gender[1
__________________________________________________________________________________________________
user (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
gender (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
concatenate_16 (Concatenate)    (None, 4, 12)        0           no_mask_25[0][0]                 
                                                                 no_mask_25[1][0]                 
__________________________________________________________________________________________________
seq_length (InputLayer)         [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_emb_user (Embedding)     (None, 1, 1)         3           user[0][0]                       
__________________________________________________________________________________________________
sparse_emb_gender (Embedding)   (None, 1, 1)         2           gender[0][0]                     
__________________________________________________________________________________________________
no_mask_27 (NoMask)             multiple             0           sparse_seq_emb_hist_item[0][0]   
                                                                 sparse_seq_emb_hist_item_gender[0
__________________________________________________________________________________________________
gru1 (DynamicGRU)               (None, 4, 12)        900         concatenate_16[0][0]             
                                                                 seq_length[0][0]                 
__________________________________________________________________________________________________
no_mask_26 (NoMask)             multiple             0           sparse_emb_user[0][0]            
                                                                 sparse_emb_gender[0][0]          
                                                                 sparse_seq_emb_hist_item[2][0]   
                                                                 sparse_seq_emb_hist_item_gender[2
__________________________________________________________________________________________________
concatenate_18 (Concatenate)    (None, 1, 12)        0           no_mask_27[0][0]                 
                                                                 no_mask_27[1][0]                 
__________________________________________________________________________________________________
gru2 (DynamicGRU)               (None, 4, 12)        900         gru1[0][0]                       
                                                                 seq_length[0][0]                 
__________________________________________________________________________________________________
concatenate_17 (Concatenate)    (None, 1, 14)        0           no_mask_26[0][0]                 
                                                                 no_mask_26[1][0]                 
                                                                 no_mask_26[2][0]                 
                                                                 no_mask_26[3][0]                 
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 12)        4433        concatenate_18[0][0]             
                                                                 gru2[0][0]                       
                                                                 seq_length[0][0]                 
__________________________________________________________________________________________________
concatenate_19 (Concatenate)    (None, 1, 26)        0           concatenate_17[0][0]             
                                                                 attention_sequence_pooling_layer[
__________________________________________________________________________________________________
flatten_8 (Flatten)             (None, 26)           0           concatenate_19[0][0]             
__________________________________________________________________________________________________
score (InputLayer)              [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_28 (NoMask)             (None, 26)           0           flatten_8[0][0]                  
__________________________________________________________________________________________________
no_mask_29 (NoMask)             (None, 1)            0           score[0][0]                      
__________________________________________________________________________________________________
flatten_9 (Flatten)             (None, 26)           0           no_mask_28[0][0]                 
__________________________________________________________________________________________________
flatten_10 (Flatten)            (None, 1)            0           no_mask_29[0][0]                 
__________________________________________________________________________________________________
no_mask_30 (NoMask)             multiple             0           flatten_9[0][0]                  
                                                                 flatten_10[0][0]                 
__________________________________________________________________________________________________
concatenate_20 (Concatenate)    (None, 27)           0           no_mask_30[0][0]                 
                                                                 no_mask_30[1][0]                 
__________________________________________________________________________________________________
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 1)            4           dnn_4[0][0]                      
__________________________________________________________________________________________________
prediction_layer_5 (PredictionL (None, 1)            1           dense_4[0][0]                    
==================================================================================================
Total params: 6,439
Trainable params: 6,279
Non-trainable params: 160
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'DIN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'DIN'} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_DIN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
2020-05-22 17:02:29.485986: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:02:29.487763: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:02:29.491338: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 17:02:29.498077: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 17:02:29.499164: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 17:02:29.500151: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:02:29.501023: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
Model: "model_6"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
user (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
gender (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
item (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
item_gender (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_emb_user (Embedding)     (None, 1, 4)         12          user[0][0]                       
__________________________________________________________________________________________________
sparse_emb_gender (Embedding)   (None, 1, 4)         8           gender[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_hist_item (Embed multiple             32          item[0][0]                       
                                                                 hist_item[0][0]                  
                                                                 item[0][0]                       
__________________________________________________________________________________________________
sparse_seq_emb_hist_item_gender multiple             12          item_gender[0][0]                
                                                                 hist_item_gender[0][0]           
                                                                 item_gender[0][0]                
__________________________________________________________________________________________________
hist_item (InputLayer)          [(None, 4)]          0                                            
__________________________________________________________________________________________________
hist_item_gender (InputLayer)   [(None, 4)]          0                                            
__________________________________________________________________________________________________
no_mask_31 (NoMask)             multiple             0           sparse_emb_user[0][0]            
                                                                 sparse_emb_gender[0][0]          
                                                                 sparse_seq_emb_hist_item[2][0]   
                                                                 sparse_seq_emb_hist_item_gender[2
__________________________________________________________________________________________________
concatenate_22 (Concatenate)    (None, 1, 20)        0           no_mask_31[0][0]                 
                                                                 no_mask_31[1][0]                 
                                                                 no_mask_31[2][0]                 
                                                                 no_mask_31[3][0]                 
__________________________________________________________________________________________________
concatenate_23 (Concatenate)    (None, 1, 12)        0           sparse_seq_emb_hist_item[0][0]   
                                                                 sparse_seq_emb_hist_item_gender[0
__________________________________________________________________________________________________
concatenate_21 (Concatenate)    (None, 4, 12)        0           sparse_seq_emb_hist_item[1][0]   
                                                                 sparse_seq_emb_hist_item_gender[1
__________________________________________________________________________________________________
no_mask_32 (NoMask)             (None, 1, 20)        0           concatenate_22[0][0]             
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 12)        7561        concatenate_23[0][0]             
                                                                 concatenate_21[0][0]             
__________________________________________________________________________________________________
concatenate_24 (Concatenate)    (None, 1, 32)        0           no_mask_32[0][0]                 
                                                                 attention_sequence_pooling_layer_
__________________________________________________________________________________________________
flatten_11 (Flatten)            (None, 32)           0           concatenate_24[0][0]             
__________________________________________________________________________________________________
score (InputLayer)              [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_33 (NoMask)             (None, 32)           0           flatten_11[0][0]                 
__________________________________________________________________________________________________
no_mask_34 (NoMask)             (None, 1)            0           score[0][0]                      
__________________________________________________________________________________________________
flatten_12 (Flatten)            (None, 32)           0           no_mask_33[0][0]                 
__________________________________________________________________________________________________
flatten_13 (Flatten)            (None, 1)            0           no_mask_34[0][0]                 
__________________________________________________________________________________________________
no_mask_35 (NoMask)             multiple             0           flatten_12[0][0]                 
                                                                 flatten_13[0][0]                 
__________________________________________________________________________________________________
concatenate_25 (Concatenate)    (None, 33)           0           no_mask_35[0][0]                 
                                                                 no_mask_35[1][0]                 
__________________________________________________________________________________________________
dnn_7 (DNN)                     (None, 4)            176         concatenate_25[0][0]             
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 1)            4           dnn_7[0][0]                      
__________________________________________________________________________________________________
prediction_layer_6 (PredictionL (None, 1)            1           dense_5[0][0]                    
==================================================================================================
Total params: 7,806
Trainable params: 7,566
Non-trainable params: 240
__________________________________________________________________________________________________
Train on 1 samples, validate on 2 samples
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2507 - val_binary_crossentropy: 0.6945
2020-05-22 17:02:31.314644: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:02:31.316010: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:02:31.318910: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 17:02:31.324363: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 17:02:31.325272: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 17:02:31.326091: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:02:31.326831: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25079471605872367}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_6"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
user (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
gender (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
item (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
item_gender (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_emb_user (Embedding)     (None, 1, 4)         12          user[0][0]                       
__________________________________________________________________________________________________
sparse_emb_gender (Embedding)   (None, 1, 4)         8           gender[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_hist_item (Embed multiple             32          item[0][0]                       
                                                                 hist_item[0][0]                  
                                                                 item[0][0]                       
__________________________________________________________________________________________________
sparse_seq_emb_hist_item_gender multiple             12          item_gender[0][0]                
                                                                 hist_item_gender[0][0]           
                                                                 item_gender[0][0]                
__________________________________________________________________________________________________
hist_item (InputLayer)          [(None, 4)]          0                                            
__________________________________________________________________________________________________
hist_item_gender (InputLayer)   [(None, 4)]          0                                            
__________________________________________________________________________________________________
no_mask_31 (NoMask)             multiple             0           sparse_emb_user[0][0]            
                                                                 sparse_emb_gender[0][0]          
                                                                 sparse_seq_emb_hist_item[2][0]   
                                                                 sparse_seq_emb_hist_item_gender[2
__________________________________________________________________________________________________
concatenate_22 (Concatenate)    (None, 1, 20)        0           no_mask_31[0][0]                 
                                                                 no_mask_31[1][0]                 
                                                                 no_mask_31[2][0]                 
                                                                 no_mask_31[3][0]                 
__________________________________________________________________________________________________
concatenate_23 (Concatenate)    (None, 1, 12)        0           sparse_seq_emb_hist_item[0][0]   
                                                                 sparse_seq_emb_hist_item_gender[0
__________________________________________________________________________________________________
concatenate_21 (Concatenate)    (None, 4, 12)        0           sparse_seq_emb_hist_item[1][0]   
                                                                 sparse_seq_emb_hist_item_gender[1
__________________________________________________________________________________________________
no_mask_32 (NoMask)             (None, 1, 20)        0           concatenate_22[0][0]             
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 12)        7561        concatenate_23[0][0]             
                                                                 concatenate_21[0][0]             
__________________________________________________________________________________________________
concatenate_24 (Concatenate)    (None, 1, 32)        0           no_mask_32[0][0]                 
                                                                 attention_sequence_pooling_layer_
__________________________________________________________________________________________________
flatten_11 (Flatten)            (None, 32)           0           concatenate_24[0][0]             
__________________________________________________________________________________________________
score (InputLayer)              [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_33 (NoMask)             (None, 32)           0           flatten_11[0][0]                 
__________________________________________________________________________________________________
no_mask_34 (NoMask)             (None, 1)            0           score[0][0]                      
__________________________________________________________________________________________________
flatten_12 (Flatten)            (None, 32)           0           no_mask_33[0][0]                 
__________________________________________________________________________________________________
flatten_13 (Flatten)            (None, 1)            0           no_mask_34[0][0]                 
__________________________________________________________________________________________________
no_mask_35 (NoMask)             multiple             0           flatten_12[0][0]                 
                                                                 flatten_13[0][0]                 
__________________________________________________________________________________________________
concatenate_25 (Concatenate)    (None, 33)           0           no_mask_35[0][0]                 
                                                                 no_mask_35[1][0]                 
__________________________________________________________________________________________________
dnn_7 (DNN)                     (None, 4)            176         concatenate_25[0][0]             
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 1)            4           dnn_7[0][0]                      
__________________________________________________________________________________________________
prediction_layer_6 (PredictionL (None, 1)            1           dense_5[0][0]                    
==================================================================================================
Total params: 7,806
Trainable params: 7,566
Non-trainable params: 240
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'DSIN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'DSIN'} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_DSIN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.string_to_hash_bucket_fast is deprecated. Please use tf.strings.to_hash_bucket_fast instead.

WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.matrix_set_diag is deprecated. Please use tf.linalg.set_diag instead.

Model: "model_7"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
item (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
item_gender (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
sess_0_item (InputLayer)        [(None, 4)]          0                                            
__________________________________________________________________________________________________
sess_0_item_gender (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
sess_1_item (InputLayer)        [(None, 4)]          0                                            
__________________________________________________________________________________________________
sess_1_item_gender (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_4 (Hash)                   (None, 1)            0           item[0][0]                       
__________________________________________________________________________________________________
hash_5 (Hash)                   (None, 1)            0           item_gender[0][0]                
__________________________________________________________________________________________________
hash (Hash)                     (None, 1)            0           item[0][0]                       
__________________________________________________________________________________________________
hash_1 (Hash)                   (None, 1)            0           item_gender[0][0]                
__________________________________________________________________________________________________
hash_6 (Hash)                   (None, 4)            0           sess_0_item[0][0]                
__________________________________________________________________________________________________
hash_7 (Hash)                   (None, 4)            0           sess_0_item_gender[0][0]         
__________________________________________________________________________________________________
hash_8 (Hash)                   (None, 4)            0           sess_1_item[0][0]                
__________________________________________________________________________________________________
hash_9 (Hash)                   (None, 4)            0           sess_1_item_gender[0][0]         
__________________________________________________________________________________________________
sparse_emb_2-item (Embedding)   multiple             16          hash[0][0]                       
                                                                 hash_4[0][0]                     
                                                                 hash_6[0][0]                     
                                                                 hash_8[0][0]                     
__________________________________________________________________________________________________
sparse_emb_3-item_gender (Embed multiple             12          hash_1[0][0]                     
                                                                 hash_5[0][0]                     
                                                                 hash_7[0][0]                     
                                                                 hash_9[0][0]                     
__________________________________________________________________________________________________
concatenate_28 (Concatenate)    (None, 4, 8)         0           sparse_emb_2-item[2][0]          
                                                                 sparse_emb_3-item_gender[2][0]   
__________________________________________________________________________________________________
concatenate_29 (Concatenate)    (None, 4, 8)         0           sparse_emb_2-item[3][0]          
                                                                 sparse_emb_3-item_gender[3][0]   
__________________________________________________________________________________________________
user (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
gender (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
transformer (Transformer)       (None, 1, 8)         704         concatenate_28[0][0]             
                                                                 concatenate_28[0][0]             
                                                                 concatenate_29[0][0]             
                                                                 concatenate_29[0][0]             
__________________________________________________________________________________________________
hash_2 (Hash)                   (None, 1)            0           user[0][0]                       
__________________________________________________________________________________________________
hash_3 (Hash)                   (None, 1)            0           gender[0][0]                     
__________________________________________________________________________________________________
no_mask_37 (NoMask)             (None, 1, 8)         0           transformer[0][0]                
                                                                 transformer[1][0]                
__________________________________________________________________________________________________
sparse_emb_0-user (Embedding)   (None, 1, 4)         12          hash_2[0][0]                     
__________________________________________________________________________________________________
sparse_emb_1-gender (Embedding) (None, 1, 4)         8           hash_3[0][0]                     
__________________________________________________________________________________________________
concatenate_30 (Concatenate)    (None, 2, 8)         0           no_mask_37[0][0]                 
                                                                 no_mask_37[1][0]                 
__________________________________________________________________________________________________
no_mask_36 (NoMask)             (None, 1, 4)         0           sparse_emb_0-user[0][0]          
                                                                 sparse_emb_1-gender[0][0]        
                                                                 sparse_emb_2-item[1][0]          
                                                                 sparse_emb_3-item_gender[1][0]   
__________________________________________________________________________________________________
concatenate_26 (Concatenate)    (None, 1, 8)         0           sparse_emb_2-item[0][0]          
                                                                 sparse_emb_3-item_gender[0][0]   
__________________________________________________________________________________________________
sess_length (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
bi_lstm (BiLSTM)                (None, 2, 8)         2176        concatenate_30[0][0]             
__________________________________________________________________________________________________
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-22 17:03:09.338291: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:03:09.343680: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:03:09.361336: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 17:03:09.391751: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 17:03:09.397048: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 17:03:09.402171: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:03:09.406998: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

                                                                 no_mask_36[1][0]                 
                                                                 no_mask_36[2][0]                 
                                                                 no_mask_36[3][0]                 
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 8)         3169        concatenate_26[0][0]             
                                                                 concatenate_30[0][0]             
                                                                 sess_length[0][0]                
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 8)         3169        concatenate_26[0][0]             
                                                                 bi_lstm[0][0]                    
                                                                 sess_length[0][0]                
__________________________________________________________________________________________________
flatten_14 (Flatten)            (None, 16)           0           concatenate_27[0][0]             
__________________________________________________________________________________________________
flatten_15 (Flatten)            (None, 8)            0           attention_sequence_pooling_layer_
__________________________________________________________________________________________________
flatten_16 (Flatten)            (None, 8)            0           attention_sequence_pooling_layer_
__________________________________________________________________________________________________
concatenate_31 (Concatenate)    (None, 32)           0           flatten_14[0][0]                 
                                                                 flatten_15[0][0]                 
                                                                 flatten_16[0][0]                 
__________________________________________________________________________________________________
score (InputLayer)              [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_38 (NoMask)             (None, 32)           0           concatenate_31[0][0]             
__________________________________________________________________________________________________
no_mask_39 (NoMask)             (None, 1)            0           score[0][0]                      
__________________________________________________________________________________________________
flatten_17 (Flatten)            (None, 32)           0           no_mask_38[0][0]                 
__________________________________________________________________________________________________
flatten_18 (Flatten)            (None, 1)            0           no_mask_39[0][0]                 
__________________________________________________________________________________________________
no_mask_40 (NoMask)             multiple             0           flatten_17[0][0]                 
                                                                 flatten_18[0][0]                 
__________________________________________________________________________________________________
concatenate_32 (Concatenate)    (None, 33)           0           no_mask_40[0][0]                 
                                                                 no_mask_40[1][0]                 
__________________________________________________________________________________________________
dnn_11 (DNN)                    (None, 4)            176         concatenate_32[0][0]             
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 1)            4           dnn_11[0][0]                     
__________________________________________________________________________________________________
prediction_layer_7 (PredictionL (None, 1)            1           dense_6[0][0]                    
==================================================================================================
Total params: 9,447
Trainable params: 9,447
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 1 samples, validate on 2 samples
1/1 [==============================] - 6s 6s/sample - loss: 0.2062 - binary_crossentropy: 0.6053 - val_loss: 0.2621 - val_binary_crossentropy: 0.7179
2020-05-22 17:03:12.120775: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:03:12.125781: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:03:12.139329: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 17:03:12.166819: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 17:03:12.171684: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 17:03:12.176299: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 17:03:12.180516: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.22534332069575194}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_7"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
item (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
item_gender (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
sess_0_item (InputLayer)        [(None, 4)]          0                                            
__________________________________________________________________________________________________
sess_0_item_gender (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
sess_1_item (InputLayer)        [(None, 4)]          0                                            
__________________________________________________________________________________________________
sess_1_item_gender (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_4 (Hash)                   (None, 1)            0           item[0][0]                       
__________________________________________________________________________________________________
hash_5 (Hash)                   (None, 1)            0           item_gender[0][0]                
__________________________________________________________________________________________________
hash (Hash)                     (None, 1)            0           item[0][0]                       
__________________________________________________________________________________________________
hash_1 (Hash)                   (None, 1)            0           item_gender[0][0]                
__________________________________________________________________________________________________
hash_6 (Hash)                   (None, 4)            0           sess_0_item[0][0]                
__________________________________________________________________________________________________
hash_7 (Hash)                   (None, 4)            0           sess_0_item_gender[0][0]         
__________________________________________________________________________________________________
hash_8 (Hash)                   (None, 4)            0           sess_1_item[0][0]                
__________________________________________________________________________________________________
hash_9 (Hash)                   (None, 4)            0           sess_1_item_gender[0][0]         
__________________________________________________________________________________________________
sparse_emb_2-item (Embedding)   multiple             16          hash[0][0]                       
                                                                 hash_4[0][0]                     
                                                                 hash_6[0][0]                     
                                                                 hash_8[0][0]                     
__________________________________________________________________________________________________
sparse_emb_3-item_gender (Embed multiple             12          hash_1[0][0]                     
                                                                 hash_5[0][0]                     
                                                                 hash_7[0][0]                     
                                                                 hash_9[0][0]                     
__________________________________________________________________________________________________
concatenate_28 (Concatenate)    (None, 4, 8)         0           sparse_emb_2-item[2][0]          
                                                                 sparse_emb_3-item_gender[2][0]   
__________________________________________________________________________________________________
concatenate_29 (Concatenate)    (None, 4, 8)         0           sparse_emb_2-item[3][0]          
                                                                 sparse_emb_3-item_gender[3][0]   
__________________________________________________________________________________________________
user (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
gender (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
transformer (Transformer)       (None, 1, 8)         704         concatenate_28[0][0]             
                                                                 concatenate_28[0][0]             
                                                                 concatenate_29[0][0]             
                                                                 concatenate_29[0][0]             
__________________________________________________________________________________________________
hash_2 (Hash)                   (None, 1)            0           user[0][0]                       
__________________________________________________________________________________________________
hash_3 (Hash)                   (None, 1)            0           gender[0][0]                     
__________________________________________________________________________________________________
no_mask_37 (NoMask)             (None, 1, 8)         0           transformer[0][0]                
                                                                 transformer[1][0]                
__________________________________________________________________________________________________
sparse_emb_0-user (Embedding)   (None, 1, 4)         12          hash_2[0][0]                     
__________________________________________________________________________________________________
sparse_emb_1-gender (Embedding) (None, 1, 4)         8           hash_3[0][0]                     
__________________________________________________________________________________________________
concatenate_30 (Concatenate)    (None, 2, 8)         0           no_mask_37[0][0]                 
                                                                 no_mask_37[1][0]                 
__________________________________________________________________________________________________
no_mask_36 (NoMask)             (None, 1, 4)         0           sparse_emb_0-user[0][0]          
                                                                 sparse_emb_1-gender[0][0]        
                                                                 sparse_emb_2-item[1][0]          
                                                                 sparse_emb_3-item_gender[1][0]   
__________________________________________________________________________________________________
concatenate_26 (Concatenate)    (None, 1, 8)         0           sparse_emb_2-item[0][0]          
                                                                 sparse_emb_3-item_gender[0][0]   
__________________________________________________________________________________________________
sess_length (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
bi_lstm (BiLSTM)                (None, 2, 8)         2176        concatenate_30[0][0]             
__________________________________________________________________________________________________
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 
                                                                 no_mask_36[1][0]                 
                                                                 no_mask_36[2][0]                 
                                                                 no_mask_36[3][0]                 
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 8)         3169        concatenate_26[0][0]             
                                                                 concatenate_30[0][0]             
                                                                 sess_length[0][0]                
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 8)         3169        concatenate_26[0][0]             
                                                                 bi_lstm[0][0]                    
                                                                 sess_length[0][0]                
__________________________________________________________________________________________________
flatten_14 (Flatten)            (None, 16)           0           concatenate_27[0][0]             
__________________________________________________________________________________________________
flatten_15 (Flatten)            (None, 8)            0           attention_sequence_pooling_layer_
__________________________________________________________________________________________________
flatten_16 (Flatten)            (None, 8)            0           attention_sequence_pooling_layer_
__________________________________________________________________________________________________
concatenate_31 (Concatenate)    (None, 32)           0           flatten_14[0][0]                 
                                                                 flatten_15[0][0]                 
                                                                 flatten_16[0][0]                 
__________________________________________________________________________________________________
score (InputLayer)              [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_38 (NoMask)             (None, 32)           0           concatenate_31[0][0]             
__________________________________________________________________________________________________
no_mask_39 (NoMask)             (None, 1)            0           score[0][0]                      
__________________________________________________________________________________________________
flatten_17 (Flatten)            (None, 32)           0           no_mask_38[0][0]                 
__________________________________________________________________________________________________
flatten_18 (Flatten)            (None, 1)            0           no_mask_39[0][0]                 
__________________________________________________________________________________________________
no_mask_40 (NoMask)             multiple             0           flatten_17[0][0]                 
                                                                 flatten_18[0][0]                 
__________________________________________________________________________________________________
concatenate_32 (Concatenate)    (None, 33)           0           no_mask_40[0][0]                 
                                                                 no_mask_40[1][0]                 
__________________________________________________________________________________________________
dnn_11 (DNN)                    (None, 4)            176         concatenate_32[0][0]             
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 1)            4           dnn_11[0][0]                     
__________________________________________________________________________________________________
prediction_layer_7 (PredictionL (None, 1)            1           dense_6[0][0]                    
==================================================================================================
Total params: 9,447
Trainable params: 9,447
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'FiBiNET', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'FiBiNET', 'sparse_feature_num': 2, 'dense_feature_num': 2} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_FiBiNET.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_8"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_15 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_60 (Sequ (None, 1, 4)         0           weighted_sequence_layer_15[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_61 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_62 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_63 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
senet_layer (SENETLayer)        [(None, 1, 4), (None 24          sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_60[0][0]  
                                                                 sequence_pooling_layer_61[0][0]  
                                                                 sequence_pooling_layer_62[0][0]  
                                                                 sequence_pooling_layer_63[0][0]  
__________________________________________________________________________________________________
bilinear_interaction (BilinearI (None, 1, 60)        16          senet_layer[0][0]                
                                                                 senet_layer[0][1]                
                                                                 senet_layer[0][2]                
                                                                 senet_layer[0][3]                
                                                                 senet_layer[0][4]                
                                                                 senet_layer[0][5]                
__________________________________________________________________________________________________
bilinear_interaction_1 (Bilinea (None, 1, 60)        16          sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_60[0][0]  
                                                                 sequence_pooling_layer_61[0][0]  
                                                                 sequence_pooling_layer_62[0][0]  
                                                                 sequence_pooling_layer_63[0][0]  
__________________________________________________________________________________________________
no_mask_47 (NoMask)             (None, 1, 60)        0           bilinear_interaction[0][0]       
                                                                 bilinear_interaction_1[0][0]     
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
concatenate_38 (Concatenate)    (None, 1, 120)       0           no_mask_47[0][0]                 
                                                                 no_mask_47[1][0]                 
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
flatten_19 (Flatten)            (None, 120)          0           concatenate_38[0][0]             
__________________________________________________________________________________________________
no_mask_49 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
weighted_sequence_layer_16 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_64 (Sequ (None, 1, 1)         0           weighted_sequence_layer_16[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_65 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_66 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_67 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_20 (Flatten)            (None, 120)          0           no_mask_48[0][0]                 
__________________________________________________________________________________________________
flatten_21 (Flatten)            (None, 2)            0           concatenate_39[0][0]             
__________________________________________________________________________________________________
no_mask_44 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_64[0][0]  
                                                                 sequence_pooling_layer_65[0][0]  
                                                                 sequence_pooling_layer_66[0][0]  
                                                                 sequence_pooling_layer_67[0][0]  
__________________________________________________________________________________________________
no_mask_45 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
no_mask_50 (NoMask)             multiple             0           flatten_20[0][0]                 
                                                                 flatten_21[0][0]                 
__________________________________________________________________________________________________
concatenate_36 (Concatenate)    (None, 1, 6)         0           no_mask_44[0][0]                 
                                                                 no_mask_44[1][0]                 
                                                                 no_mask_44[2][0]                 
                                                                 no_mask_44[3][0]                 
                                                                 no_mask_44[4][0]                 
                                                                 no_mask_44[5][0]                 
__________________________________________________________________________________________________
concatenate_37 (Concatenate)    (None, 2)            0           no_mask_45[0][0]                 
                                                                 no_mask_45[1][0]                 
__________________________________________________________________________________________________
concatenate_40 (Concatenate)    (None, 122)          0           no_mask_50[0][0]                 
                                                                 no_mask_50[1][0]                 
__________________________________________________________________________________________________
linear_5 (Linear)               (None, 1)            2           concatenate_36[0][0]             
                                                                 concatenate_37[0][0]             
__________________________________________________________________________________________________
dnn_14 (DNN)                    (None, 4)            492         concatenate_40[0][0]             
__________________________________________________________________________________________________
no_mask_46 (NoMask)             (None, 1)            0           linear_5[0][0]                   
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 1)            4           dnn_14[0][0]                     
__________________________________________________________________________________________________
add_17 (Add)                    (None, 1)            0           no_mask_46[0][0]                 
                                                                 dense_7[0][0]                    
__________________________________________________________________________________________________
prediction_layer_8 (PredictionL (None, 1)            1           add_17[0][0]                     
==================================================================================================
Total params: 715
Trainable params: 715
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2570 - binary_crossentropy: 0.7074500/500 [==============================] - 5s 10ms/sample - loss: 0.2511 - binary_crossentropy: 0.6954 - val_loss: 0.2549 - val_binary_crossentropy: 0.7029

  #### metrics   #################################################### 
{'MSE': 0.25243973829676314}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_8"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_15 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_60 (Sequ (None, 1, 4)         0           weighted_sequence_layer_15[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_61 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_62 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_63 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
senet_layer (SENETLayer)        [(None, 1, 4), (None 24          sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_60[0][0]  
                                                                 sequence_pooling_layer_61[0][0]  
                                                                 sequence_pooling_layer_62[0][0]  
                                                                 sequence_pooling_layer_63[0][0]  
__________________________________________________________________________________________________
bilinear_interaction (BilinearI (None, 1, 60)        16          senet_layer[0][0]                
                                                                 senet_layer[0][1]                
                                                                 senet_layer[0][2]                
                                                                 senet_layer[0][3]                
                                                                 senet_layer[0][4]                
                                                                 senet_layer[0][5]                
__________________________________________________________________________________________________
bilinear_interaction_1 (Bilinea (None, 1, 60)        16          sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_60[0][0]  
                                                                 sequence_pooling_layer_61[0][0]  
                                                                 sequence_pooling_layer_62[0][0]  
                                                                 sequence_pooling_layer_63[0][0]  
__________________________________________________________________________________________________
no_mask_47 (NoMask)             (None, 1, 60)        0           bilinear_interaction[0][0]       
                                                                 bilinear_interaction_1[0][0]     
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
concatenate_38 (Concatenate)    (None, 1, 120)       0           no_mask_47[0][0]                 
                                                                 no_mask_47[1][0]                 
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
flatten_19 (Flatten)            (None, 120)          0           concatenate_38[0][0]             
__________________________________________________________________________________________________
no_mask_49 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
weighted_sequence_layer_16 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_64 (Sequ (None, 1, 1)         0           weighted_sequence_layer_16[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_65 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_66 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_67 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_20 (Flatten)            (None, 120)          0           no_mask_48[0][0]                 
__________________________________________________________________________________________________
flatten_21 (Flatten)            (None, 2)            0           concatenate_39[0][0]             
__________________________________________________________________________________________________
no_mask_44 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_64[0][0]  
                                                                 sequence_pooling_layer_65[0][0]  
                                                                 sequence_pooling_layer_66[0][0]  
                                                                 sequence_pooling_layer_67[0][0]  
__________________________________________________________________________________________________
no_mask_45 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
no_mask_50 (NoMask)             multiple             0           flatten_20[0][0]                 
                                                                 flatten_21[0][0]                 
__________________________________________________________________________________________________
concatenate_36 (Concatenate)    (None, 1, 6)         0           no_mask_44[0][0]                 
                                                                 no_mask_44[1][0]                 
                                                                 no_mask_44[2][0]                 
                                                                 no_mask_44[3][0]                 
                                                                 no_mask_44[4][0]                 
                                                                 no_mask_44[5][0]                 
__________________________________________________________________________________________________
concatenate_37 (Concatenate)    (None, 2)            0           no_mask_45[0][0]                 
                                                                 no_mask_45[1][0]                 
__________________________________________________________________________________________________
concatenate_40 (Concatenate)    (None, 122)          0           no_mask_50[0][0]                 
                                                                 no_mask_50[1][0]                 
__________________________________________________________________________________________________
linear_5 (Linear)               (None, 1)            2           concatenate_36[0][0]             
                                                                 concatenate_37[0][0]             
__________________________________________________________________________________________________
dnn_14 (DNN)                    (None, 4)            492         concatenate_40[0][0]             
__________________________________________________________________________________________________
no_mask_46 (NoMask)             (None, 1)            0           linear_5[0][0]                   
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 1)            4           dnn_14[0][0]                     
__________________________________________________________________________________________________
add_17 (Add)                    (None, 1)            0           no_mask_46[0][0]                 
                                                                 dense_7[0][0]                    
__________________________________________________________________________________________________
prediction_layer_8 (PredictionL (None, 1)            1           add_17[0][0]                     
==================================================================================================
Total params: 715
Trainable params: 715
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'FLEN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'FLEN', 'embedding_size': 2, 'sparse_feature_num': 6, 'dense_feature_num': 6, 'use_group': True} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_FLEN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_9"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 2)         4           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_3 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_4 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_5 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_18 (Wei (None, 3, 2)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 2)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_2 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_3 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_4 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_5 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         18          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         6           sparse_feature_5[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_72 (Sequ (None, 1, 2)         0           weighted_sequence_layer_18[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_73 (Sequ (None, 1, 2)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_74 (Sequ (None, 1, 2)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_75 (Sequ (None, 1, 2)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_61 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_3[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_4[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sparse_emb_sparse_feature_5[0][0]
                                                                 sequence_pooling_layer_72[0][0]  
                                                                 sequence_pooling_layer_73[0][0]  
                                                                 sequence_pooling_layer_74[0][0]  
                                                                 sequence_pooling_layer_75[0][0]  
__________________________________________________________________________________________________
no_mask_62 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
                                                                 dense_feature_3[0][0]            
                                                                 dense_feature_4[0][0]            
                                                                 dense_feature_5[0][0]            
__________________________________________________________________________________________________
concatenate_50 (Concatenate)    (None, 1, 20)        0           no_mask_61[0][0]                 
                                                                 no_mask_61[1][0]                 
                                                                 no_mask_61[2][0]                 
                                                                 no_mask_61[3][0]                 
                                                                 no_mask_61[4][0]                 
                                                                 no_mask_61[5][0]                 
                                                                 no_mask_61[6][0]                 
                                                                 no_mask_61[7][0]                 
                                                                 no_mask_61[8][0]                 
                                                                 no_mask_61[9][0]                 
__________________________________________________________________________________________________
concatenate_51 (Concatenate)    (None, 6)            0           no_mask_62[0][0]                 
                                                                 no_mask_62[1][0]                 
                                                                 no_mask_62[2][0]                 
                                                                 no_mask_62[3][0]                 
                                                                 no_mask_62[4][0]                 
                                                                 no_mask_62[5][0]                 
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
flatten_22 (Flatten)            (None, 20)           0           concatenate_50[0][0]             
__________________________________________________________________________________________________
flatten_23 (Flatten)            (None, 6)            0           concatenate_51[0][0]             
__________________________________________________________________________________________________
weighted_sequence_layer_19 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_57 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_3[0][0]
__________________________________________________________________________________________________
no_mask_58 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_4[0][0]
__________________________________________________________________________________________________
no_mask_59 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_2[0][0]
                                                                 sparse_emb_sparse_feature_5[0][0]
__________________________________________________________________________________________________
no_mask_60 (NoMask)             (None, 1, 2)         0           sequence_pooling_layer_72[0][0]  
                                                                 sequence_pooling_layer_73[0][0]  
                                                                 sequence_pooling_layer_74[0][0]  
                                                                 sequence_pooling_layer_75[0][0]  
__________________________________________________________________________________________________
no_mask_63 (NoMask)             multiple             0           flatten_22[0][0]                 
                                                                 flatten_23[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_5[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_76 (Sequ (None, 1, 1)         0           weighted_sequence_layer_19[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_77 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_78 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_79 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
concatenate_46 (Concatenate)    (None, 2, 2)         0           no_mask_57[0][0]                 
                                                                 no_mask_57[1][0]                 
__________________________________________________________________________________________________
concatenate_47 (Concatenate)    (None, 2, 2)         0           no_mask_58[0][0]                 
                                                                 no_mask_58[1][0]                 
__________________________________________________________________________________________________
concatenate_48 (Concatenate)    (None, 2, 2)         0           no_mask_59[0][0]                 
                                                                 no_mask_59[1][0]                 
__________________________________________________________________________________________________
concatenate_49 (Concatenate)    (None, 4, 2)         0           no_mask_60[0][0]                 
                                                                 no_mask_60[1][0]                 
                                                                 no_mask_60[2][0]                 
                                                                 no_mask_60[3][0]                 
__________________________________________________________________________________________________
concatenate_52 (Concatenate)    (None, 26)           0           no_mask_63[0][0]                 
                                                                 no_mask_63[1][0]                 
__________________________________________________________________________________________________
no_mask_54 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_76[0][0]  
                                                                 sequence_pooling_layer_77[0][0]  
                                                                 sequence_pooling_layer_78[0][0]  
                                                                 sequence_pooling_layer_79[0][0]  
__________________________________________________________________________________________________
no_mask_55 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
                                                                 dense_feature_3[0][0]            
                                                                 dense_feature_4[0][0]            
                                                                 dense_feature_5[0][0]            
__________________________________________________________________________________________________
field_wise_bi_interaction (Fiel (None, 2)            14          concatenate_46[0][0]             
                                                                 concatenate_47[0][0]             
                                                                 concatenate_48[0][0]             
                                                                 concatenate_49[0][0]             
__________________________________________________________________________________________________
dnn_15 (DNN)                    (None, 3)            81          concatenate_52[0][0]             
__________________________________________________________________________________________________
concatenate_44 (Concatenate)    (None, 1, 10)        0           no_mask_54[0][0]                 
                                                                 no_mask_54[1][0]                 
                                                                 no_mask_54[2][0]                 
                                                                 no_mask_54[3][0]                 
                                                                 no_mask_54[4][0]                 
                                                                 no_mask_54[5][0]                 
                                                                 no_mask_54[6][0]                 
                                                                 no_mask_54[7][0]                 
                                                                 no_mask_54[8][0]                 
                                                                 no_mask_54[9][0]                 
__________________________________________________________________________________________________
concatenate_45 (Concatenate)    (None, 6)            0           no_mask_55[0][0]                 
                                                                 no_mask_55[1][0]                 
                                                                 no_mask_55[2][0]                 
                                                                 no_mask_55[3][0]                 
                                                                 no_mask_55[4][0]                 
                                                                 no_mask_55[5][0]                 
__________________________________________________________________________________________________
no_mask_64 (NoMask)             multiple             0           field_wise_bi_interaction[0][0]  
                                                                 dnn_15[0][0]                     
__________________________________________________________________________________________________
linear_6 (Linear)               (None, 1)            6           concatenate_44[0][0]             
                                                                 concatenate_45[0][0]             
__________________________________________________________________________________________________
concatenate_53 (Concatenate)    (None, 5)            0           no_mask_64[0][0]                 
                                                                 no_mask_64[1][0]                 
__________________________________________________________________________________________________
no_mask_56 (NoMask)             (None, 1)            0           linear_6[0][0]                   
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 1)            5           concatenate_53[0][0]             
__________________________________________________________________________________________________
add_20 (Add)                    (None, 1)            0           no_mask_56[0][0]                 
                                                                 dense_8[0][0]                    
__________________________________________________________________________________________________
prediction_layer_9 (PredictionL (None, 1)            1           add_20[0][0]                     
==================================================================================================
Total params: 251
Trainable params: 251
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2648 - binary_crossentropy: 1.3714500/500 [==============================] - 5s 10ms/sample - loss: 0.2920 - binary_crossentropy: 1.5850 - val_loss: 0.2704 - val_binary_crossentropy: 1.3053

  #### metrics   #################################################### 
{'MSE': 0.2794380401287787}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_9"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 2)         4           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_3 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_4 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_5 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_18 (Wei (None, 3, 2)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 2)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_2 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_3 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_4 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_5 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         18          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         6           sparse_feature_5[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_72 (Sequ (None, 1, 2)         0           weighted_sequence_layer_18[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_73 (Sequ (None, 1, 2)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_74 (Sequ (None, 1, 2)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_75 (Sequ (None, 1, 2)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_61 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_3[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_4[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sparse_emb_sparse_feature_5[0][0]
                                                                 sequence_pooling_layer_72[0][0]  
                                                                 sequence_pooling_layer_73[0][0]  
                                                                 sequence_pooling_layer_74[0][0]  
                                                                 sequence_pooling_layer_75[0][0]  
__________________________________________________________________________________________________
no_mask_62 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
                                                                 dense_feature_3[0][0]            
                                                                 dense_feature_4[0][0]            
                                                                 dense_feature_5[0][0]            
__________________________________________________________________________________________________
concatenate_50 (Concatenate)    (None, 1, 20)        0           no_mask_61[0][0]                 
                                                                 no_mask_61[1][0]                 
                                                                 no_mask_61[2][0]                 
                                                                 no_mask_61[3][0]                 
                                                                 no_mask_61[4][0]                 
                                                                 no_mask_61[5][0]                 
                                                                 no_mask_61[6][0]                 
                                                                 no_mask_61[7][0]                 
                                                                 no_mask_61[8][0]                 
                                                                 no_mask_61[9][0]                 
__________________________________________________________________________________________________
concatenate_51 (Concatenate)    (None, 6)            0           no_mask_62[0][0]                 
                                                                 no_mask_62[1][0]                 
                                                                 no_mask_62[2][0]                 
                                                                 no_mask_62[3][0]                 
                                                                 no_mask_62[4][0]                 
                                                                 no_mask_62[5][0]                 
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
flatten_22 (Flatten)            (None, 20)           0           concatenate_50[0][0]             
__________________________________________________________________________________________________
flatten_23 (Flatten)            (None, 6)            0           concatenate_51[0][0]             
__________________________________________________________________________________________________
weighted_sequence_layer_19 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_57 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_3[0][0]
__________________________________________________________________________________________________
no_mask_58 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_4[0][0]
__________________________________________________________________________________________________
no_mask_59 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_2[0][0]
                                                                 sparse_emb_sparse_feature_5[0][0]
__________________________________________________________________________________________________
no_mask_60 (NoMask)             (None, 1, 2)         0           sequence_pooling_layer_72[0][0]  
                                                                 sequence_pooling_layer_73[0][0]  
                                                                 sequence_pooling_layer_74[0][0]  
                                                                 sequence_pooling_layer_75[0][0]  
__________________________________________________________________________________________________
no_mask_63 (NoMask)             multiple             0           flatten_22[0][0]                 
                                                                 flatten_23[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_5[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_76 (Sequ (None, 1, 1)         0           weighted_sequence_layer_19[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_77 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_78 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_79 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
concatenate_46 (Concatenate)    (None, 2, 2)         0           no_mask_57[0][0]                 
                                                                 no_mask_57[1][0]                 
__________________________________________________________________________________________________
concatenate_47 (Concatenate)    (None, 2, 2)         0           no_mask_58[0][0]                 
                                                                 no_mask_58[1][0]                 
__________________________________________________________________________________________________
concatenate_48 (Concatenate)    (None, 2, 2)         0           no_mask_59[0][0]                 
                                                                 no_mask_59[1][0]                 
__________________________________________________________________________________________________
concatenate_49 (Concatenate)    (None, 4, 2)         0           no_mask_60[0][0]                 
                                                                 no_mask_60[1][0]                 
                                                                 no_mask_60[2][0]                 
                                                                 no_mask_60[3][0]                 
__________________________________________________________________________________________________
concatenate_52 (Concatenate)    (None, 26)           0           no_mask_63[0][0]                 
                                                                 no_mask_63[1][0]                 
__________________________________________________________________________________________________
no_mask_54 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_76[0][0]  
                                                                 sequence_pooling_layer_77[0][0]  
                                                                 sequence_pooling_layer_78[0][0]  
                                                                 sequence_pooling_layer_79[0][0]  
__________________________________________________________________________________________________
no_mask_55 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
                                                                 dense_feature_3[0][0]            
                                                                 dense_feature_4[0][0]            
                                                                 dense_feature_5[0][0]            
__________________________________________________________________________________________________
field_wise_bi_interaction (Fiel (None, 2)            14          concatenate_46[0][0]             
                                                                 concatenate_47[0][0]             
                                                                 concatenate_48[0][0]             
                                                                 concatenate_49[0][0]             
__________________________________________________________________________________________________
dnn_15 (DNN)                    (None, 3)            81          concatenate_52[0][0]             
__________________________________________________________________________________________________
concatenate_44 (Concatenate)    (None, 1, 10)        0           no_mask_54[0][0]                 
                                                                 no_mask_54[1][0]                 
                                                                 no_mask_54[2][0]                 
                                                                 no_mask_54[3][0]                 
                                                                 no_mask_54[4][0]                 
                                                                 no_mask_54[5][0]                 
                                                                 no_mask_54[6][0]                 
                                                                 no_mask_54[7][0]                 
                                                                 no_mask_54[8][0]                 
                                                                 no_mask_54[9][0]                 
__________________________________________________________________________________________________
concatenate_45 (Concatenate)    (None, 6)            0           no_mask_55[0][0]                 
                                                                 no_mask_55[1][0]                 
                                                                 no_mask_55[2][0]                 
                                                                 no_mask_55[3][0]                 
                                                                 no_mask_55[4][0]                 
                                                                 no_mask_55[5][0]                 
__________________________________________________________________________________________________
no_mask_64 (NoMask)             multiple             0           field_wise_bi_interaction[0][0]  
                                                                 dnn_15[0][0]                     
__________________________________________________________________________________________________
linear_6 (Linear)               (None, 1)            6           concatenate_44[0][0]             
                                                                 concatenate_45[0][0]             
__________________________________________________________________________________________________
concatenate_53 (Concatenate)    (None, 5)            0           no_mask_64[0][0]                 
                                                                 no_mask_64[1][0]                 
__________________________________________________________________________________________________
no_mask_56 (NoMask)             (None, 1)            0           linear_6[0][0]                   
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 1)            5           concatenate_53[0][0]             
__________________________________________________________________________________________________
add_20 (Add)                    (None, 1)            0           no_mask_56[0][0]                 
                                                                 dense_8[0][0]                    
__________________________________________________________________________________________________
prediction_layer_9 (PredictionL (None, 1)            1           add_20[0][0]                     
==================================================================================================
Total params: 251
Trainable params: 251
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'FNN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'FNN', 'sparse_feature_num': 1, 'dense_feature_num': 1} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_FNN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_10"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_84 (Sequ (None, 1, 4)         0           weighted_sequence_layer_21[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_85 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_86 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_87 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_68 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_84[0][0]  
                                                                 sequence_pooling_layer_85[0][0]  
                                                                 sequence_pooling_layer_86[0][0]  
                                                                 sequence_pooling_layer_87[0][0]  
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
concatenate_55 (Concatenate)    (None, 1, 20)        0           no_mask_68[0][0]                 
                                                                 no_mask_68[1][0]                 
                                                                 no_mask_68[2][0]                 
                                                                 no_mask_68[3][0]                 
                                                                 no_mask_68[4][0]                 
__________________________________________________________________________________________________
no_mask_69 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
weighted_sequence_layer_22 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_88 (Sequ (None, 1, 1)         0           weighted_sequence_layer_22[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_89 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_90 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_91 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
no_mask_70 (NoMask)             multiple             0           flatten_24[0][0]                 
                                                                 flatten_25[0][0]                 
__________________________________________________________________________________________________
no_mask_65 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_88[0][0]  
                                                                 sequence_pooling_layer_89[0][0]  
                                                                 sequence_pooling_layer_90[0][0]  
                                                                 sequence_pooling_layer_91[0][0]  
__________________________________________________________________________________________________
concatenate_56 (Concatenate)    (None, 21)           0           no_mask_70[0][0]                 
                                                                 no_mask_70[1][0]                 
__________________________________________________________________________________________________
concatenate_54 (Concatenate)    (None, 1, 5)         0           no_mask_65[0][0]                 
                                                                 no_mask_65[1][0]                 
                                                                 no_mask_65[2][0]                 
                                                                 no_mask_65[3][0]                 
                                                                 no_mask_65[4][0]                 
__________________________________________________________________________________________________
no_mask_66 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
dnn_16 (DNN)                    (None, 32)           1760        concatenate_56[0][0]             
__________________________________________________________________________________________________
linear_7 (Linear)               (None, 1)            1           concatenate_54[0][0]             
                                                                 no_mask_66[0][0]                 
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 1)            32          dnn_16[0][0]                     
__________________________________________________________________________________________________
no_mask_67 (NoMask)             (None, 1)            0           linear_7[0][0]                   
__________________________________________________________________________________________________
add_23 (Add)                    (None, 1)            0           dense_9[0][0]                    
                                                                 no_mask_67[0][0]                 
__________________________________________________________________________________________________
prediction_layer_10 (Prediction (None, 1)            1           add_23[0][0]                     
==================================================================================================
Total params: 1,894
Trainable params: 1,894
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2609 - binary_crossentropy: 0.7158500/500 [==============================] - 5s 10ms/sample - loss: 0.2550 - binary_crossentropy: 0.7033 - val_loss: 0.2526 - val_binary_crossentropy: 0.6983

  #### metrics   #################################################### 
{'MSE': 0.2516765959090291}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_10"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_84 (Sequ (None, 1, 4)         0           weighted_sequence_layer_21[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_85 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_86 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_87 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_68 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_84[0][0]  
                                                                 sequence_pooling_layer_85[0][0]  
                                                                 sequence_pooling_layer_86[0][0]  
                                                                 sequence_pooling_layer_87[0][0]  
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
concatenate_55 (Concatenate)    (None, 1, 20)        0           no_mask_68[0][0]                 
                                                                 no_mask_68[1][0]                 
                                                                 no_mask_68[2][0]                 
                                                                 no_mask_68[3][0]                 
                                                                 no_mask_68[4][0]                 
__________________________________________________________________________________________________
no_mask_69 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
weighted_sequence_layer_22 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_88 (Sequ (None, 1, 1)         0           weighted_sequence_layer_22[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_89 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_90 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_91 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
no_mask_70 (NoMask)             multiple             0           flatten_24[0][0]                 
                                                                 flatten_25[0][0]                 
__________________________________________________________________________________________________
no_mask_65 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_88[0][0]  
                                                                 sequence_pooling_layer_89[0][0]  
                                                                 sequence_pooling_layer_90[0][0]  
                                                                 sequence_pooling_layer_91[0][0]  
__________________________________________________________________________________________________
concatenate_56 (Concatenate)    (None, 21)           0           no_mask_70[0][0]                 
                                                                 no_mask_70[1][0]                 
__________________________________________________________________________________________________
concatenate_54 (Concatenate)    (None, 1, 5)         0           no_mask_65[0][0]                 
                                                                 no_mask_65[1][0]                 
                                                                 no_mask_65[2][0]                 
                                                                 no_mask_65[3][0]                 
                                                                 no_mask_65[4][0]                 
__________________________________________________________________________________________________
no_mask_66 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
dnn_16 (DNN)                    (None, 32)           1760        concatenate_56[0][0]             
__________________________________________________________________________________________________
linear_7 (Linear)               (None, 1)            1           concatenate_54[0][0]             
                                                                 no_mask_66[0][0]                 
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 1)            32          dnn_16[0][0]                     
__________________________________________________________________________________________________
no_mask_67 (NoMask)             (None, 1)            0           linear_7[0][0]                   
__________________________________________________________________________________________________
add_23 (Add)                    (None, 1)            0           dense_9[0][0]                    
                                                                 no_mask_67[0][0]                 
__________________________________________________________________________________________________
prediction_layer_10 (Prediction (None, 1)            1           add_23[0][0]                     
==================================================================================================
Total params: 1,894
Trainable params: 1,894
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'MLR', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'MLR', 'sparse_feature_num': 0, 'dense_feature_num': 2, 'prefix': 'region'} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_MLR.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_11"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
regionweighted_seq (InputLayer) [(None, 3)]          0                                            
__________________________________________________________________________________________________
region_10sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
regionweighted_seq_seq_length ( [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionweight (InputLayer)       [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
regionsequence_sum (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 8)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 1)]          0                                            
__________________________________________________________________________________________________
region_20sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_24 (Wei (None, 3, 1)         0           region_10sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
sequence_pooling_layer_96 (Sequ (None, 1, 1)         0           weighted_sequence_layer_24[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_97 (Sequ (None, 1, 1)         0           region_10sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_98 (Sequ (None, 1, 1)         0           region_10sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_99 (Sequ (None, 1, 1)         0           region_10sparse_seq_emb_regionseq
__________________________________________________________________________________________________
regiondense_feature_0 (InputLay [(None, 1)]          0                                            
__________________________________________________________________________________________________
regiondense_feature_1 (InputLay [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_pooling_layer_104 (Seq (None, 1, 1)         0           weighted_sequence_layer_26[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_105 (Seq (None, 1, 1)         0           region_20sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_106 (Seq (None, 1, 1)         0           region_20sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_107 (Seq (None, 1, 1)         0           region_20sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_112 (Seq (None, 1, 1)         0           weighted_sequence_layer_28[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_113 (Seq (None, 1, 1)         0           region_30sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_114 (Seq (None, 1, 1)         0           region_30sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_115 (Seq (None, 1, 1)         0           region_30sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_120 (Seq (None, 1, 1)         0           weighted_sequence_layer_30[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_121 (Seq (None, 1, 1)         0           region_40sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_122 (Seq (None, 1, 1)         0           region_40sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_123 (Seq (None, 1, 1)         0           region_40sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_128 (Seq (None, 1, 1)         0           weighted_sequence_layer_32[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_129 (Seq (None, 1, 1)         0           learner_10sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_130 (Seq (None, 1, 1)         0           learner_10sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_131 (Seq (None, 1, 1)         0           learner_10sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_136 (Seq (None, 1, 1)         0           weighted_sequence_layer_34[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_137 (Seq (None, 1, 1)         0           learner_20sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_138 (Seq (None, 1, 1)         0           learner_20sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_139 (Seq (None, 1, 1)         0           learner_20sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_144 (Seq (None, 1, 1)         0           weighted_sequence_layer_36[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_145 (Seq (None, 1, 1)         0           learner_30sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_146 (Seq (None, 1, 1)         0           learner_30sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_147 (Seq (None, 1, 1)         0           learner_30sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_152 (Seq (None, 1, 1)         0           weighted_sequence_layer_38[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_153 (Seq (None, 1, 1)         0           learner_40sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_154 (Seq (None, 1, 1)         0           learner_40sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_155 (Seq (None, 1, 1)         0           learner_40sparse_seq_emb_regionse
__________________________________________________________________________________________________
no_mask_71 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_96[0][0]  
                                                                 sequence_pooling_layer_97[0][0]  
                                                                 sequence_pooling_layer_98[0][0]  
                                                                 sequence_pooling_layer_99[0][0]  
__________________________________________________________________________________________________
no_mask_72 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_74 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_104[0][0] 
                                                                 sequence_pooling_layer_105[0][0] 
                                                                 sequence_pooling_layer_106[0][0] 
                                                                 sequence_pooling_layer_107[0][0] 
__________________________________________________________________________________________________
no_mask_75 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_77 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_112[0][0] 
                                                                 sequence_pooling_layer_113[0][0] 
                                                                 sequence_pooling_layer_114[0][0] 
                                                                 sequence_pooling_layer_115[0][0] 
__________________________________________________________________________________________________
no_mask_78 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_80 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_120[0][0] 
                                                                 sequence_pooling_layer_121[0][0] 
                                                                 sequence_pooling_layer_122[0][0] 
                                                                 sequence_pooling_layer_123[0][0] 
__________________________________________________________________________________________________
no_mask_81 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_84 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_128[0][0] 
                                                                 sequence_pooling_layer_129[0][0] 
                                                                 sequence_pooling_layer_130[0][0] 
                                                                 sequence_pooling_layer_131[0][0] 
__________________________________________________________________________________________________
no_mask_85 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_87 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_136[0][0] 
                                                                 sequence_pooling_layer_137[0][0] 
                                                                 sequence_pooling_layer_138[0][0] 
                                                                 sequence_pooling_layer_139[0][0] 
__________________________________________________________________________________________________
no_mask_88 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_90 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_144[0][0] 
                                                                 sequence_pooling_layer_145[0][0] 
                                                                 sequence_pooling_layer_146[0][0] 
                                                                 sequence_pooling_layer_147[0][0] 
__________________________________________________________________________________________________
no_mask_91 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_93 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_152[0][0] 
                                                                 sequence_pooling_layer_153[0][0] 
                                                                 sequence_pooling_layer_154[0][0] 
                                                                 sequence_pooling_layer_155[0][0] 
__________________________________________________________________________________________________
no_mask_94 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
concatenate_57 (Concatenate)    (None, 1, 4)         0           no_mask_71[0][0]                 
                                                                 no_mask_71[1][0]                 
                                                                 no_mask_71[2][0]                 
                                                                 no_mask_71[3][0]                 
__________________________________________________________________________________________________
concatenate_58 (Concatenate)    (None, 2)            0           no_mask_72[0][0]                 
                                                                 no_mask_72[1][0]                 
__________________________________________________________________________________________________
concatenate_59 (Concatenate)    (None, 1, 4)         0           no_mask_74[0][0]                 
                                                                 no_mask_74[1][0]                 
                                                                 no_mask_74[2][0]                 
                                                                 no_mask_74[3][0]                 
__________________________________________________________________________________________________
concatenate_60 (Concatenate)    (None, 2)            0           no_mask_75[0][0]                 
                                                                 no_mask_75[1][0]                 
__________________________________________________________________________________________________
concatenate_61 (Concatenate)    (None, 1, 4)         0           no_mask_77[0][0]                 
                                                                 no_mask_77[1][0]                 
                                                                 no_mask_77[2][0]                 
                                                                 no_mask_77[3][0]                 
__________________________________________________________________________________________________
concatenate_62 (Concatenate)    (None, 2)            0           no_mask_78[0][0]                 
                                                                 no_mask_78[1][0]                 
__________________________________________________________________________________________________
concatenate_63 (Concatenate)    (None, 1, 4)         0           no_mask_80[0][0]                 
                                                                 no_mask_80[1][0]                 
                                                                 no_mask_80[2][0]                 
                                                                 no_mask_80[3][0]                 
__________________________________________________________________________________________________
concatenate_64 (Concatenate)    (None, 2)            0           no_mask_81[0][0]                 
                                                                 no_mask_81[1][0]                 
__________________________________________________________________________________________________
concatenate_66 (Concatenate)    (None, 1, 4)         0           no_mask_84[0][0]                 
                                                                 no_mask_84[1][0]                 
                                                                 no_mask_84[2][0]                 
                                                                 no_mask_84[3][0]                 
__________________________________________________________________________________________________
concatenate_67 (Concatenate)    (None, 2)            0           no_mask_85[0][0]                 
                                                                 no_mask_85[1][0]                 
__________________________________________________________________________________________________
concatenate_68 (Concatenate)    (None, 1, 4)         0           no_mask_87[0][0]                 
                                                                 no_mask_87[1][0]                 
                                                                 no_mask_87[2][0]                 
                                                                 no_mask_87[3][0]                 
__________________________________________________________________________________________________
concatenate_69 (Concatenate)    (None, 2)            0           no_mask_88[0][0]                 
                                                                 no_mask_88[1][0]                 
__________________________________________________________________________________________________
concatenate_70 (Concatenate)    (None, 1, 4)         0           no_mask_90[0][0]                 
                                                                 no_mask_90[1][0]                 
                                                                 no_mask_90[2][0]                 
                                                                 no_mask_90[3][0]                 
__________________________________________________________________________________________________
concatenate_71 (Concatenate)    (None, 2)            0           no_mask_91[0][0]                 
                                                                 no_mask_91[1][0]                 
__________________________________________________________________________________________________
concatenate_72 (Concatenate)    (None, 1, 4)         0           no_mask_93[0][0]                 
                                                                 no_mask_93[1][0]                 
                                                                 no_mask_93[2][0]                 
                                                                 no_mask_93[3][0]                 
__________________________________________________________________________________________________
concatenate_73 (Concatenate)    (None, 2)            0           no_mask_94[0][0]                 
                                                                 no_mask_94[1][0]                 
__________________________________________________________________________________________________
linear_8 (Linear)               (None, 1)            2           concatenate_57[0][0]             
                                                                 concatenate_58[0][0]             
__________________________________________________________________________________________________
linear_9 (Linear)               (None, 1)            2           concatenate_59[0][0]             
                                                                 concatenate_60[0][0]             
__________________________________________________________________________________________________
linear_10 (Linear)              (None, 1)            2           concatenate_61[0][0]             
                                                                 concatenate_62[0][0]             
__________________________________________________________________________________________________
linear_11 (Linear)              (None, 1)            2           concatenate_63[0][0]             
                                                                 concatenate_64[0][0]             
__________________________________________________________________________________________________
linear_12 (Linear)              (None, 1)            2           concatenate_66[0][0]             
                                                                 concatenate_67[0][0]             
__________________________________________________________________________________________________
linear_13 (Linear)              (None, 1)            2           concatenate_68[0][0]             
                                                                 concatenate_69[0][0]             
__________________________________________________________________________________________________
linear_14 (Linear)              (None, 1)            2           concatenate_70[0][0]             
                                                                 concatenate_71[0][0]             
__________________________________________________________________________________________________
linear_15 (Linear)              (None, 1)            2           concatenate_72[0][0]             
                                                                 concatenate_73[0][0]             
__________________________________________________________________________________________________
no_mask_73 (NoMask)             (None, 1)            0           linear_8[0][0]                   
__________________________________________________________________________________________________
no_mask_76 (NoMask)             (None, 1)            0           linear_9[0][0]                   
__________________________________________________________________________________________________
no_mask_79 (NoMask)             (None, 1)            0           linear_10[0][0]                  
__________________________________________________________________________________________________
no_mask_82 (NoMask)             (None, 1)            0           linear_11[0][0]                  
__________________________________________________________________________________________________
no_mask_86 (NoMask)             (None, 1)            0           linear_12[0][0]                  
__________________________________________________________________________________________________
no_mask_89 (NoMask)             (None, 1)            0           linear_13[0][0]                  
__________________________________________________________________________________________________
no_mask_92 (NoMask)             (None, 1)            0           linear_14[0][0]                  
__________________________________________________________________________________________________
no_mask_95 (NoMask)             (None, 1)            0           linear_15[0][0]                  
__________________________________________________________________________________________________
no_mask_83 (NoMask)             (None, 1)            0           no_mask_73[0][0]                 
                                                                 no_mask_76[0][0]                 
                                                                 no_mask_79[0][0]                 
                                                                 no_mask_82[0][0]                 
__________________________________________________________________________________________________
prediction_layer_11 (Prediction (None, 1)            0           no_mask_86[0][0]                 
__________________________________________________________________________________________________
prediction_layer_12 (Prediction (None, 1)            0           no_mask_89[0][0]                 
__________________________________________________________________________________________________
prediction_layer_13 (Prediction (None, 1)            0           no_mask_92[0][0]                 
__________________________________________________________________________________________________
prediction_layer_14 (Prediction (None, 1)            0           no_mask_95[0][0]                 
__________________________________________________________________________________________________
concatenate_65 (Concatenate)    (None, 4)            0           no_mask_83[0][0]                 
                                                                 no_mask_83[1][0]                 
                                                                 no_mask_83[2][0]                 
                                                                 no_mask_83[3][0]                 
__________________________________________________________________________________________________
no_mask_96 (NoMask)             (None, 1)            0           prediction_layer_11[0][0]        
                                                                 prediction_layer_12[0][0]        
                                                                 prediction_layer_13[0][0]        
                                                                 prediction_layer_14[0][0]        
__________________________________________________________________________________________________
activation_40 (Activation)      (None, 4)            0           concatenate_65[0][0]             
__________________________________________________________________________________________________
concatenate_74 (Concatenate)    (None, 4)            0           no_mask_96[0][0]                 
                                                                 no_mask_96[1][0]                 
                                                                 no_mask_96[2][0]                 
                                                                 no_mask_96[3][0]                 
__________________________________________________________________________________________________
dot (Dot)                       (None, 1)            0           activation_40[0][0]              
                                                                 concatenate_74[0][0]             
==================================================================================================
Total params: 160
Trainable params: 160
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 10s - loss: 0.2851 - binary_crossentropy: 1.4115500/500 [==============================] - 6s 13ms/sample - loss: 0.2746 - binary_crossentropy: 1.2341 - val_loss: 0.2978 - val_binary_crossentropy: 1.5972

  #### metrics   #################################################### 
{'MSE': 0.28534470923466543}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_11"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
regionweighted_seq (InputLayer) [(None, 3)]          0                                            
__________________________________________________________________________________________________
region_10sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
regionweighted_seq_seq_length ( [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionweight (InputLayer)       [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
regionsequence_sum (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 8)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 1)]          0                                            
__________________________________________________________________________________________________
region_20sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_24 (Wei (None, 3, 1)         0           region_10sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
sequence_pooling_layer_96 (Sequ (None, 1, 1)         0           weighted_sequence_layer_24[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_97 (Sequ (None, 1, 1)         0           region_10sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_98 (Sequ (None, 1, 1)         0           region_10sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_99 (Sequ (None, 1, 1)         0           region_10sparse_seq_emb_regionseq
__________________________________________________________________________________________________
regiondense_feature_0 (InputLay [(None, 1)]          0                                            
__________________________________________________________________________________________________
regiondense_feature_1 (InputLay [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_pooling_layer_104 (Seq (None, 1, 1)         0           weighted_sequence_layer_26[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_105 (Seq (None, 1, 1)         0           region_20sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_106 (Seq (None, 1, 1)         0           region_20sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_107 (Seq (None, 1, 1)         0           region_20sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_112 (Seq (None, 1, 1)         0           weighted_sequence_layer_28[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_113 (Seq (None, 1, 1)         0           region_30sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_114 (Seq (None, 1, 1)         0           region_30sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_115 (Seq (None, 1, 1)         0           region_30sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_120 (Seq (None, 1, 1)         0           weighted_sequence_layer_30[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_121 (Seq (None, 1, 1)         0           region_40sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_122 (Seq (None, 1, 1)         0           region_40sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_123 (Seq (None, 1, 1)         0           region_40sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_128 (Seq (None, 1, 1)         0           weighted_sequence_layer_32[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_129 (Seq (None, 1, 1)         0           learner_10sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_130 (Seq (None, 1, 1)         0           learner_10sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_131 (Seq (None, 1, 1)         0           learner_10sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_136 (Seq (None, 1, 1)         0           weighted_sequence_layer_34[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_137 (Seq (None, 1, 1)         0           learner_20sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_138 (Seq (None, 1, 1)         0           learner_20sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_139 (Seq (None, 1, 1)         0           learner_20sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_144 (Seq (None, 1, 1)         0           weighted_sequence_layer_36[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_145 (Seq (None, 1, 1)         0           learner_30sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_146 (Seq (None, 1, 1)         0           learner_30sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_147 (Seq (None, 1, 1)         0           learner_30sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_152 (Seq (None, 1, 1)         0           weighted_sequence_layer_38[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_153 (Seq (None, 1, 1)         0           learner_40sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_154 (Seq (None, 1, 1)         0           learner_40sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_155 (Seq (None, 1, 1)         0           learner_40sparse_seq_emb_regionse
__________________________________________________________________________________________________
no_mask_71 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_96[0][0]  
                                                                 sequence_pooling_layer_97[0][0]  
                                                                 sequence_pooling_layer_98[0][0]  
                                                                 sequence_pooling_layer_99[0][0]  
__________________________________________________________________________________________________
no_mask_72 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_74 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_104[0][0] 
                                                                 sequence_pooling_layer_105[0][0] 
                                                                 sequence_pooling_layer_106[0][0] 
                                                                 sequence_pooling_layer_107[0][0] 
__________________________________________________________________________________________________
no_mask_75 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_77 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_112[0][0] 
                                                                 sequence_pooling_layer_113[0][0] 
                                                                 sequence_pooling_layer_114[0][0] 
                                                                 sequence_pooling_layer_115[0][0] 
__________________________________________________________________________________________________
no_mask_78 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_80 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_120[0][0] 
                                                                 sequence_pooling_layer_121[0][0] 
                                                                 sequence_pooling_layer_122[0][0] 
                                                                 sequence_pooling_layer_123[0][0] 
__________________________________________________________________________________________________
no_mask_81 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_84 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_128[0][0] 
                                                                 sequence_pooling_layer_129[0][0] 
                                                                 sequence_pooling_layer_130[0][0] 
                                                                 sequence_pooling_layer_131[0][0] 
__________________________________________________________________________________________________
no_mask_85 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_87 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_136[0][0] 
                                                                 sequence_pooling_layer_137[0][0] 
                                                                 sequence_pooling_layer_138[0][0] 
                                                                 sequence_pooling_layer_139[0][0] 
__________________________________________________________________________________________________
no_mask_88 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_90 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_144[0][0] 
                                                                 sequence_pooling_layer_145[0][0] 
                                                                 sequence_pooling_layer_146[0][0] 
                                                                 sequence_pooling_layer_147[0][0] 
__________________________________________________________________________________________________
no_mask_91 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_93 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_152[0][0] 
                                                                 sequence_pooling_layer_153[0][0] 
                                                                 sequence_pooling_layer_154[0][0] 
                                                                 sequence_pooling_layer_155[0][0] 
__________________________________________________________________________________________________
no_mask_94 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
concatenate_57 (Concatenate)    (None, 1, 4)         0           no_mask_71[0][0]                 
                                                                 no_mask_71[1][0]                 
                                                                 no_mask_71[2][0]                 
                                                                 no_mask_71[3][0]                 
__________________________________________________________________________________________________
concatenate_58 (Concatenate)    (None, 2)            0           no_mask_72[0][0]                 
                                                                 no_mask_72[1][0]                 
__________________________________________________________________________________________________
concatenate_59 (Concatenate)    (None, 1, 4)         0           no_mask_74[0][0]                 
                                                                 no_mask_74[1][0]                 
                                                                 no_mask_74[2][0]                 
                                                                 no_mask_74[3][0]                 
__________________________________________________________________________________________________
concatenate_60 (Concatenate)    (None, 2)            0           no_mask_75[0][0]                 
                                                                 no_mask_75[1][0]                 
__________________________________________________________________________________________________
concatenate_61 (Concatenate)    (None, 1, 4)         0           no_mask_77[0][0]                 
                                                                 no_mask_77[1][0]                 
                                                                 no_mask_77[2][0]                 
                                                                 no_mask_77[3][0]                 
__________________________________________________________________________________________________
concatenate_62 (Concatenate)    (None, 2)            0           no_mask_78[0][0]                 
                                                                 no_mask_78[1][0]                 
__________________________________________________________________________________________________
concatenate_63 (Concatenate)    (None, 1, 4)         0           no_mask_80[0][0]                 
                                                                 no_mask_80[1][0]                 
                                                                 no_mask_80[2][0]                 
                                                                 no_mask_80[3][0]                 
__________________________________________________________________________________________________
concatenate_64 (Concatenate)    (None, 2)            0           no_mask_81[0][0]                 
                                                                 no_mask_81[1][0]                 
__________________________________________________________________________________________________
concatenate_66 (Concatenate)    (None, 1, 4)         0           no_mask_84[0][0]                 
                                                                 no_mask_84[1][0]                 
                                                                 no_mask_84[2][0]                 
                                                                 no_mask_84[3][0]                 
__________________________________________________________________________________________________
concatenate_67 (Concatenate)    (None, 2)            0           no_mask_85[0][0]                 
                                                                 no_mask_85[1][0]                 
__________________________________________________________________________________________________
concatenate_68 (Concatenate)    (None, 1, 4)         0           no_mask_87[0][0]                 
                                                                 no_mask_87[1][0]                 
                                                                 no_mask_87[2][0]                 
                                                                 no_mask_87[3][0]                 
__________________________________________________________________________________________________
concatenate_69 (Concatenate)    (None, 2)            0           no_mask_88[0][0]                 
                                                                 no_mask_88[1][0]                 
__________________________________________________________________________________________________
concatenate_70 (Concatenate)    (None, 1, 4)         0           no_mask_90[0][0]                 
                                                                 no_mask_90[1][0]                 
                                                                 no_mask_90[2][0]                 
                                                                 no_mask_90[3][0]                 
__________________________________________________________________________________________________
concatenate_71 (Concatenate)    (None, 2)            0           no_mask_91[0][0]                 
                                                                 no_mask_91[1][0]                 
__________________________________________________________________________________________________
concatenate_72 (Concatenate)    (None, 1, 4)         0           no_mask_93[0][0]                 
                                                                 no_mask_93[1][0]                 
                                                                 no_mask_93[2][0]                 
                                                                 no_mask_93[3][0]                 
__________________________________________________________________________________________________
concatenate_73 (Concatenate)    (None, 2)            0           no_mask_94[0][0]                 
                                                                 no_mask_94[1][0]                 
__________________________________________________________________________________________________
linear_8 (Linear)               (None, 1)            2           concatenate_57[0][0]             
                                                                 concatenate_58[0][0]             
__________________________________________________________________________________________________
linear_9 (Linear)               (None, 1)            2           concatenate_59[0][0]             
                                                                 concatenate_60[0][0]             
__________________________________________________________________________________________________
linear_10 (Linear)              (None, 1)            2           concatenate_61[0][0]             
                                                                 concatenate_62[0][0]             
__________________________________________________________________________________________________
linear_11 (Linear)              (None, 1)            2           concatenate_63[0][0]             
                                                                 concatenate_64[0][0]             
__________________________________________________________________________________________________
linear_12 (Linear)              (None, 1)            2           concatenate_66[0][0]             
                                                                 concatenate_67[0][0]             
__________________________________________________________________________________________________
linear_13 (Linear)              (None, 1)            2           concatenate_68[0][0]             
                                                                 concatenate_69[0][0]             
__________________________________________________________________________________________________
linear_14 (Linear)              (None, 1)            2           concatenate_70[0][0]             
                                                                 concatenate_71[0][0]             
__________________________________________________________________________________________________
linear_15 (Linear)              (None, 1)            2           concatenate_72[0][0]             
                                                                 concatenate_73[0][0]             
__________________________________________________________________________________________________
no_mask_73 (NoMask)             (None, 1)            0           linear_8[0][0]                   
__________________________________________________________________________________________________
no_mask_76 (NoMask)             (None, 1)            0           linear_9[0][0]                   
__________________________________________________________________________________________________
no_mask_79 (NoMask)             (None, 1)            0           linear_10[0][0]                  
__________________________________________________________________________________________________
no_mask_82 (NoMask)             (None, 1)            0           linear_11[0][0]                  
__________________________________________________________________________________________________
no_mask_86 (NoMask)             (None, 1)            0           linear_12[0][0]                  
__________________________________________________________________________________________________
no_mask_89 (NoMask)             (None, 1)            0           linear_13[0][0]                  
__________________________________________________________________________________________________
no_mask_92 (NoMask)             (None, 1)            0           linear_14[0][0]                  
__________________________________________________________________________________________________
no_mask_95 (NoMask)             (None, 1)            0           linear_15[0][0]                  
__________________________________________________________________________________________________
no_mask_83 (NoMask)             (None, 1)            0           no_mask_73[0][0]                 
                                                                 no_mask_76[0][0]                 
                                                                 no_mask_79[0][0]                 
                                                                 no_mask_82[0][0]                 
__________________________________________________________________________________________________
prediction_layer_11 (Prediction (None, 1)            0           no_mask_86[0][0]                 
__________________________________________________________________________________________________
prediction_layer_12 (Prediction (None, 1)            0           no_mask_89[0][0]                 
__________________________________________________________________________________________________
prediction_layer_13 (Prediction (None, 1)            0           no_mask_92[0][0]                 
__________________________________________________________________________________________________
prediction_layer_14 (Prediction (None, 1)            0           no_mask_95[0][0]                 
__________________________________________________________________________________________________
concatenate_65 (Concatenate)    (None, 4)            0           no_mask_83[0][0]                 
                                                                 no_mask_83[1][0]                 
                                                                 no_mask_83[2][0]                 
                                                                 no_mask_83[3][0]                 
__________________________________________________________________________________________________
no_mask_96 (NoMask)             (None, 1)            0           prediction_layer_11[0][0]        
                                                                 prediction_layer_12[0][0]        
                                                                 prediction_layer_13[0][0]        
                                                                 prediction_layer_14[0][0]        
__________________________________________________________________________________________________
activation_40 (Activation)      (None, 4)            0           concatenate_65[0][0]             
__________________________________________________________________________________________________
concatenate_74 (Concatenate)    (None, 4)            0           no_mask_96[0][0]                 
                                                                 no_mask_96[1][0]                 
                                                                 no_mask_96[2][0]                 
                                                                 no_mask_96[3][0]                 
__________________________________________________________________________________________________
dot (Dot)                       (None, 1)            0           activation_40[0][0]              
                                                                 concatenate_74[0][0]             
==================================================================================================
Total params: 160
Trainable params: 160
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'NFM', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'NFM', 'sparse_feature_num': 1, 'dense_feature_num': 1} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_NFM.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_12"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_160 (Seq (None, 1, 4)         0           weighted_sequence_layer_40[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_161 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_162 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_163 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_100 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_160[0][0] 
                                                                 sequence_pooling_layer_161[0][0] 
                                                                 sequence_pooling_layer_162[0][0] 
                                                                 sequence_pooling_layer_163[0][0] 
__________________________________________________________________________________________________
concatenate_76 (Concatenate)    (None, 5, 4)         0           no_mask_100[0][0]                
                                                                 no_mask_100[1][0]                
                                                                 no_mask_100[2][0]                
                                                                 no_mask_100[3][0]                
                                                                 no_mask_100[4][0]                
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
bi_interaction_pooling (BiInter (None, 1, 4)         0           concatenate_76[0][0]             
__________________________________________________________________________________________________
weighted_sequence_layer_41 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_164 (Seq (None, 1, 1)         0           weighted_sequence_layer_41[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_165 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_166 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_167 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_26 (Flatten)            (None, 4)            0           no_mask_101[0][0]                
__________________________________________________________________________________________________
flatten_27 (Flatten)            (None, 1)            0           no_mask_102[0][0]                
__________________________________________________________________________________________________
no_mask_97 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_164[0][0] 
                                                                 sequence_pooling_layer_165[0][0] 
                                                                 sequence_pooling_layer_166[0][0] 
                                                                 sequence_pooling_layer_167[0][0] 
__________________________________________________________________________________________________
no_mask_103 (NoMask)            multiple             0           flatten_26[0][0]                 
                                                                 flatten_27[0][0]                 
__________________________________________________________________________________________________
concatenate_75 (Concatenate)    (None, 1, 5)         0           no_mask_97[0][0]                 
                                                                 no_mask_97[1][0]                 
                                                                 no_mask_97[2][0]                 
                                                                 no_mask_97[3][0]                 
                                                                 no_mask_97[4][0]                 
__________________________________________________________________________________________________
no_mask_98 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
concatenate_77 (Concatenate)    (None, 5)            0           no_mask_103[0][0]                
                                                                 no_mask_103[1][0]                
__________________________________________________________________________________________________
linear_16 (Linear)              (None, 1)            1           concatenate_75[0][0]             
                                                                 no_mask_98[0][0]                 
__________________________________________________________________________________________________
dnn_17 (DNN)                    (None, 32)           1248        concatenate_77[0][0]             
__________________________________________________________________________________________________
no_mask_99 (NoMask)             (None, 1)            0           linear_16[0][0]                  
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 1)            32          dnn_17[0][0]                     
__________________________________________________________________________________________________
add_26 (Add)                    (None, 1)            0           no_mask_99[0][0]                 
                                                                 dense_10[0][0]                   
__________________________________________________________________________________________________
prediction_layer_15 (Prediction (None, 1)            1           add_26[0][0]                     
==================================================================================================
Total params: 1,402
Trainable params: 1,402
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.3058 - binary_crossentropy: 0.8198500/500 [==============================] - 6s 12ms/sample - loss: 0.2743 - binary_crossentropy: 0.7496 - val_loss: 0.2728 - val_binary_crossentropy: 0.7416

  #### metrics   #################################################### 
{'MSE': 0.2682297215932996}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_12"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_160 (Seq (None, 1, 4)         0           weighted_sequence_layer_40[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_161 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_162 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_163 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_100 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_160[0][0] 
                                                                 sequence_pooling_layer_161[0][0] 
                                                                 sequence_pooling_layer_162[0][0] 
                                                                 sequence_pooling_layer_163[0][0] 
__________________________________________________________________________________________________
concatenate_76 (Concatenate)    (None, 5, 4)         0           no_mask_100[0][0]                
                                                                 no_mask_100[1][0]                
                                                                 no_mask_100[2][0]                
                                                                 no_mask_100[3][0]                
                                                                 no_mask_100[4][0]                
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
bi_interaction_pooling (BiInter (None, 1, 4)         0           concatenate_76[0][0]             
__________________________________________________________________________________________________
weighted_sequence_layer_41 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_164 (Seq (None, 1, 1)         0           weighted_sequence_layer_41[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_165 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_166 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_167 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_26 (Flatten)            (None, 4)            0           no_mask_101[0][0]                
__________________________________________________________________________________________________
flatten_27 (Flatten)            (None, 1)            0           no_mask_102[0][0]                
__________________________________________________________________________________________________
no_mask_97 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_164[0][0] 
                                                                 sequence_pooling_layer_165[0][0] 
                                                                 sequence_pooling_layer_166[0][0] 
                                                                 sequence_pooling_layer_167[0][0] 
__________________________________________________________________________________________________
no_mask_103 (NoMask)            multiple             0           flatten_26[0][0]                 
                                                                 flatten_27[0][0]                 
__________________________________________________________________________________________________
concatenate_75 (Concatenate)    (None, 1, 5)         0           no_mask_97[0][0]                 
                                                                 no_mask_97[1][0]                 
                                                                 no_mask_97[2][0]                 
                                                                 no_mask_97[3][0]                 
                                                                 no_mask_97[4][0]                 
__________________________________________________________________________________________________
no_mask_98 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
concatenate_77 (Concatenate)    (None, 5)            0           no_mask_103[0][0]                
                                                                 no_mask_103[1][0]                
__________________________________________________________________________________________________
linear_16 (Linear)              (None, 1)            1           concatenate_75[0][0]             
                                                                 no_mask_98[0][0]                 
__________________________________________________________________________________________________
dnn_17 (DNN)                    (None, 32)           1248        concatenate_77[0][0]             
__________________________________________________________________________________________________
no_mask_99 (NoMask)             (None, 1)            0           linear_16[0][0]                  
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 1)            32          dnn_17[0][0]                     
__________________________________________________________________________________________________
add_26 (Add)                    (None, 1)            0           no_mask_99[0][0]                 
                                                                 dense_10[0][0]                   
__________________________________________________________________________________________________
prediction_layer_15 (Prediction (None, 1)            1           add_26[0][0]                     
==================================================================================================
Total params: 1,402
Trainable params: 1,402
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'ONN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'ONN', 'sparse_feature_num': 2, 'dense_feature_num': 2, 'sequence_feature': ('sum', 'mean', 'max'), 'hash_flag': True} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_ONN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_13"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_14 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
hash_15 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_16 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         8           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         8           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_107 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_spars
__________________________________________________________________________________________________
no_mask_108 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_spars
__________________________________________________________________________________________________
no_mask_109 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_seque
__________________________________________________________________________________________________
sequence_pooling_layer_178 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sparse_fe
__________________________________________________________________________________________________
no_mask_110 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_seque
__________________________________________________________________________________________________
sequence_pooling_layer_179 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sparse_f
__________________________________________________________________________________________________
no_mask_111 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_seque
__________________________________________________________________________________________________
sequence_pooling_layer_180 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sparse_fe
__________________________________________________________________________________________________
no_mask_112 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_seque
__________________________________________________________________________________________________
sequence_pooling_layer_181 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sparse_fe
__________________________________________________________________________________________________
no_mask_113 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_seque
__________________________________________________________________________________________________
sequence_pooling_layer_182 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sparse_f
__________________________________________________________________________________________________
no_mask_114 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_seque
__________________________________________________________________________________________________
sequence_pooling_layer_183 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sparse_fe
__________________________________________________________________________________________________
sequence_pooling_layer_184 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sequence_
__________________________________________________________________________________________________
sequence_pooling_layer_185 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sequence
__________________________________________________________________________________________________
sequence_pooling_layer_186 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sequence_
__________________________________________________________________________________________________
sequence_pooling_layer_187 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sequence_
__________________________________________________________________________________________________
sequence_pooling_layer_188 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sequence
__________________________________________________________________________________________________
sequence_pooling_layer_189 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sequence_
__________________________________________________________________________________________________
multiply (Multiply)             (None, 1, 4)         0           no_mask_107[0][0]                
                                                                 no_mask_108[0][0]                
__________________________________________________________________________________________________
multiply_1 (Multiply)           (None, 1, 4)         0           no_mask_109[0][0]                
                                                                 sequence_pooling_layer_178[0][0] 
__________________________________________________________________________________________________
multiply_2 (Multiply)           (None, 1, 4)         0           no_mask_110[0][0]                
                                                                 sequence_pooling_layer_179[0][0] 
__________________________________________________________________________________________________
multiply_3 (Multiply)           (None, 1, 4)         0           no_mask_111[0][0]                
                                                                 sequence_pooling_layer_180[0][0] 
__________________________________________________________________________________________________
multiply_4 (Multiply)           (None, 1, 4)         0           no_mask_112[0][0]                
                                                                 sequence_pooling_layer_181[0][0] 
__________________________________________________________________________________________________
multiply_5 (Multiply)           (None, 1, 4)         0           no_mask_113[0][0]                
                                                                 sequence_pooling_layer_182[0][0] 
__________________________________________________________________________________________________
multiply_6 (Multiply)           (None, 1, 4)         0           no_mask_114[0][0]                
                                                                 sequence_pooling_layer_183[0][0] 
__________________________________________________________________________________________________
multiply_7 (Multiply)           (None, 1, 4)         0           sequence_pooling_layer_184[0][0] 
                                                                 sequence_pooling_layer_185[0][0] 
__________________________________________________________________________________________________
multiply_8 (Multiply)           (None, 1, 4)         0           sequence_pooling_layer_186[0][0] 
                                                                 sequence_pooling_layer_187[0][0] 
__________________________________________________________________________________________________
multiply_9 (Multiply)           (None, 1, 4)         0           sequence_pooling_layer_188[0][0] 
                                                                 sequence_pooling_layer_189[0][0] 
__________________________________________________________________________________________________
no_mask_115 (NoMask)            (None, 1, 4)         0           multiply[0][0]                   
                                                                 multiply_1[0][0]                 
                                                                 multiply_2[0][0]                 
                                                                 multiply_3[0][0]                 
                                                                 multiply_4[0][0]                 
                                                                 multiply_5[0][0]                 
                                                                 multiply_6[0][0]                 
                                                                 multiply_7[0][0]                 
                                                                 multiply_8[0][0]                 
                                                                 multiply_9[0][0]                 
__________________________________________________________________________________________________
concatenate_80 (Concatenate)    (None, 10, 4)        0           no_mask_115[0][0]                
                                                                 no_mask_115[1][0]                
                                                                 no_mask_115[2][0]                
                                                                 no_mask_115[3][0]                
                                                                 no_mask_115[4][0]                
                                                                 no_mask_115[5][0]                
                                                                 no_mask_115[6][0]                
                                                                 no_mask_115[7][0]                
                                                                 no_mask_115[8][0]                
                                                                 no_mask_115[9][0]                
__________________________________________________________________________________________________
flatten_28 (Flatten)            (None, 40)           0           concatenate_80[0][0]             
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 40)           160         flatten_28[0][0]                 
__________________________________________________________________________________________________
no_mask_117 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
no_mask_116 (NoMask)            (None, 40)           0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
concatenate_81 (Concatenate)    (None, 2)            0           no_mask_117[0][0]                
                                                                 no_mask_117[1][0]                
__________________________________________________________________________________________________
hash_10 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
hash_11 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_11[0][0]                    
__________________________________________________________________________________________________
sequence_pooling_layer_172 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_173 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_174 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
no_mask_118 (NoMask)            multiple             0           flatten_29[0][0]                 
                                                                 flatten_30[0][0]                 
__________________________________________________________________________________________________
no_mask_104 (NoMask)            (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_172[0][0] 
                                                                 sequence_pooling_layer_173[0][0] 
                                                                 sequence_pooling_layer_174[0][0] 
__________________________________________________________________________________________________
no_mask_105 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
concatenate_82 (Concatenate)    (None, 42)           0           no_mask_118[0][0]                
                                                                 no_mask_118[1][0]                
__________________________________________________________________________________________________
concatenate_78 (Concatenate)    (None, 1, 5)         0           no_mask_104[0][0]                
                                                                 no_mask_104[1][0]                
                                                                 no_mask_104[2][0]                
                                                                 no_mask_104[3][0]                
                                                                 no_mask_104[4][0]                
__________________________________________________________________________________________________
concatenate_79 (Concatenate)    (None, 2)            0           no_mask_105[0][0]                
                                                                 no_mask_105[1][0]                
__________________________________________________________________________________________________
dnn_18 (DNN)                    (None, 32)           2432        concatenate_82[0][0]             
__________________________________________________________________________________________________
linear_17 (Linear)              (None, 1)            2           concatenate_78[0][0]             
                                                                 concatenate_79[0][0]             
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 1)            32          dnn_18[0][0]                     
__________________________________________________________________________________________________
no_mask_106 (NoMask)            (None, 1)            0           linear_17[0][0]                  
__________________________________________________________________________________________________
add_29 (Add)                    (None, 1)            0           dense_11[0][0]                   
                                                                 no_mask_106[0][0]                
__________________________________________________________________________________________________
prediction_layer_16 (Prediction (None, 1)            1           add_29[0][0]                     
==================================================================================================
Total params: 2,950
Trainable params: 2,870
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2498 - binary_crossentropy: 0.6924500/500 [==============================] - 7s 14ms/sample - loss: 0.2534 - binary_crossentropy: 0.7002 - val_loss: 0.2580 - val_binary_crossentropy: 0.7098

  #### metrics   #################################################### 
{'MSE': 0.25477987131437224}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/init_ops.py:97: calling Ones.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
Model: "model_13"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_14 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
hash_15 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_16 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         8           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         8           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_107 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_spars
__________________________________________________________________________________________________
no_mask_108 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_spars
__________________________________________________________________________________________________
no_mask_109 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_seque
__________________________________________________________________________________________________
sequence_pooling_layer_178 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sparse_fe
__________________________________________________________________________________________________
no_mask_110 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_seque
__________________________________________________________________________________________________
sequence_pooling_layer_179 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sparse_f
__________________________________________________________________________________________________
no_mask_111 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_seque
__________________________________________________________________________________________________
sequence_pooling_layer_180 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sparse_fe
__________________________________________________________________________________________________
no_mask_112 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_seque
__________________________________________________________________________________________________
sequence_pooling_layer_181 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sparse_fe
__________________________________________________________________________________________________
no_mask_113 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_seque
__________________________________________________________________________________________________
sequence_pooling_layer_182 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sparse_f
__________________________________________________________________________________________________
no_mask_114 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_seque
__________________________________________________________________________________________________
sequence_pooling_layer_183 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sparse_fe
__________________________________________________________________________________________________
sequence_pooling_layer_184 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sequence_
__________________________________________________________________________________________________
sequence_pooling_layer_185 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sequence
__________________________________________________________________________________________________
sequence_pooling_layer_186 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sequence_
__________________________________________________________________________________________________
sequence_pooling_layer_187 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sequence_
__________________________________________________________________________________________________
sequence_pooling_layer_188 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sequence
__________________________________________________________________________________________________
sequence_pooling_layer_189 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sequence_
__________________________________________________________________________________________________
multiply (Multiply)             (None, 1, 4)         0           no_mask_107[0][0]                
                                                                 no_mask_108[0][0]                
__________________________________________________________________________________________________
multiply_1 (Multiply)           (None, 1, 4)         0           no_mask_109[0][0]                
                                                                 sequence_pooling_layer_178[0][0] 
__________________________________________________________________________________________________
multiply_2 (Multiply)           (None, 1, 4)         0           no_mask_110[0][0]                
                                                                 sequence_pooling_layer_179[0][0] 
__________________________________________________________________________________________________
multiply_3 (Multiply)           (None, 1, 4)         0           no_mask_111[0][0]                
                                                                 sequence_pooling_layer_180[0][0] 
__________________________________________________________________________________________________
multiply_4 (Multiply)           (None, 1, 4)         0           no_mask_112[0][0]                
                                                                 sequence_pooling_layer_181[0][0] 
__________________________________________________________________________________________________
multiply_5 (Multiply)           (None, 1, 4)         0           no_mask_113[0][0]                
                                                                 sequence_pooling_layer_182[0][0] 
__________________________________________________________________________________________________
multiply_6 (Multiply)           (None, 1, 4)         0           no_mask_114[0][0]                
                                                                 sequence_pooling_layer_183[0][0] 
__________________________________________________________________________________________________
multiply_7 (Multiply)           (None, 1, 4)         0           sequence_pooling_layer_184[0][0] 
                                                                 sequence_pooling_layer_185[0][0] 
__________________________________________________________________________________________________
multiply_8 (Multiply)           (None, 1, 4)         0           sequence_pooling_layer_186[0][0] 
                                                                 sequence_pooling_layer_187[0][0] 
__________________________________________________________________________________________________
multiply_9 (Multiply)           (None, 1, 4)         0           sequence_pooling_layer_188[0][0] 
                                                                 sequence_pooling_layer_189[0][0] 
__________________________________________________________________________________________________
no_mask_115 (NoMask)            (None, 1, 4)         0           multiply[0][0]                   
                                                                 multiply_1[0][0]                 
                                                                 multiply_2[0][0]                 
                                                                 multiply_3[0][0]                 
                                                                 multiply_4[0][0]                 
                                                                 multiply_5[0][0]                 
                                                                 multiply_6[0][0]                 
                                                                 multiply_7[0][0]                 
                                                                 multiply_8[0][0]                 
                                                                 multiply_9[0][0]                 
__________________________________________________________________________________________________
concatenate_80 (Concatenate)    (None, 10, 4)        0           no_mask_115[0][0]                
                                                                 no_mask_115[1][0]                
                                                                 no_mask_115[2][0]                
                                                                 no_mask_115[3][0]                
                                                                 no_mask_115[4][0]                
                                                                 no_mask_115[5][0]                
                                                                 no_mask_115[6][0]                
                                                                 no_mask_115[7][0]                
                                                                 no_mask_115[8][0]                
                                                                 no_mask_115[9][0]                
__________________________________________________________________________________________________
flatten_28 (Flatten)            (None, 40)           0           concatenate_80[0][0]             
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 40)           160         flatten_28[0][0]                 
__________________________________________________________________________________________________
no_mask_117 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
no_mask_116 (NoMask)            (None, 40)           0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
concatenate_81 (Concatenate)    (None, 2)            0           no_mask_117[0][0]                
                                                                 no_mask_117[1][0]                
__________________________________________________________________________________________________
hash_10 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
hash_11 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_11[0][0]                    
__________________________________________________________________________________________________
sequence_pooling_layer_172 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_173 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_174 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
no_mask_118 (NoMask)            multiple             0           flatten_29[0][0]                 
                                                                 flatten_30[0][0]                 
__________________________________________________________________________________________________
no_mask_104 (NoMask)            (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_172[0][0] 
                                                                 sequence_pooling_layer_173[0][0] 
                                                                 sequence_pooling_layer_174[0][0] 
__________________________________________________________________________________________________
no_mask_105 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
concatenate_82 (Concatenate)    (None, 42)           0           no_mask_118[0][0]                
                                                                 no_mask_118[1][0]                
__________________________________________________________________________________________________
concatenate_78 (Concatenate)    (None, 1, 5)         0           no_mask_104[0][0]                
                                                                 no_mask_104[1][0]                
                                                                 no_mask_104[2][0]                
                                                                 no_mask_104[3][0]                
                                                                 no_mask_104[4][0]                
__________________________________________________________________________________________________
concatenate_79 (Concatenate)    (None, 2)            0           no_mask_105[0][0]                
                                                                 no_mask_105[1][0]                
__________________________________________________________________________________________________
dnn_18 (DNN)                    (None, 32)           2432        concatenate_82[0][0]             
__________________________________________________________________________________________________
linear_17 (Linear)              (None, 1)            2           concatenate_78[0][0]             
                                                                 concatenate_79[0][0]             
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 1)            32          dnn_18[0][0]                     
__________________________________________________________________________________________________
no_mask_106 (NoMask)            (None, 1)            0           linear_17[0][0]                  
__________________________________________________________________________________________________
add_29 (Add)                    (None, 1)            0           dense_11[0][0]                   
                                                                 no_mask_106[0][0]                
__________________________________________________________________________________________________
prediction_layer_16 (Prediction (None, 1)            1           add_29[0][0]                     
==================================================================================================
Total params: 2,950
Trainable params: 2,870
Non-trainable params: 80
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'PNN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'PNN', 'sparse_feature_num': 1, 'dense_feature_num': 1} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_PNN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//01_deepctr.py", line 541, in <module>
    test(pars_choice=5, **{"model_name": model_name})
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//01_deepctr.py", line 517, in test
    module, model = module_load_full("model_keras.01_deepctr", model_pars, data_pars, compute_pars, dataset=dataset)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 101, in module_load_full
    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/01_deepctr.py", line 155, in __init__
    self.model = modeli(feature_columns, **MODEL_PARAMS[model_name])
TypeError: PNN() got an unexpected keyword argument 'embedding_size'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Already up to date.
[master 80c29b2] ml_store
 1 file changed, 4948 insertions(+)
To github.com:arita37/mlmodels_store.git
   eeb5063..80c29b2  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//nbeats.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Using TensorFlow backend.
Loading data...
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//nbeats.py", line 315, in <module>
    test(pars_choice="test01")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//nbeats.py", line 278, in test
    Xtuple = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//nbeats.py", line 172, in get_dataset
    train_data = Data(data_source= path_norm( data_pars["train_data_source"]) ,
NameError: name 'Data' is not defined

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Already up to date.
[master 675e9f1] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   80c29b2..675e9f1  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//keras_gan.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//keras_gan.py", line 31, in <module>
    'AAE' : kg.aae.aae,
AttributeError: module 'mlmodels.model_keras.raw.keras_gan' has no attribute 'aae'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Already up to date.
[master 268c192] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   675e9f1..268c192  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//02_cnn.py 

  ('#### Loading params   ##############################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/',) 

  ('#### Model params   ################################################',) 

  ('#### Loading dataset   #############################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz

    8192/11490434 [..............................] - ETA: 0s
 4227072/11490434 [==========>...................] - ETA: 0s
11108352/11490434 [============================>.] - ETA: 0s
11493376/11490434 [==============================] - 0s 0us/step

  ('#### Model init, fit   #############################################',) 
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.


  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 60000 samples, validate on 10000 samples
Epoch 1/1

   32/60000 [..............................] - ETA: 8:10 - loss: 2.3140 - categorical_accuracy: 0.1250
   64/60000 [..............................] - ETA: 4:55 - loss: 2.2529 - categorical_accuracy: 0.1562
  128/60000 [..............................] - ETA: 3:13 - loss: 2.2355 - categorical_accuracy: 0.2188
  160/60000 [..............................] - ETA: 2:53 - loss: 2.2356 - categorical_accuracy: 0.2188
  224/60000 [..............................] - ETA: 2:28 - loss: 2.1713 - categorical_accuracy: 0.2366
  288/60000 [..............................] - ETA: 2:15 - loss: 2.1089 - categorical_accuracy: 0.2604
  352/60000 [..............................] - ETA: 2:06 - loss: 1.9894 - categorical_accuracy: 0.3125
  416/60000 [..............................] - ETA: 2:00 - loss: 1.9536 - categorical_accuracy: 0.3317
  480/60000 [..............................] - ETA: 1:56 - loss: 1.8921 - categorical_accuracy: 0.3542
  544/60000 [..............................] - ETA: 1:53 - loss: 1.8910 - categorical_accuracy: 0.3566
  576/60000 [..............................] - ETA: 1:52 - loss: 1.8666 - categorical_accuracy: 0.3681
  608/60000 [..............................] - ETA: 1:51 - loss: 1.8331 - categorical_accuracy: 0.3832
  672/60000 [..............................] - ETA: 1:49 - loss: 1.7546 - categorical_accuracy: 0.4077
  704/60000 [..............................] - ETA: 1:50 - loss: 1.7181 - categorical_accuracy: 0.4247
  736/60000 [..............................] - ETA: 1:49 - loss: 1.7008 - categorical_accuracy: 0.4307
  768/60000 [..............................] - ETA: 1:48 - loss: 1.6842 - categorical_accuracy: 0.4362
  800/60000 [..............................] - ETA: 1:48 - loss: 1.6699 - categorical_accuracy: 0.4387
  864/60000 [..............................] - ETA: 1:46 - loss: 1.6135 - categorical_accuracy: 0.4572
  928/60000 [..............................] - ETA: 1:45 - loss: 1.5799 - categorical_accuracy: 0.4688
  992/60000 [..............................] - ETA: 1:44 - loss: 1.5438 - categorical_accuracy: 0.4839
 1056/60000 [..............................] - ETA: 1:43 - loss: 1.5031 - categorical_accuracy: 0.5000
 1120/60000 [..............................] - ETA: 1:42 - loss: 1.4523 - categorical_accuracy: 0.5179
 1184/60000 [..............................] - ETA: 1:41 - loss: 1.4091 - categorical_accuracy: 0.5338
 1248/60000 [..............................] - ETA: 1:41 - loss: 1.3894 - categorical_accuracy: 0.5409
 1280/60000 [..............................] - ETA: 1:40 - loss: 1.3778 - categorical_accuracy: 0.5430
 1344/60000 [..............................] - ETA: 1:40 - loss: 1.3574 - categorical_accuracy: 0.5499
 1408/60000 [..............................] - ETA: 1:39 - loss: 1.3281 - categorical_accuracy: 0.5589
 1440/60000 [..............................] - ETA: 1:39 - loss: 1.3214 - categorical_accuracy: 0.5618
 1472/60000 [..............................] - ETA: 1:39 - loss: 1.3105 - categorical_accuracy: 0.5659
 1504/60000 [..............................] - ETA: 1:39 - loss: 1.2932 - categorical_accuracy: 0.5725
 1536/60000 [..............................] - ETA: 1:38 - loss: 1.2770 - categorical_accuracy: 0.5775
 1600/60000 [..............................] - ETA: 1:38 - loss: 1.2515 - categorical_accuracy: 0.5844
 1664/60000 [..............................] - ETA: 1:38 - loss: 1.2275 - categorical_accuracy: 0.5925
 1696/60000 [..............................] - ETA: 1:37 - loss: 1.2220 - categorical_accuracy: 0.5938
 1760/60000 [..............................] - ETA: 1:37 - loss: 1.2015 - categorical_accuracy: 0.5989
 1824/60000 [..............................] - ETA: 1:37 - loss: 1.1742 - categorical_accuracy: 0.6086
 1888/60000 [..............................] - ETA: 1:36 - loss: 1.1568 - categorical_accuracy: 0.6139
 1952/60000 [..............................] - ETA: 1:36 - loss: 1.1385 - categorical_accuracy: 0.6189
 2016/60000 [>.............................] - ETA: 1:36 - loss: 1.1172 - categorical_accuracy: 0.6270
 2080/60000 [>.............................] - ETA: 1:35 - loss: 1.0986 - categorical_accuracy: 0.6313
 2144/60000 [>.............................] - ETA: 1:35 - loss: 1.0873 - categorical_accuracy: 0.6367
 2176/60000 [>.............................] - ETA: 1:35 - loss: 1.0796 - categorical_accuracy: 0.6397
 2240/60000 [>.............................] - ETA: 1:34 - loss: 1.0653 - categorical_accuracy: 0.6451
 2304/60000 [>.............................] - ETA: 1:34 - loss: 1.0532 - categorical_accuracy: 0.6502
 2368/60000 [>.............................] - ETA: 1:34 - loss: 1.0413 - categorical_accuracy: 0.6546
 2400/60000 [>.............................] - ETA: 1:34 - loss: 1.0412 - categorical_accuracy: 0.6554
 2464/60000 [>.............................] - ETA: 1:33 - loss: 1.0262 - categorical_accuracy: 0.6615
 2528/60000 [>.............................] - ETA: 1:33 - loss: 1.0100 - categorical_accuracy: 0.6677
 2592/60000 [>.............................] - ETA: 1:33 - loss: 0.9991 - categorical_accuracy: 0.6717
 2624/60000 [>.............................] - ETA: 1:33 - loss: 0.9930 - categorical_accuracy: 0.6738
 2688/60000 [>.............................] - ETA: 1:32 - loss: 0.9808 - categorical_accuracy: 0.6775
 2720/60000 [>.............................] - ETA: 1:32 - loss: 0.9723 - categorical_accuracy: 0.6794
 2784/60000 [>.............................] - ETA: 1:32 - loss: 0.9562 - categorical_accuracy: 0.6857
 2848/60000 [>.............................] - ETA: 1:32 - loss: 0.9465 - categorical_accuracy: 0.6875
 2912/60000 [>.............................] - ETA: 1:32 - loss: 0.9316 - categorical_accuracy: 0.6923
 2976/60000 [>.............................] - ETA: 1:31 - loss: 0.9253 - categorical_accuracy: 0.6949
 3008/60000 [>.............................] - ETA: 1:31 - loss: 0.9219 - categorical_accuracy: 0.6965
 3072/60000 [>.............................] - ETA: 1:31 - loss: 0.9146 - categorical_accuracy: 0.6999
 3136/60000 [>.............................] - ETA: 1:31 - loss: 0.9042 - categorical_accuracy: 0.7041
 3200/60000 [>.............................] - ETA: 1:31 - loss: 0.8974 - categorical_accuracy: 0.7069
 3232/60000 [>.............................] - ETA: 1:31 - loss: 0.8921 - categorical_accuracy: 0.7092
 3264/60000 [>.............................] - ETA: 1:31 - loss: 0.8884 - categorical_accuracy: 0.7102
 3328/60000 [>.............................] - ETA: 1:31 - loss: 0.8799 - categorical_accuracy: 0.7139
 3392/60000 [>.............................] - ETA: 1:30 - loss: 0.8690 - categorical_accuracy: 0.7179
 3456/60000 [>.............................] - ETA: 1:30 - loss: 0.8628 - categorical_accuracy: 0.7211
 3488/60000 [>.............................] - ETA: 1:30 - loss: 0.8559 - categorical_accuracy: 0.7236
 3552/60000 [>.............................] - ETA: 1:30 - loss: 0.8481 - categorical_accuracy: 0.7266
 3616/60000 [>.............................] - ETA: 1:30 - loss: 0.8400 - categorical_accuracy: 0.7290
 3680/60000 [>.............................] - ETA: 1:30 - loss: 0.8325 - categorical_accuracy: 0.7304
 3744/60000 [>.............................] - ETA: 1:29 - loss: 0.8236 - categorical_accuracy: 0.7326
 3808/60000 [>.............................] - ETA: 1:29 - loss: 0.8183 - categorical_accuracy: 0.7348
 3872/60000 [>.............................] - ETA: 1:29 - loss: 0.8102 - categorical_accuracy: 0.7376
 3936/60000 [>.............................] - ETA: 1:29 - loss: 0.8013 - categorical_accuracy: 0.7406
 3968/60000 [>.............................] - ETA: 1:29 - loss: 0.7982 - categorical_accuracy: 0.7417
 4032/60000 [=>............................] - ETA: 1:28 - loss: 0.7931 - categorical_accuracy: 0.7436
 4096/60000 [=>............................] - ETA: 1:28 - loss: 0.7873 - categorical_accuracy: 0.7449
 4160/60000 [=>............................] - ETA: 1:28 - loss: 0.7812 - categorical_accuracy: 0.7474
 4224/60000 [=>............................] - ETA: 1:28 - loss: 0.7728 - categorical_accuracy: 0.7500
 4288/60000 [=>............................] - ETA: 1:28 - loss: 0.7665 - categorical_accuracy: 0.7521
 4352/60000 [=>............................] - ETA: 1:28 - loss: 0.7620 - categorical_accuracy: 0.7539
 4384/60000 [=>............................] - ETA: 1:28 - loss: 0.7583 - categorical_accuracy: 0.7552
 4448/60000 [=>............................] - ETA: 1:28 - loss: 0.7498 - categorical_accuracy: 0.7579
 4512/60000 [=>............................] - ETA: 1:27 - loss: 0.7439 - categorical_accuracy: 0.7604
 4576/60000 [=>............................] - ETA: 1:27 - loss: 0.7378 - categorical_accuracy: 0.7627
 4608/60000 [=>............................] - ETA: 1:27 - loss: 0.7362 - categorical_accuracy: 0.7630
 4672/60000 [=>............................] - ETA: 1:27 - loss: 0.7290 - categorical_accuracy: 0.7648
 4736/60000 [=>............................] - ETA: 1:27 - loss: 0.7240 - categorical_accuracy: 0.7660
 4800/60000 [=>............................] - ETA: 1:27 - loss: 0.7200 - categorical_accuracy: 0.7677
 4864/60000 [=>............................] - ETA: 1:27 - loss: 0.7162 - categorical_accuracy: 0.7687
 4928/60000 [=>............................] - ETA: 1:26 - loss: 0.7121 - categorical_accuracy: 0.7703
 4960/60000 [=>............................] - ETA: 1:26 - loss: 0.7101 - categorical_accuracy: 0.7710
 5024/60000 [=>............................] - ETA: 1:26 - loss: 0.7076 - categorical_accuracy: 0.7721
 5088/60000 [=>............................] - ETA: 1:26 - loss: 0.7034 - categorical_accuracy: 0.7730
 5120/60000 [=>............................] - ETA: 1:26 - loss: 0.7018 - categorical_accuracy: 0.7732
 5184/60000 [=>............................] - ETA: 1:26 - loss: 0.6955 - categorical_accuracy: 0.7755
 5248/60000 [=>............................] - ETA: 1:26 - loss: 0.6909 - categorical_accuracy: 0.7767
 5280/60000 [=>............................] - ETA: 1:26 - loss: 0.6890 - categorical_accuracy: 0.7773
 5312/60000 [=>............................] - ETA: 1:26 - loss: 0.6863 - categorical_accuracy: 0.7779
 5376/60000 [=>............................] - ETA: 1:26 - loss: 0.6807 - categorical_accuracy: 0.7803
 5440/60000 [=>............................] - ETA: 1:25 - loss: 0.6781 - categorical_accuracy: 0.7816
 5472/60000 [=>............................] - ETA: 1:25 - loss: 0.6786 - categorical_accuracy: 0.7820
 5536/60000 [=>............................] - ETA: 1:25 - loss: 0.6751 - categorical_accuracy: 0.7831
 5568/60000 [=>............................] - ETA: 1:25 - loss: 0.6748 - categorical_accuracy: 0.7827
 5600/60000 [=>............................] - ETA: 1:25 - loss: 0.6726 - categorical_accuracy: 0.7832
 5664/60000 [=>............................] - ETA: 1:25 - loss: 0.6684 - categorical_accuracy: 0.7848
 5696/60000 [=>............................] - ETA: 1:25 - loss: 0.6664 - categorical_accuracy: 0.7855
 5760/60000 [=>............................] - ETA: 1:25 - loss: 0.6623 - categorical_accuracy: 0.7870
 5824/60000 [=>............................] - ETA: 1:25 - loss: 0.6564 - categorical_accuracy: 0.7888
 5856/60000 [=>............................] - ETA: 1:25 - loss: 0.6534 - categorical_accuracy: 0.7896
 5888/60000 [=>............................] - ETA: 1:25 - loss: 0.6522 - categorical_accuracy: 0.7896
 5920/60000 [=>............................] - ETA: 1:25 - loss: 0.6508 - categorical_accuracy: 0.7897
 5984/60000 [=>............................] - ETA: 1:25 - loss: 0.6477 - categorical_accuracy: 0.7906
 6048/60000 [==>...........................] - ETA: 1:24 - loss: 0.6434 - categorical_accuracy: 0.7920
 6080/60000 [==>...........................] - ETA: 1:24 - loss: 0.6415 - categorical_accuracy: 0.7928
 6144/60000 [==>...........................] - ETA: 1:24 - loss: 0.6388 - categorical_accuracy: 0.7933
 6208/60000 [==>...........................] - ETA: 1:24 - loss: 0.6353 - categorical_accuracy: 0.7945
 6240/60000 [==>...........................] - ETA: 1:24 - loss: 0.6338 - categorical_accuracy: 0.7952
 6304/60000 [==>...........................] - ETA: 1:24 - loss: 0.6307 - categorical_accuracy: 0.7963
 6368/60000 [==>...........................] - ETA: 1:24 - loss: 0.6282 - categorical_accuracy: 0.7970
 6400/60000 [==>...........................] - ETA: 1:24 - loss: 0.6261 - categorical_accuracy: 0.7975
 6464/60000 [==>...........................] - ETA: 1:23 - loss: 0.6211 - categorical_accuracy: 0.7990
 6528/60000 [==>...........................] - ETA: 1:23 - loss: 0.6195 - categorical_accuracy: 0.7999
 6592/60000 [==>...........................] - ETA: 1:23 - loss: 0.6169 - categorical_accuracy: 0.8007
 6656/60000 [==>...........................] - ETA: 1:23 - loss: 0.6122 - categorical_accuracy: 0.8023
 6720/60000 [==>...........................] - ETA: 1:23 - loss: 0.6087 - categorical_accuracy: 0.8034
 6784/60000 [==>...........................] - ETA: 1:23 - loss: 0.6048 - categorical_accuracy: 0.8048
 6848/60000 [==>...........................] - ETA: 1:23 - loss: 0.6016 - categorical_accuracy: 0.8058
 6912/60000 [==>...........................] - ETA: 1:22 - loss: 0.5977 - categorical_accuracy: 0.8071
 6944/60000 [==>...........................] - ETA: 1:22 - loss: 0.5959 - categorical_accuracy: 0.8077
 7008/60000 [==>...........................] - ETA: 1:22 - loss: 0.5947 - categorical_accuracy: 0.8085
 7040/60000 [==>...........................] - ETA: 1:22 - loss: 0.5928 - categorical_accuracy: 0.8092
 7104/60000 [==>...........................] - ETA: 1:22 - loss: 0.5905 - categorical_accuracy: 0.8097
 7136/60000 [==>...........................] - ETA: 1:22 - loss: 0.5895 - categorical_accuracy: 0.8101
 7168/60000 [==>...........................] - ETA: 1:22 - loss: 0.5875 - categorical_accuracy: 0.8108
 7232/60000 [==>...........................] - ETA: 1:22 - loss: 0.5840 - categorical_accuracy: 0.8118
 7264/60000 [==>...........................] - ETA: 1:22 - loss: 0.5820 - categorical_accuracy: 0.8125
 7328/60000 [==>...........................] - ETA: 1:22 - loss: 0.5786 - categorical_accuracy: 0.8137
 7360/60000 [==>...........................] - ETA: 1:22 - loss: 0.5776 - categorical_accuracy: 0.8141
 7392/60000 [==>...........................] - ETA: 1:22 - loss: 0.5770 - categorical_accuracy: 0.8143
 7456/60000 [==>...........................] - ETA: 1:22 - loss: 0.5744 - categorical_accuracy: 0.8152
 7488/60000 [==>...........................] - ETA: 1:22 - loss: 0.5728 - categorical_accuracy: 0.8157
 7520/60000 [==>...........................] - ETA: 1:22 - loss: 0.5716 - categorical_accuracy: 0.8160
 7584/60000 [==>...........................] - ETA: 1:21 - loss: 0.5690 - categorical_accuracy: 0.8166
 7648/60000 [==>...........................] - ETA: 1:21 - loss: 0.5671 - categorical_accuracy: 0.8172
 7712/60000 [==>...........................] - ETA: 1:21 - loss: 0.5648 - categorical_accuracy: 0.8182
 7776/60000 [==>...........................] - ETA: 1:21 - loss: 0.5610 - categorical_accuracy: 0.8197
 7808/60000 [==>...........................] - ETA: 1:21 - loss: 0.5590 - categorical_accuracy: 0.8204
 7872/60000 [==>...........................] - ETA: 1:21 - loss: 0.5587 - categorical_accuracy: 0.8208
 7936/60000 [==>...........................] - ETA: 1:21 - loss: 0.5567 - categorical_accuracy: 0.8211
 8000/60000 [===>..........................] - ETA: 1:21 - loss: 0.5562 - categorical_accuracy: 0.8211
 8032/60000 [===>..........................] - ETA: 1:21 - loss: 0.5549 - categorical_accuracy: 0.8216
 8096/60000 [===>..........................] - ETA: 1:21 - loss: 0.5523 - categorical_accuracy: 0.8225
 8160/60000 [===>..........................] - ETA: 1:20 - loss: 0.5492 - categorical_accuracy: 0.8237
 8192/60000 [===>..........................] - ETA: 1:20 - loss: 0.5477 - categorical_accuracy: 0.8242
 8224/60000 [===>..........................] - ETA: 1:20 - loss: 0.5472 - categorical_accuracy: 0.8243
 8256/60000 [===>..........................] - ETA: 1:20 - loss: 0.5456 - categorical_accuracy: 0.8249
 8320/60000 [===>..........................] - ETA: 1:20 - loss: 0.5430 - categorical_accuracy: 0.8258
 8352/60000 [===>..........................] - ETA: 1:20 - loss: 0.5419 - categorical_accuracy: 0.8264
 8416/60000 [===>..........................] - ETA: 1:20 - loss: 0.5391 - categorical_accuracy: 0.8274
 8480/60000 [===>..........................] - ETA: 1:20 - loss: 0.5362 - categorical_accuracy: 0.8283
 8544/60000 [===>..........................] - ETA: 1:20 - loss: 0.5332 - categorical_accuracy: 0.8294
 8608/60000 [===>..........................] - ETA: 1:20 - loss: 0.5317 - categorical_accuracy: 0.8298
 8672/60000 [===>..........................] - ETA: 1:20 - loss: 0.5292 - categorical_accuracy: 0.8307
 8704/60000 [===>..........................] - ETA: 1:20 - loss: 0.5278 - categorical_accuracy: 0.8311
 8768/60000 [===>..........................] - ETA: 1:19 - loss: 0.5252 - categorical_accuracy: 0.8319
 8832/60000 [===>..........................] - ETA: 1:19 - loss: 0.5220 - categorical_accuracy: 0.8329
 8896/60000 [===>..........................] - ETA: 1:19 - loss: 0.5198 - categorical_accuracy: 0.8335
 8928/60000 [===>..........................] - ETA: 1:19 - loss: 0.5184 - categorical_accuracy: 0.8339
 8992/60000 [===>..........................] - ETA: 1:19 - loss: 0.5163 - categorical_accuracy: 0.8344
 9056/60000 [===>..........................] - ETA: 1:19 - loss: 0.5143 - categorical_accuracy: 0.8351
 9120/60000 [===>..........................] - ETA: 1:19 - loss: 0.5128 - categorical_accuracy: 0.8355
 9184/60000 [===>..........................] - ETA: 1:19 - loss: 0.5119 - categorical_accuracy: 0.8358
 9248/60000 [===>..........................] - ETA: 1:19 - loss: 0.5107 - categorical_accuracy: 0.8362
 9312/60000 [===>..........................] - ETA: 1:18 - loss: 0.5084 - categorical_accuracy: 0.8370
 9376/60000 [===>..........................] - ETA: 1:18 - loss: 0.5063 - categorical_accuracy: 0.8377
 9440/60000 [===>..........................] - ETA: 1:18 - loss: 0.5038 - categorical_accuracy: 0.8385
 9504/60000 [===>..........................] - ETA: 1:18 - loss: 0.5022 - categorical_accuracy: 0.8388
 9568/60000 [===>..........................] - ETA: 1:18 - loss: 0.5000 - categorical_accuracy: 0.8396
 9600/60000 [===>..........................] - ETA: 1:18 - loss: 0.4989 - categorical_accuracy: 0.8400
 9632/60000 [===>..........................] - ETA: 1:18 - loss: 0.4982 - categorical_accuracy: 0.8401
 9696/60000 [===>..........................] - ETA: 1:18 - loss: 0.4958 - categorical_accuracy: 0.8409
 9760/60000 [===>..........................] - ETA: 1:18 - loss: 0.4938 - categorical_accuracy: 0.8413
 9824/60000 [===>..........................] - ETA: 1:18 - loss: 0.4915 - categorical_accuracy: 0.8419
 9888/60000 [===>..........................] - ETA: 1:17 - loss: 0.4892 - categorical_accuracy: 0.8425
 9952/60000 [===>..........................] - ETA: 1:17 - loss: 0.4886 - categorical_accuracy: 0.8427
10016/60000 [====>.........................] - ETA: 1:17 - loss: 0.4868 - categorical_accuracy: 0.8433
10080/60000 [====>.........................] - ETA: 1:17 - loss: 0.4852 - categorical_accuracy: 0.8439
10144/60000 [====>.........................] - ETA: 1:17 - loss: 0.4833 - categorical_accuracy: 0.8446
10208/60000 [====>.........................] - ETA: 1:17 - loss: 0.4826 - categorical_accuracy: 0.8451
10272/60000 [====>.........................] - ETA: 1:17 - loss: 0.4813 - categorical_accuracy: 0.8456
10336/60000 [====>.........................] - ETA: 1:17 - loss: 0.4792 - categorical_accuracy: 0.8463
10368/60000 [====>.........................] - ETA: 1:17 - loss: 0.4779 - categorical_accuracy: 0.8467
10400/60000 [====>.........................] - ETA: 1:17 - loss: 0.4767 - categorical_accuracy: 0.8472
10464/60000 [====>.........................] - ETA: 1:16 - loss: 0.4759 - categorical_accuracy: 0.8475
10496/60000 [====>.........................] - ETA: 1:16 - loss: 0.4757 - categorical_accuracy: 0.8473
10560/60000 [====>.........................] - ETA: 1:16 - loss: 0.4739 - categorical_accuracy: 0.8477
10592/60000 [====>.........................] - ETA: 1:16 - loss: 0.4727 - categorical_accuracy: 0.8482
10656/60000 [====>.........................] - ETA: 1:16 - loss: 0.4711 - categorical_accuracy: 0.8487
10720/60000 [====>.........................] - ETA: 1:16 - loss: 0.4693 - categorical_accuracy: 0.8492
10784/60000 [====>.........................] - ETA: 1:16 - loss: 0.4675 - categorical_accuracy: 0.8499
10848/60000 [====>.........................] - ETA: 1:16 - loss: 0.4657 - categorical_accuracy: 0.8506
10912/60000 [====>.........................] - ETA: 1:16 - loss: 0.4641 - categorical_accuracy: 0.8511
10976/60000 [====>.........................] - ETA: 1:16 - loss: 0.4633 - categorical_accuracy: 0.8513
11040/60000 [====>.........................] - ETA: 1:15 - loss: 0.4621 - categorical_accuracy: 0.8516
11072/60000 [====>.........................] - ETA: 1:15 - loss: 0.4612 - categorical_accuracy: 0.8518
11136/60000 [====>.........................] - ETA: 1:15 - loss: 0.4597 - categorical_accuracy: 0.8523
11168/60000 [====>.........................] - ETA: 1:15 - loss: 0.4590 - categorical_accuracy: 0.8525
11232/60000 [====>.........................] - ETA: 1:15 - loss: 0.4582 - categorical_accuracy: 0.8530
11264/60000 [====>.........................] - ETA: 1:15 - loss: 0.4581 - categorical_accuracy: 0.8531
11328/60000 [====>.........................] - ETA: 1:15 - loss: 0.4566 - categorical_accuracy: 0.8536
11392/60000 [====>.........................] - ETA: 1:15 - loss: 0.4560 - categorical_accuracy: 0.8539
11424/60000 [====>.........................] - ETA: 1:15 - loss: 0.4554 - categorical_accuracy: 0.8541
11456/60000 [====>.........................] - ETA: 1:15 - loss: 0.4545 - categorical_accuracy: 0.8544
11520/60000 [====>.........................] - ETA: 1:15 - loss: 0.4532 - categorical_accuracy: 0.8549
11584/60000 [====>.........................] - ETA: 1:15 - loss: 0.4518 - categorical_accuracy: 0.8550
11616/60000 [====>.........................] - ETA: 1:15 - loss: 0.4509 - categorical_accuracy: 0.8554
11680/60000 [====>.........................] - ETA: 1:14 - loss: 0.4494 - categorical_accuracy: 0.8557
11744/60000 [====>.........................] - ETA: 1:14 - loss: 0.4476 - categorical_accuracy: 0.8564
11808/60000 [====>.........................] - ETA: 1:14 - loss: 0.4460 - categorical_accuracy: 0.8569
11872/60000 [====>.........................] - ETA: 1:14 - loss: 0.4453 - categorical_accuracy: 0.8571
11936/60000 [====>.........................] - ETA: 1:14 - loss: 0.4447 - categorical_accuracy: 0.8572
12000/60000 [=====>........................] - ETA: 1:14 - loss: 0.4431 - categorical_accuracy: 0.8577
12032/60000 [=====>........................] - ETA: 1:14 - loss: 0.4426 - categorical_accuracy: 0.8578
12096/60000 [=====>........................] - ETA: 1:14 - loss: 0.4419 - categorical_accuracy: 0.8581
12128/60000 [=====>........................] - ETA: 1:14 - loss: 0.4415 - categorical_accuracy: 0.8583
12192/60000 [=====>........................] - ETA: 1:14 - loss: 0.4398 - categorical_accuracy: 0.8589
12256/60000 [=====>........................] - ETA: 1:13 - loss: 0.4392 - categorical_accuracy: 0.8593
12320/60000 [=====>........................] - ETA: 1:13 - loss: 0.4384 - categorical_accuracy: 0.8594
12384/60000 [=====>........................] - ETA: 1:13 - loss: 0.4367 - categorical_accuracy: 0.8600
12448/60000 [=====>........................] - ETA: 1:13 - loss: 0.4353 - categorical_accuracy: 0.8604
12480/60000 [=====>........................] - ETA: 1:13 - loss: 0.4347 - categorical_accuracy: 0.8606
12544/60000 [=====>........................] - ETA: 1:13 - loss: 0.4331 - categorical_accuracy: 0.8610
12576/60000 [=====>........................] - ETA: 1:13 - loss: 0.4323 - categorical_accuracy: 0.8613
12608/60000 [=====>........................] - ETA: 1:13 - loss: 0.4320 - categorical_accuracy: 0.8614
12672/60000 [=====>........................] - ETA: 1:13 - loss: 0.4304 - categorical_accuracy: 0.8620
12704/60000 [=====>........................] - ETA: 1:13 - loss: 0.4299 - categorical_accuracy: 0.8622
12736/60000 [=====>........................] - ETA: 1:13 - loss: 0.4291 - categorical_accuracy: 0.8624
12768/60000 [=====>........................] - ETA: 1:13 - loss: 0.4287 - categorical_accuracy: 0.8626
12800/60000 [=====>........................] - ETA: 1:13 - loss: 0.4284 - categorical_accuracy: 0.8627
12864/60000 [=====>........................] - ETA: 1:13 - loss: 0.4283 - categorical_accuracy: 0.8626
12896/60000 [=====>........................] - ETA: 1:13 - loss: 0.4277 - categorical_accuracy: 0.8628
12928/60000 [=====>........................] - ETA: 1:13 - loss: 0.4271 - categorical_accuracy: 0.8630
12992/60000 [=====>........................] - ETA: 1:12 - loss: 0.4265 - categorical_accuracy: 0.8632
13056/60000 [=====>........................] - ETA: 1:12 - loss: 0.4258 - categorical_accuracy: 0.8634
13088/60000 [=====>........................] - ETA: 1:12 - loss: 0.4255 - categorical_accuracy: 0.8635
13152/60000 [=====>........................] - ETA: 1:12 - loss: 0.4239 - categorical_accuracy: 0.8641
13216/60000 [=====>........................] - ETA: 1:12 - loss: 0.4225 - categorical_accuracy: 0.8646
13280/60000 [=====>........................] - ETA: 1:12 - loss: 0.4219 - categorical_accuracy: 0.8648
13344/60000 [=====>........................] - ETA: 1:12 - loss: 0.4204 - categorical_accuracy: 0.8654
13408/60000 [=====>........................] - ETA: 1:12 - loss: 0.4188 - categorical_accuracy: 0.8660
13440/60000 [=====>........................] - ETA: 1:12 - loss: 0.4184 - categorical_accuracy: 0.8661
13504/60000 [=====>........................] - ETA: 1:12 - loss: 0.4178 - categorical_accuracy: 0.8663
13568/60000 [=====>........................] - ETA: 1:11 - loss: 0.4168 - categorical_accuracy: 0.8667
13632/60000 [=====>........................] - ETA: 1:11 - loss: 0.4157 - categorical_accuracy: 0.8670
13696/60000 [=====>........................] - ETA: 1:11 - loss: 0.4142 - categorical_accuracy: 0.8675
13760/60000 [=====>........................] - ETA: 1:11 - loss: 0.4130 - categorical_accuracy: 0.8678
13792/60000 [=====>........................] - ETA: 1:11 - loss: 0.4122 - categorical_accuracy: 0.8681
13856/60000 [=====>........................] - ETA: 1:11 - loss: 0.4114 - categorical_accuracy: 0.8684
13888/60000 [=====>........................] - ETA: 1:11 - loss: 0.4108 - categorical_accuracy: 0.8686
13920/60000 [=====>........................] - ETA: 1:11 - loss: 0.4107 - categorical_accuracy: 0.8687
13984/60000 [=====>........................] - ETA: 1:11 - loss: 0.4094 - categorical_accuracy: 0.8692
14048/60000 [======>.......................] - ETA: 1:11 - loss: 0.4090 - categorical_accuracy: 0.8693
14112/60000 [======>.......................] - ETA: 1:11 - loss: 0.4084 - categorical_accuracy: 0.8695
14176/60000 [======>.......................] - ETA: 1:10 - loss: 0.4075 - categorical_accuracy: 0.8698
14240/60000 [======>.......................] - ETA: 1:10 - loss: 0.4062 - categorical_accuracy: 0.8703
14304/60000 [======>.......................] - ETA: 1:10 - loss: 0.4060 - categorical_accuracy: 0.8705
14336/60000 [======>.......................] - ETA: 1:10 - loss: 0.4053 - categorical_accuracy: 0.8706
14400/60000 [======>.......................] - ETA: 1:10 - loss: 0.4045 - categorical_accuracy: 0.8708
14464/60000 [======>.......................] - ETA: 1:10 - loss: 0.4039 - categorical_accuracy: 0.8710
14528/60000 [======>.......................] - ETA: 1:10 - loss: 0.4026 - categorical_accuracy: 0.8713
14560/60000 [======>.......................] - ETA: 1:10 - loss: 0.4020 - categorical_accuracy: 0.8715
14592/60000 [======>.......................] - ETA: 1:10 - loss: 0.4015 - categorical_accuracy: 0.8716
14656/60000 [======>.......................] - ETA: 1:10 - loss: 0.4005 - categorical_accuracy: 0.8719
14688/60000 [======>.......................] - ETA: 1:10 - loss: 0.3999 - categorical_accuracy: 0.8721
14720/60000 [======>.......................] - ETA: 1:10 - loss: 0.3997 - categorical_accuracy: 0.8721
14784/60000 [======>.......................] - ETA: 1:10 - loss: 0.3995 - categorical_accuracy: 0.8722
14848/60000 [======>.......................] - ETA: 1:09 - loss: 0.3985 - categorical_accuracy: 0.8726
14912/60000 [======>.......................] - ETA: 1:09 - loss: 0.3979 - categorical_accuracy: 0.8728
14976/60000 [======>.......................] - ETA: 1:09 - loss: 0.3969 - categorical_accuracy: 0.8731
15040/60000 [======>.......................] - ETA: 1:09 - loss: 0.3960 - categorical_accuracy: 0.8734
15104/60000 [======>.......................] - ETA: 1:09 - loss: 0.3950 - categorical_accuracy: 0.8737
15168/60000 [======>.......................] - ETA: 1:09 - loss: 0.3938 - categorical_accuracy: 0.8741
15200/60000 [======>.......................] - ETA: 1:09 - loss: 0.3933 - categorical_accuracy: 0.8743
15232/60000 [======>.......................] - ETA: 1:09 - loss: 0.3929 - categorical_accuracy: 0.8745
15264/60000 [======>.......................] - ETA: 1:09 - loss: 0.3923 - categorical_accuracy: 0.8746
15328/60000 [======>.......................] - ETA: 1:09 - loss: 0.3910 - categorical_accuracy: 0.8751
15360/60000 [======>.......................] - ETA: 1:09 - loss: 0.3903 - categorical_accuracy: 0.8753
15392/60000 [======>.......................] - ETA: 1:09 - loss: 0.3901 - categorical_accuracy: 0.8753
15424/60000 [======>.......................] - ETA: 1:09 - loss: 0.3894 - categorical_accuracy: 0.8756
15488/60000 [======>.......................] - ETA: 1:08 - loss: 0.3882 - categorical_accuracy: 0.8760
15552/60000 [======>.......................] - ETA: 1:08 - loss: 0.3870 - categorical_accuracy: 0.8764
15616/60000 [======>.......................] - ETA: 1:08 - loss: 0.3856 - categorical_accuracy: 0.8769
15648/60000 [======>.......................] - ETA: 1:08 - loss: 0.3849 - categorical_accuracy: 0.8772
15712/60000 [======>.......................] - ETA: 1:08 - loss: 0.3844 - categorical_accuracy: 0.8775
15744/60000 [======>.......................] - ETA: 1:08 - loss: 0.3839 - categorical_accuracy: 0.8777
15776/60000 [======>.......................] - ETA: 1:08 - loss: 0.3836 - categorical_accuracy: 0.8777
15840/60000 [======>.......................] - ETA: 1:08 - loss: 0.3826 - categorical_accuracy: 0.8780
15904/60000 [======>.......................] - ETA: 1:08 - loss: 0.3819 - categorical_accuracy: 0.8783
15968/60000 [======>.......................] - ETA: 1:08 - loss: 0.3809 - categorical_accuracy: 0.8786
16032/60000 [=======>......................] - ETA: 1:08 - loss: 0.3803 - categorical_accuracy: 0.8789
16096/60000 [=======>......................] - ETA: 1:07 - loss: 0.3799 - categorical_accuracy: 0.8792
16160/60000 [=======>......................] - ETA: 1:07 - loss: 0.3786 - categorical_accuracy: 0.8797
16192/60000 [=======>......................] - ETA: 1:07 - loss: 0.3783 - categorical_accuracy: 0.8799
16256/60000 [=======>......................] - ETA: 1:07 - loss: 0.3777 - categorical_accuracy: 0.8801
16320/60000 [=======>......................] - ETA: 1:07 - loss: 0.3777 - categorical_accuracy: 0.8803
16384/60000 [=======>......................] - ETA: 1:07 - loss: 0.3766 - categorical_accuracy: 0.8807
16416/60000 [=======>......................] - ETA: 1:07 - loss: 0.3764 - categorical_accuracy: 0.8807
16480/60000 [=======>......................] - ETA: 1:07 - loss: 0.3762 - categorical_accuracy: 0.8809
16544/60000 [=======>......................] - ETA: 1:07 - loss: 0.3757 - categorical_accuracy: 0.8812
16576/60000 [=======>......................] - ETA: 1:07 - loss: 0.3754 - categorical_accuracy: 0.8813
16640/60000 [=======>......................] - ETA: 1:07 - loss: 0.3743 - categorical_accuracy: 0.8816
16704/60000 [=======>......................] - ETA: 1:06 - loss: 0.3733 - categorical_accuracy: 0.8819
16768/60000 [=======>......................] - ETA: 1:06 - loss: 0.3724 - categorical_accuracy: 0.8821
16832/60000 [=======>......................] - ETA: 1:06 - loss: 0.3713 - categorical_accuracy: 0.8825
16864/60000 [=======>......................] - ETA: 1:06 - loss: 0.3708 - categorical_accuracy: 0.8826
16896/60000 [=======>......................] - ETA: 1:06 - loss: 0.3702 - categorical_accuracy: 0.8828
16960/60000 [=======>......................] - ETA: 1:06 - loss: 0.3692 - categorical_accuracy: 0.8831
16992/60000 [=======>......................] - ETA: 1:06 - loss: 0.3687 - categorical_accuracy: 0.8833
17024/60000 [=======>......................] - ETA: 1:06 - loss: 0.3685 - categorical_accuracy: 0.8833
17088/60000 [=======>......................] - ETA: 1:06 - loss: 0.3677 - categorical_accuracy: 0.8837
17152/60000 [=======>......................] - ETA: 1:06 - loss: 0.3675 - categorical_accuracy: 0.8839
17216/60000 [=======>......................] - ETA: 1:06 - loss: 0.3671 - categorical_accuracy: 0.8839
17280/60000 [=======>......................] - ETA: 1:06 - loss: 0.3665 - categorical_accuracy: 0.8842
17312/60000 [=======>......................] - ETA: 1:05 - loss: 0.3660 - categorical_accuracy: 0.8844
17344/60000 [=======>......................] - ETA: 1:05 - loss: 0.3655 - categorical_accuracy: 0.8846
17376/60000 [=======>......................] - ETA: 1:05 - loss: 0.3649 - categorical_accuracy: 0.8848
17408/60000 [=======>......................] - ETA: 1:05 - loss: 0.3646 - categorical_accuracy: 0.8849
17440/60000 [=======>......................] - ETA: 1:05 - loss: 0.3640 - categorical_accuracy: 0.8851
17472/60000 [=======>......................] - ETA: 1:05 - loss: 0.3635 - categorical_accuracy: 0.8853
17504/60000 [=======>......................] - ETA: 1:05 - loss: 0.3629 - categorical_accuracy: 0.8855
17536/60000 [=======>......................] - ETA: 1:05 - loss: 0.3626 - categorical_accuracy: 0.8855
17568/60000 [=======>......................] - ETA: 1:05 - loss: 0.3621 - categorical_accuracy: 0.8858
17632/60000 [=======>......................] - ETA: 1:05 - loss: 0.3621 - categorical_accuracy: 0.8859
17696/60000 [=======>......................] - ETA: 1:05 - loss: 0.3615 - categorical_accuracy: 0.8861
17760/60000 [=======>......................] - ETA: 1:05 - loss: 0.3608 - categorical_accuracy: 0.8862
17824/60000 [=======>......................] - ETA: 1:05 - loss: 0.3607 - categorical_accuracy: 0.8863
17856/60000 [=======>......................] - ETA: 1:05 - loss: 0.3602 - categorical_accuracy: 0.8865
17920/60000 [=======>......................] - ETA: 1:05 - loss: 0.3595 - categorical_accuracy: 0.8868
17952/60000 [=======>......................] - ETA: 1:05 - loss: 0.3592 - categorical_accuracy: 0.8870
18016/60000 [========>.....................] - ETA: 1:04 - loss: 0.3582 - categorical_accuracy: 0.8873
18048/60000 [========>.....................] - ETA: 1:04 - loss: 0.3580 - categorical_accuracy: 0.8872
18112/60000 [========>.....................] - ETA: 1:04 - loss: 0.3576 - categorical_accuracy: 0.8874
18176/60000 [========>.....................] - ETA: 1:04 - loss: 0.3568 - categorical_accuracy: 0.8877
18240/60000 [========>.....................] - ETA: 1:04 - loss: 0.3565 - categorical_accuracy: 0.8877
18304/60000 [========>.....................] - ETA: 1:04 - loss: 0.3556 - categorical_accuracy: 0.8879
18368/60000 [========>.....................] - ETA: 1:04 - loss: 0.3548 - categorical_accuracy: 0.8882
18432/60000 [========>.....................] - ETA: 1:04 - loss: 0.3541 - categorical_accuracy: 0.8884
18496/60000 [========>.....................] - ETA: 1:04 - loss: 0.3536 - categorical_accuracy: 0.8886
18560/60000 [========>.....................] - ETA: 1:04 - loss: 0.3527 - categorical_accuracy: 0.8887
18624/60000 [========>.....................] - ETA: 1:03 - loss: 0.3517 - categorical_accuracy: 0.8890
18656/60000 [========>.....................] - ETA: 1:03 - loss: 0.3511 - categorical_accuracy: 0.8892
18720/60000 [========>.....................] - ETA: 1:03 - loss: 0.3501 - categorical_accuracy: 0.8896
18784/60000 [========>.....................] - ETA: 1:03 - loss: 0.3493 - categorical_accuracy: 0.8898
18848/60000 [========>.....................] - ETA: 1:03 - loss: 0.3484 - categorical_accuracy: 0.8901
18912/60000 [========>.....................] - ETA: 1:03 - loss: 0.3480 - categorical_accuracy: 0.8902
18976/60000 [========>.....................] - ETA: 1:03 - loss: 0.3476 - categorical_accuracy: 0.8904
19040/60000 [========>.....................] - ETA: 1:03 - loss: 0.3469 - categorical_accuracy: 0.8907
19104/60000 [========>.....................] - ETA: 1:03 - loss: 0.3462 - categorical_accuracy: 0.8909
19136/60000 [========>.....................] - ETA: 1:03 - loss: 0.3467 - categorical_accuracy: 0.8909
19200/60000 [========>.....................] - ETA: 1:03 - loss: 0.3461 - categorical_accuracy: 0.8911
19232/60000 [========>.....................] - ETA: 1:02 - loss: 0.3456 - categorical_accuracy: 0.8912
19296/60000 [========>.....................] - ETA: 1:02 - loss: 0.3452 - categorical_accuracy: 0.8913
19360/60000 [========>.....................] - ETA: 1:02 - loss: 0.3444 - categorical_accuracy: 0.8915
19392/60000 [========>.....................] - ETA: 1:02 - loss: 0.3438 - categorical_accuracy: 0.8917
19456/60000 [========>.....................] - ETA: 1:02 - loss: 0.3430 - categorical_accuracy: 0.8919
19488/60000 [========>.....................] - ETA: 1:02 - loss: 0.3428 - categorical_accuracy: 0.8920
19520/60000 [========>.....................] - ETA: 1:02 - loss: 0.3426 - categorical_accuracy: 0.8921
19552/60000 [========>.....................] - ETA: 1:02 - loss: 0.3423 - categorical_accuracy: 0.8922
19584/60000 [========>.....................] - ETA: 1:02 - loss: 0.3418 - categorical_accuracy: 0.8923
19616/60000 [========>.....................] - ETA: 1:02 - loss: 0.3416 - categorical_accuracy: 0.8924
19680/60000 [========>.....................] - ETA: 1:02 - loss: 0.3406 - categorical_accuracy: 0.8928
19744/60000 [========>.....................] - ETA: 1:02 - loss: 0.3403 - categorical_accuracy: 0.8928
19808/60000 [========>.....................] - ETA: 1:02 - loss: 0.3399 - categorical_accuracy: 0.8931
19840/60000 [========>.....................] - ETA: 1:02 - loss: 0.3394 - categorical_accuracy: 0.8932
19904/60000 [========>.....................] - ETA: 1:01 - loss: 0.3388 - categorical_accuracy: 0.8934
19968/60000 [========>.....................] - ETA: 1:01 - loss: 0.3382 - categorical_accuracy: 0.8936
20032/60000 [=========>....................] - ETA: 1:01 - loss: 0.3387 - categorical_accuracy: 0.8936
20096/60000 [=========>....................] - ETA: 1:01 - loss: 0.3380 - categorical_accuracy: 0.8939
20160/60000 [=========>....................] - ETA: 1:01 - loss: 0.3373 - categorical_accuracy: 0.8941
20224/60000 [=========>....................] - ETA: 1:01 - loss: 0.3368 - categorical_accuracy: 0.8943
20256/60000 [=========>....................] - ETA: 1:01 - loss: 0.3363 - categorical_accuracy: 0.8945
20320/60000 [=========>....................] - ETA: 1:01 - loss: 0.3355 - categorical_accuracy: 0.8947
20352/60000 [=========>....................] - ETA: 1:01 - loss: 0.3351 - categorical_accuracy: 0.8949
20416/60000 [=========>....................] - ETA: 1:01 - loss: 0.3348 - categorical_accuracy: 0.8949
20480/60000 [=========>....................] - ETA: 1:01 - loss: 0.3342 - categorical_accuracy: 0.8951
20544/60000 [=========>....................] - ETA: 1:00 - loss: 0.3334 - categorical_accuracy: 0.8953
20608/60000 [=========>....................] - ETA: 1:00 - loss: 0.3327 - categorical_accuracy: 0.8956
20640/60000 [=========>....................] - ETA: 1:00 - loss: 0.3326 - categorical_accuracy: 0.8956
20704/60000 [=========>....................] - ETA: 1:00 - loss: 0.3327 - categorical_accuracy: 0.8955
20768/60000 [=========>....................] - ETA: 1:00 - loss: 0.3319 - categorical_accuracy: 0.8958
20832/60000 [=========>....................] - ETA: 1:00 - loss: 0.3310 - categorical_accuracy: 0.8961
20864/60000 [=========>....................] - ETA: 1:00 - loss: 0.3308 - categorical_accuracy: 0.8961
20928/60000 [=========>....................] - ETA: 1:00 - loss: 0.3303 - categorical_accuracy: 0.8964
20960/60000 [=========>....................] - ETA: 1:00 - loss: 0.3303 - categorical_accuracy: 0.8964
21024/60000 [=========>....................] - ETA: 1:00 - loss: 0.3298 - categorical_accuracy: 0.8965
21088/60000 [=========>....................] - ETA: 1:00 - loss: 0.3291 - categorical_accuracy: 0.8967
21152/60000 [=========>....................] - ETA: 1:00 - loss: 0.3287 - categorical_accuracy: 0.8968
21216/60000 [=========>....................] - ETA: 59s - loss: 0.3281 - categorical_accuracy: 0.8971 
21280/60000 [=========>....................] - ETA: 59s - loss: 0.3277 - categorical_accuracy: 0.8972
21344/60000 [=========>....................] - ETA: 59s - loss: 0.3270 - categorical_accuracy: 0.8975
21376/60000 [=========>....................] - ETA: 59s - loss: 0.3267 - categorical_accuracy: 0.8976
21408/60000 [=========>....................] - ETA: 59s - loss: 0.3266 - categorical_accuracy: 0.8976
21472/60000 [=========>....................] - ETA: 59s - loss: 0.3258 - categorical_accuracy: 0.8978
21504/60000 [=========>....................] - ETA: 59s - loss: 0.3255 - categorical_accuracy: 0.8979
21568/60000 [=========>....................] - ETA: 59s - loss: 0.3252 - categorical_accuracy: 0.8980
21632/60000 [=========>....................] - ETA: 59s - loss: 0.3251 - categorical_accuracy: 0.8980
21696/60000 [=========>....................] - ETA: 59s - loss: 0.3245 - categorical_accuracy: 0.8982
21760/60000 [=========>....................] - ETA: 59s - loss: 0.3239 - categorical_accuracy: 0.8983
21824/60000 [=========>....................] - ETA: 58s - loss: 0.3236 - categorical_accuracy: 0.8984
21888/60000 [=========>....................] - ETA: 58s - loss: 0.3232 - categorical_accuracy: 0.8985
21920/60000 [=========>....................] - ETA: 58s - loss: 0.3230 - categorical_accuracy: 0.8986
21984/60000 [=========>....................] - ETA: 58s - loss: 0.3225 - categorical_accuracy: 0.8987
22048/60000 [==========>...................] - ETA: 58s - loss: 0.3219 - categorical_accuracy: 0.8989
22112/60000 [==========>...................] - ETA: 58s - loss: 0.3213 - categorical_accuracy: 0.8990
22176/60000 [==========>...................] - ETA: 58s - loss: 0.3205 - categorical_accuracy: 0.8993
22240/60000 [==========>...................] - ETA: 58s - loss: 0.3197 - categorical_accuracy: 0.8996
22304/60000 [==========>...................] - ETA: 58s - loss: 0.3191 - categorical_accuracy: 0.8997
22336/60000 [==========>...................] - ETA: 58s - loss: 0.3187 - categorical_accuracy: 0.8999
22400/60000 [==========>...................] - ETA: 58s - loss: 0.3179 - categorical_accuracy: 0.9001
22464/60000 [==========>...................] - ETA: 57s - loss: 0.3175 - categorical_accuracy: 0.9002
22528/60000 [==========>...................] - ETA: 57s - loss: 0.3174 - categorical_accuracy: 0.9003
22592/60000 [==========>...................] - ETA: 57s - loss: 0.3169 - categorical_accuracy: 0.9005
22624/60000 [==========>...................] - ETA: 57s - loss: 0.3165 - categorical_accuracy: 0.9006
22656/60000 [==========>...................] - ETA: 57s - loss: 0.3162 - categorical_accuracy: 0.9007
22688/60000 [==========>...................] - ETA: 57s - loss: 0.3159 - categorical_accuracy: 0.9008
22752/60000 [==========>...................] - ETA: 57s - loss: 0.3153 - categorical_accuracy: 0.9011
22816/60000 [==========>...................] - ETA: 57s - loss: 0.3147 - categorical_accuracy: 0.9012
22848/60000 [==========>...................] - ETA: 57s - loss: 0.3143 - categorical_accuracy: 0.9013
22880/60000 [==========>...................] - ETA: 57s - loss: 0.3142 - categorical_accuracy: 0.9013
22912/60000 [==========>...................] - ETA: 57s - loss: 0.3140 - categorical_accuracy: 0.9014
22976/60000 [==========>...................] - ETA: 57s - loss: 0.3140 - categorical_accuracy: 0.9015
23008/60000 [==========>...................] - ETA: 57s - loss: 0.3138 - categorical_accuracy: 0.9015
23072/60000 [==========>...................] - ETA: 57s - loss: 0.3135 - categorical_accuracy: 0.9016
23136/60000 [==========>...................] - ETA: 56s - loss: 0.3131 - categorical_accuracy: 0.9017
23200/60000 [==========>...................] - ETA: 56s - loss: 0.3127 - categorical_accuracy: 0.9018
23264/60000 [==========>...................] - ETA: 56s - loss: 0.3120 - categorical_accuracy: 0.9020
23328/60000 [==========>...................] - ETA: 56s - loss: 0.3115 - categorical_accuracy: 0.9022
23360/60000 [==========>...................] - ETA: 56s - loss: 0.3111 - categorical_accuracy: 0.9023
23424/60000 [==========>...................] - ETA: 56s - loss: 0.3109 - categorical_accuracy: 0.9023
23456/60000 [==========>...................] - ETA: 56s - loss: 0.3105 - categorical_accuracy: 0.9025
23488/60000 [==========>...................] - ETA: 56s - loss: 0.3101 - categorical_accuracy: 0.9026
23552/60000 [==========>...................] - ETA: 56s - loss: 0.3097 - categorical_accuracy: 0.9027
23584/60000 [==========>...................] - ETA: 56s - loss: 0.3093 - categorical_accuracy: 0.9029
23648/60000 [==========>...................] - ETA: 56s - loss: 0.3087 - categorical_accuracy: 0.9030
23680/60000 [==========>...................] - ETA: 56s - loss: 0.3084 - categorical_accuracy: 0.9032
23712/60000 [==========>...................] - ETA: 56s - loss: 0.3080 - categorical_accuracy: 0.9033
23776/60000 [==========>...................] - ETA: 56s - loss: 0.3076 - categorical_accuracy: 0.9034
23840/60000 [==========>...................] - ETA: 55s - loss: 0.3070 - categorical_accuracy: 0.9036
23904/60000 [==========>...................] - ETA: 55s - loss: 0.3066 - categorical_accuracy: 0.9037
23968/60000 [==========>...................] - ETA: 55s - loss: 0.3062 - categorical_accuracy: 0.9038
24032/60000 [===========>..................] - ETA: 55s - loss: 0.3055 - categorical_accuracy: 0.9040
24096/60000 [===========>..................] - ETA: 55s - loss: 0.3049 - categorical_accuracy: 0.9042
24128/60000 [===========>..................] - ETA: 55s - loss: 0.3046 - categorical_accuracy: 0.9043
24192/60000 [===========>..................] - ETA: 55s - loss: 0.3044 - categorical_accuracy: 0.9042
24224/60000 [===========>..................] - ETA: 55s - loss: 0.3040 - categorical_accuracy: 0.9043
24288/60000 [===========>..................] - ETA: 55s - loss: 0.3036 - categorical_accuracy: 0.9044
24320/60000 [===========>..................] - ETA: 55s - loss: 0.3035 - categorical_accuracy: 0.9044
24384/60000 [===========>..................] - ETA: 55s - loss: 0.3031 - categorical_accuracy: 0.9046
24448/60000 [===========>..................] - ETA: 54s - loss: 0.3027 - categorical_accuracy: 0.9047
24512/60000 [===========>..................] - ETA: 54s - loss: 0.3025 - categorical_accuracy: 0.9048
24576/60000 [===========>..................] - ETA: 54s - loss: 0.3019 - categorical_accuracy: 0.9050
24608/60000 [===========>..................] - ETA: 54s - loss: 0.3015 - categorical_accuracy: 0.9051
24640/60000 [===========>..................] - ETA: 54s - loss: 0.3013 - categorical_accuracy: 0.9051
24704/60000 [===========>..................] - ETA: 54s - loss: 0.3006 - categorical_accuracy: 0.9053
24768/60000 [===========>..................] - ETA: 54s - loss: 0.3004 - categorical_accuracy: 0.9054
24800/60000 [===========>..................] - ETA: 54s - loss: 0.3001 - categorical_accuracy: 0.9055
24864/60000 [===========>..................] - ETA: 54s - loss: 0.2996 - categorical_accuracy: 0.9057
24928/60000 [===========>..................] - ETA: 54s - loss: 0.2991 - categorical_accuracy: 0.9058
24960/60000 [===========>..................] - ETA: 54s - loss: 0.2988 - categorical_accuracy: 0.9059
24992/60000 [===========>..................] - ETA: 54s - loss: 0.2991 - categorical_accuracy: 0.9059
25056/60000 [===========>..................] - ETA: 54s - loss: 0.2991 - categorical_accuracy: 0.9059
25120/60000 [===========>..................] - ETA: 53s - loss: 0.2986 - categorical_accuracy: 0.9060
25184/60000 [===========>..................] - ETA: 53s - loss: 0.2981 - categorical_accuracy: 0.9062
25248/60000 [===========>..................] - ETA: 53s - loss: 0.2976 - categorical_accuracy: 0.9064
25312/60000 [===========>..................] - ETA: 53s - loss: 0.2971 - categorical_accuracy: 0.9066
25344/60000 [===========>..................] - ETA: 53s - loss: 0.2967 - categorical_accuracy: 0.9067
25408/60000 [===========>..................] - ETA: 53s - loss: 0.2964 - categorical_accuracy: 0.9068
25472/60000 [===========>..................] - ETA: 53s - loss: 0.2958 - categorical_accuracy: 0.9070
25536/60000 [===========>..................] - ETA: 53s - loss: 0.2952 - categorical_accuracy: 0.9071
25600/60000 [===========>..................] - ETA: 53s - loss: 0.2950 - categorical_accuracy: 0.9072
25664/60000 [===========>..................] - ETA: 53s - loss: 0.2945 - categorical_accuracy: 0.9074
25728/60000 [===========>..................] - ETA: 52s - loss: 0.2940 - categorical_accuracy: 0.9075
25792/60000 [===========>..................] - ETA: 52s - loss: 0.2936 - categorical_accuracy: 0.9076
25856/60000 [===========>..................] - ETA: 52s - loss: 0.2931 - categorical_accuracy: 0.9077
25920/60000 [===========>..................] - ETA: 52s - loss: 0.2928 - categorical_accuracy: 0.9078
25952/60000 [===========>..................] - ETA: 52s - loss: 0.2926 - categorical_accuracy: 0.9078
26016/60000 [============>.................] - ETA: 52s - loss: 0.2925 - categorical_accuracy: 0.9079
26080/60000 [============>.................] - ETA: 52s - loss: 0.2920 - categorical_accuracy: 0.9081
26144/60000 [============>.................] - ETA: 52s - loss: 0.2915 - categorical_accuracy: 0.9082
26208/60000 [============>.................] - ETA: 52s - loss: 0.2909 - categorical_accuracy: 0.9084
26240/60000 [============>.................] - ETA: 52s - loss: 0.2908 - categorical_accuracy: 0.9085
26272/60000 [============>.................] - ETA: 52s - loss: 0.2905 - categorical_accuracy: 0.9086
26336/60000 [============>.................] - ETA: 52s - loss: 0.2899 - categorical_accuracy: 0.9088
26368/60000 [============>.................] - ETA: 51s - loss: 0.2896 - categorical_accuracy: 0.9089
26432/60000 [============>.................] - ETA: 51s - loss: 0.2894 - categorical_accuracy: 0.9090
26496/60000 [============>.................] - ETA: 51s - loss: 0.2892 - categorical_accuracy: 0.9091
26560/60000 [============>.................] - ETA: 51s - loss: 0.2887 - categorical_accuracy: 0.9093
26592/60000 [============>.................] - ETA: 51s - loss: 0.2884 - categorical_accuracy: 0.9094
26656/60000 [============>.................] - ETA: 51s - loss: 0.2879 - categorical_accuracy: 0.9095
26720/60000 [============>.................] - ETA: 51s - loss: 0.2875 - categorical_accuracy: 0.9096
26784/60000 [============>.................] - ETA: 51s - loss: 0.2869 - categorical_accuracy: 0.9098
26848/60000 [============>.................] - ETA: 51s - loss: 0.2866 - categorical_accuracy: 0.9099
26912/60000 [============>.................] - ETA: 51s - loss: 0.2863 - categorical_accuracy: 0.9099
26976/60000 [============>.................] - ETA: 51s - loss: 0.2861 - categorical_accuracy: 0.9100
27008/60000 [============>.................] - ETA: 50s - loss: 0.2859 - categorical_accuracy: 0.9100
27072/60000 [============>.................] - ETA: 50s - loss: 0.2856 - categorical_accuracy: 0.9101
27104/60000 [============>.................] - ETA: 50s - loss: 0.2855 - categorical_accuracy: 0.9102
27168/60000 [============>.................] - ETA: 50s - loss: 0.2850 - categorical_accuracy: 0.9103
27232/60000 [============>.................] - ETA: 50s - loss: 0.2847 - categorical_accuracy: 0.9104
27264/60000 [============>.................] - ETA: 50s - loss: 0.2846 - categorical_accuracy: 0.9104
27328/60000 [============>.................] - ETA: 50s - loss: 0.2840 - categorical_accuracy: 0.9106
27392/60000 [============>.................] - ETA: 50s - loss: 0.2834 - categorical_accuracy: 0.9108
27456/60000 [============>.................] - ETA: 50s - loss: 0.2829 - categorical_accuracy: 0.9109
27520/60000 [============>.................] - ETA: 50s - loss: 0.2827 - categorical_accuracy: 0.9110
27584/60000 [============>.................] - ETA: 50s - loss: 0.2824 - categorical_accuracy: 0.9112
27648/60000 [============>.................] - ETA: 49s - loss: 0.2822 - categorical_accuracy: 0.9112
27680/60000 [============>.................] - ETA: 49s - loss: 0.2819 - categorical_accuracy: 0.9113
27744/60000 [============>.................] - ETA: 49s - loss: 0.2815 - categorical_accuracy: 0.9114
27776/60000 [============>.................] - ETA: 49s - loss: 0.2815 - categorical_accuracy: 0.9114
27840/60000 [============>.................] - ETA: 49s - loss: 0.2814 - categorical_accuracy: 0.9115
27872/60000 [============>.................] - ETA: 49s - loss: 0.2815 - categorical_accuracy: 0.9115
27904/60000 [============>.................] - ETA: 49s - loss: 0.2812 - categorical_accuracy: 0.9115
27936/60000 [============>.................] - ETA: 49s - loss: 0.2812 - categorical_accuracy: 0.9115
27968/60000 [============>.................] - ETA: 49s - loss: 0.2811 - categorical_accuracy: 0.9115
28032/60000 [=============>................] - ETA: 49s - loss: 0.2808 - categorical_accuracy: 0.9116
28096/60000 [=============>................] - ETA: 49s - loss: 0.2803 - categorical_accuracy: 0.9117
28160/60000 [=============>................] - ETA: 49s - loss: 0.2798 - categorical_accuracy: 0.9119
28192/60000 [=============>................] - ETA: 49s - loss: 0.2799 - categorical_accuracy: 0.9119
28256/60000 [=============>................] - ETA: 49s - loss: 0.2797 - categorical_accuracy: 0.9119
28288/60000 [=============>................] - ETA: 49s - loss: 0.2794 - categorical_accuracy: 0.9120
28320/60000 [=============>................] - ETA: 48s - loss: 0.2795 - categorical_accuracy: 0.9120
28384/60000 [=============>................] - ETA: 48s - loss: 0.2791 - categorical_accuracy: 0.9122
28448/60000 [=============>................] - ETA: 48s - loss: 0.2786 - categorical_accuracy: 0.9123
28480/60000 [=============>................] - ETA: 48s - loss: 0.2784 - categorical_accuracy: 0.9124
28512/60000 [=============>................] - ETA: 48s - loss: 0.2784 - categorical_accuracy: 0.9125
28576/60000 [=============>................] - ETA: 48s - loss: 0.2782 - categorical_accuracy: 0.9126
28608/60000 [=============>................] - ETA: 48s - loss: 0.2779 - categorical_accuracy: 0.9126
28640/60000 [=============>................] - ETA: 48s - loss: 0.2779 - categorical_accuracy: 0.9126
28704/60000 [=============>................] - ETA: 48s - loss: 0.2775 - categorical_accuracy: 0.9128
28768/60000 [=============>................] - ETA: 48s - loss: 0.2773 - categorical_accuracy: 0.9129
28800/60000 [=============>................] - ETA: 48s - loss: 0.2770 - categorical_accuracy: 0.9130
28864/60000 [=============>................] - ETA: 48s - loss: 0.2765 - categorical_accuracy: 0.9131
28928/60000 [=============>................] - ETA: 48s - loss: 0.2760 - categorical_accuracy: 0.9133
28960/60000 [=============>................] - ETA: 47s - loss: 0.2758 - categorical_accuracy: 0.9133
28992/60000 [=============>................] - ETA: 47s - loss: 0.2758 - categorical_accuracy: 0.9133
29056/60000 [=============>................] - ETA: 47s - loss: 0.2758 - categorical_accuracy: 0.9134
29088/60000 [=============>................] - ETA: 47s - loss: 0.2757 - categorical_accuracy: 0.9134
29152/60000 [=============>................] - ETA: 47s - loss: 0.2753 - categorical_accuracy: 0.9136
29216/60000 [=============>................] - ETA: 47s - loss: 0.2750 - categorical_accuracy: 0.9136
29280/60000 [=============>................] - ETA: 47s - loss: 0.2747 - categorical_accuracy: 0.9137
29344/60000 [=============>................] - ETA: 47s - loss: 0.2743 - categorical_accuracy: 0.9139
29376/60000 [=============>................] - ETA: 47s - loss: 0.2741 - categorical_accuracy: 0.9139
29408/60000 [=============>................] - ETA: 47s - loss: 0.2738 - categorical_accuracy: 0.9140
29440/60000 [=============>................] - ETA: 47s - loss: 0.2737 - categorical_accuracy: 0.9140
29472/60000 [=============>................] - ETA: 47s - loss: 0.2735 - categorical_accuracy: 0.9141
29504/60000 [=============>................] - ETA: 47s - loss: 0.2733 - categorical_accuracy: 0.9141
29568/60000 [=============>................] - ETA: 47s - loss: 0.2729 - categorical_accuracy: 0.9143
29600/60000 [=============>................] - ETA: 47s - loss: 0.2727 - categorical_accuracy: 0.9143
29632/60000 [=============>................] - ETA: 46s - loss: 0.2726 - categorical_accuracy: 0.9143
29664/60000 [=============>................] - ETA: 46s - loss: 0.2725 - categorical_accuracy: 0.9143
29728/60000 [=============>................] - ETA: 46s - loss: 0.2720 - categorical_accuracy: 0.9145
29760/60000 [=============>................] - ETA: 46s - loss: 0.2719 - categorical_accuracy: 0.9145
29792/60000 [=============>................] - ETA: 46s - loss: 0.2717 - categorical_accuracy: 0.9145
29856/60000 [=============>................] - ETA: 46s - loss: 0.2711 - categorical_accuracy: 0.9147
29920/60000 [=============>................] - ETA: 46s - loss: 0.2707 - categorical_accuracy: 0.9148
29952/60000 [=============>................] - ETA: 46s - loss: 0.2706 - categorical_accuracy: 0.9148
29984/60000 [=============>................] - ETA: 46s - loss: 0.2704 - categorical_accuracy: 0.9149
30016/60000 [==============>...............] - ETA: 46s - loss: 0.2701 - categorical_accuracy: 0.9150
30080/60000 [==============>...............] - ETA: 46s - loss: 0.2697 - categorical_accuracy: 0.9152
30144/60000 [==============>...............] - ETA: 46s - loss: 0.2693 - categorical_accuracy: 0.9153
30208/60000 [==============>...............] - ETA: 46s - loss: 0.2689 - categorical_accuracy: 0.9154
30272/60000 [==============>...............] - ETA: 46s - loss: 0.2685 - categorical_accuracy: 0.9155
30304/60000 [==============>...............] - ETA: 45s - loss: 0.2685 - categorical_accuracy: 0.9156
30368/60000 [==============>...............] - ETA: 45s - loss: 0.2680 - categorical_accuracy: 0.9157
30400/60000 [==============>...............] - ETA: 45s - loss: 0.2679 - categorical_accuracy: 0.9157
30464/60000 [==============>...............] - ETA: 45s - loss: 0.2678 - categorical_accuracy: 0.9157
30528/60000 [==============>...............] - ETA: 45s - loss: 0.2675 - categorical_accuracy: 0.9159
30560/60000 [==============>...............] - ETA: 45s - loss: 0.2672 - categorical_accuracy: 0.9160
30624/60000 [==============>...............] - ETA: 45s - loss: 0.2668 - categorical_accuracy: 0.9161
30656/60000 [==============>...............] - ETA: 45s - loss: 0.2666 - categorical_accuracy: 0.9161
30720/60000 [==============>...............] - ETA: 45s - loss: 0.2662 - categorical_accuracy: 0.9162
30784/60000 [==============>...............] - ETA: 45s - loss: 0.2657 - categorical_accuracy: 0.9164
30848/60000 [==============>...............] - ETA: 45s - loss: 0.2652 - categorical_accuracy: 0.9165
30880/60000 [==============>...............] - ETA: 45s - loss: 0.2652 - categorical_accuracy: 0.9165
30944/60000 [==============>...............] - ETA: 44s - loss: 0.2649 - categorical_accuracy: 0.9166
31008/60000 [==============>...............] - ETA: 44s - loss: 0.2645 - categorical_accuracy: 0.9166
31072/60000 [==============>...............] - ETA: 44s - loss: 0.2642 - categorical_accuracy: 0.9167
31136/60000 [==============>...............] - ETA: 44s - loss: 0.2638 - categorical_accuracy: 0.9169
31200/60000 [==============>...............] - ETA: 44s - loss: 0.2634 - categorical_accuracy: 0.9170
31232/60000 [==============>...............] - ETA: 44s - loss: 0.2633 - categorical_accuracy: 0.9171
31264/60000 [==============>...............] - ETA: 44s - loss: 0.2632 - categorical_accuracy: 0.9171
31328/60000 [==============>...............] - ETA: 44s - loss: 0.2629 - categorical_accuracy: 0.9171
31392/60000 [==============>...............] - ETA: 44s - loss: 0.2626 - categorical_accuracy: 0.9172
31456/60000 [==============>...............] - ETA: 44s - loss: 0.2625 - categorical_accuracy: 0.9172
31520/60000 [==============>...............] - ETA: 44s - loss: 0.2623 - categorical_accuracy: 0.9173
31584/60000 [==============>...............] - ETA: 43s - loss: 0.2621 - categorical_accuracy: 0.9174
31616/60000 [==============>...............] - ETA: 43s - loss: 0.2619 - categorical_accuracy: 0.9174
31680/60000 [==============>...............] - ETA: 43s - loss: 0.2618 - categorical_accuracy: 0.9175
31712/60000 [==============>...............] - ETA: 43s - loss: 0.2616 - categorical_accuracy: 0.9176
31776/60000 [==============>...............] - ETA: 43s - loss: 0.2612 - categorical_accuracy: 0.9177
31840/60000 [==============>...............] - ETA: 43s - loss: 0.2609 - categorical_accuracy: 0.9178
31904/60000 [==============>...............] - ETA: 43s - loss: 0.2606 - categorical_accuracy: 0.9178
31968/60000 [==============>...............] - ETA: 43s - loss: 0.2603 - categorical_accuracy: 0.9179
32032/60000 [===============>..............] - ETA: 43s - loss: 0.2600 - categorical_accuracy: 0.9181
32096/60000 [===============>..............] - ETA: 43s - loss: 0.2596 - categorical_accuracy: 0.9182
32160/60000 [===============>..............] - ETA: 43s - loss: 0.2593 - categorical_accuracy: 0.9183
32224/60000 [===============>..............] - ETA: 42s - loss: 0.2590 - categorical_accuracy: 0.9183
32288/60000 [===============>..............] - ETA: 42s - loss: 0.2588 - categorical_accuracy: 0.9184
32352/60000 [===============>..............] - ETA: 42s - loss: 0.2584 - categorical_accuracy: 0.9185
32416/60000 [===============>..............] - ETA: 42s - loss: 0.2583 - categorical_accuracy: 0.9186
32480/60000 [===============>..............] - ETA: 42s - loss: 0.2582 - categorical_accuracy: 0.9186
32544/60000 [===============>..............] - ETA: 42s - loss: 0.2580 - categorical_accuracy: 0.9187
32608/60000 [===============>..............] - ETA: 42s - loss: 0.2579 - categorical_accuracy: 0.9187
32672/60000 [===============>..............] - ETA: 42s - loss: 0.2576 - categorical_accuracy: 0.9188
32736/60000 [===============>..............] - ETA: 42s - loss: 0.2572 - categorical_accuracy: 0.9189
32800/60000 [===============>..............] - ETA: 42s - loss: 0.2569 - categorical_accuracy: 0.9191
32864/60000 [===============>..............] - ETA: 41s - loss: 0.2567 - categorical_accuracy: 0.9191
32928/60000 [===============>..............] - ETA: 41s - loss: 0.2565 - categorical_accuracy: 0.9192
32992/60000 [===============>..............] - ETA: 41s - loss: 0.2560 - categorical_accuracy: 0.9193
33056/60000 [===============>..............] - ETA: 41s - loss: 0.2559 - categorical_accuracy: 0.9193
33120/60000 [===============>..............] - ETA: 41s - loss: 0.2557 - categorical_accuracy: 0.9194
33184/60000 [===============>..............] - ETA: 41s - loss: 0.2553 - categorical_accuracy: 0.9195
33248/60000 [===============>..............] - ETA: 41s - loss: 0.2551 - categorical_accuracy: 0.9196
33312/60000 [===============>..............] - ETA: 41s - loss: 0.2548 - categorical_accuracy: 0.9198
33376/60000 [===============>..............] - ETA: 41s - loss: 0.2545 - categorical_accuracy: 0.9199
33440/60000 [===============>..............] - ETA: 41s - loss: 0.2541 - categorical_accuracy: 0.9200
33504/60000 [===============>..............] - ETA: 40s - loss: 0.2537 - categorical_accuracy: 0.9201
33568/60000 [===============>..............] - ETA: 40s - loss: 0.2533 - categorical_accuracy: 0.9202
33632/60000 [===============>..............] - ETA: 40s - loss: 0.2529 - categorical_accuracy: 0.9204
33696/60000 [===============>..............] - ETA: 40s - loss: 0.2528 - categorical_accuracy: 0.9205
33760/60000 [===============>..............] - ETA: 40s - loss: 0.2525 - categorical_accuracy: 0.9206
33824/60000 [===============>..............] - ETA: 40s - loss: 0.2526 - categorical_accuracy: 0.9206
33856/60000 [===============>..............] - ETA: 40s - loss: 0.2526 - categorical_accuracy: 0.9207
33920/60000 [===============>..............] - ETA: 40s - loss: 0.2524 - categorical_accuracy: 0.9208
33984/60000 [===============>..............] - ETA: 40s - loss: 0.2523 - categorical_accuracy: 0.9208
34048/60000 [================>.............] - ETA: 40s - loss: 0.2520 - categorical_accuracy: 0.9209
34112/60000 [================>.............] - ETA: 39s - loss: 0.2518 - categorical_accuracy: 0.9209
34176/60000 [================>.............] - ETA: 39s - loss: 0.2516 - categorical_accuracy: 0.9209
34240/60000 [================>.............] - ETA: 39s - loss: 0.2517 - categorical_accuracy: 0.9210
34272/60000 [================>.............] - ETA: 39s - loss: 0.2515 - categorical_accuracy: 0.9210
34336/60000 [================>.............] - ETA: 39s - loss: 0.2516 - categorical_accuracy: 0.9210
34400/60000 [================>.............] - ETA: 39s - loss: 0.2514 - categorical_accuracy: 0.9211
34464/60000 [================>.............] - ETA: 39s - loss: 0.2510 - categorical_accuracy: 0.9212
34528/60000 [================>.............] - ETA: 39s - loss: 0.2510 - categorical_accuracy: 0.9212
34592/60000 [================>.............] - ETA: 39s - loss: 0.2510 - categorical_accuracy: 0.9213
34656/60000 [================>.............] - ETA: 39s - loss: 0.2507 - categorical_accuracy: 0.9213
34720/60000 [================>.............] - ETA: 39s - loss: 0.2505 - categorical_accuracy: 0.9214
34784/60000 [================>.............] - ETA: 38s - loss: 0.2501 - categorical_accuracy: 0.9215
34848/60000 [================>.............] - ETA: 38s - loss: 0.2503 - categorical_accuracy: 0.9215
34880/60000 [================>.............] - ETA: 38s - loss: 0.2501 - categorical_accuracy: 0.9216
34944/60000 [================>.............] - ETA: 38s - loss: 0.2498 - categorical_accuracy: 0.9216
35008/60000 [================>.............] - ETA: 38s - loss: 0.2494 - categorical_accuracy: 0.9218
35072/60000 [================>.............] - ETA: 38s - loss: 0.2493 - categorical_accuracy: 0.9218
35136/60000 [================>.............] - ETA: 38s - loss: 0.2489 - categorical_accuracy: 0.9219
35200/60000 [================>.............] - ETA: 38s - loss: 0.2486 - categorical_accuracy: 0.9220
35264/60000 [================>.............] - ETA: 38s - loss: 0.2482 - categorical_accuracy: 0.9220
35296/60000 [================>.............] - ETA: 38s - loss: 0.2481 - categorical_accuracy: 0.9221
35328/60000 [================>.............] - ETA: 38s - loss: 0.2479 - categorical_accuracy: 0.9222
35392/60000 [================>.............] - ETA: 37s - loss: 0.2477 - categorical_accuracy: 0.9222
35456/60000 [================>.............] - ETA: 37s - loss: 0.2475 - categorical_accuracy: 0.9222
35520/60000 [================>.............] - ETA: 37s - loss: 0.2471 - categorical_accuracy: 0.9224
35584/60000 [================>.............] - ETA: 37s - loss: 0.2470 - categorical_accuracy: 0.9224
35616/60000 [================>.............] - ETA: 37s - loss: 0.2468 - categorical_accuracy: 0.9225
35680/60000 [================>.............] - ETA: 37s - loss: 0.2465 - categorical_accuracy: 0.9226
35712/60000 [================>.............] - ETA: 37s - loss: 0.2463 - categorical_accuracy: 0.9226
35744/60000 [================>.............] - ETA: 37s - loss: 0.2462 - categorical_accuracy: 0.9226
35808/60000 [================>.............] - ETA: 37s - loss: 0.2460 - categorical_accuracy: 0.9227
35872/60000 [================>.............] - ETA: 37s - loss: 0.2456 - categorical_accuracy: 0.9228
35936/60000 [================>.............] - ETA: 37s - loss: 0.2455 - categorical_accuracy: 0.9229
36000/60000 [=================>............] - ETA: 36s - loss: 0.2454 - categorical_accuracy: 0.9229
36064/60000 [=================>............] - ETA: 36s - loss: 0.2454 - categorical_accuracy: 0.9229
36128/60000 [=================>............] - ETA: 36s - loss: 0.2451 - categorical_accuracy: 0.9230
36160/60000 [=================>............] - ETA: 36s - loss: 0.2452 - categorical_accuracy: 0.9230
36192/60000 [=================>............] - ETA: 36s - loss: 0.2451 - categorical_accuracy: 0.9231
36256/60000 [=================>............] - ETA: 36s - loss: 0.2448 - categorical_accuracy: 0.9232
36288/60000 [=================>............] - ETA: 36s - loss: 0.2448 - categorical_accuracy: 0.9232
36320/60000 [=================>............] - ETA: 36s - loss: 0.2447 - categorical_accuracy: 0.9232
36352/60000 [=================>............] - ETA: 36s - loss: 0.2447 - categorical_accuracy: 0.9232
36416/60000 [=================>............] - ETA: 36s - loss: 0.2445 - categorical_accuracy: 0.9233
36448/60000 [=================>............] - ETA: 36s - loss: 0.2443 - categorical_accuracy: 0.9234
36512/60000 [=================>............] - ETA: 36s - loss: 0.2439 - categorical_accuracy: 0.9235
36544/60000 [=================>............] - ETA: 36s - loss: 0.2437 - categorical_accuracy: 0.9236
36576/60000 [=================>............] - ETA: 36s - loss: 0.2436 - categorical_accuracy: 0.9236
36640/60000 [=================>............] - ETA: 36s - loss: 0.2434 - categorical_accuracy: 0.9237
36672/60000 [=================>............] - ETA: 35s - loss: 0.2433 - categorical_accuracy: 0.9237
36736/60000 [=================>............] - ETA: 35s - loss: 0.2436 - categorical_accuracy: 0.9237
36800/60000 [=================>............] - ETA: 35s - loss: 0.2435 - categorical_accuracy: 0.9237
36832/60000 [=================>............] - ETA: 35s - loss: 0.2433 - categorical_accuracy: 0.9238
36896/60000 [=================>............] - ETA: 35s - loss: 0.2431 - categorical_accuracy: 0.9238
36928/60000 [=================>............] - ETA: 35s - loss: 0.2429 - categorical_accuracy: 0.9239
36992/60000 [=================>............] - ETA: 35s - loss: 0.2427 - categorical_accuracy: 0.9240
37056/60000 [=================>............] - ETA: 35s - loss: 0.2423 - categorical_accuracy: 0.9241
37120/60000 [=================>............] - ETA: 35s - loss: 0.2422 - categorical_accuracy: 0.9241
37152/60000 [=================>............] - ETA: 35s - loss: 0.2421 - categorical_accuracy: 0.9241
37216/60000 [=================>............] - ETA: 35s - loss: 0.2420 - categorical_accuracy: 0.9242
37248/60000 [=================>............] - ETA: 35s - loss: 0.2420 - categorical_accuracy: 0.9242
37280/60000 [=================>............] - ETA: 35s - loss: 0.2419 - categorical_accuracy: 0.9242
37344/60000 [=================>............] - ETA: 34s - loss: 0.2416 - categorical_accuracy: 0.9243
37408/60000 [=================>............] - ETA: 34s - loss: 0.2414 - categorical_accuracy: 0.9244
37472/60000 [=================>............] - ETA: 34s - loss: 0.2411 - categorical_accuracy: 0.9245
37536/60000 [=================>............] - ETA: 34s - loss: 0.2408 - categorical_accuracy: 0.9246
37568/60000 [=================>............] - ETA: 34s - loss: 0.2407 - categorical_accuracy: 0.9246
37632/60000 [=================>............] - ETA: 34s - loss: 0.2405 - categorical_accuracy: 0.9246
37696/60000 [=================>............] - ETA: 34s - loss: 0.2402 - categorical_accuracy: 0.9247
37760/60000 [=================>............] - ETA: 34s - loss: 0.2400 - categorical_accuracy: 0.9248
37824/60000 [=================>............] - ETA: 34s - loss: 0.2398 - categorical_accuracy: 0.9249
37888/60000 [=================>............] - ETA: 34s - loss: 0.2396 - categorical_accuracy: 0.9249
37920/60000 [=================>............] - ETA: 34s - loss: 0.2397 - categorical_accuracy: 0.9249
37952/60000 [=================>............] - ETA: 33s - loss: 0.2396 - categorical_accuracy: 0.9250
38016/60000 [==================>...........] - ETA: 33s - loss: 0.2393 - categorical_accuracy: 0.9250
38048/60000 [==================>...........] - ETA: 33s - loss: 0.2391 - categorical_accuracy: 0.9251
38080/60000 [==================>...........] - ETA: 33s - loss: 0.2391 - categorical_accuracy: 0.9251
38144/60000 [==================>...........] - ETA: 33s - loss: 0.2388 - categorical_accuracy: 0.9252
38208/60000 [==================>...........] - ETA: 33s - loss: 0.2385 - categorical_accuracy: 0.9253
38240/60000 [==================>...........] - ETA: 33s - loss: 0.2384 - categorical_accuracy: 0.9253
38304/60000 [==================>...........] - ETA: 33s - loss: 0.2383 - categorical_accuracy: 0.9254
38368/60000 [==================>...........] - ETA: 33s - loss: 0.2380 - categorical_accuracy: 0.9255
38400/60000 [==================>...........] - ETA: 33s - loss: 0.2379 - categorical_accuracy: 0.9255
38464/60000 [==================>...........] - ETA: 33s - loss: 0.2377 - categorical_accuracy: 0.9255
38528/60000 [==================>...........] - ETA: 33s - loss: 0.2375 - categorical_accuracy: 0.9256
38592/60000 [==================>...........] - ETA: 33s - loss: 0.2375 - categorical_accuracy: 0.9256
38656/60000 [==================>...........] - ETA: 32s - loss: 0.2373 - categorical_accuracy: 0.9257
38720/60000 [==================>...........] - ETA: 32s - loss: 0.2370 - categorical_accuracy: 0.9257
38784/60000 [==================>...........] - ETA: 32s - loss: 0.2367 - categorical_accuracy: 0.9258
38816/60000 [==================>...........] - ETA: 32s - loss: 0.2365 - categorical_accuracy: 0.9259
38848/60000 [==================>...........] - ETA: 32s - loss: 0.2364 - categorical_accuracy: 0.9259
38912/60000 [==================>...........] - ETA: 32s - loss: 0.2362 - categorical_accuracy: 0.9260
38976/60000 [==================>...........] - ETA: 32s - loss: 0.2360 - categorical_accuracy: 0.9260
39040/60000 [==================>...........] - ETA: 32s - loss: 0.2358 - categorical_accuracy: 0.9261
39072/60000 [==================>...........] - ETA: 32s - loss: 0.2356 - categorical_accuracy: 0.9261
39136/60000 [==================>...........] - ETA: 32s - loss: 0.2355 - categorical_accuracy: 0.9261
39200/60000 [==================>...........] - ETA: 32s - loss: 0.2352 - categorical_accuracy: 0.9262
39264/60000 [==================>...........] - ETA: 31s - loss: 0.2349 - categorical_accuracy: 0.9263
39296/60000 [==================>...........] - ETA: 31s - loss: 0.2348 - categorical_accuracy: 0.9264
39360/60000 [==================>...........] - ETA: 31s - loss: 0.2346 - categorical_accuracy: 0.9264
39424/60000 [==================>...........] - ETA: 31s - loss: 0.2343 - categorical_accuracy: 0.9265
39488/60000 [==================>...........] - ETA: 31s - loss: 0.2340 - categorical_accuracy: 0.9266
39552/60000 [==================>...........] - ETA: 31s - loss: 0.2337 - categorical_accuracy: 0.9267
39616/60000 [==================>...........] - ETA: 31s - loss: 0.2334 - categorical_accuracy: 0.9267
39680/60000 [==================>...........] - ETA: 31s - loss: 0.2332 - categorical_accuracy: 0.9268
39712/60000 [==================>...........] - ETA: 31s - loss: 0.2331 - categorical_accuracy: 0.9268
39744/60000 [==================>...........] - ETA: 31s - loss: 0.2329 - categorical_accuracy: 0.9269
39776/60000 [==================>...........] - ETA: 31s - loss: 0.2329 - categorical_accuracy: 0.9269
39840/60000 [==================>...........] - ETA: 31s - loss: 0.2327 - categorical_accuracy: 0.9269
39872/60000 [==================>...........] - ETA: 31s - loss: 0.2325 - categorical_accuracy: 0.9270
39904/60000 [==================>...........] - ETA: 30s - loss: 0.2324 - categorical_accuracy: 0.9270
39968/60000 [==================>...........] - ETA: 30s - loss: 0.2322 - categorical_accuracy: 0.9271
40000/60000 [===================>..........] - ETA: 30s - loss: 0.2324 - categorical_accuracy: 0.9271
40032/60000 [===================>..........] - ETA: 30s - loss: 0.2323 - categorical_accuracy: 0.9271
40064/60000 [===================>..........] - ETA: 30s - loss: 0.2322 - categorical_accuracy: 0.9271
40096/60000 [===================>..........] - ETA: 30s - loss: 0.2321 - categorical_accuracy: 0.9272
40160/60000 [===================>..........] - ETA: 30s - loss: 0.2317 - categorical_accuracy: 0.9273
40192/60000 [===================>..........] - ETA: 30s - loss: 0.2316 - categorical_accuracy: 0.9273
40256/60000 [===================>..........] - ETA: 30s - loss: 0.2318 - categorical_accuracy: 0.9273
40320/60000 [===================>..........] - ETA: 30s - loss: 0.2316 - categorical_accuracy: 0.9274
40384/60000 [===================>..........] - ETA: 30s - loss: 0.2315 - categorical_accuracy: 0.9274
40416/60000 [===================>..........] - ETA: 30s - loss: 0.2314 - categorical_accuracy: 0.9275
40480/60000 [===================>..........] - ETA: 30s - loss: 0.2313 - categorical_accuracy: 0.9275
40544/60000 [===================>..........] - ETA: 29s - loss: 0.2313 - categorical_accuracy: 0.9276
40608/60000 [===================>..........] - ETA: 29s - loss: 0.2310 - categorical_accuracy: 0.9276
40672/60000 [===================>..........] - ETA: 29s - loss: 0.2308 - categorical_accuracy: 0.9277
40736/60000 [===================>..........] - ETA: 29s - loss: 0.2305 - categorical_accuracy: 0.9278
40800/60000 [===================>..........] - ETA: 29s - loss: 0.2303 - categorical_accuracy: 0.9278
40832/60000 [===================>..........] - ETA: 29s - loss: 0.2302 - categorical_accuracy: 0.9279
40896/60000 [===================>..........] - ETA: 29s - loss: 0.2300 - categorical_accuracy: 0.9280
40928/60000 [===================>..........] - ETA: 29s - loss: 0.2299 - categorical_accuracy: 0.9279
40960/60000 [===================>..........] - ETA: 29s - loss: 0.2301 - categorical_accuracy: 0.9280
41024/60000 [===================>..........] - ETA: 29s - loss: 0.2300 - categorical_accuracy: 0.9280
41088/60000 [===================>..........] - ETA: 29s - loss: 0.2298 - categorical_accuracy: 0.9281
41152/60000 [===================>..........] - ETA: 29s - loss: 0.2296 - categorical_accuracy: 0.9281
41216/60000 [===================>..........] - ETA: 28s - loss: 0.2295 - categorical_accuracy: 0.9281
41280/60000 [===================>..........] - ETA: 28s - loss: 0.2293 - categorical_accuracy: 0.9282
41312/60000 [===================>..........] - ETA: 28s - loss: 0.2292 - categorical_accuracy: 0.9282
41376/60000 [===================>..........] - ETA: 28s - loss: 0.2290 - categorical_accuracy: 0.9282
41440/60000 [===================>..........] - ETA: 28s - loss: 0.2289 - categorical_accuracy: 0.9283
41504/60000 [===================>..........] - ETA: 28s - loss: 0.2287 - categorical_accuracy: 0.9283
41568/60000 [===================>..........] - ETA: 28s - loss: 0.2285 - categorical_accuracy: 0.9284
41632/60000 [===================>..........] - ETA: 28s - loss: 0.2282 - categorical_accuracy: 0.9285
41696/60000 [===================>..........] - ETA: 28s - loss: 0.2280 - categorical_accuracy: 0.9285
41760/60000 [===================>..........] - ETA: 28s - loss: 0.2278 - categorical_accuracy: 0.9286
41824/60000 [===================>..........] - ETA: 27s - loss: 0.2277 - categorical_accuracy: 0.9286
41888/60000 [===================>..........] - ETA: 27s - loss: 0.2275 - categorical_accuracy: 0.9287
41952/60000 [===================>..........] - ETA: 27s - loss: 0.2276 - categorical_accuracy: 0.9287
42016/60000 [====================>.........] - ETA: 27s - loss: 0.2274 - categorical_accuracy: 0.9288
42080/60000 [====================>.........] - ETA: 27s - loss: 0.2272 - categorical_accuracy: 0.9288
42144/60000 [====================>.........] - ETA: 27s - loss: 0.2270 - categorical_accuracy: 0.9289
42208/60000 [====================>.........] - ETA: 27s - loss: 0.2270 - categorical_accuracy: 0.9289
42272/60000 [====================>.........] - ETA: 27s - loss: 0.2268 - categorical_accuracy: 0.9290
42336/60000 [====================>.........] - ETA: 27s - loss: 0.2266 - categorical_accuracy: 0.9290
42400/60000 [====================>.........] - ETA: 27s - loss: 0.2265 - categorical_accuracy: 0.9291
42464/60000 [====================>.........] - ETA: 26s - loss: 0.2263 - categorical_accuracy: 0.9291
42528/60000 [====================>.........] - ETA: 26s - loss: 0.2261 - categorical_accuracy: 0.9292
42592/60000 [====================>.........] - ETA: 26s - loss: 0.2258 - categorical_accuracy: 0.9293
42624/60000 [====================>.........] - ETA: 26s - loss: 0.2260 - categorical_accuracy: 0.9293
42688/60000 [====================>.........] - ETA: 26s - loss: 0.2259 - categorical_accuracy: 0.9293
42720/60000 [====================>.........] - ETA: 26s - loss: 0.2260 - categorical_accuracy: 0.9293
42784/60000 [====================>.........] - ETA: 26s - loss: 0.2258 - categorical_accuracy: 0.9294
42848/60000 [====================>.........] - ETA: 26s - loss: 0.2256 - categorical_accuracy: 0.9294
42912/60000 [====================>.........] - ETA: 26s - loss: 0.2254 - categorical_accuracy: 0.9295
42944/60000 [====================>.........] - ETA: 26s - loss: 0.2253 - categorical_accuracy: 0.9295
43008/60000 [====================>.........] - ETA: 26s - loss: 0.2252 - categorical_accuracy: 0.9295
43072/60000 [====================>.........] - ETA: 26s - loss: 0.2250 - categorical_accuracy: 0.9296
43104/60000 [====================>.........] - ETA: 25s - loss: 0.2249 - categorical_accuracy: 0.9296
43168/60000 [====================>.........] - ETA: 25s - loss: 0.2248 - categorical_accuracy: 0.9296
43200/60000 [====================>.........] - ETA: 25s - loss: 0.2248 - categorical_accuracy: 0.9297
43264/60000 [====================>.........] - ETA: 25s - loss: 0.2246 - categorical_accuracy: 0.9297
43328/60000 [====================>.........] - ETA: 25s - loss: 0.2244 - categorical_accuracy: 0.9297
43392/60000 [====================>.........] - ETA: 25s - loss: 0.2243 - categorical_accuracy: 0.9298
43456/60000 [====================>.........] - ETA: 25s - loss: 0.2241 - categorical_accuracy: 0.9299
43520/60000 [====================>.........] - ETA: 25s - loss: 0.2238 - categorical_accuracy: 0.9300
43584/60000 [====================>.........] - ETA: 25s - loss: 0.2236 - categorical_accuracy: 0.9301
43648/60000 [====================>.........] - ETA: 25s - loss: 0.2234 - categorical_accuracy: 0.9301
43712/60000 [====================>.........] - ETA: 25s - loss: 0.2232 - categorical_accuracy: 0.9301
43744/60000 [====================>.........] - ETA: 25s - loss: 0.2231 - categorical_accuracy: 0.9302
43776/60000 [====================>.........] - ETA: 24s - loss: 0.2229 - categorical_accuracy: 0.9302
43840/60000 [====================>.........] - ETA: 24s - loss: 0.2228 - categorical_accuracy: 0.9303
43904/60000 [====================>.........] - ETA: 24s - loss: 0.2226 - categorical_accuracy: 0.9303
43968/60000 [====================>.........] - ETA: 24s - loss: 0.2224 - categorical_accuracy: 0.9304
44032/60000 [=====================>........] - ETA: 24s - loss: 0.2223 - categorical_accuracy: 0.9304
44064/60000 [=====================>........] - ETA: 24s - loss: 0.2223 - categorical_accuracy: 0.9305
44128/60000 [=====================>........] - ETA: 24s - loss: 0.2220 - categorical_accuracy: 0.9305
44192/60000 [=====================>........] - ETA: 24s - loss: 0.2219 - categorical_accuracy: 0.9306
44256/60000 [=====================>........] - ETA: 24s - loss: 0.2216 - categorical_accuracy: 0.9306
44320/60000 [=====================>........] - ETA: 24s - loss: 0.2216 - categorical_accuracy: 0.9307
44352/60000 [=====================>........] - ETA: 24s - loss: 0.2215 - categorical_accuracy: 0.9307
44416/60000 [=====================>........] - ETA: 23s - loss: 0.2213 - categorical_accuracy: 0.9307
44480/60000 [=====================>........] - ETA: 23s - loss: 0.2210 - categorical_accuracy: 0.9308
44544/60000 [=====================>........] - ETA: 23s - loss: 0.2210 - categorical_accuracy: 0.9309
44608/60000 [=====================>........] - ETA: 23s - loss: 0.2207 - categorical_accuracy: 0.9310
44672/60000 [=====================>........] - ETA: 23s - loss: 0.2207 - categorical_accuracy: 0.9310
44736/60000 [=====================>........] - ETA: 23s - loss: 0.2204 - categorical_accuracy: 0.9310
44800/60000 [=====================>........] - ETA: 23s - loss: 0.2202 - categorical_accuracy: 0.9311
44864/60000 [=====================>........] - ETA: 23s - loss: 0.2202 - categorical_accuracy: 0.9311
44928/60000 [=====================>........] - ETA: 23s - loss: 0.2200 - categorical_accuracy: 0.9312
44992/60000 [=====================>........] - ETA: 23s - loss: 0.2197 - categorical_accuracy: 0.9313
45056/60000 [=====================>........] - ETA: 22s - loss: 0.2195 - categorical_accuracy: 0.9314
45120/60000 [=====================>........] - ETA: 22s - loss: 0.2194 - categorical_accuracy: 0.9314
45184/60000 [=====================>........] - ETA: 22s - loss: 0.2192 - categorical_accuracy: 0.9314
45248/60000 [=====================>........] - ETA: 22s - loss: 0.2190 - categorical_accuracy: 0.9315
45312/60000 [=====================>........] - ETA: 22s - loss: 0.2188 - categorical_accuracy: 0.9315
45376/60000 [=====================>........] - ETA: 22s - loss: 0.2185 - categorical_accuracy: 0.9316
45408/60000 [=====================>........] - ETA: 22s - loss: 0.2185 - categorical_accuracy: 0.9316
45472/60000 [=====================>........] - ETA: 22s - loss: 0.2183 - categorical_accuracy: 0.9316
45536/60000 [=====================>........] - ETA: 22s - loss: 0.2181 - categorical_accuracy: 0.9317
45600/60000 [=====================>........] - ETA: 22s - loss: 0.2180 - categorical_accuracy: 0.9317
45664/60000 [=====================>........] - ETA: 22s - loss: 0.2181 - categorical_accuracy: 0.9317
45728/60000 [=====================>........] - ETA: 21s - loss: 0.2180 - categorical_accuracy: 0.9317
45792/60000 [=====================>........] - ETA: 21s - loss: 0.2177 - categorical_accuracy: 0.9318
45856/60000 [=====================>........] - ETA: 21s - loss: 0.2175 - categorical_accuracy: 0.9319
45920/60000 [=====================>........] - ETA: 21s - loss: 0.2173 - categorical_accuracy: 0.9320
45984/60000 [=====================>........] - ETA: 21s - loss: 0.2170 - categorical_accuracy: 0.9320
46048/60000 [======================>.......] - ETA: 21s - loss: 0.2169 - categorical_accuracy: 0.9321
46112/60000 [======================>.......] - ETA: 21s - loss: 0.2169 - categorical_accuracy: 0.9321
46144/60000 [======================>.......] - ETA: 21s - loss: 0.2168 - categorical_accuracy: 0.9321
46208/60000 [======================>.......] - ETA: 21s - loss: 0.2165 - categorical_accuracy: 0.9322
46272/60000 [======================>.......] - ETA: 21s - loss: 0.2164 - categorical_accuracy: 0.9323
46336/60000 [======================>.......] - ETA: 20s - loss: 0.2163 - categorical_accuracy: 0.9324
46400/60000 [======================>.......] - ETA: 20s - loss: 0.2162 - categorical_accuracy: 0.9325
46432/60000 [======================>.......] - ETA: 20s - loss: 0.2161 - categorical_accuracy: 0.9325
46496/60000 [======================>.......] - ETA: 20s - loss: 0.2161 - categorical_accuracy: 0.9325
46560/60000 [======================>.......] - ETA: 20s - loss: 0.2159 - categorical_accuracy: 0.9325
46624/60000 [======================>.......] - ETA: 20s - loss: 0.2159 - categorical_accuracy: 0.9326
46688/60000 [======================>.......] - ETA: 20s - loss: 0.2157 - categorical_accuracy: 0.9327
46752/60000 [======================>.......] - ETA: 20s - loss: 0.2154 - categorical_accuracy: 0.9327
46784/60000 [======================>.......] - ETA: 20s - loss: 0.2154 - categorical_accuracy: 0.9328
46816/60000 [======================>.......] - ETA: 20s - loss: 0.2153 - categorical_accuracy: 0.9328
46848/60000 [======================>.......] - ETA: 20s - loss: 0.2155 - categorical_accuracy: 0.9328
46880/60000 [======================>.......] - ETA: 20s - loss: 0.2154 - categorical_accuracy: 0.9328
46944/60000 [======================>.......] - ETA: 20s - loss: 0.2152 - categorical_accuracy: 0.9329
47008/60000 [======================>.......] - ETA: 19s - loss: 0.2150 - categorical_accuracy: 0.9330
47072/60000 [======================>.......] - ETA: 19s - loss: 0.2149 - categorical_accuracy: 0.9330
47136/60000 [======================>.......] - ETA: 19s - loss: 0.2146 - categorical_accuracy: 0.9331
47200/60000 [======================>.......] - ETA: 19s - loss: 0.2144 - categorical_accuracy: 0.9332
47264/60000 [======================>.......] - ETA: 19s - loss: 0.2142 - categorical_accuracy: 0.9332
47328/60000 [======================>.......] - ETA: 19s - loss: 0.2141 - categorical_accuracy: 0.9333
47392/60000 [======================>.......] - ETA: 19s - loss: 0.2139 - categorical_accuracy: 0.9334
47424/60000 [======================>.......] - ETA: 19s - loss: 0.2139 - categorical_accuracy: 0.9333
47488/60000 [======================>.......] - ETA: 19s - loss: 0.2138 - categorical_accuracy: 0.9334
47552/60000 [======================>.......] - ETA: 19s - loss: 0.2136 - categorical_accuracy: 0.9334
47584/60000 [======================>.......] - ETA: 19s - loss: 0.2135 - categorical_accuracy: 0.9335
47616/60000 [======================>.......] - ETA: 19s - loss: 0.2134 - categorical_accuracy: 0.9335
47680/60000 [======================>.......] - ETA: 18s - loss: 0.2134 - categorical_accuracy: 0.9336
47744/60000 [======================>.......] - ETA: 18s - loss: 0.2135 - categorical_accuracy: 0.9336
47808/60000 [======================>.......] - ETA: 18s - loss: 0.2132 - categorical_accuracy: 0.9337
47872/60000 [======================>.......] - ETA: 18s - loss: 0.2130 - categorical_accuracy: 0.9338
47936/60000 [======================>.......] - ETA: 18s - loss: 0.2128 - categorical_accuracy: 0.9338
48000/60000 [=======================>......] - ETA: 18s - loss: 0.2127 - categorical_accuracy: 0.9339
48064/60000 [=======================>......] - ETA: 18s - loss: 0.2125 - categorical_accuracy: 0.9339
48128/60000 [=======================>......] - ETA: 18s - loss: 0.2125 - categorical_accuracy: 0.9339
48192/60000 [=======================>......] - ETA: 18s - loss: 0.2124 - categorical_accuracy: 0.9340
48256/60000 [=======================>......] - ETA: 18s - loss: 0.2122 - categorical_accuracy: 0.9341
48320/60000 [=======================>......] - ETA: 17s - loss: 0.2120 - categorical_accuracy: 0.9341
48384/60000 [=======================>......] - ETA: 17s - loss: 0.2119 - categorical_accuracy: 0.9342
48448/60000 [=======================>......] - ETA: 17s - loss: 0.2118 - categorical_accuracy: 0.9342
48480/60000 [=======================>......] - ETA: 17s - loss: 0.2117 - categorical_accuracy: 0.9342
48544/60000 [=======================>......] - ETA: 17s - loss: 0.2116 - categorical_accuracy: 0.9343
48608/60000 [=======================>......] - ETA: 17s - loss: 0.2114 - categorical_accuracy: 0.9344
48672/60000 [=======================>......] - ETA: 17s - loss: 0.2112 - categorical_accuracy: 0.9345
48736/60000 [=======================>......] - ETA: 17s - loss: 0.2111 - categorical_accuracy: 0.9345
48800/60000 [=======================>......] - ETA: 17s - loss: 0.2109 - categorical_accuracy: 0.9345
48864/60000 [=======================>......] - ETA: 17s - loss: 0.2107 - categorical_accuracy: 0.9346
48928/60000 [=======================>......] - ETA: 16s - loss: 0.2106 - categorical_accuracy: 0.9346
48992/60000 [=======================>......] - ETA: 16s - loss: 0.2104 - categorical_accuracy: 0.9347
49056/60000 [=======================>......] - ETA: 16s - loss: 0.2103 - categorical_accuracy: 0.9347
49120/60000 [=======================>......] - ETA: 16s - loss: 0.2102 - categorical_accuracy: 0.9348
49152/60000 [=======================>......] - ETA: 16s - loss: 0.2100 - categorical_accuracy: 0.9348
49216/60000 [=======================>......] - ETA: 16s - loss: 0.2098 - categorical_accuracy: 0.9349
49280/60000 [=======================>......] - ETA: 16s - loss: 0.2098 - categorical_accuracy: 0.9349
49344/60000 [=======================>......] - ETA: 16s - loss: 0.2096 - categorical_accuracy: 0.9349
49408/60000 [=======================>......] - ETA: 16s - loss: 0.2093 - categorical_accuracy: 0.9350
49472/60000 [=======================>......] - ETA: 16s - loss: 0.2095 - categorical_accuracy: 0.9351
49504/60000 [=======================>......] - ETA: 16s - loss: 0.2094 - categorical_accuracy: 0.9351
49536/60000 [=======================>......] - ETA: 16s - loss: 0.2093 - categorical_accuracy: 0.9351
49600/60000 [=======================>......] - ETA: 15s - loss: 0.2092 - categorical_accuracy: 0.9351
49632/60000 [=======================>......] - ETA: 15s - loss: 0.2092 - categorical_accuracy: 0.9351
49664/60000 [=======================>......] - ETA: 15s - loss: 0.2091 - categorical_accuracy: 0.9352
49728/60000 [=======================>......] - ETA: 15s - loss: 0.2090 - categorical_accuracy: 0.9352
49792/60000 [=======================>......] - ETA: 15s - loss: 0.2089 - categorical_accuracy: 0.9352
49824/60000 [=======================>......] - ETA: 15s - loss: 0.2089 - categorical_accuracy: 0.9353
49856/60000 [=======================>......] - ETA: 15s - loss: 0.2088 - categorical_accuracy: 0.9353
49888/60000 [=======================>......] - ETA: 15s - loss: 0.2087 - categorical_accuracy: 0.9353
49952/60000 [=======================>......] - ETA: 15s - loss: 0.2085 - categorical_accuracy: 0.9354
50016/60000 [========================>.....] - ETA: 15s - loss: 0.2082 - categorical_accuracy: 0.9355
50080/60000 [========================>.....] - ETA: 15s - loss: 0.2080 - categorical_accuracy: 0.9355
50144/60000 [========================>.....] - ETA: 15s - loss: 0.2080 - categorical_accuracy: 0.9356
50208/60000 [========================>.....] - ETA: 14s - loss: 0.2078 - categorical_accuracy: 0.9356
50272/60000 [========================>.....] - ETA: 14s - loss: 0.2076 - categorical_accuracy: 0.9357
50336/60000 [========================>.....] - ETA: 14s - loss: 0.2076 - categorical_accuracy: 0.9357
50400/60000 [========================>.....] - ETA: 14s - loss: 0.2075 - categorical_accuracy: 0.9357
50464/60000 [========================>.....] - ETA: 14s - loss: 0.2074 - categorical_accuracy: 0.9358
50496/60000 [========================>.....] - ETA: 14s - loss: 0.2073 - categorical_accuracy: 0.9358
50528/60000 [========================>.....] - ETA: 14s - loss: 0.2073 - categorical_accuracy: 0.9358
50592/60000 [========================>.....] - ETA: 14s - loss: 0.2071 - categorical_accuracy: 0.9359
50656/60000 [========================>.....] - ETA: 14s - loss: 0.2069 - categorical_accuracy: 0.9359
50720/60000 [========================>.....] - ETA: 14s - loss: 0.2067 - categorical_accuracy: 0.9360
50784/60000 [========================>.....] - ETA: 14s - loss: 0.2066 - categorical_accuracy: 0.9360
50848/60000 [========================>.....] - ETA: 14s - loss: 0.2065 - categorical_accuracy: 0.9360
50912/60000 [========================>.....] - ETA: 13s - loss: 0.2063 - categorical_accuracy: 0.9361
50976/60000 [========================>.....] - ETA: 13s - loss: 0.2060 - categorical_accuracy: 0.9362
51040/60000 [========================>.....] - ETA: 13s - loss: 0.2058 - categorical_accuracy: 0.9363
51104/60000 [========================>.....] - ETA: 13s - loss: 0.2057 - categorical_accuracy: 0.9363
51168/60000 [========================>.....] - ETA: 13s - loss: 0.2056 - categorical_accuracy: 0.9363
51232/60000 [========================>.....] - ETA: 13s - loss: 0.2055 - categorical_accuracy: 0.9364
51296/60000 [========================>.....] - ETA: 13s - loss: 0.2052 - categorical_accuracy: 0.9365
51360/60000 [========================>.....] - ETA: 13s - loss: 0.2053 - categorical_accuracy: 0.9365
51424/60000 [========================>.....] - ETA: 13s - loss: 0.2051 - categorical_accuracy: 0.9365
51488/60000 [========================>.....] - ETA: 13s - loss: 0.2049 - categorical_accuracy: 0.9366
51552/60000 [========================>.....] - ETA: 12s - loss: 0.2048 - categorical_accuracy: 0.9367
51616/60000 [========================>.....] - ETA: 12s - loss: 0.2046 - categorical_accuracy: 0.9367
51680/60000 [========================>.....] - ETA: 12s - loss: 0.2043 - categorical_accuracy: 0.9368
51744/60000 [========================>.....] - ETA: 12s - loss: 0.2042 - categorical_accuracy: 0.9368
51776/60000 [========================>.....] - ETA: 12s - loss: 0.2041 - categorical_accuracy: 0.9369
51840/60000 [========================>.....] - ETA: 12s - loss: 0.2039 - categorical_accuracy: 0.9369
51904/60000 [========================>.....] - ETA: 12s - loss: 0.2037 - categorical_accuracy: 0.9370
51968/60000 [========================>.....] - ETA: 12s - loss: 0.2037 - categorical_accuracy: 0.9370
52032/60000 [=========================>....] - ETA: 12s - loss: 0.2035 - categorical_accuracy: 0.9370
52096/60000 [=========================>....] - ETA: 12s - loss: 0.2034 - categorical_accuracy: 0.9371
52160/60000 [=========================>....] - ETA: 11s - loss: 0.2032 - categorical_accuracy: 0.9371
52224/60000 [=========================>....] - ETA: 11s - loss: 0.2033 - categorical_accuracy: 0.9371
52288/60000 [=========================>....] - ETA: 11s - loss: 0.2034 - categorical_accuracy: 0.9371
52352/60000 [=========================>....] - ETA: 11s - loss: 0.2032 - categorical_accuracy: 0.9371
52384/60000 [=========================>....] - ETA: 11s - loss: 0.2032 - categorical_accuracy: 0.9371
52448/60000 [=========================>....] - ETA: 11s - loss: 0.2030 - categorical_accuracy: 0.9372
52512/60000 [=========================>....] - ETA: 11s - loss: 0.2029 - categorical_accuracy: 0.9372
52576/60000 [=========================>....] - ETA: 11s - loss: 0.2029 - categorical_accuracy: 0.9373
52608/60000 [=========================>....] - ETA: 11s - loss: 0.2028 - categorical_accuracy: 0.9373
52672/60000 [=========================>....] - ETA: 11s - loss: 0.2026 - categorical_accuracy: 0.9373
52736/60000 [=========================>....] - ETA: 11s - loss: 0.2024 - categorical_accuracy: 0.9374
52800/60000 [=========================>....] - ETA: 11s - loss: 0.2023 - categorical_accuracy: 0.9374
52864/60000 [=========================>....] - ETA: 10s - loss: 0.2021 - categorical_accuracy: 0.9375
52896/60000 [=========================>....] - ETA: 10s - loss: 0.2020 - categorical_accuracy: 0.9375
52960/60000 [=========================>....] - ETA: 10s - loss: 0.2019 - categorical_accuracy: 0.9376
53024/60000 [=========================>....] - ETA: 10s - loss: 0.2019 - categorical_accuracy: 0.9376
53056/60000 [=========================>....] - ETA: 10s - loss: 0.2018 - categorical_accuracy: 0.9376
53120/60000 [=========================>....] - ETA: 10s - loss: 0.2016 - categorical_accuracy: 0.9377
53184/60000 [=========================>....] - ETA: 10s - loss: 0.2014 - categorical_accuracy: 0.9377
53248/60000 [=========================>....] - ETA: 10s - loss: 0.2012 - categorical_accuracy: 0.9378
53312/60000 [=========================>....] - ETA: 10s - loss: 0.2010 - categorical_accuracy: 0.9379
53344/60000 [=========================>....] - ETA: 10s - loss: 0.2009 - categorical_accuracy: 0.9379
53408/60000 [=========================>....] - ETA: 10s - loss: 0.2008 - categorical_accuracy: 0.9379
53472/60000 [=========================>....] - ETA: 9s - loss: 0.2006 - categorical_accuracy: 0.9380 
53536/60000 [=========================>....] - ETA: 9s - loss: 0.2004 - categorical_accuracy: 0.9380
53600/60000 [=========================>....] - ETA: 9s - loss: 0.2003 - categorical_accuracy: 0.9381
53632/60000 [=========================>....] - ETA: 9s - loss: 0.2002 - categorical_accuracy: 0.9381
53696/60000 [=========================>....] - ETA: 9s - loss: 0.2000 - categorical_accuracy: 0.9381
53760/60000 [=========================>....] - ETA: 9s - loss: 0.1998 - categorical_accuracy: 0.9382
53824/60000 [=========================>....] - ETA: 9s - loss: 0.1996 - categorical_accuracy: 0.9383
53888/60000 [=========================>....] - ETA: 9s - loss: 0.1995 - categorical_accuracy: 0.9383
53952/60000 [=========================>....] - ETA: 9s - loss: 0.1993 - categorical_accuracy: 0.9384
54016/60000 [==========================>...] - ETA: 9s - loss: 0.1991 - categorical_accuracy: 0.9384
54048/60000 [==========================>...] - ETA: 9s - loss: 0.1990 - categorical_accuracy: 0.9384
54112/60000 [==========================>...] - ETA: 9s - loss: 0.1988 - categorical_accuracy: 0.9385
54176/60000 [==========================>...] - ETA: 8s - loss: 0.1986 - categorical_accuracy: 0.9386
54240/60000 [==========================>...] - ETA: 8s - loss: 0.1985 - categorical_accuracy: 0.9386
54272/60000 [==========================>...] - ETA: 8s - loss: 0.1984 - categorical_accuracy: 0.9386
54304/60000 [==========================>...] - ETA: 8s - loss: 0.1984 - categorical_accuracy: 0.9386
54336/60000 [==========================>...] - ETA: 8s - loss: 0.1983 - categorical_accuracy: 0.9387
54400/60000 [==========================>...] - ETA: 8s - loss: 0.1982 - categorical_accuracy: 0.9387
54464/60000 [==========================>...] - ETA: 8s - loss: 0.1983 - categorical_accuracy: 0.9387
54528/60000 [==========================>...] - ETA: 8s - loss: 0.1982 - categorical_accuracy: 0.9387
54592/60000 [==========================>...] - ETA: 8s - loss: 0.1981 - categorical_accuracy: 0.9388
54656/60000 [==========================>...] - ETA: 8s - loss: 0.1981 - categorical_accuracy: 0.9388
54720/60000 [==========================>...] - ETA: 8s - loss: 0.1981 - categorical_accuracy: 0.9388
54784/60000 [==========================>...] - ETA: 7s - loss: 0.1980 - categorical_accuracy: 0.9388
54816/60000 [==========================>...] - ETA: 7s - loss: 0.1979 - categorical_accuracy: 0.9388
54880/60000 [==========================>...] - ETA: 7s - loss: 0.1978 - categorical_accuracy: 0.9389
54944/60000 [==========================>...] - ETA: 7s - loss: 0.1977 - categorical_accuracy: 0.9389
55008/60000 [==========================>...] - ETA: 7s - loss: 0.1976 - categorical_accuracy: 0.9390
55072/60000 [==========================>...] - ETA: 7s - loss: 0.1974 - categorical_accuracy: 0.9391
55136/60000 [==========================>...] - ETA: 7s - loss: 0.1972 - categorical_accuracy: 0.9391
55200/60000 [==========================>...] - ETA: 7s - loss: 0.1970 - categorical_accuracy: 0.9392
55264/60000 [==========================>...] - ETA: 7s - loss: 0.1969 - categorical_accuracy: 0.9392
55328/60000 [==========================>...] - ETA: 7s - loss: 0.1968 - categorical_accuracy: 0.9392
55392/60000 [==========================>...] - ETA: 7s - loss: 0.1967 - categorical_accuracy: 0.9393
55424/60000 [==========================>...] - ETA: 6s - loss: 0.1966 - categorical_accuracy: 0.9393
55488/60000 [==========================>...] - ETA: 6s - loss: 0.1965 - categorical_accuracy: 0.9394
55520/60000 [==========================>...] - ETA: 6s - loss: 0.1964 - categorical_accuracy: 0.9394
55552/60000 [==========================>...] - ETA: 6s - loss: 0.1964 - categorical_accuracy: 0.9394
55616/60000 [==========================>...] - ETA: 6s - loss: 0.1963 - categorical_accuracy: 0.9394
55680/60000 [==========================>...] - ETA: 6s - loss: 0.1963 - categorical_accuracy: 0.9394
55744/60000 [==========================>...] - ETA: 6s - loss: 0.1962 - categorical_accuracy: 0.9394
55808/60000 [==========================>...] - ETA: 6s - loss: 0.1960 - categorical_accuracy: 0.9394
55872/60000 [==========================>...] - ETA: 6s - loss: 0.1959 - categorical_accuracy: 0.9395
55936/60000 [==========================>...] - ETA: 6s - loss: 0.1958 - categorical_accuracy: 0.9395
56000/60000 [===========================>..] - ETA: 6s - loss: 0.1957 - categorical_accuracy: 0.9396
56064/60000 [===========================>..] - ETA: 6s - loss: 0.1955 - categorical_accuracy: 0.9396
56128/60000 [===========================>..] - ETA: 5s - loss: 0.1953 - categorical_accuracy: 0.9397
56192/60000 [===========================>..] - ETA: 5s - loss: 0.1952 - categorical_accuracy: 0.9397
56224/60000 [===========================>..] - ETA: 5s - loss: 0.1951 - categorical_accuracy: 0.9397
56288/60000 [===========================>..] - ETA: 5s - loss: 0.1951 - categorical_accuracy: 0.9398
56352/60000 [===========================>..] - ETA: 5s - loss: 0.1949 - categorical_accuracy: 0.9398
56416/60000 [===========================>..] - ETA: 5s - loss: 0.1947 - categorical_accuracy: 0.9399
56448/60000 [===========================>..] - ETA: 5s - loss: 0.1947 - categorical_accuracy: 0.9399
56512/60000 [===========================>..] - ETA: 5s - loss: 0.1945 - categorical_accuracy: 0.9400
56544/60000 [===========================>..] - ETA: 5s - loss: 0.1944 - categorical_accuracy: 0.9400
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1943 - categorical_accuracy: 0.9400
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1942 - categorical_accuracy: 0.9401
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1940 - categorical_accuracy: 0.9401
56736/60000 [===========================>..] - ETA: 4s - loss: 0.1940 - categorical_accuracy: 0.9401
56800/60000 [===========================>..] - ETA: 4s - loss: 0.1939 - categorical_accuracy: 0.9401
56864/60000 [===========================>..] - ETA: 4s - loss: 0.1937 - categorical_accuracy: 0.9402
56928/60000 [===========================>..] - ETA: 4s - loss: 0.1936 - categorical_accuracy: 0.9402
56960/60000 [===========================>..] - ETA: 4s - loss: 0.1935 - categorical_accuracy: 0.9402
56992/60000 [===========================>..] - ETA: 4s - loss: 0.1935 - categorical_accuracy: 0.9402
57056/60000 [===========================>..] - ETA: 4s - loss: 0.1933 - categorical_accuracy: 0.9403
57120/60000 [===========================>..] - ETA: 4s - loss: 0.1934 - categorical_accuracy: 0.9403
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1932 - categorical_accuracy: 0.9404
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1934 - categorical_accuracy: 0.9404
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1932 - categorical_accuracy: 0.9404
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1931 - categorical_accuracy: 0.9404
57440/60000 [===========================>..] - ETA: 3s - loss: 0.1930 - categorical_accuracy: 0.9405
57472/60000 [===========================>..] - ETA: 3s - loss: 0.1929 - categorical_accuracy: 0.9405
57536/60000 [===========================>..] - ETA: 3s - loss: 0.1927 - categorical_accuracy: 0.9406
57568/60000 [===========================>..] - ETA: 3s - loss: 0.1926 - categorical_accuracy: 0.9406
57600/60000 [===========================>..] - ETA: 3s - loss: 0.1926 - categorical_accuracy: 0.9406
57632/60000 [===========================>..] - ETA: 3s - loss: 0.1925 - categorical_accuracy: 0.9406
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1923 - categorical_accuracy: 0.9407
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1922 - categorical_accuracy: 0.9407
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1920 - categorical_accuracy: 0.9408
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1918 - categorical_accuracy: 0.9409
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1917 - categorical_accuracy: 0.9408
58016/60000 [============================>.] - ETA: 3s - loss: 0.1916 - categorical_accuracy: 0.9409
58080/60000 [============================>.] - ETA: 2s - loss: 0.1915 - categorical_accuracy: 0.9409
58144/60000 [============================>.] - ETA: 2s - loss: 0.1914 - categorical_accuracy: 0.9409
58208/60000 [============================>.] - ETA: 2s - loss: 0.1913 - categorical_accuracy: 0.9410
58240/60000 [============================>.] - ETA: 2s - loss: 0.1913 - categorical_accuracy: 0.9410
58304/60000 [============================>.] - ETA: 2s - loss: 0.1912 - categorical_accuracy: 0.9410
58368/60000 [============================>.] - ETA: 2s - loss: 0.1910 - categorical_accuracy: 0.9411
58432/60000 [============================>.] - ETA: 2s - loss: 0.1909 - categorical_accuracy: 0.9411
58496/60000 [============================>.] - ETA: 2s - loss: 0.1907 - categorical_accuracy: 0.9412
58528/60000 [============================>.] - ETA: 2s - loss: 0.1907 - categorical_accuracy: 0.9412
58592/60000 [============================>.] - ETA: 2s - loss: 0.1905 - categorical_accuracy: 0.9412
58656/60000 [============================>.] - ETA: 2s - loss: 0.1904 - categorical_accuracy: 0.9413
58688/60000 [============================>.] - ETA: 2s - loss: 0.1903 - categorical_accuracy: 0.9413
58720/60000 [============================>.] - ETA: 1s - loss: 0.1903 - categorical_accuracy: 0.9413
58784/60000 [============================>.] - ETA: 1s - loss: 0.1903 - categorical_accuracy: 0.9413
58848/60000 [============================>.] - ETA: 1s - loss: 0.1902 - categorical_accuracy: 0.9413
58880/60000 [============================>.] - ETA: 1s - loss: 0.1903 - categorical_accuracy: 0.9413
58912/60000 [============================>.] - ETA: 1s - loss: 0.1902 - categorical_accuracy: 0.9414
58944/60000 [============================>.] - ETA: 1s - loss: 0.1901 - categorical_accuracy: 0.9414
59008/60000 [============================>.] - ETA: 1s - loss: 0.1899 - categorical_accuracy: 0.9414
59072/60000 [============================>.] - ETA: 1s - loss: 0.1898 - categorical_accuracy: 0.9415
59136/60000 [============================>.] - ETA: 1s - loss: 0.1897 - categorical_accuracy: 0.9415
59200/60000 [============================>.] - ETA: 1s - loss: 0.1896 - categorical_accuracy: 0.9416
59264/60000 [============================>.] - ETA: 1s - loss: 0.1895 - categorical_accuracy: 0.9416
59328/60000 [============================>.] - ETA: 1s - loss: 0.1894 - categorical_accuracy: 0.9416
59392/60000 [============================>.] - ETA: 0s - loss: 0.1893 - categorical_accuracy: 0.9417
59456/60000 [============================>.] - ETA: 0s - loss: 0.1891 - categorical_accuracy: 0.9417
59520/60000 [============================>.] - ETA: 0s - loss: 0.1889 - categorical_accuracy: 0.9418
59584/60000 [============================>.] - ETA: 0s - loss: 0.1888 - categorical_accuracy: 0.9418
59648/60000 [============================>.] - ETA: 0s - loss: 0.1887 - categorical_accuracy: 0.9418
59680/60000 [============================>.] - ETA: 0s - loss: 0.1886 - categorical_accuracy: 0.9419
59744/60000 [============================>.] - ETA: 0s - loss: 0.1886 - categorical_accuracy: 0.9419
59808/60000 [============================>.] - ETA: 0s - loss: 0.1884 - categorical_accuracy: 0.9419
59840/60000 [============================>.] - ETA: 0s - loss: 0.1883 - categorical_accuracy: 0.9419
59872/60000 [============================>.] - ETA: 0s - loss: 0.1883 - categorical_accuracy: 0.9419
59936/60000 [============================>.] - ETA: 0s - loss: 0.1882 - categorical_accuracy: 0.9420
60000/60000 [==============================] - 95s 2ms/step - loss: 0.1880 - categorical_accuracy: 0.9420 - val_loss: 0.0497 - val_categorical_accuracy: 0.9834

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 16s
  224/10000 [..............................] - ETA: 4s 
  416/10000 [>.............................] - ETA: 3s
  608/10000 [>.............................] - ETA: 3s
  800/10000 [=>............................] - ETA: 3s
  992/10000 [=>............................] - ETA: 2s
 1184/10000 [==>...........................] - ETA: 2s
 1376/10000 [===>..........................] - ETA: 2s
 1568/10000 [===>..........................] - ETA: 2s
 1760/10000 [====>.........................] - ETA: 2s
 1952/10000 [====>.........................] - ETA: 2s
 2144/10000 [=====>........................] - ETA: 2s
 2336/10000 [======>.......................] - ETA: 2s
 2528/10000 [======>.......................] - ETA: 2s
 2720/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 2s
 3104/10000 [========>.....................] - ETA: 2s
 3296/10000 [========>.....................] - ETA: 1s
 3488/10000 [=========>....................] - ETA: 1s
 3680/10000 [==========>...................] - ETA: 1s
 3872/10000 [==========>...................] - ETA: 1s
 4064/10000 [===========>..................] - ETA: 1s
 4256/10000 [===========>..................] - ETA: 1s
 4448/10000 [============>.................] - ETA: 1s
 4640/10000 [============>.................] - ETA: 1s
 4832/10000 [=============>................] - ETA: 1s
 5024/10000 [==============>...............] - ETA: 1s
 5216/10000 [==============>...............] - ETA: 1s
 5408/10000 [===============>..............] - ETA: 1s
 5600/10000 [===============>..............] - ETA: 1s
 5792/10000 [================>.............] - ETA: 1s
 5984/10000 [================>.............] - ETA: 1s
 6176/10000 [=================>............] - ETA: 1s
 6368/10000 [==================>...........] - ETA: 1s
 6560/10000 [==================>...........] - ETA: 0s
 6752/10000 [===================>..........] - ETA: 0s
 6944/10000 [===================>..........] - ETA: 0s
 7136/10000 [====================>.........] - ETA: 0s
 7296/10000 [====================>.........] - ETA: 0s
 7488/10000 [=====================>........] - ETA: 0s
 7648/10000 [=====================>........] - ETA: 0s
 7840/10000 [======================>.......] - ETA: 0s
 8000/10000 [=======================>......] - ETA: 0s
 8192/10000 [=======================>......] - ETA: 0s
 8384/10000 [========================>.....] - ETA: 0s
 8576/10000 [========================>.....] - ETA: 0s
 8768/10000 [=========================>....] - ETA: 0s
 8960/10000 [=========================>....] - ETA: 0s
 9152/10000 [==========================>...] - ETA: 0s
 9344/10000 [===========================>..] - ETA: 0s
 9536/10000 [===========================>..] - ETA: 0s
 9728/10000 [============================>.] - ETA: 0s
 9920/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 291us/step
[[5.1040319e-09 1.8384988e-08 1.1486867e-06 ... 9.9999809e-01
  4.3192880e-08 1.3931614e-07]
 [7.2937360e-06 3.4190236e-06 9.9998796e-01 ... 5.7205169e-09
  1.4015028e-07 3.7141615e-10]
 [1.6664828e-06 9.9992371e-01 8.5462234e-06 ... 2.8557128e-05
  1.2301880e-05 1.3379090e-06]
 ...
 [8.2359447e-10 7.4153905e-07 1.0923743e-09 ... 2.7737264e-07
  1.0620396e-06 3.3269469e-06]
 [1.1004070e-07 4.0758771e-08 1.8336540e-09 ... 7.4929680e-08
  5.6724894e-05 1.8596138e-07]
 [2.6217404e-06 9.2660059e-08 7.7462255e-06 ... 2.8177845e-09
  1.1667107e-07 2.4663665e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04969238245636225, 'accuracy_test:': 0.9833999872207642}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Already up to date.
[master 32fee8a] ml_store
 1 file changed, 1251 insertions(+)
To github.com:arita37/mlmodels_store.git
   268c192..32fee8a  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Using TensorFlow backend.
Loading data...
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py", line 357, in <module>
    test(pars_choice="test01")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py", line 320, in test
    Xtuple = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py", line 216, in get_dataset
    if data_pars['type'] == "npz":
KeyError: 'type'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Already up to date.
[master af7c24f] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   32fee8a..af7c24f  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//namentity_crm_bilstm.py 

  #### Loading params   ############################################## 

  #### Loading dataset   ############################################# 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//namentity_crm_bilstm.py", line 348, in <module>
    test(pars_choice="json", data_path=f"model_keras/namentity_crm_bilstm.json")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//namentity_crm_bilstm.py", line 311, in test
    Xtuple = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//namentity_crm_bilstm.py", line 193, in get_dataset
    raise Exception(f"Not support dataset yet")
Exception: Not support dataset yet

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Already up to date.

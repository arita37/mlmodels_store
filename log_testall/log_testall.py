
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '51b64e342c7b2661e79b8abaa33db92672ae95c7', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ************************************************************************************************************************

  ############Check model ################################ 

  ['model_gluon.fb_prophet', 'model_gluon.gluon_automl', 'model_gluon.gluonts_model', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn', 'model_keras.textvae', 'model_keras.charcnn', 'model_keras.charcnn_zhang', 'model_keras.keras_gan', 'model_keras.namentity_crm_bilstm', 'model_keras.nbeats', 'model_keras.armdn', 'model_keras.02_cnn', 'model_keras.Autokeras', 'model_keras.textcnn', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.01_deepctr', 'model_tch.pplm', 'model_tch.mlp', 'model_tch.nbeats', 'model_tch.transformer_classifier', 'model_tch.03_nbeats_dataloader', 'model_tch.transformer_sentence', 'model_tch.matchzoo_models', 'model_tch.textcnn', 'model_tch.pytorch_vae', 'model_tch.torchhub', 'model_tf.temporal_fusion_google', 'model_tf.1_lstm'] 

  Used ['model_gluon.fb_prophet', 'model_gluon.gluon_automl', 'model_gluon.gluonts_model', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn', 'model_keras.textvae', 'model_keras.charcnn', 'model_keras.charcnn_zhang', 'model_keras.keras_gan', 'model_keras.namentity_crm_bilstm', 'model_keras.nbeats', 'model_keras.armdn', 'model_keras.02_cnn', 'model_keras.Autokeras', 'model_keras.textcnn', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.01_deepctr', 'model_tch.pplm', 'model_tch.mlp', 'model_tch.nbeats', 'model_tch.transformer_classifier', 'model_tch.03_nbeats_dataloader', 'model_tch.transformer_sentence', 'model_tch.matchzoo_models', 'model_tch.textcnn', 'model_tch.pytorch_vae', 'model_tch.torchhub', 'model_tf.temporal_fusion_google', 'model_tf.1_lstm'] 





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
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
Already up to date.
[master 139abf8] ml_store
 2 files changed, 64 insertions(+), 11026 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   c6b695d..139abf8  master -> master





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
	Data preprocessing and feature engineering runtime = 0.19s ...
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
 40%|████      | 2/5 [00:17<00:26,  8.80s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.8466177484323181, 'learning_rate': 0.1136163494985154, 'min_data_in_leaf': 9, 'num_leaves': 59} and reward: 0.3898
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\x17~\x1a\xb7[\xd8X\r\x00\x00\x00learning_rateq\x02G?\xbd\x15\xf6\tc\x14`X\x10\x00\x00\x00min_data_in_leafq\x03K\tX\n\x00\x00\x00num_leavesq\x04K;u.' and reward: 0.3898
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\x17~\x1a\xb7[\xd8X\r\x00\x00\x00learning_rateq\x02G?\xbd\x15\xf6\tc\x14`X\x10\x00\x00\x00min_data_in_leafq\x03K\tX\n\x00\x00\x00num_leavesq\x04K;u.' and reward: 0.3898
 60%|██████    | 3/5 [00:42<00:27, 13.75s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9823034781950581, 'learning_rate': 0.1904019523350776, 'min_data_in_leaf': 15, 'num_leaves': 30} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xefo\x07\xb43\x08\xccX\r\x00\x00\x00learning_rateq\x02G?\xc8_\x17W/\xd5\xb3X\x10\x00\x00\x00min_data_in_leafq\x03K\x0fX\n\x00\x00\x00num_leavesq\x04K\x1eu.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xefo\x07\xb43\x08\xccX\r\x00\x00\x00learning_rateq\x02G?\xc8_\x17W/\xd5\xb3X\x10\x00\x00\x00min_data_in_leafq\x03K\x0fX\n\x00\x00\x00num_leavesq\x04K\x1eu.' and reward: 0.3894
 80%|████████  | 4/5 [00:57<00:14, 14.12s/it] 80%|████████  | 4/5 [00:57<00:14, 14.47s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.7577050701117096, 'learning_rate': 0.18016604361494487, 'min_data_in_leaf': 12, 'num_leaves': 50} and reward: 0.3864
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8?\x1e\xb4\x04\x94\xeeX\r\x00\x00\x00learning_rateq\x02G?\xc7\x0f\xaeP\x96\x83\xd2X\x10\x00\x00\x00min_data_in_leafq\x03K\x0cX\n\x00\x00\x00num_leavesq\x04K2u.' and reward: 0.3864
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8?\x1e\xb4\x04\x94\xeeX\r\x00\x00\x00learning_rateq\x02G?\xc7\x0f\xaeP\x96\x83\xd2X\x10\x00\x00\x00min_data_in_leafq\x03K\x0cX\n\x00\x00\x00num_leavesq\x04K2u.' and reward: 0.3864
Time for Gradient Boosting hyperparameter optimization: 79.59723949432373
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 1.0, 'learning_rate': 0.1, 'min_data_in_leaf': 20, 'num_leaves': 36}
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
Saving dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
 40%|████      | 2/5 [00:40<01:01, 20.43s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.4040946607495949, 'embedding_size_factor': 0.8495241795688466, 'layers.choice': 3, 'learning_rate': 0.00013498826002366203, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 5.061592916630433e-06} and reward: 0.3148
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd9\xdc\xaf\xda\x1a\x18*X\x15\x00\x00\x00embedding_size_factorq\x03G?\xeb/MU\r\x1a\x08X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?!\xb1tS\x04\xbe\x97X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xd5:\xd8\x11\x14\xf3\xacu.' and reward: 0.3148
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd9\xdc\xaf\xda\x1a\x18*X\x15\x00\x00\x00embedding_size_factorq\x03G?\xeb/MU\r\x1a\x08X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?!\xb1tS\x04\xbe\x97X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xd5:\xd8\x11\x14\xf3\xacu.' and reward: 0.3148
 60%|██████    | 3/5 [01:22<00:53, 26.77s/it] 60%|██████    | 3/5 [01:22<00:54, 27.48s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1023283444020884, 'embedding_size_factor': 1.0238742802477523, 'layers.choice': 0, 'learning_rate': 0.00046576816902273887, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 1.5341998794042092e-11} and reward: 0.3872
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xba20\xbc\xa9)\xd2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0a\xc9\xffN\x12\xfcX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?>\x86K\r\xb0=\x85X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xb0\xdec\x85U5\xfdu.' and reward: 0.3872
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xba20\xbc\xa9)\xd2X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0a\xc9\xffN\x12\xfcX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?>\x86K\r\xb0=\x85X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xb0\xdec\x85U5\xfdu.' and reward: 0.3872
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 125.42460989952087
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.8s of the -87.67s of remaining time.
Ensemble size: 20
Ensemble weights: 
[0.25 0.05 0.   0.35 0.15 0.   0.2 ]
	0.4002	 = Validation accuracy score
	1.4s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 209.11s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7fb478e3ec18>

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
From github.com:arita37/mlmodels_store
   139abf8..8c18f2b  master     -> origin/master
Updating 139abf8..8c18f2b
Fast-forward
 .../20200522/list_log_pullrequest_20200522.md      |   2 +-
 error_list/20200522/list_log_testall_20200522.md   | 713 +--------------------
 ...-10_51b64e342c7b2661e79b8abaa33db92672ae95c7.py | 621 ++++++++++++++++++
 3 files changed, 623 insertions(+), 713 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-22-08-10_51b64e342c7b2661e79b8abaa33db92672ae95c7.py
[master 05d276e] ml_store
 1 file changed, 215 insertions(+)
To github.com:arita37/mlmodels_store.git
   8c18f2b..05d276e  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  4.27it/s, avg_epoch_loss=5.19]
INFO:root:Epoch[0] Elapsed time 2.346 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.188932
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.188931798934936 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fe329a383c8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fe329a383c8>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['type.txt', 'version.json', 'prediction_net-network.json', 'parameters.json', 'glutonts_model_pars.pkl', 'prediction_net-0000.params', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]WARNING:root:multiple 5 does not divide base seasonality 1.Falling back to seasonality 1
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 105.18it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1123.23681640625,
    "abs_error": 381.79443359375,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.529750759256139,
    "sMAPE": 0.5240403017370577,
    "MSIS": 101.19003037024555,
    "QuantileLoss[0.5]": 381.79443359375,
    "Coverage[0.5]": 1.0,
    "RMSE": 33.51472536671381,
    "NRMSE": 0.7055731656150276,
    "ND": 0.6698147957785088,
    "wQuantileLoss[0.5]": 0.6698147957785088,
    "mean_wQuantileLoss": 0.6698147957785088,
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
100%|██████████| 10/10 [00:01<00:00,  9.04it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.107 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fe3220d99b0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fe3220d99b0>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['type.txt', 'version.json', 'prediction_net-network.json', 'parameters.json', 'glutonts_model_pars.pkl', 'prediction_net-0000.params', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 182.23it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  6.64it/s, avg_epoch_loss=5.21]
INFO:root:Epoch[0] Elapsed time 1.506 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.213087
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.21308741569519 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fe3220d99b0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fe3220d99b0>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['type.txt', 'version.json', 'prediction_net-network.json', 'parameters.json', 'glutonts_model_pars.pkl', 'prediction_net-0000.params', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 166.67it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 262.86362711588544,
    "abs_error": 178.07469177246094,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.1799139722268082,
    "sMAPE": 0.2941017454586304,
    "MSIS": 47.196556462577036,
    "QuantileLoss[0.5]": 178.07469177246094,
    "Coverage[0.5]": 0.75,
    "RMSE": 16.213069638901988,
    "NRMSE": 0.34132778187162083,
    "ND": 0.31241173995168586,
    "wQuantileLoss[0.5]": 0.31241173995168586,
    "mean_wQuantileLoss": 0.31241173995168586,
    "MAE_Coverage": 0.25
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
 30%|███       | 3/10 [00:10<00:25,  3.65s/it, avg_epoch_loss=6.94] 70%|███████   | 7/10 [00:24<00:10,  3.54s/it, avg_epoch_loss=6.91]100%|██████████| 10/10 [00:33<00:00,  3.38s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 33.822 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.873941
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.873940658569336 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fe2fc09af98>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fe2fc09af98>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['type.txt', 'version.json', 'prediction_net-network.json', 'parameters.json', 'glutonts_model_pars.pkl', 'prediction_net-0000.params', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU
INFO:gluonts.model.wavenet._estimator:Using dilation depth 10 and receptive field length 1024

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 138.06it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 54289.473958333336,
    "abs_error": 2749.054931640625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 18.21510003457669,
    "sMAPE": 1.4168317149302723,
    "MSIS": 728.6040919722253,
    "QuantileLoss[0.5]": 2749.055145263672,
    "Coverage[0.5]": 1.0,
    "RMSE": 233.0010170757487,
    "NRMSE": 4.905284570015763,
    "ND": 4.822903388843201,
    "wQuantileLoss[0.5]": 4.822903763620477,
    "mean_wQuantileLoss": 4.822903763620477,
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
100%|██████████| 10/10 [00:00<00:00, 61.38it/s, avg_epoch_loss=5.18]
INFO:root:Epoch[0] Elapsed time 0.163 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.177750
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.177749824523926 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fe2e6287080>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fe2e6287080>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['type.txt', 'version.json', 'prediction_net-network.json', 'parameters.json', 'glutonts_model_pars.pkl', 'prediction_net-0000.params', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 186.37it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 517.294677734375,
    "abs_error": 190.82241821289062,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.264379767963843,
    "sMAPE": 0.3201404580366494,
    "MSIS": 50.575189100890185,
    "QuantileLoss[0.5]": 190.8224105834961,
    "Coverage[0.5]": 0.6666666666666666,
    "RMSE": 22.744113034681632,
    "NRMSE": 0.47882343230908697,
    "ND": 0.3347761723033169,
    "wQuantileLoss[0.5]": 0.3347761589184142,
    "mean_wQuantileLoss": 0.3347761589184142,
    "MAE_Coverage": 0.16666666666666663
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
100%|██████████| 10/10 [00:00<00:00, 10.31it/s, avg_epoch_loss=161]
INFO:root:Epoch[0] Elapsed time 0.970 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=161.108291
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 161.1082910709855 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fe2e633b630>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fe2e633b630>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['type.txt', 'version.json', 'prediction_net-network.json', 'parameters.json', 'glutonts_model_pars.pkl', 'prediction_net-0000.params', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 183.76it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 439.8828247741723,
    "abs_error": 223.62751574072416,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.4817446980843618,
    "sMAPE": 0.42600252840216285,
    "MSIS": 59.26978792337448,
    "QuantileLoss[0.5]": 223.62751574072416,
    "Coverage[0.5]": 0.3333333333333333,
    "RMSE": 20.973383722570194,
    "NRMSE": 0.441544920475162,
    "ND": 0.3923289749837266,
    "wQuantileLoss[0.5]": 0.3923289749837266,
    "mean_wQuantileLoss": 0.3923289749837266,
    "MAE_Coverage": 0.16666666666666669
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
 10%|█         | 1/10 [01:49<16:29, 109.91s/it, avg_epoch_loss=0.582] 20%|██        | 2/10 [04:42<17:10, 128.76s/it, avg_epoch_loss=0.565] 30%|███       | 3/10 [07:48<17:00, 145.81s/it, avg_epoch_loss=0.548] 40%|████      | 4/10 [11:08<16:13, 162.26s/it, avg_epoch_loss=0.531] 50%|█████     | 5/10 [14:40<14:45, 177.11s/it, avg_epoch_loss=0.515] 60%|██████    | 6/10 [18:02<12:18, 184.63s/it, avg_epoch_loss=0.499] 70%|███████   | 7/10 [21:00<09:07, 182.64s/it, avg_epoch_loss=0.484] 80%|████████  | 8/10 [24:41<06:28, 194.18s/it, avg_epoch_loss=0.47]  90%|█████████ | 9/10 [27:56<03:14, 194.42s/it, avg_epoch_loss=0.457]100%|██████████| 10/10 [30:47<00:00, 187.32s/it, avg_epoch_loss=0.447]100%|██████████| 10/10 [30:47<00:00, 184.77s/it, avg_epoch_loss=0.447]
INFO:root:Epoch[0] Elapsed time 1847.727 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.446552
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.4465524971485138 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fe2e637da20>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fe2e637da20>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['type.txt', 'version.json', 'prediction_net-network.json', 'parameters.json', 'glutonts_model_pars.pkl', 'prediction_net-0000.params', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 19.79it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 164.05573527018228,
    "abs_error": 114.22518920898438,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 0.7568503720907435,
    "sMAPE": 0.20283677998139085,
    "MSIS": 30.274016501293268,
    "QuantileLoss[0.5]": 114.22519302368164,
    "Coverage[0.5]": 0.4166666666666667,
    "RMSE": 12.808424386714483,
    "NRMSE": 0.2696510397203049,
    "ND": 0.2003950687876919,
    "wQuantileLoss[0.5]": 0.20039507548014324,
    "mean_wQuantileLoss": 0.20039507548014324,
    "MAE_Coverage": 0.08333333333333331
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
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
Already up to date.
[master fea8668] ml_store
 1 file changed, 501 insertions(+)
To github.com:arita37/mlmodels_store.git
   05d276e..fea8668  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 0.98042741  1.93752881 -0.23083974  0.36633201  1.10018476 -1.04458938
  -0.34498721  2.05117344  0.585662   -2.793085  ]
 [ 0.35413361  0.21112476  0.92145007  0.01652757  0.90394545  0.17718772
   0.09542509 -1.11647002  0.0809271   0.0607502 ]
 [ 1.18468624 -1.00016919 -0.59384307  1.04499441  0.96548233  0.6085147
  -0.625342   -0.0693287  -0.10839207 -0.34390071]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 1.34740825  0.73302323  0.83863475 -1.89881206 -0.54245992 -1.11711069
  -1.09715436 -0.50897228 -0.16648595 -1.03918232]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 1.838294    0.50274088  0.12910158  1.55880554  1.32551412  0.1094027
   1.40754    -1.2197444   2.44936865  1.6169496 ]
 [ 0.44118981  0.47985237 -0.1920037  -1.55269878 -1.88873982  0.57846442
   0.39859839 -0.9612636  -1.45832446 -3.05376438]
 [ 1.66752297  1.22372221 -0.4599301  -0.0593679  -0.493857    1.4489894
  -1.18110317 -0.47758085  0.02599999 -0.79079995]
 [ 0.61363671  0.3166589   1.34710546 -1.89526695 -0.76045809  0.08972912
  -0.32905155  0.41026575  0.85987097 -1.04906775]
 [ 0.55853873 -0.51634791 -0.51814555  0.3511169   0.82550695 -0.06877046
  -0.9520621  -1.34776494  1.47073986 -1.4614036 ]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 0.6109426  -2.79099641 -1.33520272 -0.45611756 -0.94495995 -0.97989025
  -0.15699367  0.69257435 -0.47867236 -0.10646012]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 0.85877496  2.29371761 -1.47023709 -0.83001099 -0.67204982 -1.01951985
   0.59921324 -0.21465384  1.02124813  0.60640394]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 0.79032389  1.61336137 -2.09424782 -0.37480469  0.91588404 -0.74996962
   0.31027229  2.0546241   0.05340954 -0.22876583]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 0.8786438   1.03703898 -0.47712421  0.67261975 -1.04948638  2.42887697
   0.52475049  1.00568668  0.35356722 -0.03599018]
 [ 0.70017571  0.55607351  0.08968641  1.69380911  0.88239331  0.19686978
  -0.56378873  0.16986926 -1.16400797 -0.6011568 ]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 0.9292506  -1.10657307 -1.95816909 -0.3592241  -1.21258781  0.5053819
   0.54264529  1.2179409  -1.94068096  0.67780757]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7fa0d277cd30>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7fa0f37f8f98> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 1.06040861e+00  5.10307597e-01  5.01725109e-01 -9.15791849e-01
  -9.07318361e-01 -4.07252043e-01 -1.79612295e-01  9.84951672e-01
   1.07125243e+00 -5.93343754e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 8.88389445e-01  2.82995534e-01  1.79558917e-02  1.08030817e-01
  -8.49671873e-01  2.94176190e-02 -5.03973949e-01 -1.34793129e-01
   1.04921829e+00 -1.27046078e+00]
 [ 6.23688521e-01  1.20660790e+00  9.03999174e-01 -2.82863552e-01
  -1.18913787e+00 -2.66326884e-01  1.42361443e+00  1.06897162e+00
   4.03714310e-02  1.57546791e+00]
 [ 6.67591795e-01 -4.52524973e-01 -6.05981321e-01  1.16128569e+00
  -1.44620987e+00  1.06996554e+00  1.92381543e+00 -1.04553425e+00
   3.55284507e-01  1.80358898e+00]
 [ 1.83829400e+00  5.02740882e-01  1.29101580e-01  1.55880554e+00
   1.32551412e+00  1.09402696e-01  1.40754000e+00 -1.21974440e+00
   2.44936865e+00  1.61694960e+00]
 [ 1.64661853e+00 -1.52568032e+00 -6.06998398e-01  7.95026094e-01
   1.08480038e+00 -3.74438319e-01  4.29526140e-01  1.34048197e-01
   1.20205486e+00  1.06222724e-01]
 [ 7.90323893e-01  1.61336137e+00 -2.09424782e+00 -3.74804687e-01
   9.15884042e-01 -7.49969617e-01  3.10272288e-01  2.05462410e+00
   5.34095368e-02 -2.28765829e-01]
 [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]
 [ 1.36586461e+00  3.95860270e+00  5.48129585e-01  6.48643644e-01
   8.49176066e-01  1.07343294e-01  1.38631426e+00 -1.39881282e+00
   8.17678188e-02 -1.63744959e+00]
 [ 9.97855163e-01 -6.00138799e-01  4.57947076e-01  1.46765263e-01
  -9.33557290e-01  5.71804879e-01  5.72962726e-01 -3.68176565e-02
   1.12368489e-01 -1.78175491e-02]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 7.73703613e-01  1.27852808e+00 -2.11416392e+00 -4.42229280e-01
   1.06821044e+00  3.23527354e-01 -2.50644065e+00 -1.09991490e-01
   8.54894544e-03 -4.11639163e-01]
 [ 1.13545112e+00  8.61623101e-01  4.90616924e-02 -2.08639057e+00
  -1.11469020e+00  3.61801641e-01 -8.06178212e-01  4.25920177e-01
   4.90803971e-02 -5.96086335e-01]
 [ 8.72267394e-01 -2.51630386e+00 -7.75070287e-01 -5.95667881e-01
   1.02600767e+00 -3.09121319e-01  1.74643509e+00  5.10937774e-01
   1.71066184e+00  1.41640538e-01]
 [ 1.17867274e+00 -5.99804531e-01 -6.94693595e-01  1.12341216e+00
   1.17899425e+00  3.05267040e-01  1.33526763e-02  1.38877940e+00
  -6.61344243e-01  6.21803504e-01]
 [ 6.91743730e-01  1.00978733e+00 -1.21333813e+00 -1.55694156e+00
  -1.20257258e+00 -6.12442128e-01 -2.69836174e+00 -1.39351805e-01
  -7.28537489e-01  7.22518992e-02]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 9.83799588e-01 -4.07240024e-01  9.32721414e-01  1.60564992e-01
  -1.27861800e+00 -1.20149976e-01  1.99759555e-01  3.85602292e-01
   7.18290736e-01 -5.30119800e-01]
 [ 1.09488485e+00 -6.96245395e-02 -1.16444148e-01  3.53870427e-01
  -1.44189096e+00 -1.86955017e-01  1.29118890e+00 -1.53236162e-01
  -2.43250851e+00 -2.27729800e+00]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]]
None

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

  ############ Model preparation   ################################## 

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  ############ Model fit   ########################################## 
fit success None

  ############ Prediction############################################ 
[[ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 9.64572049e-01 -1.06793987e-01  1.12232832e+00  1.45142926e+00
   1.21828168e+00 -6.18036848e-01  4.38166347e-01 -2.03720123e+00
  -1.94258918e+00 -9.97019796e-01]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]
 [ 8.95623122e-01 -2.29820588e+00 -1.95225583e-02  1.45652739e+00
  -1.85064099e+00  3.16637236e-01  1.11337266e-01 -2.66412594e+00
  -4.26428618e-01 -8.39988915e-01]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 8.57719529e-01  9.81122462e-02 -2.60466059e-01  1.06032751e+00
  -1.39003042e+00 -1.71116766e+00  2.65642403e-01  1.65712464e+00
   1.41767401e+00  4.45096710e-01]
 [ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 8.57296491e-01  9.56121704e-01 -8.26097432e-01 -7.05840507e-01
   1.13872896e+00  1.19268607e+00  2.82675712e-01 -2.37941936e-01
   1.15528789e+00  6.21082701e-01]
 [ 8.78740711e-01 -1.92316341e-02  3.19656942e-01  1.50016279e-01
  -1.46662161e+00  4.63534322e-01 -8.98683193e-01  3.97880425e-01
  -9.96010889e-01  3.18154200e-01]
 [ 9.26869810e-01  3.92334911e-01 -4.23478297e-01  4.48380651e-01
  -1.09230828e+00  1.12532350e+00 -9.48439656e-01  1.04053390e-01
   5.28003422e-01  1.00796648e+00]
 [ 9.67037267e-01  3.82715174e-01 -8.06184817e-01 -2.88997343e-01
   9.08526041e-01 -3.91816240e-01  1.62091229e+00  6.84001328e-01
  -3.53409983e-01 -2.51674208e-01]
 [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
  -6.87487393e-01  3.32107876e+00]
 [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
  -9.18440038e-01  1.46089238e+00]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 7.88018455e-01  3.01960045e-01  7.00982122e-01 -3.94689681e-01
  -1.20376927e+00 -1.17181338e+00  7.55392029e-01  9.84012237e-01
  -5.59681422e-01 -1.98937450e-01]
 [ 8.58774962e-01  2.29371761e+00 -1.47023709e+00 -8.30010986e-01
  -6.72049816e-01 -1.01951985e+00  5.99213235e-01 -2.14653842e-01
   1.02124813e+00  6.06403944e-01]
 [ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 6.81889336e-01 -1.15498263e+00  1.22895559e+00 -1.77632196e-01
   9.98545187e-01 -1.51045638e+00 -2.75846063e-01  1.01120706e+00
  -1.47656266e+00  1.30970591e+00]
 [ 9.71395338e-01  7.13049050e-01  1.76041518e+00  1.30620607e+00
   1.05765490e+00 -6.04602969e-01  1.28376990e-01  6.36583409e-01
   1.40925339e+00  9.66539250e-01]
 [ 1.44682180e+00  8.07455917e-01  1.49810818e+00  3.12238689e-01
  -6.82430193e-01 -1.93321640e-01  2.88078167e-01 -2.07680202e+00
   9.47501167e-01 -3.00976154e-01]
 [ 6.91743730e-01  1.00978733e+00 -1.21333813e+00 -1.55694156e+00
  -1.20257258e+00 -6.12442128e-01 -2.69836174e+00 -1.39351805e-01
  -7.28537489e-01  7.22518992e-02]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]]
None

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
[master ea0f8bc] ml_store
 1 file changed, 297 insertions(+)
To github.com:arita37/mlmodels_store.git
   fea8668..ea0f8bc  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_sklearn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 

  #### metrics   ##################################################### 
{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}
{'roc_auc_score': 1.0}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_sklearn/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_sklearn/model.pkl'}
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=4, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)
{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}
{'roc_auc_score': 1.0}

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7ff131b994a8> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
None

  #### Get  metrics   ################################################ 
{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}

  #### Save   ######################################################## 

  #### Load   ######################################################## 

  ############ Model preparation   ################################## 

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  ############ Model fit   ########################################## 
fit success None

  ############ Prediction############################################ 
None

  ############ Save/ Load ############################################ 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)

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
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
Already up to date.
[master 2e3ef35] ml_store
 1 file changed, 110 insertions(+)
To github.com:arita37/mlmodels_store.git
   ea0f8bc..2e3ef35  master -> master





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
[master e2b72de] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   2e3ef35..e2b72de  master -> master





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
[master eca7d0b] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   e2b72de..eca7d0b  master -> master





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

2020-05-22 08:47:43.457423: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-22 08:47:43.469848: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-22 08:47:43.470579: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5587b58713c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 08:47:43.470612: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 7s - loss: 1.3855
256/354 [====================>.........] - ETA: 3s - loss: 1.1353
354/354 [==============================] - 14s 38ms/step - loss: 1.5926 - val_loss: 3.2307

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
[master d19d79d] ml_store
 1 file changed, 150 insertions(+)
To github.com:arita37/mlmodels_store.git
   eca7d0b..d19d79d  master -> master





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
[master a95f81f] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   d19d79d..a95f81f  master -> master





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
[master 089e3d3] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   a95f81f..089e3d3  master -> master





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
[master 76320f1] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   089e3d3..76320f1  master -> master





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

13/13 [==============================] - 1s 97ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 3ms/step - loss: nan
Epoch 8/10

13/13 [==============================] - 0s 3ms/step - loss: nan
Epoch 9/10

13/13 [==============================] - 0s 3ms/step - loss: nan
Epoch 10/10

13/13 [==============================] - 0s 3ms/step - loss: nan

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
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
Already up to date.
[master 4c257af] ml_store
 1 file changed, 127 insertions(+)
To github.com:arita37/mlmodels_store.git
   76320f1..4c257af  master -> master





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
 2621440/11490434 [=====>........................] - ETA: 0s
11173888/11490434 [============================>.] - ETA: 0s
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

   32/60000 [..............................] - ETA: 6:41 - loss: 2.2910 - categorical_accuracy: 0.0625
   64/60000 [..............................] - ETA: 4:13 - loss: 2.2736 - categorical_accuracy: 0.1406
   96/60000 [..............................] - ETA: 3:23 - loss: 2.2582 - categorical_accuracy: 0.1667
  128/60000 [..............................] - ETA: 2:56 - loss: 2.2409 - categorical_accuracy: 0.1953
  192/60000 [..............................] - ETA: 2:30 - loss: 2.1819 - categorical_accuracy: 0.2188
  256/60000 [..............................] - ETA: 2:15 - loss: 2.0872 - categorical_accuracy: 0.2617
  320/60000 [..............................] - ETA: 2:07 - loss: 2.0137 - categorical_accuracy: 0.3000
  352/60000 [..............................] - ETA: 2:04 - loss: 1.9764 - categorical_accuracy: 0.3153
  384/60000 [..............................] - ETA: 2:03 - loss: 1.9413 - categorical_accuracy: 0.3333
  416/60000 [..............................] - ETA: 2:01 - loss: 1.9002 - categorical_accuracy: 0.3558
  448/60000 [..............................] - ETA: 1:59 - loss: 1.8412 - categorical_accuracy: 0.3750
  480/60000 [..............................] - ETA: 1:57 - loss: 1.8366 - categorical_accuracy: 0.3875
  512/60000 [..............................] - ETA: 1:56 - loss: 1.8136 - categorical_accuracy: 0.3926
  544/60000 [..............................] - ETA: 1:55 - loss: 1.7678 - categorical_accuracy: 0.4081
  576/60000 [..............................] - ETA: 1:53 - loss: 1.7332 - categorical_accuracy: 0.4201
  640/60000 [..............................] - ETA: 1:51 - loss: 1.6773 - categorical_accuracy: 0.4313
  672/60000 [..............................] - ETA: 1:50 - loss: 1.6549 - categorical_accuracy: 0.4390
  704/60000 [..............................] - ETA: 1:49 - loss: 1.6283 - categorical_accuracy: 0.4460
  736/60000 [..............................] - ETA: 1:49 - loss: 1.6142 - categorical_accuracy: 0.4538
  800/60000 [..............................] - ETA: 1:48 - loss: 1.5608 - categorical_accuracy: 0.4750
  832/60000 [..............................] - ETA: 1:47 - loss: 1.5463 - categorical_accuracy: 0.4796
  864/60000 [..............................] - ETA: 1:47 - loss: 1.5218 - categorical_accuracy: 0.4907
  896/60000 [..............................] - ETA: 1:47 - loss: 1.4909 - categorical_accuracy: 0.5011
  928/60000 [..............................] - ETA: 1:46 - loss: 1.4724 - categorical_accuracy: 0.5097
  992/60000 [..............................] - ETA: 1:45 - loss: 1.4507 - categorical_accuracy: 0.5151
 1024/60000 [..............................] - ETA: 1:45 - loss: 1.4346 - categorical_accuracy: 0.5215
 1056/60000 [..............................] - ETA: 1:44 - loss: 1.4214 - categorical_accuracy: 0.5237
 1120/60000 [..............................] - ETA: 1:43 - loss: 1.3844 - categorical_accuracy: 0.5375
 1184/60000 [..............................] - ETA: 1:43 - loss: 1.3751 - categorical_accuracy: 0.5448
 1216/60000 [..............................] - ETA: 1:42 - loss: 1.3562 - categorical_accuracy: 0.5518
 1248/60000 [..............................] - ETA: 1:42 - loss: 1.3444 - categorical_accuracy: 0.5545
 1280/60000 [..............................] - ETA: 1:42 - loss: 1.3275 - categorical_accuracy: 0.5586
 1344/60000 [..............................] - ETA: 1:42 - loss: 1.2879 - categorical_accuracy: 0.5737
 1376/60000 [..............................] - ETA: 1:42 - loss: 1.2639 - categorical_accuracy: 0.5828
 1408/60000 [..............................] - ETA: 1:41 - loss: 1.2559 - categorical_accuracy: 0.5838
 1440/60000 [..............................] - ETA: 1:41 - loss: 1.2440 - categorical_accuracy: 0.5882
 1472/60000 [..............................] - ETA: 1:41 - loss: 1.2334 - categorical_accuracy: 0.5931
 1504/60000 [..............................] - ETA: 1:42 - loss: 1.2127 - categorical_accuracy: 0.6011
 1536/60000 [..............................] - ETA: 1:41 - loss: 1.2133 - categorical_accuracy: 0.6035
 1568/60000 [..............................] - ETA: 1:41 - loss: 1.2076 - categorical_accuracy: 0.6046
 1600/60000 [..............................] - ETA: 1:41 - loss: 1.1933 - categorical_accuracy: 0.6100
 1632/60000 [..............................] - ETA: 1:41 - loss: 1.1853 - categorical_accuracy: 0.6134
 1696/60000 [..............................] - ETA: 1:40 - loss: 1.1627 - categorical_accuracy: 0.6221
 1728/60000 [..............................] - ETA: 1:40 - loss: 1.1590 - categorical_accuracy: 0.6238
 1760/60000 [..............................] - ETA: 1:40 - loss: 1.1533 - categorical_accuracy: 0.6256
 1792/60000 [..............................] - ETA: 1:40 - loss: 1.1412 - categorical_accuracy: 0.6300
 1856/60000 [..............................] - ETA: 1:40 - loss: 1.1217 - categorical_accuracy: 0.6363
 1888/60000 [..............................] - ETA: 1:40 - loss: 1.1078 - categorical_accuracy: 0.6414
 1920/60000 [..............................] - ETA: 1:40 - loss: 1.0998 - categorical_accuracy: 0.6443
 1952/60000 [..............................] - ETA: 1:40 - loss: 1.0890 - categorical_accuracy: 0.6491
 2016/60000 [>.............................] - ETA: 1:39 - loss: 1.0767 - categorical_accuracy: 0.6538
 2048/60000 [>.............................] - ETA: 1:39 - loss: 1.0651 - categorical_accuracy: 0.6577
 2080/60000 [>.............................] - ETA: 1:39 - loss: 1.0534 - categorical_accuracy: 0.6611
 2112/60000 [>.............................] - ETA: 1:39 - loss: 1.0484 - categorical_accuracy: 0.6629
 2144/60000 [>.............................] - ETA: 1:39 - loss: 1.0407 - categorical_accuracy: 0.6656
 2176/60000 [>.............................] - ETA: 1:39 - loss: 1.0352 - categorical_accuracy: 0.6668
 2208/60000 [>.............................] - ETA: 1:39 - loss: 1.0317 - categorical_accuracy: 0.6680
 2240/60000 [>.............................] - ETA: 1:39 - loss: 1.0252 - categorical_accuracy: 0.6705
 2304/60000 [>.............................] - ETA: 1:38 - loss: 1.0119 - categorical_accuracy: 0.6766
 2368/60000 [>.............................] - ETA: 1:38 - loss: 0.9992 - categorical_accuracy: 0.6803
 2432/60000 [>.............................] - ETA: 1:38 - loss: 0.9944 - categorical_accuracy: 0.6817
 2464/60000 [>.............................] - ETA: 1:38 - loss: 0.9909 - categorical_accuracy: 0.6830
 2496/60000 [>.............................] - ETA: 1:37 - loss: 0.9839 - categorical_accuracy: 0.6847
 2528/60000 [>.............................] - ETA: 1:37 - loss: 0.9771 - categorical_accuracy: 0.6871
 2560/60000 [>.............................] - ETA: 1:38 - loss: 0.9678 - categorical_accuracy: 0.6902
 2624/60000 [>.............................] - ETA: 1:37 - loss: 0.9603 - categorical_accuracy: 0.6925
 2688/60000 [>.............................] - ETA: 1:37 - loss: 0.9504 - categorical_accuracy: 0.6964
 2752/60000 [>.............................] - ETA: 1:37 - loss: 0.9364 - categorical_accuracy: 0.7006
 2784/60000 [>.............................] - ETA: 1:37 - loss: 0.9303 - categorical_accuracy: 0.7019
 2816/60000 [>.............................] - ETA: 1:36 - loss: 0.9247 - categorical_accuracy: 0.7042
 2880/60000 [>.............................] - ETA: 1:36 - loss: 0.9116 - categorical_accuracy: 0.7090
 2912/60000 [>.............................] - ETA: 1:36 - loss: 0.9045 - categorical_accuracy: 0.7105
 2944/60000 [>.............................] - ETA: 1:36 - loss: 0.9002 - categorical_accuracy: 0.7116
 3008/60000 [>.............................] - ETA: 1:36 - loss: 0.8925 - categorical_accuracy: 0.7138
 3040/60000 [>.............................] - ETA: 1:36 - loss: 0.8864 - categorical_accuracy: 0.7158
 3072/60000 [>.............................] - ETA: 1:37 - loss: 0.8793 - categorical_accuracy: 0.7178
 3104/60000 [>.............................] - ETA: 1:37 - loss: 0.8720 - categorical_accuracy: 0.7200
 3136/60000 [>.............................] - ETA: 1:37 - loss: 0.8666 - categorical_accuracy: 0.7223
 3168/60000 [>.............................] - ETA: 1:36 - loss: 0.8615 - categorical_accuracy: 0.7235
 3232/60000 [>.............................] - ETA: 1:36 - loss: 0.8529 - categorical_accuracy: 0.7268
 3264/60000 [>.............................] - ETA: 1:36 - loss: 0.8497 - categorical_accuracy: 0.7279
 3296/60000 [>.............................] - ETA: 1:36 - loss: 0.8478 - categorical_accuracy: 0.7285
 3328/60000 [>.............................] - ETA: 1:36 - loss: 0.8422 - categorical_accuracy: 0.7302
 3360/60000 [>.............................] - ETA: 1:36 - loss: 0.8365 - categorical_accuracy: 0.7321
 3392/60000 [>.............................] - ETA: 1:36 - loss: 0.8339 - categorical_accuracy: 0.7332
 3456/60000 [>.............................] - ETA: 1:36 - loss: 0.8267 - categorical_accuracy: 0.7358
 3520/60000 [>.............................] - ETA: 1:35 - loss: 0.8214 - categorical_accuracy: 0.7372
 3552/60000 [>.............................] - ETA: 1:35 - loss: 0.8197 - categorical_accuracy: 0.7385
 3616/60000 [>.............................] - ETA: 1:35 - loss: 0.8113 - categorical_accuracy: 0.7412
 3648/60000 [>.............................] - ETA: 1:35 - loss: 0.8068 - categorical_accuracy: 0.7423
 3680/60000 [>.............................] - ETA: 1:35 - loss: 0.8042 - categorical_accuracy: 0.7437
 3744/60000 [>.............................] - ETA: 1:35 - loss: 0.7944 - categorical_accuracy: 0.7471
 3776/60000 [>.............................] - ETA: 1:35 - loss: 0.7889 - categorical_accuracy: 0.7487
 3840/60000 [>.............................] - ETA: 1:34 - loss: 0.7814 - categorical_accuracy: 0.7513
 3872/60000 [>.............................] - ETA: 1:34 - loss: 0.7797 - categorical_accuracy: 0.7526
 3904/60000 [>.............................] - ETA: 1:34 - loss: 0.7782 - categorical_accuracy: 0.7528
 3936/60000 [>.............................] - ETA: 1:34 - loss: 0.7744 - categorical_accuracy: 0.7543
 3968/60000 [>.............................] - ETA: 1:34 - loss: 0.7702 - categorical_accuracy: 0.7555
 4000/60000 [=>............................] - ETA: 1:34 - loss: 0.7662 - categorical_accuracy: 0.7565
 4064/60000 [=>............................] - ETA: 1:34 - loss: 0.7592 - categorical_accuracy: 0.7589
 4096/60000 [=>............................] - ETA: 1:34 - loss: 0.7565 - categorical_accuracy: 0.7598
 4160/60000 [=>............................] - ETA: 1:33 - loss: 0.7484 - categorical_accuracy: 0.7620
 4192/60000 [=>............................] - ETA: 1:33 - loss: 0.7439 - categorical_accuracy: 0.7636
 4224/60000 [=>............................] - ETA: 1:33 - loss: 0.7399 - categorical_accuracy: 0.7649
 4288/60000 [=>............................] - ETA: 1:33 - loss: 0.7345 - categorical_accuracy: 0.7670
 4352/60000 [=>............................] - ETA: 1:33 - loss: 0.7285 - categorical_accuracy: 0.7693
 4384/60000 [=>............................] - ETA: 1:33 - loss: 0.7267 - categorical_accuracy: 0.7696
 4416/60000 [=>............................] - ETA: 1:33 - loss: 0.7249 - categorical_accuracy: 0.7699
 4480/60000 [=>............................] - ETA: 1:32 - loss: 0.7198 - categorical_accuracy: 0.7717
 4512/60000 [=>............................] - ETA: 1:32 - loss: 0.7181 - categorical_accuracy: 0.7724
 4544/60000 [=>............................] - ETA: 1:32 - loss: 0.7147 - categorical_accuracy: 0.7738
 4608/60000 [=>............................] - ETA: 1:32 - loss: 0.7106 - categorical_accuracy: 0.7750
 4672/60000 [=>............................] - ETA: 1:32 - loss: 0.7068 - categorical_accuracy: 0.7765
 4736/60000 [=>............................] - ETA: 1:32 - loss: 0.7015 - categorical_accuracy: 0.7781
 4768/60000 [=>............................] - ETA: 1:32 - loss: 0.6977 - categorical_accuracy: 0.7794
 4800/60000 [=>............................] - ETA: 1:31 - loss: 0.6950 - categorical_accuracy: 0.7798
 4864/60000 [=>............................] - ETA: 1:31 - loss: 0.6937 - categorical_accuracy: 0.7804
 4896/60000 [=>............................] - ETA: 1:31 - loss: 0.6906 - categorical_accuracy: 0.7812
 4928/60000 [=>............................] - ETA: 1:31 - loss: 0.6895 - categorical_accuracy: 0.7812
 4960/60000 [=>............................] - ETA: 1:31 - loss: 0.6873 - categorical_accuracy: 0.7821
 4992/60000 [=>............................] - ETA: 1:31 - loss: 0.6846 - categorical_accuracy: 0.7831
 5024/60000 [=>............................] - ETA: 1:31 - loss: 0.6822 - categorical_accuracy: 0.7832
 5056/60000 [=>............................] - ETA: 1:31 - loss: 0.6799 - categorical_accuracy: 0.7838
 5120/60000 [=>............................] - ETA: 1:31 - loss: 0.6753 - categorical_accuracy: 0.7854
 5152/60000 [=>............................] - ETA: 1:31 - loss: 0.6728 - categorical_accuracy: 0.7861
 5184/60000 [=>............................] - ETA: 1:31 - loss: 0.6693 - categorical_accuracy: 0.7874
 5248/60000 [=>............................] - ETA: 1:31 - loss: 0.6641 - categorical_accuracy: 0.7891
 5280/60000 [=>............................] - ETA: 1:31 - loss: 0.6612 - categorical_accuracy: 0.7900
 5344/60000 [=>............................] - ETA: 1:31 - loss: 0.6583 - categorical_accuracy: 0.7908
 5408/60000 [=>............................] - ETA: 1:30 - loss: 0.6532 - categorical_accuracy: 0.7922
 5472/60000 [=>............................] - ETA: 1:30 - loss: 0.6475 - categorical_accuracy: 0.7942
 5536/60000 [=>............................] - ETA: 1:30 - loss: 0.6431 - categorical_accuracy: 0.7953
 5600/60000 [=>............................] - ETA: 1:30 - loss: 0.6369 - categorical_accuracy: 0.7973
 5632/60000 [=>............................] - ETA: 1:30 - loss: 0.6355 - categorical_accuracy: 0.7983
 5664/60000 [=>............................] - ETA: 1:30 - loss: 0.6342 - categorical_accuracy: 0.7987
 5696/60000 [=>............................] - ETA: 1:30 - loss: 0.6323 - categorical_accuracy: 0.7993
 5760/60000 [=>............................] - ETA: 1:29 - loss: 0.6298 - categorical_accuracy: 0.8003
 5792/60000 [=>............................] - ETA: 1:29 - loss: 0.6273 - categorical_accuracy: 0.8013
 5856/60000 [=>............................] - ETA: 1:29 - loss: 0.6241 - categorical_accuracy: 0.8021
 5888/60000 [=>............................] - ETA: 1:29 - loss: 0.6227 - categorical_accuracy: 0.8023
 5952/60000 [=>............................] - ETA: 1:29 - loss: 0.6197 - categorical_accuracy: 0.8033
 5984/60000 [=>............................] - ETA: 1:29 - loss: 0.6187 - categorical_accuracy: 0.8033
 6016/60000 [==>...........................] - ETA: 1:29 - loss: 0.6170 - categorical_accuracy: 0.8037
 6080/60000 [==>...........................] - ETA: 1:29 - loss: 0.6145 - categorical_accuracy: 0.8043
 6144/60000 [==>...........................] - ETA: 1:29 - loss: 0.6117 - categorical_accuracy: 0.8053
 6208/60000 [==>...........................] - ETA: 1:28 - loss: 0.6066 - categorical_accuracy: 0.8070
 6240/60000 [==>...........................] - ETA: 1:28 - loss: 0.6040 - categorical_accuracy: 0.8079
 6272/60000 [==>...........................] - ETA: 1:28 - loss: 0.6026 - categorical_accuracy: 0.8082
 6336/60000 [==>...........................] - ETA: 1:28 - loss: 0.5994 - categorical_accuracy: 0.8090
 6368/60000 [==>...........................] - ETA: 1:28 - loss: 0.5972 - categorical_accuracy: 0.8097
 6400/60000 [==>...........................] - ETA: 1:28 - loss: 0.5949 - categorical_accuracy: 0.8105
 6464/60000 [==>...........................] - ETA: 1:28 - loss: 0.5919 - categorical_accuracy: 0.8114
 6528/60000 [==>...........................] - ETA: 1:28 - loss: 0.5886 - categorical_accuracy: 0.8120
 6560/60000 [==>...........................] - ETA: 1:28 - loss: 0.5873 - categorical_accuracy: 0.8125
 6624/60000 [==>...........................] - ETA: 1:28 - loss: 0.5836 - categorical_accuracy: 0.8137
 6656/60000 [==>...........................] - ETA: 1:27 - loss: 0.5818 - categorical_accuracy: 0.8142
 6688/60000 [==>...........................] - ETA: 1:27 - loss: 0.5796 - categorical_accuracy: 0.8149
 6720/60000 [==>...........................] - ETA: 1:27 - loss: 0.5785 - categorical_accuracy: 0.8150
 6784/60000 [==>...........................] - ETA: 1:27 - loss: 0.5756 - categorical_accuracy: 0.8159
 6816/60000 [==>...........................] - ETA: 1:27 - loss: 0.5745 - categorical_accuracy: 0.8162
 6880/60000 [==>...........................] - ETA: 1:27 - loss: 0.5711 - categorical_accuracy: 0.8173
 6912/60000 [==>...........................] - ETA: 1:27 - loss: 0.5699 - categorical_accuracy: 0.8176
 6944/60000 [==>...........................] - ETA: 1:27 - loss: 0.5686 - categorical_accuracy: 0.8180
 6976/60000 [==>...........................] - ETA: 1:27 - loss: 0.5689 - categorical_accuracy: 0.8179
 7008/60000 [==>...........................] - ETA: 1:27 - loss: 0.5666 - categorical_accuracy: 0.8188
 7040/60000 [==>...........................] - ETA: 1:27 - loss: 0.5651 - categorical_accuracy: 0.8192
 7072/60000 [==>...........................] - ETA: 1:27 - loss: 0.5629 - categorical_accuracy: 0.8200
 7104/60000 [==>...........................] - ETA: 1:27 - loss: 0.5617 - categorical_accuracy: 0.8204
 7136/60000 [==>...........................] - ETA: 1:27 - loss: 0.5599 - categorical_accuracy: 0.8208
 7168/60000 [==>...........................] - ETA: 1:26 - loss: 0.5581 - categorical_accuracy: 0.8214
 7232/60000 [==>...........................] - ETA: 1:26 - loss: 0.5564 - categorical_accuracy: 0.8223
 7264/60000 [==>...........................] - ETA: 1:26 - loss: 0.5549 - categorical_accuracy: 0.8228
 7328/60000 [==>...........................] - ETA: 1:26 - loss: 0.5514 - categorical_accuracy: 0.8240
 7392/60000 [==>...........................] - ETA: 1:26 - loss: 0.5475 - categorical_accuracy: 0.8254
 7456/60000 [==>...........................] - ETA: 1:26 - loss: 0.5450 - categorical_accuracy: 0.8262
 7520/60000 [==>...........................] - ETA: 1:26 - loss: 0.5418 - categorical_accuracy: 0.8274
 7584/60000 [==>...........................] - ETA: 1:26 - loss: 0.5395 - categorical_accuracy: 0.8282
 7616/60000 [==>...........................] - ETA: 1:25 - loss: 0.5381 - categorical_accuracy: 0.8288
 7648/60000 [==>...........................] - ETA: 1:25 - loss: 0.5365 - categorical_accuracy: 0.8295
 7680/60000 [==>...........................] - ETA: 1:25 - loss: 0.5356 - categorical_accuracy: 0.8297
 7712/60000 [==>...........................] - ETA: 1:25 - loss: 0.5339 - categorical_accuracy: 0.8301
 7744/60000 [==>...........................] - ETA: 1:25 - loss: 0.5323 - categorical_accuracy: 0.8307
 7776/60000 [==>...........................] - ETA: 1:25 - loss: 0.5310 - categorical_accuracy: 0.8311
 7808/60000 [==>...........................] - ETA: 1:25 - loss: 0.5294 - categorical_accuracy: 0.8317
 7872/60000 [==>...........................] - ETA: 1:25 - loss: 0.5256 - categorical_accuracy: 0.8330
 7904/60000 [==>...........................] - ETA: 1:25 - loss: 0.5237 - categorical_accuracy: 0.8336
 7936/60000 [==>...........................] - ETA: 1:25 - loss: 0.5219 - categorical_accuracy: 0.8342
 8000/60000 [===>..........................] - ETA: 1:25 - loss: 0.5218 - categorical_accuracy: 0.8346
 8032/60000 [===>..........................] - ETA: 1:25 - loss: 0.5202 - categorical_accuracy: 0.8352
 8064/60000 [===>..........................] - ETA: 1:25 - loss: 0.5199 - categorical_accuracy: 0.8353
 8096/60000 [===>..........................] - ETA: 1:25 - loss: 0.5191 - categorical_accuracy: 0.8355
 8160/60000 [===>..........................] - ETA: 1:24 - loss: 0.5156 - categorical_accuracy: 0.8365
 8224/60000 [===>..........................] - ETA: 1:24 - loss: 0.5126 - categorical_accuracy: 0.8374
 8256/60000 [===>..........................] - ETA: 1:24 - loss: 0.5113 - categorical_accuracy: 0.8377
 8288/60000 [===>..........................] - ETA: 1:24 - loss: 0.5113 - categorical_accuracy: 0.8378
 8352/60000 [===>..........................] - ETA: 1:24 - loss: 0.5084 - categorical_accuracy: 0.8386
 8416/60000 [===>..........................] - ETA: 1:24 - loss: 0.5051 - categorical_accuracy: 0.8395
 8480/60000 [===>..........................] - ETA: 1:24 - loss: 0.5027 - categorical_accuracy: 0.8403
 8544/60000 [===>..........................] - ETA: 1:24 - loss: 0.5017 - categorical_accuracy: 0.8409
 8576/60000 [===>..........................] - ETA: 1:24 - loss: 0.5002 - categorical_accuracy: 0.8414
 8640/60000 [===>..........................] - ETA: 1:23 - loss: 0.4991 - categorical_accuracy: 0.8422
 8672/60000 [===>..........................] - ETA: 1:23 - loss: 0.4987 - categorical_accuracy: 0.8421
 8736/60000 [===>..........................] - ETA: 1:23 - loss: 0.4969 - categorical_accuracy: 0.8428
 8768/60000 [===>..........................] - ETA: 1:23 - loss: 0.4958 - categorical_accuracy: 0.8431
 8800/60000 [===>..........................] - ETA: 1:23 - loss: 0.4947 - categorical_accuracy: 0.8434
 8864/60000 [===>..........................] - ETA: 1:23 - loss: 0.4929 - categorical_accuracy: 0.8441
 8896/60000 [===>..........................] - ETA: 1:23 - loss: 0.4921 - categorical_accuracy: 0.8445
 8960/60000 [===>..........................] - ETA: 1:23 - loss: 0.4907 - categorical_accuracy: 0.8452
 9024/60000 [===>..........................] - ETA: 1:23 - loss: 0.4884 - categorical_accuracy: 0.8459
 9056/60000 [===>..........................] - ETA: 1:23 - loss: 0.4875 - categorical_accuracy: 0.8461
 9088/60000 [===>..........................] - ETA: 1:23 - loss: 0.4866 - categorical_accuracy: 0.8464
 9120/60000 [===>..........................] - ETA: 1:23 - loss: 0.4858 - categorical_accuracy: 0.8466
 9184/60000 [===>..........................] - ETA: 1:23 - loss: 0.4855 - categorical_accuracy: 0.8472
 9216/60000 [===>..........................] - ETA: 1:23 - loss: 0.4849 - categorical_accuracy: 0.8474
 9280/60000 [===>..........................] - ETA: 1:22 - loss: 0.4834 - categorical_accuracy: 0.8480
 9344/60000 [===>..........................] - ETA: 1:22 - loss: 0.4809 - categorical_accuracy: 0.8489
 9408/60000 [===>..........................] - ETA: 1:22 - loss: 0.4803 - categorical_accuracy: 0.8494
 9440/60000 [===>..........................] - ETA: 1:22 - loss: 0.4804 - categorical_accuracy: 0.8497
 9472/60000 [===>..........................] - ETA: 1:22 - loss: 0.4792 - categorical_accuracy: 0.8501
 9504/60000 [===>..........................] - ETA: 1:22 - loss: 0.4780 - categorical_accuracy: 0.8504
 9568/60000 [===>..........................] - ETA: 1:22 - loss: 0.4763 - categorical_accuracy: 0.8510
 9632/60000 [===>..........................] - ETA: 1:22 - loss: 0.4735 - categorical_accuracy: 0.8520
 9664/60000 [===>..........................] - ETA: 1:22 - loss: 0.4724 - categorical_accuracy: 0.8522
 9728/60000 [===>..........................] - ETA: 1:22 - loss: 0.4708 - categorical_accuracy: 0.8528
 9760/60000 [===>..........................] - ETA: 1:22 - loss: 0.4697 - categorical_accuracy: 0.8532
 9792/60000 [===>..........................] - ETA: 1:21 - loss: 0.4686 - categorical_accuracy: 0.8536
 9824/60000 [===>..........................] - ETA: 1:21 - loss: 0.4678 - categorical_accuracy: 0.8538
 9888/60000 [===>..........................] - ETA: 1:21 - loss: 0.4661 - categorical_accuracy: 0.8544
 9952/60000 [===>..........................] - ETA: 1:21 - loss: 0.4649 - categorical_accuracy: 0.8546
10016/60000 [====>.........................] - ETA: 1:21 - loss: 0.4631 - categorical_accuracy: 0.8553
10080/60000 [====>.........................] - ETA: 1:21 - loss: 0.4617 - categorical_accuracy: 0.8559
10112/60000 [====>.........................] - ETA: 1:21 - loss: 0.4607 - categorical_accuracy: 0.8562
10144/60000 [====>.........................] - ETA: 1:21 - loss: 0.4600 - categorical_accuracy: 0.8564
10176/60000 [====>.........................] - ETA: 1:21 - loss: 0.4594 - categorical_accuracy: 0.8565
10208/60000 [====>.........................] - ETA: 1:21 - loss: 0.4583 - categorical_accuracy: 0.8569
10272/60000 [====>.........................] - ETA: 1:21 - loss: 0.4567 - categorical_accuracy: 0.8576
10336/60000 [====>.........................] - ETA: 1:20 - loss: 0.4549 - categorical_accuracy: 0.8583
10368/60000 [====>.........................] - ETA: 1:20 - loss: 0.4539 - categorical_accuracy: 0.8586
10432/60000 [====>.........................] - ETA: 1:20 - loss: 0.4520 - categorical_accuracy: 0.8591
10464/60000 [====>.........................] - ETA: 1:20 - loss: 0.4517 - categorical_accuracy: 0.8591
10528/60000 [====>.........................] - ETA: 1:20 - loss: 0.4496 - categorical_accuracy: 0.8597
10592/60000 [====>.........................] - ETA: 1:20 - loss: 0.4473 - categorical_accuracy: 0.8605
10656/60000 [====>.........................] - ETA: 1:20 - loss: 0.4458 - categorical_accuracy: 0.8610
10720/60000 [====>.........................] - ETA: 1:20 - loss: 0.4451 - categorical_accuracy: 0.8612
10752/60000 [====>.........................] - ETA: 1:20 - loss: 0.4460 - categorical_accuracy: 0.8610
10784/60000 [====>.........................] - ETA: 1:20 - loss: 0.4456 - categorical_accuracy: 0.8611
10816/60000 [====>.........................] - ETA: 1:20 - loss: 0.4447 - categorical_accuracy: 0.8614
10848/60000 [====>.........................] - ETA: 1:19 - loss: 0.4442 - categorical_accuracy: 0.8616
10912/60000 [====>.........................] - ETA: 1:19 - loss: 0.4426 - categorical_accuracy: 0.8619
10944/60000 [====>.........................] - ETA: 1:19 - loss: 0.4419 - categorical_accuracy: 0.8620
10976/60000 [====>.........................] - ETA: 1:19 - loss: 0.4413 - categorical_accuracy: 0.8622
11008/60000 [====>.........................] - ETA: 1:19 - loss: 0.4407 - categorical_accuracy: 0.8623
11040/60000 [====>.........................] - ETA: 1:19 - loss: 0.4399 - categorical_accuracy: 0.8626
11104/60000 [====>.........................] - ETA: 1:19 - loss: 0.4379 - categorical_accuracy: 0.8632
11136/60000 [====>.........................] - ETA: 1:19 - loss: 0.4376 - categorical_accuracy: 0.8633
11168/60000 [====>.........................] - ETA: 1:19 - loss: 0.4369 - categorical_accuracy: 0.8635
11200/60000 [====>.........................] - ETA: 1:19 - loss: 0.4364 - categorical_accuracy: 0.8637
11264/60000 [====>.........................] - ETA: 1:19 - loss: 0.4347 - categorical_accuracy: 0.8642
11328/60000 [====>.........................] - ETA: 1:19 - loss: 0.4334 - categorical_accuracy: 0.8647
11360/60000 [====>.........................] - ETA: 1:19 - loss: 0.4329 - categorical_accuracy: 0.8649
11392/60000 [====>.........................] - ETA: 1:19 - loss: 0.4322 - categorical_accuracy: 0.8651
11424/60000 [====>.........................] - ETA: 1:19 - loss: 0.4320 - categorical_accuracy: 0.8651
11488/60000 [====>.........................] - ETA: 1:18 - loss: 0.4305 - categorical_accuracy: 0.8655
11552/60000 [====>.........................] - ETA: 1:18 - loss: 0.4289 - categorical_accuracy: 0.8660
11584/60000 [====>.........................] - ETA: 1:18 - loss: 0.4283 - categorical_accuracy: 0.8662
11648/60000 [====>.........................] - ETA: 1:18 - loss: 0.4267 - categorical_accuracy: 0.8667
11680/60000 [====>.........................] - ETA: 1:18 - loss: 0.4258 - categorical_accuracy: 0.8670
11712/60000 [====>.........................] - ETA: 1:18 - loss: 0.4257 - categorical_accuracy: 0.8670
11744/60000 [====>.........................] - ETA: 1:18 - loss: 0.4251 - categorical_accuracy: 0.8673
11808/60000 [====>.........................] - ETA: 1:18 - loss: 0.4246 - categorical_accuracy: 0.8674
11872/60000 [====>.........................] - ETA: 1:18 - loss: 0.4234 - categorical_accuracy: 0.8679
11936/60000 [====>.........................] - ETA: 1:18 - loss: 0.4218 - categorical_accuracy: 0.8685
11968/60000 [====>.........................] - ETA: 1:18 - loss: 0.4209 - categorical_accuracy: 0.8687
12032/60000 [=====>........................] - ETA: 1:17 - loss: 0.4200 - categorical_accuracy: 0.8690
12064/60000 [=====>........................] - ETA: 1:17 - loss: 0.4190 - categorical_accuracy: 0.8694
12096/60000 [=====>........................] - ETA: 1:17 - loss: 0.4181 - categorical_accuracy: 0.8696
12128/60000 [=====>........................] - ETA: 1:17 - loss: 0.4178 - categorical_accuracy: 0.8696
12160/60000 [=====>........................] - ETA: 1:17 - loss: 0.4170 - categorical_accuracy: 0.8699
12192/60000 [=====>........................] - ETA: 1:17 - loss: 0.4167 - categorical_accuracy: 0.8701
12224/60000 [=====>........................] - ETA: 1:17 - loss: 0.4157 - categorical_accuracy: 0.8704
12288/60000 [=====>........................] - ETA: 1:17 - loss: 0.4148 - categorical_accuracy: 0.8707
12320/60000 [=====>........................] - ETA: 1:17 - loss: 0.4143 - categorical_accuracy: 0.8707
12352/60000 [=====>........................] - ETA: 1:17 - loss: 0.4143 - categorical_accuracy: 0.8708
12384/60000 [=====>........................] - ETA: 1:17 - loss: 0.4144 - categorical_accuracy: 0.8710
12416/60000 [=====>........................] - ETA: 1:17 - loss: 0.4140 - categorical_accuracy: 0.8711
12448/60000 [=====>........................] - ETA: 1:17 - loss: 0.4134 - categorical_accuracy: 0.8715
12512/60000 [=====>........................] - ETA: 1:17 - loss: 0.4122 - categorical_accuracy: 0.8720
12576/60000 [=====>........................] - ETA: 1:16 - loss: 0.4106 - categorical_accuracy: 0.8723
12608/60000 [=====>........................] - ETA: 1:16 - loss: 0.4103 - categorical_accuracy: 0.8725
12672/60000 [=====>........................] - ETA: 1:16 - loss: 0.4088 - categorical_accuracy: 0.8730
12704/60000 [=====>........................] - ETA: 1:16 - loss: 0.4092 - categorical_accuracy: 0.8730
12736/60000 [=====>........................] - ETA: 1:16 - loss: 0.4084 - categorical_accuracy: 0.8733
12768/60000 [=====>........................] - ETA: 1:16 - loss: 0.4079 - categorical_accuracy: 0.8735
12832/60000 [=====>........................] - ETA: 1:16 - loss: 0.4067 - categorical_accuracy: 0.8738
12864/60000 [=====>........................] - ETA: 1:16 - loss: 0.4061 - categorical_accuracy: 0.8741
12896/60000 [=====>........................] - ETA: 1:16 - loss: 0.4059 - categorical_accuracy: 0.8742
12928/60000 [=====>........................] - ETA: 1:16 - loss: 0.4052 - categorical_accuracy: 0.8745
12992/60000 [=====>........................] - ETA: 1:16 - loss: 0.4038 - categorical_accuracy: 0.8749
13024/60000 [=====>........................] - ETA: 1:16 - loss: 0.4032 - categorical_accuracy: 0.8752
13056/60000 [=====>........................] - ETA: 1:16 - loss: 0.4034 - categorical_accuracy: 0.8751
13088/60000 [=====>........................] - ETA: 1:16 - loss: 0.4027 - categorical_accuracy: 0.8754
13152/60000 [=====>........................] - ETA: 1:16 - loss: 0.4013 - categorical_accuracy: 0.8759
13216/60000 [=====>........................] - ETA: 1:15 - loss: 0.4004 - categorical_accuracy: 0.8762
13248/60000 [=====>........................] - ETA: 1:15 - loss: 0.3999 - categorical_accuracy: 0.8764
13312/60000 [=====>........................] - ETA: 1:15 - loss: 0.3984 - categorical_accuracy: 0.8769
13344/60000 [=====>........................] - ETA: 1:15 - loss: 0.3978 - categorical_accuracy: 0.8771
13408/60000 [=====>........................] - ETA: 1:15 - loss: 0.3975 - categorical_accuracy: 0.8773
13472/60000 [=====>........................] - ETA: 1:15 - loss: 0.3964 - categorical_accuracy: 0.8774
13536/60000 [=====>........................] - ETA: 1:15 - loss: 0.3956 - categorical_accuracy: 0.8776
13600/60000 [=====>........................] - ETA: 1:15 - loss: 0.3953 - categorical_accuracy: 0.8777
13664/60000 [=====>........................] - ETA: 1:15 - loss: 0.3943 - categorical_accuracy: 0.8781
13696/60000 [=====>........................] - ETA: 1:15 - loss: 0.3936 - categorical_accuracy: 0.8783
13760/60000 [=====>........................] - ETA: 1:14 - loss: 0.3924 - categorical_accuracy: 0.8786
13792/60000 [=====>........................] - ETA: 1:14 - loss: 0.3920 - categorical_accuracy: 0.8788
13824/60000 [=====>........................] - ETA: 1:14 - loss: 0.3912 - categorical_accuracy: 0.8791
13888/60000 [=====>........................] - ETA: 1:14 - loss: 0.3906 - categorical_accuracy: 0.8794
13920/60000 [=====>........................] - ETA: 1:14 - loss: 0.3898 - categorical_accuracy: 0.8797
13984/60000 [=====>........................] - ETA: 1:14 - loss: 0.3893 - categorical_accuracy: 0.8798
14016/60000 [======>.......................] - ETA: 1:14 - loss: 0.3885 - categorical_accuracy: 0.8801
14080/60000 [======>.......................] - ETA: 1:14 - loss: 0.3872 - categorical_accuracy: 0.8804
14144/60000 [======>.......................] - ETA: 1:14 - loss: 0.3856 - categorical_accuracy: 0.8809
14176/60000 [======>.......................] - ETA: 1:14 - loss: 0.3856 - categorical_accuracy: 0.8811
14208/60000 [======>.......................] - ETA: 1:14 - loss: 0.3868 - categorical_accuracy: 0.8811
14272/60000 [======>.......................] - ETA: 1:14 - loss: 0.3862 - categorical_accuracy: 0.8812
14304/60000 [======>.......................] - ETA: 1:14 - loss: 0.3856 - categorical_accuracy: 0.8814
14368/60000 [======>.......................] - ETA: 1:13 - loss: 0.3850 - categorical_accuracy: 0.8814
14432/60000 [======>.......................] - ETA: 1:13 - loss: 0.3837 - categorical_accuracy: 0.8818
14496/60000 [======>.......................] - ETA: 1:13 - loss: 0.3832 - categorical_accuracy: 0.8821
14528/60000 [======>.......................] - ETA: 1:13 - loss: 0.3825 - categorical_accuracy: 0.8824
14592/60000 [======>.......................] - ETA: 1:13 - loss: 0.3811 - categorical_accuracy: 0.8828
14624/60000 [======>.......................] - ETA: 1:13 - loss: 0.3805 - categorical_accuracy: 0.8830
14656/60000 [======>.......................] - ETA: 1:13 - loss: 0.3797 - categorical_accuracy: 0.8832
14720/60000 [======>.......................] - ETA: 1:13 - loss: 0.3791 - categorical_accuracy: 0.8835
14784/60000 [======>.......................] - ETA: 1:13 - loss: 0.3779 - categorical_accuracy: 0.8839
14816/60000 [======>.......................] - ETA: 1:13 - loss: 0.3771 - categorical_accuracy: 0.8842
14880/60000 [======>.......................] - ETA: 1:12 - loss: 0.3758 - categorical_accuracy: 0.8846
14912/60000 [======>.......................] - ETA: 1:12 - loss: 0.3751 - categorical_accuracy: 0.8849
14944/60000 [======>.......................] - ETA: 1:12 - loss: 0.3744 - categorical_accuracy: 0.8850
15008/60000 [======>.......................] - ETA: 1:12 - loss: 0.3734 - categorical_accuracy: 0.8853
15072/60000 [======>.......................] - ETA: 1:12 - loss: 0.3725 - categorical_accuracy: 0.8855
15104/60000 [======>.......................] - ETA: 1:12 - loss: 0.3722 - categorical_accuracy: 0.8856
15168/60000 [======>.......................] - ETA: 1:12 - loss: 0.3709 - categorical_accuracy: 0.8860
15200/60000 [======>.......................] - ETA: 1:12 - loss: 0.3702 - categorical_accuracy: 0.8863
15264/60000 [======>.......................] - ETA: 1:12 - loss: 0.3693 - categorical_accuracy: 0.8865
15296/60000 [======>.......................] - ETA: 1:12 - loss: 0.3687 - categorical_accuracy: 0.8867
15328/60000 [======>.......................] - ETA: 1:12 - loss: 0.3684 - categorical_accuracy: 0.8869
15360/60000 [======>.......................] - ETA: 1:12 - loss: 0.3679 - categorical_accuracy: 0.8870
15424/60000 [======>.......................] - ETA: 1:12 - loss: 0.3666 - categorical_accuracy: 0.8874
15488/60000 [======>.......................] - ETA: 1:11 - loss: 0.3653 - categorical_accuracy: 0.8878
15552/60000 [======>.......................] - ETA: 1:11 - loss: 0.3641 - categorical_accuracy: 0.8881
15584/60000 [======>.......................] - ETA: 1:11 - loss: 0.3635 - categorical_accuracy: 0.8883
15616/60000 [======>.......................] - ETA: 1:11 - loss: 0.3632 - categorical_accuracy: 0.8884
15680/60000 [======>.......................] - ETA: 1:11 - loss: 0.3625 - categorical_accuracy: 0.8886
15744/60000 [======>.......................] - ETA: 1:11 - loss: 0.3621 - categorical_accuracy: 0.8887
15808/60000 [======>.......................] - ETA: 1:11 - loss: 0.3613 - categorical_accuracy: 0.8890
15840/60000 [======>.......................] - ETA: 1:11 - loss: 0.3616 - categorical_accuracy: 0.8889
15904/60000 [======>.......................] - ETA: 1:11 - loss: 0.3607 - categorical_accuracy: 0.8891
15936/60000 [======>.......................] - ETA: 1:11 - loss: 0.3601 - categorical_accuracy: 0.8894
15968/60000 [======>.......................] - ETA: 1:11 - loss: 0.3595 - categorical_accuracy: 0.8896
16000/60000 [=======>......................] - ETA: 1:11 - loss: 0.3594 - categorical_accuracy: 0.8896
16064/60000 [=======>......................] - ETA: 1:10 - loss: 0.3586 - categorical_accuracy: 0.8899
16128/60000 [=======>......................] - ETA: 1:10 - loss: 0.3577 - categorical_accuracy: 0.8901
16192/60000 [=======>......................] - ETA: 1:10 - loss: 0.3571 - categorical_accuracy: 0.8902
16224/60000 [=======>......................] - ETA: 1:10 - loss: 0.3565 - categorical_accuracy: 0.8904
16256/60000 [=======>......................] - ETA: 1:10 - loss: 0.3559 - categorical_accuracy: 0.8906
16288/60000 [=======>......................] - ETA: 1:10 - loss: 0.3553 - categorical_accuracy: 0.8908
16352/60000 [=======>......................] - ETA: 1:10 - loss: 0.3543 - categorical_accuracy: 0.8911
16384/60000 [=======>......................] - ETA: 1:10 - loss: 0.3539 - categorical_accuracy: 0.8912
16448/60000 [=======>......................] - ETA: 1:10 - loss: 0.3534 - categorical_accuracy: 0.8914
16480/60000 [=======>......................] - ETA: 1:10 - loss: 0.3528 - categorical_accuracy: 0.8916
16544/60000 [=======>......................] - ETA: 1:10 - loss: 0.3520 - categorical_accuracy: 0.8919
16608/60000 [=======>......................] - ETA: 1:09 - loss: 0.3515 - categorical_accuracy: 0.8919
16672/60000 [=======>......................] - ETA: 1:09 - loss: 0.3510 - categorical_accuracy: 0.8921
16736/60000 [=======>......................] - ETA: 1:09 - loss: 0.3502 - categorical_accuracy: 0.8923
16768/60000 [=======>......................] - ETA: 1:09 - loss: 0.3502 - categorical_accuracy: 0.8924
16800/60000 [=======>......................] - ETA: 1:09 - loss: 0.3500 - categorical_accuracy: 0.8924
16832/60000 [=======>......................] - ETA: 1:09 - loss: 0.3499 - categorical_accuracy: 0.8924
16864/60000 [=======>......................] - ETA: 1:09 - loss: 0.3495 - categorical_accuracy: 0.8926
16896/60000 [=======>......................] - ETA: 1:09 - loss: 0.3489 - categorical_accuracy: 0.8928
16960/60000 [=======>......................] - ETA: 1:09 - loss: 0.3487 - categorical_accuracy: 0.8929
16992/60000 [=======>......................] - ETA: 1:09 - loss: 0.3483 - categorical_accuracy: 0.8929
17024/60000 [=======>......................] - ETA: 1:09 - loss: 0.3481 - categorical_accuracy: 0.8930
17088/60000 [=======>......................] - ETA: 1:09 - loss: 0.3471 - categorical_accuracy: 0.8932
17152/60000 [=======>......................] - ETA: 1:09 - loss: 0.3467 - categorical_accuracy: 0.8933
17184/60000 [=======>......................] - ETA: 1:09 - loss: 0.3466 - categorical_accuracy: 0.8934
17248/60000 [=======>......................] - ETA: 1:08 - loss: 0.3457 - categorical_accuracy: 0.8936
17312/60000 [=======>......................] - ETA: 1:08 - loss: 0.3448 - categorical_accuracy: 0.8938
17344/60000 [=======>......................] - ETA: 1:08 - loss: 0.3443 - categorical_accuracy: 0.8940
17376/60000 [=======>......................] - ETA: 1:08 - loss: 0.3440 - categorical_accuracy: 0.8941
17408/60000 [=======>......................] - ETA: 1:08 - loss: 0.3436 - categorical_accuracy: 0.8942
17440/60000 [=======>......................] - ETA: 1:08 - loss: 0.3434 - categorical_accuracy: 0.8943
17472/60000 [=======>......................] - ETA: 1:08 - loss: 0.3433 - categorical_accuracy: 0.8943
17504/60000 [=======>......................] - ETA: 1:08 - loss: 0.3430 - categorical_accuracy: 0.8944
17536/60000 [=======>......................] - ETA: 1:08 - loss: 0.3424 - categorical_accuracy: 0.8946
17600/60000 [=======>......................] - ETA: 1:08 - loss: 0.3413 - categorical_accuracy: 0.8950
17664/60000 [=======>......................] - ETA: 1:08 - loss: 0.3407 - categorical_accuracy: 0.8953
17728/60000 [=======>......................] - ETA: 1:08 - loss: 0.3397 - categorical_accuracy: 0.8956
17792/60000 [=======>......................] - ETA: 1:08 - loss: 0.3390 - categorical_accuracy: 0.8959
17824/60000 [=======>......................] - ETA: 1:07 - loss: 0.3389 - categorical_accuracy: 0.8960
17888/60000 [=======>......................] - ETA: 1:07 - loss: 0.3383 - categorical_accuracy: 0.8962
17920/60000 [=======>......................] - ETA: 1:07 - loss: 0.3380 - categorical_accuracy: 0.8962
17984/60000 [=======>......................] - ETA: 1:07 - loss: 0.3372 - categorical_accuracy: 0.8965
18016/60000 [========>.....................] - ETA: 1:07 - loss: 0.3369 - categorical_accuracy: 0.8965
18048/60000 [========>.....................] - ETA: 1:07 - loss: 0.3364 - categorical_accuracy: 0.8967
18080/60000 [========>.....................] - ETA: 1:07 - loss: 0.3358 - categorical_accuracy: 0.8968
18144/60000 [========>.....................] - ETA: 1:07 - loss: 0.3350 - categorical_accuracy: 0.8970
18208/60000 [========>.....................] - ETA: 1:07 - loss: 0.3345 - categorical_accuracy: 0.8971
18272/60000 [========>.....................] - ETA: 1:07 - loss: 0.3336 - categorical_accuracy: 0.8974
18336/60000 [========>.....................] - ETA: 1:07 - loss: 0.3329 - categorical_accuracy: 0.8976
18400/60000 [========>.....................] - ETA: 1:07 - loss: 0.3325 - categorical_accuracy: 0.8978
18432/60000 [========>.....................] - ETA: 1:06 - loss: 0.3322 - categorical_accuracy: 0.8979
18464/60000 [========>.....................] - ETA: 1:06 - loss: 0.3318 - categorical_accuracy: 0.8980
18496/60000 [========>.....................] - ETA: 1:06 - loss: 0.3313 - categorical_accuracy: 0.8982
18560/60000 [========>.....................] - ETA: 1:06 - loss: 0.3305 - categorical_accuracy: 0.8984
18624/60000 [========>.....................] - ETA: 1:06 - loss: 0.3296 - categorical_accuracy: 0.8987
18688/60000 [========>.....................] - ETA: 1:06 - loss: 0.3298 - categorical_accuracy: 0.8988
18720/60000 [========>.....................] - ETA: 1:06 - loss: 0.3297 - categorical_accuracy: 0.8988
18784/60000 [========>.....................] - ETA: 1:06 - loss: 0.3288 - categorical_accuracy: 0.8991
18816/60000 [========>.....................] - ETA: 1:06 - loss: 0.3286 - categorical_accuracy: 0.8991
18848/60000 [========>.....................] - ETA: 1:06 - loss: 0.3282 - categorical_accuracy: 0.8992
18880/60000 [========>.....................] - ETA: 1:06 - loss: 0.3284 - categorical_accuracy: 0.8993
18944/60000 [========>.....................] - ETA: 1:06 - loss: 0.3274 - categorical_accuracy: 0.8995
19008/60000 [========>.....................] - ETA: 1:05 - loss: 0.3264 - categorical_accuracy: 0.8998
19072/60000 [========>.....................] - ETA: 1:05 - loss: 0.3256 - categorical_accuracy: 0.9001
19136/60000 [========>.....................] - ETA: 1:05 - loss: 0.3252 - categorical_accuracy: 0.9002
19168/60000 [========>.....................] - ETA: 1:05 - loss: 0.3250 - categorical_accuracy: 0.9004
19200/60000 [========>.....................] - ETA: 1:05 - loss: 0.3247 - categorical_accuracy: 0.9005
19232/60000 [========>.....................] - ETA: 1:05 - loss: 0.3246 - categorical_accuracy: 0.9004
19264/60000 [========>.....................] - ETA: 1:05 - loss: 0.3242 - categorical_accuracy: 0.9005
19296/60000 [========>.....................] - ETA: 1:05 - loss: 0.3238 - categorical_accuracy: 0.9006
19328/60000 [========>.....................] - ETA: 1:05 - loss: 0.3234 - categorical_accuracy: 0.9007
19360/60000 [========>.....................] - ETA: 1:05 - loss: 0.3232 - categorical_accuracy: 0.9008
19424/60000 [========>.....................] - ETA: 1:05 - loss: 0.3225 - categorical_accuracy: 0.9011
19456/60000 [========>.....................] - ETA: 1:05 - loss: 0.3222 - categorical_accuracy: 0.9012
19488/60000 [========>.....................] - ETA: 1:05 - loss: 0.3217 - categorical_accuracy: 0.9013
19552/60000 [========>.....................] - ETA: 1:05 - loss: 0.3212 - categorical_accuracy: 0.9015
19584/60000 [========>.....................] - ETA: 1:05 - loss: 0.3214 - categorical_accuracy: 0.9016
19648/60000 [========>.....................] - ETA: 1:04 - loss: 0.3207 - categorical_accuracy: 0.9017
19712/60000 [========>.....................] - ETA: 1:04 - loss: 0.3198 - categorical_accuracy: 0.9020
19776/60000 [========>.....................] - ETA: 1:04 - loss: 0.3190 - categorical_accuracy: 0.9023
19808/60000 [========>.....................] - ETA: 1:04 - loss: 0.3190 - categorical_accuracy: 0.9021
19872/60000 [========>.....................] - ETA: 1:04 - loss: 0.3192 - categorical_accuracy: 0.9021
19936/60000 [========>.....................] - ETA: 1:04 - loss: 0.3184 - categorical_accuracy: 0.9023
20000/60000 [=========>....................] - ETA: 1:04 - loss: 0.3176 - categorical_accuracy: 0.9025
20064/60000 [=========>....................] - ETA: 1:04 - loss: 0.3173 - categorical_accuracy: 0.9026
20096/60000 [=========>....................] - ETA: 1:04 - loss: 0.3170 - categorical_accuracy: 0.9027
20128/60000 [=========>....................] - ETA: 1:04 - loss: 0.3166 - categorical_accuracy: 0.9028
20192/60000 [=========>....................] - ETA: 1:04 - loss: 0.3160 - categorical_accuracy: 0.9030
20224/60000 [=========>....................] - ETA: 1:03 - loss: 0.3155 - categorical_accuracy: 0.9032
20288/60000 [=========>....................] - ETA: 1:03 - loss: 0.3150 - categorical_accuracy: 0.9034
20320/60000 [=========>....................] - ETA: 1:03 - loss: 0.3145 - categorical_accuracy: 0.9035
20352/60000 [=========>....................] - ETA: 1:03 - loss: 0.3142 - categorical_accuracy: 0.9035
20384/60000 [=========>....................] - ETA: 1:03 - loss: 0.3140 - categorical_accuracy: 0.9036
20416/60000 [=========>....................] - ETA: 1:03 - loss: 0.3136 - categorical_accuracy: 0.9038
20480/60000 [=========>....................] - ETA: 1:03 - loss: 0.3133 - categorical_accuracy: 0.9037
20512/60000 [=========>....................] - ETA: 1:03 - loss: 0.3128 - categorical_accuracy: 0.9039
20544/60000 [=========>....................] - ETA: 1:03 - loss: 0.3125 - categorical_accuracy: 0.9040
20576/60000 [=========>....................] - ETA: 1:03 - loss: 0.3123 - categorical_accuracy: 0.9040
20608/60000 [=========>....................] - ETA: 1:03 - loss: 0.3120 - categorical_accuracy: 0.9041
20640/60000 [=========>....................] - ETA: 1:03 - loss: 0.3116 - categorical_accuracy: 0.9042
20672/60000 [=========>....................] - ETA: 1:03 - loss: 0.3114 - categorical_accuracy: 0.9042
20704/60000 [=========>....................] - ETA: 1:03 - loss: 0.3113 - categorical_accuracy: 0.9042
20768/60000 [=========>....................] - ETA: 1:03 - loss: 0.3106 - categorical_accuracy: 0.9044
20832/60000 [=========>....................] - ETA: 1:03 - loss: 0.3103 - categorical_accuracy: 0.9046
20896/60000 [=========>....................] - ETA: 1:02 - loss: 0.3094 - categorical_accuracy: 0.9049
20928/60000 [=========>....................] - ETA: 1:02 - loss: 0.3090 - categorical_accuracy: 0.9050
20960/60000 [=========>....................] - ETA: 1:02 - loss: 0.3087 - categorical_accuracy: 0.9051
21024/60000 [=========>....................] - ETA: 1:02 - loss: 0.3079 - categorical_accuracy: 0.9053
21056/60000 [=========>....................] - ETA: 1:02 - loss: 0.3079 - categorical_accuracy: 0.9053
21120/60000 [=========>....................] - ETA: 1:02 - loss: 0.3074 - categorical_accuracy: 0.9054
21184/60000 [=========>....................] - ETA: 1:02 - loss: 0.3066 - categorical_accuracy: 0.9057
21248/60000 [=========>....................] - ETA: 1:02 - loss: 0.3061 - categorical_accuracy: 0.9059
21312/60000 [=========>....................] - ETA: 1:02 - loss: 0.3055 - categorical_accuracy: 0.9061
21344/60000 [=========>....................] - ETA: 1:02 - loss: 0.3052 - categorical_accuracy: 0.9062
21376/60000 [=========>....................] - ETA: 1:02 - loss: 0.3049 - categorical_accuracy: 0.9062
21440/60000 [=========>....................] - ETA: 1:02 - loss: 0.3049 - categorical_accuracy: 0.9062
21504/60000 [=========>....................] - ETA: 1:01 - loss: 0.3041 - categorical_accuracy: 0.9065
21536/60000 [=========>....................] - ETA: 1:01 - loss: 0.3040 - categorical_accuracy: 0.9065
21568/60000 [=========>....................] - ETA: 1:01 - loss: 0.3036 - categorical_accuracy: 0.9067
21600/60000 [=========>....................] - ETA: 1:01 - loss: 0.3035 - categorical_accuracy: 0.9067
21632/60000 [=========>....................] - ETA: 1:01 - loss: 0.3032 - categorical_accuracy: 0.9068
21696/60000 [=========>....................] - ETA: 1:01 - loss: 0.3025 - categorical_accuracy: 0.9070
21760/60000 [=========>....................] - ETA: 1:01 - loss: 0.3020 - categorical_accuracy: 0.9072
21792/60000 [=========>....................] - ETA: 1:01 - loss: 0.3017 - categorical_accuracy: 0.9072
21856/60000 [=========>....................] - ETA: 1:01 - loss: 0.3010 - categorical_accuracy: 0.9074
21920/60000 [=========>....................] - ETA: 1:01 - loss: 0.3006 - categorical_accuracy: 0.9076
21952/60000 [=========>....................] - ETA: 1:01 - loss: 0.3002 - categorical_accuracy: 0.9077
21984/60000 [=========>....................] - ETA: 1:01 - loss: 0.3002 - categorical_accuracy: 0.9077
22016/60000 [==========>...................] - ETA: 1:01 - loss: 0.2999 - categorical_accuracy: 0.9077
22048/60000 [==========>...................] - ETA: 1:01 - loss: 0.3001 - categorical_accuracy: 0.9077
22080/60000 [==========>...................] - ETA: 1:00 - loss: 0.2998 - categorical_accuracy: 0.9078
22112/60000 [==========>...................] - ETA: 1:00 - loss: 0.2995 - categorical_accuracy: 0.9079
22144/60000 [==========>...................] - ETA: 1:00 - loss: 0.2991 - categorical_accuracy: 0.9081
22176/60000 [==========>...................] - ETA: 1:00 - loss: 0.2989 - categorical_accuracy: 0.9081
22240/60000 [==========>...................] - ETA: 1:00 - loss: 0.2983 - categorical_accuracy: 0.9083
22272/60000 [==========>...................] - ETA: 1:00 - loss: 0.2980 - categorical_accuracy: 0.9084
22336/60000 [==========>...................] - ETA: 1:00 - loss: 0.2974 - categorical_accuracy: 0.9085
22400/60000 [==========>...................] - ETA: 1:00 - loss: 0.2972 - categorical_accuracy: 0.9086
22432/60000 [==========>...................] - ETA: 1:00 - loss: 0.2973 - categorical_accuracy: 0.9086
22496/60000 [==========>...................] - ETA: 1:00 - loss: 0.2970 - categorical_accuracy: 0.9087
22528/60000 [==========>...................] - ETA: 1:00 - loss: 0.2967 - categorical_accuracy: 0.9088
22560/60000 [==========>...................] - ETA: 1:00 - loss: 0.2963 - categorical_accuracy: 0.9089
22592/60000 [==========>...................] - ETA: 1:00 - loss: 0.2961 - categorical_accuracy: 0.9090
22624/60000 [==========>...................] - ETA: 1:00 - loss: 0.2958 - categorical_accuracy: 0.9090
22656/60000 [==========>...................] - ETA: 1:00 - loss: 0.2960 - categorical_accuracy: 0.9090
22688/60000 [==========>...................] - ETA: 1:00 - loss: 0.2957 - categorical_accuracy: 0.9092
22752/60000 [==========>...................] - ETA: 59s - loss: 0.2956 - categorical_accuracy: 0.9092 
22784/60000 [==========>...................] - ETA: 59s - loss: 0.2957 - categorical_accuracy: 0.9092
22816/60000 [==========>...................] - ETA: 59s - loss: 0.2954 - categorical_accuracy: 0.9093
22848/60000 [==========>...................] - ETA: 59s - loss: 0.2951 - categorical_accuracy: 0.9094
22880/60000 [==========>...................] - ETA: 59s - loss: 0.2948 - categorical_accuracy: 0.9095
22944/60000 [==========>...................] - ETA: 59s - loss: 0.2944 - categorical_accuracy: 0.9097
22976/60000 [==========>...................] - ETA: 59s - loss: 0.2944 - categorical_accuracy: 0.9097
23040/60000 [==========>...................] - ETA: 59s - loss: 0.2940 - categorical_accuracy: 0.9099
23072/60000 [==========>...................] - ETA: 59s - loss: 0.2938 - categorical_accuracy: 0.9099
23136/60000 [==========>...................] - ETA: 59s - loss: 0.2933 - categorical_accuracy: 0.9101
23200/60000 [==========>...................] - ETA: 59s - loss: 0.2930 - categorical_accuracy: 0.9102
23264/60000 [==========>...................] - ETA: 59s - loss: 0.2923 - categorical_accuracy: 0.9104
23296/60000 [==========>...................] - ETA: 59s - loss: 0.2921 - categorical_accuracy: 0.9105
23360/60000 [==========>...................] - ETA: 58s - loss: 0.2917 - categorical_accuracy: 0.9106
23392/60000 [==========>...................] - ETA: 58s - loss: 0.2914 - categorical_accuracy: 0.9107
23424/60000 [==========>...................] - ETA: 58s - loss: 0.2911 - categorical_accuracy: 0.9108
23488/60000 [==========>...................] - ETA: 58s - loss: 0.2911 - categorical_accuracy: 0.9108
23552/60000 [==========>...................] - ETA: 58s - loss: 0.2909 - categorical_accuracy: 0.9110
23584/60000 [==========>...................] - ETA: 58s - loss: 0.2908 - categorical_accuracy: 0.9110
23616/60000 [==========>...................] - ETA: 58s - loss: 0.2905 - categorical_accuracy: 0.9111
23648/60000 [==========>...................] - ETA: 58s - loss: 0.2902 - categorical_accuracy: 0.9112
23712/60000 [==========>...................] - ETA: 58s - loss: 0.2899 - categorical_accuracy: 0.9113
23776/60000 [==========>...................] - ETA: 58s - loss: 0.2895 - categorical_accuracy: 0.9114
23840/60000 [==========>...................] - ETA: 58s - loss: 0.2891 - categorical_accuracy: 0.9115
23904/60000 [==========>...................] - ETA: 58s - loss: 0.2887 - categorical_accuracy: 0.9117
23968/60000 [==========>...................] - ETA: 58s - loss: 0.2882 - categorical_accuracy: 0.9118
24032/60000 [===========>..................] - ETA: 57s - loss: 0.2881 - categorical_accuracy: 0.9119
24096/60000 [===========>..................] - ETA: 57s - loss: 0.2879 - categorical_accuracy: 0.9119
24160/60000 [===========>..................] - ETA: 57s - loss: 0.2873 - categorical_accuracy: 0.9120
24224/60000 [===========>..................] - ETA: 57s - loss: 0.2869 - categorical_accuracy: 0.9121
24256/60000 [===========>..................] - ETA: 57s - loss: 0.2867 - categorical_accuracy: 0.9121
24320/60000 [===========>..................] - ETA: 57s - loss: 0.2863 - categorical_accuracy: 0.9123
24352/60000 [===========>..................] - ETA: 57s - loss: 0.2861 - categorical_accuracy: 0.9123
24416/60000 [===========>..................] - ETA: 57s - loss: 0.2861 - categorical_accuracy: 0.9124
24480/60000 [===========>..................] - ETA: 57s - loss: 0.2856 - categorical_accuracy: 0.9125
24512/60000 [===========>..................] - ETA: 57s - loss: 0.2855 - categorical_accuracy: 0.9125
24544/60000 [===========>..................] - ETA: 57s - loss: 0.2852 - categorical_accuracy: 0.9125
24576/60000 [===========>..................] - ETA: 56s - loss: 0.2849 - categorical_accuracy: 0.9126
24608/60000 [===========>..................] - ETA: 56s - loss: 0.2846 - categorical_accuracy: 0.9127
24640/60000 [===========>..................] - ETA: 56s - loss: 0.2843 - categorical_accuracy: 0.9128
24704/60000 [===========>..................] - ETA: 56s - loss: 0.2840 - categorical_accuracy: 0.9130
24768/60000 [===========>..................] - ETA: 56s - loss: 0.2835 - categorical_accuracy: 0.9132
24832/60000 [===========>..................] - ETA: 56s - loss: 0.2829 - categorical_accuracy: 0.9134
24864/60000 [===========>..................] - ETA: 56s - loss: 0.2826 - categorical_accuracy: 0.9134
24928/60000 [===========>..................] - ETA: 56s - loss: 0.2821 - categorical_accuracy: 0.9136
24960/60000 [===========>..................] - ETA: 56s - loss: 0.2818 - categorical_accuracy: 0.9136
25024/60000 [===========>..................] - ETA: 56s - loss: 0.2813 - categorical_accuracy: 0.9138
25088/60000 [===========>..................] - ETA: 56s - loss: 0.2807 - categorical_accuracy: 0.9140
25120/60000 [===========>..................] - ETA: 56s - loss: 0.2804 - categorical_accuracy: 0.9141
25184/60000 [===========>..................] - ETA: 55s - loss: 0.2802 - categorical_accuracy: 0.9142
25248/60000 [===========>..................] - ETA: 55s - loss: 0.2799 - categorical_accuracy: 0.9142
25280/60000 [===========>..................] - ETA: 55s - loss: 0.2795 - categorical_accuracy: 0.9143
25344/60000 [===========>..................] - ETA: 55s - loss: 0.2791 - categorical_accuracy: 0.9145
25408/60000 [===========>..................] - ETA: 55s - loss: 0.2784 - categorical_accuracy: 0.9147
25472/60000 [===========>..................] - ETA: 55s - loss: 0.2782 - categorical_accuracy: 0.9148
25504/60000 [===========>..................] - ETA: 55s - loss: 0.2780 - categorical_accuracy: 0.9149
25568/60000 [===========>..................] - ETA: 55s - loss: 0.2781 - categorical_accuracy: 0.9149
25600/60000 [===========>..................] - ETA: 55s - loss: 0.2780 - categorical_accuracy: 0.9149
25632/60000 [===========>..................] - ETA: 55s - loss: 0.2777 - categorical_accuracy: 0.9150
25664/60000 [===========>..................] - ETA: 55s - loss: 0.2775 - categorical_accuracy: 0.9151
25696/60000 [===========>..................] - ETA: 55s - loss: 0.2772 - categorical_accuracy: 0.9152
25728/60000 [===========>..................] - ETA: 55s - loss: 0.2769 - categorical_accuracy: 0.9153
25792/60000 [===========>..................] - ETA: 54s - loss: 0.2766 - categorical_accuracy: 0.9153
25824/60000 [===========>..................] - ETA: 54s - loss: 0.2764 - categorical_accuracy: 0.9153
25888/60000 [===========>..................] - ETA: 54s - loss: 0.2761 - categorical_accuracy: 0.9154
25952/60000 [===========>..................] - ETA: 54s - loss: 0.2758 - categorical_accuracy: 0.9155
25984/60000 [===========>..................] - ETA: 54s - loss: 0.2758 - categorical_accuracy: 0.9156
26016/60000 [============>.................] - ETA: 54s - loss: 0.2755 - categorical_accuracy: 0.9156
26080/60000 [============>.................] - ETA: 54s - loss: 0.2750 - categorical_accuracy: 0.9158
26144/60000 [============>.................] - ETA: 54s - loss: 0.2744 - categorical_accuracy: 0.9159
26208/60000 [============>.................] - ETA: 54s - loss: 0.2740 - categorical_accuracy: 0.9161
26272/60000 [============>.................] - ETA: 54s - loss: 0.2738 - categorical_accuracy: 0.9161
26336/60000 [============>.................] - ETA: 54s - loss: 0.2738 - categorical_accuracy: 0.9162
26368/60000 [============>.................] - ETA: 54s - loss: 0.2736 - categorical_accuracy: 0.9163
26400/60000 [============>.................] - ETA: 53s - loss: 0.2735 - categorical_accuracy: 0.9163
26432/60000 [============>.................] - ETA: 53s - loss: 0.2733 - categorical_accuracy: 0.9164
26464/60000 [============>.................] - ETA: 53s - loss: 0.2732 - categorical_accuracy: 0.9164
26496/60000 [============>.................] - ETA: 53s - loss: 0.2732 - categorical_accuracy: 0.9164
26560/60000 [============>.................] - ETA: 53s - loss: 0.2728 - categorical_accuracy: 0.9166
26624/60000 [============>.................] - ETA: 53s - loss: 0.2723 - categorical_accuracy: 0.9168
26656/60000 [============>.................] - ETA: 53s - loss: 0.2720 - categorical_accuracy: 0.9168
26688/60000 [============>.................] - ETA: 53s - loss: 0.2718 - categorical_accuracy: 0.9169
26720/60000 [============>.................] - ETA: 53s - loss: 0.2715 - categorical_accuracy: 0.9170
26784/60000 [============>.................] - ETA: 53s - loss: 0.2711 - categorical_accuracy: 0.9170
26816/60000 [============>.................] - ETA: 53s - loss: 0.2709 - categorical_accuracy: 0.9171
26880/60000 [============>.................] - ETA: 53s - loss: 0.2710 - categorical_accuracy: 0.9171
26944/60000 [============>.................] - ETA: 53s - loss: 0.2707 - categorical_accuracy: 0.9172
27008/60000 [============>.................] - ETA: 52s - loss: 0.2705 - categorical_accuracy: 0.9172
27072/60000 [============>.................] - ETA: 52s - loss: 0.2701 - categorical_accuracy: 0.9173
27136/60000 [============>.................] - ETA: 52s - loss: 0.2697 - categorical_accuracy: 0.9175
27168/60000 [============>.................] - ETA: 52s - loss: 0.2694 - categorical_accuracy: 0.9176
27232/60000 [============>.................] - ETA: 52s - loss: 0.2690 - categorical_accuracy: 0.9177
27296/60000 [============>.................] - ETA: 52s - loss: 0.2685 - categorical_accuracy: 0.9178
27360/60000 [============>.................] - ETA: 52s - loss: 0.2681 - categorical_accuracy: 0.9179
27424/60000 [============>.................] - ETA: 52s - loss: 0.2675 - categorical_accuracy: 0.9181
27456/60000 [============>.................] - ETA: 52s - loss: 0.2673 - categorical_accuracy: 0.9182
27520/60000 [============>.................] - ETA: 52s - loss: 0.2667 - categorical_accuracy: 0.9184
27552/60000 [============>.................] - ETA: 52s - loss: 0.2665 - categorical_accuracy: 0.9184
27616/60000 [============>.................] - ETA: 51s - loss: 0.2661 - categorical_accuracy: 0.9186
27680/60000 [============>.................] - ETA: 51s - loss: 0.2659 - categorical_accuracy: 0.9186
27712/60000 [============>.................] - ETA: 51s - loss: 0.2660 - categorical_accuracy: 0.9187
27744/60000 [============>.................] - ETA: 51s - loss: 0.2658 - categorical_accuracy: 0.9187
27776/60000 [============>.................] - ETA: 51s - loss: 0.2656 - categorical_accuracy: 0.9187
27840/60000 [============>.................] - ETA: 51s - loss: 0.2652 - categorical_accuracy: 0.9188
27904/60000 [============>.................] - ETA: 51s - loss: 0.2648 - categorical_accuracy: 0.9189
27968/60000 [============>.................] - ETA: 51s - loss: 0.2645 - categorical_accuracy: 0.9189
28032/60000 [=============>................] - ETA: 51s - loss: 0.2641 - categorical_accuracy: 0.9191
28064/60000 [=============>................] - ETA: 51s - loss: 0.2639 - categorical_accuracy: 0.9192
28096/60000 [=============>................] - ETA: 51s - loss: 0.2636 - categorical_accuracy: 0.9192
28160/60000 [=============>................] - ETA: 51s - loss: 0.2634 - categorical_accuracy: 0.9193
28224/60000 [=============>................] - ETA: 50s - loss: 0.2630 - categorical_accuracy: 0.9194
28288/60000 [=============>................] - ETA: 50s - loss: 0.2625 - categorical_accuracy: 0.9196
28320/60000 [=============>................] - ETA: 50s - loss: 0.2623 - categorical_accuracy: 0.9196
28352/60000 [=============>................] - ETA: 50s - loss: 0.2620 - categorical_accuracy: 0.9197
28416/60000 [=============>................] - ETA: 50s - loss: 0.2616 - categorical_accuracy: 0.9198
28448/60000 [=============>................] - ETA: 50s - loss: 0.2614 - categorical_accuracy: 0.9199
28512/60000 [=============>................] - ETA: 50s - loss: 0.2608 - categorical_accuracy: 0.9201
28576/60000 [=============>................] - ETA: 50s - loss: 0.2604 - categorical_accuracy: 0.9201
28640/60000 [=============>................] - ETA: 50s - loss: 0.2601 - categorical_accuracy: 0.9203
28704/60000 [=============>................] - ETA: 50s - loss: 0.2597 - categorical_accuracy: 0.9204
28736/60000 [=============>................] - ETA: 50s - loss: 0.2597 - categorical_accuracy: 0.9204
28768/60000 [=============>................] - ETA: 50s - loss: 0.2597 - categorical_accuracy: 0.9204
28800/60000 [=============>................] - ETA: 50s - loss: 0.2596 - categorical_accuracy: 0.9204
28832/60000 [=============>................] - ETA: 49s - loss: 0.2594 - categorical_accuracy: 0.9205
28864/60000 [=============>................] - ETA: 49s - loss: 0.2591 - categorical_accuracy: 0.9206
28896/60000 [=============>................] - ETA: 49s - loss: 0.2590 - categorical_accuracy: 0.9205
28928/60000 [=============>................] - ETA: 49s - loss: 0.2590 - categorical_accuracy: 0.9206
28960/60000 [=============>................] - ETA: 49s - loss: 0.2588 - categorical_accuracy: 0.9206
29024/60000 [=============>................] - ETA: 49s - loss: 0.2584 - categorical_accuracy: 0.9207
29056/60000 [=============>................] - ETA: 49s - loss: 0.2584 - categorical_accuracy: 0.9207
29120/60000 [=============>................] - ETA: 49s - loss: 0.2580 - categorical_accuracy: 0.9208
29152/60000 [=============>................] - ETA: 49s - loss: 0.2579 - categorical_accuracy: 0.9208
29184/60000 [=============>................] - ETA: 49s - loss: 0.2578 - categorical_accuracy: 0.9209
29216/60000 [=============>................] - ETA: 49s - loss: 0.2576 - categorical_accuracy: 0.9209
29280/60000 [=============>................] - ETA: 49s - loss: 0.2574 - categorical_accuracy: 0.9210
29312/60000 [=============>................] - ETA: 49s - loss: 0.2571 - categorical_accuracy: 0.9211
29376/60000 [=============>................] - ETA: 49s - loss: 0.2568 - categorical_accuracy: 0.9212
29408/60000 [=============>................] - ETA: 49s - loss: 0.2566 - categorical_accuracy: 0.9212
29472/60000 [=============>................] - ETA: 48s - loss: 0.2566 - categorical_accuracy: 0.9213
29536/60000 [=============>................] - ETA: 48s - loss: 0.2562 - categorical_accuracy: 0.9214
29600/60000 [=============>................] - ETA: 48s - loss: 0.2561 - categorical_accuracy: 0.9214
29664/60000 [=============>................] - ETA: 48s - loss: 0.2556 - categorical_accuracy: 0.9216
29728/60000 [=============>................] - ETA: 48s - loss: 0.2552 - categorical_accuracy: 0.9217
29792/60000 [=============>................] - ETA: 48s - loss: 0.2547 - categorical_accuracy: 0.9219
29824/60000 [=============>................] - ETA: 48s - loss: 0.2545 - categorical_accuracy: 0.9219
29888/60000 [=============>................] - ETA: 48s - loss: 0.2542 - categorical_accuracy: 0.9220
29920/60000 [=============>................] - ETA: 48s - loss: 0.2540 - categorical_accuracy: 0.9221
29952/60000 [=============>................] - ETA: 48s - loss: 0.2538 - categorical_accuracy: 0.9221
30016/60000 [==============>...............] - ETA: 48s - loss: 0.2534 - categorical_accuracy: 0.9222
30048/60000 [==============>...............] - ETA: 47s - loss: 0.2533 - categorical_accuracy: 0.9222
30080/60000 [==============>...............] - ETA: 47s - loss: 0.2532 - categorical_accuracy: 0.9222
30144/60000 [==============>...............] - ETA: 47s - loss: 0.2527 - categorical_accuracy: 0.9224
30208/60000 [==============>...............] - ETA: 47s - loss: 0.2525 - categorical_accuracy: 0.9225
30240/60000 [==============>...............] - ETA: 47s - loss: 0.2523 - categorical_accuracy: 0.9226
30272/60000 [==============>...............] - ETA: 47s - loss: 0.2521 - categorical_accuracy: 0.9226
30336/60000 [==============>...............] - ETA: 47s - loss: 0.2520 - categorical_accuracy: 0.9226
30368/60000 [==============>...............] - ETA: 47s - loss: 0.2517 - categorical_accuracy: 0.9227
30400/60000 [==============>...............] - ETA: 47s - loss: 0.2517 - categorical_accuracy: 0.9227
30464/60000 [==============>...............] - ETA: 47s - loss: 0.2513 - categorical_accuracy: 0.9228
30528/60000 [==============>...............] - ETA: 47s - loss: 0.2510 - categorical_accuracy: 0.9229
30592/60000 [==============>...............] - ETA: 47s - loss: 0.2506 - categorical_accuracy: 0.9231
30624/60000 [==============>...............] - ETA: 47s - loss: 0.2505 - categorical_accuracy: 0.9231
30688/60000 [==============>...............] - ETA: 46s - loss: 0.2503 - categorical_accuracy: 0.9231
30720/60000 [==============>...............] - ETA: 46s - loss: 0.2501 - categorical_accuracy: 0.9232
30784/60000 [==============>...............] - ETA: 46s - loss: 0.2500 - categorical_accuracy: 0.9232
30816/60000 [==============>...............] - ETA: 46s - loss: 0.2499 - categorical_accuracy: 0.9233
30848/60000 [==============>...............] - ETA: 46s - loss: 0.2497 - categorical_accuracy: 0.9234
30880/60000 [==============>...............] - ETA: 46s - loss: 0.2496 - categorical_accuracy: 0.9234
30944/60000 [==============>...............] - ETA: 46s - loss: 0.2495 - categorical_accuracy: 0.9234
31008/60000 [==============>...............] - ETA: 46s - loss: 0.2494 - categorical_accuracy: 0.9234
31072/60000 [==============>...............] - ETA: 46s - loss: 0.2491 - categorical_accuracy: 0.9235
31136/60000 [==============>...............] - ETA: 46s - loss: 0.2489 - categorical_accuracy: 0.9236
31168/60000 [==============>...............] - ETA: 46s - loss: 0.2487 - categorical_accuracy: 0.9236
31200/60000 [==============>...............] - ETA: 46s - loss: 0.2485 - categorical_accuracy: 0.9237
31232/60000 [==============>...............] - ETA: 46s - loss: 0.2483 - categorical_accuracy: 0.9237
31264/60000 [==============>...............] - ETA: 45s - loss: 0.2483 - categorical_accuracy: 0.9237
31296/60000 [==============>...............] - ETA: 45s - loss: 0.2481 - categorical_accuracy: 0.9238
31328/60000 [==============>...............] - ETA: 45s - loss: 0.2479 - categorical_accuracy: 0.9239
31392/60000 [==============>...............] - ETA: 45s - loss: 0.2476 - categorical_accuracy: 0.9240
31456/60000 [==============>...............] - ETA: 45s - loss: 0.2472 - categorical_accuracy: 0.9241
31520/60000 [==============>...............] - ETA: 45s - loss: 0.2468 - categorical_accuracy: 0.9242
31584/60000 [==============>...............] - ETA: 45s - loss: 0.2464 - categorical_accuracy: 0.9244
31648/60000 [==============>...............] - ETA: 45s - loss: 0.2462 - categorical_accuracy: 0.9244
31680/60000 [==============>...............] - ETA: 45s - loss: 0.2459 - categorical_accuracy: 0.9245
31744/60000 [==============>...............] - ETA: 45s - loss: 0.2457 - categorical_accuracy: 0.9246
31808/60000 [==============>...............] - ETA: 45s - loss: 0.2455 - categorical_accuracy: 0.9246
31840/60000 [==============>...............] - ETA: 45s - loss: 0.2452 - categorical_accuracy: 0.9247
31872/60000 [==============>...............] - ETA: 45s - loss: 0.2450 - categorical_accuracy: 0.9248
31904/60000 [==============>...............] - ETA: 44s - loss: 0.2449 - categorical_accuracy: 0.9248
31936/60000 [==============>...............] - ETA: 44s - loss: 0.2446 - categorical_accuracy: 0.9249
31968/60000 [==============>...............] - ETA: 44s - loss: 0.2444 - categorical_accuracy: 0.9250
32000/60000 [===============>..............] - ETA: 44s - loss: 0.2442 - categorical_accuracy: 0.9251
32064/60000 [===============>..............] - ETA: 44s - loss: 0.2439 - categorical_accuracy: 0.9251
32128/60000 [===============>..............] - ETA: 44s - loss: 0.2437 - categorical_accuracy: 0.9252
32160/60000 [===============>..............] - ETA: 44s - loss: 0.2435 - categorical_accuracy: 0.9252
32192/60000 [===============>..............] - ETA: 44s - loss: 0.2435 - categorical_accuracy: 0.9253
32224/60000 [===============>..............] - ETA: 44s - loss: 0.2433 - categorical_accuracy: 0.9253
32288/60000 [===============>..............] - ETA: 44s - loss: 0.2431 - categorical_accuracy: 0.9254
32320/60000 [===============>..............] - ETA: 44s - loss: 0.2430 - categorical_accuracy: 0.9254
32352/60000 [===============>..............] - ETA: 44s - loss: 0.2427 - categorical_accuracy: 0.9255
32416/60000 [===============>..............] - ETA: 44s - loss: 0.2424 - categorical_accuracy: 0.9256
32480/60000 [===============>..............] - ETA: 44s - loss: 0.2421 - categorical_accuracy: 0.9256
32544/60000 [===============>..............] - ETA: 43s - loss: 0.2421 - categorical_accuracy: 0.9257
32608/60000 [===============>..............] - ETA: 43s - loss: 0.2417 - categorical_accuracy: 0.9258
32640/60000 [===============>..............] - ETA: 43s - loss: 0.2415 - categorical_accuracy: 0.9259
32704/60000 [===============>..............] - ETA: 43s - loss: 0.2412 - categorical_accuracy: 0.9260
32768/60000 [===============>..............] - ETA: 43s - loss: 0.2410 - categorical_accuracy: 0.9260
32800/60000 [===============>..............] - ETA: 43s - loss: 0.2409 - categorical_accuracy: 0.9260
32864/60000 [===============>..............] - ETA: 43s - loss: 0.2406 - categorical_accuracy: 0.9261
32896/60000 [===============>..............] - ETA: 43s - loss: 0.2404 - categorical_accuracy: 0.9262
32960/60000 [===============>..............] - ETA: 43s - loss: 0.2401 - categorical_accuracy: 0.9263
32992/60000 [===============>..............] - ETA: 43s - loss: 0.2399 - categorical_accuracy: 0.9263
33024/60000 [===============>..............] - ETA: 43s - loss: 0.2397 - categorical_accuracy: 0.9264
33056/60000 [===============>..............] - ETA: 43s - loss: 0.2395 - categorical_accuracy: 0.9264
33088/60000 [===============>..............] - ETA: 43s - loss: 0.2393 - categorical_accuracy: 0.9265
33152/60000 [===============>..............] - ETA: 42s - loss: 0.2390 - categorical_accuracy: 0.9266
33184/60000 [===============>..............] - ETA: 42s - loss: 0.2388 - categorical_accuracy: 0.9266
33248/60000 [===============>..............] - ETA: 42s - loss: 0.2387 - categorical_accuracy: 0.9266
33280/60000 [===============>..............] - ETA: 42s - loss: 0.2387 - categorical_accuracy: 0.9266
33312/60000 [===============>..............] - ETA: 42s - loss: 0.2386 - categorical_accuracy: 0.9267
33344/60000 [===============>..............] - ETA: 42s - loss: 0.2384 - categorical_accuracy: 0.9267
33376/60000 [===============>..............] - ETA: 42s - loss: 0.2384 - categorical_accuracy: 0.9267
33408/60000 [===============>..............] - ETA: 42s - loss: 0.2382 - categorical_accuracy: 0.9268
33440/60000 [===============>..............] - ETA: 42s - loss: 0.2380 - categorical_accuracy: 0.9268
33472/60000 [===============>..............] - ETA: 42s - loss: 0.2378 - categorical_accuracy: 0.9269
33536/60000 [===============>..............] - ETA: 42s - loss: 0.2376 - categorical_accuracy: 0.9269
33568/60000 [===============>..............] - ETA: 42s - loss: 0.2374 - categorical_accuracy: 0.9270
33600/60000 [===============>..............] - ETA: 42s - loss: 0.2372 - categorical_accuracy: 0.9271
33664/60000 [===============>..............] - ETA: 42s - loss: 0.2370 - categorical_accuracy: 0.9271
33728/60000 [===============>..............] - ETA: 42s - loss: 0.2366 - categorical_accuracy: 0.9272
33792/60000 [===============>..............] - ETA: 41s - loss: 0.2367 - categorical_accuracy: 0.9273
33856/60000 [===============>..............] - ETA: 41s - loss: 0.2365 - categorical_accuracy: 0.9273
33888/60000 [===============>..............] - ETA: 41s - loss: 0.2363 - categorical_accuracy: 0.9274
33952/60000 [===============>..............] - ETA: 41s - loss: 0.2360 - categorical_accuracy: 0.9275
33984/60000 [===============>..............] - ETA: 41s - loss: 0.2361 - categorical_accuracy: 0.9275
34016/60000 [================>.............] - ETA: 41s - loss: 0.2361 - categorical_accuracy: 0.9275
34080/60000 [================>.............] - ETA: 41s - loss: 0.2357 - categorical_accuracy: 0.9276
34112/60000 [================>.............] - ETA: 41s - loss: 0.2356 - categorical_accuracy: 0.9277
34144/60000 [================>.............] - ETA: 41s - loss: 0.2356 - categorical_accuracy: 0.9277
34208/60000 [================>.............] - ETA: 41s - loss: 0.2354 - categorical_accuracy: 0.9278
34272/60000 [================>.............] - ETA: 41s - loss: 0.2354 - categorical_accuracy: 0.9278
34336/60000 [================>.............] - ETA: 41s - loss: 0.2354 - categorical_accuracy: 0.9278
34400/60000 [================>.............] - ETA: 40s - loss: 0.2351 - categorical_accuracy: 0.9279
34464/60000 [================>.............] - ETA: 40s - loss: 0.2350 - categorical_accuracy: 0.9279
34528/60000 [================>.............] - ETA: 40s - loss: 0.2348 - categorical_accuracy: 0.9280
34592/60000 [================>.............] - ETA: 40s - loss: 0.2345 - categorical_accuracy: 0.9281
34624/60000 [================>.............] - ETA: 40s - loss: 0.2343 - categorical_accuracy: 0.9281
34656/60000 [================>.............] - ETA: 40s - loss: 0.2342 - categorical_accuracy: 0.9282
34688/60000 [================>.............] - ETA: 40s - loss: 0.2341 - categorical_accuracy: 0.9282
34720/60000 [================>.............] - ETA: 40s - loss: 0.2339 - categorical_accuracy: 0.9283
34752/60000 [================>.............] - ETA: 40s - loss: 0.2337 - categorical_accuracy: 0.9283
34816/60000 [================>.............] - ETA: 40s - loss: 0.2333 - categorical_accuracy: 0.9285
34848/60000 [================>.............] - ETA: 40s - loss: 0.2331 - categorical_accuracy: 0.9285
34912/60000 [================>.............] - ETA: 40s - loss: 0.2330 - categorical_accuracy: 0.9285
34944/60000 [================>.............] - ETA: 40s - loss: 0.2328 - categorical_accuracy: 0.9286
34976/60000 [================>.............] - ETA: 40s - loss: 0.2327 - categorical_accuracy: 0.9286
35008/60000 [================>.............] - ETA: 39s - loss: 0.2327 - categorical_accuracy: 0.9286
35040/60000 [================>.............] - ETA: 39s - loss: 0.2325 - categorical_accuracy: 0.9287
35104/60000 [================>.............] - ETA: 39s - loss: 0.2323 - categorical_accuracy: 0.9288
35168/60000 [================>.............] - ETA: 39s - loss: 0.2320 - categorical_accuracy: 0.9289
35232/60000 [================>.............] - ETA: 39s - loss: 0.2318 - categorical_accuracy: 0.9290
35264/60000 [================>.............] - ETA: 39s - loss: 0.2316 - categorical_accuracy: 0.9290
35328/60000 [================>.............] - ETA: 39s - loss: 0.2315 - categorical_accuracy: 0.9290
35392/60000 [================>.............] - ETA: 39s - loss: 0.2313 - categorical_accuracy: 0.9290
35424/60000 [================>.............] - ETA: 39s - loss: 0.2311 - categorical_accuracy: 0.9291
35456/60000 [================>.............] - ETA: 39s - loss: 0.2311 - categorical_accuracy: 0.9291
35488/60000 [================>.............] - ETA: 39s - loss: 0.2311 - categorical_accuracy: 0.9291
35552/60000 [================>.............] - ETA: 39s - loss: 0.2307 - categorical_accuracy: 0.9292
35616/60000 [================>.............] - ETA: 38s - loss: 0.2305 - categorical_accuracy: 0.9293
35680/60000 [================>.............] - ETA: 38s - loss: 0.2303 - categorical_accuracy: 0.9293
35744/60000 [================>.............] - ETA: 38s - loss: 0.2301 - categorical_accuracy: 0.9294
35808/60000 [================>.............] - ETA: 38s - loss: 0.2297 - categorical_accuracy: 0.9295
35872/60000 [================>.............] - ETA: 38s - loss: 0.2295 - categorical_accuracy: 0.9296
35936/60000 [================>.............] - ETA: 38s - loss: 0.2292 - categorical_accuracy: 0.9297
36000/60000 [=================>............] - ETA: 38s - loss: 0.2294 - categorical_accuracy: 0.9297
36032/60000 [=================>............] - ETA: 38s - loss: 0.2294 - categorical_accuracy: 0.9296
36064/60000 [=================>............] - ETA: 38s - loss: 0.2294 - categorical_accuracy: 0.9297
36096/60000 [=================>............] - ETA: 38s - loss: 0.2294 - categorical_accuracy: 0.9297
36128/60000 [=================>............] - ETA: 38s - loss: 0.2294 - categorical_accuracy: 0.9297
36160/60000 [=================>............] - ETA: 38s - loss: 0.2295 - categorical_accuracy: 0.9297
36192/60000 [=================>............] - ETA: 38s - loss: 0.2294 - categorical_accuracy: 0.9298
36224/60000 [=================>............] - ETA: 37s - loss: 0.2292 - categorical_accuracy: 0.9298
36256/60000 [=================>............] - ETA: 37s - loss: 0.2291 - categorical_accuracy: 0.9299
36320/60000 [=================>............] - ETA: 37s - loss: 0.2288 - categorical_accuracy: 0.9300
36352/60000 [=================>............] - ETA: 37s - loss: 0.2286 - categorical_accuracy: 0.9300
36416/60000 [=================>............] - ETA: 37s - loss: 0.2283 - categorical_accuracy: 0.9301
36448/60000 [=================>............] - ETA: 37s - loss: 0.2283 - categorical_accuracy: 0.9301
36480/60000 [=================>............] - ETA: 37s - loss: 0.2282 - categorical_accuracy: 0.9302
36544/60000 [=================>............] - ETA: 37s - loss: 0.2279 - categorical_accuracy: 0.9302
36608/60000 [=================>............] - ETA: 37s - loss: 0.2276 - categorical_accuracy: 0.9303
36640/60000 [=================>............] - ETA: 37s - loss: 0.2276 - categorical_accuracy: 0.9303
36672/60000 [=================>............] - ETA: 37s - loss: 0.2275 - categorical_accuracy: 0.9303
36704/60000 [=================>............] - ETA: 37s - loss: 0.2274 - categorical_accuracy: 0.9304
36736/60000 [=================>............] - ETA: 37s - loss: 0.2273 - categorical_accuracy: 0.9304
36768/60000 [=================>............] - ETA: 37s - loss: 0.2273 - categorical_accuracy: 0.9304
36832/60000 [=================>............] - ETA: 37s - loss: 0.2274 - categorical_accuracy: 0.9303
36864/60000 [=================>............] - ETA: 36s - loss: 0.2273 - categorical_accuracy: 0.9303
36896/60000 [=================>............] - ETA: 36s - loss: 0.2272 - categorical_accuracy: 0.9304
36960/60000 [=================>............] - ETA: 36s - loss: 0.2269 - categorical_accuracy: 0.9305
36992/60000 [=================>............] - ETA: 36s - loss: 0.2268 - categorical_accuracy: 0.9305
37024/60000 [=================>............] - ETA: 36s - loss: 0.2266 - categorical_accuracy: 0.9306
37056/60000 [=================>............] - ETA: 36s - loss: 0.2266 - categorical_accuracy: 0.9306
37088/60000 [=================>............] - ETA: 36s - loss: 0.2266 - categorical_accuracy: 0.9306
37120/60000 [=================>............] - ETA: 36s - loss: 0.2265 - categorical_accuracy: 0.9306
37184/60000 [=================>............] - ETA: 36s - loss: 0.2263 - categorical_accuracy: 0.9307
37248/60000 [=================>............] - ETA: 36s - loss: 0.2260 - categorical_accuracy: 0.9308
37312/60000 [=================>............] - ETA: 36s - loss: 0.2257 - categorical_accuracy: 0.9309
37376/60000 [=================>............] - ETA: 36s - loss: 0.2253 - categorical_accuracy: 0.9310
37408/60000 [=================>............] - ETA: 36s - loss: 0.2252 - categorical_accuracy: 0.9311
37472/60000 [=================>............] - ETA: 36s - loss: 0.2251 - categorical_accuracy: 0.9311
37504/60000 [=================>............] - ETA: 35s - loss: 0.2250 - categorical_accuracy: 0.9311
37568/60000 [=================>............] - ETA: 35s - loss: 0.2246 - categorical_accuracy: 0.9312
37600/60000 [=================>............] - ETA: 35s - loss: 0.2246 - categorical_accuracy: 0.9312
37632/60000 [=================>............] - ETA: 35s - loss: 0.2245 - categorical_accuracy: 0.9313
37664/60000 [=================>............] - ETA: 35s - loss: 0.2243 - categorical_accuracy: 0.9313
37728/60000 [=================>............] - ETA: 35s - loss: 0.2243 - categorical_accuracy: 0.9314
37760/60000 [=================>............] - ETA: 35s - loss: 0.2241 - categorical_accuracy: 0.9314
37792/60000 [=================>............] - ETA: 35s - loss: 0.2242 - categorical_accuracy: 0.9314
37824/60000 [=================>............] - ETA: 35s - loss: 0.2241 - categorical_accuracy: 0.9314
37888/60000 [=================>............] - ETA: 35s - loss: 0.2240 - categorical_accuracy: 0.9315
37952/60000 [=================>............] - ETA: 35s - loss: 0.2238 - categorical_accuracy: 0.9315
38016/60000 [==================>...........] - ETA: 35s - loss: 0.2239 - categorical_accuracy: 0.9314
38048/60000 [==================>...........] - ETA: 35s - loss: 0.2237 - categorical_accuracy: 0.9315
38080/60000 [==================>...........] - ETA: 35s - loss: 0.2236 - categorical_accuracy: 0.9315
38112/60000 [==================>...........] - ETA: 34s - loss: 0.2235 - categorical_accuracy: 0.9315
38144/60000 [==================>...........] - ETA: 34s - loss: 0.2234 - categorical_accuracy: 0.9315
38176/60000 [==================>...........] - ETA: 34s - loss: 0.2233 - categorical_accuracy: 0.9316
38240/60000 [==================>...........] - ETA: 34s - loss: 0.2230 - categorical_accuracy: 0.9317
38272/60000 [==================>...........] - ETA: 34s - loss: 0.2231 - categorical_accuracy: 0.9317
38304/60000 [==================>...........] - ETA: 34s - loss: 0.2229 - categorical_accuracy: 0.9318
38368/60000 [==================>...........] - ETA: 34s - loss: 0.2227 - categorical_accuracy: 0.9318
38432/60000 [==================>...........] - ETA: 34s - loss: 0.2227 - categorical_accuracy: 0.9319
38464/60000 [==================>...........] - ETA: 34s - loss: 0.2225 - categorical_accuracy: 0.9319
38496/60000 [==================>...........] - ETA: 34s - loss: 0.2225 - categorical_accuracy: 0.9319
38560/60000 [==================>...........] - ETA: 34s - loss: 0.2223 - categorical_accuracy: 0.9320
38624/60000 [==================>...........] - ETA: 34s - loss: 0.2220 - categorical_accuracy: 0.9321
38656/60000 [==================>...........] - ETA: 34s - loss: 0.2218 - categorical_accuracy: 0.9321
38688/60000 [==================>...........] - ETA: 34s - loss: 0.2219 - categorical_accuracy: 0.9321
38752/60000 [==================>...........] - ETA: 33s - loss: 0.2216 - categorical_accuracy: 0.9322
38784/60000 [==================>...........] - ETA: 33s - loss: 0.2217 - categorical_accuracy: 0.9322
38848/60000 [==================>...........] - ETA: 33s - loss: 0.2215 - categorical_accuracy: 0.9323
38912/60000 [==================>...........] - ETA: 33s - loss: 0.2212 - categorical_accuracy: 0.9324
38976/60000 [==================>...........] - ETA: 33s - loss: 0.2210 - categorical_accuracy: 0.9324
39008/60000 [==================>...........] - ETA: 33s - loss: 0.2209 - categorical_accuracy: 0.9325
39072/60000 [==================>...........] - ETA: 33s - loss: 0.2206 - categorical_accuracy: 0.9326
39104/60000 [==================>...........] - ETA: 33s - loss: 0.2205 - categorical_accuracy: 0.9326
39136/60000 [==================>...........] - ETA: 33s - loss: 0.2204 - categorical_accuracy: 0.9326
39168/60000 [==================>...........] - ETA: 33s - loss: 0.2203 - categorical_accuracy: 0.9327
39200/60000 [==================>...........] - ETA: 33s - loss: 0.2202 - categorical_accuracy: 0.9327
39264/60000 [==================>...........] - ETA: 33s - loss: 0.2200 - categorical_accuracy: 0.9328
39328/60000 [==================>...........] - ETA: 33s - loss: 0.2197 - categorical_accuracy: 0.9329
39360/60000 [==================>...........] - ETA: 32s - loss: 0.2201 - categorical_accuracy: 0.9329
39392/60000 [==================>...........] - ETA: 32s - loss: 0.2199 - categorical_accuracy: 0.9329
39424/60000 [==================>...........] - ETA: 32s - loss: 0.2198 - categorical_accuracy: 0.9330
39488/60000 [==================>...........] - ETA: 32s - loss: 0.2195 - categorical_accuracy: 0.9331
39552/60000 [==================>...........] - ETA: 32s - loss: 0.2192 - categorical_accuracy: 0.9332
39584/60000 [==================>...........] - ETA: 32s - loss: 0.2191 - categorical_accuracy: 0.9332
39616/60000 [==================>...........] - ETA: 32s - loss: 0.2191 - categorical_accuracy: 0.9332
39648/60000 [==================>...........] - ETA: 32s - loss: 0.2189 - categorical_accuracy: 0.9332
39712/60000 [==================>...........] - ETA: 32s - loss: 0.2190 - categorical_accuracy: 0.9332
39744/60000 [==================>...........] - ETA: 32s - loss: 0.2189 - categorical_accuracy: 0.9333
39808/60000 [==================>...........] - ETA: 32s - loss: 0.2187 - categorical_accuracy: 0.9333
39872/60000 [==================>...........] - ETA: 32s - loss: 0.2186 - categorical_accuracy: 0.9333
39904/60000 [==================>...........] - ETA: 32s - loss: 0.2185 - categorical_accuracy: 0.9334
39936/60000 [==================>...........] - ETA: 32s - loss: 0.2183 - categorical_accuracy: 0.9334
39968/60000 [==================>...........] - ETA: 32s - loss: 0.2182 - categorical_accuracy: 0.9334
40000/60000 [===================>..........] - ETA: 31s - loss: 0.2181 - categorical_accuracy: 0.9334
40032/60000 [===================>..........] - ETA: 31s - loss: 0.2180 - categorical_accuracy: 0.9335
40064/60000 [===================>..........] - ETA: 31s - loss: 0.2178 - categorical_accuracy: 0.9336
40096/60000 [===================>..........] - ETA: 31s - loss: 0.2177 - categorical_accuracy: 0.9336
40128/60000 [===================>..........] - ETA: 31s - loss: 0.2176 - categorical_accuracy: 0.9336
40160/60000 [===================>..........] - ETA: 31s - loss: 0.2175 - categorical_accuracy: 0.9336
40192/60000 [===================>..........] - ETA: 31s - loss: 0.2174 - categorical_accuracy: 0.9336
40224/60000 [===================>..........] - ETA: 31s - loss: 0.2172 - categorical_accuracy: 0.9337
40256/60000 [===================>..........] - ETA: 31s - loss: 0.2172 - categorical_accuracy: 0.9337
40320/60000 [===================>..........] - ETA: 31s - loss: 0.2170 - categorical_accuracy: 0.9338
40384/60000 [===================>..........] - ETA: 31s - loss: 0.2168 - categorical_accuracy: 0.9339
40416/60000 [===================>..........] - ETA: 31s - loss: 0.2167 - categorical_accuracy: 0.9339
40480/60000 [===================>..........] - ETA: 31s - loss: 0.2165 - categorical_accuracy: 0.9340
40544/60000 [===================>..........] - ETA: 31s - loss: 0.2162 - categorical_accuracy: 0.9341
40576/60000 [===================>..........] - ETA: 31s - loss: 0.2161 - categorical_accuracy: 0.9341
40608/60000 [===================>..........] - ETA: 31s - loss: 0.2160 - categorical_accuracy: 0.9341
40672/60000 [===================>..........] - ETA: 30s - loss: 0.2158 - categorical_accuracy: 0.9341
40736/60000 [===================>..........] - ETA: 30s - loss: 0.2155 - categorical_accuracy: 0.9342
40768/60000 [===================>..........] - ETA: 30s - loss: 0.2154 - categorical_accuracy: 0.9343
40832/60000 [===================>..........] - ETA: 30s - loss: 0.2153 - categorical_accuracy: 0.9343
40864/60000 [===================>..........] - ETA: 30s - loss: 0.2152 - categorical_accuracy: 0.9343
40896/60000 [===================>..........] - ETA: 30s - loss: 0.2151 - categorical_accuracy: 0.9344
40960/60000 [===================>..........] - ETA: 30s - loss: 0.2151 - categorical_accuracy: 0.9344
41024/60000 [===================>..........] - ETA: 30s - loss: 0.2149 - categorical_accuracy: 0.9345
41088/60000 [===================>..........] - ETA: 30s - loss: 0.2148 - categorical_accuracy: 0.9345
41152/60000 [===================>..........] - ETA: 30s - loss: 0.2145 - categorical_accuracy: 0.9346
41184/60000 [===================>..........] - ETA: 30s - loss: 0.2147 - categorical_accuracy: 0.9345
41216/60000 [===================>..........] - ETA: 30s - loss: 0.2146 - categorical_accuracy: 0.9346
41248/60000 [===================>..........] - ETA: 29s - loss: 0.2144 - categorical_accuracy: 0.9346
41280/60000 [===================>..........] - ETA: 29s - loss: 0.2143 - categorical_accuracy: 0.9346
41344/60000 [===================>..........] - ETA: 29s - loss: 0.2143 - categorical_accuracy: 0.9347
41408/60000 [===================>..........] - ETA: 29s - loss: 0.2140 - categorical_accuracy: 0.9347
41472/60000 [===================>..........] - ETA: 29s - loss: 0.2137 - categorical_accuracy: 0.9348
41504/60000 [===================>..........] - ETA: 29s - loss: 0.2137 - categorical_accuracy: 0.9348
41536/60000 [===================>..........] - ETA: 29s - loss: 0.2136 - categorical_accuracy: 0.9348
41600/60000 [===================>..........] - ETA: 29s - loss: 0.2134 - categorical_accuracy: 0.9349
41664/60000 [===================>..........] - ETA: 29s - loss: 0.2132 - categorical_accuracy: 0.9350
41728/60000 [===================>..........] - ETA: 29s - loss: 0.2132 - categorical_accuracy: 0.9350
41792/60000 [===================>..........] - ETA: 29s - loss: 0.2129 - categorical_accuracy: 0.9351
41824/60000 [===================>..........] - ETA: 29s - loss: 0.2128 - categorical_accuracy: 0.9352
41856/60000 [===================>..........] - ETA: 28s - loss: 0.2126 - categorical_accuracy: 0.9352
41920/60000 [===================>..........] - ETA: 28s - loss: 0.2123 - categorical_accuracy: 0.9353
41952/60000 [===================>..........] - ETA: 28s - loss: 0.2122 - categorical_accuracy: 0.9354
41984/60000 [===================>..........] - ETA: 28s - loss: 0.2120 - categorical_accuracy: 0.9354
42016/60000 [====================>.........] - ETA: 28s - loss: 0.2119 - categorical_accuracy: 0.9355
42080/60000 [====================>.........] - ETA: 28s - loss: 0.2116 - categorical_accuracy: 0.9355
42112/60000 [====================>.........] - ETA: 28s - loss: 0.2115 - categorical_accuracy: 0.9356
42144/60000 [====================>.........] - ETA: 28s - loss: 0.2117 - categorical_accuracy: 0.9356
42208/60000 [====================>.........] - ETA: 28s - loss: 0.2115 - categorical_accuracy: 0.9356
42240/60000 [====================>.........] - ETA: 28s - loss: 0.2116 - categorical_accuracy: 0.9356
42272/60000 [====================>.........] - ETA: 28s - loss: 0.2115 - categorical_accuracy: 0.9356
42304/60000 [====================>.........] - ETA: 28s - loss: 0.2115 - categorical_accuracy: 0.9356
42336/60000 [====================>.........] - ETA: 28s - loss: 0.2115 - categorical_accuracy: 0.9356
42400/60000 [====================>.........] - ETA: 28s - loss: 0.2112 - categorical_accuracy: 0.9357
42432/60000 [====================>.........] - ETA: 28s - loss: 0.2111 - categorical_accuracy: 0.9357
42464/60000 [====================>.........] - ETA: 28s - loss: 0.2109 - categorical_accuracy: 0.9358
42496/60000 [====================>.........] - ETA: 27s - loss: 0.2110 - categorical_accuracy: 0.9358
42560/60000 [====================>.........] - ETA: 27s - loss: 0.2108 - categorical_accuracy: 0.9357
42592/60000 [====================>.........] - ETA: 27s - loss: 0.2108 - categorical_accuracy: 0.9357
42656/60000 [====================>.........] - ETA: 27s - loss: 0.2106 - categorical_accuracy: 0.9358
42688/60000 [====================>.........] - ETA: 27s - loss: 0.2106 - categorical_accuracy: 0.9358
42720/60000 [====================>.........] - ETA: 27s - loss: 0.2105 - categorical_accuracy: 0.9359
42752/60000 [====================>.........] - ETA: 27s - loss: 0.2104 - categorical_accuracy: 0.9359
42816/60000 [====================>.........] - ETA: 27s - loss: 0.2101 - categorical_accuracy: 0.9360
42880/60000 [====================>.........] - ETA: 27s - loss: 0.2099 - categorical_accuracy: 0.9361
42944/60000 [====================>.........] - ETA: 27s - loss: 0.2096 - categorical_accuracy: 0.9361
43008/60000 [====================>.........] - ETA: 27s - loss: 0.2096 - categorical_accuracy: 0.9362
43072/60000 [====================>.........] - ETA: 27s - loss: 0.2094 - categorical_accuracy: 0.9362
43136/60000 [====================>.........] - ETA: 26s - loss: 0.2093 - categorical_accuracy: 0.9362
43168/60000 [====================>.........] - ETA: 26s - loss: 0.2092 - categorical_accuracy: 0.9362
43200/60000 [====================>.........] - ETA: 26s - loss: 0.2091 - categorical_accuracy: 0.9363
43232/60000 [====================>.........] - ETA: 26s - loss: 0.2090 - categorical_accuracy: 0.9363
43264/60000 [====================>.........] - ETA: 26s - loss: 0.2089 - categorical_accuracy: 0.9363
43296/60000 [====================>.........] - ETA: 26s - loss: 0.2089 - categorical_accuracy: 0.9363
43360/60000 [====================>.........] - ETA: 26s - loss: 0.2088 - categorical_accuracy: 0.9363
43392/60000 [====================>.........] - ETA: 26s - loss: 0.2092 - categorical_accuracy: 0.9363
43424/60000 [====================>.........] - ETA: 26s - loss: 0.2091 - categorical_accuracy: 0.9364
43488/60000 [====================>.........] - ETA: 26s - loss: 0.2089 - categorical_accuracy: 0.9364
43552/60000 [====================>.........] - ETA: 26s - loss: 0.2087 - categorical_accuracy: 0.9365
43616/60000 [====================>.........] - ETA: 26s - loss: 0.2086 - categorical_accuracy: 0.9365
43680/60000 [====================>.........] - ETA: 26s - loss: 0.2085 - categorical_accuracy: 0.9365
43712/60000 [====================>.........] - ETA: 26s - loss: 0.2085 - categorical_accuracy: 0.9365
43744/60000 [====================>.........] - ETA: 25s - loss: 0.2084 - categorical_accuracy: 0.9366
43808/60000 [====================>.........] - ETA: 25s - loss: 0.2083 - categorical_accuracy: 0.9366
43840/60000 [====================>.........] - ETA: 25s - loss: 0.2081 - categorical_accuracy: 0.9367
43872/60000 [====================>.........] - ETA: 25s - loss: 0.2082 - categorical_accuracy: 0.9366
43904/60000 [====================>.........] - ETA: 25s - loss: 0.2080 - categorical_accuracy: 0.9367
43968/60000 [====================>.........] - ETA: 25s - loss: 0.2080 - categorical_accuracy: 0.9367
44032/60000 [=====================>........] - ETA: 25s - loss: 0.2078 - categorical_accuracy: 0.9368
44096/60000 [=====================>........] - ETA: 25s - loss: 0.2076 - categorical_accuracy: 0.9368
44160/60000 [=====================>........] - ETA: 25s - loss: 0.2078 - categorical_accuracy: 0.9368
44192/60000 [=====================>........] - ETA: 25s - loss: 0.2077 - categorical_accuracy: 0.9368
44224/60000 [=====================>........] - ETA: 25s - loss: 0.2077 - categorical_accuracy: 0.9368
44256/60000 [=====================>........] - ETA: 25s - loss: 0.2078 - categorical_accuracy: 0.9368
44288/60000 [=====================>........] - ETA: 25s - loss: 0.2077 - categorical_accuracy: 0.9368
44320/60000 [=====================>........] - ETA: 25s - loss: 0.2076 - categorical_accuracy: 0.9368
44352/60000 [=====================>........] - ETA: 25s - loss: 0.2075 - categorical_accuracy: 0.9368
44416/60000 [=====================>........] - ETA: 24s - loss: 0.2074 - categorical_accuracy: 0.9368
44480/60000 [=====================>........] - ETA: 24s - loss: 0.2072 - categorical_accuracy: 0.9369
44544/60000 [=====================>........] - ETA: 24s - loss: 0.2069 - categorical_accuracy: 0.9370
44608/60000 [=====================>........] - ETA: 24s - loss: 0.2069 - categorical_accuracy: 0.9370
44672/60000 [=====================>........] - ETA: 24s - loss: 0.2067 - categorical_accuracy: 0.9371
44736/60000 [=====================>........] - ETA: 24s - loss: 0.2068 - categorical_accuracy: 0.9371
44800/60000 [=====================>........] - ETA: 24s - loss: 0.2065 - categorical_accuracy: 0.9372
44864/60000 [=====================>........] - ETA: 24s - loss: 0.2065 - categorical_accuracy: 0.9372
44928/60000 [=====================>........] - ETA: 24s - loss: 0.2064 - categorical_accuracy: 0.9372
44992/60000 [=====================>........] - ETA: 23s - loss: 0.2063 - categorical_accuracy: 0.9373
45056/60000 [=====================>........] - ETA: 23s - loss: 0.2061 - categorical_accuracy: 0.9373
45120/60000 [=====================>........] - ETA: 23s - loss: 0.2060 - categorical_accuracy: 0.9374
45152/60000 [=====================>........] - ETA: 23s - loss: 0.2059 - categorical_accuracy: 0.9374
45184/60000 [=====================>........] - ETA: 23s - loss: 0.2057 - categorical_accuracy: 0.9375
45216/60000 [=====================>........] - ETA: 23s - loss: 0.2056 - categorical_accuracy: 0.9375
45280/60000 [=====================>........] - ETA: 23s - loss: 0.2056 - categorical_accuracy: 0.9376
45344/60000 [=====================>........] - ETA: 23s - loss: 0.2056 - categorical_accuracy: 0.9376
45376/60000 [=====================>........] - ETA: 23s - loss: 0.2055 - categorical_accuracy: 0.9376
45408/60000 [=====================>........] - ETA: 23s - loss: 0.2055 - categorical_accuracy: 0.9376
45440/60000 [=====================>........] - ETA: 23s - loss: 0.2054 - categorical_accuracy: 0.9377
45504/60000 [=====================>........] - ETA: 23s - loss: 0.2053 - categorical_accuracy: 0.9377
45568/60000 [=====================>........] - ETA: 23s - loss: 0.2052 - categorical_accuracy: 0.9377
45632/60000 [=====================>........] - ETA: 22s - loss: 0.2050 - categorical_accuracy: 0.9378
45664/60000 [=====================>........] - ETA: 22s - loss: 0.2050 - categorical_accuracy: 0.9378
45696/60000 [=====================>........] - ETA: 22s - loss: 0.2049 - categorical_accuracy: 0.9378
45760/60000 [=====================>........] - ETA: 22s - loss: 0.2048 - categorical_accuracy: 0.9378
45824/60000 [=====================>........] - ETA: 22s - loss: 0.2048 - categorical_accuracy: 0.9378
45856/60000 [=====================>........] - ETA: 22s - loss: 0.2046 - categorical_accuracy: 0.9379
45920/60000 [=====================>........] - ETA: 22s - loss: 0.2044 - categorical_accuracy: 0.9379
45952/60000 [=====================>........] - ETA: 22s - loss: 0.2044 - categorical_accuracy: 0.9380
46016/60000 [======================>.......] - ETA: 22s - loss: 0.2042 - categorical_accuracy: 0.9380
46080/60000 [======================>.......] - ETA: 22s - loss: 0.2042 - categorical_accuracy: 0.9380
46112/60000 [======================>.......] - ETA: 22s - loss: 0.2041 - categorical_accuracy: 0.9380
46176/60000 [======================>.......] - ETA: 22s - loss: 0.2040 - categorical_accuracy: 0.9381
46208/60000 [======================>.......] - ETA: 22s - loss: 0.2038 - categorical_accuracy: 0.9381
46240/60000 [======================>.......] - ETA: 21s - loss: 0.2038 - categorical_accuracy: 0.9382
46304/60000 [======================>.......] - ETA: 21s - loss: 0.2035 - categorical_accuracy: 0.9383
46368/60000 [======================>.......] - ETA: 21s - loss: 0.2035 - categorical_accuracy: 0.9383
46400/60000 [======================>.......] - ETA: 21s - loss: 0.2034 - categorical_accuracy: 0.9383
46432/60000 [======================>.......] - ETA: 21s - loss: 0.2034 - categorical_accuracy: 0.9383
46464/60000 [======================>.......] - ETA: 21s - loss: 0.2033 - categorical_accuracy: 0.9383
46496/60000 [======================>.......] - ETA: 21s - loss: 0.2034 - categorical_accuracy: 0.9383
46560/60000 [======================>.......] - ETA: 21s - loss: 0.2033 - categorical_accuracy: 0.9384
46592/60000 [======================>.......] - ETA: 21s - loss: 0.2031 - categorical_accuracy: 0.9384
46656/60000 [======================>.......] - ETA: 21s - loss: 0.2030 - categorical_accuracy: 0.9384
46720/60000 [======================>.......] - ETA: 21s - loss: 0.2029 - categorical_accuracy: 0.9385
46784/60000 [======================>.......] - ETA: 21s - loss: 0.2027 - categorical_accuracy: 0.9385
46848/60000 [======================>.......] - ETA: 20s - loss: 0.2025 - categorical_accuracy: 0.9386
46912/60000 [======================>.......] - ETA: 20s - loss: 0.2023 - categorical_accuracy: 0.9387
46944/60000 [======================>.......] - ETA: 20s - loss: 0.2023 - categorical_accuracy: 0.9387
46976/60000 [======================>.......] - ETA: 20s - loss: 0.2022 - categorical_accuracy: 0.9387
47040/60000 [======================>.......] - ETA: 20s - loss: 0.2021 - categorical_accuracy: 0.9388
47104/60000 [======================>.......] - ETA: 20s - loss: 0.2019 - categorical_accuracy: 0.9388
47168/60000 [======================>.......] - ETA: 20s - loss: 0.2017 - categorical_accuracy: 0.9389
47232/60000 [======================>.......] - ETA: 20s - loss: 0.2015 - categorical_accuracy: 0.9389
47264/60000 [======================>.......] - ETA: 20s - loss: 0.2014 - categorical_accuracy: 0.9390
47296/60000 [======================>.......] - ETA: 20s - loss: 0.2014 - categorical_accuracy: 0.9390
47328/60000 [======================>.......] - ETA: 20s - loss: 0.2013 - categorical_accuracy: 0.9390
47392/60000 [======================>.......] - ETA: 20s - loss: 0.2011 - categorical_accuracy: 0.9390
47456/60000 [======================>.......] - ETA: 20s - loss: 0.2011 - categorical_accuracy: 0.9391
47488/60000 [======================>.......] - ETA: 19s - loss: 0.2011 - categorical_accuracy: 0.9390
47552/60000 [======================>.......] - ETA: 19s - loss: 0.2010 - categorical_accuracy: 0.9391
47584/60000 [======================>.......] - ETA: 19s - loss: 0.2009 - categorical_accuracy: 0.9391
47616/60000 [======================>.......] - ETA: 19s - loss: 0.2009 - categorical_accuracy: 0.9391
47680/60000 [======================>.......] - ETA: 19s - loss: 0.2007 - categorical_accuracy: 0.9391
47744/60000 [======================>.......] - ETA: 19s - loss: 0.2008 - categorical_accuracy: 0.9391
47776/60000 [======================>.......] - ETA: 19s - loss: 0.2007 - categorical_accuracy: 0.9392
47808/60000 [======================>.......] - ETA: 19s - loss: 0.2006 - categorical_accuracy: 0.9392
47872/60000 [======================>.......] - ETA: 19s - loss: 0.2004 - categorical_accuracy: 0.9392
47904/60000 [======================>.......] - ETA: 19s - loss: 0.2003 - categorical_accuracy: 0.9392
47936/60000 [======================>.......] - ETA: 19s - loss: 0.2004 - categorical_accuracy: 0.9392
47968/60000 [======================>.......] - ETA: 19s - loss: 0.2003 - categorical_accuracy: 0.9393
48032/60000 [=======================>......] - ETA: 19s - loss: 0.2003 - categorical_accuracy: 0.9392
48096/60000 [=======================>......] - ETA: 18s - loss: 0.2004 - categorical_accuracy: 0.9392
48160/60000 [=======================>......] - ETA: 18s - loss: 0.2004 - categorical_accuracy: 0.9392
48224/60000 [=======================>......] - ETA: 18s - loss: 0.2003 - categorical_accuracy: 0.9393
48256/60000 [=======================>......] - ETA: 18s - loss: 0.2002 - categorical_accuracy: 0.9393
48288/60000 [=======================>......] - ETA: 18s - loss: 0.2002 - categorical_accuracy: 0.9393
48352/60000 [=======================>......] - ETA: 18s - loss: 0.2001 - categorical_accuracy: 0.9393
48384/60000 [=======================>......] - ETA: 18s - loss: 0.2001 - categorical_accuracy: 0.9393
48416/60000 [=======================>......] - ETA: 18s - loss: 0.2000 - categorical_accuracy: 0.9394
48448/60000 [=======================>......] - ETA: 18s - loss: 0.2000 - categorical_accuracy: 0.9394
48480/60000 [=======================>......] - ETA: 18s - loss: 0.1999 - categorical_accuracy: 0.9394
48512/60000 [=======================>......] - ETA: 18s - loss: 0.1998 - categorical_accuracy: 0.9394
48544/60000 [=======================>......] - ETA: 18s - loss: 0.1997 - categorical_accuracy: 0.9395
48576/60000 [=======================>......] - ETA: 18s - loss: 0.1995 - categorical_accuracy: 0.9395
48640/60000 [=======================>......] - ETA: 18s - loss: 0.1993 - categorical_accuracy: 0.9396
48704/60000 [=======================>......] - ETA: 18s - loss: 0.1992 - categorical_accuracy: 0.9396
48768/60000 [=======================>......] - ETA: 17s - loss: 0.1990 - categorical_accuracy: 0.9397
48800/60000 [=======================>......] - ETA: 17s - loss: 0.1989 - categorical_accuracy: 0.9397
48832/60000 [=======================>......] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9398
48864/60000 [=======================>......] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9398
48896/60000 [=======================>......] - ETA: 17s - loss: 0.1987 - categorical_accuracy: 0.9398
48960/60000 [=======================>......] - ETA: 17s - loss: 0.1986 - categorical_accuracy: 0.9398
49024/60000 [=======================>......] - ETA: 17s - loss: 0.1984 - categorical_accuracy: 0.9399
49056/60000 [=======================>......] - ETA: 17s - loss: 0.1984 - categorical_accuracy: 0.9398
49120/60000 [=======================>......] - ETA: 17s - loss: 0.1982 - categorical_accuracy: 0.9399
49152/60000 [=======================>......] - ETA: 17s - loss: 0.1982 - categorical_accuracy: 0.9399
49216/60000 [=======================>......] - ETA: 17s - loss: 0.1980 - categorical_accuracy: 0.9400
49280/60000 [=======================>......] - ETA: 17s - loss: 0.1978 - categorical_accuracy: 0.9400
49344/60000 [=======================>......] - ETA: 16s - loss: 0.1977 - categorical_accuracy: 0.9401
49408/60000 [=======================>......] - ETA: 16s - loss: 0.1975 - categorical_accuracy: 0.9401
49472/60000 [=======================>......] - ETA: 16s - loss: 0.1975 - categorical_accuracy: 0.9401
49504/60000 [=======================>......] - ETA: 16s - loss: 0.1974 - categorical_accuracy: 0.9401
49536/60000 [=======================>......] - ETA: 16s - loss: 0.1975 - categorical_accuracy: 0.9402
49568/60000 [=======================>......] - ETA: 16s - loss: 0.1974 - categorical_accuracy: 0.9402
49600/60000 [=======================>......] - ETA: 16s - loss: 0.1973 - categorical_accuracy: 0.9402
49632/60000 [=======================>......] - ETA: 16s - loss: 0.1974 - categorical_accuracy: 0.9402
49664/60000 [=======================>......] - ETA: 16s - loss: 0.1974 - categorical_accuracy: 0.9402
49696/60000 [=======================>......] - ETA: 16s - loss: 0.1973 - categorical_accuracy: 0.9402
49728/60000 [=======================>......] - ETA: 16s - loss: 0.1972 - categorical_accuracy: 0.9403
49760/60000 [=======================>......] - ETA: 16s - loss: 0.1971 - categorical_accuracy: 0.9403
49824/60000 [=======================>......] - ETA: 16s - loss: 0.1971 - categorical_accuracy: 0.9403
49856/60000 [=======================>......] - ETA: 16s - loss: 0.1970 - categorical_accuracy: 0.9403
49920/60000 [=======================>......] - ETA: 16s - loss: 0.1969 - categorical_accuracy: 0.9404
49984/60000 [=======================>......] - ETA: 15s - loss: 0.1967 - categorical_accuracy: 0.9404
50048/60000 [========================>.....] - ETA: 15s - loss: 0.1965 - categorical_accuracy: 0.9405
50080/60000 [========================>.....] - ETA: 15s - loss: 0.1964 - categorical_accuracy: 0.9406
50144/60000 [========================>.....] - ETA: 15s - loss: 0.1964 - categorical_accuracy: 0.9406
50176/60000 [========================>.....] - ETA: 15s - loss: 0.1964 - categorical_accuracy: 0.9406
50208/60000 [========================>.....] - ETA: 15s - loss: 0.1963 - categorical_accuracy: 0.9406
50272/60000 [========================>.....] - ETA: 15s - loss: 0.1962 - categorical_accuracy: 0.9406
50304/60000 [========================>.....] - ETA: 15s - loss: 0.1962 - categorical_accuracy: 0.9406
50336/60000 [========================>.....] - ETA: 15s - loss: 0.1961 - categorical_accuracy: 0.9406
50400/60000 [========================>.....] - ETA: 15s - loss: 0.1961 - categorical_accuracy: 0.9406
50464/60000 [========================>.....] - ETA: 15s - loss: 0.1960 - categorical_accuracy: 0.9407
50496/60000 [========================>.....] - ETA: 15s - loss: 0.1959 - categorical_accuracy: 0.9407
50528/60000 [========================>.....] - ETA: 15s - loss: 0.1958 - categorical_accuracy: 0.9407
50560/60000 [========================>.....] - ETA: 15s - loss: 0.1957 - categorical_accuracy: 0.9407
50592/60000 [========================>.....] - ETA: 15s - loss: 0.1956 - categorical_accuracy: 0.9408
50624/60000 [========================>.....] - ETA: 14s - loss: 0.1956 - categorical_accuracy: 0.9408
50656/60000 [========================>.....] - ETA: 14s - loss: 0.1955 - categorical_accuracy: 0.9408
50720/60000 [========================>.....] - ETA: 14s - loss: 0.1953 - categorical_accuracy: 0.9409
50752/60000 [========================>.....] - ETA: 14s - loss: 0.1953 - categorical_accuracy: 0.9408
50784/60000 [========================>.....] - ETA: 14s - loss: 0.1953 - categorical_accuracy: 0.9409
50816/60000 [========================>.....] - ETA: 14s - loss: 0.1953 - categorical_accuracy: 0.9409
50880/60000 [========================>.....] - ETA: 14s - loss: 0.1952 - categorical_accuracy: 0.9409
50944/60000 [========================>.....] - ETA: 14s - loss: 0.1952 - categorical_accuracy: 0.9410
51008/60000 [========================>.....] - ETA: 14s - loss: 0.1949 - categorical_accuracy: 0.9410
51072/60000 [========================>.....] - ETA: 14s - loss: 0.1948 - categorical_accuracy: 0.9411
51104/60000 [========================>.....] - ETA: 14s - loss: 0.1949 - categorical_accuracy: 0.9411
51168/60000 [========================>.....] - ETA: 14s - loss: 0.1948 - categorical_accuracy: 0.9411
51232/60000 [========================>.....] - ETA: 13s - loss: 0.1947 - categorical_accuracy: 0.9411
51264/60000 [========================>.....] - ETA: 13s - loss: 0.1946 - categorical_accuracy: 0.9411
51296/60000 [========================>.....] - ETA: 13s - loss: 0.1945 - categorical_accuracy: 0.9411
51328/60000 [========================>.....] - ETA: 13s - loss: 0.1944 - categorical_accuracy: 0.9412
51392/60000 [========================>.....] - ETA: 13s - loss: 0.1942 - categorical_accuracy: 0.9412
51424/60000 [========================>.....] - ETA: 13s - loss: 0.1941 - categorical_accuracy: 0.9413
51456/60000 [========================>.....] - ETA: 13s - loss: 0.1940 - categorical_accuracy: 0.9413
51488/60000 [========================>.....] - ETA: 13s - loss: 0.1939 - categorical_accuracy: 0.9413
51520/60000 [========================>.....] - ETA: 13s - loss: 0.1940 - categorical_accuracy: 0.9413
51584/60000 [========================>.....] - ETA: 13s - loss: 0.1938 - categorical_accuracy: 0.9414
51616/60000 [========================>.....] - ETA: 13s - loss: 0.1938 - categorical_accuracy: 0.9414
51648/60000 [========================>.....] - ETA: 13s - loss: 0.1940 - categorical_accuracy: 0.9413
51680/60000 [========================>.....] - ETA: 13s - loss: 0.1939 - categorical_accuracy: 0.9414
51712/60000 [========================>.....] - ETA: 13s - loss: 0.1939 - categorical_accuracy: 0.9413
51744/60000 [========================>.....] - ETA: 13s - loss: 0.1938 - categorical_accuracy: 0.9414
51776/60000 [========================>.....] - ETA: 13s - loss: 0.1938 - categorical_accuracy: 0.9414
51808/60000 [========================>.....] - ETA: 13s - loss: 0.1937 - categorical_accuracy: 0.9414
51872/60000 [========================>.....] - ETA: 12s - loss: 0.1936 - categorical_accuracy: 0.9414
51936/60000 [========================>.....] - ETA: 12s - loss: 0.1934 - categorical_accuracy: 0.9415
51968/60000 [========================>.....] - ETA: 12s - loss: 0.1933 - categorical_accuracy: 0.9415
52000/60000 [=========================>....] - ETA: 12s - loss: 0.1932 - categorical_accuracy: 0.9416
52032/60000 [=========================>....] - ETA: 12s - loss: 0.1931 - categorical_accuracy: 0.9416
52064/60000 [=========================>....] - ETA: 12s - loss: 0.1930 - categorical_accuracy: 0.9416
52128/60000 [=========================>....] - ETA: 12s - loss: 0.1931 - categorical_accuracy: 0.9416
52192/60000 [=========================>....] - ETA: 12s - loss: 0.1929 - categorical_accuracy: 0.9417
52224/60000 [=========================>....] - ETA: 12s - loss: 0.1928 - categorical_accuracy: 0.9417
52288/60000 [=========================>....] - ETA: 12s - loss: 0.1928 - categorical_accuracy: 0.9417
52320/60000 [=========================>....] - ETA: 12s - loss: 0.1928 - categorical_accuracy: 0.9417
52384/60000 [=========================>....] - ETA: 12s - loss: 0.1926 - categorical_accuracy: 0.9418
52448/60000 [=========================>....] - ETA: 12s - loss: 0.1925 - categorical_accuracy: 0.9418
52512/60000 [=========================>....] - ETA: 11s - loss: 0.1924 - categorical_accuracy: 0.9419
52576/60000 [=========================>....] - ETA: 11s - loss: 0.1923 - categorical_accuracy: 0.9419
52608/60000 [=========================>....] - ETA: 11s - loss: 0.1922 - categorical_accuracy: 0.9419
52640/60000 [=========================>....] - ETA: 11s - loss: 0.1922 - categorical_accuracy: 0.9419
52672/60000 [=========================>....] - ETA: 11s - loss: 0.1921 - categorical_accuracy: 0.9419
52704/60000 [=========================>....] - ETA: 11s - loss: 0.1920 - categorical_accuracy: 0.9420
52768/60000 [=========================>....] - ETA: 11s - loss: 0.1918 - categorical_accuracy: 0.9420
52832/60000 [=========================>....] - ETA: 11s - loss: 0.1917 - categorical_accuracy: 0.9421
52896/60000 [=========================>....] - ETA: 11s - loss: 0.1917 - categorical_accuracy: 0.9421
52960/60000 [=========================>....] - ETA: 11s - loss: 0.1916 - categorical_accuracy: 0.9421
53024/60000 [=========================>....] - ETA: 11s - loss: 0.1915 - categorical_accuracy: 0.9421
53056/60000 [=========================>....] - ETA: 11s - loss: 0.1915 - categorical_accuracy: 0.9422
53088/60000 [=========================>....] - ETA: 11s - loss: 0.1915 - categorical_accuracy: 0.9422
53120/60000 [=========================>....] - ETA: 10s - loss: 0.1915 - categorical_accuracy: 0.9421
53152/60000 [=========================>....] - ETA: 10s - loss: 0.1914 - categorical_accuracy: 0.9422
53216/60000 [=========================>....] - ETA: 10s - loss: 0.1913 - categorical_accuracy: 0.9422
53280/60000 [=========================>....] - ETA: 10s - loss: 0.1912 - categorical_accuracy: 0.9422
53344/60000 [=========================>....] - ETA: 10s - loss: 0.1910 - categorical_accuracy: 0.9423
53376/60000 [=========================>....] - ETA: 10s - loss: 0.1910 - categorical_accuracy: 0.9423
53440/60000 [=========================>....] - ETA: 10s - loss: 0.1909 - categorical_accuracy: 0.9423
53472/60000 [=========================>....] - ETA: 10s - loss: 0.1908 - categorical_accuracy: 0.9423
53504/60000 [=========================>....] - ETA: 10s - loss: 0.1908 - categorical_accuracy: 0.9424
53536/60000 [=========================>....] - ETA: 10s - loss: 0.1907 - categorical_accuracy: 0.9424
53568/60000 [=========================>....] - ETA: 10s - loss: 0.1906 - categorical_accuracy: 0.9424
53600/60000 [=========================>....] - ETA: 10s - loss: 0.1905 - categorical_accuracy: 0.9424
53632/60000 [=========================>....] - ETA: 10s - loss: 0.1905 - categorical_accuracy: 0.9424
53664/60000 [=========================>....] - ETA: 10s - loss: 0.1904 - categorical_accuracy: 0.9425
53696/60000 [=========================>....] - ETA: 10s - loss: 0.1904 - categorical_accuracy: 0.9425
53728/60000 [=========================>....] - ETA: 9s - loss: 0.1904 - categorical_accuracy: 0.9425 
53760/60000 [=========================>....] - ETA: 9s - loss: 0.1904 - categorical_accuracy: 0.9425
53824/60000 [=========================>....] - ETA: 9s - loss: 0.1902 - categorical_accuracy: 0.9426
53856/60000 [=========================>....] - ETA: 9s - loss: 0.1901 - categorical_accuracy: 0.9426
53920/60000 [=========================>....] - ETA: 9s - loss: 0.1900 - categorical_accuracy: 0.9426
53952/60000 [=========================>....] - ETA: 9s - loss: 0.1899 - categorical_accuracy: 0.9426
53984/60000 [=========================>....] - ETA: 9s - loss: 0.1898 - categorical_accuracy: 0.9426
54016/60000 [==========================>...] - ETA: 9s - loss: 0.1897 - categorical_accuracy: 0.9427
54080/60000 [==========================>...] - ETA: 9s - loss: 0.1896 - categorical_accuracy: 0.9427
54144/60000 [==========================>...] - ETA: 9s - loss: 0.1896 - categorical_accuracy: 0.9427
54208/60000 [==========================>...] - ETA: 9s - loss: 0.1894 - categorical_accuracy: 0.9428
54240/60000 [==========================>...] - ETA: 9s - loss: 0.1893 - categorical_accuracy: 0.9428
54304/60000 [==========================>...] - ETA: 9s - loss: 0.1893 - categorical_accuracy: 0.9428
54336/60000 [==========================>...] - ETA: 9s - loss: 0.1893 - categorical_accuracy: 0.9428
54368/60000 [==========================>...] - ETA: 8s - loss: 0.1892 - categorical_accuracy: 0.9429
54400/60000 [==========================>...] - ETA: 8s - loss: 0.1891 - categorical_accuracy: 0.9429
54464/60000 [==========================>...] - ETA: 8s - loss: 0.1891 - categorical_accuracy: 0.9429
54528/60000 [==========================>...] - ETA: 8s - loss: 0.1890 - categorical_accuracy: 0.9429
54560/60000 [==========================>...] - ETA: 8s - loss: 0.1890 - categorical_accuracy: 0.9429
54592/60000 [==========================>...] - ETA: 8s - loss: 0.1890 - categorical_accuracy: 0.9429
54656/60000 [==========================>...] - ETA: 8s - loss: 0.1889 - categorical_accuracy: 0.9430
54720/60000 [==========================>...] - ETA: 8s - loss: 0.1887 - categorical_accuracy: 0.9430
54752/60000 [==========================>...] - ETA: 8s - loss: 0.1886 - categorical_accuracy: 0.9430
54784/60000 [==========================>...] - ETA: 8s - loss: 0.1886 - categorical_accuracy: 0.9430
54816/60000 [==========================>...] - ETA: 8s - loss: 0.1886 - categorical_accuracy: 0.9430
54880/60000 [==========================>...] - ETA: 8s - loss: 0.1884 - categorical_accuracy: 0.9431
54912/60000 [==========================>...] - ETA: 8s - loss: 0.1883 - categorical_accuracy: 0.9431
54944/60000 [==========================>...] - ETA: 8s - loss: 0.1883 - categorical_accuracy: 0.9431
54976/60000 [==========================>...] - ETA: 8s - loss: 0.1882 - categorical_accuracy: 0.9432
55008/60000 [==========================>...] - ETA: 7s - loss: 0.1882 - categorical_accuracy: 0.9432
55072/60000 [==========================>...] - ETA: 7s - loss: 0.1881 - categorical_accuracy: 0.9432
55136/60000 [==========================>...] - ETA: 7s - loss: 0.1880 - categorical_accuracy: 0.9432
55168/60000 [==========================>...] - ETA: 7s - loss: 0.1879 - categorical_accuracy: 0.9432
55232/60000 [==========================>...] - ETA: 7s - loss: 0.1877 - categorical_accuracy: 0.9432
55296/60000 [==========================>...] - ETA: 7s - loss: 0.1876 - categorical_accuracy: 0.9433
55360/60000 [==========================>...] - ETA: 7s - loss: 0.1875 - categorical_accuracy: 0.9433
55392/60000 [==========================>...] - ETA: 7s - loss: 0.1874 - categorical_accuracy: 0.9433
55456/60000 [==========================>...] - ETA: 7s - loss: 0.1873 - categorical_accuracy: 0.9434
55520/60000 [==========================>...] - ETA: 7s - loss: 0.1872 - categorical_accuracy: 0.9434
55552/60000 [==========================>...] - ETA: 7s - loss: 0.1872 - categorical_accuracy: 0.9434
55584/60000 [==========================>...] - ETA: 7s - loss: 0.1871 - categorical_accuracy: 0.9435
55616/60000 [==========================>...] - ETA: 6s - loss: 0.1870 - categorical_accuracy: 0.9435
55680/60000 [==========================>...] - ETA: 6s - loss: 0.1872 - categorical_accuracy: 0.9434
55744/60000 [==========================>...] - ETA: 6s - loss: 0.1870 - categorical_accuracy: 0.9435
55776/60000 [==========================>...] - ETA: 6s - loss: 0.1869 - categorical_accuracy: 0.9435
55808/60000 [==========================>...] - ETA: 6s - loss: 0.1869 - categorical_accuracy: 0.9435
55840/60000 [==========================>...] - ETA: 6s - loss: 0.1869 - categorical_accuracy: 0.9435
55872/60000 [==========================>...] - ETA: 6s - loss: 0.1868 - categorical_accuracy: 0.9435
55904/60000 [==========================>...] - ETA: 6s - loss: 0.1867 - categorical_accuracy: 0.9435
55936/60000 [==========================>...] - ETA: 6s - loss: 0.1866 - categorical_accuracy: 0.9436
56000/60000 [===========================>..] - ETA: 6s - loss: 0.1865 - categorical_accuracy: 0.9436
56032/60000 [===========================>..] - ETA: 6s - loss: 0.1864 - categorical_accuracy: 0.9436
56096/60000 [===========================>..] - ETA: 6s - loss: 0.1863 - categorical_accuracy: 0.9436
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1862 - categorical_accuracy: 0.9437
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1861 - categorical_accuracy: 0.9437
56256/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9437
56288/60000 [===========================>..] - ETA: 5s - loss: 0.1859 - categorical_accuracy: 0.9437
56352/60000 [===========================>..] - ETA: 5s - loss: 0.1858 - categorical_accuracy: 0.9437
56416/60000 [===========================>..] - ETA: 5s - loss: 0.1857 - categorical_accuracy: 0.9437
56480/60000 [===========================>..] - ETA: 5s - loss: 0.1856 - categorical_accuracy: 0.9437
56544/60000 [===========================>..] - ETA: 5s - loss: 0.1855 - categorical_accuracy: 0.9437
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1854 - categorical_accuracy: 0.9438
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1853 - categorical_accuracy: 0.9438
56640/60000 [===========================>..] - ETA: 5s - loss: 0.1853 - categorical_accuracy: 0.9438
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1853 - categorical_accuracy: 0.9438
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1852 - categorical_accuracy: 0.9438
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1850 - categorical_accuracy: 0.9439
56864/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9439
56896/60000 [===========================>..] - ETA: 4s - loss: 0.1851 - categorical_accuracy: 0.9439
56928/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9439
56992/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9439
57056/60000 [===========================>..] - ETA: 4s - loss: 0.1848 - categorical_accuracy: 0.9440
57120/60000 [===========================>..] - ETA: 4s - loss: 0.1847 - categorical_accuracy: 0.9440
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1848 - categorical_accuracy: 0.9440
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1847 - categorical_accuracy: 0.9440
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1847 - categorical_accuracy: 0.9441
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1846 - categorical_accuracy: 0.9441
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1845 - categorical_accuracy: 0.9441
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1843 - categorical_accuracy: 0.9442
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1842 - categorical_accuracy: 0.9442
57504/60000 [===========================>..] - ETA: 3s - loss: 0.1841 - categorical_accuracy: 0.9442
57536/60000 [===========================>..] - ETA: 3s - loss: 0.1840 - categorical_accuracy: 0.9442
57600/60000 [===========================>..] - ETA: 3s - loss: 0.1841 - categorical_accuracy: 0.9443
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1840 - categorical_accuracy: 0.9443
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1839 - categorical_accuracy: 0.9443
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1839 - categorical_accuracy: 0.9444
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1838 - categorical_accuracy: 0.9444
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1838 - categorical_accuracy: 0.9444
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1839 - categorical_accuracy: 0.9444
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1838 - categorical_accuracy: 0.9444
58016/60000 [============================>.] - ETA: 3s - loss: 0.1837 - categorical_accuracy: 0.9444
58080/60000 [============================>.] - ETA: 3s - loss: 0.1836 - categorical_accuracy: 0.9444
58112/60000 [============================>.] - ETA: 3s - loss: 0.1836 - categorical_accuracy: 0.9444
58176/60000 [============================>.] - ETA: 2s - loss: 0.1835 - categorical_accuracy: 0.9445
58208/60000 [============================>.] - ETA: 2s - loss: 0.1835 - categorical_accuracy: 0.9445
58272/60000 [============================>.] - ETA: 2s - loss: 0.1836 - categorical_accuracy: 0.9445
58336/60000 [============================>.] - ETA: 2s - loss: 0.1834 - categorical_accuracy: 0.9445
58368/60000 [============================>.] - ETA: 2s - loss: 0.1833 - categorical_accuracy: 0.9445
58432/60000 [============================>.] - ETA: 2s - loss: 0.1833 - categorical_accuracy: 0.9446
58496/60000 [============================>.] - ETA: 2s - loss: 0.1832 - categorical_accuracy: 0.9446
58560/60000 [============================>.] - ETA: 2s - loss: 0.1830 - categorical_accuracy: 0.9446
58592/60000 [============================>.] - ETA: 2s - loss: 0.1830 - categorical_accuracy: 0.9446
58656/60000 [============================>.] - ETA: 2s - loss: 0.1828 - categorical_accuracy: 0.9447
58720/60000 [============================>.] - ETA: 2s - loss: 0.1827 - categorical_accuracy: 0.9447
58784/60000 [============================>.] - ETA: 1s - loss: 0.1827 - categorical_accuracy: 0.9448
58848/60000 [============================>.] - ETA: 1s - loss: 0.1825 - categorical_accuracy: 0.9448
58880/60000 [============================>.] - ETA: 1s - loss: 0.1825 - categorical_accuracy: 0.9448
58944/60000 [============================>.] - ETA: 1s - loss: 0.1823 - categorical_accuracy: 0.9449
58976/60000 [============================>.] - ETA: 1s - loss: 0.1822 - categorical_accuracy: 0.9449
59008/60000 [============================>.] - ETA: 1s - loss: 0.1822 - categorical_accuracy: 0.9449
59040/60000 [============================>.] - ETA: 1s - loss: 0.1821 - categorical_accuracy: 0.9449
59072/60000 [============================>.] - ETA: 1s - loss: 0.1822 - categorical_accuracy: 0.9449
59136/60000 [============================>.] - ETA: 1s - loss: 0.1821 - categorical_accuracy: 0.9449
59200/60000 [============================>.] - ETA: 1s - loss: 0.1821 - categorical_accuracy: 0.9449
59232/60000 [============================>.] - ETA: 1s - loss: 0.1820 - categorical_accuracy: 0.9450
59296/60000 [============================>.] - ETA: 1s - loss: 0.1819 - categorical_accuracy: 0.9450
59360/60000 [============================>.] - ETA: 1s - loss: 0.1817 - categorical_accuracy: 0.9451
59392/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9451
59424/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9451
59456/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9451
59488/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9451
59520/60000 [============================>.] - ETA: 0s - loss: 0.1816 - categorical_accuracy: 0.9451
59584/60000 [============================>.] - ETA: 0s - loss: 0.1815 - categorical_accuracy: 0.9451
59648/60000 [============================>.] - ETA: 0s - loss: 0.1814 - categorical_accuracy: 0.9451
59712/60000 [============================>.] - ETA: 0s - loss: 0.1813 - categorical_accuracy: 0.9451
59744/60000 [============================>.] - ETA: 0s - loss: 0.1813 - categorical_accuracy: 0.9452
59808/60000 [============================>.] - ETA: 0s - loss: 0.1813 - categorical_accuracy: 0.9452
59840/60000 [============================>.] - ETA: 0s - loss: 0.1812 - categorical_accuracy: 0.9452
59872/60000 [============================>.] - ETA: 0s - loss: 0.1812 - categorical_accuracy: 0.9452
59936/60000 [============================>.] - ETA: 0s - loss: 0.1812 - categorical_accuracy: 0.9452
60000/60000 [==============================] - 99s 2ms/step - loss: 0.1811 - categorical_accuracy: 0.9452 - val_loss: 0.0510 - val_categorical_accuracy: 0.9835

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 13s
  224/10000 [..............................] - ETA: 4s 
  416/10000 [>.............................] - ETA: 3s
  608/10000 [>.............................] - ETA: 3s
  768/10000 [=>............................] - ETA: 3s
  960/10000 [=>............................] - ETA: 3s
 1152/10000 [==>...........................] - ETA: 2s
 1344/10000 [===>..........................] - ETA: 2s
 1536/10000 [===>..........................] - ETA: 2s
 1728/10000 [====>.........................] - ETA: 2s
 1920/10000 [====>.........................] - ETA: 2s
 2112/10000 [=====>........................] - ETA: 2s
 2304/10000 [=====>........................] - ETA: 2s
 2464/10000 [======>.......................] - ETA: 2s
 2656/10000 [======>.......................] - ETA: 2s
 2816/10000 [=======>......................] - ETA: 2s
 3008/10000 [========>.....................] - ETA: 2s
 3200/10000 [========>.....................] - ETA: 2s
 3392/10000 [=========>....................] - ETA: 2s
 3584/10000 [=========>....................] - ETA: 2s
 3776/10000 [==========>...................] - ETA: 1s
 3968/10000 [==========>...................] - ETA: 1s
 4128/10000 [===========>..................] - ETA: 1s
 4320/10000 [===========>..................] - ETA: 1s
 4480/10000 [============>.................] - ETA: 1s
 4640/10000 [============>.................] - ETA: 1s
 4800/10000 [=============>................] - ETA: 1s
 4992/10000 [=============>................] - ETA: 1s
 5184/10000 [==============>...............] - ETA: 1s
 5344/10000 [===============>..............] - ETA: 1s
 5504/10000 [===============>..............] - ETA: 1s
 5664/10000 [===============>..............] - ETA: 1s
 5824/10000 [================>.............] - ETA: 1s
 6016/10000 [=================>............] - ETA: 1s
 6208/10000 [=================>............] - ETA: 1s
 6400/10000 [==================>...........] - ETA: 1s
 6592/10000 [==================>...........] - ETA: 1s
 6752/10000 [===================>..........] - ETA: 1s
 6912/10000 [===================>..........] - ETA: 0s
 7072/10000 [====================>.........] - ETA: 0s
 7232/10000 [====================>.........] - ETA: 0s
 7392/10000 [=====================>........] - ETA: 0s
 7552/10000 [=====================>........] - ETA: 0s
 7744/10000 [======================>.......] - ETA: 0s
 7936/10000 [======================>.......] - ETA: 0s
 8128/10000 [=======================>......] - ETA: 0s
 8320/10000 [=======================>......] - ETA: 0s
 8512/10000 [========================>.....] - ETA: 0s
 8704/10000 [=========================>....] - ETA: 0s
 8864/10000 [=========================>....] - ETA: 0s
 9056/10000 [==========================>...] - ETA: 0s
 9248/10000 [==========================>...] - ETA: 0s
 9440/10000 [===========================>..] - ETA: 0s
 9600/10000 [===========================>..] - ETA: 0s
 9760/10000 [============================>.] - ETA: 0s
 9920/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 313us/step
[[5.7310629e-08 1.0016669e-07 5.4252774e-07 ... 9.9999440e-01
  1.6002440e-07 4.0854839e-06]
 [2.7257784e-06 1.7508217e-04 9.9981219e-01 ... 1.8660501e-08
  1.2619188e-06 2.1565982e-09]
 [6.1453001e-08 9.9997199e-01 9.3396261e-07 ... 4.8701258e-06
  1.3800980e-06 4.0744874e-07]
 ...
 [9.2684154e-09 1.2770887e-06 6.4023857e-09 ... 4.5848196e-06
  2.2869367e-06 3.0783885e-05]
 [7.6734623e-06 3.2525756e-07 6.9147852e-08 ... 2.9917146e-06
  9.6750958e-04 6.6740171e-07]
 [6.2064546e-06 4.9066881e-07 4.6501532e-06 ... 4.0711654e-08
  1.0131271e-07 3.7580236e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.05101578844415489, 'accuracy_test:': 0.9835000038146973}

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
From github.com:arita37/mlmodels_store
   4c257af..a89a2ff  master     -> origin/master
Updating 4c257af..a89a2ff
Fast-forward
 .../20200522/list_log_pullrequest_20200522.md      |   2 +-
 error_list/20200522/list_log_testall_20200522.md   | 126 +++++++++++++++++++++
 2 files changed, 127 insertions(+), 1 deletion(-)
[master ae44b05] ml_store
 1 file changed, 1462 insertions(+)
To github.com:arita37/mlmodels_store.git
   a89a2ff..ae44b05  master -> master





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
[master 68d969f] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   ae44b05..68d969f  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2621440/17464789 [===>..........................] - ETA: 0s
11124736/17464789 [==================>...........] - ETA: 0s
16048128/17464789 [==========================>...] - ETA: 0s
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
2020-05-22 08:51:19.404243: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-22 08:51:19.408205: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-22 08:51:19.408325: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c0fa5186a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 08:51:19.408338: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 11s - loss: 7.6973 - accuracy: 0.4980
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5746 - accuracy: 0.5060 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5848 - accuracy: 0.5053
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.5555 - accuracy: 0.5073
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.5869 - accuracy: 0.5052
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6513 - accuracy: 0.5010
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6798 - accuracy: 0.4991
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7088 - accuracy: 0.4972
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7484 - accuracy: 0.4947
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7080 - accuracy: 0.4973
11000/25000 [============>.................] - ETA: 3s - loss: 7.7029 - accuracy: 0.4976
12000/25000 [=============>................] - ETA: 3s - loss: 7.6832 - accuracy: 0.4989
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6548 - accuracy: 0.5008
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6820 - accuracy: 0.4990
15000/25000 [=================>............] - ETA: 2s - loss: 7.6789 - accuracy: 0.4992
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6848 - accuracy: 0.4988
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6765 - accuracy: 0.4994
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7075 - accuracy: 0.4973
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7062 - accuracy: 0.4974
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6958 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6907 - accuracy: 0.4984
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6868 - accuracy: 0.4987
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6833 - accuracy: 0.4989
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 8s 328us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f3d2f182d30>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f3d2f182d30> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 8.0806 - accuracy: 0.4730
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7970 - accuracy: 0.4915 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7075 - accuracy: 0.4973
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6436 - accuracy: 0.5015
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7341 - accuracy: 0.4956
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7637 - accuracy: 0.4937
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.8112 - accuracy: 0.4906
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7625 - accuracy: 0.4938
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7723 - accuracy: 0.4931
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7341 - accuracy: 0.4956
11000/25000 [============>.................] - ETA: 3s - loss: 7.7294 - accuracy: 0.4959
12000/25000 [=============>................] - ETA: 3s - loss: 7.6998 - accuracy: 0.4978
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7032 - accuracy: 0.4976
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6918 - accuracy: 0.4984
15000/25000 [=================>............] - ETA: 2s - loss: 7.6881 - accuracy: 0.4986
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6810 - accuracy: 0.4991
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6684 - accuracy: 0.4999
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6768 - accuracy: 0.4993
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6892 - accuracy: 0.4985
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6843 - accuracy: 0.4988
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6695 - accuracy: 0.4998
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6604 - accuracy: 0.5004
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6749 - accuracy: 0.4995
25000/25000 [==============================] - 8s 326us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 10s - loss: 7.8966 - accuracy: 0.4850
 2000/25000 [=>............................] - ETA: 8s - loss: 7.9120 - accuracy: 0.4840 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8302 - accuracy: 0.4893
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8391 - accuracy: 0.4888
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7924 - accuracy: 0.4918
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7535 - accuracy: 0.4943
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7805 - accuracy: 0.4926
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7222 - accuracy: 0.4964
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7007 - accuracy: 0.4978
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7111 - accuracy: 0.4971
11000/25000 [============>.................] - ETA: 3s - loss: 7.7154 - accuracy: 0.4968
12000/25000 [=============>................] - ETA: 3s - loss: 7.7037 - accuracy: 0.4976
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6938 - accuracy: 0.4982
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6962 - accuracy: 0.4981
15000/25000 [=================>............] - ETA: 2s - loss: 7.6728 - accuracy: 0.4996
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7145 - accuracy: 0.4969
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7198 - accuracy: 0.4965
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7152 - accuracy: 0.4968
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6981 - accuracy: 0.4979
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6873 - accuracy: 0.4987
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6652 - accuracy: 0.5001
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6562 - accuracy: 0.5007
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6606 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6545 - accuracy: 0.5008
25000/25000 [==============================] - 8s 327us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   68d969f..5fd2530  master     -> origin/master
Updating 68d969f..5fd2530
Fast-forward
 error_list/20200522/list_log_pullrequest_20200522.md | 2 +-
 error_list/20200522/list_log_testall_20200522.md     | 7 +++++++
 2 files changed, 8 insertions(+), 1 deletion(-)


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
[master 8850eee] ml_store
 1 file changed, 325 insertions(+)
To github.com:arita37/mlmodels_store.git
   5fd2530..8850eee  master -> master





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
[master 8791248] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   8850eee..8791248  master -> master





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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-22 08:52:40.001275: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-22 08:52:40.005754: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-22 08:52:40.005930: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55d2d0c07950 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 08:52:40.005946: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 208
Trainable params: 208
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2575 - binary_crossentropy: 0.8405500/500 [==============================] - 1s 1ms/sample - loss: 0.2530 - binary_crossentropy: 0.7519 - val_loss: 0.2500 - val_binary_crossentropy: 0.6931

  #### metrics   #################################################### 
{'MSE': 0.251276211266909}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
Total params: 208
Trainable params: 208
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 478
Trainable params: 478
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2574 - binary_crossentropy: 0.7082500/500 [==============================] - 1s 1ms/sample - loss: 0.2529 - binary_crossentropy: 0.6991 - val_loss: 0.2560 - val_binary_crossentropy: 0.7053

  #### metrics   #################################################### 
{'MSE': 0.25419955989300624}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 478
Trainable params: 478
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
Total params: 607
Trainable params: 607
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2501 - binary_crossentropy: 0.6933 - val_loss: 0.2499 - val_binary_crossentropy: 0.6929

  #### metrics   #################################################### 
{'MSE': 0.2499042123996828}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
Total params: 607
Trainable params: 607
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
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
Total params: 408
Trainable params: 408
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.3400 - binary_crossentropy: 5.2304500/500 [==============================] - 1s 2ms/sample - loss: 0.4700 - binary_crossentropy: 7.2255 - val_loss: 0.5220 - val_binary_crossentropy: 8.0518

  #### metrics   #################################################### 
{'MSE': 0.502}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
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
Total params: 408
Trainable params: 408
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 138
Trainable params: 138
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2889 - binary_crossentropy: 1.8124500/500 [==============================] - 1s 3ms/sample - loss: 0.2782 - binary_crossentropy: 1.7075 - val_loss: 0.2999 - val_binary_crossentropy: 1.9659

  #### metrics   #################################################### 
{'MSE': 0.2885796827347976}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 138
Trainable params: 138
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-22 08:53:43.822099: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:53:43.824368: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:53:43.829545: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 08:53:43.839088: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 08:53:43.840649: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 08:53:43.842280: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:53:43.843728: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2524 - val_binary_crossentropy: 0.6980
2020-05-22 08:53:44.824952: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:53:44.826438: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:53:44.830169: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 08:53:44.837989: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 08:53:44.839461: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 08:53:44.840757: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:53:44.841872: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25305986658591717}

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
2020-05-22 08:54:03.805797: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:54:03.806937: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:54:03.810120: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 08:54:03.815683: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 08:54:03.816616: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 08:54:03.817482: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:54:03.818281: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2519 - val_binary_crossentropy: 0.6970
2020-05-22 08:54:05.006974: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:54:05.007944: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:54:05.010240: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 08:54:05.014795: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 08:54:05.015592: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 08:54:05.016306: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:54:05.016987: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2524432463509605}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-22 08:54:33.096164: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:54:33.100522: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:54:33.113781: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 08:54:33.135496: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 08:54:33.139337: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 08:54:33.142717: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:54:33.146324: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 4s 4s/sample - loss: 0.1624 - binary_crossentropy: 0.5159 - val_loss: 0.2622 - val_binary_crossentropy: 0.7182
2020-05-22 08:54:35.009246: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:54:35.013156: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:54:35.023804: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 08:54:35.044465: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 08:54:35.048165: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 08:54:35.051564: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 08:54:35.054851: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.29897974625456375}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
Total params: 730
Trainable params: 730
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 5s - loss: 0.2877 - binary_crossentropy: 1.4140500/500 [==============================] - 4s 7ms/sample - loss: 0.3026 - binary_crossentropy: 1.8959 - val_loss: 0.3023 - val_binary_crossentropy: 1.9721

  #### metrics   #################################################### 
{'MSE': 0.3019995253388748}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
Total params: 730
Trainable params: 730
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 2)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 2)         8           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         10          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         8           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         12          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         16          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         8           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_5[0][0]           
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
Total params: 242
Trainable params: 242
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 5s - loss: 0.2902 - binary_crossentropy: 1.0527500/500 [==============================] - 4s 8ms/sample - loss: 0.2803 - binary_crossentropy: 0.8722 - val_loss: 0.2905 - val_binary_crossentropy: 0.8346

  #### metrics   #################################################### 
{'MSE': 0.2818057327847447}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 2)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 2)         8           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         10          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         8           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         12          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         16          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         8           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_5[0][0]           
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
Total params: 242
Trainable params: 242
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 1,909
Trainable params: 1,909
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 5s - loss: 0.2407 - binary_crossentropy: 0.6743500/500 [==============================] - 4s 8ms/sample - loss: 0.2531 - binary_crossentropy: 0.7262 - val_loss: 0.2526 - val_binary_crossentropy: 0.6985

  #### metrics   #################################################### 
{'MSE': 0.2520799002693677}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 1,909
Trainable params: 1,909
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
regionsequence_sum (InputLayer) [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 4)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 7)]          0                                            
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
region_10sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         8           regionsequence_max[0][0]         
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
Total params: 176
Trainable params: 176
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2450 - binary_crossentropy: 0.6829500/500 [==============================] - 5s 10ms/sample - loss: 0.2507 - binary_crossentropy: 0.6944 - val_loss: 0.2506 - val_binary_crossentropy: 0.6940

  #### metrics   #################################################### 
{'MSE': 0.25032038290799175}

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
regionsequence_sum (InputLayer) [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 4)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 7)]          0                                            
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
region_10sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         8           regionsequence_max[0][0]         
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
Total params: 176
Trainable params: 176
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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 1,407
Trainable params: 1,407
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2830 - binary_crossentropy: 1.1547500/500 [==============================] - 5s 10ms/sample - loss: 0.2829 - binary_crossentropy: 1.0780 - val_loss: 0.2777 - val_binary_crossentropy: 0.9607

  #### metrics   #################################################### 
{'MSE': 0.27758383003742204}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 1,407
Trainable params: 1,407
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         8           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         24          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           hash_11[0][0]                    
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
Total params: 3,103
Trainable params: 3,023
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2548 - binary_crossentropy: 0.7030500/500 [==============================] - 6s 11ms/sample - loss: 0.2521 - binary_crossentropy: 0.6975 - val_loss: 0.2502 - val_binary_crossentropy: 0.6936

  #### metrics   #################################################### 
{'MSE': 0.24989868758093686}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         8           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         24          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           hash_11[0][0]                    
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
Total params: 3,103
Trainable params: 3,023
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
From github.com:arita37/mlmodels_store
   8791248..b89bba4  master     -> origin/master
Updating 8791248..b89bba4
Fast-forward
 deps.txt                                           |   47 +-
 error_list/20200522/list_log_jupyter_20200522.md   |  762 +++---
 .../20200522/list_log_pullrequest_20200522.md      |    2 +-
 error_list/20200522/list_log_testall_20200522.md   |   11 +
 log_jupyter/log_jupyter.py                         | 2493 ++++++++++----------
 5 files changed, 1656 insertions(+), 1659 deletions(-)
[master 8ad70a7] ml_store
 1 file changed, 4956 insertions(+)
To github.com:arita37/mlmodels_store.git
   b89bba4..8ad70a7  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pplm.py 
 Generating text ... 
= Prefix of sentence =
<|endoftext|>The potato

 Unperturbed generated text :

<|endoftext|>The potato-shaped, potato-eating insect of modern times (Ophiocordyceps elegans) has a unique ability to adapt quickly to a wide range of environments. It is able to survive in many different environments, including the Arctic, deserts

 Perturbed generated text :

<|endoftext|>The potato bomb is nothing new. It's been on the news a lot since 9/11. In fact, since the bombing in Paris last November, a bomb has been detonated in every major European country in the European Union.

The bomb in Brussels


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
   8ad70a7..ceef662  master     -> origin/master
Updating 8ad70a7..ceef662
Fast-forward
 error_list/20200522/list_log_pullrequest_20200522.md |  2 +-
 error_list/20200522/list_log_testall_20200522.md     | 13 +++++++++++++
 2 files changed, 14 insertions(+), 1 deletion(-)
[master e674c15] ml_store
 2 files changed, 61 insertions(+), 38 deletions(-)
To github.com:arita37/mlmodels_store.git
   ceef662..e674c15  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//mlp.py 

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
[master 9618967] ml_store
 1 file changed, 32 insertions(+)
To github.com:arita37/mlmodels_store.git
   e674c15..9618967  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//nbeats.py 

  #### Loading params   ####################################### 

  #### Loading dataset  ####################################### 
   milk_production_pounds
0                     589
1                     561
2                     640
3                     656
4                     727
[[0.60784314]
 [0.57894737]
 [0.66047472]
 [0.67698658]
 [0.750258  ]
 [0.71929825]
 [0.66047472]
 [0.61816305]
 [0.58617131]
 [0.59545924]
 [0.57069143]
 [0.6006192 ]
 [0.61919505]
 [0.58410733]
 [0.67389061]
 [0.69453044]
 [0.76573787]
 [0.73890609]
 [0.68111455]
 [0.63673891]
 [0.60165119]
 [0.60577915]
 [0.58307534]
 [0.61713106]
 [0.64809082]
 [0.6377709 ]
 [0.71001032]
 [0.72755418]
 [0.79463364]
 [0.75954592]
 [0.6996904 ]
 [0.65944272]
 [0.62332301]
 [0.63054696]
 [0.6130031 ]
 [0.65428277]
 [0.67905057]
 [0.64189886]
 [0.73168215]
 [0.74509804]
 [0.80701754]
 [0.78018576]
 [0.7244582 ]
 [0.67389061]
 [0.63467492]
 [0.64086687]
 [0.62125903]
 [0.65531476]
 [0.69865841]
 [0.65531476]
 [0.75954592]
 [0.77915377]
 [0.8369453 ]
 [0.82352941]
 [0.75851393]
 [0.71929825]
 [0.68214654]
 [0.68833849]
 [0.66563467]
 [0.71001032]
 [0.73581011]
 [0.68833849]
 [0.78637771]
 [0.80908153]
 [0.86377709]
 [0.84313725]
 [0.79153767]
 [0.74509804]
 [0.70278638]
 [0.70897833]
 [0.68111455]
 [0.72033024]
 [0.73993808]
 [0.71826625]
 [0.7997936 ]
 [0.82146543]
 [0.88544892]
 [0.85242518]
 [0.80804954]
 [0.76367389]
 [0.72342621]
 [0.72858617]
 [0.69865841]
 [0.73374613]
 [0.75748194]
 [0.7120743 ]
 [0.81011352]
 [0.83075335]
 [0.89886481]
 [0.87203302]
 [0.82662539]
 [0.78844169]
 [0.74819401]
 [0.74613003]
 [0.7120743 ]
 [0.75748194]
 [0.77399381]
 [0.72961816]
 [0.83281734]
 [0.8503612 ]
 [0.91434469]
 [0.88648091]
 [0.84520124]
 [0.80804954]
 [0.76367389]
 [0.77089783]
 [0.73374613]
 [0.7750258 ]
 [0.82972136]
 [0.78018576]
 [0.8875129 ]
 [0.90608875]
 [0.97213622]
 [0.94220846]
 [0.89680083]
 [0.86068111]
 [0.81527348]
 [0.8255934 ]
 [0.7874097 ]
 [0.8255934 ]
 [0.85242518]
 [0.8245614 ]
 [0.91847265]
 [0.92879257]
 [0.99174407]
 [0.96491228]
 [0.92260062]
 [0.88235294]
 [0.83488132]
 [0.83591331]
 [0.79050568]
 [0.83075335]
 [0.84726522]
 [0.79772962]
 [0.91124871]
 [0.92672859]
 [0.9876161 ]
 [0.95356037]
 [0.90918473]
 [0.86377709]
 [0.80908153]
 [0.81630547]
 [0.78431373]
 [0.82765738]
 [0.85448916]
 [0.80288958]
 [0.91744066]
 [0.93085655]
 [1.        ]
 [0.97729618]
 [0.9370485 ]
 [0.89473684]
 [0.84107327]
 [0.8379773 ]
 [0.79772962]
 [0.83900929]
 [0.86068111]
 [0.80701754]
 [0.92053664]
 [0.93188854]
 [0.99690402]
 [0.96697626]
 [0.9246646 ]
 [0.88544892]
 [0.84313725]
 [0.85345717]
 [0.82249742]
 [0.86996904]]

  #### Model setup   ########################################## 
| N-Beats
| --  Stack Generic (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140108539285456
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140108539285232
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140108539284000
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140108539283552
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140108539283048
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140108539282712

  #### Model fit   ############################################ 
   milk_production_pounds
0                     589
1                     561
2                     640
3                     656
4                     727
[[0.60784314]
 [0.57894737]
 [0.66047472]
 [0.67698658]
 [0.750258  ]
 [0.71929825]
 [0.66047472]
 [0.61816305]
 [0.58617131]
 [0.59545924]
 [0.57069143]
 [0.6006192 ]
 [0.61919505]
 [0.58410733]
 [0.67389061]
 [0.69453044]
 [0.76573787]
 [0.73890609]
 [0.68111455]
 [0.63673891]
 [0.60165119]
 [0.60577915]
 [0.58307534]
 [0.61713106]
 [0.64809082]
 [0.6377709 ]
 [0.71001032]
 [0.72755418]
 [0.79463364]
 [0.75954592]
 [0.6996904 ]
 [0.65944272]
 [0.62332301]
 [0.63054696]
 [0.6130031 ]
 [0.65428277]
 [0.67905057]
 [0.64189886]
 [0.73168215]
 [0.74509804]
 [0.80701754]
 [0.78018576]
 [0.7244582 ]
 [0.67389061]
 [0.63467492]
 [0.64086687]
 [0.62125903]
 [0.65531476]
 [0.69865841]
 [0.65531476]
 [0.75954592]
 [0.77915377]
 [0.8369453 ]
 [0.82352941]
 [0.75851393]
 [0.71929825]
 [0.68214654]
 [0.68833849]
 [0.66563467]
 [0.71001032]
 [0.73581011]
 [0.68833849]
 [0.78637771]
 [0.80908153]
 [0.86377709]
 [0.84313725]
 [0.79153767]
 [0.74509804]
 [0.70278638]
 [0.70897833]
 [0.68111455]
 [0.72033024]
 [0.73993808]
 [0.71826625]
 [0.7997936 ]
 [0.82146543]
 [0.88544892]
 [0.85242518]
 [0.80804954]
 [0.76367389]
 [0.72342621]
 [0.72858617]
 [0.69865841]
 [0.73374613]
 [0.75748194]
 [0.7120743 ]
 [0.81011352]
 [0.83075335]
 [0.89886481]
 [0.87203302]
 [0.82662539]
 [0.78844169]
 [0.74819401]
 [0.74613003]
 [0.7120743 ]
 [0.75748194]
 [0.77399381]
 [0.72961816]
 [0.83281734]
 [0.8503612 ]
 [0.91434469]
 [0.88648091]
 [0.84520124]
 [0.80804954]
 [0.76367389]
 [0.77089783]
 [0.73374613]
 [0.7750258 ]
 [0.82972136]
 [0.78018576]
 [0.8875129 ]
 [0.90608875]
 [0.97213622]
 [0.94220846]
 [0.89680083]
 [0.86068111]
 [0.81527348]
 [0.8255934 ]
 [0.7874097 ]
 [0.8255934 ]
 [0.85242518]
 [0.8245614 ]
 [0.91847265]
 [0.92879257]
 [0.99174407]
 [0.96491228]
 [0.92260062]
 [0.88235294]
 [0.83488132]
 [0.83591331]
 [0.79050568]
 [0.83075335]
 [0.84726522]
 [0.79772962]
 [0.91124871]
 [0.92672859]
 [0.9876161 ]
 [0.95356037]
 [0.90918473]
 [0.86377709]
 [0.80908153]
 [0.81630547]
 [0.78431373]
 [0.82765738]
 [0.85448916]
 [0.80288958]
 [0.91744066]
 [0.93085655]
 [1.        ]
 [0.97729618]
 [0.9370485 ]
 [0.89473684]
 [0.84107327]
 [0.8379773 ]
 [0.79772962]
 [0.83900929]
 [0.86068111]
 [0.80701754]
 [0.92053664]
 [0.93188854]
 [0.99690402]
 [0.96697626]
 [0.9246646 ]
 [0.88544892]
 [0.84313725]
 [0.85345717]
 [0.82249742]
 [0.86996904]]
--- fiting ---
grad_step = 000000, loss = 0.732677
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.575314
grad_step = 000002, loss = 0.483721
grad_step = 000003, loss = 0.392136
grad_step = 000004, loss = 0.292418
grad_step = 000005, loss = 0.200880
grad_step = 000006, loss = 0.147129
grad_step = 000007, loss = 0.138141
grad_step = 000008, loss = 0.116594
grad_step = 000009, loss = 0.067791
grad_step = 000010, loss = 0.027661
grad_step = 000011, loss = 0.012702
grad_step = 000012, loss = 0.019405
grad_step = 000013, loss = 0.030510
grad_step = 000014, loss = 0.032722
grad_step = 000015, loss = 0.027210
grad_step = 000016, loss = 0.019240
grad_step = 000017, loss = 0.013981
grad_step = 000018, loss = 0.013648
grad_step = 000019, loss = 0.015740
grad_step = 000020, loss = 0.016401
grad_step = 000021, loss = 0.014640
grad_step = 000022, loss = 0.012020
grad_step = 000023, loss = 0.009975
grad_step = 000024, loss = 0.008644
grad_step = 000025, loss = 0.007650
grad_step = 000026, loss = 0.006922
grad_step = 000027, loss = 0.006761
grad_step = 000028, loss = 0.007331
grad_step = 000029, loss = 0.008167
grad_step = 000030, loss = 0.008501
grad_step = 000031, loss = 0.007975
grad_step = 000032, loss = 0.006981
grad_step = 000033, loss = 0.006170
grad_step = 000034, loss = 0.005834
grad_step = 000035, loss = 0.005848
grad_step = 000036, loss = 0.005916
grad_step = 000037, loss = 0.005894
grad_step = 000038, loss = 0.005815
grad_step = 000039, loss = 0.005730
grad_step = 000040, loss = 0.005611
grad_step = 000041, loss = 0.005400
grad_step = 000042, loss = 0.005157
grad_step = 000043, loss = 0.005006
grad_step = 000044, loss = 0.005006
grad_step = 000045, loss = 0.005074
grad_step = 000046, loss = 0.005086
grad_step = 000047, loss = 0.004995
grad_step = 000048, loss = 0.004854
grad_step = 000049, loss = 0.004735
grad_step = 000050, loss = 0.004657
grad_step = 000051, loss = 0.004587
grad_step = 000052, loss = 0.004509
grad_step = 000053, loss = 0.004447
grad_step = 000054, loss = 0.004419
grad_step = 000055, loss = 0.004403
grad_step = 000056, loss = 0.004363
grad_step = 000057, loss = 0.004299
grad_step = 000058, loss = 0.004240
grad_step = 000059, loss = 0.004188
grad_step = 000060, loss = 0.004120
grad_step = 000061, loss = 0.004045
grad_step = 000062, loss = 0.003990
grad_step = 000063, loss = 0.003958
grad_step = 000064, loss = 0.003922
grad_step = 000065, loss = 0.003872
grad_step = 000066, loss = 0.003819
grad_step = 000067, loss = 0.003760
grad_step = 000068, loss = 0.003689
grad_step = 000069, loss = 0.003621
grad_step = 000070, loss = 0.003563
grad_step = 000071, loss = 0.003502
grad_step = 000072, loss = 0.003434
grad_step = 000073, loss = 0.003365
grad_step = 000074, loss = 0.003291
grad_step = 000075, loss = 0.003205
grad_step = 000076, loss = 0.003122
grad_step = 000077, loss = 0.003033
grad_step = 000078, loss = 0.002937
grad_step = 000079, loss = 0.002848
grad_step = 000080, loss = 0.002751
grad_step = 000081, loss = 0.002659
grad_step = 000082, loss = 0.002558
grad_step = 000083, loss = 0.002462
grad_step = 000084, loss = 0.002357
grad_step = 000085, loss = 0.002259
grad_step = 000086, loss = 0.002162
grad_step = 000087, loss = 0.002067
grad_step = 000088, loss = 0.001984
grad_step = 000089, loss = 0.001931
grad_step = 000090, loss = 0.001832
grad_step = 000091, loss = 0.001743
grad_step = 000092, loss = 0.001707
grad_step = 000093, loss = 0.001604
grad_step = 000094, loss = 0.001541
grad_step = 000095, loss = 0.001479
grad_step = 000096, loss = 0.001389
grad_step = 000097, loss = 0.001345
grad_step = 000098, loss = 0.001264
grad_step = 000099, loss = 0.001219
grad_step = 000100, loss = 0.001162
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.001114
grad_step = 000102, loss = 0.001080
grad_step = 000103, loss = 0.001032
grad_step = 000104, loss = 0.001013
grad_step = 000105, loss = 0.000978
grad_step = 000106, loss = 0.000955
grad_step = 000107, loss = 0.000931
grad_step = 000108, loss = 0.000906
grad_step = 000109, loss = 0.000888
grad_step = 000110, loss = 0.000868
grad_step = 000111, loss = 0.000855
grad_step = 000112, loss = 0.000844
grad_step = 000113, loss = 0.000843
grad_step = 000114, loss = 0.000867
grad_step = 000115, loss = 0.000880
grad_step = 000116, loss = 0.000837
grad_step = 000117, loss = 0.000810
grad_step = 000118, loss = 0.000840
grad_step = 000119, loss = 0.000806
grad_step = 000120, loss = 0.000794
grad_step = 000121, loss = 0.000800
grad_step = 000122, loss = 0.000773
grad_step = 000123, loss = 0.000767
grad_step = 000124, loss = 0.000765
grad_step = 000125, loss = 0.000739
grad_step = 000126, loss = 0.000742
grad_step = 000127, loss = 0.000730
grad_step = 000128, loss = 0.000717
grad_step = 000129, loss = 0.000717
grad_step = 000130, loss = 0.000705
grad_step = 000131, loss = 0.000698
grad_step = 000132, loss = 0.000697
grad_step = 000133, loss = 0.000684
grad_step = 000134, loss = 0.000682
grad_step = 000135, loss = 0.000675
grad_step = 000136, loss = 0.000667
grad_step = 000137, loss = 0.000664
grad_step = 000138, loss = 0.000656
grad_step = 000139, loss = 0.000649
grad_step = 000140, loss = 0.000644
grad_step = 000141, loss = 0.000639
grad_step = 000142, loss = 0.000631
grad_step = 000143, loss = 0.000625
grad_step = 000144, loss = 0.000621
grad_step = 000145, loss = 0.000615
grad_step = 000146, loss = 0.000606
grad_step = 000147, loss = 0.000602
grad_step = 000148, loss = 0.000597
grad_step = 000149, loss = 0.000591
grad_step = 000150, loss = 0.000586
grad_step = 000151, loss = 0.000580
grad_step = 000152, loss = 0.000574
grad_step = 000153, loss = 0.000571
grad_step = 000154, loss = 0.000570
grad_step = 000155, loss = 0.000570
grad_step = 000156, loss = 0.000570
grad_step = 000157, loss = 0.000567
grad_step = 000158, loss = 0.000561
grad_step = 000159, loss = 0.000552
grad_step = 000160, loss = 0.000542
grad_step = 000161, loss = 0.000535
grad_step = 000162, loss = 0.000531
grad_step = 000163, loss = 0.000530
grad_step = 000164, loss = 0.000530
grad_step = 000165, loss = 0.000529
grad_step = 000166, loss = 0.000527
grad_step = 000167, loss = 0.000523
grad_step = 000168, loss = 0.000516
grad_step = 000169, loss = 0.000509
grad_step = 000170, loss = 0.000502
grad_step = 000171, loss = 0.000496
grad_step = 000172, loss = 0.000492
grad_step = 000173, loss = 0.000490
grad_step = 000174, loss = 0.000487
grad_step = 000175, loss = 0.000486
grad_step = 000176, loss = 0.000487
grad_step = 000177, loss = 0.000490
grad_step = 000178, loss = 0.000498
grad_step = 000179, loss = 0.000507
grad_step = 000180, loss = 0.000514
grad_step = 000181, loss = 0.000505
grad_step = 000182, loss = 0.000480
grad_step = 000183, loss = 0.000458
grad_step = 000184, loss = 0.000457
grad_step = 000185, loss = 0.000469
grad_step = 000186, loss = 0.000471
grad_step = 000187, loss = 0.000458
grad_step = 000188, loss = 0.000444
grad_step = 000189, loss = 0.000442
grad_step = 000190, loss = 0.000447
grad_step = 000191, loss = 0.000447
grad_step = 000192, loss = 0.000439
grad_step = 000193, loss = 0.000429
grad_step = 000194, loss = 0.000427
grad_step = 000195, loss = 0.000428
grad_step = 000196, loss = 0.000429
grad_step = 000197, loss = 0.000426
grad_step = 000198, loss = 0.000421
grad_step = 000199, loss = 0.000416
grad_step = 000200, loss = 0.000412
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000408
grad_step = 000202, loss = 0.000405
grad_step = 000203, loss = 0.000404
grad_step = 000204, loss = 0.000404
grad_step = 000205, loss = 0.000406
grad_step = 000206, loss = 0.000406
grad_step = 000207, loss = 0.000403
grad_step = 000208, loss = 0.000399
grad_step = 000209, loss = 0.000397
grad_step = 000210, loss = 0.000399
grad_step = 000211, loss = 0.000402
grad_step = 000212, loss = 0.000401
grad_step = 000213, loss = 0.000396
grad_step = 000214, loss = 0.000394
grad_step = 000215, loss = 0.000395
grad_step = 000216, loss = 0.000398
grad_step = 000217, loss = 0.000393
grad_step = 000218, loss = 0.000383
grad_step = 000219, loss = 0.000375
grad_step = 000220, loss = 0.000372
grad_step = 000221, loss = 0.000367
grad_step = 000222, loss = 0.000360
grad_step = 000223, loss = 0.000357
grad_step = 000224, loss = 0.000359
grad_step = 000225, loss = 0.000362
grad_step = 000226, loss = 0.000361
grad_step = 000227, loss = 0.000360
grad_step = 000228, loss = 0.000362
grad_step = 000229, loss = 0.000371
grad_step = 000230, loss = 0.000378
grad_step = 000231, loss = 0.000386
grad_step = 000232, loss = 0.000391
grad_step = 000233, loss = 0.000396
grad_step = 000234, loss = 0.000383
grad_step = 000235, loss = 0.000355
grad_step = 000236, loss = 0.000334
grad_step = 000237, loss = 0.000337
grad_step = 000238, loss = 0.000350
grad_step = 000239, loss = 0.000355
grad_step = 000240, loss = 0.000349
grad_step = 000241, loss = 0.000338
grad_step = 000242, loss = 0.000330
grad_step = 000243, loss = 0.000325
grad_step = 000244, loss = 0.000326
grad_step = 000245, loss = 0.000333
grad_step = 000246, loss = 0.000333
grad_step = 000247, loss = 0.000325
grad_step = 000248, loss = 0.000316
grad_step = 000249, loss = 0.000314
grad_step = 000250, loss = 0.000314
grad_step = 000251, loss = 0.000314
grad_step = 000252, loss = 0.000313
grad_step = 000253, loss = 0.000314
grad_step = 000254, loss = 0.000316
grad_step = 000255, loss = 0.000313
grad_step = 000256, loss = 0.000307
grad_step = 000257, loss = 0.000302
grad_step = 000258, loss = 0.000301
grad_step = 000259, loss = 0.000300
grad_step = 000260, loss = 0.000298
grad_step = 000261, loss = 0.000295
grad_step = 000262, loss = 0.000292
grad_step = 000263, loss = 0.000292
grad_step = 000264, loss = 0.000292
grad_step = 000265, loss = 0.000293
grad_step = 000266, loss = 0.000295
grad_step = 000267, loss = 0.000302
grad_step = 000268, loss = 0.000318
grad_step = 000269, loss = 0.000354
grad_step = 000270, loss = 0.000397
grad_step = 000271, loss = 0.000452
grad_step = 000272, loss = 0.000425
grad_step = 000273, loss = 0.000350
grad_step = 000274, loss = 0.000281
grad_step = 000275, loss = 0.000308
grad_step = 000276, loss = 0.000363
grad_step = 000277, loss = 0.000329
grad_step = 000278, loss = 0.000278
grad_step = 000279, loss = 0.000292
grad_step = 000280, loss = 0.000324
grad_step = 000281, loss = 0.000303
grad_step = 000282, loss = 0.000271
grad_step = 000283, loss = 0.000288
grad_step = 000284, loss = 0.000306
grad_step = 000285, loss = 0.000281
grad_step = 000286, loss = 0.000268
grad_step = 000287, loss = 0.000284
grad_step = 000288, loss = 0.000284
grad_step = 000289, loss = 0.000268
grad_step = 000290, loss = 0.000267
grad_step = 000291, loss = 0.000275
grad_step = 000292, loss = 0.000271
grad_step = 000293, loss = 0.000261
grad_step = 000294, loss = 0.000263
grad_step = 000295, loss = 0.000269
grad_step = 000296, loss = 0.000263
grad_step = 000297, loss = 0.000254
grad_step = 000298, loss = 0.000256
grad_step = 000299, loss = 0.000261
grad_step = 000300, loss = 0.000258
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000251
grad_step = 000302, loss = 0.000250
grad_step = 000303, loss = 0.000252
grad_step = 000304, loss = 0.000251
grad_step = 000305, loss = 0.000248
grad_step = 000306, loss = 0.000245
grad_step = 000307, loss = 0.000246
grad_step = 000308, loss = 0.000247
grad_step = 000309, loss = 0.000245
grad_step = 000310, loss = 0.000242
grad_step = 000311, loss = 0.000241
grad_step = 000312, loss = 0.000241
grad_step = 000313, loss = 0.000242
grad_step = 000314, loss = 0.000242
grad_step = 000315, loss = 0.000240
grad_step = 000316, loss = 0.000237
grad_step = 000317, loss = 0.000236
grad_step = 000318, loss = 0.000237
grad_step = 000319, loss = 0.000237
grad_step = 000320, loss = 0.000238
grad_step = 000321, loss = 0.000239
grad_step = 000322, loss = 0.000241
grad_step = 000323, loss = 0.000243
grad_step = 000324, loss = 0.000246
grad_step = 000325, loss = 0.000247
grad_step = 000326, loss = 0.000245
grad_step = 000327, loss = 0.000241
grad_step = 000328, loss = 0.000236
grad_step = 000329, loss = 0.000232
grad_step = 000330, loss = 0.000229
grad_step = 000331, loss = 0.000229
grad_step = 000332, loss = 0.000229
grad_step = 000333, loss = 0.000228
grad_step = 000334, loss = 0.000226
grad_step = 000335, loss = 0.000224
grad_step = 000336, loss = 0.000223
grad_step = 000337, loss = 0.000223
grad_step = 000338, loss = 0.000225
grad_step = 000339, loss = 0.000229
grad_step = 000340, loss = 0.000233
grad_step = 000341, loss = 0.000239
grad_step = 000342, loss = 0.000245
grad_step = 000343, loss = 0.000257
grad_step = 000344, loss = 0.000268
grad_step = 000345, loss = 0.000287
grad_step = 000346, loss = 0.000292
grad_step = 000347, loss = 0.000287
grad_step = 000348, loss = 0.000258
grad_step = 000349, loss = 0.000227
grad_step = 000350, loss = 0.000215
grad_step = 000351, loss = 0.000228
grad_step = 000352, loss = 0.000248
grad_step = 000353, loss = 0.000247
grad_step = 000354, loss = 0.000229
grad_step = 000355, loss = 0.000212
grad_step = 000356, loss = 0.000213
grad_step = 000357, loss = 0.000224
grad_step = 000358, loss = 0.000231
grad_step = 000359, loss = 0.000226
grad_step = 000360, loss = 0.000215
grad_step = 000361, loss = 0.000210
grad_step = 000362, loss = 0.000214
grad_step = 000363, loss = 0.000221
grad_step = 000364, loss = 0.000221
grad_step = 000365, loss = 0.000214
grad_step = 000366, loss = 0.000206
grad_step = 000367, loss = 0.000204
grad_step = 000368, loss = 0.000207
grad_step = 000369, loss = 0.000210
grad_step = 000370, loss = 0.000211
grad_step = 000371, loss = 0.000207
grad_step = 000372, loss = 0.000203
grad_step = 000373, loss = 0.000200
grad_step = 000374, loss = 0.000201
grad_step = 000375, loss = 0.000204
grad_step = 000376, loss = 0.000208
grad_step = 000377, loss = 0.000212
grad_step = 000378, loss = 0.000221
grad_step = 000379, loss = 0.000241
grad_step = 000380, loss = 0.000274
grad_step = 000381, loss = 0.000290
grad_step = 000382, loss = 0.000279
grad_step = 000383, loss = 0.000230
grad_step = 000384, loss = 0.000213
grad_step = 000385, loss = 0.000233
grad_step = 000386, loss = 0.000239
grad_step = 000387, loss = 0.000213
grad_step = 000388, loss = 0.000202
grad_step = 000389, loss = 0.000220
grad_step = 000390, loss = 0.000225
grad_step = 000391, loss = 0.000202
grad_step = 000392, loss = 0.000195
grad_step = 000393, loss = 0.000210
grad_step = 000394, loss = 0.000212
grad_step = 000395, loss = 0.000199
grad_step = 000396, loss = 0.000194
grad_step = 000397, loss = 0.000200
grad_step = 000398, loss = 0.000204
grad_step = 000399, loss = 0.000197
grad_step = 000400, loss = 0.000191
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000193
grad_step = 000402, loss = 0.000198
grad_step = 000403, loss = 0.000197
grad_step = 000404, loss = 0.000191
grad_step = 000405, loss = 0.000188
grad_step = 000406, loss = 0.000189
grad_step = 000407, loss = 0.000190
grad_step = 000408, loss = 0.000191
grad_step = 000409, loss = 0.000189
grad_step = 000410, loss = 0.000187
grad_step = 000411, loss = 0.000187
grad_step = 000412, loss = 0.000188
grad_step = 000413, loss = 0.000187
grad_step = 000414, loss = 0.000185
grad_step = 000415, loss = 0.000184
grad_step = 000416, loss = 0.000184
grad_step = 000417, loss = 0.000184
grad_step = 000418, loss = 0.000184
grad_step = 000419, loss = 0.000184
grad_step = 000420, loss = 0.000183
grad_step = 000421, loss = 0.000182
grad_step = 000422, loss = 0.000181
grad_step = 000423, loss = 0.000181
grad_step = 000424, loss = 0.000182
grad_step = 000425, loss = 0.000183
grad_step = 000426, loss = 0.000185
grad_step = 000427, loss = 0.000189
grad_step = 000428, loss = 0.000196
grad_step = 000429, loss = 0.000206
grad_step = 000430, loss = 0.000224
grad_step = 000431, loss = 0.000247
grad_step = 000432, loss = 0.000285
grad_step = 000433, loss = 0.000307
grad_step = 000434, loss = 0.000319
grad_step = 000435, loss = 0.000264
grad_step = 000436, loss = 0.000199
grad_step = 000437, loss = 0.000181
grad_step = 000438, loss = 0.000216
grad_step = 000439, loss = 0.000241
grad_step = 000440, loss = 0.000214
grad_step = 000441, loss = 0.000186
grad_step = 000442, loss = 0.000193
grad_step = 000443, loss = 0.000205
grad_step = 000444, loss = 0.000197
grad_step = 000445, loss = 0.000186
grad_step = 000446, loss = 0.000195
grad_step = 000447, loss = 0.000200
grad_step = 000448, loss = 0.000184
grad_step = 000449, loss = 0.000176
grad_step = 000450, loss = 0.000186
grad_step = 000451, loss = 0.000190
grad_step = 000452, loss = 0.000181
grad_step = 000453, loss = 0.000174
grad_step = 000454, loss = 0.000178
grad_step = 000455, loss = 0.000182
grad_step = 000456, loss = 0.000178
grad_step = 000457, loss = 0.000174
grad_step = 000458, loss = 0.000176
grad_step = 000459, loss = 0.000178
grad_step = 000460, loss = 0.000175
grad_step = 000461, loss = 0.000171
grad_step = 000462, loss = 0.000171
grad_step = 000463, loss = 0.000174
grad_step = 000464, loss = 0.000175
grad_step = 000465, loss = 0.000172
grad_step = 000466, loss = 0.000170
grad_step = 000467, loss = 0.000171
grad_step = 000468, loss = 0.000172
grad_step = 000469, loss = 0.000171
grad_step = 000470, loss = 0.000169
grad_step = 000471, loss = 0.000168
grad_step = 000472, loss = 0.000169
grad_step = 000473, loss = 0.000169
grad_step = 000474, loss = 0.000169
grad_step = 000475, loss = 0.000168
grad_step = 000476, loss = 0.000168
grad_step = 000477, loss = 0.000168
grad_step = 000478, loss = 0.000169
grad_step = 000479, loss = 0.000169
grad_step = 000480, loss = 0.000171
grad_step = 000481, loss = 0.000172
grad_step = 000482, loss = 0.000173
grad_step = 000483, loss = 0.000175
grad_step = 000484, loss = 0.000177
grad_step = 000485, loss = 0.000180
grad_step = 000486, loss = 0.000180
grad_step = 000487, loss = 0.000177
grad_step = 000488, loss = 0.000174
grad_step = 000489, loss = 0.000170
grad_step = 000490, loss = 0.000168
grad_step = 000491, loss = 0.000166
grad_step = 000492, loss = 0.000166
grad_step = 000493, loss = 0.000166
grad_step = 000494, loss = 0.000165
grad_step = 000495, loss = 0.000165
grad_step = 000496, loss = 0.000167
grad_step = 000497, loss = 0.000169
grad_step = 000498, loss = 0.000172
grad_step = 000499, loss = 0.000175
grad_step = 000500, loss = 0.000177
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000178
Finished.

  #### Predict    ############################################# 
   milk_production_pounds
0                     589
1                     561
2                     640
3                     656
4                     727
[[0.60784314]
 [0.57894737]
 [0.66047472]
 [0.67698658]
 [0.750258  ]
 [0.71929825]
 [0.66047472]
 [0.61816305]
 [0.58617131]
 [0.59545924]
 [0.57069143]
 [0.6006192 ]
 [0.61919505]
 [0.58410733]
 [0.67389061]
 [0.69453044]
 [0.76573787]
 [0.73890609]
 [0.68111455]
 [0.63673891]
 [0.60165119]
 [0.60577915]
 [0.58307534]
 [0.61713106]
 [0.64809082]
 [0.6377709 ]
 [0.71001032]
 [0.72755418]
 [0.79463364]
 [0.75954592]
 [0.6996904 ]
 [0.65944272]
 [0.62332301]
 [0.63054696]
 [0.6130031 ]
 [0.65428277]
 [0.67905057]
 [0.64189886]
 [0.73168215]
 [0.74509804]
 [0.80701754]
 [0.78018576]
 [0.7244582 ]
 [0.67389061]
 [0.63467492]
 [0.64086687]
 [0.62125903]
 [0.65531476]
 [0.69865841]
 [0.65531476]
 [0.75954592]
 [0.77915377]
 [0.8369453 ]
 [0.82352941]
 [0.75851393]
 [0.71929825]
 [0.68214654]
 [0.68833849]
 [0.66563467]
 [0.71001032]
 [0.73581011]
 [0.68833849]
 [0.78637771]
 [0.80908153]
 [0.86377709]
 [0.84313725]
 [0.79153767]
 [0.74509804]
 [0.70278638]
 [0.70897833]
 [0.68111455]
 [0.72033024]
 [0.73993808]
 [0.71826625]
 [0.7997936 ]
 [0.82146543]
 [0.88544892]
 [0.85242518]
 [0.80804954]
 [0.76367389]
 [0.72342621]
 [0.72858617]
 [0.69865841]
 [0.73374613]
 [0.75748194]
 [0.7120743 ]
 [0.81011352]
 [0.83075335]
 [0.89886481]
 [0.87203302]
 [0.82662539]
 [0.78844169]
 [0.74819401]
 [0.74613003]
 [0.7120743 ]
 [0.75748194]
 [0.77399381]
 [0.72961816]
 [0.83281734]
 [0.8503612 ]
 [0.91434469]
 [0.88648091]
 [0.84520124]
 [0.80804954]
 [0.76367389]
 [0.77089783]
 [0.73374613]
 [0.7750258 ]
 [0.82972136]
 [0.78018576]
 [0.8875129 ]
 [0.90608875]
 [0.97213622]
 [0.94220846]
 [0.89680083]
 [0.86068111]
 [0.81527348]
 [0.8255934 ]
 [0.7874097 ]
 [0.8255934 ]
 [0.85242518]
 [0.8245614 ]
 [0.91847265]
 [0.92879257]
 [0.99174407]
 [0.96491228]
 [0.92260062]
 [0.88235294]
 [0.83488132]
 [0.83591331]
 [0.79050568]
 [0.83075335]
 [0.84726522]
 [0.79772962]
 [0.91124871]
 [0.92672859]
 [0.9876161 ]
 [0.95356037]
 [0.90918473]
 [0.86377709]
 [0.80908153]
 [0.81630547]
 [0.78431373]
 [0.82765738]
 [0.85448916]
 [0.80288958]
 [0.91744066]
 [0.93085655]
 [1.        ]
 [0.97729618]
 [0.9370485 ]
 [0.89473684]
 [0.84107327]
 [0.8379773 ]
 [0.79772962]
 [0.83900929]
 [0.86068111]
 [0.80701754]
 [0.92053664]
 [0.93188854]
 [0.99690402]
 [0.96697626]
 [0.9246646 ]
 [0.88544892]
 [0.84313725]
 [0.85345717]
 [0.82249742]
 [0.86996904]]
[[0.8425521  0.864361   0.94412494 0.9627873  1.0120288 ]
 [0.8273498  0.9435921  0.96285415 1.0062658  0.9987292 ]
 [0.9010215  0.92715883 1.0106072  0.9938376  0.94890046]
 [0.91423804 0.9960027  0.99516475 0.9549961  0.92765725]
 [0.9836344  0.99783385 0.9592943  0.9191967  0.85103035]
 [0.9940376  0.95958096 0.92197376 0.84997785 0.87021625]
 [0.9248503  0.919094   0.8695641  0.86128616 0.8063426 ]
 [0.9005285  0.845929   0.85303557 0.8070773  0.8392347 ]
 [0.82856643 0.8503918  0.809797   0.8405484  0.84743893]
 [0.831003   0.806605   0.8442664  0.86186314 0.8171419 ]
 [0.7903378  0.8320444  0.8612788  0.82053006 0.93005073]
 [0.8222903  0.8534502  0.80763394 0.92723083 0.9467604 ]
 [0.83423424 0.86014426 0.94349897 0.96443915 1.0123291 ]
 [0.82362896 0.9437277  0.963722   1.0060146  0.9921422 ]
 [0.91127324 0.93589085 1.0074741  0.98335457 0.93492174]
 [0.9279189  1.0011219  0.98381984 0.9359933  0.90476954]
 [0.9842025  0.99886614 0.9387076  0.9044651  0.8373693 ]
 [0.98057723 0.94064236 0.9027553  0.83232534 0.85896754]
 [0.91581094 0.903834   0.8547968  0.8520117  0.80402935]
 [0.8953809  0.8423785  0.845747   0.8073455  0.8444079 ]
 [0.8401131  0.85901654 0.80934197 0.8422204  0.85852957]
 [0.8485379  0.81656724 0.84660685 0.8727602  0.8235918 ]
 [0.804166   0.8407946  0.86689574 0.8280326  0.9321189 ]
 [0.8324909  0.8599295  0.8071864  0.92719936 0.9491751 ]
 [0.84691495 0.87082124 0.9450692  0.9642812  1.0199403 ]
 [0.8326988  0.95408666 0.9681772  1.0100317  1.0124321 ]
 [0.90335184 0.9370492  1.0209503  0.99950147 0.9624102 ]
 [0.92454565 1.0045072  1.0081586  0.9684069  0.9402497 ]
 [0.9937342  1.01105    0.97003055 0.93015325 0.85751414]
 [1.0026528  0.97169983 0.93239534 0.8586283  0.8786181 ]
 [0.9292098  0.9277611  0.8774009  0.867283   0.81464386]]

  #### Plot     ############################################### 
Saved image to ztest/model_tch/nbeats//n_beats_test.png.

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
[master 089c163] ml_store
 1 file changed, 1123 insertions(+)
To github.com:arita37/mlmodels_store.git
   9618967..089c163  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py", line 522, in <module>
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py", line 418, in get_params
    cf = json.load(open(data_path, mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: 'model_tch/transformer_classifier.json'

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
[master eb46fa7] ml_store
 1 file changed, 38 insertions(+)
To github.com:arita37/mlmodels_store.git
   089c163..eb46fa7  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//03_nbeats_dataloader.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//03_nbeats_dataloader.py", line 9, in <module>
    from dataloader import DataLoader
ModuleNotFoundError: No module named 'dataloader'

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
[master bf234a8] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   eb46fa7..bf234a8  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
Iteration:   0%|          | 0/29440 [00:00<?, ?it/s][A
Iteration:   0%|          | 1/29440 [00:30<247:46:48, 30.30s/it][A
Iteration:   0%|          | 2/29440 [00:42<204:22:12, 24.99s/it][A
Iteration:   0%|          | 3/29440 [00:52<166:37:16, 20.38s/it][A
Iteration:   0%|          | 4/29440 [01:01<139:35:16, 17.07s/it][A
Iteration:   0%|          | 5/29440 [01:11<122:30:16, 14.98s/it][A
Iteration:   0%|          | 6/29440 [01:33<138:47:24, 16.98s/it][A
Iteration:   0%|          | 7/29440 [02:39<259:45:53, 31.77s/it][A
Iteration:   0%|          | 8/29440 [03:58<375:38:30, 45.95s/it][A
Iteration:   0%|          | 9/29440 [04:57<405:39:23, 49.62s/it][A
Iteration:   0%|          | 10/29440 [08:01<735:39:20, 89.99s/it][A
Iteration:   0%|          | 11/29440 [09:08<679:27:55, 83.12s/it][A
Iteration:   0%|          | 12/29440 [10:55<739:29:45, 90.46s/it][A
Iteration:   0%|          | 13/29440 [11:34<610:53:58, 74.74s/it][A
Iteration:   0%|          | 14/29440 [12:18<536:43:48, 65.66s/it][A
Iteration:   0%|          | 15/29440 [13:49<598:29:22, 73.22s/it][A
Iteration:   0%|          | 16/29440 [14:42<548:17:14, 67.08s/it][A
Iteration:   0%|          | 17/29440 [15:13<461:51:21, 56.51s/it][A
Iteration:   0%|          | 18/29440 [16:13<469:43:13, 57.47s/it][A
Iteration:   0%|          | 19/29440 [17:50<565:19:08, 69.17s/it][A
Iteration:   0%|          | 20/29440 [21:01<863:33:06, 105.67s/it][A
Iteration:   0%|          | 21/29440 [21:55<737:22:23, 90.23s/it] [A
Iteration:   0%|          | 22/29440 [23:40<774:24:45, 94.77s/it][A
Iteration:   0%|          | 23/29440 [24:18<634:00:39, 77.59s/it][A
Iteration:   0%|          | 24/29440 [25:21<599:28:09, 73.36s/it][A
Iteration:   0%|          | 25/29440 [27:37<752:22:51, 92.08s/it][A
Iteration:   0%|          | 26/29440 [28:45<693:00:06, 84.82s/it][A
Iteration:   0%|          | 27/29440 [29:36<610:21:35, 74.70s/it][A
Iteration:   0%|          | 28/29440 [29:53<468:10:10, 57.30s/it][A
Iteration:   0%|          | 29/29440 [30:11<373:43:50, 45.75s/it][A
Iteration:   0%|          | 30/29440 [36:36<1203:26:25, 147.31s/it][A
Iteration:   0%|          | 31/29440 [38:36<1137:06:24, 139.19s/it][A
Iteration:   0%|          | 32/29440 [39:41<956:17:01, 117.06s/it] [A
Iteration:   0%|          | 33/29440 [40:23<772:37:13, 94.58s/it] [A
Iteration:   0%|          | 34/29440 [41:09<653:03:43, 79.95s/it][A
Iteration:   0%|          | 35/29440 [42:17<623:12:16, 76.30s/it][A
Iteration:   0%|          | 36/29440 [43:03<549:42:22, 67.30s/it][A
Iteration:   0%|          | 37/29440 [44:24<582:00:15, 71.26s/it][A
Iteration:   0%|          | 38/29440 [45:04<505:43:10, 61.92s/it][A
Iteration:   0%|          | 39/29440 [47:16<677:54:03, 83.01s/it][A
Iteration:   0%|          | 40/29440 [51:55<1157:29:05, 141.73s/it][A
Iteration:   0%|          | 41/29440 [52:46<935:20:28, 114.54s/it] [A
Iteration:   0%|          | 42/29440 [53:05<700:28:14, 85.78s/it] [A
Iteration:   0%|          | 43/29440 [55:55<908:14:46, 111.23s/it][AKilled

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
   bf234a8..55d620d  master     -> origin/master
Updating bf234a8..55d620d
Fast-forward
 deps.txt                                           |    8 +-
 error_list/20200522/list_log_jupyter_20200522.md   | 2021 ++++++++-----
 .../20200522/list_log_pullrequest_20200522.md      |    2 +-
 error_list/20200522/list_log_testall_20200522.md   |   16 +
 log_jupyter/log_jupyter.py                         | 3002 ++++++++------------
 ...-10_09dbb573cf89ddf861ea945ff13f39f474d48070.py |  637 +++++
 6 files changed, 3198 insertions(+), 2488 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-22-10-10_09dbb573cf89ddf861ea945ff13f39f474d48070.py
[master 57ac47a] ml_store
 1 file changed, 96 insertions(+)
To github.com:arita37/mlmodels_store.git
   55d620d..57ac47a  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 10871070.97B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 481860.87B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 4315136/440473133 [00:00<00:10, 43143480.77B/s]  2%|▏         | 10286080/440473133 [00:00<00:09, 47059274.29B/s]  4%|▍         | 16602112/440473133 [00:00<00:08, 50955063.76B/s]  5%|▌         | 23026688/440473133 [00:00<00:07, 54326229.36B/s]  7%|▋         | 29330432/440473133 [00:00<00:07, 56659950.25B/s]  8%|▊         | 35067904/440473133 [00:00<00:07, 56862844.62B/s]  9%|▉         | 41411584/440473133 [00:00<00:06, 58687275.44B/s] 11%|█         | 47106048/440473133 [00:00<00:06, 58151875.52B/s] 12%|█▏        | 53434368/440473133 [00:00<00:06, 59599647.20B/s] 14%|█▎        | 59777024/440473133 [00:01<00:06, 60696111.08B/s] 15%|█▍        | 65767424/440473133 [00:01<00:06, 58639873.35B/s] 16%|█▋        | 72235008/440473133 [00:01<00:06, 60328291.52B/s] 18%|█▊        | 78641152/440473133 [00:01<00:05, 61400155.51B/s] 19%|█▉        | 85104640/440473133 [00:01<00:05, 62335473.94B/s] 21%|██        | 91602944/440473133 [00:01<00:05, 63105075.34B/s] 22%|██▏       | 97914880/440473133 [00:01<00:05, 61818344.63B/s] 24%|██▎       | 104103936/440473133 [00:01<00:05, 61134658.49B/s] 25%|██▌       | 110224384/440473133 [00:01<00:05, 60501906.70B/s] 26%|██▋       | 116294656/440473133 [00:01<00:05, 60561034.36B/s] 28%|██▊       | 122371072/440473133 [00:02<00:05, 60620462.38B/s] 29%|██▉       | 128436224/440473133 [00:02<00:05, 60052881.63B/s] 31%|███       | 134445056/440473133 [00:02<00:05, 58512402.55B/s] 32%|███▏      | 140407808/440473133 [00:02<00:05, 58837317.56B/s] 33%|███▎      | 146636800/440473133 [00:02<00:04, 59831712.95B/s] 35%|███▍      | 153017344/440473133 [00:02<00:04, 60969304.14B/s] 36%|███▌      | 159126528/440473133 [00:02<00:04, 60936044.68B/s] 38%|███▊      | 165464064/440473133 [00:02<00:04, 61646365.30B/s] 39%|███▉      | 171863040/440473133 [00:02<00:04, 62330353.92B/s] 40%|████      | 178128896/440473133 [00:02<00:04, 62421221.19B/s] 42%|████▏     | 184384512/440473133 [00:03<00:04, 62449483.75B/s] 43%|████▎     | 190633984/440473133 [00:03<00:04, 61279195.63B/s] 45%|████▍     | 197021696/440473133 [00:03<00:03, 62035955.29B/s] 46%|████▌     | 203233280/440473133 [00:03<00:03, 61928215.07B/s] 48%|████▊     | 209632256/440473133 [00:03<00:03, 62527673.83B/s] 49%|████▉     | 215890944/440473133 [00:03<00:03, 60589005.56B/s] 50%|█████     | 221967360/440473133 [00:03<00:03, 60102912.42B/s] 52%|█████▏    | 228198400/440473133 [00:03<00:03, 60745738.01B/s] 53%|█████▎    | 234284032/440473133 [00:03<00:03, 58981241.44B/s] 55%|█████▍    | 240200704/440473133 [00:03<00:03, 58478405.00B/s] 56%|█████▌    | 246127616/440473133 [00:04<00:03, 58712829.53B/s] 57%|█████▋    | 252009472/440473133 [00:04<00:03, 57964209.68B/s] 59%|█████▊    | 257862656/440473133 [00:04<00:03, 58131910.56B/s] 60%|█████▉    | 263796736/440473133 [00:04<00:03, 58488606.87B/s] 61%|██████▏   | 269842432/440473133 [00:04<00:02, 59064919.50B/s] 63%|██████▎   | 276091904/440473133 [00:04<00:02, 60052736.31B/s] 64%|██████▍   | 282104832/440473133 [00:04<00:02, 58752272.70B/s] 65%|██████▌   | 287991808/440473133 [00:04<00:02, 58742634.08B/s] 67%|██████▋   | 294109184/440473133 [00:04<00:02, 59451164.90B/s] 68%|██████▊   | 300149760/440473133 [00:04<00:02, 59733532.28B/s] 70%|██████▉   | 306192384/440473133 [00:05<00:02, 59937094.21B/s] 71%|███████   | 312190976/440473133 [00:05<00:02, 59640692.72B/s] 72%|███████▏  | 318193664/440473133 [00:05<00:02, 59750852.75B/s] 74%|███████▎  | 324241408/440473133 [00:05<00:01, 59963263.76B/s] 75%|███████▍  | 330240000/440473133 [00:05<00:01, 58692413.80B/s] 76%|███████▋  | 336250880/440473133 [00:05<00:01, 59108541.41B/s] 78%|███████▊  | 342167552/440473133 [00:05<00:01, 58533785.64B/s] 79%|███████▉  | 348338176/440473133 [00:05<00:01, 59450135.65B/s] 80%|████████  | 354363392/440473133 [00:05<00:01, 59686478.95B/s] 82%|████████▏ | 360337408/440473133 [00:06<00:01, 58394804.43B/s] 83%|████████▎ | 366296064/440473133 [00:06<00:01, 58745282.11B/s] 85%|████████▍ | 372588544/440473133 [00:06<00:01, 59938119.24B/s] 86%|████████▌ | 378787840/440473133 [00:06<00:01, 60537646.70B/s] 87%|████████▋ | 385081344/440473133 [00:06<00:00, 61235649.38B/s] 89%|████████▉ | 391469056/440473133 [00:06<00:00, 62002083.03B/s] 90%|█████████ | 397838336/440473133 [00:06<00:00, 62498236.10B/s] 92%|█████████▏| 404094976/440473133 [00:06<00:00, 62110026.99B/s] 93%|█████████▎| 410577920/440473133 [00:06<00:00, 62901258.77B/s] 95%|█████████▍| 416874496/440473133 [00:06<00:00, 62730865.43B/s] 96%|█████████▌| 423152640/440473133 [00:07<00:00, 62109013.43B/s] 97%|█████████▋| 429368320/440473133 [00:07<00:00, 60373258.46B/s] 99%|█████████▉| 435420160/440473133 [00:07<00:00, 60021886.11B/s]100%|██████████| 440473133/440473133 [00:07<00:00, 60357150.08B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
4857856/7094233 [===================>..........] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:  23%|██▎       | 483/2118 [00:00<00:00, 4829.73it/s]Processing text_left with encode:  45%|████▍     | 952/2118 [00:00<00:00, 4784.27it/s]Processing text_left with encode:  71%|███████   | 1507/2118 [00:00<00:00, 4988.98it/s]Processing text_left with encode:  98%|█████████▊| 2074/2118 [00:00<00:00, 5174.84it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 5169.85it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 218/18841 [00:00<00:08, 2179.34it/s]Processing text_right with encode:   2%|▏         | 430/18841 [00:00<00:08, 2159.58it/s]Processing text_right with encode:   4%|▎         | 662/18841 [00:00<00:08, 2201.09it/s]Processing text_right with encode:   5%|▍         | 865/18841 [00:00<00:08, 2144.50it/s]Processing text_right with encode:   6%|▌         | 1064/18841 [00:00<00:08, 2093.95it/s]Processing text_right with encode:   7%|▋         | 1268/18841 [00:00<00:08, 2076.03it/s]Processing text_right with encode:   8%|▊         | 1476/18841 [00:00<00:08, 2076.77it/s]Processing text_right with encode:   9%|▉         | 1700/18841 [00:00<00:08, 2122.16it/s]Processing text_right with encode:  10%|█         | 1901/18841 [00:00<00:08, 2065.09it/s]Processing text_right with encode:  11%|█         | 2100/18841 [00:01<00:08, 2026.91it/s]Processing text_right with encode:  12%|█▏        | 2323/18841 [00:01<00:07, 2081.33it/s]Processing text_right with encode:  14%|█▎        | 2546/18841 [00:01<00:07, 2123.49it/s]Processing text_right with encode:  15%|█▍        | 2782/18841 [00:01<00:07, 2188.48it/s]Processing text_right with encode:  16%|█▌        | 3000/18841 [00:01<00:07, 2098.77it/s]Processing text_right with encode:  17%|█▋        | 3210/18841 [00:01<00:07, 2021.20it/s]Processing text_right with encode:  18%|█▊        | 3421/18841 [00:01<00:07, 2046.14it/s]Processing text_right with encode:  19%|█▉        | 3630/18841 [00:01<00:07, 2057.82it/s]Processing text_right with encode:  20%|██        | 3841/18841 [00:01<00:07, 2068.81it/s]Processing text_right with encode:  22%|██▏       | 4074/18841 [00:01<00:06, 2139.18it/s]Processing text_right with encode:  23%|██▎       | 4289/18841 [00:02<00:07, 2067.14it/s]Processing text_right with encode:  24%|██▍       | 4497/18841 [00:02<00:06, 2056.92it/s]Processing text_right with encode:  25%|██▌       | 4715/18841 [00:02<00:06, 2091.43it/s]Processing text_right with encode:  26%|██▋       | 4947/18841 [00:02<00:06, 2152.44it/s]Processing text_right with encode:  27%|██▋       | 5164/18841 [00:02<00:06, 2128.77it/s]Processing text_right with encode:  29%|██▊       | 5378/18841 [00:02<00:06, 2101.96it/s]Processing text_right with encode:  30%|██▉       | 5592/18841 [00:02<00:06, 2110.32it/s]Processing text_right with encode:  31%|███       | 5810/18841 [00:02<00:06, 2128.07it/s]Processing text_right with encode:  32%|███▏      | 6024/18841 [00:02<00:06, 2119.97it/s]Processing text_right with encode:  33%|███▎      | 6237/18841 [00:02<00:05, 2116.37it/s]Processing text_right with encode:  34%|███▍      | 6449/18841 [00:03<00:05, 2114.90it/s]Processing text_right with encode:  35%|███▌      | 6677/18841 [00:03<00:05, 2159.92it/s]Processing text_right with encode:  37%|███▋      | 6894/18841 [00:03<00:05, 2134.78it/s]Processing text_right with encode:  38%|███▊      | 7108/18841 [00:03<00:05, 2119.02it/s]Processing text_right with encode:  39%|███▉      | 7337/18841 [00:03<00:05, 2167.42it/s]Processing text_right with encode:  40%|████      | 7555/18841 [00:03<00:05, 2139.29it/s]Processing text_right with encode:  41%|████▏     | 7774/18841 [00:03<00:05, 2153.51it/s]Processing text_right with encode:  42%|████▏     | 7990/18841 [00:03<00:05, 2138.80it/s]Processing text_right with encode:  44%|████▎     | 8205/18841 [00:03<00:05, 2124.16it/s]Processing text_right with encode:  45%|████▍     | 8429/18841 [00:03<00:04, 2155.51it/s]Processing text_right with encode:  46%|████▌     | 8645/18841 [00:04<00:04, 2082.53it/s]Processing text_right with encode:  47%|████▋     | 8861/18841 [00:04<00:04, 2103.77it/s]Processing text_right with encode:  48%|████▊     | 9072/18841 [00:04<00:04, 2084.25it/s]Processing text_right with encode:  49%|████▉     | 9293/18841 [00:04<00:04, 2118.67it/s]Processing text_right with encode:  50%|█████     | 9513/18841 [00:04<00:04, 2140.67it/s]Processing text_right with encode:  52%|█████▏    | 9728/18841 [00:04<00:04, 2107.57it/s]Processing text_right with encode:  53%|█████▎    | 9950/18841 [00:04<00:04, 2136.42it/s]Processing text_right with encode:  54%|█████▍    | 10171/18841 [00:04<00:04, 2156.11it/s]Processing text_right with encode:  55%|█████▌    | 10387/18841 [00:04<00:03, 2136.50it/s]Processing text_right with encode:  56%|█████▋    | 10615/18841 [00:05<00:03, 2176.66it/s]Processing text_right with encode:  58%|█████▊    | 10834/18841 [00:05<00:03, 2116.55it/s]Processing text_right with encode:  59%|█████▊    | 11048/18841 [00:05<00:03, 2123.24it/s]Processing text_right with encode:  60%|█████▉    | 11262/18841 [00:05<00:03, 2124.06it/s]Processing text_right with encode:  61%|██████    | 11476/18841 [00:05<00:03, 2128.07it/s]Processing text_right with encode:  62%|██████▏   | 11690/18841 [00:05<00:03, 2115.04it/s]Processing text_right with encode:  63%|██████▎   | 11902/18841 [00:05<00:03, 2096.91it/s]Processing text_right with encode:  64%|██████▍   | 12118/18841 [00:05<00:03, 2112.51it/s]Processing text_right with encode:  65%|██████▌   | 12332/18841 [00:05<00:03, 2120.09it/s]Processing text_right with encode:  67%|██████▋   | 12545/18841 [00:05<00:02, 2122.63it/s]Processing text_right with encode:  68%|██████▊   | 12759/18841 [00:06<00:02, 2127.36it/s]Processing text_right with encode:  69%|██████▉   | 12972/18841 [00:06<00:02, 2108.78it/s]Processing text_right with encode:  70%|██████▉   | 13183/18841 [00:06<00:02, 2104.49it/s]Processing text_right with encode:  71%|███████   | 13402/18841 [00:06<00:02, 2128.38it/s]Processing text_right with encode:  72%|███████▏  | 13625/18841 [00:06<00:02, 2157.69it/s]Processing text_right with encode:  74%|███████▎  | 13850/18841 [00:06<00:02, 2181.41it/s]Processing text_right with encode:  75%|███████▍  | 14069/18841 [00:06<00:02, 2156.42it/s]Processing text_right with encode:  76%|███████▌  | 14285/18841 [00:06<00:02, 2147.67it/s]Processing text_right with encode:  77%|███████▋  | 14500/18841 [00:06<00:02, 2142.17it/s]Processing text_right with encode:  78%|███████▊  | 14735/18841 [00:06<00:01, 2199.02it/s]Processing text_right with encode:  79%|███████▉  | 14957/18841 [00:07<00:01, 2200.92it/s]Processing text_right with encode:  81%|████████  | 15178/18841 [00:07<00:01, 2143.25it/s]Processing text_right with encode:  82%|████████▏ | 15394/18841 [00:07<00:01, 2145.56it/s]Processing text_right with encode:  83%|████████▎ | 15616/18841 [00:07<00:01, 2166.93it/s]Processing text_right with encode:  84%|████████▍ | 15834/18841 [00:07<00:01, 2151.15it/s]Processing text_right with encode:  85%|████████▌ | 16050/18841 [00:07<00:01, 2142.55it/s]Processing text_right with encode:  86%|████████▋ | 16265/18841 [00:07<00:01, 2078.94it/s]Processing text_right with encode:  87%|████████▋ | 16485/18841 [00:07<00:01, 2112.68it/s]Processing text_right with encode:  89%|████████▊ | 16697/18841 [00:07<00:01, 2027.55it/s]Processing text_right with encode:  90%|████████▉ | 16901/18841 [00:07<00:00, 1993.77it/s]Processing text_right with encode:  91%|█████████ | 17115/18841 [00:08<00:00, 2032.57it/s]Processing text_right with encode:  92%|█████████▏| 17320/18841 [00:08<00:00, 2013.87it/s]Processing text_right with encode:  93%|█████████▎| 17531/18841 [00:08<00:00, 2039.01it/s]Processing text_right with encode:  94%|█████████▍| 17763/18841 [00:08<00:00, 2115.16it/s]Processing text_right with encode:  95%|█████████▌| 17976/18841 [00:08<00:00, 2093.83it/s]Processing text_right with encode:  97%|█████████▋| 18201/18841 [00:08<00:00, 2135.54it/s]Processing text_right with encode:  98%|█████████▊| 18416/18841 [00:08<00:00, 2119.62it/s]Processing text_right with encode:  99%|█████████▉| 18647/18841 [00:08<00:00, 2170.13it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:08<00:00, 2118.96it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 820195.35it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 962509.06it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  93%|█████████▎| 589/633 [00:00<00:00, 5887.17it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 5833.19it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   4%|▎         | 222/5961 [00:00<00:02, 2211.64it/s]Processing text_right with encode:   7%|▋         | 419/5961 [00:00<00:02, 2130.21it/s]Processing text_right with encode:  11%|█         | 640/5961 [00:00<00:02, 2150.18it/s]Processing text_right with encode:  15%|█▍        | 870/5961 [00:00<00:02, 2192.11it/s]Processing text_right with encode:  18%|█▊        | 1088/5961 [00:00<00:02, 2187.62it/s]Processing text_right with encode:  22%|██▏       | 1321/5961 [00:00<00:02, 2226.35it/s]Processing text_right with encode:  26%|██▌       | 1524/5961 [00:00<00:02, 2162.02it/s]Processing text_right with encode:  29%|██▉       | 1723/5961 [00:00<00:02, 2106.50it/s]Processing text_right with encode:  33%|███▎      | 1943/5961 [00:00<00:01, 2132.77it/s]Processing text_right with encode:  36%|███▋      | 2161/5961 [00:01<00:01, 2145.65it/s]Processing text_right with encode:  40%|███▉      | 2370/5961 [00:01<00:01, 2108.10it/s]Processing text_right with encode:  43%|████▎     | 2581/5961 [00:01<00:01, 2107.53it/s]Processing text_right with encode:  47%|████▋     | 2820/5961 [00:01<00:01, 2184.80it/s]Processing text_right with encode:  51%|█████     | 3049/5961 [00:01<00:01, 2213.85it/s]Processing text_right with encode:  55%|█████▍    | 3275/5961 [00:01<00:01, 2226.48it/s]Processing text_right with encode:  59%|█████▊    | 3498/5961 [00:01<00:01, 2180.42it/s]Processing text_right with encode:  62%|██████▏   | 3716/5961 [00:01<00:01, 2152.57it/s]Processing text_right with encode:  66%|██████▌   | 3947/5961 [00:01<00:00, 2196.28it/s]Processing text_right with encode:  70%|██████▉   | 4168/5961 [00:01<00:00, 2200.21it/s]Processing text_right with encode:  74%|███████▎  | 4389/5961 [00:02<00:00, 2201.34it/s]Processing text_right with encode:  77%|███████▋  | 4612/5961 [00:02<00:00, 2207.85it/s]Processing text_right with encode:  81%|████████  | 4833/5961 [00:02<00:00, 2136.74it/s]Processing text_right with encode:  85%|████████▍ | 5048/5961 [00:02<00:00, 2119.52it/s]Processing text_right with encode:  89%|████████▊ | 5283/5961 [00:02<00:00, 2183.50it/s]Processing text_right with encode:  92%|█████████▏| 5503/5961 [00:02<00:00, 2136.40it/s]Processing text_right with encode:  96%|█████████▌| 5718/5961 [00:02<00:00, 2006.79it/s]Processing text_right with encode: 100%|█████████▉| 5938/5961 [00:02<00:00, 2060.52it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:02<00:00, 2151.53it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 578682.31it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 976650.24it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:23<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:23<?, ?it/s, loss=1.077]Epoch 1/1:   1%|          | 1/102 [00:23<39:07, 23.24s/it, loss=1.077]Epoch 1/1:   1%|          | 1/102 [00:47<39:07, 23.24s/it, loss=1.077]Epoch 1/1:   1%|          | 1/102 [00:47<39:07, 23.24s/it, loss=0.922]Epoch 1/1:   2%|▏         | 2/102 [00:47<39:04, 23.44s/it, loss=0.922]Epoch 1/1:   2%|▏         | 2/102 [01:46<39:04, 23.44s/it, loss=0.922]Epoch 1/1:   2%|▏         | 2/102 [01:46<39:04, 23.44s/it, loss=0.995]Epoch 1/1:   3%|▎         | 3/102 [01:46<56:19, 34.13s/it, loss=0.995]Epoch 1/1:   3%|▎         | 3/102 [02:38<56:19, 34.13s/it, loss=0.995]Epoch 1/1:   3%|▎         | 3/102 [02:38<56:19, 34.13s/it, loss=0.902]Epoch 1/1:   4%|▍         | 4/102 [02:38<1:04:39, 39.59s/it, loss=0.902]Epoch 1/1:   4%|▍         | 4/102 [04:11<1:04:39, 39.59s/it, loss=0.902]Epoch 1/1:   4%|▍         | 4/102 [04:11<1:04:39, 39.59s/it, loss=1.003]Epoch 1/1:   5%|▍         | 5/102 [04:11<1:30:00, 55.67s/it, loss=1.003]Epoch 1/1:   5%|▍         | 5/102 [04:44<1:30:00, 55.67s/it, loss=1.003]Epoch 1/1:   5%|▍         | 5/102 [04:44<1:30:00, 55.67s/it, loss=0.818]Epoch 1/1:   6%|▌         | 6/102 [04:44<1:18:16, 48.93s/it, loss=0.818]Epoch 1/1:   6%|▌         | 6/102 [05:34<1:18:16, 48.93s/it, loss=0.818]Epoch 1/1:   6%|▌         | 6/102 [05:34<1:18:16, 48.93s/it, loss=0.718]Epoch 1/1:   7%|▋         | 7/102 [05:34<1:17:43, 49.09s/it, loss=0.718]Epoch 1/1:   7%|▋         | 7/102 [06:49<1:17:43, 49.09s/it, loss=0.718]Epoch 1/1:   7%|▋         | 7/102 [06:49<1:17:43, 49.09s/it, loss=0.817]Epoch 1/1:   8%|▊         | 8/102 [06:49<1:29:00, 56.81s/it, loss=0.817]Killed

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
   57ac47a..69bdd0f  master     -> origin/master
Updating 57ac47a..69bdd0f
Fast-forward
 error_list/20200522/list_log_pullrequest_20200522.md | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
[master ed9c9ee] ml_store
 2 files changed, 66 insertions(+), 6 deletions(-)
To github.com:arita37/mlmodels_store.git
   69bdd0f..ed9c9ee  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/textcnn/model', 'checkpointdir': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/textcnn//checkpoint/'}

  #### Loading dataset   ############################################# 
>>>>> load:  {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=9841a1e7eece4346b49f54d8e4fa3f0773c021b96608cabe65067349d1cb677a
  Stored in directory: /tmp/pip-ephem-wheel-cache-iyeni5s_/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 153, in create_tabular_dataset
    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
    return util.load_model(name, **overrides)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
    raise IOError(Errors.E050.format(name=name))
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 477, in <module>
    test( data_path="model_tch/textcnn.json", pars_choice = "test01" )
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 442, in test
    Xtuple = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 334, in get_dataset
    trainset, validset, vocab = create_tabular_dataset( data_pars['train_path'], data_pars['valid_path'], lang, pretrained_emb)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 159, in create_tabular_dataset
    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
    return util.load_model(name, **overrides)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
    raise IOError(Errors.E050.format(name=name))
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.

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
[master 116b1b1] ml_store
 2 files changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   ed9c9ee..116b1b1  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pytorch_vae.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pytorch_vae.py", line 34, in <module>
    "beta_vae": md.model.beta_vae,
AttributeError: module 'mlmodels.model_tch.raw.pytorch_vae' has no attribute 'model'

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
[master aa6c374] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   116b1b1..aa6c374  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels/preprocess/generic.py::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f491d631ea0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f491d631ea0>

  function with postional parmater data_info <function get_dataset_torch at 0x7f491d631ea0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 145196.51it/s] 78%|███████▊  | 7692288/9912422 [00:00<00:10, 207254.70it/s]9920512it [00:00, 43437175.45it/s]                           
0it [00:00, ?it/s]32768it [00:00, 597927.22it/s]
0it [00:00, ?it/s]  4%|▍         | 73728/1648877 [00:00<00:02, 737208.64it/s]1654784it [00:00, 12173341.45it/s]                         
0it [00:00, ?it/s]8192it [00:00, 221906.23it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Processing...
Done!

  #### Model init, fit   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f491c96abf8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f491c96abf8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f491c96abf8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.001999993582566579 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.01121086049079895 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0012690487504005432 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.009925252318382264 	 Accuracy: 1
model saves at 1 accuracy

  #### Predict   ##################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f491c96a9d8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f491c96a9d8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f491c96a9d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  #### metrics   ##################################################### 
None

  #### Plot   ######################################################## 

  #### Save  ######################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 

  #### Load   ######################################################## 
<__main__.Model object at 0x7f491d3e6978>

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels.preprocess.generic::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/pytorch_GAN_zoo/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/pytorch_GAN_zoo/'} 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### Predict   ##################################################### 
img_01.png

  #### metrics   ##################################################### 

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/pytorch_GAN_zoo//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 
img_01.png
torch_model

Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/facebookresearch_pytorch_GAN_zoo_hub
<__main__.Model object at 0x7f491556eb00>

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


  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_all GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/f56aed39a4e8698b1d43abde0792c92364346a5c', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': 'f56aed39a4e8698b1d43abde0792c92364346a5c', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/f56aed39a4e8698b1d43abde0792c92364346a5c

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/f56aed39a4e8698b1d43abde0792c92364346a5c

 ************************************************************************************************************************

  ############Check model ################################ 

  ['model_tf.temporal_fusion_google', 'model_tf.1_lstm', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn', 'model_tch.pplm', 'model_tch.transformer_sentence', 'model_tch.nbeats', 'model_tch.textcnn', 'model_tch.mlp', 'model_tch.pytorch_vae', 'model_tch.03_nbeats_dataloader', 'model_tch.torchhub', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_gluon.gluonts_model', 'model_gluon.fb_prophet', 'model_gluon.gluon_automl', 'model_keras.charcnn', 'model_keras.charcnn_zhang', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.01_deepctr', 'model_keras.namentity_crm_bilstm', 'model_keras.nbeats', 'model_keras.textcnn', 'model_keras.keras_gan', 'model_keras.textcnn_dataloader', 'model_keras.02_cnn', 'model_keras.Autokeras', 'model_keras.armdn', 'model_keras.textvae'] 

  Used ['model_tf.temporal_fusion_google', 'model_tf.1_lstm', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn', 'model_tch.pplm', 'model_tch.transformer_sentence', 'model_tch.nbeats', 'model_tch.textcnn', 'model_tch.mlp', 'model_tch.pytorch_vae', 'model_tch.03_nbeats_dataloader', 'model_tch.torchhub', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_gluon.gluonts_model', 'model_gluon.fb_prophet', 'model_gluon.gluon_automl', 'model_keras.charcnn', 'model_keras.charcnn_zhang', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.01_deepctr', 'model_keras.namentity_crm_bilstm', 'model_keras.nbeats', 'model_keras.textcnn', 'model_keras.keras_gan', 'model_keras.textcnn_dataloader', 'model_keras.02_cnn', 'model_keras.Autokeras', 'model_keras.armdn', 'model_keras.textvae'] 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
    from mlmodels.mode_tf.raw  import temporal_fusion_google
ModuleNotFoundError: No module named 'mlmodels.mode_tf'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   813609c..2d7a588  master     -> origin/master
Updating 813609c..2d7a588
Fast-forward
 ...-06_46ba20fe091e28b621f61cf8993a32b6038feb3d.py |  167 +
 ...-07_f56aed39a4e8698b1d43abde0792c92364346a5c.py |  167 +
 ...-09_3c412e105084f9d92d8ff55ba1516753739477a1.py |  167 +
 ...-02_3c412e105084f9d92d8ff55ba1516753739477a1.py | 4985 ++++++++++++++++++++
 ...-05_f56aed39a4e8698b1d43abde0792c92364346a5c.py | 4984 +++++++++++++++++++
 ...-06_46ba20fe091e28b621f61cf8993a32b6038feb3d.py | 4984 +++++++++++++++++++
 ...-04_f56aed39a4e8698b1d43abde0792c92364346a5c.py | 1914 ++++++++
 ...-03_3c412e105084f9d92d8ff55ba1516753739477a1.py | 3250 +++++++++++++
 ...-04_f56aed39a4e8698b1d43abde0792c92364346a5c.py | 3220 +++++++++++++
 ...-06_46ba20fe091e28b621f61cf8993a32b6038feb3d.py | 3220 +++++++++++++
 10 files changed, 27058 insertions(+)
 create mode 100644 log_dataloader/log_2020-05-10-20-06_46ba20fe091e28b621f61cf8993a32b6038feb3d.py
 create mode 100644 log_dataloader/log_2020-05-10-20-07_f56aed39a4e8698b1d43abde0792c92364346a5c.py
 create mode 100644 log_dataloader/log_2020-05-10-20-09_3c412e105084f9d92d8ff55ba1516753739477a1.py
 create mode 100644 log_json/log_json_2020-05-10-20-02_3c412e105084f9d92d8ff55ba1516753739477a1.py
 create mode 100644 log_json/log_json_2020-05-10-20-05_f56aed39a4e8698b1d43abde0792c92364346a5c.py
 create mode 100644 log_json/log_json_2020-05-10-20-06_46ba20fe091e28b621f61cf8993a32b6038feb3d.py
 create mode 100644 log_jupyter/log_jupyter_2020-05-10-20-04_f56aed39a4e8698b1d43abde0792c92364346a5c.py
 create mode 100644 log_test_cli/log_cli_2020-05-10-20-03_3c412e105084f9d92d8ff55ba1516753739477a1.py
 create mode 100644 log_test_cli/log_cli_2020-05-10-20-04_f56aed39a4e8698b1d43abde0792c92364346a5c.py
 create mode 100644 log_test_cli/log_cli_2020-05-10-20-06_46ba20fe091e28b621f61cf8993a32b6038feb3d.py
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
